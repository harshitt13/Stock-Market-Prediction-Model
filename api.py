import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# ---------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import run_pipeline


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Stock Market Prediction API",
    description="API for the hybrid stock prediction pipeline.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Job management
# ---------------------------------------------------------

executor = ThreadPoolExecutor(max_workers=1)

jobs: Dict[str, Dict[str, Any]] = {}

jobs_lock = Lock()


# ---------------------------------------------------------
# Cache
# ---------------------------------------------------------

CACHE_DIR = PROJECT_ROOT / "data" / "api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------

API_KEY = os.getenv("STOCK_API_KEY")
security = HTTPBearer()

def require_api_key(
    credentials: HTTPAuthorizationCredentials,
) -> None:
    if API_KEY is None:
        raise HTTPException(
            status_code=500,
            detail="STOCK_API_KEY is not configured.",
        )

import secrets

    if not secrets.compare_digest(credentials.credentials, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class RetrainRequest(BaseModel):
    ticker: str


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()

def cache_path(ticker: str, days: int) -> Path:
    safe_ticker = normalize_ticker(ticker)

    if not safe_ticker:
        raise ValueError("Ticker is required.")

    safe_ticker = "".join(
        char for char in safe_ticker
        if char.isalnum() or char in ("_", "-")
    )

    if not safe_ticker:
        raise ValueError("Invalid ticker.")

    return CACHE_DIR / f"{safe_ticker}_{days}.json"

def save_cache(ticker: str, days: int, result: Dict[str, Any]) -> None:
    path = cache_path(ticker, days)

    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def load_cache(ticker: str, days: int) -> Optional[Dict[str, Any]]:
    path = cache_path(ticker, days)

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def is_cache_fresh(result: Dict[str, Any]) -> bool:
    """
    Development freshness rule:

    Treat a prediction as fresh if it was generated today.

    Later we can make this stricter by checking the latest
    market-data date / market close.
    """

    generated_at = result.get("generated_at")

    if not generated_at:
        return False

    try:
        generated_date = datetime.fromisoformat(
            generated_at.replace("Z", "+00:00")
        ).date()

        today = datetime.now(timezone.utc).date()

        return generated_date == today

    except (ValueError, TypeError):
        return False


def direction_from_price(
    current_price: float,
    predicted_price: float,
) -> str:

    if predicted_price > current_price:
        return "up"

    if predicted_price < current_price:
        return "down"

    return "flat"


def model_direction(
    current_price: float,
    prediction: float,
) -> str:
    return direction_from_price(current_price, prediction)


def build_api_result(
    ticker: str,
    pipeline_result: Dict[str, Any],
) -> Dict[str, Any]:

    stock_data = pipeline_result["stock_data"]

    hybrid_future = pipeline_result.get("hybrid_future")

    conf_lower = pipeline_result.get("conf_lower")
    conf_upper = pipeline_result.get("conf_upper")

    base_results = pipeline_result["base_results"]

    if hybrid_future is None or len(hybrid_future) == 0:
        raise RuntimeError("Pipeline did not produce hybrid predictions.")

    current_price = float(stock_data["Close"].iloc[-1])

    last_data_used = pd_to_iso_date(
        stock_data["Date"].iloc[-1]
    )

    predicted_price = float(hybrid_future[0])

    predicted_return_pct = (
        (predicted_price - current_price)
        / current_price
        * 100
        if current_price != 0
        else 0.0
    )

    future_dates = base_results["tree"]["future_dates"]

    target_date = pd_to_iso_date(future_dates[0])

    result = {
        "ticker": normalize_ticker(ticker),
        "prediction_date": datetime.now(timezone.utc).date().isoformat(),
        "target_date": target_date,
        "direction": direction_from_price(
            current_price,
            predicted_price,
        ),
        # IMPORTANT:
        # We currently do not have a scientifically defined
        # per-prediction confidence score in the existing pipeline.
        "confidence": None,
        "predicted_return_pct": round(
            predicted_return_pct,
            4,
        ),
        "confidence_interval": build_confidence_interval(
            conf_lower,
            conf_upper,
            0,
        ),
        "model_breakdown": {
            "tree": model_direction(
                current_price,
                float(
                    base_results["tree"]["future_predictions"][0]
                ),
            ),
            "lstm": model_direction(
                current_price,
                float(
                    base_results["lstm"]["future_predictions"][0]
                ),
            ),
            "transformer": model_direction(
                current_price,
                float(
                    base_results["transformer"]["future_predictions"][0]
                ),
            ),
        },
        "last_data_used": last_data_used,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return result


def pd_to_iso_date(value: Any) -> str:
    """
    Convert pandas/numpy/Python date-like values to YYYY-MM-DD.
    """

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")

    return str(value)[:10]


def build_confidence_interval(
    lower: Any,
    upper: Any,
    index: int,
) -> Optional[Dict[str, float]]:

    if lower is None or upper is None:
        return None

    if len(lower) <= index or len(upper) <= index:
        return None

    return {
        "lower": round(float(lower[index]), 4),
        "upper": round(float(upper[index]), 4),
    }


# ---------------------------------------------------------
# Background pipeline execution
# ---------------------------------------------------------

def execute_pipeline_job(
    job_id: str,
    ticker: str,
    days: int,
) -> None:

    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = "running_pipeline"

    try:

        pipeline_result = run_pipeline(
            ticker=ticker,
            future_days=days,
        )

        if "error" in pipeline_result:
            raise RuntimeError(
                pipeline_result["error"]
            )

        api_result = build_api_result(
            ticker,
            pipeline_result,
        )

        save_cache(
            ticker,
            days,
            api_result,
        )

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = "finished"
            jobs[job_id]["result"] = api_result
            jobs[job_id]["completed_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

    except Exception as exc:

        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["progress"] = "failed"
            jobs[job_id]["error"] = str(exc)

def create_pipeline_job(
    ticker: str,
    days: int,
) -> str:

    with jobs_lock:
        active_job = next(
            (
                job
                for job in jobs.values()
                if job["status"] in ("queued", "running")
            ),
            None,
        )

        if active_job is not None:
            raise HTTPException(
                status_code=409,
                detail="A pipeline job is already running.",
            )

        job_id = str(uuid.uuid4())

        started_at = datetime.now(timezone.utc).isoformat()

        jobs[job_id] = {
            "job_id": job_id,
            "ticker": ticker,
            "status": "queued",
            "progress": "queued",
            "started_at": started_at,
        }

    executor.submit(
        execute_pipeline_job,
        job_id,
        ticker,
        days,
    )

    return job_id


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Stock Prediction API is running"
    }

@app.get("/predict")
def predict(
    ticker: str,
    days: int = 30,
    force_refresh: bool = False,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
):
    ticker = normalize_ticker(ticker)

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Ticker is required.",
        )

    if days <= 0 or days > 365:
        raise HTTPException(
            status_code=400,
            detail="days must be between 1 and 365.",
        )

    if not force_refresh:
        cached_result = load_cache(
            ticker,
            days,
        )

        if cached_result and is_cache_fresh(
            cached_result
        ):
            return cached_result

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="API key is required to generate a new prediction.",
        )

    require_api_key(credentials)

    job_id = create_pipeline_job(
        ticker,
        days,
    )

    return {
        "status": "processing",
        "job_id": job_id,
        "ticker": ticker,
    }
@app.post("/retrain")
def retrain(
    request: RetrainRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    require_api_key(credentials)

    ticker = normalize_ticker(
        request.ticker
    )

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Ticker is required.",
        )

    job_id = create_pipeline_job(
        ticker,
        30,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "ticker": ticker,
        "started_at": jobs[job_id]["started_at"],
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):

    with jobs_lock:

        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found.",
            )

        return dict(job)
