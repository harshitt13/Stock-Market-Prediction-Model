"""Run the two linear comparators everywhere the tree ran, and persist them.

Cheap models only: RidgeCV on returns and an L2 logistic direction
classifier (src/linear_models.py). Two jobs:

1. The headline run: docs/frozen_aapl_raw.csv, the headline 1008/252/252
   grid, seed 42. Written in the sweep's long schema to
   results/headline/AAPL__frozen__seed42__linear.parquet, beside the
   headline parquet, so every loader of that directory sees twelve models.
   The per-fold fits (ridge penalty, logistic inverse penalty, dispersion)
   go to results/linear_fits_headline.csv.
2. The thirty sweep tickers: each ticker's cached raw frame under
   results/raw/ (gitignored), the sweep's own 1008/252/252 grid, seed 42,
   written to results/comparators/ as one parquet per ticker. Kept out of
   results/predictions/ so the committed sweep runs stay as tagged. Per-fold
   fits go to results/linear_fits_sweep.csv.

Each model is fitted once per fold; the long frame and the fits table come
from the same fit.

    python analysis/run_linear_comparators.py
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from dataset import build_dataset  # noqa: E402
from experiments import DEFAULT_TICKERS, PREDICTION_SCHEMA, cached_loader, save_run  # noqa: E402
from linear_models import run_linear_comparators  # noqa: E402
from walk_forward import WalkForwardSplitter  # noqa: E402

FROZEN_CSV = REPO / "docs" / "frozen_aapl_raw.csv"
RAW_DIR = REPO / "results" / "raw"
HEADLINE_OUT = REPO / "results" / "headline" / "AAPL__frozen__seed42__linear.parquet"
SWEEP_DIR = REPO / "results" / "comparators"
FITS_HEADLINE = REPO / "results" / "linear_fits_headline.csv"
FITS_SWEEP = REPO / "results" / "linear_fits_sweep.csv"
FOLD_CFG = (1008, 252, 252)
SEED = 42


def run(raw: pd.DataFrame, ticker: str, regime: str):
    """Fit both comparators once; return the long prediction frame and the fits."""
    ds = build_dataset(raw)
    folds = WalkForwardSplitter(*FOLD_CFG).split(len(ds))
    if not folds:
        return None, None
    results = run_linear_comparators(ds, folds, seed=SEED)
    blocks, fits = [], []
    for result in results.values():
        block = result["predictions"].copy()
        block["model"], block["ticker"], block["regime"], block["seed"] = result["model_name"], ticker, regime, SEED
        blocks.append(block[PREDICTION_SCHEMA])
        f = result["fold_fits"].copy()
        f.insert(0, "model", result["model_name"])
        f.insert(0, "ticker", ticker)
        fits.append(f)
    return pd.concat(blocks, ignore_index=True), pd.concat(fits, ignore_index=True)


def headline() -> None:
    t0 = time.time()
    frame, fits = run(pd.read_csv(FROZEN_CSV, parse_dates=["Date"]), "AAPL", "frozen")
    save_run(frame, str(HEADLINE_OUT))
    fits.to_csv(FITS_HEADLINE, index=False)
    print(f"headline: {len(frame)} rows -> {HEADLINE_OUT.relative_to(REPO)}; fits -> {FITS_HEADLINE.relative_to(REPO)} "
          f"({time.time() - t0:.0f} s)", flush=True)


def sweep() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    load = cached_loader(str(RAW_DIR))
    all_fits = []
    t0 = time.time()
    for i, ticker in enumerate(DEFAULT_TICKERS, 1):
        raw = load(ticker, "2010-01-01", None)
        if raw is None:
            print(f"  {ticker}: no cache, skipped")
            continue
        frame, fits = run(raw, ticker, "full")
        if frame is None:
            print(f"  {ticker}: too short, skipped")
            continue
        save_run(frame, str(SWEEP_DIR / f"{ticker}__full__seed{SEED}.parquet"))
        all_fits.append(fits)
        print(f"  [{i:>2}/30] {ticker:<6} {len(frame)} rows  ({time.time() - t0:.0f} s)", flush=True)
    pd.concat(all_fits, ignore_index=True).to_csv(FITS_SWEEP, index=False)
    print(f"sweep: {len(all_fits)} tickers -> {SWEEP_DIR.relative_to(REPO)}/; fits -> {FITS_SWEEP.relative_to(REPO)}")


if __name__ == "__main__":
    headline()
    sweep()
