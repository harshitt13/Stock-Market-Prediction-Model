"""XGBoost + Random Forest ensemble, forecasting next-day log return.

Ported to the contract in REFACTOR_PLAN.md section 1: consumes a
:class:`~dataset.Dataset`, predicts ``y`` directly, and returns predictions
keyed by ``target_date``.

There is no scaler here any more. Trees split on rank, so standardising the
inputs changes nothing about the fitted model; the old StandardScaler was pure
ceremony. What did matter was the old target: next-day *Close*, against a
feature matrix that included today's Close. That is the identity function with
extra steps, and it is what produced the previous R2 of 0.975.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

from dataset import Dataset
from model_utils import (
    assemble_predictions,
    default_folds,
    fold_predictions,
    recursive_demo_forecast,
)

MODEL_NAME = "Tree Ensemble"
MODEL_PATH = "models/tree_ensemble_model.pkl"


def _build_tree_ensemble(random_state: int = 42) -> VotingRegressor:
    """XGBoost + RF VotingRegressor (deterministic given the seed)."""
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=0,
    )
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    return VotingRegressor(estimators=[("xgb", xgb), ("rf", rf)])


def train_tree_on_fold(
    dataset: Dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_id: int,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit on one fold's training rows, predict its test rows."""
    model = _build_tree_ensemble(random_state)
    model.fit(dataset.X[train_idx], dataset.y[train_idx])
    y_pred = model.predict(dataset.X[test_idx])

    return {
        "model": model,
        "fold_id": fold_id,
        "predictions": fold_predictions(dataset, fold_id, test_idx, y_pred),
        "feature_importances": model.named_estimators_["xgb"].feature_importances_,
    }


def train_tree_model(
    dataset: Dataset,
    fold_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    random_state: int = 42,
    save_model: bool = True,
    demo_forecast_days: int = 0,
    demo_history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Walk-forward training of the tree ensemble on log returns.

    Parameters
    ----------
    dataset
        Output of :func:`dataset.build_dataset`. The target is already defined;
        this function does not construct one.
    fold_indices
        ``(train_idx, test_idx)`` pairs indexing dataset rows. Test folds must
        not overlap, or the resulting frame will have duplicate target_dates
        and fail validation.
    demo_forecast_days
        If > 0, also produce a recursive multi-step forecast. Illustrative
        only; see ``model_utils.DEMO_FORECAST_CAVEAT``.

    Returns
    -------
    dict with ``predictions`` (the standard result frame), the last fold's
    model, and per-fold feature importances.
    """
    n = len(dataset)
    if fold_indices is None:
        fold_indices = default_folds(n)

    fold_frames: List[pd.DataFrame] = []
    importances: List[np.ndarray] = []
    last_model = None

    for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
        train_idx = np.asarray(train_idx)[np.asarray(train_idx) < n]
        test_idx = np.asarray(test_idx)[np.asarray(test_idx) < n]
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"  Tree fold {fold_id + 1}: skipped, empty train or test")
            continue

        print(
            f"  Tree fold {fold_id + 1}/{len(fold_indices)} - "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )
        result = train_tree_on_fold(dataset, train_idx, test_idx, fold_id, random_state)
        fold_frames.append(result["predictions"])
        importances.append(result["feature_importances"])
        last_model = result["model"]

    predictions = assemble_predictions(fold_frames, MODEL_NAME)

    if save_model and last_model is not None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(
            {"model": last_model, "feature_names": dataset.feature_names}, MODEL_PATH
        )
        print(f"Model saved to '{MODEL_PATH}'")

    output: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "predictions": predictions,
        "model": last_model,
        "feature_names": list(dataset.feature_names),
        "feature_importances": importances[-1] if importances else None,
        "mean_feature_importances": (
            np.mean(importances, axis=0) if importances else None
        ),
    }

    if demo_forecast_days > 0:
        output["demo_forecast"] = _demo_forecast(
            last_model, dataset, demo_history, demo_forecast_days
        )

    return output


def _demo_forecast(model, dataset: Dataset, history, horizon: int) -> pd.DataFrame:
    """Recursive forecast for display. Never enters a metrics table."""
    if history is None:
        raise ValueError("demo_forecast_days requires demo_history (the raw frame)")

    feature_names = list(dataset.feature_names)

    def predict_next_return(engineered: pd.DataFrame) -> float:
        row = engineered.iloc[-1][feature_names].to_numpy(dtype=float).reshape(1, -1)
        return float(model.predict(row)[0])

    forecast = recursive_demo_forecast(
        history, predict_next_return, horizon, label="Predicted Close Tree"
    )
    os.makedirs("data", exist_ok=True)
    forecast.to_csv("data/future_predictions_tree.csv", index=False)
    return forecast
