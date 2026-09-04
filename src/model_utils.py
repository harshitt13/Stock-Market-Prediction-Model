"""Shared plumbing for the model modules.

The fold-scaling and demo-forecast code lives here rather than being copied
into each model, because these are exactly the places a leak hides: a scaler
fitted one line too early sees the test fold and nothing crashes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from contracts import make_predictions, validate_predictions
from dataset import Dataset

#: Printed and stored wherever a recursive forecast is produced.
DEMO_FORECAST_CAVEAT = (
    "ILLUSTRATIVE ONLY - excluded from every scientific claim. Each recursive "
    "step sets High = Low = Close = the predicted price, which destroys the "
    "ATR, Bollinger and intraday-range features, and the errors compound over "
    "the horizon. For real multi-horizon results, train separate direct models "
    "for h = 1, 5, 21 rather than recursing."
)


def fit_feature_scaler(dataset: Dataset, train_idx: np.ndarray) -> StandardScaler:
    """Fit on the training rows of one fold and nothing else.

    StandardScaler, not MinMaxScaler: returns are roughly symmetric and
    unbounded, whereas MinMax squeezes the training range into [0, 1] and then
    puts every test value that exceeds the training extremes outside it. On a
    trending series that is most of the test set.
    """
    return StandardScaler().fit(dataset.X[np.asarray(train_idx)])


def apply_feature_scaler(dataset: Dataset, scaler: StandardScaler) -> Dataset:
    """A copy of ``dataset`` with X scaled. Keys are untouched."""
    return replace(dataset, X=scaler.transform(dataset.X))


def fit_target_scaler(y_train) -> StandardScaler:
    """Standardise the target on the training fold.

    Daily log returns have a standard deviation around 0.02, so an unscaled
    Huber loss with delta=1.0 is purely quadratic and the gradients are tiny.
    Scaling is fitted on training returns only and inverted before any metric
    is computed.
    """
    return StandardScaler().fit(np.asarray(y_train, dtype=float).reshape(-1, 1))


def scale_targets(scaler: StandardScaler, y) -> np.ndarray:
    return scaler.transform(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()


def unscale_targets(scaler: StandardScaler, y) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()


def default_folds(n: int, train_fraction: float = 0.8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """A single chronological split, for callers that pass no folds."""
    split = int(n * train_fraction)
    if split < 1 or split >= n:
        raise ValueError(f"cannot split {n} rows at fraction {train_fraction}")
    return [(np.arange(0, split), np.arange(split, n))]


def assemble_predictions(fold_frames: List[pd.DataFrame], model_name: str) -> pd.DataFrame:
    """Concatenate per-fold result frames and enforce the contract."""
    if not fold_frames:
        raise RuntimeError(f"{model_name}: no fold produced predictions")
    combined = (
        pd.concat(fold_frames, ignore_index=True)
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    return validate_predictions(combined, name=model_name)


def fold_predictions(
    dataset: Dataset,
    fold_id: int,
    row_index: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """One fold's rows as a standard result frame, keyed by target_date."""
    row_index = np.asarray(row_index, dtype=int)
    return make_predictions(
        target_date=dataset.target_date[row_index],
        fold_id=np.full(len(row_index), fold_id, dtype=int),
        close_t=dataset.close_t[row_index],
        y_true=dataset.y[row_index],
        y_pred=np.asarray(y_pred, dtype=float),
    )


def recursive_demo_forecast(
    history: pd.DataFrame,
    predict_next_return: Callable[[pd.DataFrame], float],
    horizon: int,
    label: str = "Predicted Close",
) -> pd.DataFrame:
    """Recursive multi-step forecast. A demo feature, not a result.

    See :data:`DEMO_FORECAST_CAVEAT`. Kept because it makes a nice chart and
    because removing a feature users have seen is its own kind of dishonesty,
    but it must never appear in a metrics table.

    ``predict_next_return`` receives the engineered history so far and returns
    the predicted log return for the next trading day. The price is then
    reconstructed as ``Close * exp(y_hat)``, which at least keeps the demo in
    the same units as the real model rather than inventing a second target.
    """
    from fetch_data import engineer_features

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    history = history.copy().reset_index(drop=True)
    last_date = pd.to_datetime(history["Date"]).max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon, freq="B"
    )

    prices: List[float] = []
    returns: List[float] = []

    for step in range(horizon):
        y_hat = float(predict_next_return(history))
        close_now = float(history["Close"].iloc[-1])
        close_next = close_now * np.exp(y_hat)

        returns.append(y_hat)
        prices.append(close_next)

        new_row = history.iloc[-1].copy()
        new_row["Date"] = future_dates[step]
        new_row["Open"] = close_now
        # The indefensible part: a synthetic bar has no real high or low.
        new_row["Close"] = close_next
        new_row["High"] = max(close_now, close_next)
        new_row["Low"] = min(close_now, close_next)

        history.loc[len(history)] = new_row
        history = engineer_features(history)

    return pd.DataFrame(
        {
            "date": future_dates,
            "predicted_log_return": returns,
            label: prices,
            "caveat": DEMO_FORECAST_CAVEAT,
        }
    )
