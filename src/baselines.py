"""
Baseline models for comparison: Naive (zero-change) and ARIMA.

Both baselines follow the same interface as the main models and integrate
with the walk-forward validation framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from evaluate import compute_metrics, print_metrics


def naive_baseline(
    prices: np.ndarray,
    dates: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Any]:
    """
    Naive baseline: predict tomorrow's close = today's close.

    Parameters
    ----------
    prices : array
        Full price series (Close).
    dates : array
        Corresponding date array.
    train_idx, test_idx : arrays
        Walk-forward fold indices.

    Returns
    -------
    dict with y_true, y_pred, test_dates, metrics.
    """
    # For the test period, the prediction for day T is the close of day T-1
    # We need the day *before* each test day as the prediction
    y_true = prices[test_idx]
    # Naive: predict close[T] = close[T-1]
    pred_idx = test_idx - 1
    # Ensure we don't go below 0
    pred_idx = np.clip(pred_idx, 0, len(prices) - 1)
    y_pred = prices[pred_idx]
    test_dates = dates[test_idx]

    metrics = compute_metrics(y_true, y_pred, model_name="Naive (Zero-Change)")
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "test_dates": test_dates,
        "metrics": metrics,
    }


def arima_baseline(
    prices: np.ndarray,
    dates: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    order: Tuple[int, int, int] = (5, 1, 0),
) -> Dict[str, Any]:
    """
    ARIMA baseline: fit ARIMA on training prices, forecast test period.

    Uses statsmodels ARIMA with a fixed order (default (5,1,0)).

    Parameters
    ----------
    prices : array
        Full price series (Close).
    dates : array
        Corresponding date array.
    train_idx, test_idx : arrays
        Walk-forward fold indices.
    order : tuple
        ARIMA (p, d, q) order.

    Returns
    -------
    dict with y_true, y_pred, test_dates, metrics.
    """
    from statsmodels.tsa.arima.model import ARIMA
    import warnings

    train_prices = prices[train_idx]
    y_true = prices[test_idx]
    test_dates = dates[test_idx]
    n_forecast = len(test_idx)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train_prices, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=n_forecast)
            y_pred = np.asarray(forecast, dtype=np.float64)
    except Exception as e:
        print(f"  ARIMA fitting failed: {e}. Falling back to naive.")
        # Fallback to naive
        y_pred = np.full(n_forecast, train_prices[-1])

    metrics = compute_metrics(y_true, y_pred, model_name="ARIMA (5,1,0)")
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "test_dates": test_dates,
        "metrics": metrics,
    }


def run_baselines_walkforward(
    prices: np.ndarray,
    dates: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Dict[str, Any]]:
    """
    Run both baselines across all walk-forward folds and aggregate.

    Parameters
    ----------
    prices : array
        Full Close price series.
    dates : array
        Corresponding dates.
    folds : list of (train_idx, test_idx)
        Walk-forward folds.

    Returns
    -------
    dict with keys "naive" and "arima", each containing aggregated results.
    """
    from walk_forward import aggregate_fold_results

    naive_folds = []
    arima_folds = []

    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"  Baselines — Fold {i+1}/{len(folds)}...")
        naive_res = naive_baseline(prices, dates, train_idx, test_idx)
        arima_res = arima_baseline(prices, dates, train_idx, test_idx)
        naive_folds.append(naive_res)
        arima_folds.append(arima_res)

    # Aggregate
    naive_agg = aggregate_fold_results(naive_folds)
    arima_agg = aggregate_fold_results(arima_folds)

    # Compute aggregate metrics
    naive_metrics = compute_metrics(
        naive_agg["y_true"], naive_agg["y_pred"], model_name="Naive (Zero-Change)"
    )
    arima_metrics = compute_metrics(
        arima_agg["y_true"], arima_agg["y_pred"], model_name="ARIMA (5,1,0)"
    )

    print_metrics(naive_metrics)
    print_metrics(arima_metrics)

    return {
        "naive": {**naive_agg, "metrics": naive_metrics},
        "arima": {**arima_agg, "metrics": arima_metrics},
    }
