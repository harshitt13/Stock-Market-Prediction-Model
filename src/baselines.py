"""Baselines, on the same contract as the models.

REFACTOR_PLAN.md section 6. Each baseline consumes the same Dataset and the
same folds and emits the same standard result frame, so comparing a model to a
baseline is an exact merge on ``target_date`` rather than an assumption that
two similarly-shaped arrays describe the same days.

The one that matters is :func:`zero_return_baseline`. In return space the old
"predict tomorrow's close = today's close" naive baseline *is* a prediction of
zero return, and it is the benchmark every model here has to beat. Most of
them will not, and that is the honest result.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dataset import Dataset
from model_utils import assemble_predictions, default_folds, fold_predictions

Folds = List[Tuple[np.ndarray, np.ndarray]]


def _run_baseline(
    dataset: Dataset,
    fold_indices: Optional[Folds],
    model_name: str,
    predict_fold,
) -> Dict[str, Any]:
    """Apply ``predict_fold(train_idx, test_idx) -> y_pred`` over every fold."""
    n = len(dataset)
    if fold_indices is None:
        fold_indices = default_folds(n)

    frames = []
    for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
        train_idx = np.asarray(train_idx)[np.asarray(train_idx) < n]
        test_idx = np.asarray(test_idx)[np.asarray(test_idx) < n]
        if len(test_idx) == 0:
            continue
        y_pred = np.asarray(predict_fold(train_idx, test_idx), dtype=float)
        if len(y_pred) != len(test_idx):
            raise RuntimeError(
                f"{model_name}: produced {len(y_pred)} predictions for "
                f"{len(test_idx)} test rows"
            )
        frames.append(fold_predictions(dataset, fold_id, test_idx, y_pred))

    return {
        "model_name": model_name,
        "predictions": assemble_predictions(frames, model_name),
    }


def zero_return_baseline(
    dataset: Dataset, fold_indices: Optional[Folds] = None
) -> Dict[str, Any]:
    """``y_pred = 0`` everywhere.

    This is what the naive price baseline actually was: predicting no change.
    It is the benchmark, not a strawman -- for daily equity returns it is hard
    to beat, and a model that cannot beat it has found nothing.
    """
    return _run_baseline(
        dataset,
        fold_indices,
        "Zero return",
        lambda train_idx, test_idx: np.zeros(len(test_idx)),
    )


def historical_mean_baseline(
    dataset: Dataset, fold_indices: Optional[Folds] = None
) -> Dict[str, Any]:
    """Expanding-window mean of every return observed before each day.

    Seeded with the training returns and expanded with realised test returns.
    Never the test-set mean, which would use the future.
    """
    from evaluate import expanding_mean_benchmark

    def predict(train_idx, test_idx):
        return expanding_mean_benchmark(dataset.y[test_idx], dataset.y[train_idx])

    return _run_baseline(dataset, fold_indices, "Historical mean", predict)


def ar1_baseline(
    dataset: Dataset, fold_indices: Optional[Folds] = None
) -> Dict[str, Any]:
    """AR(1) on returns: ``y_hat[t] = c + phi * y[t-1]``.

    ``y[t-1]`` is the return realised from close t-1 to close t, so it is known
    at the close of day t, when the forecast for t+1 is made.
    """

    def predict(train_idx, test_idx):
        train_idx = np.asarray(train_idx)
        # Regress y[i] on y[i-1] over the training rows that have a predecessor.
        usable = train_idx[train_idx > 0]
        if len(usable) < 2:
            return np.zeros(len(test_idx))
        phi, c = np.polyfit(dataset.y[usable - 1], dataset.y[usable], 1)

        prev = np.where(test_idx > 0, dataset.y[np.maximum(test_idx - 1, 0)], 0.0)
        return c + phi * prev

    return _run_baseline(dataset, fold_indices, "AR(1) returns", predict)


def arima_baseline(
    dataset: Dataset,
    fold_indices: Optional[Folds] = None,
    order: Tuple[int, int, int] = (5, 0, 0),
) -> Dict[str, Any]:
    """ARIMA on log returns, not on prices.

    Parameters are estimated on the training fold only. Test-fold forecasts are
    genuine one-step-ahead predictions: the realised returns are appended with
    ``refit=False``, so each prediction conditions on data before it while the
    coefficients stay frozen at their training values.

    ``d=0`` by default because returns are already differenced; the old (5,1,0)
    was differencing a price series.
    """
    from statsmodels.tsa.arima.model import ARIMA

    def predict(train_idx, test_idx):
        y_train = dataset.y[train_idx]
        y_test = dataset.y[test_idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(y_train, order=order).fit()
                extended = fitted.append(y_test, refit=False)
                predictions = extended.predict(
                    start=len(y_train), end=len(y_train) + len(y_test) - 1
                )
            return np.asarray(predictions, dtype=float)
        except Exception as exc:  # pragma: no cover - depends on statsmodels
            print(f"  ARIMA fit failed ({exc}); falling back to the training mean.")
            return np.full(len(test_idx), float(np.mean(y_train)))

    return _run_baseline(
        dataset, fold_indices, f"ARIMA{order} returns", predict
    )


def random_sign_baseline(
    dataset: Dataset,
    fold_indices: Optional[Folds] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Random signs at the training-fold return scale.

    A sanity check for directional accuracy: anything that cannot beat this is
    not forecasting direction. Its expected accuracy is 50%, and seeing a real
    model land near it is informative rather than embarrassing.
    """
    rng = np.random.default_rng(seed)

    def predict(train_idx, test_idx):
        scale = float(np.std(dataset.y[train_idx])) or 1.0
        return rng.choice([-1.0, 1.0], size=len(test_idx)) * scale

    return _run_baseline(dataset, fold_indices, "Random sign", predict)


#: Every baseline, in the order they belong in the results table.
BASELINES = {
    "zero_return": zero_return_baseline,
    "historical_mean": historical_mean_baseline,
    "ar1": ar1_baseline,
    "arima": arima_baseline,
    "random_sign": random_sign_baseline,
}


def run_all_baselines(
    dataset: Dataset,
    fold_indices: Optional[Folds] = None,
    seed: int = 42,
    include: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run every baseline over the same folds.

    All of them emit identical ``target_date`` values, which is what makes the
    comparison exact rather than approximately-the-same-distribution.
    """
    names = list(BASELINES) if include is None else list(include)
    results: Dict[str, Dict[str, Any]] = {}

    for name in names:
        if name not in BASELINES:
            raise KeyError(f"unknown baseline {name!r}; choose from {list(BASELINES)}")
        print(f"  Baseline: {name}")
        if name == "random_sign":
            results[name] = BASELINES[name](dataset, fold_indices, seed=seed)
        else:
            results[name] = BASELINES[name](dataset, fold_indices)

    return results
