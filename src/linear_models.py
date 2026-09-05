"""Two well-regularised linear comparators, on the same contract as every
other model.

They exist to span the capacity range from linear to Transformer with
something cheap at the bottom of it, so that "you built bad models" can be
met with a model that has almost no capacity to misbuild.

- ``Ridge (returns)``: standardised features, RidgeCV over the same penalty
  grid as the meta-learner, fitted per fold on the training rows, predicting
  the next-day log return. Its selected penalty per fold is recorded so that
  whether it collapses towards its intercept, as the meta-learner does, can
  be read off rather than guessed.
- ``Logistic (direction)``: standardised features, L2 logistic regression
  with the inverse penalty chosen by a time-respecting inner cross-validation
  on the training rows, predicting the probability that the next-day return
  is positive. The contract needs a return-scale number, so its output is
  ``(2p - 1)`` times the training fold's return standard deviation: the sign
  is the class decision, the magnitude is confidence at the training scale.
  Its R2_OOS is reported for completeness and is not a return forecast.

Both are fitted from scratch on each fold's training rows; nothing is tuned
against a test fold.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from dataset import Dataset
from model_utils import assemble_predictions, clip_fold, default_folds, fit_feature_scaler, fold_predictions

Folds = Sequence[Tuple[np.ndarray, np.ndarray]]

#: The meta-learner's grid, so the two ridge fits are comparable.
RIDGE_ALPHAS = tuple(10.0 ** np.arange(-3, 7))
#: Inverse-penalty grid for the logistic classifier, decades from 1e-4 to 1e4.
LOGISTIC_CS = tuple(10.0 ** np.arange(-4, 5))
RIDGE_NAME = "Ridge (returns)"
LOGISTIC_NAME = "Logistic (direction)"


def _run(dataset: Dataset, folds: Optional[Folds], name: str, fit_predict) -> Dict[str, Any]:
    """Apply ``fit_predict(train_idx, test_idx) -> (y_pred, fit_record)`` per fold."""
    n = len(dataset)
    if folds is None:
        folds = default_folds(n)
    frames: List[pd.DataFrame] = []
    fits: List[Dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train_idx, test_idx = clip_fold(train_idx, test_idx, n)
        if len(test_idx) == 0:
            continue
        y_pred, record = fit_predict(train_idx, test_idx)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(test_idx):
            raise RuntimeError(f"{name}: produced {len(y_pred)} predictions for {len(test_idx)} test rows")
        frames.append(fold_predictions(dataset, fold_id, test_idx, y_pred))
        fits.append({"fold_id": fold_id, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), **record})
    return {
        "model_name": name,
        "predictions": assemble_predictions(frames, name),
        "fold_fits": pd.DataFrame(fits),
    }


def train_ridge_model(dataset: Dataset, folds: Optional[Folds] = None) -> Dict[str, Any]:
    """RidgeCV on standardised features, per fold, predicting the log return."""

    def fit_predict(train_idx, test_idx):
        scaler = fit_feature_scaler(dataset, train_idx)
        x_train = scaler.transform(dataset.X[train_idx])
        y_train = dataset.y[train_idx]
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(x_train, y_train)
        y_pred = model.predict(scaler.transform(dataset.X[test_idx]))
        return y_pred, {
            "alpha": float(model.alpha_),
            "intercept": float(model.intercept_),
            "coef_abs_sum": float(np.abs(model.coef_).sum()),
            "coef_abs_max": float(np.abs(model.coef_).max()),
            "pred_sd_over_train_sd": float(np.std(y_pred) / (np.std(y_train) or 1.0)),
        }

    return _run(dataset, folds, RIDGE_NAME, fit_predict)


def train_logistic_model(
    dataset: Dataset, folds: Optional[Folds] = None, seed: int = 42, inner_splits: int = 5
) -> Dict[str, Any]:
    """L2 logistic regression on the sign of the return, per fold.

    The inverse penalty is chosen by ``LogisticRegressionCV`` over
    ``LOGISTIC_CS`` with a ``TimeSeriesSplit`` inner cross-validation on the
    training rows, so the selection respects time. The output is
    ``(2p - 1) * sd(y_train)``.
    """

    def fit_predict(train_idx, test_idx):
        scaler = fit_feature_scaler(dataset, train_idx)
        x_train = scaler.transform(dataset.X[train_idx])
        y_train = dataset.y[train_idx]
        labels = (y_train > 0).astype(int)
        scale = float(np.std(y_train)) or 1.0
        if labels.min() == labels.max():
            # One class only: no direction to learn. Predict the majority sign.
            sign = 1.0 if labels[0] == 1 else -1.0
            return np.full(len(test_idx), sign * scale), {"C": float("nan"), "intercept": float("nan"),
                                                           "p_up_mean": float(labels[0]), "single_class": True}
        with warnings.catch_warnings():
            # scikit-learn 1.9 announces a change to LogisticRegressionCV's fitted
            # attributes in 1.10; the attributes read below exist in both layouts.
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            model = LogisticRegressionCV(
                Cs=list(LOGISTIC_CS), cv=TimeSeriesSplit(n_splits=inner_splits), penalty="l2",
                scoring="neg_log_loss", max_iter=2000, random_state=seed,
            ).fit(x_train, labels)
        p_up = model.predict_proba(scaler.transform(dataset.X[test_idx]))[:, 1]
        return (2.0 * p_up - 1.0) * scale, {
            "C": float(np.ravel(model.C_)[0]),
            "intercept": float(np.ravel(model.intercept_)[0]),
            "coef_abs_sum": float(np.abs(model.coef_).sum()),
            "p_up_mean": float(p_up.mean()),
            "p_up_sd": float(p_up.std()),
            "single_class": False,
        }

    return _run(dataset, folds, LOGISTIC_NAME, fit_predict)


def run_linear_comparators(dataset: Dataset, folds: Optional[Folds] = None, seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """Both comparators over the same folds, keyed like ``run_all_baselines``."""
    return {
        "ridge": train_ridge_model(dataset, folds),
        "logistic": train_logistic_model(dataset, folds, seed=seed),
    }
