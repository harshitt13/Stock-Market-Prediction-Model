"""The prediction contract every model must satisfy.

From REFACTOR_PLAN.md section 1. Every model, including the baselines, returns
a tidy frame keyed by ``target_date`` -- the day being forecast, never the day
the features came from. Keying by ``target_date`` is what removes the
off-by-one between the tree model and the sequence models: a sequence model
consuming a 90-day window and a tree consuming one row both name the same
forecast day, so their predictions merge on an exact key instead of on an
assumption about alignment.

Call :func:`validate_predictions` at the end of every train function. It is
meant to fail loudly: a silent contract violation shows up as a plausible
number in a results table, not as a crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Exactly the columns of a standard result frame, in order.
PREDICTION_COLUMNS = ["target_date", "fold_id", "close_t", "y_true", "y_pred"]


class ContractViolation(AssertionError):
    """A result frame does not satisfy the prediction contract."""


def _fail(name: str, message: str) -> None:
    raise ContractViolation(f"{name}: {message}")


def validate_predictions(df: pd.DataFrame, name: str = "predictions") -> pd.DataFrame:
    """Assert the standard result schema and return the frame unchanged.

    Checks, in order: it is a DataFrame; it has exactly ``PREDICTION_COLUMNS``;
    the dtypes are right; there are no NaNs; ``target_date`` is unique and
    sorted ascending.

    Raises
    ------
    ContractViolation
        On any failure, with the offending column named.
    """
    if not isinstance(df, pd.DataFrame):
        _fail(name, f"expected a DataFrame, got {type(df).__name__}")

    actual = list(df.columns)
    if actual != PREDICTION_COLUMNS:
        missing = [c for c in PREDICTION_COLUMNS if c not in actual]
        extra = [c for c in actual if c not in PREDICTION_COLUMNS]
        _fail(
            name,
            f"columns must be exactly {PREDICTION_COLUMNS}, got {actual}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else ""),
        )

    if len(df) == 0:
        _fail(name, "no rows; a model that predicts nothing is a bug, not a result")

    if not pd.api.types.is_datetime64_any_dtype(df["target_date"]):
        _fail(name, f"target_date must be datetime64, got {df['target_date'].dtype}")

    if df["fold_id"].dtype.kind not in "iu":
        _fail(name, f"fold_id must be an integer, got {df['fold_id'].dtype}")

    for col in ("close_t", "y_true", "y_pred"):
        if df[col].dtype.kind != "f":
            _fail(name, f"{col} must be float, got {df[col].dtype}")

    nan_counts = {c: int(df[c].isna().sum()) for c in PREDICTION_COLUMNS}
    if any(nan_counts.values()):
        offenders = {c: n for c, n in nan_counts.items() if n}
        _fail(name, f"NaNs are not allowed: {offenders}")

    if not np.isfinite(df[["close_t", "y_true", "y_pred"]].to_numpy(float)).all():
        _fail(name, "close_t, y_true and y_pred must all be finite")

    if df["target_date"].duplicated().any():
        dupes = df.loc[df["target_date"].duplicated(), "target_date"].unique()[:5]
        _fail(
            name,
            f"target_date must be unique; {len(dupes)}+ duplicates, e.g. {list(dupes)}. "
            "Overlapping walk-forward test folds would do this.",
        )

    if not df["target_date"].is_monotonic_increasing:
        _fail(name, "target_date must be sorted ascending")

    if (df["close_t"] <= 0).any():
        _fail(name, "close_t must be positive; price reconstruction takes its log")

    return df


def reconstruct_close(close_t, y) -> np.ndarray:
    """``close_hat[t+1] = close_t[t] * exp(y_hat[t])``.

    Price is a display quantity only. Never compute a primary metric on the
    output of this function -- the naive baseline scores about the same on any
    of them, so they measure the price series, not the forecast.
    """
    return np.asarray(close_t, dtype=float) * np.exp(np.asarray(y, dtype=float))


def empty_predictions() -> pd.DataFrame:
    """A correctly typed, zero-row result frame, for building up per fold."""
    return pd.DataFrame(
        {
            "target_date": pd.Series([], dtype="datetime64[ns]"),
            "fold_id": pd.Series([], dtype="int64"),
            "close_t": pd.Series([], dtype="float64"),
            "y_true": pd.Series([], dtype="float64"),
            "y_pred": pd.Series([], dtype="float64"),
        }
    )


def make_predictions(target_date, fold_id, close_t, y_true, y_pred) -> pd.DataFrame:
    """Assemble a standard result frame with the right dtypes and order.

    Sorts by ``target_date`` so that per-fold frames concatenated by a caller
    still satisfy the contract.
    """
    df = pd.DataFrame(
        {
            "target_date": pd.to_datetime(pd.Series(target_date)).to_numpy("datetime64[ns]"),
            "fold_id": np.asarray(fold_id, dtype="int64"),
            "close_t": np.asarray(close_t, dtype="float64"),
            "y_true": np.asarray(y_true, dtype="float64"),
            "y_pred": np.asarray(y_pred, dtype="float64"),
        }
    )
    return df.sort_values("target_date").reset_index(drop=True)
