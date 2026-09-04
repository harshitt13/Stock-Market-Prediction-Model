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

from dataclasses import dataclass

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


@dataclass(frozen=True)
class EvaluationWindow:
    """The set of forecast days every model has in common.

    Fold-respecting stacking forbids the meta-learner from forecasting the
    earliest folds, so it covers fewer days than the base models. Scoring each
    model over its own range compares a model measured across a volatile
    stretch against one that never saw it -- on the AAPL fixture the excluded
    folds are the 2020 crash, which is exactly the period that would flatter or
    punish a model most. Every primary comparison runs on this window instead.
    """

    dates: np.ndarray
    dropped_by_model: dict
    reference_model: str

    @property
    def n(self) -> int:
        return len(self.dates)

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates[0])

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates[-1])

    def describe(self) -> str:
        return (
            f"{self.n} forecast days, "
            f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"
        )


def _prediction_frames(results_by_model: dict) -> dict:
    frames = {}
    for name, item in results_by_model.items():
        df = item["predictions"] if isinstance(item, dict) else item
        validate_predictions(df, name=name)
        frames[name] = df
    return frames


def common_evaluation_window(
    results_by_model: dict, require_contiguous: bool = True
) -> EvaluationWindow:
    """Intersect ``target_date`` across every model, including the meta.

    Raises if the intersection is empty, or (by default) if it is not a
    contiguous run of trading days within the longest-covering model. A hole in
    the middle would mean some model is missing days the others have, which is
    a bug rather than a windowing decision.
    """
    frames = _prediction_frames(results_by_model)
    if not frames:
        raise ValueError("no models to window")

    date_sets = {name: set(df["target_date"]) for name, df in frames.items()}
    shared = set.intersection(*date_sets.values())
    if not shared:
        raise ContractViolation(
            "models share no forecast days at all; a common evaluation window "
            "does not exist. Check that every model ran on the same folds."
        )

    reference_model = max(frames, key=lambda n: len(frames[n]))
    reference_dates = frames[reference_model]["target_date"].to_numpy()
    ordered = np.sort(np.array(sorted(shared), dtype="datetime64[ns]"))

    if require_contiguous:
        positions = np.flatnonzero(np.isin(reference_dates, ordered))
        span = positions[-1] - positions[0] + 1
        if span != len(positions):
            missing = span - len(positions)
            raise ContractViolation(
                f"the shared window is not contiguous: {missing} day(s) inside "
                f"{pd.Timestamp(ordered[0]):%Y-%m-%d}.."
                f"{pd.Timestamp(ordered[-1]):%Y-%m-%d} are missing from at "
                "least one model. That is a bug, not a windowing choice."
            )

    return EvaluationWindow(
        dates=ordered,
        dropped_by_model={n: len(df) - len(shared) for n, df in frames.items()},
        reference_model=reference_model,
    )


def restrict_to_window(
    predictions: pd.DataFrame, window: EvaluationWindow, name: str = "predictions"
) -> pd.DataFrame:
    """Cut a result frame down to the common window, revalidating after."""
    df = predictions[predictions["target_date"].isin(window.dates)]
    df = df.sort_values("target_date").reset_index(drop=True)
    if len(df) != window.n:
        _fail(
            name,
            f"has {len(df)} of the window's {window.n} days; it does not cover "
            "the common evaluation window",
        )
    return validate_predictions(df, name=name)


def restrict_all(results_by_model: dict, window: EvaluationWindow) -> dict:
    """Every model, cut to the common window. Keys preserved."""
    return {
        name: restrict_to_window(df, window, name)
        for name, df in _prediction_frames(results_by_model).items()
    }


def align_predictions(
    results_by_model: dict, require_identical: bool = True
) -> pd.DataFrame:
    """Merge several models' result frames into one wide frame on target_date.

    Because every model and baseline emits the same ``target_date`` values,
    alignment is a merge with an assertion rather than a reindexing exercise.
    ``require_identical`` enforces that the date sets match exactly; set it
    False to fall back to the intersection (a sequence model with a longer
    lookback, say), in which case the shrinkage is reported.

    Returns a frame with ``target_date``, ``close_t``, ``y_true`` and one
    ``y_pred_<model>`` column per model.
    """
    if not results_by_model:
        raise ValueError("no models to align")

    frames = {}
    for name, item in results_by_model.items():
        df = item["predictions"] if isinstance(item, dict) else item
        validate_predictions(df, name=name)
        frames[name] = df

    date_sets = {name: set(df["target_date"]) for name, df in frames.items()}
    reference_name, reference_dates = next(iter(date_sets.items()))

    if require_identical:
        for name, dates in date_sets.items():
            if dates != reference_dates:
                only_ref = len(reference_dates - dates)
                only_this = len(dates - reference_dates)
                _fail(
                    name,
                    f"target_date set differs from {reference_name!r}: "
                    f"{only_ref} days missing here, {only_this} extra. "
                    "Every model must forecast the same days.",
                )

    shared = set.intersection(*date_sets.values())
    if not shared:
        raise ContractViolation("models share no forecast days at all")
    if len(shared) < len(reference_dates):
        print(
            f"align_predictions: keeping {len(shared)} shared days "
            f"of {len(reference_dates)}"
        )

    keep = sorted(shared)
    out = frames[reference_name][
        frames[reference_name]["target_date"].isin(keep)
    ][["target_date", "close_t", "y_true"]].reset_index(drop=True)

    for name, df in frames.items():
        subset = df[df["target_date"].isin(keep)].sort_values("target_date")
        out[f"y_pred_{name}"] = subset["y_pred"].to_numpy()

    return out


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
