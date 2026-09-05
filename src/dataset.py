"""The one place the target and the row alignment are defined.

REFACTOR_PLAN.md section 3. Every model imports :func:`build_dataset`; no model
computes its own target ever again. That is the whole point -- the previous
version defined the target three times, in three different files, with two
different alignments, and the disagreement was invisible because none of the
definitions crashed.

Row semantics, from section 1::

    feature_date[t]  trading day t; every feature uses information available
                     at the close of day t
    target_date[t]   trading day t+1; the day being forecast
    close_t[t]       Close[t], the last observed price
    y[t]             log(Close[t+1] / Close[t]), the next-day log return
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fetch_data import FEATURE_COLUMNS, available_feature_columns, engineer_features


@dataclass(frozen=True)
class Dataset:
    """Aligned features, target and keys for one ticker."""

    X: np.ndarray  # (n, d) features at feature_date
    y: np.ndarray  # (n,)   log return from close_t to the next close
    feature_date: np.ndarray  # (n,) datetime64[ns]
    target_date: np.ndarray  # (n,) datetime64[ns], the next trading day
    close_t: np.ndarray  # (n,) last observed close
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        """Width of the feature matrix."""
        return self.X.shape[1]

    def subset(self, idx) -> "Dataset":
        """Row subset, keys included. Used to slice a fold."""
        idx = np.asarray(idx)
        return Dataset(
            X=self.X[idx],
            y=self.y[idx],
            feature_date=self.feature_date[idx],
            target_date=self.target_date[idx],
            close_t=self.close_t[idx],
            feature_names=list(self.feature_names),
        )


@dataclass(frozen=True)
class Sequences:
    """Lookback windows keyed identically to the parent :class:`Dataset`."""

    X: np.ndarray  # (m, lookback, d)
    y: np.ndarray  # (m,)
    feature_date: np.ndarray
    target_date: np.ndarray
    close_t: np.ndarray
    row_index: np.ndarray  # positions into the parent Dataset
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)


def build_dataset(raw: pd.DataFrame) -> Dataset:
    """Build the aligned dataset for one ticker.

    Accepts either a raw OHLCV frame or one that already carries the engineered
    features; the features are computed if they are not all present.

    The final row is dropped because its target is unknown: there is no
    Close[t+1] for the last observed day.
    """
    if "Close" not in raw.columns or "Date" not in raw.columns:
        raise ValueError("build_dataset needs at least 'Date' and 'Close' columns")

    df = raw.copy()
    if not set(FEATURE_COLUMNS).issubset(df.columns):
        df = engineer_features(df)

    features = available_feature_columns(df)
    if not features:
        raise ValueError("no FEATURE_COLUMNS present after feature engineering")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Warm-up rows are dropped here, not filled, so a NaN never reaches a model
    # disguised as a zero.
    df = df.dropna(subset=features + ["Close"]).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(f"need at least 2 usable rows, got {len(df)}")

    close = df["Close"].to_numpy(dtype=float)
    dates = df["Date"].to_numpy(dtype="datetime64[ns]")

    if (close <= 0).any():
        raise ValueError("Close must be positive; the target is a log ratio")

    n = len(df) - 1  # the last row has no target
    y = np.log(close[1:] / close[:-1])

    dataset = Dataset(
        X=df[features].to_numpy(dtype=float)[:n],
        y=y,
        feature_date=dates[:n],
        target_date=dates[1:],
        close_t=close[:n],
        feature_names=list(features),
    )
    _validate(dataset)
    return dataset


def _validate(dataset: Dataset) -> None:
    n = len(dataset)
    if dataset.X.shape[0] != n:
        raise ValueError(f"X has {dataset.X.shape[0]} rows, y has {n}")
    if not np.isfinite(dataset.X).all():
        bad = [
            name
            for i, name in enumerate(dataset.feature_names)
            if not np.isfinite(dataset.X[:, i]).all()
        ]
        raise ValueError(f"non-finite feature values in {bad}")
    if not np.isfinite(dataset.y).all():
        raise ValueError("non-finite target values")
    if not (dataset.feature_date < dataset.target_date).all():
        raise ValueError("target_date must be strictly after feature_date")
    if not (np.diff(dataset.target_date) > np.timedelta64(0, "ns")).all():
        raise ValueError("target_date must be strictly increasing")


def build_sequences(dataset: Dataset, lookback: int, fold=None) -> Sequences:
    """Lookback windows for the sequence models.

    ``fold`` is an array of row positions into ``dataset`` -- typically one
    fold's train or test indices. Each returned window ends at row ``i`` and
    covers rows ``i - lookback + 1 .. i`` inclusive, so it uses only
    information available at ``feature_date[i]``. Rows with fewer than
    ``lookback`` predecessors are dropped, which is why a test fold keeps its
    full length: its windows reach back into earlier data, and reaching
    backwards is not leakage.

    The returned ``target_date`` and ``close_t`` are the parent dataset's, so
    an LSTM keys its output exactly as the tree model does.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")

    n = len(dataset)
    fold = np.arange(n) if fold is None else np.asarray(fold, dtype=int)
    if fold.size and (fold.min() < 0 or fold.max() >= n):
        raise ValueError(f"fold indices out of range for a dataset of {n} rows")

    usable = fold[fold >= lookback - 1]
    if usable.size == 0:
        raise ValueError(
            f"no row in this fold has {lookback} predecessors; "
            f"lookback={lookback} is too long for {fold.size} rows at "
            f"positions {fold.min() if fold.size else 'n/a'}.."
            f"{fold.max() if fold.size else 'n/a'}"
        )

    # (m, lookback) gather matrix: row j holds the window positions for usable[j]
    offsets = np.arange(-lookback + 1, 1)
    window_idx = usable[:, None] + offsets[None, :]

    return Sequences(
        X=dataset.X[window_idx],
        y=dataset.y[usable],
        feature_date=dataset.feature_date[usable],
        target_date=dataset.target_date[usable],
        close_t=dataset.close_t[usable],
        row_index=usable,
        feature_names=list(dataset.feature_names),
    )
