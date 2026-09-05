"""Shared helpers for the fold-grid experiments.

A "grid" here is the list of calendar test windows of a committed run. Given
a dataset, fold k tests on the rows whose target_date falls inside window k
and trains on every row before the first of them, so the same calendar grid
can be applied to a dataset that starts on a different day. An "offset" grid
is the headline 1008/252/252 configuration with its first boundary moved
earlier by a number of rows.

Every experiment evaluates with the benchmark seeded from each fold's own
training returns, and reports the pooled R2_OOS beside the across-fold mean,
standard deviation and median, because the pooled ratio is dominated by the
fold with the largest squared errors.
"""

import contextlib
import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HEADLINE_PARQUET = REPO / "results" / "headline" / "AAPL__frozen__seed42.parquet"
SWEEP_DIR = REPO / "results" / "predictions"
FROZEN_CSV = REPO / "docs" / "frozen_aapl_raw.csv"
RAW_DIR = REPO / "results" / "raw"
FOLD_CFG = (1008, 252, 252)
FULL_MODEL_TICKERS = ("AAPL", "JNJ", "JPM", "WMT", "XOM")

Grid = List[Tuple[pd.Timestamp, pd.Timestamp, int]]
Folds = List[Tuple[np.ndarray, np.ndarray]]


def calendar_grid(parquet: Path, ticker: Optional[str] = None, model: str = "Tree Ensemble") -> Grid:
    """The test windows of a committed run, one (first, last, n) per fold."""
    d = pd.read_parquet(parquet)
    d = d[d["model"] == model]
    if ticker is not None:
        d = d[d["ticker"] == ticker]
    g = d.groupby("fold_id")["target_date"].agg(["min", "max", "size"]).sort_index()
    return [(pd.Timestamp(a), pd.Timestamp(b), int(n)) for a, b, n in g.itertuples(index=False)]


def committed_predictions(parquet: Path, ticker: Optional[str] = None, model: str = "Tree Ensemble") -> pd.Series:
    d = pd.read_parquet(parquet)
    d = d[d["model"] == model]
    if ticker is not None:
        d = d[d["ticker"] == ticker]
    return d.set_index("target_date")["y_pred"].sort_index()


def folds_for_grid(dataset, grid: Grid) -> Folds:
    """Apply a calendar grid to a dataset. Asserts every window is fully present."""
    td = pd.to_datetime(pd.Series(np.asarray(dataset.target_date)))
    folds: Folds = []
    for first, last, n in grid:
        test = np.where((td >= first) & (td <= last))[0]
        if len(test) != n:
            raise ValueError(f"window {first.date()}..{last.date()} has {len(test)} rows in this dataset, expected {n}")
        if test[0] == 0:
            raise ValueError("a test window starts at the first row; nothing to train on")
        folds.append((np.arange(0, test[0]), test))
    return folds


def offset_folds(n_rows: int, offset: int, cfg: Tuple[int, int, int] = FOLD_CFG) -> Folds:
    """The headline grid with every boundary moved earlier by ``offset`` rows
    (a negative offset moves it later). Always exactly 12 folds."""
    min_train, test, step = cfg
    start = min_train - offset
    if start < 1:
        raise ValueError(f"offset {offset} leaves no training rows")
    folds: Folds = []
    for k in range(12):
        a = start + k * step
        if a + test > n_rows:
            raise ValueError(f"offset {offset}: fold {k} runs past the data ({a + test} > {n_rows})")
        folds.append((np.arange(0, a), np.arange(a, a + test)))
    return folds


def tree_predictions(dataset, folds: Folds, seed: int = 42) -> pd.DataFrame:
    from tree_model import train_tree_model

    with contextlib.redirect_stdout(io.StringIO()):
        return train_tree_model(dataset, folds, random_state=seed, save_model=False)["predictions"]


def fold_stats(predictions: pd.DataFrame, name: str, dataset, folds: Folds) -> Dict[str, object]:
    """Pooled and across-fold R2_OOS, direction, and the per-fold series."""
    from evaluate import evaluate_predictions

    y_train = {k: dataset.y[tr] for k, (tr, _) in enumerate(folds)}
    with contextlib.redirect_stdout(io.StringIO()):
        ev = evaluate_predictions(predictions, name, y_train_by_fold=y_train, validate=False)
    pooled = ev["pooled"]
    pf = ev["per_fold"].sort_values("fold_id")
    r2 = pf["r2_oos"].to_numpy(float)
    worst = int(np.argmin(r2))
    out: Dict[str, object] = {
        "n_folds": int(len(r2)),
        "r2_pooled": float(pooled["r2_oos"]),
        "r2_fold_mean": float(r2.mean()),
        "r2_fold_sd": float(r2.std(ddof=1)) if len(r2) > 1 else float("nan"),
        "r2_fold_median": float(np.median(r2)),
        "r2_fold_mean_excl_worst": float(np.delete(r2, worst).mean()) if len(r2) > 1 else float("nan"),
        "worst_fold": int(pf["fold_id"].iloc[worst]),
        "da": float(pooled["directional_accuracy"]),
        "majority": float(pooled["majority_class_rate"]),
        "da_minus_majority": float(pooled["da_minus_majority"]),
    }
    for fid, v in zip(pf["fold_id"].to_numpy(int), r2):
        out[f"r2_fold_{fid:02d}"] = float(v)
    return out


def max_abs_gap(a: pd.Series, b: pd.Series) -> float:
    """Largest absolute difference between two prediction series on shared dates."""
    b = b.reindex(a.index)
    return float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))))


def load_dataset(csv: Path):
    from dataset import build_dataset

    return build_dataset(pd.read_csv(csv, parse_dates=["Date"]))


def y_train_by_fold(dataset, folds: Folds) -> Dict[int, np.ndarray]:
    return {k: dataset.y[tr] for k, (tr, _) in enumerate(folds)}
