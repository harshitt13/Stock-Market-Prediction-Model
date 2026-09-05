"""Across-fold aggregates of R2_OOS beside the pooled value, for every
committed run that the paper reports.

Pooled R2_OOS is 1 - sum(SSE_model) / sum(SSE_benchmark) over all test days,
so it is a variance-weighted ratio that the fold with the largest squared
errors dominates. The across-fold mean, standard deviation and median weight
every fold equally. This script computes both from the committed predictions
and writes them so the paper can report the fold median as its headline
magnitude with the pooled value beside it.

    results/fold_aggregates_headline.csv   every model of the headline run on
                                           the common window (folds 2..11),
                                           from results/headline/ and the
                                           frozen CSV; no cache needed
    results/fold_aggregates_sweep.csv      the tree on all 30 sweep tickers,
                                           every fold, from results/predictions/
                                           and the gitignored results/raw/ cache

No training. Seconds.

    python analysis/fold_aggregates.py
"""

import contextlib
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from contracts import common_evaluation_window, make_predictions, restrict_all  # noqa: E402
from evaluate import evaluate_predictions  # noqa: E402
from experiments import load_runs  # noqa: E402
from gridlib import FOLD_CFG, FROZEN_CSV, HEADLINE_PARQUET, RAW_DIR, SWEEP_DIR, load_dataset  # noqa: E402
from walk_forward import WalkForwardSplitter  # noqa: E402

OUT_HEADLINE = REPO / "results" / "fold_aggregates_headline.csv"
OUT_SWEEP = REPO / "results" / "fold_aggregates_sweep.csv"


def frame(block: pd.DataFrame) -> pd.DataFrame:
    block = block.sort_values("target_date").reset_index(drop=True)
    return make_predictions(
        target_date=block["target_date"], fold_id=block["fold_id"].to_numpy(int),
        close_t=block["close_t"].to_numpy(float), y_true=block["y_true"].to_numpy(float),
        y_pred=block["y_pred"].to_numpy(float),
    )


def aggregates(predictions: pd.DataFrame, name: str, y_train: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        ev = evaluate_predictions(predictions, name, y_train_by_fold=y_train, validate=False)
    pf = ev["per_fold"].sort_values("fold_id")
    r2 = pf["r2_oos"].to_numpy(float)
    worst = int(np.argmin(r2))
    return {
        "n_folds": int(len(r2)), "n_days": int(len(predictions)),
        "r2_pooled": float(ev["pooled"]["r2_oos"]),
        "r2_fold_mean": float(r2.mean()), "r2_fold_sd": float(r2.std(ddof=1)),
        "r2_fold_median": float(np.median(r2)),
        "r2_fold_mean_excl_worst": float(np.delete(r2, worst).mean()),
        "worst_fold": int(pf["fold_id"].iloc[worst]), "worst_fold_r2": float(r2[worst]),
        "da_minus_majority_pooled": float(ev["pooled"]["da_minus_majority"]),
        "da_minus_majority_fold_median": float(np.median(pf["da_minus_majority"].to_numpy(float))),
    }


def headline() -> pd.DataFrame:
    runs = load_runs(str(HEADLINE_PARQUET.parent))
    runs = runs[(runs["ticker"] == "AAPL") & (runs["regime"] == "frozen")]
    frames = {m: frame(b) for m, b in runs.groupby("model")}
    ds = load_dataset(FROZEN_CSV)
    folds = WalkForwardSplitter(*FOLD_CFG).split(len(ds))
    y_train = {i: ds.y[np.asarray(tr)] for i, (tr, _) in enumerate(folds)}
    window = common_evaluation_window(frames)
    primary = restrict_all(frames, window)
    rows = [{"model": m, **aggregates(p, m, y_train)} for m, p in primary.items()]
    table = pd.DataFrame(rows)
    table.to_csv(OUT_HEADLINE, index=False)
    print(f"headline run, common window {window.describe()}:")
    print(table[["model", "n_folds", "r2_pooled", "r2_fold_mean", "r2_fold_sd", "r2_fold_median", "r2_fold_mean_excl_worst", "worst_fold"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    print(f"wrote {OUT_HEADLINE.relative_to(REPO)}")
    return table


def sweep() -> pd.DataFrame:
    runs = load_runs(str(SWEEP_DIR))
    tree = runs[runs["model"] == "Tree Ensemble"]
    rows = []
    for ticker, block in tree.groupby("ticker"):
        ds = load_dataset(RAW_DIR / f"{ticker}.csv")
        folds = WalkForwardSplitter(*FOLD_CFG).split(len(ds))
        y_train = {i: ds.y[np.asarray(tr)] for i, (tr, _) in enumerate(folds)}
        rows.append({"ticker": ticker, **aggregates(frame(block), f"{ticker} tree", y_train)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_SWEEP, index=False)
    print("\ntree ensemble across the sweep:")
    for col in ("r2_pooled", "r2_fold_mean", "r2_fold_median"):
        v = table[col]
        print(f"  {col:<16} mean {v.mean():+.4f}  median {v.median():+.4f}  sd {v.std(ddof=1):.4f}  "
              f"min {v.min():+.4f}  max {v.max():+.4f}  tickers > 0: {int((v > 0).sum())}/{len(v)}")
    print(f"  worst fold is fold 6 on {int((table['worst_fold'] == 6).sum())}/{len(table)} tickers; "
          f"worst-fold R2 mean {table['worst_fold_r2'].mean():+.3f}")
    a = table.set_index("ticker")
    print(f"  AAPL: pooled {a.loc['AAPL', 'r2_pooled']:+.4f}  fold mean {a.loc['AAPL', 'r2_fold_mean']:+.4f}  "
          f"median {a.loc['AAPL', 'r2_fold_median']:+.4f}; rank by median {int(a['r2_fold_median'].rank(ascending=False)['AAPL'])}/30")
    print(f"wrote {OUT_SWEEP.relative_to(REPO)}")
    return table


if __name__ == "__main__":
    headline()
    sweep()
