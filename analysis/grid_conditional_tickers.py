"""Is the sweep's cross-ticker R2_OOS magnitude conditional on the fold grid?

Tree only, seed 42, the five tickers that carried every model in the sweep
(AAPL, JNJ, JPM, WMT, XOM), each run two ways on its own sweep cache:

    sweep grid   the cache's own 1008/252/252 grid (must reproduce
                 results/predictions/, fold 0 opens 2014-03-19)
    frozen grid  the headline run's calendar test windows applied to the
                 same cache (fold 0 opens 2014-05-29), training on every
                 row before each window

Writes results/grid_conditional_tickers.csv with pooled and across-fold
R2_OOS per ticker and grid, and prints the mean over the five tickers under
each grid and whether any R2_OOS is positive under either. Needs the
gitignored results/raw/ cache.

    python analysis/grid_conditional_tickers.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gridlib import (  # noqa: E402
    FULL_MODEL_TICKERS, HEADLINE_PARQUET, RAW_DIR, SWEEP_DIR, calendar_grid, committed_predictions,
    fold_stats, folds_for_grid, load_dataset, max_abs_gap, tree_predictions,
)

OUT = REPO / "results" / "grid_conditional_tickers.csv"


def main() -> None:
    frozen_grid = calendar_grid(HEADLINE_PARQUET)
    rows = []
    for ticker in FULL_MODEL_TICKERS:
        ds = load_dataset(RAW_DIR / f"{ticker}.csv")
        parquet = SWEEP_DIR / f"{ticker}__full__seed42.parquet"
        grids = {"sweep": calendar_grid(parquet, ticker=ticker), "frozen": frozen_grid}
        for grid_name, grid in grids.items():
            folds = folds_for_grid(ds, grid)
            preds = tree_predictions(ds, folds)
            stats = fold_stats(preds, f"{ticker} {grid_name}", ds, folds)
            gap = (max_abs_gap(preds.set_index("target_date")["y_pred"], committed_predictions(parquet, ticker=ticker))
                   if grid_name == "sweep" else np.nan)
            rows.append({"ticker": ticker, "grid": grid_name, "train_rows_fold0": int(len(folds[0][0])),
                         "fold0_test_opens": str(grid[0][0].date()), "reproduces_committed_max_abs_diff": gap, **stats})
            rep = f"  reproduces committed run to {gap:.1e}" if grid_name == "sweep" else ""
            print(f"{ticker:<5} {grid_name:<6} grid  train0={int(len(folds[0][0])):4d}  pooled {stats['r2_pooled']:+.4f}  "
                  f"fold mean {stats['r2_fold_mean']:+.4f}  median {stats['r2_fold_median']:+.4f}  "
                  f"worst fold {stats['worst_fold']}  DA-maj {stats['da_minus_majority']*100:+.2f}pp{rep}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}")
    for grid_name in ("sweep", "frozen"):
        g = table[table["grid"] == grid_name]
        print(f"  {grid_name:<6} grid, 5 tickers: mean pooled {g['r2_pooled'].mean():+.4f}  mean fold-mean {g['r2_fold_mean'].mean():+.4f}  "
              f"mean median {g['r2_fold_median'].mean():+.4f};  pooled > 0 on {int((g['r2_pooled'] > 0).sum())}/5, "
              f"median > 0 on {int((g['r2_fold_median'] > 0).sum())}/5")
    wide = table.pivot(index="ticker", columns="grid", values="r2_pooled")
    print(f"  pooled, frozen minus sweep grid per ticker: {(wide['frozen'] - wide['sweep']).round(4).to_dict()}")


if __name__ == "__main__":
    main()
