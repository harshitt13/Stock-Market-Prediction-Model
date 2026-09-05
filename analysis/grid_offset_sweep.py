"""How much does the AAPL tree's R2_OOS depend on where the fold boundaries
fall? Tree only, frozen CSV, seed 42, the headline 1008/252/252 grid with
every boundary moved earlier by 0, 12, 25, 37, 49 and 63 trading days (49 is
the sweep grid's offset), and later by 12, 25, 37 and 49 where twelve full
folds still fit (later by 63 does not: the last fold would run past the
data). Twelve folds every time; the benchmark is seeded per fold.

Writes results/grid_offset_sweep.csv with pooled and across-fold R2_OOS and
the per-fold series at each offset. No network, no cache: the frozen CSV is
committed.

    python analysis/grid_offset_sweep.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gridlib import FROZEN_CSV, fold_stats, load_dataset, offset_folds, tree_predictions  # noqa: E402

EARLIER = (0, 12, 25, 37, 49, 63)
LATER = (12, 25, 37, 49)
OUT = REPO / "results" / "grid_offset_sweep.csv"


def main() -> None:
    ds = load_dataset(FROZEN_CSV)
    td = pd.to_datetime(pd.Series(np.asarray(ds.target_date)))
    rows = []
    for offset in list(EARLIER) + [-k for k in LATER]:
        folds = offset_folds(len(ds), offset)
        preds = tree_predictions(ds, folds)
        stats = fold_stats(preds, f"offset {offset}", ds, folds)
        row = {"offset_days_earlier": offset, "min_train_rows": int(len(folds[0][0])),
               "fold0_test_opens": str(td.iloc[folds[0][1][0]].date()),
               "last_test_day": str(td.iloc[folds[-1][1][-1]].date()), **stats}
        rows.append(row)
        print(f"offset {offset:+4d} days earlier  min_train {row['min_train_rows']:4d}  fold 0 opens {row['fold0_test_opens']}  "
              f"pooled {stats['r2_pooled']:+.4f}  fold mean {stats['r2_fold_mean']:+.4f} (sd {stats['r2_fold_sd']:.3f})  "
              f"median {stats['r2_fold_median']:+.4f}  excl. worst {stats['r2_fold_mean_excl_worst']:+.4f}  "
              f"worst fold {stats['worst_fold']}  DA-maj {stats['da_minus_majority']*100:+.2f}pp")
    table = pd.DataFrame(rows).sort_values("offset_days_earlier", ascending=False)
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}")
    req = table[table["offset_days_earlier"].isin(EARLIER)]
    for col in ("r2_pooled", "r2_fold_mean", "r2_fold_median"):
        print(f"  {col:<16} over the six requested offsets: min {req[col].min():+.4f}  max {req[col].max():+.4f}  "
              f"range {req[col].max() - req[col].min():.4f}")


if __name__ == "__main__":
    main()
