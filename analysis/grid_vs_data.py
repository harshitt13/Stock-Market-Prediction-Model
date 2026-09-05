"""Separate the two causes of the AAPL tree's R2_OOS gap between the headline
run (frozen CSV, fold 0 opens 2014-05-29) and the sweep run (its own fetch
cache, fold 0 opens 2014-03-19): the data, or the fold grid.

Tree only, seed 42, twelve folds, benchmark seeded per fold. Four runs:

    A  frozen CSV,  frozen grid   (must reproduce results/headline/)
    B  frozen CSV,  sweep grid
    C  sweep cache, frozen grid
    D  sweep cache, sweep grid    (must reproduce results/predictions/)

plus C', the sweep cache with its 49 early rows removed on the frozen grid,
which separates vendor differences in the prices from the extra history the
cache carries. A grid is a run's calendar test windows; training is every row
before each window (analysis/gridlib.py).

Writes results/grid_vs_data.csv (one row per run, pooled and across-fold
R2_OOS, direction, per-fold R2_OOS) and results/grid_vs_data_rawdiff.csv (the
sweep cache against the frozen CSV, column by column). The feature-level
comparison is printed. Needs results/raw/AAPL.csv, the gitignored cache.

    python analysis/grid_vs_data.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gridlib import (  # noqa: E402
    FROZEN_CSV, HEADLINE_PARQUET, RAW_DIR, SWEEP_DIR, calendar_grid, committed_predictions,
    fold_stats, folds_for_grid, load_dataset, max_abs_gap, tree_predictions,
)

SWEEP_CACHE = RAW_DIR / "AAPL.csv"
OUT_RUNS = REPO / "results" / "grid_vs_data.csv"
OUT_DIFF = REPO / "results" / "grid_vs_data_rawdiff.csv"


def main() -> None:
    frozen_raw = pd.read_csv(FROZEN_CSV, parse_dates=["Date"])
    sweep_raw = pd.read_csv(SWEEP_CACHE, parse_dates=["Date"])
    frozen = load_dataset(FROZEN_CSV)
    sweep = load_dataset(SWEEP_CACHE)
    first_frozen_day = pd.Timestamp(np.asarray(frozen.feature_date)[0])
    from dataset import build_dataset
    sweep_trim = build_dataset(sweep_raw[sweep_raw["Date"] >= first_frozen_day].reset_index(drop=True))

    frozen_grid = calendar_grid(HEADLINE_PARQUET)
    sweep_grid = calendar_grid(SWEEP_DIR / "AAPL__full__seed42.parquet", ticker="AAPL")
    print(f"frozen dataset {len(frozen)} rows from {first_frozen_day.date()}; sweep cache dataset {len(sweep)} rows "
          f"from {pd.Timestamp(np.asarray(sweep.feature_date)[0]).date()}")
    print(f"frozen grid fold 0 test {frozen_grid[0][0].date()}..{frozen_grid[0][1].date()}; "
          f"sweep grid fold 0 test {sweep_grid[0][0].date()}..{sweep_grid[0][1].date()}")

    runs = [
        ("A", "frozen CSV", "frozen", frozen, frozen_grid, committed_predictions(HEADLINE_PARQUET)),
        ("B", "frozen CSV", "sweep", frozen, sweep_grid, None),
        ("C", "sweep cache", "frozen", sweep, frozen_grid, None),
        ("D", "sweep cache", "sweep", sweep, sweep_grid,
         committed_predictions(SWEEP_DIR / "AAPL__full__seed42.parquet", ticker="AAPL")),
        ("C'", "sweep cache minus its 49 early rows", "frozen", sweep_trim, frozen_grid, None),
    ]
    rows = []
    for label, data, grid_name, ds, grid, committed in runs:
        folds = folds_for_grid(ds, grid)
        preds = tree_predictions(ds, folds)
        stats = fold_stats(preds, label, ds, folds)
        r2 = np.array([stats[f"r2_fold_{k:02d}"] for k in range(12)])
        row = {"run": label, "data": data, "grid": grid_name, "train_rows_fold0": int(len(folds[0][0])),
               "fold0_test_opens": str(grid[0][0].date()),
               "reproduces_committed_max_abs_diff": (
                   max_abs_gap(preds.set_index("target_date")["y_pred"], committed) if committed is not None else np.nan),
               **stats,
               "r2_fold_mean_excl_fold6": float(np.delete(r2, 6).mean())}
        rows.append(row)
        rep = f"  reproduces committed run to {row['reproduces_committed_max_abs_diff']:.1e}" if committed is not None else ""
        print(f"{label:<3} {data:<38} {grid_name:<7} grid  train0={row['train_rows_fold0']:4d}  pooled {stats['r2_pooled']:+.4f}  "
              f"fold mean {stats['r2_fold_mean']:+.4f}  median {stats['r2_fold_median']:+.4f}  "
              f"DA-maj {stats['da_minus_majority']*100:+.2f}pp{rep}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT_RUNS, index=False)
    print(f"wrote {OUT_RUNS.relative_to(REPO)}")

    # The raw data, column by column, on the shared dates.
    m = sweep_raw.merge(frozen_raw, on="Date", suffixes=("_sweep", "_frozen"))
    diff_rows = []
    for col in ["Open", "High", "Low", "Close", "Volume", "SP500", "VIX", "TNX_Yield"]:
        a = m[f"{col}_sweep"].to_numpy(float)
        b = m[f"{col}_frozen"].to_numpy(float)
        d = a - b
        rel = np.abs(d) / np.where(np.abs(b) > 0, np.abs(b), np.nan)
        differs = np.abs(d) > 1e-6
        diff_rows.append({
            "column": col, "shared_dates": int(len(m)),
            "max_abs_diff": float(np.nanmax(np.abs(d))), "max_rel_diff": float(np.nanmax(rel)),
            "rows_abs_diff_gt_1e-6": int(differs.sum()), "rows_rel_diff_gt_1e-6": int(np.nansum(rel > 1e-6)),
            "mean_rel_diff_on_differing_rows": float(np.nanmean(rel[differs])) if differs.any() else 0.0,
            "first_differing_date": str(m.loc[differs, "Date"].min().date()) if differs.any() else "",
            "last_differing_date": str(m.loc[differs, "Date"].max().date()) if differs.any() else "",
        })
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUT_DIFF, index=False)
    print(diff.to_string(index=False, float_format=lambda v: f"{v:.3e}"))
    print(f"wrote {OUT_DIFF.relative_to(REPO)}")

    # Features on the shared dataset rows: where the two differ and by how much.
    fdates = pd.to_datetime(pd.Series(np.asarray(frozen.feature_date)))
    sdates = pd.to_datetime(pd.Series(np.asarray(sweep.feature_date)))
    fi, si = fdates.isin(sdates).to_numpy(), sdates.isin(fdates).to_numpy()
    xf, xs = frozen.X[fi], sweep.X[si]
    feats = []
    for j, name in enumerate(frozen.feature_names):
        d = xs[:, j] - xf[:, j]
        feats.append({"feature": name, "max_abs_diff": float(np.max(np.abs(d))),
                      "max_abs_diff_over_sd": float(np.max(np.abs(d)) / (np.std(xf[:, j]) or 1.0)),
                      "rows_gt_1e-9": int((np.abs(d) > 1e-9).sum())})
    feats = pd.DataFrame(feats).sort_values("max_abs_diff_over_sd", ascending=False)
    print("\nfeature differences on shared rows (sweep minus frozen), largest first:")
    print(feats.head(8).to_string(index=False, float_format=lambda v: f"{v:.3e}"))
    yd = sweep.y[si] - frozen.y[fi]
    print(f"target y: max |diff| {np.max(np.abs(yd)):.3e} over {len(yd)} shared rows")


if __name__ == "__main__":
    main()
