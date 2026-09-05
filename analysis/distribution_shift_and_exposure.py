"""Tasks 1 and 2: does distribution shift predict degradation, and what is
the tree's ~50% exposure actually made of?"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from backtest import prediction_diagnostics
from contracts import make_predictions
from dataset import build_dataset
from evaluate import directional_accuracy, expanding_mean_benchmark, r2_oos
from experiments import load_runs
from walk_forward import WalkForwardSplitter

pd.set_option("display.width", 240, "display.max_columns", 40)
RULE = "=" * 116


def banner(t):
    print("\n" + RULE); print(t); print(RULE)


def ols(y, x, label):
    """Simple regression with a t-stat, reported honestly."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = n - 2
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * s2)
    t = coef[1] / se[1]
    r2 = 1 - resid.var() / y.var()
    return {
        "outcome": label, "n": n, "slope": coef[1], "se": se[1], "t": t,
        "p": 2 * stats.t.sf(abs(t), dof), "r_squared": r2,
    }


def main():
    runs = load_runs(str(REPO / "results/predictions"))
    tree = runs[runs["model"] == "Tree Ensemble"]
    ks = pd.read_csv(REPO / "results/fold_diagnostics.csv")

    # Per-fold metrics for the tree, seeded per fold.
    y_train = {}
    for ticker in sorted(tree["ticker"].unique()):
        raw = pd.read_csv(REPO / f"results/raw/{ticker}.csv", parse_dates=["Date"])
        ds = build_dataset(raw)
        folds = WalkForwardSplitter(1008, 252, 252).split(len(ds))
        y_train[ticker] = {i: ds.y[np.asarray(tr)] for i, (tr, _) in enumerate(folds)}

    rows = []
    for (ticker, fold_id), block in tree.groupby(["ticker", "fold_id"], sort=True):
        y_true = block["y_true"].to_numpy(float)
        y_pred = block["y_pred"].to_numpy(float)
        bench = expanding_mean_benchmark(y_true, y_train[ticker].get(int(fold_id)))
        rows.append({
            "ticker": ticker, "fold_id": int(fold_id), "n": len(block),
            "r2_oos": r2_oos(y_true, y_pred, bench),
            "DA": directional_accuracy(y_true, y_pred)["directional_accuracy"] * 100,
            "da_minus_majority":
                directional_accuracy(y_true, y_pred)["da_minus_majority"] * 100,
        })
    per_fold = pd.DataFrame(rows).merge(
        ks[["ticker", "fold_id", "ks_stat", "ks_p", "ks_reject_5pct",
            "test_outside_train_range"]],
        on=["ticker", "fold_id"], how="inner",
    )
    print(f"matched {len(per_fold)} fold-ticker pairs")

    banner("1a. REGRESSION OF PER-FOLD PERFORMANCE ON THE FOLD'S KS STATISTIC")
    results = [
        ols(per_fold["r2_oos"], per_fold["ks_stat"], "R2_OOS ~ KS"),
        ols(per_fold["DA"], per_fold["ks_stat"], "DA (%) ~ KS"),
        ols(per_fold["da_minus_majority"], per_fold["ks_stat"], "DA-majority ~ KS"),
    ]
    table = pd.DataFrame(results)
    print(table.to_string(index=False, float_format=lambda v: f"{v: .5f}"))
    print("\n  slope is the change in the outcome per 1.0 of KS statistic; the")
    print("  observed KS range is roughly 0.03-0.20, so multiply by ~0.1 for a")
    print("  realistic move.")
    for r in results:
        move = r["slope"] * 0.10
        print(f"    {r['outcome']:<20} a 0.10 rise in KS moves it by {move:+.4f}"
              f"   (t={r['t']:+.2f}, p={r['p']:.4f}, R2={r['r_squared']:.4f})")

    banner("1b. KS-SIGNIFICANT FOLDS vs THE REST")
    grouped = per_fold.groupby("ks_reject_5pct").agg(
        folds=("n", "size"),
        mean_r2_oos=("r2_oos", "mean"),
        median_r2_oos=("r2_oos", "median"),
        mean_DA=("DA", "mean"),
        mean_da_minus_majority=("da_minus_majority", "mean"),
        mean_ks=("ks_stat", "mean"),
    )
    grouped.index = ["KS not significant", "KS significant (p<0.05)"]
    print(grouped.to_string(float_format=lambda v: f"{v: .4f}"))

    a = per_fold[per_fold["ks_reject_5pct"]]
    b = per_fold[~per_fold["ks_reject_5pct"]]
    shift_tests = [{"test": "OLS " + r["outcome"], "statistic": r["t"], "p_value": r["p"],
                    "slope": r["slope"], "r_squared": r["r_squared"], "n": r["n"]} for r in results]
    for col in ("r2_oos", "DA", "da_minus_majority"):
        t = stats.ttest_ind(a[col], b[col], equal_var=False)
        u = stats.mannwhitneyu(a[col], b[col])
        common = {"mean_shifted": float(a[col].mean()), "mean_stable": float(b[col].mean()),
                  "n_shifted": int(len(a)), "n_stable": int(len(b))}
        shift_tests.append({"test": f"Welch t, {col}, shifted vs stable", "statistic": float(t.statistic),
                            "p_value": float(t.pvalue), **common})
        shift_tests.append({"test": f"Mann-Whitney U, {col}, shifted vs stable", "statistic": float(u.statistic),
                            "p_value": float(u.pvalue), **common})
        print(f"\n  {col}: shifted {a[col].mean():+.4f} vs stable {b[col].mean():+.4f}"
              f"   diff {a[col].mean()-b[col].mean():+.4f}")
        print(f"    Welch t = {t.statistic:+.3f}, p = {t.pvalue:.4f}   |   "
              f"Mann-Whitney p = {u.pvalue:.4f}")

    banner("2. WHAT IS THE TREE'S ~50% EXPOSURE MADE OF?")
    rows = []
    for ticker, block in tree.groupby("ticker", sort=True):
        block = block.sort_values("target_date").reset_index(drop=True)
        preds = make_predictions(
            target_date=block["target_date"],
            fold_id=block["fold_id"].to_numpy(int),
            close_t=block["close_t"].to_numpy(float),
            y_true=block["y_true"].to_numpy(float),
            y_pred=block["y_pred"].to_numpy(float),
        )
        d = prediction_diagnostics(preds)
        rows.append({
            "ticker": ticker,
            "mean_pred_bps": d["mean_pred_bps"],
            "std_pred_bps": d["std_pred_bps"],
            "std_true_bps": d["std_true_bps"],
            "std_ratio": d["std_ratio"],
            "frac_positive": d["frac_positive"],
            "sign_flip_rate": d["sign_flip_rate"],
            "mean_realised_bps": block["y_true"].mean() * 1e4,
        })
    exposure = pd.DataFrame(rows).set_index("ticker")
    print(exposure.to_string(float_format=lambda v: f"{v: .4f}"))

    print(f"\n  mean predicted return  : {exposure['mean_pred_bps'].mean():+.2f} bps"
          f"   (realised {exposure['mean_realised_bps'].mean():+.2f} bps)")
    print(f"  tickers with mean pred < 0: "
          f"{int((exposure['mean_pred_bps'] < 0).sum())}/30")
    print(f"  mean fraction positive : {exposure['frac_positive'].mean():.4f}")
    print(f"  mean sign-flip rate    : {exposure['sign_flip_rate'].mean():.4f}"
          f"   (a coin flip on iid signs would give ~0.50)")
    print(f"  mean std ratio         : {exposure['std_ratio'].mean():.4f}")

    print("\n  Interpretation test:")
    print("    If exposure ~50% came from SYSTEMATIC bearishness, mean predicted")
    print("    return would be negative and the sign-flip rate low (persistent).")
    print("    If it came from NEAR-ZERO predictions flipping at random, the mean")
    print("    would be near zero and the flip rate near 0.5.")
    flip = exposure["sign_flip_rate"].mean()
    mean_pred = exposure["mean_pred_bps"].mean()
    print(f"\n    observed: mean {mean_pred:+.2f} bps, flip rate {flip:.3f}")

    # Committed outputs: the per-fold series the KS regression and the shifted-
    # versus-stable tests are computed from, the tests themselves, and the
    # exposure decomposition. Sections 6.4 and 8.4 cite the first two.
    per_fold.to_csv(REPO / "results" / "fold_r2_oos.csv", index=False)
    pd.DataFrame(shift_tests).to_csv(REPO / "results" / "shift_tests.csv", index=False)
    exposure.to_csv(REPO / "results" / "tree_exposure.csv")
    print("wrote results/fold_r2_oos.csv (%d rows), results/shift_tests.csv (%d rows), results/tree_exposure.csv"
          % (len(per_fold), len(shift_tests)))


if __name__ == "__main__":
    main()
