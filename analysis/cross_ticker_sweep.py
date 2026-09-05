"""Cross-ticker aggregation of the 30-ticker sweep. Reads from disk only."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from backtest import backtest, holm_bonferroni
from contracts import make_predictions
from evaluate import evaluate_predictions
from experiments import FULL_MODEL_TICKERS, load_runs

pd.set_option("display.width", 250, "display.max_columns", 60)
RULE = "=" * 120


def banner(t):
    print("\n" + RULE); print(t); print(RULE)


def main():
    runs = load_runs(str(REPO / "results/predictions"))
    print(f"loaded {len(runs):,} prediction rows from disk")
    print(f"  tickers {runs['ticker'].nunique()}, models {runs['model'].nunique()}, "
          f"seeds {sorted(runs['seed'].unique())}")

    # R2_OOS needs each fold's training returns to seed the expanding-mean
    # benchmark. Without them the benchmark starts cold and every model's
    # R2_OOS is biased upward -- the zero-return baseline scores +0.035 instead
    # of ~0. Rebuild the datasets from the cached raw CSVs to recover them.
    from dataset import build_dataset
    from walk_forward import WalkForwardSplitter

    y_train_by_ticker = {}
    for ticker in sorted(runs["ticker"].unique()):
        raw = pd.read_csv(REPO / f"results/raw/{ticker}.csv", parse_dates=["Date"])
        ds = build_dataset(raw)
        folds = WalkForwardSplitter(1008, 252, 252).split(len(ds))
        y_train_by_ticker[ticker] = {
            i: ds.y[np.asarray(tr)] for i, (tr, _) in enumerate(folds)
        }
    print(f"  rebuilt training returns for {len(y_train_by_ticker)} tickers "
          f"(R2_OOS benchmark seeding)")

    rows = []
    for (ticker, model), block in runs.groupby(["ticker", "model"], sort=True):
        block = block.sort_values("target_date").reset_index(drop=True)
        preds = make_predictions(
            target_date=block["target_date"],
            fold_id=block["fold_id"].to_numpy(int),
            close_t=block["close_t"].to_numpy(float),
            y_true=block["y_true"].to_numpy(float),
            y_pred=block["y_pred"].to_numpy(float),
        )
        ev = evaluate_predictions(
            preds, model, validate=False,
            y_train_by_fold=y_train_by_ticker[ticker],
        )["pooled"]
        bt = backtest(preds, cost_bps=7.5, validate=False)
        adj = bt["market_adjusted"]
        rows.append({
            "ticker": ticker, "model": model, "n": len(preds),
            "DA": ev["directional_accuracy"] * 100,
            "majority": ev["majority_class_rate"] * 100,
            "DA_minus_majority": ev["da_minus_majority"] * 100,
            "PT_p": ev.get("pt_p_value", np.nan),
            "R2_OOS": ev["r2_oos"],
            "RMSE_bps": ev["rmse_bps"],
            "alpha_ann": adj["alpha_annualised"],
            "t_alpha": adj["t_alpha"],
            "p_alpha": adj["p_alpha"],
            "beta": adj["beta"],
            "exposure": bt["average_exposure"],
        })
    per = pd.DataFrame(rows)
    per.to_csv(REPO / "results/per_ticker_model.csv", index=False)

    tree = per[per["model"] == "Tree Ensemble"].set_index("ticker").sort_index()

    banner("A. CROSS-TICKER DISTRIBUTION, Tree Ensemble on all 30 tickers")
    for col, label in [
        ("DA_minus_majority", "DA - majority (pp)"),
        ("R2_OOS", "R2_OOS"),
        ("PT_p", "Pesaran-Timmermann p"),
        ("t_alpha", "t(alpha)"),
    ]:
        v = tree[col].dropna()
        print(f"  {label:<24} n={len(v):>2}  mean {v.mean():+8.4f}  median "
              f"{v.median():+8.4f}  sd {v.std():7.4f}  min {v.min():+8.4f}  "
              f"max {v.max():+8.4f}")

    banner("B. DOES ANY TICKER BREAK THE NULL?  (Tree Ensemble, 30 tests)")
    for label, col, better in [
        ("PT p < 0.05 (directional skill)", "PT_p", "lower"),
        ("alpha p < 0.05", "p_alpha", "lower"),
    ]:
        p = tree[col].dropna()
        hits = p[p < 0.05]
        holm = pd.Series(holm_bonferroni(p.to_numpy()), index=p.index)
        holm_hits = holm[holm < 0.05]
        expected = 0.05 * len(p)
        print(f"\n  {label}")
        print(f"    raw p<0.05      : {len(hits):>2}/{len(p)}   "
              f"expected by chance at 5%: {expected:.1f}")
        if len(hits):
            print(f"      {', '.join(f'{t}={v:.4f}' for t, v in hits.items())}")
        print(f"    Holm-corrected  : {len(holm_hits):>2}/{len(p)}")
        if len(holm_hits):
            print(f"      {', '.join(f'{t}={v:.4f}' for t, v in holm_hits.items())}")
        # Binomial: how surprising is this many hits out of 30?
        k, n = len(hits), len(p)
        print(f"    P(>= {k} hits | pure null) = "
              f"{stats.binom.sf(k - 1, n, 0.05) if k else 1.0:.3f}")

    banner("C. POSITIVE R2_OOS AND DA ABOVE MAJORITY, PER TICKER")
    print(f"  R2_OOS > 0            : {int((tree['R2_OOS'] > 0).sum())}/30")
    print(f"  DA above majority     : {int((tree['DA_minus_majority'] > 0).sum())}/30")
    print(f"  alpha t > 1.96        : {int((tree['t_alpha'] > 1.96).sum())}/30")
    print(f"  alpha t < -1.96       : {int((tree['t_alpha'] < -1.96).sum())}/30")
    print("\n  Per-ticker detail, sorted by DA - majority:")
    print(tree.sort_values("DA_minus_majority", ascending=False)[
        ["DA", "majority", "DA_minus_majority", "PT_p", "R2_OOS", "t_alpha",
         "p_alpha", "beta", "exposure"]
    ].to_string(float_format=lambda v: f"{v: .4f}"))

    banner("D. ALL MODELS ON THE 5 FULL-MODEL TICKERS")
    full = per[per["ticker"].isin(FULL_MODEL_TICKERS)]
    pivot = full.pivot_table(index="model", values=["DA_minus_majority", "R2_OOS",
                                                    "t_alpha", "PT_p"], aggfunc="mean")
    print(pivot.to_string(float_format=lambda v: f"{v: .4f}"))
    print(f"\n  tickers: {sorted(full['ticker'].unique())}")

    banner("E. TRAIN vs TEST RETURN DISTRIBUTION SHIFT (KS per fold)")
    ks = pd.read_csv(REPO / "results/fold_diagnostics.csv")
    pooled_rate = ks["ks_reject_5pct"].mean()
    print(f"  folds total                    : {len(ks)}")
    print(f"  KS rejects at 5% (pooled)      : {ks['ks_reject_5pct'].sum()}/{len(ks)} "
          f"= {pooled_rate:.1%}")
    print(f"  expected under no shift        : 5.0%")
    print(f"  mean KS statistic              : {ks['ks_stat'].mean():.4f}")
    print(f"  mean test days outside train range: "
          f"{ks['test_outside_train_range'].mean():.4%}")
    by_ticker = ks.groupby("ticker")["ks_reject_5pct"].agg(["sum", "count", "mean"])
    by_ticker.columns = ["rejects", "folds", "fraction"]
    print(f"\n  per-ticker rejection fraction: mean {by_ticker['fraction'].mean():.1%}, "
          f"min {by_ticker['fraction'].min():.1%}, max {by_ticker['fraction'].max():.1%}")
    print(f"  tickers with >50% of folds shifted: "
          f"{int((by_ticker['fraction'] > 0.5).sum())}/30")
    print("\n  worst 8 tickers by shifted-fold fraction:")
    print(by_ticker.sort_values("fraction", ascending=False).head(8).to_string())

    by_fold = ks.groupby("fold_id")["ks_reject_5pct"].mean()
    print("\n  rejection rate by fold index (all tickers):")
    print("   " + "  ".join(f"f{i}:{v:.0%}" for i, v in by_fold.items()))

    banner("F. BASELINES ACROSS ALL 30 TICKERS")
    cheap = per[per["model"].isin([
        "Zero return", "Historical mean", "AR(1) returns",
        "ARIMA(5, 0, 0) returns", "Random sign", "Tree Ensemble"])]
    summary = cheap.groupby("model").agg(
        tickers=("ticker", "nunique"),
        DA_minus_majority=("DA_minus_majority", "mean"),
        R2_OOS=("R2_OOS", "mean"),
        R2_positive=("R2_OOS", lambda s: float((s > 0).mean())),
        t_alpha=("t_alpha", "mean"),
        alpha_sig=("p_alpha", lambda s: int((s < 0.05).sum())),
    )
    print(summary.to_string(float_format=lambda v: f"{v: .4f}"))


if __name__ == "__main__":
    main()
