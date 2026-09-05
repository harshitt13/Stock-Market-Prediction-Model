"""How many Pesaran-Timmermann hits at the 5% level should thirty correlated
tickers produce, and ninety correlated ticker-model tests?

The binomial reference (1.5 hits expected in 30, 4.5 in 90) assumes the tests
are independent. They are not: the thirty tickers are US equities over the
same sixteen years on the same fold grid, and the three models are
near-constant predictors fitted to identical data. This script does two
things.

1. Estimates the correlations from the committed predictions rather than
   assuming them: the mean pairwise cross-ticker correlation of daily
   realised returns, of realised signs, and of each model's daily direction
   hits (the quantity the PT statistic is built from); and the cross-model
   correlation within a ticker, both of daily hits and of the per-ticker PT
   statistics.
2. Simulates equicorrelated standard-normal test statistics, one-sided at
   the 5% level as the pipeline's PT p-value is, and reports the expected
   hit count, its standard deviation, and the probability of at least the
   observed count, for each model's thirty tests at rho_ticker in
   {0, 0.3, 0.5, estimate} and for the ninety-test pool under a Kronecker
   structure (rho_ticker across tickers, rho_model across models within a
   ticker) at rho_model in {0, 0.8, estimate}.

Writes results/cross_ticker_correlation.csv and results/hit_count_null.csv.
Needs results/predictions/ and results/comparators/ (committed) and
results/per_ticker_model.csv (committed); no cache.

    python analysis/correlated_hit_null.py
"""

import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from evaluate import DIRECTION_THRESHOLD, pesaran_timmermann  # noqa: E402
from experiments import load_runs  # noqa: E402

MODELS = ["Tree Ensemble", "Ridge (returns)", "Logistic (direction)"]
Z_HIT = stats.norm.isf(0.05)  # one-sided 5%: 1.645
N_SIMS = 200_000
N_BOOT = 400
BLOCK_DAYS = 21
OUT_CORR = REPO / "results" / "cross_ticker_correlation.csv"
OUT_NULL = REPO / "results" / "hit_count_null.csv"


def mean_offdiag(corr: pd.DataFrame) -> float:
    """Mean of the off-diagonal entries, ignoring pairs that could not be
    estimated (a replicate on which a model predicted one direction only has
    no PT statistic)."""
    a = corr.to_numpy(float).copy()
    np.fill_diagonal(a, np.nan)
    return float(np.nanmean(a))


def daily_hits(block: pd.DataFrame) -> pd.Series:
    keep = block["y_true"].abs() > DIRECTION_THRESHOLD
    hit = ((block["y_pred"] > 0) == (block["y_true"] > 0)).astype(float)
    return hit.where(keep).set_axis(block["target_date"])


def bootstrap_pt_correlation(runs: pd.DataFrame, rng) -> dict:
    """The correlation of the PT statistics themselves, by a moving-block
    bootstrap over calendar days shared across models and tickers.

    Each replicate resamples blocks of BLOCK_DAYS consecutive trading days with
    replacement, recomputes every (model, ticker) PT statistic on the resampled
    days, and the correlation matrices of those replicated statistics give the
    cross-ticker correlation within each model and the cross-model correlation
    within each ticker. This is the correlation the hit-count null needs; the
    daily hit-indicator correlation is only a proxy for it."""
    wide = {}
    for (model, ticker), b in runs.groupby(["model", "ticker"]):
        wide[(model, ticker)] = b.sort_values("target_date").set_index("target_date")[["y_true", "y_pred"]]
    dates = np.array(sorted(set().union(*[set(v.index) for v in wide.values()])))
    n_blocks = int(np.ceil(len(dates) / BLOCK_DAYS))
    stats_boot = {k: np.empty(N_BOOT) for k in wide}
    for r in range(N_BOOT):
        starts = rng.integers(0, len(dates) - BLOCK_DAYS + 1, size=n_blocks)
        picked = np.concatenate([dates[st:st + BLOCK_DAYS] for st in starts])[: len(dates)]
        for k, v in wide.items():
            sel = v.reindex(picked).dropna()
            stats_boot[k][r] = pesaran_timmermann(sel["y_true"].to_numpy(float), sel["y_pred"].to_numpy(float))["pt_statistic"]
    out = {}
    for model in MODELS:
        m = pd.DataFrame({t: stats_boot[(model, t)] for (mm, t) in wide if mm == model})
        out[f"PT statistic, cross-ticker, {model} (block bootstrap)"] = mean_offdiag(m.corr())
    for a, b in combinations(MODELS, 2):
        cs = [np.corrcoef(stats_boot[(a, t)], stats_boot[(b, t)])[0, 1] for (mm, t) in wide if mm == a]
        out[f"PT statistic, cross-model within ticker, {a} vs {b} (block bootstrap)"] = float(np.nanmean(cs))
    return out


def estimate_correlations(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tree = runs[runs["model"] == "Tree Ensemble"]
    ret = tree.pivot_table(index="target_date", columns="ticker", values="y_true").dropna()
    rows.append({"quantity": "daily realised return, cross-ticker", "estimate": mean_offdiag(ret.corr()), "n_days": len(ret)})
    rows.append({"quantity": "daily realised sign, cross-ticker", "estimate": mean_offdiag((ret > 0).astype(float).corr()), "n_days": len(ret)})
    hits = {}
    for model in MODELS:
        block = runs[runs["model"] == model]
        h = pd.concat({t: daily_hits(b.sort_values("target_date")) for t, b in block.groupby("ticker")}, axis=1)
        hits[model] = h
        # Pairwise-complete: a day counts for a pair whenever both tickers are
        # outside the dead zone, rather than only when all thirty are.
        rows.append({"quantity": f"daily direction hit, cross-ticker, {model}", "estimate": mean_offdiag(h.corr(min_periods=250)),
                     "n_days": int(h.notna().sum().median())})
    # cross-model within ticker: daily hits
    for a, b in combinations(MODELS, 2):
        cs = []
        for t in hits[a].columns:
            pair = pd.concat([hits[a][t], hits[b][t]], axis=1).dropna()
            if len(pair) > 10:
                cs.append(pair.corr().iloc[0, 1])
        rows.append({"quantity": f"daily direction hit, cross-model within ticker, {a} vs {b}", "estimate": float(np.mean(cs)), "n_days": int(np.mean([len(hits[a][t].dropna()) for t in hits[a].columns]))})
    # cross-model across the 30 per-ticker PT statistics
    per = pd.read_csv(REPO / "results" / "per_ticker_model.csv")
    z = per[per["model"].isin(MODELS)].pivot(index="ticker", columns="model", values="PT_p").apply(lambda c: stats.norm.isf(c))
    for a, b in combinations(MODELS, 2):
        rows.append({"quantity": f"per-ticker PT statistic, cross-model, {a} vs {b}", "estimate": float(z[a].corr(z[b])), "n_days": 30})
    return pd.DataFrame(rows)


def chol(n: int, rho: float) -> np.ndarray:
    return np.linalg.cholesky((1 - rho) * np.eye(n) + rho * np.ones((n, n)))


def simulate_family(n_tests: int, rho: float, observed: int, rng) -> dict:
    L = chol(n_tests, rho)
    e = rng.standard_normal((N_SIMS, n_tests))
    z = e @ L.T
    counts = (z > Z_HIT).sum(axis=1)
    return {"expected": float(counts.mean()), "sd": float(counts.std(ddof=1)),
            "p_at_least_observed": float((counts >= observed).mean())}


def simulate_pool(n_tickers: int, n_models: int, rho_t: float, rho_m: float, observed: int, rng) -> dict:
    Lt, Lm = chol(n_tickers, rho_t), chol(n_models, rho_m)
    e = rng.standard_normal((N_SIMS, n_models, n_tickers))
    z = Lm @ e @ Lt.T  # corr = rho_m^[m!=m'] * rho_t^[i!=j]
    counts = (z > Z_HIT).sum(axis=(1, 2))
    return {"expected": float(counts.mean()), "sd": float(counts.std(ddof=1)),
            "p_at_least_observed": float((counts >= observed).mean())}


def main() -> None:
    runs = pd.concat([load_runs(str(REPO / "results" / "predictions")), load_runs(str(REPO / "results" / "comparators"))], ignore_index=True)
    runs = runs[runs["model"].isin(MODELS)]
    corr = estimate_correlations(runs)
    rng = np.random.default_rng(0)
    boot = bootstrap_pt_correlation(runs, rng)
    corr = pd.concat([corr, pd.DataFrame([{"quantity": k, "estimate": v, "n_days": N_BOOT} for k, v in boot.items()])],
                     ignore_index=True)
    corr.to_csv(OUT_CORR, index=False)
    print(corr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    per = pd.read_csv(REPO / "results" / "per_ticker_model.csv")
    observed = {m: int((per.loc[per["model"] == m, "PT_p"] < 0.05).sum()) for m in MODELS}
    n_tickers = {m: int((per["model"] == m).sum()) for m in MODELS}
    rho_hit = {m: float(boot[f"PT statistic, cross-ticker, {m} (block bootstrap)"]) for m in MODELS}
    rho_model_daily = float(np.mean([v for k, v in boot.items() if "cross-model" in k]))
    rho_model_stat = float(corr.loc[corr["quantity"].str.startswith("per-ticker PT statistic"), "estimate"].mean())

    rows = []
    print(f"\nhit rule: one-sided z > {Z_HIT:.3f} (PT p < 0.05); {N_SIMS} simulations")
    for m in MODELS:
        for rho in (0.0, 0.3, 0.5, round(rho_hit[m], 3)):
            r = simulate_family(n_tickers[m], rho, observed[m], rng)
            rows.append({"family": m, "n_tests": n_tickers[m], "observed_hits": observed[m], "rho_ticker": rho, "rho_model": np.nan,
                         "rho_source": "estimate" if rho == round(rho_hit[m], 3) else "assumed", **r})
            print(f"{m:<22} {n_tickers[m]} tests, observed {observed[m]}: rho_ticker={rho:<5} expected {r['expected']:.2f} sd {r['sd']:.2f} "
                  f"P(>= {observed[m]}) = {r['p_at_least_observed']:.3f}")
    k_all = sum(observed.values()); n_all = sum(n_tickers.values())
    rho_t_est = round(float(np.mean(list(rho_hit.values()))), 3)
    for rho_t in (0.0, 0.3, 0.5, rho_t_est):
        for rho_m in (0.0, 0.8, round(rho_model_daily, 3)):
            r = simulate_pool(30, 3, rho_t, rho_m, k_all, rng)
            rows.append({"family": "all three models", "n_tests": n_all, "observed_hits": k_all, "rho_ticker": rho_t, "rho_model": rho_m,
                         "rho_source": ("estimate" if rho_t == rho_t_est else "assumed") + "/" + ("estimate" if rho_m == round(rho_model_daily, 3) else "assumed"), **r})
            print(f"pooled 90 tests, observed {k_all}: rho_ticker={rho_t:<5} rho_model={rho_m:<5} expected {r['expected']:.2f} sd {r['sd']:.2f} "
                  f"P(>= {k_all}) = {r['p_at_least_observed']:.3f}")
    print(f"\nestimated rho_ticker (bootstrapped PT-statistic correlation, mean over models): {rho_t_est}; "
          f"estimated rho_model: bootstrapped PT statistics within ticker {rho_model_daily:.3f}; cross-sectional {rho_model_stat:.3f}")
    pd.DataFrame(rows).to_csv(OUT_NULL, index=False)
    print(f"wrote {OUT_CORR.relative_to(REPO)} and {OUT_NULL.relative_to(REPO)}")


if __name__ == "__main__":
    main()
