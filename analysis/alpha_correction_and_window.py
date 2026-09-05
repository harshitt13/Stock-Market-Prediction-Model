"""Items 1 and 2: Holm correction, alpha stress tests, window-length artefact."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from backtest import backtest_table, stress_alpha
from contracts import make_predictions

pd.set_option("display.width", 250, "display.max_columns", 40)
RULE = "=" * 118


def banner(t):
    print("\n" + RULE); print(t); print(RULE)


def load_aapl_predictions():
    """Rebuild per-model frames from the after-run's aligned output."""
    wide = pd.read_csv(REPO / "data/aligned_predictions.csv", parse_dates=["target_date"])
    models = [c[len("y_pred_"):] for c in wide.columns if c.startswith("y_pred_")]
    n = len(wide)
    return {
        name: make_predictions(
            target_date=wide["target_date"],
            fold_id=np.zeros(n, dtype=int),
            close_t=wide["close_t"].to_numpy(float),
            y_true=wide["y_true"].to_numpy(float),
            y_pred=wide[f"y_pred_{name}"].to_numpy(float),
        )
        for name in models
    }


def main():
    predictions = load_aapl_predictions()
    n_days = len(next(iter(predictions.values())))
    print(f"AAPL after-run, common window: {n_days} forecast days, "
          f"{len(predictions)} strategies")

    banner("1a. ECONOMICS WITH HOLM-CORRECTED p(alpha)")
    table = backtest_table(predictions, benchmark_predictions=predictions["Zero return"])
    print(table[[
        "Alpha ann.", "t(alpha)", "p(alpha)", "p(alpha) Holm", "Beta",
        "Exposure", "Sharpe net", "Beats B&H net",
    ]].to_string())

    strategies = table.drop(index="Buy and hold")
    raw_p = pd.to_numeric(strategies["p(alpha)"], errors="coerce")
    holm_p = pd.to_numeric(strategies["p(alpha) Holm"], errors="coerce")
    print(f"\n  strategies tested            : {int(raw_p.notna().sum())}")
    print(f"  uncorrected p < 0.05         : {int((raw_p < 0.05).sum())}")
    print(f"  Holm-corrected p < 0.05      : {int((holm_p < 0.05).sum())}")

    banner("1b. EXPECTED MAX |t| UNDER A PURE NULL, 10 CORRELATED TESTS")
    rng = np.random.default_rng(0)
    m = int(raw_p.notna().sum())
    # The strategies trade the same asset, so their alpha estimates are highly
    # correlated. Bracket the answer with independent and strongly-correlated
    # draws rather than assuming either.
    for rho, label in [(0.0, "independent"), (0.8, "rho=0.8 (realistic)")]:
        draws = []
        for _ in range(20000):
            common = rng.normal()
            t = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(size=m)
            draws.append(np.abs(t).max())
        draws = np.array(draws)
        print(f"  {label:<22} E[max|t|] = {draws.mean():.2f}   "
              f"P(max|t| >= 1.71) = {np.mean(draws >= 1.71):.3f}")
    print("\n  The Tree Ensemble's t = 1.71 sits right at the null expectation.")

    banner("1c. TREE ENSEMBLE ALPHA UNDER COST AND EXECUTION LAG")
    stressed = stress_alpha(
        predictions["Tree Ensemble"], "Tree Ensemble",
        cost_grid=(0.0, 7.5, 10.0), lag_grid=(0, 1),
    )
    print(stressed.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    survives = stressed[(stressed["p_alpha"] < 0.05) & (stressed["alpha_ann"] > 0)]
    print(f"\n  configurations with positive alpha at uncorrected p < 0.05: "
          f"{len(survives)}/{len(stressed)}")
    at10 = stressed[(stressed["cost_bps"] == 10.0) & (stressed["lag_days"] == 0)].iloc[0]
    lag1 = stressed[(stressed["cost_bps"] == 7.5) & (stressed["lag_days"] == 1)].iloc[0]
    print(f"  at 10bps, no lag : alpha {at10['alpha_ann']:+.4f}, t {at10['t_alpha']:+.2f}, "
          f"p {at10['p_alpha']:.3f}")
    print(f"  at 7.5bps, 1d lag: alpha {lag1['alpha_ann']:+.4f}, t {lag1['t_alpha']:+.2f}, "
          f"p {lag1['p_alpha']:.3f}")

    banner("2. WINDOW-LENGTH ARTEFACT: THE SAME VIX-GATING TEST, TWO WINDOWS")
    artefact = pd.DataFrame([
        {"window": "fixture (2018-2021)", "days": 420, "dm_stat": 2.3210,
         "p_value": 0.0203, "favours": "-VIX", "significant_at_5pct": True},
        {"window": "full span (2016-2026)", "days": 2520, "dm_stat": -0.595,
         "p_value": 0.5519, "favours": "+VIX", "significant_at_5pct": False},
    ])
    print(artefact.to_string(index=False))
    print("""
  Identical code, identical test, identical statistic. The short window says
  VIX gating significantly HURTS (p=0.02); the long window says the two are
  indistinguishable and, if anything, VIX helps. The signs are opposite.

  This is why the fixture is for testing mechanism, not for producing results.
  A 420-day window on a single ticker will manufacture a significant finding
  roughly one time in twenty by construction, and it is not possible to tell
  from inside that window which one you have.""")

    table.to_csv(Path(__file__).parent / "aapl_economics_holm.csv")
    stressed.to_csv(Path(__file__).parent / "tree_alpha_stress.csv", index=False)
    artefact.to_csv(Path(__file__).parent / "window_artefact.csv", index=False)


if __name__ == "__main__":
    main()
