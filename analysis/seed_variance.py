"""Seed variance vs architecture variance on AAPL."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from backtest import backtest, prediction_diagnostics
from contracts import make_predictions
from dataset import build_dataset
from evaluate import evaluate_predictions
from experiments import load_runs
from walk_forward import WalkForwardSplitter

pd.set_option("display.width", 240, "display.max_columns", 40)
RULE = "=" * 112


def banner(t):
    print("\n" + RULE); print(t); print(RULE)


def main():
    runs = load_runs(str(REPO / "results/seeds"))
    raw = pd.read_csv(REPO / "results/raw/AAPL.csv", parse_dates=["Date"])
    ds = build_dataset(raw)
    folds = WalkForwardSplitter(1008, 252, 252).split(len(ds))
    y_train = {i: ds.y[np.asarray(tr)] for i, (tr, _) in enumerate(folds)}

    rows = []
    for (model, seed), block in runs.groupby(["model", "seed"], sort=True):
        block = block.sort_values("target_date").reset_index(drop=True)
        preds = make_predictions(
            target_date=block["target_date"],
            fold_id=block["fold_id"].to_numpy(int),
            close_t=block["close_t"].to_numpy(float),
            y_true=block["y_true"].to_numpy(float),
            y_pred=block["y_pred"].to_numpy(float),
        )
        ev = evaluate_predictions(preds, model, validate=False,
                                  y_train_by_fold=y_train)["pooled"]
        d = prediction_diagnostics(preds)
        bt = backtest(preds, cost_bps=7.5, validate=False)
        rows.append({
            "model": model, "seed": int(seed), "n": len(preds),
            "R2_OOS": ev["r2_oos"],
            "DA": ev["directional_accuracy"] * 100,
            "DA_minus_majority": ev["da_minus_majority"] * 100,
            "PT_p": ev.get("pt_p_value", np.nan),
            "sd_pred_bps": d["std_pred_bps"],
            "std_ratio": d["std_ratio"],
            "frac_positive": d["frac_positive"],
            "t_alpha": bt["market_adjusted"]["t_alpha"],
        })
    per = pd.DataFrame(rows)

    banner("PER SEED, AAPL, 12-fold config (same as the headline run)")
    for model in sorted(per["model"].unique()):
        block = per[per["model"] == model].set_index("seed").sort_index()
        print(f"\n  {model}")
        print(block[["R2_OOS", "DA", "DA_minus_majority", "PT_p",
                     "sd_pred_bps", "std_ratio", "frac_positive", "t_alpha"]]
              .to_string(float_format=lambda v: f"{v: .4f}"))

    banner("SEED SPREAD vs ARCHITECTURE SPREAD")
    metrics = ["R2_OOS", "DA", "sd_pred_bps"]
    seed_sd = per.groupby("model")[metrics].std(ddof=1)
    seed_sd.index = [f"across 5 seeds: {i}" for i in seed_sd.index]

    # Architecture spread: for each seed, the gap between the two models.
    wide = per.pivot(index="seed", columns="model", values=metrics)
    arch_gap = {}
    for m in metrics:
        cols = wide[m]
        arch_gap[m] = (cols.iloc[:, 0] - cols.iloc[:, 1]).abs()
    arch = pd.DataFrame(arch_gap)

    print("\n  Standard deviation ACROSS SEEDS, within each architecture:")
    print(seed_sd.to_string(float_format=lambda v: f"{v: .4f}"))
    print("\n  |BiLSTM - Transformer| ACROSS ARCHITECTURES, within each seed:")
    print(arch.to_string(float_format=lambda v: f"{v: .4f}"))

    print("\n  Head to head:")
    print(f"  {'metric':<14}{'mean seed sd':>14}{'mean arch gap':>16}{'ratio':>10}")
    for m in metrics:
        s = seed_sd[m].mean()
        a = arch[m].mean()
        print(f"  {m:<14}{s:>14.4f}{a:>16.4f}{a / s:>10.2f}x")

    print("\n  A ratio near or below 1 means the architecture difference is no")
    print("  larger than reseeding the same architecture: the comparison between")
    print("  BiLSTM and Transformer is not identifiable at one seed.")

    banner("RANGE ACROSS SEEDS (what a single-seed result could have reported)")
    for model in sorted(per["model"].unique()):
        b = per[per["model"] == model]
        print(f"\n  {model}")
        for m in ["R2_OOS", "DA", "sd_pred_bps"]:
            print(f"    {m:<14} min {b[m].min():+9.4f}   max {b[m].max():+9.4f}   "
                  f"range {b[m].max() - b[m].min():.4f}")
        print(f"    R2_OOS sign flips across seeds: "
              f"{'YES' if (b['R2_OOS'] > 0).any() and (b['R2_OOS'] < 0).any() else 'no'}")

    per.to_csv(Path(__file__).parent / "seed_study.csv", index=False)


if __name__ == "__main__":
    main()
