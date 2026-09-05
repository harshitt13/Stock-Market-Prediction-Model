"""Tasks 1 and 2: widened ridge grid, and why the tree scores PT = -2.12."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from contracts import common_evaluation_window, restrict_all
from dataset import build_dataset
from evaluate import directional_accuracy, pesaran_timmermann
from fetch_data import engineer_features
from lstm_model import train_lstm_model
from meta_ensemble import RIDGE_ALPHAS, build_meta_frame, fit_stacked_meta
from transformer_model import train_transformer_model
from tree_model import train_tree_model
from walk_forward import WalkForwardSplitter

pd.set_option("display.width", 250, "display.max_columns", 60)
RULE = "=" * 122


def banner(text):
    print("\n" + RULE)
    print(text)
    print(RULE)


def main():
    raw = pd.read_csv(REPO / "tests/fixtures/aapl_raw.csv", parse_dates=["Date"])
    dataset = build_dataset(engineer_features(raw).dropna().reset_index(drop=True))
    folds = WalkForwardSplitter(400, 60, 60).split(len(dataset))

    tree = train_tree_model(dataset, folds, save_model=False)
    lstm = train_lstm_model(dataset, folds, epochs=2, save_model=False, verbose=False)
    transformer = train_transformer_model(
        dataset, folds, epochs=2, save_model=False, verbose=False)
    base = {"tree": tree, "lstm": lstm, "transformer": transformer}

    # ---- Task 1: where does the widened ridge grid land? -----------------
    banner("1. RIDGE ALPHA WITH THE WIDENED GRID (1e-3 .. 1e6)")
    print(f"  grid searched: {[f'{a:g}' for a in RIDGE_ALPHAS]}")
    meta_frame = build_meta_frame(base, dataset)
    for use_vix in (True, False):
        fitted = fit_stacked_meta(meta_frame, use_vix=use_vix, min_train_folds=2)
        print(f"\n  {fitted['model_name']}  features={fitted['features']}")
        weights = fitted["fold_weights"].copy()
        ceiling = max(RIDGE_ALPHAS)
        floor = min(RIDGE_ALPHAS)
        weights["at_ceiling"] = weights["alpha"] >= ceiling
        weights["at_floor"] = weights["alpha"] <= floor
        print(weights.to_string(index=False, float_format=lambda v: f"{v: .5g}"))
        n_ceiling = int(weights["at_ceiling"].sum())
        print(f"    folds pinned at the ceiling ({ceiling:g}): "
              f"{n_ceiling}/{len(weights)}")

    # ---- Task 2: why is the tree worse than chance? ----------------------
    banner("2. TREE ENSEMBLE, PER FOLD: PREDICTION vs TRAINING TARGET RANGE")
    rows = []
    predictions = tree["predictions"]
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        block = predictions[predictions["fold_id"] == fold_id]
        if block.empty:
            continue
        y_train = dataset.y[np.asarray(train_idx)]
        y_test = block["y_true"].to_numpy(float)
        y_pred = block["y_pred"].to_numpy(float)

        outside = float(
            np.mean((y_test < y_train.min()) | (y_test > y_train.max()))
        )
        ks = stats.ks_2samp(y_train, y_test)
        da = directional_accuracy(y_test, y_pred)["directional_accuracy"]

        rows.append({
            "fold": fold_id,
            "n_train": len(y_train),
            "mean_pred_bps": y_pred.mean() * 1e4,
            "mean_true_bps": y_test.mean() * 1e4,
            "train_mean_bps": y_train.mean() * 1e4,
            "pred_min_bps": y_pred.min() * 1e4,
            "pred_max_bps": y_pred.max() * 1e4,
            "train_y_min_bps": y_train.min() * 1e4,
            "train_y_max_bps": y_train.max() * 1e4,
            "test_outside_train_range": outside,
            "ks_stat": ks.statistic,
            "ks_p": ks.pvalue,
            "DA": da * 100,
        })
    per_fold = pd.DataFrame(rows).set_index("fold")
    print(per_fold[[
        "n_train", "mean_pred_bps", "mean_true_bps", "train_mean_bps", "DA",
        "test_outside_train_range", "ks_stat", "ks_p",
    ]].to_string(float_format=lambda v: f"{v: .4f}"))

    banner("2b. ARE THE TREE'S PREDICTIONS CAPPED BY ITS TRAINING TARGET RANGE?")
    caps = per_fold[[
        "pred_min_bps", "pred_max_bps", "train_y_min_bps", "train_y_max_bps",
    ]].copy()
    caps["pred_range_bps"] = caps["pred_max_bps"] - caps["pred_min_bps"]
    caps["train_range_bps"] = caps["train_y_max_bps"] - caps["train_y_min_bps"]
    caps["range_ratio"] = caps["pred_range_bps"] / caps["train_range_bps"]
    caps["within_train_range"] = (
        (caps["pred_min_bps"] >= caps["train_y_min_bps"])
        & (caps["pred_max_bps"] <= caps["train_y_max_bps"])
    )
    print(caps.to_string(float_format=lambda v: f"{v: .2f}"))
    print("\n  A tree averages training targets, so its predictions cannot leave")
    print("  [min(train_y), max(train_y)]. within_train_range confirms that;")
    print("  range_ratio shows how much of that span it actually uses.")

    banner("2c. DIRECTION OF THE ERROR - IS IT SYSTEMATIC?")
    y_true = predictions["y_true"].to_numpy(float)
    y_pred = predictions["y_pred"].to_numpy(float)
    pt = pesaran_timmermann(y_true, y_pred)
    print(f"  pooled PT statistic : {pt['pt_statistic']:.4f}")
    print(f"  hit rate            : {pt['hit_rate']:.4f}")
    print(f"  under independence  : {pt['hit_rate_under_independence']:.4f}")
    print(f"  corr(pred, true)    : {np.corrcoef(y_pred, y_true)[0, 1]:+.4f}")
    print()
    print("  Inverting the tree's sign would give:")
    inv = directional_accuracy(y_true, -y_pred)
    print(f"    DA {inv['directional_accuracy'] * 100:.2f}% on "
          f"{inv['n_evaluated']} evaluated days")
    inv_pt = pesaran_timmermann(y_true, -y_pred)
    print(f"    PT {inv_pt['pt_statistic']:.4f}, p = {inv_pt['pt_p_value']:.4f}")
    print("\n  A model reliably worse than chance is only interesting if the")
    print("  inversion is significant. If neither direction is significant, the")
    print("  negative statistic is noise on 420 days, not an inverted signal.")

    per_fold.to_csv(Path(__file__).parent / "tree_per_fold.csv")


if __name__ == "__main__":
    main()
