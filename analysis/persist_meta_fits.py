"""Persist the headline meta-learner's per-fold fits, including the RidgeCV
penalty it selected, to a committed CSV.

fit_stacked_meta already records, for every fold it fits, the selected alpha,
the intercept and the standardised coefficients (its ``fold_weights``); the
pipeline prints them and never writes them. This script rebuilds the meta
frame from the committed base predictions of the headline run
(results/headline/AAPL__frozen__seed42.parquet) and the frozen CSV (for the
VIX level at each feature date), refits both meta variants exactly as
main.py does, asserts that the refit reproduces the parquet's own meta
predictions, and writes the fold records to
results/headline/AAPL__frozen__seed42_meta_fits.csv.

    python analysis/persist_meta_fits.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from contracts import make_predictions  # noqa: E402
from dataset import build_dataset  # noqa: E402
from experiments import load_runs  # noqa: E402
from meta_ensemble import MIN_TRAIN_FOLDS, build_meta_frame, fit_stacked_meta  # noqa: E402

PARQUET_DIR = REPO / "results" / "headline"
FROZEN_CSV = REPO / "docs" / "frozen_aapl_raw.csv"
OUT = PARQUET_DIR / "AAPL__frozen__seed42_meta_fits.csv"
BASE = {"tree": "Tree Ensemble", "lstm": "BiLSTM", "transformer": "Transformer"}


def frame(block: pd.DataFrame) -> pd.DataFrame:
    block = block.sort_values("target_date").reset_index(drop=True)
    return make_predictions(
        target_date=block["target_date"], fold_id=block["fold_id"].to_numpy(int),
        close_t=block["close_t"].to_numpy(float), y_true=block["y_true"].to_numpy(float),
        y_pred=block["y_pred"].to_numpy(float),
    )


def main() -> None:
    runs = load_runs(str(PARQUET_DIR))
    runs = runs[(runs["ticker"] == "AAPL") & (runs["regime"] == "frozen")]
    by_model = {m: frame(b) for m, b in runs.groupby("model")}
    base_results = {key: {"model_name": name, "predictions": by_model[name]} for key, name in BASE.items()}

    dataset = build_dataset(pd.read_csv(FROZEN_CSV, parse_dates=["Date"]))
    meta_frame = build_meta_frame(base_results, dataset)

    records = []
    for use_vix in (True, False):
        result = fit_stacked_meta(meta_frame, use_vix=use_vix, min_train_folds=MIN_TRAIN_FOLDS)
        name = result["model_name"]
        stored = by_model[name].set_index("target_date")["y_pred"]
        refit = result["predictions"].set_index("target_date")["y_pred"]
        assert len(stored) == len(refit), (name, len(stored), len(refit))
        gap = float(np.max(np.abs(stored.reindex(refit.index).to_numpy() - refit.to_numpy())))
        assert gap < 1e-9, f"{name}: refit differs from the committed parquet by {gap:.2e}"
        weights = result["fold_weights"].copy()
        weights.insert(0, "variant", name)
        weights["skipped_folds"] = ",".join(str(k) for k in result["skipped_folds"])
        records.append(weights)
        print(f"  {name:<22} refit == parquet (max |diff| {gap:.1e}); "
              f"alphas by fold: {weights['alpha'].astype(int).tolist()}")

    table = pd.concat(records, ignore_index=True)
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}  ({len(table)} rows)")


if __name__ == "__main__":
    main()
