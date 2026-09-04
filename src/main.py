"""Hybrid Stock Market Prediction Model - CLI orchestrator.

Every model, baseline and ensemble in this pipeline forecasts the next-day log
return and reports predictions keyed by ``target_date``. Price appears only as
a clearly-labelled secondary diagnostic; see REFACTOR_PLAN.md sections 1 and 4
for why.

Expect modest numbers. Honest daily equity return forecasting produces R2_OOS
somewhere around zero and directional accuracy in the low fifties, and most
models will be statistically indistinguishable from the zero-return baseline.
The previous MAPE of 0.89% and R2 of 0.975 were the naive baseline wearing a
transformer costume.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from baselines import run_all_baselines
from contracts import align_predictions
from dataset import Dataset, build_dataset
from evaluate import compare_evaluations, dm_table, evaluate_predictions, print_evaluation
from fetch_data import fetch_stock_data
from lstm_model import train_lstm_model
from meta_ensemble import (
    MIN_TRAIN_FOLDS,
    build_meta_frame,
    compare_vix_gating,
    fold_respecting_intervals,
    print_calibration_table,
    print_vix_gating_report,
    residual_quantiles,
    scale_for_horizon,
    vix_tercile_weights,
)
from model_utils import DEMO_FORECAST_CAVEAT
from transformer_model import train_transformer_model
from tree_model import train_tree_model
from walk_forward import WalkForwardSplitter, print_fold_summary

Folds = List[Tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Pipeline building blocks (importable)
# ---------------------------------------------------------------------------


def fetch_and_prepare_data(
    ticker: str, start_date: str, end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV plus macro context and engineer the leak-free feature set."""
    return fetch_stock_data(ticker, start_date, end_date)


def create_walk_forward_folds(
    dataset: Dataset,
    min_train_size: int = 504,
    test_size: int = 63,
    step_size: int = 63,
) -> Folds:
    """Expanding-window folds over dataset rows.

    Note these index the *dataset*, whose final row was dropped because its
    target is unknown, not the raw frame.
    """
    splitter = WalkForwardSplitter(
        min_train_size=min_train_size, test_size=test_size, step_size=step_size
    )
    folds = splitter.split(len(dataset))
    print_fold_summary(folds, dataset.feature_date)
    return folds


def training_returns_by_fold(dataset: Dataset, folds: Folds) -> Dict[int, np.ndarray]:
    """The history each fold's R2_OOS benchmark must be seeded with.

    Without this the expanding-mean benchmark starts cold and every model's
    R2_OOS is biased upward; see evaluate.evaluate_predictions.
    """
    return {i: dataset.y[np.asarray(train)] for i, (train, _) in enumerate(folds)}


def train_base_models(
    dataset: Dataset,
    folds: Folds,
    optimize: bool = False,
    seed: int = 42,
    epochs: int = 100,
    demo_forecast_days: int = 0,
    demo_history: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, Any]]:
    """Train all three base models over the same folds."""
    demo_kwargs = {
        "demo_forecast_days": demo_forecast_days,
        "demo_history": demo_history,
    }

    print("\n[2/7] Tree ensemble (walk-forward)...")
    tree = train_tree_model(dataset, folds, random_state=seed, **demo_kwargs)

    print("\n[3/7] Bidirectional LSTM (walk-forward)...")
    lstm = train_lstm_model(dataset, folds, epochs=epochs, seed=seed, **demo_kwargs)

    print("\n[4/7] Time-series Transformer (walk-forward)...")
    transformer = train_transformer_model(
        dataset, folds, epochs=epochs, optimize=optimize, seed=seed, **demo_kwargs
    )

    return {"tree": tree, "lstm": lstm, "transformer": transformer}


def run_pipeline(
    ticker: str = "AAPL",
    start_date: str = "2010-01-01",
    optimize: bool = False,
    min_train_size: int = 504,
    test_size: int = 63,
    step_size: int = 63,
    seed: int = 42,
    epochs: int = 100,
    demo_forecast_days: int = 0,
    meta_min_train_folds: int = MIN_TRAIN_FOLDS,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Fetch, walk-forward train, stack, evaluate, report."""
    print("=" * 72)
    print(f"  HYBRID STOCK PREDICTOR - {ticker}   (next-day log return)")
    print(f"  History from {start_date} | seed {seed}")
    print(f"  Optuna: {'on' if optimize else 'off'} | "
          f"Walk-forward: min_train={min_train_size}, test={test_size}, step={step_size}")
    if demo_forecast_days:
        print(f"  Demo forecast: {demo_forecast_days} days. {DEMO_FORECAST_CAVEAT}")
    print("=" * 72)

    print("\n[1/7] Fetching data and engineering features...")
    raw = fetch_and_prepare_data(ticker, start_date)
    if raw is None:
        print("ERROR: failed to fetch data.")
        return {"error": "data_fetch_failed"}

    dataset = build_dataset(raw)
    print(f"  Dataset: {len(dataset)} rows x {dataset.n_features} features")

    folds = create_walk_forward_folds(dataset, min_train_size, test_size, step_size)
    if not folds:
        print(
            "ERROR: not enough data for walk-forward validation. "
            "Reduce --min-train or fetch a longer history."
        )
        return {"error": "insufficient_data_for_wf"}
    if len(folds) <= meta_min_train_folds:
        print(
            f"WARNING: {len(folds)} folds is not enough for out-of-fold stacking "
            f"with meta_min_train_folds={meta_min_train_folds}; the meta-ensemble "
            "will be skipped."
        )

    y_train_by_fold = training_returns_by_fold(dataset, folds)

    base_results = train_base_models(
        dataset,
        folds,
        optimize=optimize,
        seed=seed,
        epochs=epochs,
        demo_forecast_days=demo_forecast_days,
        demo_history=raw,
    )

    print("\n[5/7] Baselines...")
    baseline_results = run_all_baselines(dataset, folds, seed=seed)

    print("\n[6/7] Out-of-fold meta-ensemble...")
    meta_frame = build_meta_frame(base_results, dataset)

    gating = None
    calibration = None
    try:
        gating = compare_vix_gating(
            meta_frame,
            min_train_folds=meta_min_train_folds,
            y_train_by_fold=y_train_by_fold,
        )
        skipped = gating["with_vix"]["skipped_folds"]
        print(
            f"  Folds {skipped} produce no meta-prediction: a meta-model for "
            f"fold k is fitted only on folds 0..k-1. This is correct, and every "
            f"meta-prediction reported below is genuinely out-of-sample."
        )
        print_vix_gating_report(gating)

        terciles = vix_tercile_weights(meta_frame, meta_min_train_folds)
        print(f"\n  {'-' * 18} BASE WEIGHTS BY VIX TERCILE {'-' * 17}")
        print(terciles.to_string())

        calibration = fold_respecting_intervals(gating["with_vix"]["predictions"])
        print_calibration_table(calibration)
    except (RuntimeError, ValueError) as exc:
        print(f"  Meta-ensemble skipped: {exc}")
        terciles = None

    print("\n[7/7] Evaluating...")
    evaluations = []
    all_predictions: Dict[str, pd.DataFrame] = {}

    for key, result in base_results.items():
        evaluation = evaluate_predictions(
            result["predictions"], result["model_name"], y_train_by_fold=y_train_by_fold
        )
        print_evaluation(evaluation)
        evaluations.append(evaluation)
        all_predictions[result["model_name"]] = result["predictions"]

    if gating is not None:
        for side in ("with_vix", "without_vix"):
            evaluations.append(gating["evaluations"][side])
            all_predictions[gating[side]["model_name"]] = gating[side]["predictions"]
        print_evaluation(gating["evaluations"]["with_vix"])

    for key, result in baseline_results.items():
        evaluation = evaluate_predictions(
            result["predictions"], result["model_name"], y_train_by_fold=y_train_by_fold
        )
        evaluations.append(evaluation)
        all_predictions[result["model_name"]] = result["predictions"]

    os.makedirs("data", exist_ok=True)
    comparison = compare_evaluations(
        evaluations, save_path="data/model_comparison.csv"
    )

    # Every comparison in the paper needs a DM test against the benchmark.
    reference = baseline_results["zero_return"]["model_name"]
    dm = dm_table(all_predictions, reference=reference)
    if not dm.empty:
        print(f"\n  {'-' * 14} DIEBOLD-MARIANO vs {reference} {'-' * 14}")
        print(dm.to_string())
        print(
            "\n  A model is only better than the benchmark if its p-value is "
            "small. Lower RMSE alone is not a finding."
        )
        dm.to_csv("data/diebold_mariano.csv")

    aligned = align_predictions(
        {name: df for name, df in all_predictions.items()}, require_identical=False
    )
    aligned.to_csv("data/aligned_predictions.csv", index=False)

    demo = _collect_demo_forecasts(base_results, gating, calibration)
    if demo is not None:
        demo.to_csv("data/combined_predictions.csv", index=False)
        print(f"\n  Demo forecast written. {DEMO_FORECAST_CAVEAT}")

    if make_plots:
        _make_plots(dataset, folds, base_results, aligned, comparison)

    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE - primary metrics are in return space")
    print("=" * 72)

    return {
        "raw": raw,
        "dataset": dataset,
        "folds": folds,
        "base_results": base_results,
        "baseline_results": baseline_results,
        "meta_frame": meta_frame,
        "gating": gating,
        "vix_terciles": terciles,
        "calibration": calibration,
        "evaluations": evaluations,
        "comparison_df": comparison,
        "diebold_mariano": dm,
        "aligned": aligned,
    }


def _collect_demo_forecasts(base_results, gating, calibration) -> Optional[pd.DataFrame]:
    """Assemble the illustrative multi-step forecasts, if any were produced."""
    demos = {
        key: result["demo_forecast"]
        for key, result in base_results.items()
        if "demo_forecast" in result
    }
    if not demos:
        return None

    first = next(iter(demos.values()))
    combined = pd.DataFrame({"date": first["date"]})
    for key, frame in demos.items():
        combined[f"predicted_log_return_{key}"] = frame["predicted_log_return"].to_numpy()
        price_column = [c for c in frame.columns if c.startswith("Predicted Close")][0]
        combined[price_column] = frame[price_column].to_numpy()

    # Horizon-scaled intervals around the mean demo return, per 7.4.
    if gating is not None and calibration is not None:
        q_low, q_high = residual_quantiles(gating["with_vix"]["predictions"])
        horizons = np.arange(1, len(combined) + 1)
        low, high = scale_for_horizon(q_low, q_high, horizons)
        mean_return = combined[
            [c for c in combined.columns if c.startswith("predicted_log_return")]
        ].mean(axis=1)
        combined["interval_lower_log_return"] = mean_return + low
        combined["interval_upper_log_return"] = mean_return + high

    combined["caveat"] = DEMO_FORECAST_CAVEAT
    return combined


def _make_plots(dataset, folds, base_results, aligned, comparison) -> None:
    """Charts. Kept deliberately thin; they are not evidence."""
    try:
        from visualize import plot_feature_importance, plot_walk_forward_folds

        os.makedirs("images", exist_ok=True)
        plot_walk_forward_folds(
            dataset.feature_date, dataset.close_t, folds
        )
        plot_feature_importance(
            base_results["tree"]["feature_names"],
            base_results["tree"]["mean_feature_importances"],
        )
    except Exception as exc:  # pragma: no cover - plotting is cosmetic
        print(f"  Plotting skipped: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid stock predictor - walk-forward validated, "
        "return-space evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epochs", type=int, default=100, help="Neural training epochs. Default: 100"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna search. Safe: it only ever sees training sequences.",
    )
    parser.add_argument(
        "--demo-forecast",
        type=int,
        default=0,
        metavar="DAYS",
        help=(
            "Produce a recursive multi-step forecast of DAYS business days.\n"
            "ILLUSTRATIVE ONLY - excluded from every metric and every claim.\n"
            "Each step feeds a synthetic bar back into the features and the\n"
            "errors compound. Off by default."
        ),
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=504,
        help="Minimum training window (trading days). Default: 504 (~2 years)",
    )
    parser.add_argument(
        "--test-size", type=int, default=63, help="Test fold size. Default: 63 (~3 months)"
    )
    parser.add_argument(
        "--step-size", type=int, default=63, help="Step between folds. Default: 63"
    )
    parser.add_argument(
        "--meta-min-folds",
        type=int,
        default=MIN_TRAIN_FOLDS,
        help=(
            "Folds before this index produce no meta-prediction, because a\n"
            "meta-model for fold k is fitted only on folds 0..k-1. Default: 2"
        ),
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    result = run_pipeline(
        ticker=args.ticker.upper(),
        start_date=args.start,
        optimize=args.optimize,
        min_train_size=args.min_train,
        test_size=args.test_size,
        step_size=args.step_size,
        seed=args.seed,
        epochs=args.epochs,
        demo_forecast_days=args.demo_forecast,
        meta_min_train_folds=args.meta_min_folds,
        make_plots=not args.no_plots,
    )

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
