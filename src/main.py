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

from backtest import DEFAULT_COST_BPS, NOT_APPLICABLE, backtest_table
from baselines import run_all_baselines
from contracts import (
    align_predictions,
    common_evaluation_window,
    restrict_all,
)
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
    cost_bps: float = DEFAULT_COST_BPS,
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

        # The interval calibration table is computed further down, on the
        # common evaluation window, so it covers the same days as every other
        # primary result.
    except (RuntimeError, ValueError) as exc:
        print(f"  Meta-ensemble skipped: {exc}")
        terciles = None

    print("\n[7/7] Evaluating...")

    # Collect every model's full-range predictions first. Nothing is scored
    # until the common evaluation window is known.
    all_predictions: Dict[str, pd.DataFrame] = {
        result["model_name"]: result["predictions"]
        for result in list(base_results.values()) + list(baseline_results.values())
    }
    if gating is not None:
        for side in ("with_vix", "without_vix"):
            all_predictions[gating[side]["model_name"]] = gating[side]["predictions"]

    window = common_evaluation_window(all_predictions)
    primary = restrict_all(all_predictions, window)

    print(f"\n  {'-' * 16} COMMON EVALUATION WINDOW {'-' * 16}")
    print(f"  {window.describe()}")
    print(
        "  Every primary comparison below runs on exactly these days. A "
        "meta-model for\n  fold k is fitted only on folds 0..k-1, so the "
        "meta-learner cannot forecast the\n  earliest folds. Scoring each "
        "model over its own range would compare a base\n  model measured "
        "across those folds against a meta-learner that never saw them."
    )
    dropped = {n: d for n, d in window.dropped_by_model.items() if d}
    if dropped:
        print("  Days dropped from each model to reach this window:")
        for name, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<26} -{count}")

    os.makedirs("data", exist_ok=True)

    base_model_names = {r["model_name"] for r in base_results.values()}
    evaluations = []
    for name, predictions in primary.items():
        evaluation = evaluate_predictions(
            predictions, name, y_train_by_fold=y_train_by_fold
        )
        evaluations.append(evaluation)
        if name in base_model_names or "+VIX" in name:
            print_evaluation(evaluation)

    comparison = compare_evaluations(evaluations, save_path="data/model_comparison.csv")

    # Interval calibration, on the same window as everything else.
    if gating is not None:
        calibration = fold_respecting_intervals(
            primary[gating["with_vix"]["model_name"]]
        )
        print_calibration_table(calibration)

    # Every comparison in the paper needs a DM test against the benchmark.
    reference = baseline_results["zero_return"]["model_name"]
    dm = dm_table(primary, reference=reference)
    if not dm.empty:
        print(f"\n  {'-' * 14} DIEBOLD-MARIANO vs {reference} {'-' * 14}")
        print(dm.to_string())
        print(
            "\n  A model is only better than the benchmark if its p-value is "
            "small. Lower RMSE alone is not a finding."
        )
        dm.to_csv("data/diebold_mariano.csv")

    # Identical date sets now, so this is an exact merge with an assertion.
    aligned = align_predictions(primary, require_identical=True)
    aligned.to_csv("data/aligned_predictions.csv", index=False)

    # Section 8: directional accuracy does not pay for lunch. Buy-and-hold is
    # computed on the window, not on whichever model came first in the dict.
    economics = backtest_table(
        primary, cost_bps=cost_bps, benchmark_predictions=primary[reference]
    )
    print()
    print(f"  {'-' * 12} ECONOMICS (long/flat, {cost_bps:.1f} bps round-trip) "
          f"{'-' * 12}")
    print(f"  All rows and buy-and-hold on the common window: {window.describe()}")
    print(economics.to_string())
    print(
        f"  '{NOT_APPLICABLE}' marks a strategy that never takes a position, so "
        "its Sharpe is 0/0 rather than bad."
    )
    if not economics.drop(index="Buy and hold")["Beats B&H net"].any():
        print(
            "\n  No model beats buy-and-hold after costs. Stated plainly: that "
            "is a publishable finding, and reviewers respect it."
        )
    economics.to_csv("data/backtest.csv")

    # Secondary. Models covering more than the window, over their own full
    # range. Clearly labelled, never mixed into the primary table.
    secondary = _full_range_table(all_predictions, window, y_train_by_fold)

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
        "economics": economics,
        "window": window,
        "primary_predictions": primary,
        "secondary_comparison_df": secondary,
    }


def _full_range_table(all_predictions, window, y_train_by_fold):
    """SECONDARY table: models that cover more days than the common window.

    Reported for completeness only. These rows are NOT comparable with each
    other or with the primary table, because each covers a different set of
    days -- which is exactly the problem the common window exists to remove.
    """
    wider = {
        name: df
        for name, df in all_predictions.items()
        if len(df) > window.n
    }
    if not wider:
        return None

    evaluations = [
        evaluate_predictions(df, name, y_train_by_fold=y_train_by_fold)
        for name, df in wider.items()
    ]
    table = compare_evaluations(
        evaluations, save_path="data/model_comparison_full_range.csv"
    )
    print(
        "  SECONDARY, full range per model. Rows above cover different day "
        "sets from\n  each other and from the primary table; they are not "
        "comparable. Primary\n  results are the common-window table only."
    )
    return table


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
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=DEFAULT_COST_BPS,
        help=(
            "Round-trip transaction cost in basis points for the backtest.\n"
            f"Default: {DEFAULT_COST_BPS} (liquid US equities)"
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
        cost_bps=args.cost_bps,
        make_plots=not args.no_plots,
    )

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
