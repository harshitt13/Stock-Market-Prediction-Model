"""Experiment runner: sweep tickers, regimes, models and seeds.

REFACTOR_PLAN.md section 9. One stock is an anecdote. A result that holds on
AAPL and nowhere else is a property of AAPL.

Two rules this module exists to enforce:

- **Raw per-fold, per-ticker predictions are persisted to disk**, and every
  aggregate is computed by reading those files back. Aggregates are recomputed
  many times while writing a paper and re-training for each one is both slow
  and a good way to end up reporting numbers from a run whose code no longer
  exists.
- **Neural results are run over several seeds and the seed variance is
  reported.** A single-seed neural result is not credible; the spread across
  seeds is frequently larger than the difference between models.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from baselines import run_all_baselines
from dataset import build_dataset
from evaluate import evaluate_predictions
from lstm_model import train_lstm_model
from meta_ensemble import build_meta_frame, fit_stacked_meta
from transformer_model import train_transformer_model
from tree_model import train_tree_model
from walk_forward import WalkForwardSplitter

RESULTS_DIR = "results"
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

#: Thirty large-cap US names across sectors. All are current constituents.
DEFAULT_TICKERS: List[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL",
    # Communication services
    "GOOGL", "META", "NFLX", "DIS",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "PG", "KO", "WMT",
    # Financials
    "JPM", "BAC", "GS", "BRK-B",
    # Health care
    "JNJ", "UNH", "PFE", "MRK",
    # Industrials / energy / utilities
    "CAT", "BA", "XOM", "NEE",
]

SURVIVORSHIP_WARNING = (
    "SURVIVORSHIP BIAS: DEFAULT_TICKERS are all names that still trade today. "
    "Companies that were delisted, acquired or went bankrupt over the sample "
    "are absent, which biases every aggregate upward. Any paper using this "
    "ticker list must state this explicitly. To fix it properly, source "
    "point-in-time index constituents including delisted names."
)

#: Regime windows from section 9. End dates are exclusive.
REGIMES: Dict[str, Tuple[str, Optional[str]]] = {
    "pre_2020": ("2010-01-01", "2020-01-01"),
    "crash_2020": ("2019-06-01", "2021-01-01"),
    "drawdown_2021_2022": ("2020-06-01", "2023-01-01"),
    "post_2023": ("2022-06-01", None),
    "full": ("2010-01-01", None),
}

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

PREDICTION_SCHEMA = [
    "target_date", "fold_id", "close_t", "y_true", "y_pred",
    "model", "ticker", "regime", "seed",
]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


def run_path(ticker: str, regime: str, seed: int, results_dir: str = PREDICTIONS_DIR) -> str:
    """Where one (ticker, regime, seed) run's predictions live."""
    extension = "parquet" if _parquet_available() else "csv"
    safe = ticker.replace("/", "-")
    return os.path.join(results_dir, f"{safe}__{regime}__seed{seed}.{extension}")


def save_run(frame: pd.DataFrame, path: str) -> str:
    """Persist one run. Parquet when an engine is installed, CSV otherwise."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".parquet"):
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def load_runs(results_dir: str = PREDICTIONS_DIR) -> pd.DataFrame:
    """Read every persisted run back off disk.

    Aggregates are computed from this, never from whatever happens to still be
    in memory after a sweep.
    """
    paths = sorted(
        glob.glob(os.path.join(results_dir, "*.parquet"))
        + glob.glob(os.path.join(results_dir, "*.csv"))
    )
    if not paths:
        raise FileNotFoundError(f"no persisted runs under {results_dir!r}")

    frames = []
    for path in paths:
        if path.endswith(".parquet"):
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_csv(path, parse_dates=["target_date"]))

    combined = pd.concat(frames, ignore_index=True)
    combined["target_date"] = pd.to_datetime(combined["target_date"])
    return combined


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _default_loader(ticker: str, start: str, end: Optional[str]) -> Optional[pd.DataFrame]:
    from fetch_data import fetch_stock_data

    return fetch_stock_data(ticker, start, end)


def run_single(
    ticker: str,
    regime: str,
    seed: int,
    load_raw: Callable[..., Optional[pd.DataFrame]] = _default_loader,
    min_train_size: int = 504,
    test_size: int = 63,
    step_size: int = 63,
    epochs: int = 100,
    include_models: Optional[Sequence[str]] = None,
    meta_min_train_folds: int = 2,
) -> Optional[pd.DataFrame]:
    """One ticker, one regime, one seed. Returns a long prediction frame.

    Returns None when the window does not contain enough data for even one
    walk-forward fold, which is common for the shorter regime windows.
    """
    start, end = REGIMES[regime]
    raw = load_raw(ticker, start, end)
    if raw is None or len(raw) < min_train_size + test_size + 2:
        print(f"  {ticker}/{regime}/seed{seed}: insufficient data, skipped")
        return None

    dataset = build_dataset(raw)
    folds = WalkForwardSplitter(min_train_size, test_size, step_size).split(len(dataset))
    if not folds:
        print(f"  {ticker}/{regime}/seed{seed}: no folds, skipped")
        return None

    wanted = set(include_models or ["tree", "lstm", "transformer", "baselines", "meta"])
    blocks: List[pd.DataFrame] = []
    base_results: Dict[str, Any] = {}

    def collect(name: str, predictions: pd.DataFrame) -> None:
        block = predictions.copy()
        block["model"] = name
        block["ticker"] = ticker
        block["regime"] = regime
        block["seed"] = seed
        blocks.append(block[PREDICTION_SCHEMA])

    if "tree" in wanted:
        result = train_tree_model(dataset, folds, random_state=seed, save_model=False)
        base_results["tree"] = result
        collect(result["model_name"], result["predictions"])

    if "lstm" in wanted:
        result = train_lstm_model(
            dataset, folds, epochs=epochs, seed=seed, save_model=False, verbose=False
        )
        base_results["lstm"] = result
        collect(result["model_name"], result["predictions"])

    if "transformer" in wanted:
        result = train_transformer_model(
            dataset, folds, epochs=epochs, seed=seed, save_model=False, verbose=False
        )
        base_results["transformer"] = result
        collect(result["model_name"], result["predictions"])

    if "baselines" in wanted:
        for result in run_all_baselines(dataset, folds, seed=seed).values():
            collect(result["model_name"], result["predictions"])

    if "meta" in wanted and len(base_results) >= 2:
        meta_frame = build_meta_frame(base_results, dataset)
        for use_vix in (True, False):
            try:
                result = fit_stacked_meta(
                    meta_frame, use_vix=use_vix, min_train_folds=meta_min_train_folds
                )
                collect(result["model_name"], result["predictions"])
            except RuntimeError as exc:
                print(f"  {ticker}/{regime}/seed{seed}: meta skipped ({exc})")

    if not blocks:
        return None
    return pd.concat(blocks, ignore_index=True)


def sweep(
    tickers: Sequence[str] = tuple(DEFAULT_TICKERS),
    regimes: Sequence[str] = ("full",),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    results_dir: str = PREDICTIONS_DIR,
    overwrite: bool = False,
    **run_kwargs: Any,
) -> List[str]:
    """Run the full sweep, persisting each run as it completes.

    Existing files are skipped unless ``overwrite``, so an interrupted sweep
    resumes instead of starting over.
    """
    print(f"\n{SURVIVORSHIP_WARNING}\n")
    total = len(tickers) * len(regimes) * len(seeds)
    print(f"Sweep: {len(tickers)} tickers x {len(regimes)} regimes x "
          f"{len(seeds)} seeds = {total} runs")

    written: List[str] = []
    for i, ticker in enumerate(tickers, 1):
        for regime in regimes:
            for seed in seeds:
                path = run_path(ticker, regime, seed, results_dir)
                if os.path.exists(path) and not overwrite:
                    print(f"  [{i}/{len(tickers)}] {ticker}/{regime}/seed{seed}: cached")
                    written.append(path)
                    continue

                print(f"  [{i}/{len(tickers)}] {ticker}/{regime}/seed{seed}")
                frame = run_single(ticker, regime, seed, **run_kwargs)
                if frame is not None:
                    written.append(save_run(frame, path))

    print(f"\nPersisted {len(written)} runs to {results_dir!r}")
    if not _parquet_available():
        print(
            "  NOTE: written as CSV. `pip install pyarrow` for the intended "
            "parquet format."
        )
    return written


# ---------------------------------------------------------------------------
# Aggregation, always from disk
# ---------------------------------------------------------------------------

GROUP_KEYS = ["regime", "model", "ticker", "seed"]


def aggregate(
    results_dir: str = PREDICTIONS_DIR, runs: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Metrics for every (regime, model, ticker, seed) run, read from disk."""
    runs = load_runs(results_dir) if runs is None else runs

    rows = []
    for keys, block in runs.groupby(GROUP_KEYS, sort=True):
        predictions = (
            block[["target_date", "fold_id", "close_t", "y_true", "y_pred"]]
            .sort_values("target_date")
            .reset_index(drop=True)
        )
        predictions["fold_id"] = predictions["fold_id"].astype("int64")
        evaluation = evaluate_predictions(
            predictions, model_name=str(keys[1]), validate=False
        )
        pooled = evaluation["pooled"]
        rows.append(
            {
                **dict(zip(GROUP_KEYS, keys)),
                "n": len(predictions),
                "directional_accuracy": pooled["directional_accuracy"],
                "pt_statistic": pooled.get("pt_statistic"),
                "pt_p_value": pooled.get("pt_p_value"),
                "r2_oos": pooled["r2_oos"],
                "rmse_bps": pooled["rmse_bps"],
                "mae_bps": pooled["mae_bps"],
                "excluded_fraction": pooled["excluded_fraction"],
            }
        )

    return pd.DataFrame(rows)


def seed_variance(per_run: pd.DataFrame) -> pd.DataFrame:
    """Spread across seeds, per (regime, model, ticker).

    Single-seed neural results are not credible. If this std is comparable to
    the gap between two models, the gap is not a finding.
    """
    return (
        per_run.groupby(["regime", "model", "ticker"])
        .agg(
            n_seeds=("seed", "nunique"),
            da_mean=("directional_accuracy", "mean"),
            da_std=("directional_accuracy", "std"),
            r2_mean=("r2_oos", "mean"),
            r2_std=("r2_oos", "std"),
        )
        .reset_index()
    )


def across_tickers(per_run: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional summary per (regime, model): the headline table.

    Seeds are averaged within a ticker first, so a model run over more seeds
    does not get extra weight in the cross-sectional mean.
    """
    by_ticker = (
        per_run.groupby(["regime", "model", "ticker"])[
            ["directional_accuracy", "r2_oos", "rmse_bps"]
        ]
        .mean()
        .reset_index()
    )
    return (
        by_ticker.groupby(["regime", "model"])
        .agg(
            n_tickers=("ticker", "nunique"),
            da_mean=("directional_accuracy", "mean"),
            da_std=("directional_accuracy", "std"),
            da_frac_above_half=("directional_accuracy", lambda s: float((s > 0.5).mean())),
            r2_mean=("r2_oos", "mean"),
            r2_std=("r2_oos", "std"),
            r2_frac_positive=("r2_oos", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )


def write_aggregates(results_dir: str = PREDICTIONS_DIR, out_dir: str = RESULTS_DIR):
    """Recompute every aggregate from the persisted runs and write them out."""
    os.makedirs(out_dir, exist_ok=True)
    per_run = aggregate(results_dir)

    outputs = {
        "per_run.csv": per_run,
        "seed_variance.csv": seed_variance(per_run),
        "across_tickers.csv": across_tickers(per_run),
    }
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(out_dir, name), index=False)
        print(f"  wrote {os.path.join(out_dir, name)}  ({len(frame)} rows)")

    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sweep tickers, regimes and seeds")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument(
        "--regimes", nargs="*", default=["full"], choices=sorted(REGIMES)
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-train", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=63)
    parser.add_argument("--step-size", type=int, default=63)
    parser.add_argument("--results-dir", default=PREDICTIONS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training and just recompute aggregates from persisted runs",
    )
    args = parser.parse_args()

    if not args.aggregate_only:
        sweep(
            tickers=args.tickers,
            regimes=args.regimes,
            seeds=args.seeds,
            results_dir=args.results_dir,
            overwrite=args.overwrite,
            epochs=args.epochs,
            min_train_size=args.min_train,
            test_size=args.test_size,
            step_size=args.step_size,
        )

    write_aggregates(args.results_dir)


if __name__ == "__main__":
    main()
