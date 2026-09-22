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
from linear_models import run_linear_comparators
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

    wanted = set(include_models or ["tree", "lstm", "transformer", "baselines", "linear", "meta"])
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

    if "linear" in wanted:
        for result in run_linear_comparators(dataset, folds, seed=seed).values():
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


def training_returns_for_run(dataset, predictions: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Each fold's training returns, recovered from the run's own test windows.

    An expanding-window fold trains on every dataset row before its first
    test day, so fold k's training history is the dataset's returns up to the
    row before its earliest ``target_date``. This needs no record of the
    min_train/test/step configuration.
    """
    target_dates = pd.to_datetime(pd.Series(np.asarray(dataset.target_date)))
    out: Dict[int, np.ndarray] = {}
    for fold_id, block in predictions.groupby("fold_id", sort=True):
        first = pd.Timestamp(block["target_date"].min())
        n_before = int((target_dates < first).sum())
        if n_before == 0:
            raise ValueError(f"fold {int(fold_id)} has no dataset rows before {first.date()}")
        out[int(fold_id)] = dataset.y[:n_before]
    return out


def aggregate(
    results_dir: str = PREDICTIONS_DIR,
    runs: Optional[pd.DataFrame] = None,
    *,
    load_raw: Callable[..., Optional[pd.DataFrame]],
) -> pd.DataFrame:
    """Metrics for every (regime, model, ticker, seed) run, read from disk.

    ``load_raw`` is required. It serves each ticker's raw frame so that every
    fold's training returns can be rebuilt and the R2_OOS benchmark seeded.
    An earlier version of this function evaluated without them; the
    zero-return baseline then scored +0.035 on all thirty sweep tickers, and
    the flag that recorded the omission was never read on this path. The
    evaluator now raises without the seeds, and this function rebuilds them
    from the run's own fold windows.
    """
    runs = load_runs(results_dir) if runs is None else runs

    datasets: Dict[Tuple[str, str], Any] = {}

    def dataset_for(ticker: str, regime: str):
        key = (ticker, regime)
        if key not in datasets:
            start, end = REGIMES[regime]
            raw = load_raw(ticker, start, end)
            if raw is None:
                raise FileNotFoundError(f"no raw data for {ticker}; cannot seed the R2_OOS benchmark")
            raw = raw[raw["Date"] >= pd.Timestamp(start)]
            if end is not None:
                raw = raw[raw["Date"] < pd.Timestamp(end)]
            datasets[key] = build_dataset(raw.reset_index(drop=True))
        return datasets[key]

    rows = []
    for keys, block in runs.groupby(GROUP_KEYS, sort=True):
        predictions = (
            block[["target_date", "fold_id", "close_t", "y_true", "y_pred"]]
            .sort_values("target_date")
            .reset_index(drop=True)
        )
        predictions["fold_id"] = predictions["fold_id"].astype("int64")
        regime, model, ticker = str(keys[0]), str(keys[1]), str(keys[2])
        evaluation = evaluate_predictions(
            predictions, model_name=model, validate=False,
            y_train_by_fold=training_returns_for_run(dataset_for(ticker, regime), predictions),
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


def write_aggregates(
    results_dir: str = PREDICTIONS_DIR,
    out_dir: str = RESULTS_DIR,
    *,
    load_raw: Callable[..., Optional[pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """Recompute every aggregate from the persisted runs and write them out."""
    os.makedirs(out_dir, exist_ok=True)
    per_run = aggregate(results_dir, load_raw=load_raw)

    outputs = {
        "per_run.csv": per_run,
        "seed_variance.csv": seed_variance(per_run),
        "across_tickers.csv": across_tickers(per_run),
    }
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(out_dir, name), index=False)
        print(f"  wrote {os.path.join(out_dir, name)}  ({len(frame)} rows)")

    return outputs




# ---------------------------------------------------------------------------
# Parallel sweep over a cached universe
# ---------------------------------------------------------------------------

#: Tickers that get every model. The rest get the cheap ones only, because the
#: neural models cost roughly twenty times the tree and the question the sweep
#: answers -- is the null universal -- does not need them on all thirty.
FULL_MODEL_TICKERS = ("AAPL", "JPM", "JNJ", "XOM", "WMT")

CHEAP_MODELS = ("tree", "baselines", "linear")
ALL_MODELS = ("tree", "lstm", "transformer", "baselines", "linear", "meta")


def cached_loader(raw_dir: str) -> Callable[..., Optional[pd.DataFrame]]:
    """A load_raw callable that reads the cached CSV for a ticker."""

    def load(ticker: str, start: str, end: Optional[str]) -> Optional[pd.DataFrame]:
        path = os.path.join(raw_dir, f"{ticker.replace('/', '-')}.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path, parse_dates=["Date"])

    return load


def fold_return_diagnostics(
    dataset, folds, ticker: str, regime: str = "full"
) -> pd.DataFrame:
    """Per-fold train/test return-distribution comparison.

    A walk-forward fold trains on one return distribution and is scored on the
    next one. The KS test says how often those differ enough to notice, which
    bounds how much any model could have learned that still applies.
    """
    from scipy import stats

    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        y_train = dataset.y[np.asarray(train_idx)]
        y_test = dataset.y[np.asarray(test_idx)]
        ks = stats.ks_2samp(y_train, y_test)
        rows.append(
            {
                "ticker": ticker,
                "regime": regime,
                "fold_id": fold_id,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "train_mean_bps": float(y_train.mean()) * 1e4,
                "test_mean_bps": float(y_test.mean()) * 1e4,
                "train_std_bps": float(y_train.std()) * 1e4,
                "test_std_bps": float(y_test.std()) * 1e4,
                "ks_stat": float(ks.statistic),
                "ks_p": float(ks.pvalue),
                "ks_reject_5pct": bool(ks.pvalue < 0.05),
                "test_outside_train_range": float(
                    np.mean((y_test < y_train.min()) | (y_test > y_train.max()))
                ),
            }
        )
    return pd.DataFrame(rows)


def run_ticker_task(task: dict) -> dict:
    """Worker entry point. Picklable, returns paths and small frames only.

    Large prediction frames are written to disk here rather than shipped back
    through the pool, so inter-process traffic stays small.
    """
    import time

    import torch

    torch.set_num_threads(task.get("torch_threads", 2))

    ticker = task["ticker"]
    started = time.time()

    try:
        loader = cached_loader(task["raw_dir"])
        raw = loader(ticker, None, None)
        if raw is None:
            return {"ticker": ticker, "ok": False, "error": "no cached CSV"}

        dataset = build_dataset(raw)
        folds = WalkForwardSplitter(
            task["min_train"], task["test_size"], task["step_size"]
        ).split(len(dataset))
        if not folds:
            return {"ticker": ticker, "ok": False, "error": "no folds"}

        frame = run_single(
            ticker,
            task["regime"],
            task["seed"],
            load_raw=loader,
            min_train_size=task["min_train"],
            test_size=task["test_size"],
            step_size=task["step_size"],
            epochs=task["epochs"],
            include_models=task["models"],
            meta_min_train_folds=task.get("meta_min_train_folds", 2),
        )
        if frame is None:
            return {"ticker": ticker, "ok": False, "error": "no predictions"}

        path = save_run(frame, run_path(ticker, task["regime"], task["seed"],
                                        task["results_dir"]))
        diagnostics = fold_return_diagnostics(dataset, folds, ticker, task["regime"])

        return {
            "ticker": ticker,
            "ok": True,
            "path": path,
            "n_rows": len(frame),
            "n_models": frame["model"].nunique(),
            "models": task["models"],
            "diagnostics": diagnostics,
            "seconds": time.time() - started,
        }
    except Exception as exc:  # pragma: no cover - worker isolation
        import traceback

        return {
            "ticker": ticker,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
            "seconds": time.time() - started,
        }


def sweep_parallel(
    tickers: Sequence[str],
    full_model_tickers: Sequence[str] = FULL_MODEL_TICKERS,
    raw_dir: str = os.path.join("results", "raw"),
    results_dir: str = PREDICTIONS_DIR,
    regime: str = "full",
    seed: int = 42,
    min_train: int = 1008,
    test_size: int = 252,
    step_size: int = 252,
    epochs: int = 100,
    max_workers: Optional[int] = None,
    torch_threads: int = 2,
) -> Dict[str, Any]:
    """Run the sweep across tickers in a process pool.

    Heavy tasks are submitted first so the long neural runs start immediately
    and the cheap ones fill in around them, rather than the pool draining with
    one five-model ticker still going.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if max_workers is None:
        max_workers = max(1, min(len(tickers), (os.cpu_count() or 4) // torch_threads))

    full = [t for t in tickers if t in set(full_model_tickers)]
    cheap = [t for t in tickers if t not in set(full_model_tickers)]

    def make(ticker, models):
        return {
            "ticker": ticker, "regime": regime, "seed": seed, "models": list(models),
            "raw_dir": raw_dir, "results_dir": results_dir, "min_train": min_train,
            "test_size": test_size, "step_size": step_size, "epochs": epochs,
            "torch_threads": torch_threads,
        }

    tasks = [make(t, ALL_MODELS) for t in full] + [make(t, CHEAP_MODELS) for t in cheap]

    print(f"\n{SURVIVORSHIP_WARNING}\n")
    print(f"Sweep: {len(tasks)} tasks "
          f"({len(full)} all-model, {len(cheap)} cheap-model) "
          f"on {max_workers} workers x {torch_threads} torch threads "
          f"({os.cpu_count()} logical cores)")
    print(f"Folds: min_train={min_train} test={test_size} step={step_size}, "
          f"seed={seed}, epochs={epochs}")

    started = time.time()
    results, diagnostics = [], []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_ticker_task, t): t["ticker"] for t in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            outcome = future.result()
            results.append(outcome)
            if outcome.get("ok"):
                diagnostics.append(outcome.pop("diagnostics"))
                print(f"  [{done:>2}/{len(tasks)}] {outcome['ticker']:<6} ok  "
                      f"{outcome['n_rows']:>6} rows, {outcome['n_models']} models, "
                      f"{outcome['seconds'] / 60:.1f} min")
            else:
                print(f"  [{done:>2}/{len(tasks)}] {outcome['ticker']:<6} FAILED: "
                      f"{outcome.get('error')}")

    wall = time.time() - started
    cpu_minutes = sum(r.get("seconds", 0) for r in results) / 60
    ok = [r for r in results if r.get("ok")]

    print(f"\nWall clock : {wall / 60:.1f} min on {max_workers} workers")
    print(f"CPU time   : {cpu_minutes:.1f} min  (speed-up {cpu_minutes / (wall / 60):.1f}x)")
    print(f"Succeeded  : {len(ok)}/{len(tasks)}")

    fold_diagnostics = (
        pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    )
    if not fold_diagnostics.empty:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fold_diagnostics.to_csv(os.path.join(RESULTS_DIR, "fold_diagnostics.csv"),
                                index=False)

    return {
        "results": results,
        "fold_diagnostics": fold_diagnostics,
        "wall_minutes": wall / 60,
        "cpu_minutes": cpu_minutes,
        "max_workers": max_workers,
    }


def main() -> None:
    """CLI: run a sweep (serial or --parallel) and recompute aggregates."""
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
    parser.add_argument("--raw-dir", default=os.path.join("results", "raw"),
                        help="Cached raw CSVs; needed to seed the R2_OOS benchmark when aggregating")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tickers in a process pool, reading the cache from "
             "src/fetch_universe.py. Uses --workers and --torch-threads.",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--full-models",
        nargs="*",
        default=list(FULL_MODEL_TICKERS),
        help="Tickers that get every model; the rest get tree + baselines.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training and just recompute aggregates from persisted runs",
    )
    args = parser.parse_args()

    if args.parallel and not args.aggregate_only:
        sweep_parallel(
            tickers=args.tickers,
            full_model_tickers=args.full_models,
            regime=args.regimes[0],
            seed=args.seeds[0],
            min_train=args.min_train,
            test_size=args.test_size,
            step_size=args.step_size,
            epochs=args.epochs,
            max_workers=args.workers,
            torch_threads=args.torch_threads,
            results_dir=args.results_dir,
            raw_dir=args.raw_dir,
        )
    elif not args.aggregate_only:
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

    write_aggregates(args.results_dir, load_raw=cached_loader(args.raw_dir))


if __name__ == "__main__":
    main()
