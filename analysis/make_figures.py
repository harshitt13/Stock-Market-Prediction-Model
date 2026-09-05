"""Build every figure from the committed artefacts, into docs/figures/.

Reads results/ and docs/ only; retrains nothing; writes only PNGs.

Sources, by figure:
- The AAPL headline set is rebuilt from results/headline/AAPL__frozen__seed42.parquet,
  the per-fold predictions of the README section 3.1 run on docs/frozen_aapl_raw.csv
  (written by analysis/freeze_headline_predictions.py). Its fold grid is Table 1's,
  and headline_frames() asserts that the comparison table rebuilt from it equals
  docs/baseline_after_refactor.csv, which is Table 1, before anything is drawn.
- The cross-ticker figures come from the 30-ticker sweep under results/predictions/
  and results/fold_diagnostics.csv, where every ticker sits on its own fetch grid
  and no offset arises.
- The seed-variance figure comes from results/seeds/, which the seed study ran on
  the sweep's AAPL fetch cache. That grid opens about 50 trading days before
  Table 1's; grid_offset() computes the exact figure from committed files and the
  two figures that touch the sweep's AAPL run print it in their titles.

    python analysis/make_figures.py
"""

import functools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest import backtest_table  # noqa: E402
from contracts import align_predictions, common_evaluation_window, make_predictions, restrict_all  # noqa: E402
from dataset import build_dataset  # noqa: E402
from evaluate import compare_evaluations, evaluate_predictions, expanding_mean_benchmark, r2_oos  # noqa: E402
from experiments import load_runs  # noqa: E402
from figures import (  # noqa: E402
    ROLE_COLOURS, _canonical, _colour, _offscale_barh, _save, _style,
    fig_alpha_beta, fig_calibration, fig_equity_curves, fig_model_comparison,
    fig_per_fold_metric, fig_predicted_vs_realised, fig_prediction_dispersion,
    fig_walk_forward_folds,
)
from meta_ensemble import fold_respecting_intervals  # noqa: E402
from walk_forward import WalkForwardSplitter  # noqa: E402

OUT = REPO / "docs" / "figures"
FOLD_CFG = (1008, 252, 252)  # the headline configuration


def frames_from_runs(runs: pd.DataFrame) -> dict:
    """Standard result frames, one per model, from the long parquet layout."""
    out = {}
    for model, block in runs.groupby("model", sort=True):
        block = block.sort_values("target_date").reset_index(drop=True)
        out[model] = make_predictions(
            target_date=block["target_date"], fold_id=block["fold_id"].to_numpy(int),
            close_t=block["close_t"].to_numpy(float), y_true=block["y_true"].to_numpy(float),
            y_pred=block["y_pred"].to_numpy(float),
        )
    return out


FROZEN_CSV = REPO / "docs" / "frozen_aapl_raw.csv"
TABLE_1 = REPO / "docs" / "baseline_after_refactor.csv"
HEADLINE_DIR = REPO / "results" / "headline"
SWEEP_DIR = REPO / "results" / "predictions"
SWEEP_RAW = REPO / "results" / "raw"


def _dataset(csv: Path) -> tuple:
    ds = build_dataset(pd.read_csv(csv, parse_dates=["Date"]))
    folds = WalkForwardSplitter(*FOLD_CFG).split(len(ds))
    return ds, folds, {i: ds.y[np.asarray(tr)] for i, (tr, _) in enumerate(folds)}


@functools.lru_cache(maxsize=1)
def frozen_dataset() -> tuple:
    """Table 1's grid: docs/frozen_aapl_raw.csv, the headline run's input."""
    return _dataset(FROZEN_CSV)


@functools.lru_cache(maxsize=1)
def sweep_dataset() -> tuple:
    """The sweep's grid: results/raw/AAPL.csv, the fetch cache the 30-ticker
    sweep and the seed study ran on. Gitignored. Only the seed figure needs it,
    for the fold-seeded R2_OOS benchmark of runs that were made on that grid.
    """
    sweep = SWEEP_RAW / "AAPL.csv"
    if not sweep.exists():
        raise FileNotFoundError(
            f"{sweep} is absent (a gitignored fetch cache); regenerate it with "
            "`python src/fetch_universe.py`."
        )
    return _dataset(sweep)


@functools.lru_cache(maxsize=1)
def grid_offset() -> dict:
    """How far the sweep's AAPL fold grid sits from Table 1's, from committed
    files only: the sweep parquet's fold-0 opening date against the frozen
    dataset's.

    The frozen CSV is the pre-refactor pipeline's already-warmed-up frame, so
    re-engineering it costs a second warm-up (4144 -> 4094 rows), while the
    sweep cache was saved post-engineering and keeps those rows (4144 -> 4143).
    Every sweep fold boundary is therefore earlier by that many trading days.
    """
    ds, folds, _ = frozen_dataset()
    dates = pd.to_datetime(pd.Series(np.asarray(ds.target_date)))
    frozen_open = dates.iloc[folds[0][1][0]]
    sweep = load_runs(str(SWEEP_DIR))
    sweep_open = sweep.loc[(sweep["ticker"] == "AAPL") & (sweep["fold_id"] == 0), "target_date"].min()
    rows = int(((dates >= sweep_open) & (dates < frozen_open)).sum())
    return {"sweep_open": str(sweep_open)[:10], "frozen_open": str(frozen_open)[:10], "rows": rows}


def sweep_grid_note() -> str:
    o = grid_offset()
    return (f"sweep fold grid: fold 0 opens {o['sweep_open']}, {o['rows']} trading days "
            f"before Table 1's {o['frozen_open']}")


# ---------------------------------------------------------------------------
# AAPL headline set, from results/headline/AAPL__frozen__seed42.parquet (Table 1's run)
# ---------------------------------------------------------------------------


def headline_frames() -> tuple:
    """The headline run's frames, checked against its dataset and against Table 1.

    Two assertions, both on committed files. The parquet's fold-0 realised
    returns must equal the frozen dataset's, so the fold grid is the one Table 1
    was computed on; and the comparison table rebuilt here must equal
    docs/baseline_after_refactor.csv, which is Table 1, to the printed digits.
    """
    runs = load_runs(str(HEADLINE_DIR))
    runs = runs[(runs["ticker"] == "AAPL") & (runs["regime"] == "frozen")]
    frames = frames_from_runs(runs)
    ds, folds, y_train = frozen_dataset()

    tree0 = frames["Tree Ensemble"]
    f0 = tree0[tree0["fold_id"] == 0]
    np.testing.assert_allclose(f0["y_true"].to_numpy(), ds.y[folds[0][1]],
                               err_msg="headline parquet / frozen dataset drift")

    window = common_evaluation_window(frames)
    primary = restrict_all(frames, window)
    evaluations = [evaluate_predictions(p, n, y_train_by_fold=y_train) for n, p in primary.items()]
    comparison = compare_evaluations(evaluations)
    table1 = pd.read_csv(TABLE_1).set_index("Model")
    pd.testing.assert_frame_equal(comparison.loc[table1.index, table1.columns], table1,
                                  check_dtype=False, check_names=False, rtol=0, atol=1e-9)
    print(f"  AAPL headline set: {len(runs)} rows from {HEADLINE_DIR.relative_to(REPO)}, "
          f"fold 0 opens {str(f0['target_date'].min())[:10]}; comparison table equals "
          f"{TABLE_1.relative_to(REPO)} (Table 1)")
    return ds, folds, y_train, primary, window, evaluations, comparison


def aapl_figures() -> list:
    ds, folds, y_train, primary, window, evaluations, comparison = headline_frames()
    subtitle = f"AAPL, {window.describe()}, 12 folds, seed 42, frozen CSV (Table 1 grid)"
    aligned = align_predictions(primary)
    economics = backtest_table(primary, benchmark_predictions=primary["Zero return"])
    meta_name = next(n for n in primary if "+VIX" in n)
    calibration = fold_respecting_intervals(primary[meta_name])
    models = ["Tree Ensemble", "BiLSTM", "Transformer", "Hybrid meta (+VIX)", "Hybrid meta (no VIX)"]

    return [
        fig_model_comparison(comparison, subtitle, OUT / "aapl_model_comparison.png"),
        fig_predicted_vs_realised(aligned, models, subtitle, OUT / "aapl_predicted_vs_realised.png"),
        fig_prediction_dispersion(aligned, list(primary), subtitle, OUT / "aapl_prediction_dispersion.png"),
        fig_equity_curves(primary, subtitle, OUT / "aapl_equity_curves.png"),
        fig_alpha_beta(economics, subtitle, OUT / "aapl_alpha_beta.png"),
        fig_per_fold_metric(evaluations, "r2_oos", subtitle, OUT / "aapl_r2_oos_by_fold.png", "R²_OOS"),
        fig_per_fold_metric(evaluations, "da_minus_majority", subtitle,
                            OUT / "aapl_da_minus_majority_by_fold.png", "DA minus majority (pp)"),
        fig_calibration(calibration, subtitle, OUT / "aapl_calibration.png"),
        fig_walk_forward_folds(ds.feature_date, ds.close_t, folds, subtitle, OUT / "aapl_walk_forward_folds.png"),
    ]


# ---------------------------------------------------------------------------
# Cross-run figures
# ---------------------------------------------------------------------------


def fig_before_after() -> Path:
    """R² on the price target vs R²_OOS on the return target, identical data."""
    _style()
    before = pd.read_csv(REPO / "docs/baseline_before_refactor.csv").set_index("Model")
    after = pd.read_csv(REPO / "docs/baseline_after_refactor.csv").set_index("Model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    b = pd.to_numeric(before["R2"], errors="coerce").dropna()
    ax.barh(b.index, b.values, color=[ROLE_COLOURS["baseline"] if "Naive" in n or "ARIMA" in n else ROLE_COLOURS["base"] for n in b.index])
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("R² on next-day close price")
    ax.set_title("BEFORE: price target\nthe naive baseline scores 0.9992 — the metric measures the price series")
    for i, v in enumerate(b.values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)

    ax = axes[1]
    a = pd.to_numeric(after["R2_OOS"], errors="coerce").dropna()
    a = a.loc[_canonical(a.index)]
    _offscale_barh(ax, list(a.index), a.values, "{:+.4f}")
    ax.set_xlabel("out-of-sample R² on next-day log return")
    ax.set_title("AFTER: return target, same frozen data, same 12 folds\nno model is above zero")

    fig.suptitle("The same models, before and after the evaluation refactor — AAPL 2010-2026", fontsize=11)
    fig.tight_layout()
    return _save(fig, OUT / "before_after.png")


def fig_cross_ticker() -> Path:
    """Tree Ensemble across 30 tickers: is AAPL's null universal?"""
    _style()
    per = pd.read_csv(REPO / "results/per_ticker_model.csv")
    tree = per[per["model"] == "Tree Ensemble"].set_index("ticker").sort_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [("DA_minus_majority", "DA minus majority-class rate (pp)", None),
              ("R2_OOS", "out-of-sample R²", None),
              ("t_alpha", "t-statistic on alpha vs buy-and-hold", 1.96)]
    rng = np.random.default_rng(0)
    for ax, (col, label, band) in zip(axes, panels):
        v = tree[col].to_numpy(float)
        jitter = rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(v, jitter, s=22, color=ROLE_COLOURS["base"], alpha=0.75, zorder=3)
        if "AAPL" in tree.index:
            ax.scatter(tree.loc["AAPL", col], jitter[list(tree.index).index("AAPL")], s=80,
                       facecolor="none", edgecolor="#d62728", lw=1.6, zorder=4, label="AAPL")
        ax.axvline(0, color="black", lw=1)
        if band:
            ax.axvspan(-band, band, color="grey", alpha=0.12, label=f"|t| < {band}")
        ax.set_yticks([])
        ax.set_xlabel(label)
        sd = v.std(ddof=1)  # matches the README tables
        if band:
            pos = int((v > band).sum())
            neg = int((v < -band).sum())
            head = f"{pos}/{len(v)} with t > +{band};  {neg}/{len(v)} with t < -{band}"
        else:
            head = f"{int((v > 0).sum())}/{len(v)} tickers above zero"
        ax.set_title(f"{head}\nmean {v.mean():+.3f}, sd {sd:.3f}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Tree Ensemble across 30 US large caps, 12 folds each, seed 42\n"
                 f"AAPL ringed is the sweep's own run, not Table 1's ({sweep_grid_note()})",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, OUT / "cross_ticker_null.png")


def fig_seed_variance() -> Path | None:
    """Five seeds per neural architecture on AAPL: seed noise vs architecture gap.

    The seed study ran on the sweep's fetch cache, so its benchmark needs that
    grid's training returns, and its title carries the offset from Table 1.
    """
    if not (SWEEP_RAW / "AAPL.csv").exists():
        print("  seed_variance skipped: results/raw/AAPL.csv absent (python src/fetch_universe.py)")
        return None
    _style()
    runs = load_runs(str(REPO / "results/seeds"))
    _, _, y_train = sweep_dataset()
    rows = []
    for (model, seed), block in runs.groupby(["model", "seed"]):
        block = block.sort_values("target_date").reset_index(drop=True)
        p = make_predictions(block["target_date"], block["fold_id"].to_numpy(int),
                             block["close_t"].to_numpy(float), block["y_true"].to_numpy(float),
                             block["y_pred"].to_numpy(float))
        ev = evaluate_predictions(p, model, validate=False, y_train_by_fold=y_train)["pooled"]
        rows.append({"model": model, "seed": int(seed), "R2_OOS": ev["r2_oos"],
                     "DA": ev["directional_accuracy"] * 100})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (col, label) in zip(axes, [("R2_OOS", "out-of-sample R²"), ("DA", "directional accuracy (%)")]):
        for i, model in enumerate(sorted(d["model"].unique())):
            v = d.loc[d["model"] == model, col].to_numpy()
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(v)),
                       v, s=40, color=_colour(model), zorder=3, label=model)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black", lw=1.5)
        if col == "R2_OOS":
            ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(d["model"].nunique()))
        ax.set_xticklabels(sorted(d["model"].unique()))
        ax.set_ylabel(label)
        ax.set_title(f"{label}: 5 seeds each (bar = mean)", fontsize=9)
    fig.suptitle("Seed variance vs architecture difference — AAPL, 12 folds, seeds 0-4\n"
                 f"run on the sweep's fetch cache, not the frozen CSV ({sweep_grid_note()})",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, OUT / "seed_variance.png")


def fig_shift_vs_r2() -> Path | None:
    """Per-fold R²_OOS against the fold's train/test KS statistic, all tickers."""
    raw_dir = REPO / "results/raw"
    if not raw_dir.exists():
        print("  shift_vs_r2 skipped: results/raw/ absent (python src/fetch_universe.py)")
        return None
    _style()
    ks = pd.read_csv(REPO / "results/fold_diagnostics.csv")
    runs = load_runs(str(REPO / "results/predictions"))
    tree = runs[runs["model"] == "Tree Ensemble"]
    rows = []
    for ticker, block in tree.groupby("ticker"):
        raw = pd.read_csv(raw_dir / f"{ticker}.csv", parse_dates=["Date"])
        ds = build_dataset(raw)
        folds = WalkForwardSplitter(*FOLD_CFG).split(len(ds))
        y_train = {i: ds.y[np.asarray(tr)] for i, (tr, _) in enumerate(folds)}
        for fold_id, fb in block.groupby("fold_id"):
            yt, yp = fb["y_true"].to_numpy(float), fb["y_pred"].to_numpy(float)
            rows.append({"ticker": ticker, "fold_id": int(fold_id),
                         "r2_oos": r2_oos(yt, yp, expanding_mean_benchmark(yt, y_train.get(int(fold_id))))})
    d = pd.DataFrame(rows).merge(ks, on=["ticker", "fold_id"])
    x, y = d["ks_stat"].to_numpy(), d["r2_oos"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    r2 = 1.0 - resid.var() / y.var()  # the ceiling: how much of R2_OOS shift explains
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sig = d["ks_reject_5pct"].astype(bool).to_numpy()
    ax.scatter(x[~sig], y[~sig], s=14, alpha=0.6, color=ROLE_COLOURS["base"], label="KS not significant")
    ax.scatter(x[sig], y[sig], s=14, alpha=0.8, color="#d62728", label="KS significant (p<0.05)")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, intercept + slope * xs, color="black", lw=1.2, label=f"OLS slope {slope:.2f}, R² = {r2:.3f}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("KS statistic, train vs test return distribution")
    ax.set_ylabel("Tree Ensemble R²_OOS for that fold")
    ax.set_title(f"Distribution shift predicts magnitude error — {len(d)} fold-ticker pairs\n"
                 f"shifted folds mean {y[sig].mean():+.3f}, stable folds mean {y[~sig].mean():+.3f};  "
                 f"KS explains R² = {r2:.3f} of R2_OOS variance", fontsize=9.5)
    ax.legend(fontsize=7)
    return _save(fig, OUT / "shift_vs_r2_oos.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = aapl_figures()
    written += [fig_before_after(), fig_cross_ticker()]
    written += [p for p in (fig_seed_variance(), fig_shift_vs_r2()) if p]
    print(f"\n{len(written)} figures written to {OUT.relative_to(REPO)}/")
    for p in written:
        print(f"  {p.name:<40} {p.stat().st_size / 1024:6.0f} KB")


if __name__ == "__main__":
    main()
