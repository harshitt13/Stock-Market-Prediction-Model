"""Figures. Read-only consumers of pipeline results.

Nothing in this module computes a metric. Every function takes tables the
pipeline has already produced -- the comparison table, the aligned predictions,
the economics table, the per-fold evaluations -- and draws them. Where a figure
needs a derived series (an equity curve from daily returns, say) it calls the
same evaluation function the pipeline used, on the same inputs, so the picture
cannot disagree with the table.

Two entry points:

- :func:`render_run_figures` is called by ``main.run_pipeline`` at the end of
  every run and writes the per-run set to ``images/``.
- ``analysis/make_figures.py`` rebuilds the same figures, plus the cross-run
  ones (30 tickers, five seeds, before/after), from the committed artefacts
  under ``results/`` and ``docs/``, and writes them to ``docs/figures/``.

matplotlib only. Every figure states on its face what data it shows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest import DEFAULT_COST_BPS, backtest  # noqa: E402

DPI = 150

# One colour per model *role*, so a reader can tell a baseline from a model at
# a glance without a legend for every figure.
ROLE_COLOURS = {
    "base": "#1f77b4",       # tree / LSTM / transformer
    "linear": "#17becf",     # ridge / logistic comparators
    "meta": "#9467bd",       # hybrid meta
    "baseline": "#7f7f7f",   # zero, historical mean, AR, ARIMA, random
    "benchmark": "#2ca02c",  # buy and hold
    "realised": "#d62728",
}
BASELINE_NAMES = {
    "Zero return", "Historical mean", "AR(1) returns",
    "ARIMA(5, 0, 0) returns", "Random sign",
}


def _role(name: str) -> str:
    if name == "Buy and hold":
        return "benchmark"
    if name in BASELINE_NAMES:
        return "baseline"
    if "Hybrid meta" in name:
        return "meta"
    if name.startswith(("Ridge", "Logistic")):
        return "linear"
    return "base"


def _colour(name: str) -> str:
    return ROLE_COLOURS[_role(name)]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8,
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, out: str | os.PathLike) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _numeric(series: pd.Series) -> pd.Series:
    """Tables carry 'n/a' strings in some cells; plot only the numbers."""
    return pd.to_numeric(series, errors="coerce")


def _role_legend(ax: plt.Axes, roles: Iterable[str]) -> None:
    from matplotlib.patches import Patch

    labels = {"base": "base model", "linear": "linear comparator", "meta": "hybrid meta", "baseline": "baseline",
              "benchmark": "buy and hold"}
    seen = [r for r in ("base", "linear", "meta", "baseline", "benchmark") if r in set(roles)]
    ax.legend(handles=[Patch(color=ROLE_COLOURS[r], label=labels[r]) for r in seen],
              loc="best")


#: Rows are drawn models-first, baselines last, whatever order the input has.
CANONICAL_ORDER = [
    "Tree Ensemble", "BiLSTM", "Transformer",
    "Ridge (returns)", "Logistic (direction)",
    "Hybrid meta (+VIX)", "Hybrid meta (no VIX)",
    "Zero return", "Historical mean", "AR(1) returns", "ARIMA(5, 0, 0) returns",
    "Random sign", "Buy and hold",
]

#: Rows allowed to fall off the axis. Random sign is a sanity check whose
#: R2_OOS is ~-0.9 by construction; letting it set the scale squashes every
#: real model into an unreadable sliver at zero.
OFF_SCALE_OK = {"Random sign"}


def _canonical(names: Iterable[str]) -> List[str]:
    names = list(names)
    rank = {n: i for i, n in enumerate(CANONICAL_ORDER)}
    return sorted(names, key=lambda n: (rank.get(n, len(rank)), n))


def _offscale_barh(ax: plt.Axes, names: Sequence[str], values, fmt: str,
                   exclude: Iterable[str] = OFF_SCALE_OK, margin: float = 0.45) -> None:
    """Horizontal bars whose axis is set by the rows that matter.

    Rows in ``exclude`` that fall outside the resulting range are drawn to the
    edge, hatched, and annotated with their true value and "(off scale)", so
    nothing is hidden but nothing is squashed either.
    """
    vals = np.asarray(_numeric(pd.Series(list(values))), dtype=float)
    excluded = set(exclude)
    ref = np.array([v for n, v in zip(names, vals) if n not in excluded and np.isfinite(v)])
    if ref.size == 0:
        ref = vals[np.isfinite(vals)]
    span = float(np.max(np.abs(ref))) if ref.size else 1.0
    span = span if span > 0 else 1.0
    if (ref < 0).any() and not (ref > 0).any():
        lo, hi = -span, span * 0.15
    elif (ref > 0).any() and not (ref < 0).any():
        lo, hi = -span * 0.15, span
    else:
        lo, hi = -span, span
    # Pad only where annotation text will sit: past the bar ends. A one-signed
    # panel gets a token margin on its empty side rather than a dead band.
    pad = (hi - lo) * margin
    pad_lo = pad if (ref < 0).any() else (hi - lo) * 0.04
    pad_hi = pad if (ref > 0).any() else (hi - lo) * 0.04

    ys = np.arange(len(names))
    for y, n, v in zip(ys, names, vals):
        if not np.isfinite(v):
            ax.text(0, y, "  n/a", va="center", fontsize=7, color="grey")
            continue
        off = v < lo or v > hi
        drawn = float(np.clip(v, lo, hi))
        colour = _colour(n)
        ax.barh(y, drawn, color=colour, hatch="//" if off else None,
                edgecolor="white" if off else colour, alpha=0.55 if off else 1.0)
        label = fmt.format(v) + ("  (off scale)" if off else "")
        # Off-scale bars are annotated INSIDE the hatched bar, reading toward
        # zero, so the text cannot run into the row labels.
        if off:
            ax.text(drawn, y, f" {label} ", va="center", fontsize=7, color="black",
                    ha="left" if v < 0 else "right")
        else:
            ax.text(drawn, y, f" {label} ", va="center", fontsize=7,
                    ha="left" if v >= 0 else "right")
    ax.set_yticks(ys)
    ax.set_yticklabels(list(names))
    ax.set_xlim(lo - pad_lo, hi + pad_hi)
    ax.axvline(0, color="black", lw=1)
    ax.invert_yaxis()


# ---------------------------------------------------------------------------
# Per-run figures
# ---------------------------------------------------------------------------


def fig_model_comparison(comparison: pd.DataFrame, subtitle: str, out) -> Path:
    """Directional accuracy against the majority class, and R2_OOS, per model.

    Both panels have a zero line: above it a model is doing something, below it
    it is worse than the trivial strategy the axis is measured against. Rows are
    models first, baselines last, regardless of the table's order.
    """
    _style()
    df = comparison.loc[_canonical(comparison.index)]
    names = list(df.index)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    _offscale_barh(axes[0], names, df["DA - majority"], "{:+.2f}")
    axes[0].set_xlabel("directional accuracy minus majority-class rate (pp)")
    axes[0].set_title("Direction: does the model beat 'always predict the common class'?")

    _offscale_barh(axes[1], names, df["R2_OOS"], "{:+.4f}")
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("out-of-sample R\u00b2 (Campbell-Thompson)")
    axes[1].set_title("Magnitude: does the model beat the historical mean?")

    from matplotlib.patches import Patch
    roles = {_role(n) for n in names}
    labels = {"base": "base model", "linear": "linear comparator", "meta": "hybrid meta", "baseline": "baseline"}
    fig.legend(handles=[Patch(color=ROLE_COLOURS[r], label=labels[r])
                        for r in ("base", "linear", "meta", "baseline") if r in roles],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Model comparison \u2014 {subtitle}", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out)


def fig_predicted_vs_realised(aligned: pd.DataFrame, models: Sequence[str],
                              subtitle: str, out) -> Path:
    """Scatter of predicted against realised next-day return, one panel per model.

    A forecasting model shows a diagonal. A model emitting a near-constant
    shows a horizontal band. The correlation is printed on each panel.
    """
    _style()
    models = [m for m in models if f"y_pred_{m}" in aligned.columns]
    n = len(models)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)

    y = aligned["y_true"].to_numpy(float) * 1e4
    lim = np.nanpercentile(np.abs(y), 99)

    for ax, m in zip(axes.flat, models):
        p = aligned[f"y_pred_{m}"].to_numpy(float) * 1e4
        ok = np.isfinite(p) & np.isfinite(y)
        corr = np.corrcoef(p[ok], y[ok])[0, 1] if ok.sum() > 2 and p[ok].std() > 0 else np.nan
        ax.scatter(y[ok], p[ok], s=4, alpha=0.35, color=_colour(m))
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.8, color="grey", label="perfect")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{m}\ncorr = {corr:+.3f}   sd(pred)/sd(true) = "
                     f"{np.nanstd(p[ok]) / np.nanstd(y[ok]):.2f}", fontsize=8.5)
        ax.set_xlabel("realised return (bps)")
        ax.set_ylabel("predicted return (bps)")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Predicted vs realised next-day return — {subtitle}", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def fig_prediction_dispersion(aligned: pd.DataFrame, models: Sequence[str],
                              subtitle: str, out) -> Path:
    """Distribution of predicted returns per model, against the realised one.

    The realised distribution is the wide one on the left. Every model that
    is picking a level rather than forecasting collapses to a narrow spike.
    """
    _style()
    models = _canonical(m for m in models if f"y_pred_{m}" in aligned.columns)
    data = [aligned["y_true"].to_numpy(float) * 1e4] + [
        aligned[f"y_pred_{m}"].to_numpy(float) * 1e4 for m in models
    ]
    labels = ["realised"] + list(models)
    colours = [ROLE_COLOURS["realised"]] + [_colour(m) for m in models]

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels) + 3), 4.5))
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.85)
    for body, c in zip(parts["bodies"], colours):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("next-day return (bps)")
    sds = [np.nanstd(d) for d in data]
    ax.set_title(f"Dispersion of predictions — {subtitle}\n"
                 f"sd realised = {sds[0]:.0f} bps; model sds = "
                 + ", ".join(f"{s:.0f}" for s in sds[1:]) + " bps", fontsize=9.5)
    return _save(fig, out)


def fig_equity_curves(predictions: Dict[str, pd.DataFrame], subtitle: str, out,
                      cost_bps: float = DEFAULT_COST_BPS) -> Path:
    """Net-of-cost equity curves, long/flat on the predicted sign, vs buy-and-hold.

    Uses :func:`backtest.backtest` on each model's frame, exactly as the
    economics table does, so the curves and the table agree.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))

    hold_plotted = False
    for name in _canonical(predictions):
        frame = predictions[name]
        bt = backtest(frame, cost_bps=cost_bps, validate=False)
        daily = bt["daily"]
        equity = np.cumprod(1.0 + daily["net_return"].to_numpy(float))
        ax.plot(daily["target_date"], equity, lw=1.3 if _role(name) != "baseline" else 0.9,
                alpha=1.0 if _role(name) != "baseline" else 0.6,
                color=_colour(name), label=f"{name}  (Sharpe {bt['net']['sharpe']:.2f})"
                if np.isfinite(bt["net"]["sharpe"]) else f"{name}  (never trades)")
        if not hold_plotted:
            hold = np.cumprod(1.0 + daily["asset_return"].to_numpy(float))
            ax.plot(daily["target_date"], hold, lw=2.2, color=ROLE_COLOURS["benchmark"],
                    label=f"Buy and hold  (Sharpe {bt['buy_and_hold']['sharpe']:.2f})")
            hold_plotted = True

    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log scale)")
    ax.set_title(f"Equity curves, long/flat on predicted sign, {cost_bps:.1f} bps round-trip — "
                 f"{subtitle}", fontsize=9.5)
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    fig.autofmt_xdate()
    return _save(fig, out)


def fig_alpha_beta(economics: pd.DataFrame, subtitle: str, out) -> Path:
    """Annualised alpha with its t-statistic, and beta, per strategy.

    Raw Sharpe cannot separate skill from market exposure; alpha and beta can.
    The t-statistic is printed at the right edge, red where |t| > 1.96, so a
    reader can see at a glance that none is.
    """
    _style()
    df = economics.loc[_canonical(economics.index)]
    names = list(df.index)
    t = _numeric(df["t(alpha)"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    _offscale_barh(ax, names, df["Alpha ann."], "{:+.4f}", exclude=set())
    right = ax.get_xlim()[1]
    for y, tv in enumerate(t):
        if np.isfinite(tv):
            ax.text(right, y, f"t={tv:+.2f}", va="center", ha="right", fontsize=7,
                    color="#d62728" if abs(tv) > 1.96 else "dimgrey")
    ax.set_xlabel("annualised alpha vs buy-and-hold")
    ax.set_title("Alpha  (t-statistic at right; red if |t| > 1.96)")

    ax = axes[1]
    _offscale_barh(ax, names, df["Beta"], "{:.3f}", exclude=set())
    ax.axvline(1, color=ROLE_COLOURS["benchmark"], lw=1.2, ls="--")
    ax.text(1, -0.7, "beta = 1: is the market", ha="center", va="bottom", fontsize=7,
            color=ROLE_COLOURS["benchmark"])
    ax.set_yticklabels([])
    ax.set_xlabel("beta vs buy-and-hold")
    ax.set_title("Beta")

    fig.suptitle(f"Market-adjusted economics \u2014 {subtitle}", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def fig_per_fold_metric(evaluations: Sequence[Dict[str, Any]], metric: str,
                        subtitle: str, out, label: Optional[str] = None) -> Path:
    """One line per model across walk-forward folds, for a per-fold metric."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for ev in evaluations:
        pf = ev["per_fold"]
        if metric not in pf.columns:
            continue
        name = ev["model"]
        ax.plot(pf["fold_id"], pf[metric], marker="o", ms=3, lw=1.2,
                alpha=0.55 if _role(name) == "baseline" else 1.0,
                color=_colour(name), label=name)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("walk-forward fold")
    ax.set_ylabel(label or metric)
    ax.set_title(f"{label or metric} by fold — {subtitle}")
    ax.legend(ncol=3, fontsize=7)
    return _save(fig, out)


def fig_calibration(calibration: Dict[str, Any], subtitle: str, out) -> Path:
    """Empirical coverage of the 95% interval per fold, against nominal."""
    _style()
    per_fold = calibration.get("per_fold")
    if per_fold is None or len(per_fold) == 0:
        raise ValueError("no per-fold calibration to plot")
    level = calibration.get("level", 0.95)

    fig, ax = plt.subplots(figsize=(8, 4))
    cov = per_fold["empirical_coverage"].to_numpy(float)
    ax.bar(per_fold["fold_id"].astype(str), cov,
           color=np.where(cov < level, "#d62728", ROLE_COLOURS["meta"]))
    ax.axhline(level, color="black", lw=1.2, ls="--", label=f"nominal {level:.0%}")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("walk-forward fold")
    ax.set_ylabel("empirical coverage")
    overall = calibration.get("overall") or {}
    ax.set_title(f"Interval calibration — {subtitle}\n"
                 f"overall {overall.get('empirical_coverage', float('nan')):.1%} "
                 f"({overall.get('verdict', '')})", fontsize=9.5)
    ax.legend()
    return _save(fig, out)


def fig_walk_forward_folds(dates, close, folds, subtitle: str, out) -> Path:
    """Price history with each fold's train/test span shaded."""
    _style()
    dates = pd.to_datetime(np.asarray(dates))
    close = np.asarray(close, float)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, close, color="black", lw=0.8, label="close")
    for i, (tr, te) in enumerate(folds):
        tr, te = np.asarray(tr), np.asarray(te)
        ax.axvspan(dates[te[0]], dates[te[-1]], color="#ff7f0e", alpha=0.18,
                   label="test fold" if i == 0 else None)
        ax.axvline(dates[tr[-1]], color="grey", lw=0.5, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("close (log)")
    ax.set_title(f"Walk-forward folds — {subtitle}\n"
                 f"{len(folds)} expanding folds; each test span is shaded, training "
                 f"is everything before it", fontsize=9.5)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    return _save(fig, out)


def fig_feature_importance(names: Sequence[str], importances, subtitle: str, out,
                           top: int = 20) -> Path:
    """Tree-ensemble feature importances, top ``top``."""
    _style()
    imp = np.asarray(importances, float)
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) + 1.2))
    ax.barh([names[i] for i in order][::-1], imp[order][::-1], color=ROLE_COLOURS["base"])
    ax.set_xlabel("XGBoost gain importance (mean over folds)")
    ax.set_title(f"Tree ensemble feature importance — {subtitle}")
    return _save(fig, out)


def render_run_figures(result: Dict[str, Any], out_dir: str | os.PathLike = "images",
                       ticker: str = "") -> List[Path]:
    """Every per-run figure, from ``run_pipeline``'s return dict.

    Each figure is attempted independently: a failure in one is reported and
    does not stop the others, because a chart must never take a run down.
    """
    out_dir = Path(out_dir)
    window = result.get("window")
    subtitle = f"{ticker or 'run'}, {window.describe()}" if window is not None else (ticker or "run")

    primary: Dict[str, pd.DataFrame] = result.get("primary_predictions") or {}
    base_names = [r["model_name"] for r in result.get("base_results", {}).values()]
    meta_names = [n for n in primary if "Hybrid meta" in n]
    model_names = base_names + meta_names

    jobs = [
        ("model_comparison.png",
         lambda p: fig_model_comparison(result["comparison_df"], subtitle, p)),
        ("predicted_vs_realised.png",
         lambda p: fig_predicted_vs_realised(result["aligned"], model_names, subtitle, p)),
        ("prediction_dispersion.png",
         lambda p: fig_prediction_dispersion(result["aligned"], list(primary), subtitle, p)),
        ("equity_curves.png",
         lambda p: fig_equity_curves(primary, subtitle, p, result.get("cost_bps", DEFAULT_COST_BPS))),
        ("alpha_beta.png",
         lambda p: fig_alpha_beta(result["economics"], subtitle, p)),
        ("r2_oos_by_fold.png",
         lambda p: fig_per_fold_metric(result["evaluations"], "r2_oos", subtitle, p, "R²_OOS")),
        ("directional_accuracy_by_fold.png",
         lambda p: fig_per_fold_metric(result["evaluations"], "da_minus_majority", subtitle, p,
                                       "DA minus majority")),
        ("calibration.png",
         lambda p: fig_calibration(result["calibration"], subtitle, p)),
        ("walk_forward_folds.png",
         lambda p: fig_walk_forward_folds(result["dataset"].feature_date,
                                          result["dataset"].close_t, result["folds"], subtitle, p)),
        ("feature_importance.png",
         lambda p: fig_feature_importance(result["base_results"]["tree"]["feature_names"],
                                          result["base_results"]["tree"]["mean_feature_importances"],
                                          subtitle, p)),
    ]

    written: List[Path] = []
    for filename, draw in jobs:
        try:
            written.append(draw(out_dir / filename))
        except Exception as exc:  # pragma: no cover - cosmetic path
            print(f"  figure {filename} skipped: {exc}")
    return written
