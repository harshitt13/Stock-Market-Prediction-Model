"""
Visualization dashboard — dark-themed, professional charts.

Includes new plots:
  - Confusion matrix heatmap
  - Walk-forward fold boundaries
  - Updated model comparison (classification + regression)
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from typing import List, Tuple


# ── Consistent style ────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "font.family": "sans-serif",
        "font.size": 10,
    }
)

COLORS = {
    "actual": "#58a6ff",
    "tree": "#3fb950",
    "lstm": "#f78166",
    "transformer": "#ffa657",
    "hybrid": "#d2a8ff",
    "confidence": "#d2a8ff",
    "naive": "#8b949e",
    "arima": "#79c0ff",
}


def plot_feature_importance(
    feature_names, importances, save_path="images/feature_importance.png", top_n=15
) -> None:
    """
    Plot feature importances from a tree-based model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = [importances[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(names)), vals[::-1],
        color=COLORS["tree"], alpha=0.9, edgecolor="#30363d",
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Importance")
    ax.set_title(
        "Top Feature Importances (XGBoost)", fontsize=14, fontweight="bold", pad=15
    )
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_walk_forward_folds(
    dates: np.ndarray,
    prices: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    save_path: str = "images/walk_forward_folds.png",
) -> None:
    """
    Visualize walk-forward fold boundaries overlaid on the price chart.

    Each fold's test region is highlighted with a colored band.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, prices, color=COLORS["actual"], linewidth=1.5, label="Close Price")

    fold_colors = ["#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff", "#8b949e"]
    for i, (train_idx, test_idx) in enumerate(folds):
        color = fold_colors[i % len(fold_colors)]
        test_start = dates[test_idx[0]]
        test_end = dates[test_idx[-1]]
        ax.axvspan(test_start, test_end, alpha=0.15, color=color, label=f"Fold {i+1} test")

    ax.set_title(
        "Walk-Forward Validation — Fold Boundaries",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price ($)")
    ax.legend(framealpha=0.9, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
