import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os


# ── Consistent style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'font.family': 'sans-serif',
    'font.size': 10,
})

COLORS = {
    'actual': '#58a6ff',
    'tree': '#3fb950',
    'lstm': '#f78166',
    'hybrid': '#d2a8ff',
    'confidence': '#d2a8ff',
}


def plot_actual_vs_predicted(dates, y_true, predictions_dict, save_path='images/actual_vs_predicted.png'):
    """
    Plot actual vs predicted close prices for multiple models.

    Args:
        dates: Array-like of dates for the test period.
        y_true: Actual close prices.
        predictions_dict: Dict of {model_name: predictions_array}.
        save_path: Where to save the plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, y_true, label='Actual', color=COLORS['actual'], linewidth=2)

    model_colors = list(COLORS.values())[1:]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        color = model_colors[i % len(model_colors)]
        ax.plot(dates, preds, label=name, color=color, linewidth=1.5, alpha=0.85)

    ax.set_title('Actual vs Predicted Close Prices', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price ($)')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_residuals(y_true, predictions_dict, save_path='images/residual_distribution.png'):
    """
    Plot residual distribution histograms for each model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    model_colors = list(COLORS.values())[1:]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        ax = axes[0][i]
        residuals = np.asarray(y_true) - np.asarray(preds)
        color = model_colors[i % len(model_colors)]
        ax.hist(residuals, bins=40, color=color, alpha=0.8, edgecolor='#30363d')
        ax.axvline(0, color='#f0f6fc', linestyle='--', linewidth=1)
        ax.set_title(f'{name} Residuals', fontsize=12, fontweight='bold')
        ax.set_xlabel('Error ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_future_predictions(future_dates, future_preds_dict, confidence_lower=None, confidence_upper=None,
                            save_path='images/future_predictions.png'):
    """
    Plot future price predictions with optional confidence bands.

    Args:
        future_dates: Array of future dates.
        future_preds_dict: Dict of {model_name: future_predictions_array}.
        confidence_lower: Lower bound array for the hybrid model.
        confidence_upper: Upper bound array for the hybrid model.
        save_path: Where to save.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    model_colors = list(COLORS.values())[1:]
    for i, (name, preds) in enumerate(future_preds_dict.items()):
        color = model_colors[i % len(model_colors)]
        ax.plot(future_dates, preds, label=name, color=color, linewidth=2, marker='o', markersize=3)

    if confidence_lower is not None and confidence_upper is not None:
        ax.fill_between(future_dates, confidence_lower, confidence_upper,
                        color=COLORS['confidence'], alpha=0.15, label='95% Confidence')

    ax.set_title('Future Price Predictions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Close Price ($)')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_model_comparison_bars(metrics_df, save_path='images/model_comparison.png'):
    """
    Plot a grouped bar chart comparing model metrics.

    Args:
        metrics_df: DataFrame with models as index, metrics as columns.
        save_path: Where to save.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Select a subset of key metrics for visual comparison
    key_metrics = ['RMSE', 'MAE', 'MAPE (%)', 'Directional Accuracy (%)']
    available = [m for m in key_metrics if m in metrics_df.columns]
    plot_df = metrics_df[available]

    n_models = len(plot_df)
    n_metrics = len(available)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))

    model_colors = [COLORS['tree'], COLORS['lstm'], COLORS['hybrid'], COLORS['actual']]
    for i, (model_name, row) in enumerate(plot_df.iterrows()):
        offset = (i - n_models / 2 + 0.5) * width
        color = model_colors[i % len(model_colors)]
        bars = ax.bar(x + offset, row.values, width, label=model_name, color=color, alpha=0.9, edgecolor='#30363d')
        # Add value labels
        for bar, val in zip(bars, row.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(row.values),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='#c9d1d9')

    ax.set_xticks(x)
    ax.set_xticklabels(available, fontsize=11)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.legend(framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_feature_importance(feature_names, importances, save_path='images/feature_importance.png', top_n=15):
    """
    Plot feature importances from a tree-based model.

    Args:
        feature_names: List of feature names.
        importances: Corresponding importance values.
        save_path: Where to save.
        top_n: Number of top features to display.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = [importances[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), vals[::-1], color=COLORS['tree'], alpha=0.9, edgecolor='#30363d')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel('Importance')
    ax.set_title('Top Feature Importances (XGBoost)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, axis='x', alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")
