"""
Evaluation metrics module — Classification-primary, Regression-secondary.

Primary metrics  (directional / classification):
    - Directional Accuracy
    - F1 Score (macro)
    - Precision / Recall
    - Confusion Matrix

Secondary metrics (regression):
    - RMSE, MAE, MAPE, R², Max Error
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

DIRECTION_THRESHOLD = 0.001  # 0.1 % move threshold for up / down


def derive_direction_labels(
    prices: np.ndarray, threshold: float = DIRECTION_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive binary direction labels from a price series.

    Returns
    -------
    labels : np.ndarray
        1 = up, 0 = down.  Sub-threshold moves are marked as -1 (to be
        filtered out).
    mask : np.ndarray (bool)
        True where the absolute return exceeds *threshold*.
    """
    returns = np.diff(prices) / prices[:-1]
    labels = np.where(returns > threshold, 1, np.where(returns < -threshold, 0, -1))
    mask = labels != -1
    return labels, mask


def compute_classification_metrics(
    y_true_prices: np.ndarray,
    y_pred_prices: np.ndarray,
    model_name: str = "Model",
    threshold: float = DIRECTION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute classification (directional) metrics from price arrays.

    The actual and predicted *direction* is derived from consecutive price
    differences.  Sub-threshold moves are dropped from evaluation.

    Parameters
    ----------
    y_true_prices : array
        Actual closing prices (length N).
    y_pred_prices : array
        Predicted closing prices (length N).
    model_name : str
        Label for the model.
    threshold : float
        Minimum absolute return to count as up/down.

    Returns
    -------
    dict with classification metrics.
    """
    y_true_prices = np.asarray(y_true_prices, dtype=np.float64)
    y_pred_prices = np.asarray(y_pred_prices, dtype=np.float64)

    if len(y_true_prices) < 2:
        return {
            "Model": model_name,
            "Directional Accuracy (%)": float("nan"),
            "F1 Score": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "Confusion Matrix": None,
        }

    # Actual direction from true price changes
    actual_labels, actual_mask = derive_direction_labels(y_true_prices, threshold)
    # Predicted direction from predicted price changes
    pred_labels, _ = derive_direction_labels(y_pred_prices, threshold)

    # Align: only keep indices where ACTUAL direction is significant
    # For predicted direction, if sub-threshold assign 0 (down) as conservative
    pred_labels_aligned = pred_labels.copy()
    pred_labels_aligned[pred_labels_aligned == -1] = 0  # treat flat as down

    # Apply actual mask
    a = actual_labels[actual_mask].astype(int)
    p = pred_labels_aligned[actual_mask].astype(int)

    if len(a) == 0:
        return {
            "Model": model_name,
            "Directional Accuracy (%)": float("nan"),
            "F1 Score": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "Confusion Matrix": None,
        }

    da = accuracy_score(a, p) * 100
    f1 = f1_score(a, p, average="macro", zero_division=0)
    prec = precision_score(a, p, average="macro", zero_division=0)
    rec = recall_score(a, p, average="macro", zero_division=0)
    cm = confusion_matrix(a, p, labels=[0, 1])

    return {
        "Model": model_name,
        "Directional Accuracy (%)": round(da, 2),
        "F1 Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Confusion Matrix": cm,
    }


# ---------------------------------------------------------------------------
# Regression metrics (kept as secondary)
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Compute regression metrics for stock price prediction.

    Returns
    -------
    dict with MSE, RMSE, MAE, MAPE, R², Max Error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE — handle zeros by masking
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")

    max_err = np.max(np.abs(y_true - y_pred))

    return {
        "Model": model_name,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE (%)": round(mape, 2),
        "R2": round(r2, 4),
        "Max Error": round(max_err, 4),
    }


# ---------------------------------------------------------------------------
# Combined metrics (classification-primary + regression-secondary)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute both classification and regression metrics.

    Classification metrics are derived from price direction changes.
    Regression metrics are standard error metrics.

    Returns
    -------
    dict merging both metric sets, with 'Model' key.
    """
    cls_metrics = compute_classification_metrics(y_true, y_pred, model_name)
    reg_metrics = compute_regression_metrics(y_true, y_pred, model_name)

    # Merge: classification first (primary), then regression (secondary)
    combined = {"Model": model_name}

    # Primary — classification
    for k in ["Directional Accuracy (%)", "F1 Score", "Precision", "Recall"]:
        combined[k] = cls_metrics.get(k)
    combined["Confusion Matrix"] = cls_metrics.get("Confusion Matrix")

    # Secondary — regression
    for k in ["RMSE", "MAE", "MAPE (%)", "R2", "Max Error"]:
        combined[k] = reg_metrics.get(k)

    return combined


# ---------------------------------------------------------------------------
# Display / comparison
# ---------------------------------------------------------------------------

def print_metrics(metrics_dict: Dict[str, Any]):
    """Pretty-print a single model's metrics (classification-first)."""
    print(f"\n{'='*60}")
    print(f"  [EVAL] {metrics_dict['Model']} — Evaluation Results")
    print(f"{'='*60}")

    # Primary — Classification
    print(f"\n  {'-'*20} PRIMARY (Classification) {'-'*20}")
    print(f"  Directional Accuracy : {metrics_dict.get('Directional Accuracy (%)', np.nan):.2f}%")
    print(f"  F1 Score             : {metrics_dict.get('F1 Score', np.nan):.4f}")
    print(f"  Precision            : {metrics_dict.get('Precision', np.nan):.4f}")
    print(f"  Recall               : {metrics_dict.get('Recall', np.nan):.4f}")

    cm = metrics_dict.get("Confusion Matrix")
    if cm is not None:
        print("\n  Confusion Matrix (Actual \\ Predicted):")
        print("                   Pred DOWN   Pred UP")
        print(f"    Actual DOWN  | {cm[0,0]:9d} | {cm[0,1]:7d}")
        print(f"    Actual UP    | {cm[1,0]:9d} | {cm[1,1]:7d}")

    print(f"\n  {'-'*20} SECONDARY (Regression) {'-'*20}")
    print(f"  RMSE                 : {metrics_dict.get('RMSE', np.nan):.4f}")
    print(f"  MAE                  : {metrics_dict.get('MAE', np.nan):.4f}")
    print(f"  MAPE                 : {metrics_dict.get('MAPE (%)', np.nan):.2f}%")
    print(f"  R-Squared            : {metrics_dict.get('R2', np.nan):.4f}")


def compare_models(
    metrics_list: List[Dict[str, Any]], save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Print a comparison table of multiple models and optionally save to CSV.

    Parameters
    ----------
    metrics_list : list of dicts
        Each dict from ``compute_metrics()``.
    save_path : str, optional
        CSV path to save.

    Returns
    -------
    pd.DataFrame : comparison table (models as index).
    """
    # Build a clean DF excluding the confusion matrix object
    rows = []
    for m in metrics_list:
        row = {k: v for k, v in m.items() if k != "Confusion Matrix"}
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    print(f"\n{'='*80}")
    print("  MODEL COMPARISON  (Classification = Primary, Regression = Secondary)")
    print(f"{'='*80}")
    print(df.to_string())
    print(f"{'='*80}\n")

    if save_path:
        df.to_csv(save_path)
        print(f"Comparison table saved to '{save_path}'")

    return df
