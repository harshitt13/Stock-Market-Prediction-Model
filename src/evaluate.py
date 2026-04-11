import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Compute a comprehensive set of regression metrics for stock price prediction.

    Args:
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.
        model_name (str): Label for the model.

    Returns:
        dict: Dictionary of metric_name -> value.
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
        mape = float('nan')

    # Directional Accuracy — % of days where predicted direction matches actual
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        da = np.mean(actual_dir == pred_dir) * 100
    else:
        da = float('nan')

    # Max error
    max_err = np.max(np.abs(y_true - y_pred))

    metrics = {
        'Model': model_name,
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE (%)': round(mape, 2),
        'R2': round(r2, 4),
        'Directional Accuracy (%)': round(da, 2),
        'Max Error': round(max_err, 4),
    }

    return metrics


def print_metrics(metrics_dict):
    """Pretty-print a single model's metrics."""
    print(f"\n{'=' * 50}")
    print(f"  [EVAL] {metrics_dict['Model']} -- Evaluation Results")
    print(f"{'=' * 50}")
    for k, v in metrics_dict.items():
        if k == 'Model':
            continue
        print(f"  {k:<28s}: {v}")
    print(f"{'=' * 50}\n")


def compare_models(metrics_list, save_path=None):
    """
    Print a comparison table of multiple models and optionally save to CSV.

    Args:
        metrics_list (list[dict]): List of metric dictionaries from compute_metrics.
        save_path (str): Optional CSV path to save the comparison table.

    Returns:
        pd.DataFrame: Comparison DataFrame.
    """
    df = pd.DataFrame(metrics_list)
    df = df.set_index('Model')

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70 + "\n")

    if save_path:
        df.to_csv(save_path)
        print(f"Comparison table saved to '{save_path}'")

    return df
