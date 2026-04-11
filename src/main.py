import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from fetch_data import fetch_stock_data
from tree_model import train_tree_model
from lstm_model import train_lstm_model
from evaluate import compute_metrics, print_metrics, compare_models
from visualize import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_future_predictions,
    plot_model_comparison_bars,
    plot_feature_importance,
)


def align_test_results(tree_result, lstm_result):
    """
    Align the test-period predictions from both models to the same date range
    so they can be stacked. The tree model predicts next-day close from features,
    while the LSTM predicts from sequences — their test sets may differ in length.

    Returns aligned y_true, tree_pred, lstm_pred, dates arrays.
    """
    def _strip_tz(dt_index):
        dt_index = pd.to_datetime(dt_index)
        if hasattr(dt_index, 'tz') and dt_index.tz is not None:
            dt_index = dt_index.tz_localize(None)
        return dt_index.normalize()

    tree_dates = _strip_tz(tree_result['test_dates'])
    lstm_dates = _strip_tz(lstm_result['test_dates'])

    # Build DataFrames for merge
    tree_df = pd.DataFrame({
        'date': tree_dates,
        'y_true_tree': tree_result['y_test'],
        'tree_pred': tree_result['y_pred'],
    })
    lstm_df = pd.DataFrame({
        'date': lstm_dates,
        'y_true_lstm': lstm_result['y_test'],
        'lstm_pred': lstm_result['y_pred'],
    })

    merged = pd.merge(tree_df, lstm_df, on='date', how='inner')

    if merged.empty:
        print("WARNING: No overlapping test dates between models. Falling back to simple average.")
        return None, None, None, None

    # Use tree's y_true (they should be identical on overlapping dates)
    y_true = merged['y_true_tree'].values
    tree_pred = merged['tree_pred'].values
    lstm_pred = merged['lstm_pred'].values
    dates = merged['date'].values

    return y_true, tree_pred, lstm_pred, dates


def train_meta_ensemble(y_true, tree_pred, lstm_pred):
    """
    Train a Ridge Regression meta-learner that learns optimal weights
    to combine the tree and LSTM predictions.

    Returns:
        meta_model: Fitted Ridge model.
        residual_std: Std of residuals on training data (for confidence intervals).
    """
    # Stack sub-model predictions as features
    X_meta = np.column_stack([tree_pred, lstm_pred])

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta, y_true)

    # Residual std for confidence intervals
    meta_pred = meta_model.predict(X_meta)
    residual_std = np.std(y_true - meta_pred)

    w_tree, w_lstm = meta_model.coef_
    print(f"\n[META] Meta-Ensemble Weights: Tree={w_tree:.4f}, LSTM={w_lstm:.4f}, Intercept={meta_model.intercept_:.4f}")
    print(f"   Residual Std (for confidence): +/-${residual_std:.2f}")

    return meta_model, residual_std


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Stock Market Prediction Model — XGBoost + BiLSTM + Meta-Ensemble',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (e.g., AAPL, TSLA, GOOGL)')
    parser.add_argument('--start', type=str, default='2010-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of future business days to predict')
    args = parser.parse_args()

    ticker = args.ticker.upper()
    start_date = args.start
    future_days = args.days

    print("=" * 70)
    print(f"  >> HYBRID STOCK PREDICTION -- {ticker}")
    print(f"  Historical data from {start_date} | Predicting {future_days} future days")
    print("=" * 70)

    # ── 1. Fetch Data ────────────────────────────────────────────────────
    print("\n[1/5] Fetching & engineering features...")
    stock_data = fetch_stock_data(ticker, start_date)

    if stock_data is None:
        print("Failed to fetch stock data. Exiting.")
        sys.exit(1)

    # ── 2. Train Sub-Models ──────────────────────────────────────────────
    print("\n[2/5] Training Tree Ensemble (XGBoost + Random Forest)...")
    tree_result = train_tree_model(stock_data, future_days=future_days)

    print("\n[3/5] Training Bidirectional LSTM...")
    lstm_result = train_lstm_model(stock_data, future_days=future_days)

    # ── 3. Meta-Ensemble (Stacking) ─────────────────────────────────────
    print("\n[4/5] Building meta-ensemble...")
    y_true, tree_pred, lstm_pred, aligned_dates = align_test_results(tree_result, lstm_result)

    all_metrics = [tree_result['metrics'], lstm_result['metrics']]

    if y_true is not None:
        meta_model, residual_std = train_meta_ensemble(y_true, tree_pred, lstm_pred)

        # Hybrid test predictions
        X_meta_test = np.column_stack([tree_pred, lstm_pred])
        hybrid_test_pred = meta_model.predict(X_meta_test)

        hybrid_metrics = compute_metrics(y_true, hybrid_test_pred, model_name="Hybrid (Stacking)")
        print_metrics(hybrid_metrics)
        all_metrics.append(hybrid_metrics)

        # Hybrid future predictions
        X_meta_future = np.column_stack([
            tree_result['future_predictions'],
            lstm_result['future_predictions'],
        ])
        hybrid_future = meta_model.predict(X_meta_future)

        # Confidence intervals (±1.96σ ≈ 95%)
        conf_lower = hybrid_future - 1.96 * residual_std
        conf_upper = hybrid_future + 1.96 * residual_std
    else:
        # Fallback: simple average
        hybrid_test_pred = None
        hybrid_future = (tree_result['future_predictions'] + lstm_result['future_predictions']) / 2
        conf_lower = None
        conf_upper = None
        aligned_dates = tree_result['test_dates']
        y_true = tree_result['y_test']
        tree_pred = tree_result['y_pred']
        lstm_pred = lstm_result['y_pred'][:len(tree_pred)]

    # ── 4. Comparison & Visualization ────────────────────────────────────
    print("\n[5/5] Generating visualizations & reports...")

    # Model comparison table
    os.makedirs('data', exist_ok=True)
    comparison_df = compare_models(all_metrics, save_path='data/model_comparison.csv')

    # Save combined future predictions
    future_dates = tree_result['future_dates']
    combined_future = pd.DataFrame({
        'date': future_dates,
        'Predicted Close Tree': tree_result['future_predictions'],
        'Predicted Close LSTM': lstm_result['future_predictions'],
        'Predicted Close Hybrid': hybrid_future,
    })
    if conf_lower is not None:
        combined_future['Confidence Lower (95%)'] = conf_lower
        combined_future['Confidence Upper (95%)'] = conf_upper

    combined_future.to_csv('data/combined_predictions.csv', index=False)
    print(f"Combined predictions saved to 'data/combined_predictions.csv'")

    # ── Plots ────────────────────────────────────────────────────────────
    predictions_dict = {
        'Tree Ensemble': tree_pred,
        'BiLSTM': lstm_pred,
    }
    if hybrid_test_pred is not None:
        predictions_dict['Hybrid (Stacking)'] = hybrid_test_pred

    plot_actual_vs_predicted(aligned_dates, y_true, predictions_dict)
    plot_residuals(y_true, predictions_dict)

    future_preds_dict = {
        'Tree Ensemble': tree_result['future_predictions'],
        'BiLSTM': lstm_result['future_predictions'],
        'Hybrid (Stacking)': hybrid_future,
    }
    plot_future_predictions(future_dates, future_preds_dict, conf_lower, conf_upper)
    plot_model_comparison_bars(comparison_df)
    plot_feature_importance(tree_result['features'], tree_result['feature_importances'])

    # ── Final Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [OK] PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  -> Predictions:    data/combined_predictions.csv")
    print(f"  -> Metrics:        data/model_comparison.csv")
    print(f"  -> Models:         models/")
    print(f"  -> Visualizations: images/")
    print("=" * 70)

    # Print the future prediction summary
    print(f"\n  [FORECAST] {ticker} -- Next {future_days} Business Day Predictions (Hybrid):")
    print("  " + "-" * 50)
    for i, (date, price) in enumerate(zip(future_dates, hybrid_future)):
        ci = f"  [{conf_lower[i]:.2f} - {conf_upper[i]:.2f}]" if conf_lower is not None else ""
        print(f"    {date.strftime('%Y-%m-%d')}:  ${price:.2f}{ci}")
    print()


if __name__ == "__main__":
    main()
