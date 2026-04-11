import os
import sys
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from fetch_data import fetch_stock_data
from tree_model import train_tree_model
from lstm_model import train_lstm_model
from transformer_model import train_transformer_model
from evaluate import compute_metrics, print_metrics, compare_models
from visualize import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_future_predictions,
    plot_model_comparison_bars,
    plot_feature_importance,
)

def _strip_tz(dt_index):
    dt_index = pd.to_datetime(dt_index)
    if hasattr(dt_index, 'tz') and dt_index.tz is not None:
        dt_index = dt_index.tz_localize(None)
    return dt_index.normalize()

def align_test_results(tree_res, lstm_res, tf_res):
    """Align test-period predictions from all three models."""
    tree_dates = _strip_tz(tree_res['test_dates'])
    lstm_dates = _strip_tz(lstm_res['test_dates'])
    tf_dates = _strip_tz(tf_res['test_dates'])

    df_tree = pd.DataFrame({'date': tree_dates, 'y_true': tree_res['y_test'], 'tree_pred': tree_res['y_pred']})
    df_lstm = pd.DataFrame({'date': lstm_dates, 'lstm_pred': lstm_res['y_pred']})
    df_tf = pd.DataFrame({'date': tf_dates, 'tf_pred': tf_res['y_pred']})

    merged = df_tree.merge(df_lstm, on='date', how='inner').merge(df_tf, on='date', how='inner')

    if merged.empty:
        print("WARNING: No overlapping test dates between models.")
        return None, None, None, None, None

    y_true = merged['y_true'].values
    tree_pred = merged['tree_pred'].values
    lstm_pred = merged['lstm_pred'].values
    tf_pred = merged['tf_pred'].values
    dates = merged['date'].values

    return y_true, tree_pred, lstm_pred, tf_pred, dates

def train_meta_ensemble(y_true, tree_pred, lstm_pred, tf_pred, stock_data, test_dates):
    """
    Train an XGBoost Meta-Learner on the sub-models' predictions.
    Injects Volatility feature if available to allow non-linear weighting based on market fear.
    """
    # Create Meta-features
    X_meta = np.column_stack([tree_pred, lstm_pred, tf_pred])
    
    # Try adding VIX as a contextual feature to the meta-model
    try:
        if 'VIX' in stock_data.columns:
            # Map VIX to the test dates
            df_vix = stock_data[['Date', 'VIX']].copy()
            df_vix['Date'] = _strip_tz(df_vix['Date'])
            df_meta = pd.DataFrame({'Date': test_dates})
            df_meta = df_meta.merge(df_vix, on='Date', how='left')
            vix_vals = df_meta['VIX'].fillna(df_meta['VIX'].median()).values # fillna just in case
            X_meta = np.column_stack([X_meta, vix_vals])
            print("  -> Meta-Learner incorporates 'VIX' for dynamic weighting.")
    except Exception as e:
        print(f"Warning mapping VIX to meta-learner: {e}")

    meta_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    meta_model.fit(X_meta, y_true)

    meta_pred = meta_model.predict(X_meta)
    residual_std = np.std(y_true - meta_pred)
    
    print(f"   Meta-Learner Residual Std (for base confidence): +/-${residual_std:.2f}")

    return meta_model, residual_std

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Stock Market Prediction Model -- XGBoost + BiLSTM + Transformer + Meta-XGBoost',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--start', type=str, default='2010-01-01')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--optimize', action='store_true', help='Run Optuna hyperparameter optimization per model')
    args = parser.parse_args()

    ticker, start_date, future_days = args.ticker.upper(), args.start, args.days
    optimize = args.optimize

    print("=" * 70)
    print(f"  >> PLANET'S BEST STOCK PREDICTOR -- {ticker}")
    print(f"  Historical data from {start_date} | Predicting {future_days} futures")
    print(f"  Optuna Tuning: {'ENABLED' if optimize else 'DISABLED'}")
    print("=" * 70)

    # 1. Fetch
    print("\n[1/6] Fetching & engineering features + Macro context...")
    stock_data = fetch_stock_data(ticker, start_date)
    if stock_data is None: sys.exit(1)

    # 2. Tree Model
    print("\n[2/6] Training Tree Ensemble...")
    tree_res = train_tree_model(stock_data, future_days=future_days)

    # 3. LSTM Model
    print("\n[3/6] Training Bidirectional LSTM...")
    lstm_res = train_lstm_model(stock_data, future_days=future_days)

    # 4. Transformer Model
    print("\n[4/6] Training Time-Series Transformer...")
    tf_res = train_transformer_model(stock_data, future_days=future_days, optimize=optimize)

    # 5. Meta-Ensemble
    print("\n[5/6] Building Non-Linear XGBoost Meta-Ensemble...")
    y_true, tree_p, lstm_p, tf_p, dates = align_test_results(tree_res, lstm_res, tf_res)
    all_metrics = [tree_res['metrics'], lstm_res['metrics'], tf_res['metrics']]

    if y_true is not None:
        meta_model, raw_residual_std = train_meta_ensemble(y_true, tree_p, lstm_p, tf_p, stock_data, dates)

        # Meta Test predictions
        X_meta_test = np.column_stack([tree_p, lstm_p, tf_p])
        if X_meta_test.shape[1] < meta_model.n_features_in_:
            # VIX was added, we need it here
            vix = stock_data[stock_data['Date'].apply(_strip_tz).isin(dates)]['VIX'].values
            if len(vix) == len(X_meta_test):
                X_meta_test = np.column_stack([X_meta_test, vix])
        
        hybrid_test_pred = meta_model.predict(X_meta_test)
        hybrid_metrics = compute_metrics(y_true, hybrid_test_pred, model_name="Hybrid (XGBoost Meta)")
        print_metrics(hybrid_metrics)
        all_metrics.append(hybrid_metrics)

        # Meta Future predictions
        X_meta_future = np.column_stack([
            tree_res['future_predictions'], lstm_res['future_predictions'], tf_res['future_predictions']
        ])
        
        # Volatility-adjusted Confidence Intervals
        vix_current = stock_data['VIX'].iloc[-1] if 'VIX' in stock_data.columns else 20.0
        vix_historical_mean = stock_data['VIX'].mean() if 'VIX' in stock_data.columns else 20.0
        volatility_multiplier = max(1.0, vix_current / vix_historical_mean)

        if X_meta_future.shape[1] < meta_model.n_features_in_:
            X_meta_future = np.column_stack([X_meta_future, np.full(future_days, vix_current)])

        hybrid_future = meta_model.predict(X_meta_future)
        dynamic_std = raw_residual_std * volatility_multiplier
        conf_lower = hybrid_future - 1.96 * dynamic_std
        conf_upper = hybrid_future + 1.96 * dynamic_std
    else:
        # Fallback simple average
        hybrid_test_pred = None
        hybrid_future = (tree_res['future_predictions'] + lstm_res['future_predictions'] + tf_res['future_predictions']) / 3
        conf_lower, conf_upper = None, None
        dates, y_true, tree_p, lstm_p, tf_p = tree_res['test_dates'], tree_res['y_test'], tree_res['y_pred'], lstm_res['y_pred'][:len(tree_res['y_pred'])], tf_res['y_pred'][:len(tree_res['y_pred'])]

    # 6. Output & Visuals
    print("\n[6/6] Generating reports...")
    os.makedirs('data', exist_ok=True)
    comp_df = compare_models(all_metrics, save_path='data/model_comparison.csv')

    combined_future = pd.DataFrame({
        'date': tree_res['future_dates'],
        'Predicted Close Tree': tree_res['future_predictions'],
        'Predicted Close LSTM': lstm_res['future_predictions'],
        'Predicted Close Transformer': tf_res['future_predictions'],
        'Predicted Close Hybrid': hybrid_future,
    })
    if conf_lower is not None:
        combined_future['Conf Lower'] = conf_lower
        combined_future['Conf Upper'] = conf_upper
    combined_future.to_csv('data/combined_predictions.csv', index=False)

    preds_dict = {'Tree': tree_p, 'BiLSTM': lstm_p, 'Transformer': tf_p}
    if hybrid_test_pred is not None: preds_dict['Hybrid Meta'] = hybrid_test_pred
    plot_actual_vs_predicted(dates, y_true, preds_dict)
    plot_residuals(y_true, preds_dict)
    
    future_preds_dict = {
        'Tree': tree_res['future_predictions'], 'BiLSTM': lstm_res['future_predictions'],
        'Transformer': tf_res['future_predictions'], 'Hybrid Meta': hybrid_future
    }
    plot_future_predictions(tree_res['future_dates'], future_preds_dict, conf_lower, conf_upper)
    plot_model_comparison_bars(comp_df)
    plot_feature_importance(tree_res['features'], tree_res['feature_importances'])

    print("\n" + "=" * 70)
    print("  [OK] PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  [FORECAST] {ticker} -- Next {future_days} Business Day Predictions (Hybrid):")
    for i, (d, p) in enumerate(zip(tree_res['future_dates'], hybrid_future)):
        ci = f"  [{conf_lower[i]:.2f} - {conf_upper[i]:.2f}]" if conf_lower is not None else ""
        print(f"    {d.strftime('%Y-%m-%d')}:  ${p:.2f}{ci}")

if __name__ == "__main__":
    main()
