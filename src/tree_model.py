import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
import joblib
import os

from evaluate import compute_metrics, print_metrics


def get_feature_columns(df):
    """Return the list of feature columns available in the DataFrame."""
    # All columns except Date and the target we create
    exclude = {'Date', 'Target_Close'}
    return [c for c in df.columns if c not in exclude]


def train_tree_model(data, future_days=30):
    """
    Train an XGBoost + Random Forest ensemble (VotingRegressor) for
    next-day close price prediction.

    Returns:
        dict with keys:
            'model': trained VotingRegressor
            'scaler': fitted StandardScaler
            'features': list of feature column names
            'y_test': actual test values
            'y_pred': predicted test values
            'test_dates': dates for the test period
            'future_dates': predicted future dates
            'future_predictions': predicted future close prices
            'feature_importances': array from XGBoost
            'metrics': evaluation metrics dict
    """
    df = data.copy()

    # Target: next day's Close
    df['Target_Close'] = df['Close'].shift(-1)
    df.dropna(subset=['Target_Close'], inplace=True)

    features = get_feature_columns(df)
    X = df[features].values
    y = df['Target_Close'].values
    dates = pd.to_datetime(df['Date']).values

    # Chronological split: 80% train, 20% test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = dates[train_size:]

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Build ensemble ───────────────────────────────────────────────────
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )

    model = VotingRegressor(estimators=[('xgb', xgb), ('rf', rf)])

    print("Training XGBoost + Random Forest ensemble...")
    model.fit(X_train_sc, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_sc)

    metrics = compute_metrics(y_test, y_pred, model_name="Tree Ensemble")
    print_metrics(metrics)

    # ── Feature importance from XGBoost ──────────────────────────────────
    xgb_model = model.named_estimators_['xgb']
    feat_imp = xgb_model.feature_importances_

    # ── Save model ───────────────────────────────────────────────────────
    model_path = 'models/tree_ensemble_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'features': features}, model_path)
    print(f"Model saved to '{model_path}'")

    # ── Future predictions (walk-forward) ────────────────────────────────
    last_date = pd.to_datetime(data['Date']).max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

    # Start from the last known feature row
    current_features = data[features].iloc[-1].values.copy().astype(np.float64)
    future_predictions = []

    for _ in range(future_days):
        feat_sc = scaler.transform(current_features.reshape(1, -1))
        pred = model.predict(feat_sc)[0]
        future_predictions.append(pred)

        # Update 'Close' to the new prediction (index 0 in our feature list)
        close_idx = features.index('Close')
        current_features[close_idx] = pred

        # Update simple derived features where possible
        if 'Return_1d' in features:
            ret_idx = features.index('Return_1d')
            old_close = current_features[close_idx]
            current_features[ret_idx] = (pred - old_close) / old_close if old_close != 0 else 0

    future_predictions = np.array(future_predictions)

    # Save future predictions
    future_df = pd.DataFrame({
        'date': future_dates,
        'Predicted Close Tree': future_predictions,
    })
    os.makedirs('data', exist_ok=True)
    future_df.to_csv('data/future_predictions_tree.csv', index=False)
    print("Future predictions saved to 'data/future_predictions_tree.csv'")

    return {
        'model': model,
        'scaler': scaler,
        'features': features,
        'y_test': y_test,
        'y_pred': y_pred,
        'test_dates': test_dates,
        'future_dates': future_dates,
        'future_predictions': future_predictions,
        'feature_importances': feat_imp,
        'metrics': metrics,
    }
