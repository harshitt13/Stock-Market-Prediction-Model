"""
XGBoost + Random Forest tree ensemble for next-day close price prediction.

Refactored to support walk-forward validation via external fold indices.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
import joblib
import os
from typing import Dict, Any, List, Tuple, Optional

from evaluate import compute_metrics, print_metrics


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of feature columns available in the DataFrame."""
    exclude = {"Date", "Target_Close"}
    return [c for c in df.columns if c not in exclude]


def _build_tree_ensemble() -> VotingRegressor:
    """Build the XGBoost + RF VotingRegressor (deterministic config)."""
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
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    return VotingRegressor(estimators=[("xgb", xgb), ("rf", rf)])


def train_tree_on_fold(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: List[str],
) -> Dict[str, Any]:
    """
    Train tree ensemble on a single walk-forward fold.

    Parameters
    ----------
    X : 2D array (n_samples, n_features)
    y : 1D array (n_samples,) - target (next-day close).
    dates : 1D array of dates.
    train_idx, test_idx : fold index arrays.
    features : list of feature names (for importance mapping).

    Returns
    -------
    dict with model, scaler, y_true, y_pred, test_dates, feature_importances, metrics.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_dates = dates[test_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = _build_tree_ensemble()
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    # Feature importance from XGBoost sub-estimator
    xgb_model = model.named_estimators_["xgb"]
    feat_imp = xgb_model.feature_importances_

    metrics = compute_metrics(y_test, y_pred, model_name="Tree Ensemble")

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "y_true": y_test,
        "y_pred": y_pred,
        "test_dates": test_dates,
        "feature_importances": feat_imp,
        "metrics": metrics,
    }


def train_tree_model(
    data: pd.DataFrame,
    future_days: int = 30,
    fold_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """
    Train the tree ensemble with walk-forward validation.

    Parameters
    ----------
    data : pd.DataFrame
        Full stock data with engineered features.
    future_days : int
        Number of future days to forecast.
    fold_indices : list of (train_idx, test_idx), optional
        If provided, uses these walk-forward folds.
        If None, falls back to a single 80/20 chronological split.

    Returns
    -------
    dict with aggregated results across folds + future predictions.
    """
    from walk_forward import aggregate_fold_results

    df = data.copy()

    # Target: next day's Close
    df["Target_Close"] = df["Close"].shift(-1)
    df.dropna(subset=["Target_Close"], inplace=True)

    features = get_feature_columns(df)
    X = df[features].values
    y = df["Target_Close"].values
    dates = pd.to_datetime(df["Date"]).values

    # --- Walk-forward or single split ---
    if fold_indices is None:
        # Fallback: single 80/20 split
        train_size = int(len(X) * 0.8)
        fold_indices = [(np.arange(0, train_size), np.arange(train_size, len(X)))]

    fold_results = []
    last_model = None
    last_scaler = None
    last_feat_imp = None

    for i, (train_idx, test_idx) in enumerate(fold_indices):
        # Clip indices to valid range (target shifted by -1 removes last row)
        train_idx = train_idx[train_idx < len(X)]
        test_idx = test_idx[test_idx < len(X)]
        if len(test_idx) == 0:
            continue

        print(f"  Tree Fold {i+1}/{len(fold_indices)} - "
              f"train={len(train_idx)}, test={len(test_idx)}")

        fold_res = train_tree_on_fold(X, y, dates, train_idx, test_idx, features)
        fold_results.append(fold_res)
        last_model = fold_res["model"]
        last_scaler = fold_res["scaler"]
        last_feat_imp = fold_res["feature_importances"]

    # Aggregate
    agg = aggregate_fold_results(fold_results)
    agg_metrics = compute_metrics(
        agg["y_true"], agg["y_pred"], model_name="Tree Ensemble"
    )
    print_metrics(agg_metrics)

    # --- Save model (last fold's model) ---
    model_path = "models/tree_ensemble_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {"model": last_model, "scaler": last_scaler, "features": features}, model_path
    )
    print(f"Model saved to '{model_path}'")

    # --- Future predictions (using last-fold model) ---
    last_date = pd.to_datetime(data["Date"]).max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_days, freq="B"
    )

    current_features = data[features].iloc[-1].values.copy().astype(np.float64)
    future_predictions = []

    for _ in range(future_days):
        feat_sc = last_scaler.transform(current_features.reshape(1, -1))
        pred = last_model.predict(feat_sc)[0]
        future_predictions.append(pred)
        close_idx = features.index("Close")
        current_features[close_idx] = pred
        if "Return_1d" in features:
            ret_idx = features.index("Return_1d")
            old_close = current_features[close_idx]
            current_features[ret_idx] = (
                (pred - old_close) / old_close if old_close != 0 else 0
            )

    future_predictions = np.array(future_predictions)

    # Save future predictions
    future_df = pd.DataFrame(
        {"date": future_dates, "Predicted Close Tree": future_predictions}
    )
    os.makedirs("data", exist_ok=True)
    future_df.to_csv("data/future_predictions_tree.csv", index=False)
    print("Future predictions saved to 'data/future_predictions_tree.csv'")

    return {
        "model": last_model,
        "scaler": last_scaler,
        "features": features,
        "y_test": agg["y_true"],
        "y_pred": agg["y_pred"],
        "test_dates": agg["test_dates"],
        "future_dates": future_dates,
        "future_predictions": future_predictions,
        "feature_importances": last_feat_imp,
        "metrics": agg_metrics,
        "per_fold_metrics": agg.get("per_fold_metrics", []),
    }
