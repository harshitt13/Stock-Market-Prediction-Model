"""
Hybrid Stock Market Prediction Model - CLI Orchestrator & Meta-Ensemble.

Refactored into clean, importable functions for future FastAPI integration.
Key changes from v1:
  - Walk-forward validation replaces static 80/20 splits
  - Out-of-fold meta-learner (no data leakage)
  - Classification-primary evaluation
  - Naive + ARIMA baselines included
  - Confidence interval calibration
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from xgboost import XGBRegressor
from typing import Dict, Any, Optional, List, Tuple

from fetch_data import fetch_stock_data
from tree_model import train_tree_model
from lstm_model import train_lstm_model
from transformer_model import train_transformer_model
from evaluate import compute_metrics, print_metrics, compare_models
from baselines import run_baselines_walkforward
from calibration import (
    compute_volatility_adjusted_ci,
    calibrate_confidence_interval,
    print_calibration_report,
)
from walk_forward import WalkForwardSplitter, print_fold_summary
from visualize import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_future_predictions,
    plot_model_comparison_bars,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_walk_forward_folds,
)


def _strip_tz(dt_index):
    dt_index = pd.to_datetime(dt_index)
    if hasattr(dt_index, "dt"):
        if getattr(dt_index.dt, "tz", None) is not None:
            dt_index = dt_index.dt.tz_localize(None)
        return dt_index.dt.normalize()
    else:
        if getattr(dt_index, "tz", None) is not None:
            dt_index = dt_index.tz_localize(None)
        return dt_index.normalize()


# ---------------------------------------------------------------------------
# Pipeline building blocks (importable for FastAPI)
# ---------------------------------------------------------------------------


def fetch_and_prepare_data(
    ticker: str, start_date: str, end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Step 1: Fetch stock data and engineer features.

    Returns None if data fetch fails.
    """
    return fetch_stock_data(ticker, start_date, end_date)


def create_walk_forward_folds(
    data: pd.DataFrame,
    min_train_size: int = 504,
    test_size: int = 63,
    step_size: int = 63,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Step 2: Generate walk-forward validation folds.

    Returns list of (train_idx, test_idx) tuples.
    """
    splitter = WalkForwardSplitter(
        min_train_size=min_train_size,
        test_size=test_size,
        step_size=step_size,
    )
    n = len(data)
    folds = splitter.split(n)

    dates = pd.to_datetime(data["Date"]).values
    print_fold_summary(folds, dates)

    return folds


def train_base_models(
    data: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    future_days: int = 30,
    optimize: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Step 3: Train all three base models using walk-forward validation.

    Returns dict with keys 'tree', 'lstm', 'transformer', each containing
    the model's full result dict.
    """
    print("\n[2/7] Training Tree Ensemble (walk-forward)...")
    tree_res = train_tree_model(data, future_days=future_days, fold_indices=folds)

    print("\n[3/7] Training Bidirectional LSTM (walk-forward)...")
    lstm_res = train_lstm_model(data, future_days=future_days, fold_indices=folds)

    print("\n[4/7] Training Time-Series Transformer (walk-forward)...")
    tf_res = train_transformer_model(
        data, future_days=future_days, optimize=optimize, fold_indices=folds
    )

    return {"tree": tree_res, "lstm": lstm_res, "transformer": tf_res}


def align_test_results(
    tree_res: Dict, lstm_res: Dict, tf_res: Dict
) -> Tuple[Optional[np.ndarray], ...]:
    """
    Align test-period predictions from all three models by date.

    Returns (y_true, tree_pred, lstm_pred, tf_pred, dates) or all None.
    """
    tree_dates = _strip_tz(tree_res["test_dates"])
    lstm_dates = _strip_tz(lstm_res["test_dates"])
    tf_dates = _strip_tz(tf_res["test_dates"])

    df_tree = pd.DataFrame(
        {"date": tree_dates, "y_true": tree_res["y_test"], "tree_pred": tree_res["y_pred"]}
    )
    df_lstm = pd.DataFrame({"date": lstm_dates, "lstm_pred": lstm_res["y_pred"]})
    df_tf = pd.DataFrame({"date": tf_dates, "tf_pred": tf_res["y_pred"]})

    merged = df_tree.merge(df_lstm, on="date", how="inner").merge(
        df_tf, on="date", how="inner"
    )

    if merged.empty:
        print("WARNING: No overlapping test dates between models.")
        return None, None, None, None, None

    return (
        merged["y_true"].values,
        merged["tree_pred"].values,
        merged["lstm_pred"].values,
        merged["tf_pred"].values,
        merged["date"].values,
    )


def train_meta_ensemble_oof(
    y_true: np.ndarray,
    tree_pred: np.ndarray,
    lstm_pred: np.ndarray,
    tf_pred: np.ndarray,
    stock_data: pd.DataFrame,
    test_dates: np.ndarray,
) -> Tuple[XGBRegressor, np.ndarray, float]:
    """
    Train the XGBoost meta-learner using out-of-fold predictions.

    To avoid leakage, we use time-respecting K-fold within the aligned
    test predictions: the first 80% trains the meta-learner, and the
    last 20% is used for evaluation.

    Returns (meta_model, meta_predictions_on_eval, residual_std).
    """
    n = len(y_true)
    meta_train_end = int(n * 0.8)

    # Build meta-features
    X_meta = np.column_stack([tree_pred, lstm_pred, tf_pred])

    # Add VIX as contextual feature
    has_vix = False
    try:
        if "VIX" in stock_data.columns:
            df_vix = stock_data[["Date", "VIX"]].copy()
            df_vix["Date"] = _strip_tz(df_vix["Date"])
            df_meta = pd.DataFrame({"Date": test_dates})
            df_meta = df_meta.merge(df_vix, on="Date", how="left")
            vix_vals = df_meta["VIX"].fillna(df_meta["VIX"].median()).values
            X_meta = np.column_stack([X_meta, vix_vals])
            has_vix = True
            print("  -> Meta-Learner incorporates 'VIX' for dynamic weighting.")
    except Exception as e:
        print(f"  Warning mapping VIX to meta-learner: {e}")

    # Split: train meta on first portion, evaluate on last portion
    X_meta_train = X_meta[:meta_train_end]
    y_meta_train = y_true[:meta_train_end]
    X_meta_eval = X_meta[meta_train_end:]
    y_meta_eval = y_true[meta_train_end:]

    from sklearn.linear_model import RidgeCV
    meta_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
    meta_model.fit(X_meta_train, y_meta_train)

    # Predict on evaluation portion (OOF - model never saw this data)
    meta_pred_eval = meta_model.predict(X_meta_eval)

    # Also predict on all data for reporting
    meta_pred_all = meta_model.predict(X_meta)

    # Residual std from the OOF evaluation portion
    residual_std = np.std(y_meta_eval - meta_pred_eval)
    print(f"  Meta-Learner OOF Residual Std: +/-${residual_std:.2f}")
    print(f"  Meta-Learner trained on {meta_train_end} samples, "
          f"evaluated on {len(y_meta_eval)} OOF samples.")

    return meta_model, meta_pred_all, residual_std


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_classification_meta_ensemble_oof(
    y_true: np.ndarray,
    tree_pred: np.ndarray,
    lstm_pred: np.ndarray,
    tf_pred: np.ndarray,
    stock_data: pd.DataFrame,
    test_dates: np.ndarray,
) -> Tuple[XGBClassifier, np.ndarray]:
    """
    Train an XGBoost meta-learner to predict the DIRECTION (UP=1, DOWN=0)
    rather than the magnitude.
    """
    n = len(y_true)
    meta_train_end = int(n * 0.8)

    # Get Current Close (T) to determine directions to T+1
    df_prices = stock_data[["Date", "Close"]].copy()
    df_prices["Date"] = _strip_tz(df_prices["Date"])
    df_prices.rename(columns={"Close": "Current_Close"}, inplace=True)
    
    df_meta = pd.DataFrame({"Date": _strip_tz(test_dates)})
    df_meta = df_meta.merge(df_prices, on="Date", how="left")
    
    # Forward fill just in case, though they should match exactly
    df_meta["Current_Close"] = df_meta["Current_Close"].ffill().bfill()
    current_close = df_meta["Current_Close"].values

    # Base models' predicted directions
    tree_dir = (tree_pred > current_close).astype(int)
    lstm_dir = (lstm_pred > current_close).astype(int)
    tf_dir = (tf_pred > current_close).astype(int)
    
    # True direction
    y_class = (y_true > current_close).astype(int)

    X_meta = np.column_stack([tree_dir, lstm_dir, tf_dir])

    # Add VIX as contextual feature
    try:
        if "VIX" in stock_data.columns:
            df_vix = stock_data[["Date", "VIX"]].copy()
            df_vix["Date"] = _strip_tz(df_vix["Date"])
            df_meta2 = pd.DataFrame({"Date": _strip_tz(test_dates)})
            df_meta2 = df_meta2.merge(df_vix, on="Date", how="left")
            vix_vals = df_meta2["VIX"].fillna(df_meta2["VIX"].median()).values
            X_meta = np.column_stack([X_meta, vix_vals])
    except Exception:
        pass

    X_meta_train = X_meta[:meta_train_end]
    y_meta_train = y_class[:meta_train_end]
    X_meta_eval = X_meta[meta_train_end:]
    y_meta_eval = y_class[meta_train_end:]

    clf_model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss"
    )
    clf_model.fit(X_meta_train, y_meta_train)

    clf_pred_all = clf_model.predict(X_meta)
    
    eval_acc = accuracy_score(y_meta_eval, clf_pred_all[meta_train_end:])
    print(f"  Classification Meta-Learner OOF Accuracy: {eval_acc*100:.2f}%")

    return clf_model, clf_pred_all


def run_baselines(
    data: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Dict[str, Any]]:
    """
    Step: Run naive and ARIMA baselines through walk-forward validation.
    """
    prices = data["Close"].values
    dates = pd.to_datetime(data["Date"]).values
    return run_baselines_walkforward(prices, dates, folds)


def generate_confidence_intervals(
    meta_model: XGBRegressor,
    tree_future: np.ndarray,
    lstm_future: np.ndarray,
    tf_future: np.ndarray,
    stock_data: pd.DataFrame,
    residual_std: float,
    future_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate volatility-adjusted confidence intervals for future predictions.

    Returns (hybrid_future, conf_lower, conf_upper, dynamic_std).
    """
    X_meta_future = np.column_stack([tree_future, lstm_future, tf_future])

    vix_current = (
        stock_data["VIX"].iloc[-1] if "VIX" in stock_data.columns else 20.0
    )
    vix_mean = stock_data["VIX"].mean() if "VIX" in stock_data.columns else 20.0

    # Add VIX if meta-model expects it
    if X_meta_future.shape[1] < meta_model.n_features_in_:
        X_meta_future = np.column_stack(
            [X_meta_future, np.full(future_days, vix_current)]
        )

    hybrid_future = meta_model.predict(X_meta_future)

    conf_lower, conf_upper, dynamic_std = compute_volatility_adjusted_ci(
        hybrid_future, residual_std, vix_current, vix_mean, confidence_level=0.95
    )

    return hybrid_future, conf_lower, conf_upper, dynamic_std


def run_pipeline(
    ticker: str = "AAPL",
    start_date: str = "2010-01-01",
    future_days: int = 30,
    optimize: bool = False,
    min_train_size: int = 504,
    test_size: int = 63,
    step_size: int = 63,
) -> Dict[str, Any]:
    """
    Full pipeline: fetch → walk-forward train → meta-ensemble → baselines → report.

    This is the main entry point, importable for FastAPI or notebook use.

    Returns a comprehensive result dict.
    """
    print("=" * 70)
    print(f"  >> HYBRID STOCK PREDICTOR - {ticker}")
    print(f"  Historical data from {start_date} | Predicting {future_days} days ahead")
    print(f"  Optuna Tuning: {'ENABLED' if optimize else 'DISABLED'}")
    print(f"  Validation: Walk-Forward (min_train={min_train_size}, "
          f"test={test_size}, step={step_size})")
    print("=" * 70)

    # 1. Fetch data
    print("\n[1/7] Fetching & engineering features + Macro context...")
    stock_data = fetch_and_prepare_data(ticker, start_date)
    if stock_data is None:
        print("ERROR: Failed to fetch data.")
        return {"error": "data_fetch_failed"}

    # 2. Create walk-forward folds
    folds = create_walk_forward_folds(
        stock_data,
        min_train_size=min_train_size,
        test_size=test_size,
        step_size=step_size,
    )

    if not folds:
        print("ERROR: Not enough data for walk-forward validation. "
              "Try reducing min_train_size or using more historical data.")
        return {"error": "insufficient_data_for_wf"}

    # 3. Train base models
    base_results = train_base_models(stock_data, folds, future_days, optimize)

    # 4. Baselines
    print("\n[5/7] Running Baselines (Naive + ARIMA)...")
    baseline_results = run_baselines(stock_data, folds)

    # 5. Meta-Ensemble
    print("\n[6/7] Building OOF Meta-Ensemble...")
    y_true, tree_p, lstm_p, tf_p, dates = align_test_results(
        base_results["tree"], base_results["lstm"], base_results["transformer"]
    )

    all_metrics = [
        base_results["tree"]["metrics"],
        base_results["lstm"]["metrics"],
        base_results["transformer"]["metrics"],
    ]

    hybrid_test_pred = None
    hybrid_future = None
    conf_lower = None
    conf_upper = None
    meta_model = None
    calibration_report = None
    clf_model = None

    if y_true is not None and len(y_true) > 10:
        # 1. Regression Meta-Learner (Magnitude & CI)
        meta_model, hybrid_test_pred, residual_std = train_meta_ensemble_oof(
            y_true, tree_p, lstm_p, tf_p, stock_data, dates
        )

        hybrid_metrics = compute_metrics(
            y_true, hybrid_test_pred, model_name="Hybrid Meta (Regressor)"
        )
        print_metrics(hybrid_metrics)
        all_metrics.append(hybrid_metrics)

        # 2. Classification Meta-Learner (Direction)
        clf_model, clf_pred_all = train_classification_meta_ensemble_oof(
            y_true, tree_p, lstm_p, tf_p, stock_data, dates
        )
        
        # Calculate Classification Metrics manually since compute_metrics expects prices
        # We align with evaluate.py logic:
        df_prices = stock_data[["Date", "Close"]].copy()
        df_prices["Date"] = _strip_tz(df_prices["Date"])
        df_prices.rename(columns={"Close": "Current_Close"}, inplace=True)
        df_meta = pd.DataFrame({"Date": _strip_tz(dates)})
        df_meta = df_meta.merge(df_prices, on="Date", how="left")
        df_meta["Current_Close"] = df_meta["Current_Close"].ffill().bfill()
        
        y_class_true = (y_true > df_meta["Current_Close"].values).astype(int)
        
        # WE MUST EVALUATE ON THE OOF PORTION ONLY TO AVOID LEAKAGE
        n_clf = len(y_class_true)
        meta_train_end_clf = int(n_clf * 0.8)
        y_eval_true = y_class_true[meta_train_end_clf:]
        pred_eval = clf_pred_all[meta_train_end_clf:]
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        clf_metrics = {
            "Model": "Hybrid Meta (Classifier)",
            "Directional Accuracy (%)": accuracy_score(y_eval_true, pred_eval) * 100,
            "F1 Score": f1_score(y_eval_true, pred_eval, average="macro"),
            "Precision": precision_score(y_eval_true, pred_eval, average="macro", zero_division=0),
            "Recall": recall_score(y_eval_true, pred_eval, average="macro", zero_division=0),
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE (%)": np.nan,
            "R2": np.nan,
            "Max Error": np.nan,
        }
        all_metrics.append(clf_metrics)

        # Future predictions with CI (using Regression model)
        hybrid_future, conf_lower, conf_upper, _ = generate_confidence_intervals(
            meta_model,
            base_results["tree"]["future_predictions"],
            base_results["lstm"]["future_predictions"],
            base_results["transformer"]["future_predictions"],
            stock_data,
            residual_std,
            future_days,
        )

        # Confidence interval calibration on test data
        print("\n  Calibrating confidence intervals on test data...")
        test_lower, test_upper, _ = compute_volatility_adjusted_ci(
            hybrid_test_pred,
            residual_std,
            stock_data["VIX"].iloc[-1] if "VIX" in stock_data.columns else 20.0,
            stock_data["VIX"].mean() if "VIX" in stock_data.columns else 20.0,
            confidence_level=0.95,
        )
        calibration_report = calibrate_confidence_interval(
            y_true, test_lower, test_upper, nominal_level=0.95
        )
        print_calibration_report(calibration_report)
    else:
        # Fallback: simple average
        print("  WARNING: Insufficient overlapping dates for meta-ensemble. "
              "Using simple average.")
        hybrid_future = (
            base_results["tree"]["future_predictions"]
            + base_results["lstm"]["future_predictions"]
            + base_results["transformer"]["future_predictions"]
        ) / 3

    # Add baseline metrics
    all_metrics.append(baseline_results["naive"]["metrics"])
    all_metrics.append(baseline_results["arima"]["metrics"])

    # 6. Output & Visuals
    print("\n[7/7] Generating reports...")
    os.makedirs("data", exist_ok=True)
    comp_df = compare_models(all_metrics, save_path="data/model_comparison.csv")

    # Combined future predictions
    combined_future = pd.DataFrame(
        {
            "date": base_results["tree"]["future_dates"],
            "Predicted Close Tree": base_results["tree"]["future_predictions"],
            "Predicted Close LSTM": base_results["lstm"]["future_predictions"],
            "Predicted Close Transformer": base_results["transformer"]["future_predictions"],
            "Predicted Close Hybrid": hybrid_future
            if hybrid_future is not None
            else np.nan,
        }
    )
    if conf_lower is not None:
        combined_future["Conf Lower"] = conf_lower
        combined_future["Conf Upper"] = conf_upper
    combined_future.to_csv("data/combined_predictions.csv", index=False)

    # Plots
    if y_true is not None:
        preds_dict = {
            "Tree": tree_p,
            "BiLSTM": lstm_p,
            "Transformer": tf_p,
        }
        if hybrid_test_pred is not None:
            preds_dict["Hybrid Meta"] = hybrid_test_pred
        plot_actual_vs_predicted(dates, y_true, preds_dict)
        plot_residuals(y_true, preds_dict)

        # Walk-forward fold visualization
        all_dates = pd.to_datetime(stock_data["Date"]).values
        all_prices = stock_data["Close"].values
        plot_walk_forward_folds(all_dates, all_prices, folds)

        # Confusion matrices
        for name, pred in preds_dict.items():
            cm = compute_metrics(y_true, pred, model_name=name).get("Confusion Matrix")
            if cm is not None:
                plot_confusion_matrix(
                    cm, name, save_path=f"images/confusion_matrix_{name.lower().replace(' ', '_')}.png"
                )

    future_preds_dict = {
        "Tree": base_results["tree"]["future_predictions"],
        "BiLSTM": base_results["lstm"]["future_predictions"],
        "Transformer": base_results["transformer"]["future_predictions"],
    }
    if hybrid_future is not None:
        future_preds_dict["Hybrid Meta"] = hybrid_future
    plot_future_predictions(
        base_results["tree"]["future_dates"], future_preds_dict, conf_lower, conf_upper
    )
    plot_model_comparison_bars(comp_df)
    plot_feature_importance(
        base_results["tree"]["features"],
        base_results["tree"]["feature_importances"],
    )

    # Final summary
    print("\n" + "=" * 70)
    print("  [OK] PIPELINE COMPLETE (Walk-Forward, Classification-Primary)")
    import glob
    img_count = len(glob.glob("images/*.png"))
    print(f"  [OK] Generated {img_count} visualization PNGs in 'images/' directory.")
    print("=" * 70)
    if hybrid_future is not None:
        print(f"\n  [FORECAST] {ticker} - Next {future_days} Business Day Predictions (Hybrid):")
        for i, (d, p) in enumerate(
            zip(base_results["tree"]["future_dates"], hybrid_future)
        ):
            ci = (
                f"  [{conf_lower[i]:.2f} - {conf_upper[i]:.2f}]"
                if conf_lower is not None
                else ""
            )
            print(f"    {d.strftime('%Y-%m-%d')}:  ${p:.2f}{ci}")

    return {
        "stock_data": stock_data,
        "base_results": base_results,
        "baseline_results": baseline_results,
        "meta_model": meta_model,
        "hybrid_test_pred": hybrid_test_pred,
        "hybrid_future": hybrid_future,
        "conf_lower": conf_lower,
        "conf_upper": conf_upper,
        "calibration_report": calibration_report,
        "comparison_df": comp_df,
        "folds": folds,
        "aligned_test": {
            "y_true": y_true,
            "tree_pred": tree_p if y_true is not None else None,
            "lstm_pred": lstm_p if y_true is not None else None,
            "tf_pred": tf_p if y_true is not None else None,
            "dates": dates if y_true is not None else None,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Stock Market Prediction Model - "
        "Walk-Forward Validated, Classification-Primary",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization (safe - never touches test data)",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=504,
        help="Minimum training window size (trading days). Default: 504 (~2 years)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=63,
        help="Test fold size (trading days). Default: 63 (~3 months)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=63,
        help="Step size between folds. Default: 63",
    )
    args = parser.parse_args()

    result = run_pipeline(
        ticker=args.ticker.upper(),
        start_date=args.start,
        future_days=args.days,
        optimize=args.optimize,
        min_train_size=args.min_train,
        test_size=args.test_size,
        step_size=args.step_size,
    )

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
