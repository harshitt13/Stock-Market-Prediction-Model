"""
Bidirectional LSTM (PyTorch) for next-day close price prediction.

Refactored to support walk-forward validation via external fold indices.
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, List, Tuple, Optional

from evaluate import compute_metrics, print_metrics

# ── Device Selection ─────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of feature columns (everything except Date)."""
    return [c for c in df.columns if c != "Date"]


def create_sequences(dataset: np.ndarray, time_step: int = 90):
    """
    Create overlapping sequences for LSTM training.

    Args:
        dataset: 2D numpy array (samples, features). Close must be column 0.
        time_step: Number of lookback days.

    Returns:
        X: 3D array (samples, time_step, features)
        y: 1D array of next-day Close (scaled)
    """
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i : i + time_step])
        y.append(dataset[i + time_step, 0])  # Close is column 0
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── BiLSTM Model ─────────────────────────────────────────────────────────
class BiLSTMModel(nn.Module):
    """
    Deep Bidirectional LSTM with BatchNorm and Dropout for stock price prediction.
    Architecture: 3× BiLSTM(128→64→32) → Dense(64) → Dense(32) → Dense(1)
    """

    def __init__(self, n_features, hidden_sizes=(128, 64, 32), dropout=0.3):
        super().__init__()

        self.lstm1 = nn.LSTM(
            n_features, hidden_sizes[0], batch_first=True, bidirectional=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0] * 2)
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            hidden_sizes[0] * 2, hidden_sizes[1], batch_first=True, bidirectional=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1] * 2)
        self.drop2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(
            hidden_sizes[1] * 2, hidden_sizes[2], batch_first=True, bidirectional=True
        )
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2] * 2)
        self.drop3 = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_sizes[2] * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop1(out)

        out, _ = self.lstm2(out)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop2(out)

        out, _ = self.lstm3(out)
        out = self.bn3(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop3(out)

        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


def _train_lstm_on_sequences(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    epochs: int = 100,
    patience: int = 15,
) -> BiLSTMModel:
    """Train a BiLSTM model on pre-built sequences with early stopping."""
    model = BiLSTMModel(n_features).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Train / val split (time-respecting within fold)
    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    tr_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    tr_loader = DataLoader(tr_dataset, batch_size=64, shuffle=False)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in tr_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            if len(X_val) > 0:
                val_preds = model(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()
            else:
                val_loss = epoch_loss / max(len(tr_loader), 1)

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch+1}/{epochs} - "
                f"Train: {epoch_loss/max(len(tr_loader),1):.6f}, "
                f"Val: {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_lstm_model(
    data: pd.DataFrame,
    future_days: int = 30,
    fold_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """
    Train BiLSTM with walk-forward validation.

    Parameters
    ----------
    data : pd.DataFrame
        Full stock data with engineered features.
    future_days : int
        Number of future days to forecast.
    fold_indices : list of (train_idx, test_idx), optional
        Walk-forward folds on the *raw data* (before sequencing).
        If None, falls back to single 80/20 split.

    Returns
    -------
    dict with aggregated results + future predictions.
    """
    from walk_forward import aggregate_fold_results

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    dates_raw = df["Date"].values

    features = get_feature_columns(data)
    if features[0] != "Close":
        features.remove("Close")
        features.insert(0, "Close")

    n_features = len(features)
    time_step = 90

    # --- Walk-forward or single split ---
    if fold_indices is None:
        n = len(df)
        train_size = int(n * 0.8)
        fold_indices = [(np.arange(0, train_size), np.arange(train_size, n))]

    fold_results = []
    last_model = None
    last_scaler = None

    for fi, (train_idx, test_idx) in enumerate(fold_indices):
        print(
            f"  LSTM Fold {fi+1}/{len(fold_indices)} - "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

        # Fit scaler only on training data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[features].iloc[train_idx])
        scaled_all = scaler.transform(df[features])

        # Build sequences from the combined train+test region
        # but ensure test sequences only use training data in their lookback
        fold_start = train_idx[0]
        fold_end = test_idx[-1] + 1
        scaled_fold = scaled_all[fold_start:fold_end]

        X_seq, y_seq = create_sequences(scaled_fold, time_step)
        if len(X_seq) == 0:
            print(f"    Skipping fold {fi+1}: not enough data for sequences")
            continue

        # Map sequence indices back to original data indices
        # Sequence i corresponds to predicting data point (fold_start + time_step + i)
        seq_orig_idx = np.arange(fold_start + time_step, fold_start + time_step + len(X_seq))

        # Split: sequences whose target falls in train vs test
        train_end_orig = train_idx[-1] + 1
        seq_train_mask = seq_orig_idx < train_end_orig
        seq_test_mask = ~seq_train_mask

        X_train_seq = X_seq[seq_train_mask]
        y_train_seq = y_seq[seq_train_mask]
        X_test_seq = X_seq[seq_test_mask]
        y_test_seq = y_seq[seq_test_mask]

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"    Skipping fold {fi+1}: insufficient train/test sequences")
            continue

        # Train
        model = _train_lstm_on_sequences(
            X_train_seq, y_train_seq, n_features, epochs=100, patience=15
        )

        # Predict
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_seq).to(DEVICE)
            y_pred_scaled = model(X_test_t).cpu().numpy()

        # Inverse transform
        def inverse_close(arr):
            arr = np.asarray(arr, dtype=np.float64).flatten()
            padded = np.column_stack(
                [arr.reshape(-1, 1), np.zeros((len(arr), n_features - 1))]
            )
            return scaler.inverse_transform(padded)[:, 0]

        y_pred_inv = inverse_close(y_pred_scaled)
        y_test_inv = inverse_close(y_test_seq)

        # Dates for test sequences
        test_seq_dates = dates_raw[seq_orig_idx[seq_test_mask]]

        metrics = compute_metrics(y_test_inv, y_pred_inv, model_name="BiLSTM")

        fold_results.append(
            {
                "y_true": y_test_inv,
                "y_pred": y_pred_inv,
                "test_dates": test_seq_dates,
                "metrics": metrics,
            }
        )
        last_model = model
        last_scaler = scaler

    if not fold_results:
        raise RuntimeError("LSTM: No valid folds produced results.")

    # Aggregate
    agg = aggregate_fold_results(fold_results)
    agg_metrics = compute_metrics(agg["y_true"], agg["y_pred"], model_name="BiLSTM")
    print_metrics(agg_metrics)

    # --- Save model ---
    model_path = "models/lstm_model.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": last_model.state_dict(),
            "n_features": n_features,
            "time_step": time_step,
        },
        model_path,
    )
    print(f"Model saved to '{model_path}'")

    from fetch_data import engineer_features

    # --- Future predictions ---
    last_date = pd.to_datetime(data["Date"]).max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_days, freq="B"
    )

    def inverse_close_single(arr):
        arr = np.asarray(arr, dtype=np.float64).flatten()
        padded = np.column_stack(
            [arr.reshape(-1, 1), np.zeros((len(arr), n_features - 1))]
        )
        return last_scaler.inverse_transform(padded)[:, 0]

    future_preds_scaled = []
    
    # Track the raw data to dynamically rebuild features
    history_df = data.copy().iloc[-100:].reset_index(drop=True)

    last_model.eval()
    for i in range(future_days):
        # We need the last `time_step` scaled features
        # Recalculate scaled data for the current history window
        scaled_history = last_scaler.transform(history_df[features])
        current_sequence = scaled_history[-time_step:].reshape(1, time_step, n_features)
        
        with torch.no_grad():
            seq_t = torch.tensor(current_sequence, dtype=torch.float32).to(DEVICE)
            pred_scaled = last_model(seq_t).cpu().numpy()[0]
        future_preds_scaled.append(pred_scaled)
        
        pred_inv = inverse_close_single([pred_scaled])[0]
        
        # Build next day's base row in raw scale
        new_row = history_df.iloc[-1].copy()
        new_row["Date"] = future_dates[i]
        new_row["Close"] = pred_inv
        new_row["High"] = pred_inv
        new_row["Low"] = pred_inv
        
        # Append and re-engineer features
        history_df.loc[len(history_df)] = new_row
        history_df = engineer_features(history_df)

    future_preds_inv = inverse_close_single(np.array(future_preds_scaled))

    future_df = pd.DataFrame(
        {"date": future_dates, "Predicted Close LSTM": future_preds_inv}
    )
    os.makedirs("data", exist_ok=True)
    future_df.to_csv("data/future_predictions_lstm.csv", index=False)
    print("Future predictions saved to 'data/future_predictions_lstm.csv'")

    return {
        "model": last_model,
        "scaler": last_scaler,
        "features": features,
        "y_test": agg["y_true"],
        "y_pred": agg["y_pred"],
        "test_dates": agg["test_dates"],
        "future_dates": future_dates,
        "future_predictions": future_preds_inv,
        "metrics": agg_metrics,
        "per_fold_metrics": agg.get("per_fold_metrics", []),
    }
