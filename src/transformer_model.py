"""
Time-Series Transformer (PyTorch) for next-day close price prediction.

Refactored to support walk-forward validation and Optuna safety (tuning
never touches the outer test fold).
"""

import pandas as pd
import numpy as np
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import optuna
from typing import Dict, Any, List, Tuple, Optional

from evaluate import compute_metrics, print_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "Date"]


def create_sequences(dataset: np.ndarray, time_step: int = 90):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i : i + time_step])
        y.append(dataset[i + time_step, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.2,
    ):
        super().__init__()
        self.input_linear = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = output[:, -1, :]
        return self.regressor(output).squeeze(-1)


def _run_optuna_on_training_data(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    n_features: int,
    n_trials: int = 10,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search ONLY on the training sequences.

    Safety: splits training data into inner-train (80%) and inner-val (20%)
    for each trial.  The outer test fold is never seen.
    """
    print("  Optimizing Transformer hyperparameters with Optuna (safe)...")

    # Inner split (time-respecting)
    inner_split = int(len(X_train_seq) * 0.8)
    X_inner_tr = torch.tensor(X_train_seq[:inner_split]).to(DEVICE)
    y_inner_tr = torch.tensor(y_train_seq[:inner_split]).to(DEVICE)
    X_inner_val = torch.tensor(X_train_seq[inner_split:]).to(DEVICE)
    y_inner_val = torch.tensor(y_train_seq[inner_split:]).to(DEVICE)

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned()
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

        model = TimeSeriesTransformer(
            n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(DEVICE)
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        dataset = TensorDataset(X_inner_tr, y_inner_tr)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        for _ in range(15):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_inner_val), y_inner_val).item()
        return val_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    # Ensure lr is present
    if "lr" not in best:
        best["lr"] = 1e-3
    print(f"  Best params: {best}")
    return best


def _train_transformer_on_sequences(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    params: Dict[str, Any],
    epochs: int = 100,
    patience: int = 15,
) -> TimeSeriesTransformer:
    """Train a Transformer on pre-built sequences with early stopping."""
    model = TimeSeriesTransformer(
        n_features,
        d_model=params.get("d_model", 64),
        nhead=params.get("nhead", 4),
        num_layers=params.get("num_layers", 3),
        dim_feedforward=params.get("dim_feedforward", 128),
        dropout=params.get("dropout", 0.2),
    ).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=64,
        shuffle=False,
    )
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
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            if len(X_val) > 0:
                val_loss = criterion(model(X_val_t), y_val_t).item()
            else:
                val_loss = epoch_loss / max(len(tr_loader), 1)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch+1}/{epochs} - "
                f"Train: {epoch_loss/max(len(tr_loader),1):.6f}, Val: {val_loss:.6f}"
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


def train_transformer_model(
    data: pd.DataFrame,
    future_days: int = 30,
    optimize: bool = False,
    fold_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """
    Train Time-Series Transformer with walk-forward validation.

    Parameters
    ----------
    data : pd.DataFrame
        Full stock data with features.
    future_days : int
        Number of future days to forecast.
    optimize : bool
        Whether to run Optuna hyperparameter search.
    fold_indices : list of (train_idx, test_idx), optional
        Walk-forward folds. If None, single 80/20 split.

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

    if fold_indices is None:
        n = len(df)
        train_size = int(n * 0.8)
        fold_indices = [(np.arange(0, train_size), np.arange(train_size, n))]

    # Default params
    best_params = {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 128,
        "dropout": 0.2,
        "lr": 1e-3,
    }

    fold_results = []
    last_model = None
    last_scaler = None

    for fi, (train_idx, test_idx) in enumerate(fold_indices):
        print(
            f"  Transformer Fold {fi+1}/{len(fold_indices)} - "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[features].iloc[train_idx])
        scaled_all = scaler.transform(df[features])

        fold_start = train_idx[0]
        fold_end = test_idx[-1] + 1
        scaled_fold = scaled_all[fold_start:fold_end]

        X_seq, y_seq = create_sequences(scaled_fold, time_step)
        if len(X_seq) == 0:
            print(f"    Skipping fold {fi+1}: not enough data")
            continue

        seq_orig_idx = np.arange(
            fold_start + time_step, fold_start + time_step + len(X_seq)
        )
        train_end_orig = train_idx[-1] + 1
        seq_train_mask = seq_orig_idx < train_end_orig
        seq_test_mask = ~seq_train_mask

        X_train_seq = X_seq[seq_train_mask]
        y_train_seq = y_seq[seq_train_mask]
        X_test_seq = X_seq[seq_test_mask]
        y_test_seq = y_seq[seq_test_mask]

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"    Skipping fold {fi+1}: insufficient sequences")
            continue

        # Optuna (safe): only on first fold's training data, never on test
        if optimize and fi == 0:
            best_params = _run_optuna_on_training_data(
                X_train_seq, y_train_seq, n_features, n_trials=10
            )

        model = _train_transformer_on_sequences(
            X_train_seq, y_train_seq, n_features, best_params, epochs=100, patience=15
        )

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_seq).to(DEVICE)
            y_pred_scaled = model(X_test_t).cpu().numpy()

        def inverse_close(arr):
            arr = np.asarray(arr, dtype=np.float64).flatten()
            padded = np.column_stack(
                [arr.reshape(-1, 1), np.zeros((len(arr), n_features - 1))]
            )
            return scaler.inverse_transform(padded)[:, 0]

        y_pred_inv = inverse_close(y_pred_scaled)
        y_test_inv = inverse_close(y_test_seq)
        test_seq_dates = dates_raw[seq_orig_idx[seq_test_mask]]

        metrics = compute_metrics(y_test_inv, y_pred_inv, model_name="Transformer")
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
        raise RuntimeError("Transformer: No valid folds produced results.")

    agg = aggregate_fold_results(fold_results)
    agg_metrics = compute_metrics(
        agg["y_true"], agg["y_pred"], model_name="Transformer"
    )
    print_metrics(agg_metrics)

    model_path = "models/transformer_model.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(last_model.state_dict(), model_path)

    # Future predictions
    last_date = pd.to_datetime(data["Date"]).max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_days, freq="B"
    )
    scaled_data = last_scaler.transform(df[features])
    last_sequence = scaled_data[-time_step:].copy()
    current_sequence = last_sequence.reshape(1, time_step, n_features)

    def inverse_close_future(arr):
        arr = np.asarray(arr, dtype=np.float64).flatten()
        padded = np.column_stack(
            [arr.reshape(-1, 1), np.zeros((len(arr), n_features - 1))]
        )
        return last_scaler.inverse_transform(padded)[:, 0]

    future_preds_scaled = []
    last_model.eval()
    for _ in range(future_days):
        with torch.no_grad():
            seq_t = torch.tensor(current_sequence, dtype=torch.float32).to(DEVICE)
            pred_scaled = last_model(seq_t).cpu().numpy()[0]
        future_preds_scaled.append(pred_scaled)
        next_day = current_sequence[0, -1, :].copy()
        next_day[0] = pred_scaled
        current_sequence = np.append(
            current_sequence[:, 1:, :], [[next_day]], axis=1
        ).astype(np.float32)

    future_preds_inv = inverse_close_future(np.array(future_preds_scaled))
    future_df = pd.DataFrame(
        {"date": future_dates, "Predicted Close Transformer": future_preds_inv}
    )
    os.makedirs("data", exist_ok=True)
    future_df.to_csv("data/future_predictions_transformer.csv", index=False)

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
