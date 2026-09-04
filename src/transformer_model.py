"""Time-Series Transformer (PyTorch), forecasting next-day log return.

Ported to the contract in REFACTOR_PLAN.md section 1, with the same two fixes
as the LSTM: windows and targets come from :func:`dataset.build_sequences`
rather than from column 0 of the scaled feature matrix, and the scaler is a
StandardScaler fitted on the fold's training rows.

The Optuna search was already safe -- it splits the *training* sequences into
an inner train/validation pair and never sees the outer test fold -- and that
property is preserved here.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import Dataset, build_sequences
from model_utils import (
    apply_feature_scaler,
    assemble_predictions,
    default_folds,
    fit_feature_scaler,
    fit_target_scaler,
    fold_predictions,
    recursive_demo_forecast,
    scale_targets,
    unscale_targets,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "Transformer"
MODEL_PATH = "models/transformer_model.pt"
DEFAULT_LOOKBACK = 90

DEFAULT_PARAMS: Dict[str, Any] = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 128,
    "dropout": 0.2,
    "lr": 1e-3,
}


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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


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
        x = self.pos_encoder(self.input_linear(x))
        output = self.transformer_encoder(x)[:, -1, :]
        return self.regressor(output).squeeze(-1)


def _run_optuna_on_training_data(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    n_features: int,
    n_trials: int = 10,
    inner_epochs: int = 15,
) -> Dict[str, Any]:
    """Hyperparameter search on the training sequences only.

    The training windows are split 80/20 in time for each trial. The outer
    test fold is never constructed here, let alone scored.
    """
    print("  Optimizing Transformer hyperparameters with Optuna (train-only)...")

    inner_split = max(1, int(len(X_train_seq) * 0.8))
    X_inner_tr = torch.tensor(X_train_seq[:inner_split]).to(DEVICE)
    y_inner_tr = torch.tensor(y_train_seq[:inner_split]).to(DEVICE)
    X_inner_val = torch.tensor(X_train_seq[inner_split:]).to(DEVICE)
    y_inner_val = torch.tensor(y_train_seq[inner_split:]).to(DEVICE)

    if len(X_inner_val) == 0:
        print("  Not enough training windows to tune; using defaults.")
        return dict(DEFAULT_PARAMS)

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned()
        params = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dim_feedforward": trial.suggest_categorical(
                "dim_feedforward", [64, 128, 256]
            ),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        }
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

        model = TimeSeriesTransformer(n_features, **params).to(DEVICE)
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(
            TensorDataset(X_inner_tr, y_inner_tr), batch_size=64, shuffle=False
        )

        for _ in range(inner_epochs):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            return criterion(model(X_inner_val), y_inner_val).item()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = {**DEFAULT_PARAMS, **study.best_params}
    print(f"  Best params: {best}")
    return best


def _train_on_sequences(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    params: Dict[str, Any],
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 64,
    verbose: bool = True,
) -> TimeSeriesTransformer:
    """Train with early stopping on a time-respecting tail validation split."""
    architecture = {k: v for k, v in params.items() if k != "lr"}
    model = TimeSeriesTransformer(n_features, **architecture).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    val_split = max(1, int(len(X_train) * 0.9))
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size,
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
        n_batches = 0
        for batch_X, batch_y in tr_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            if len(X_val) > 0:
                val_loss = criterion(model(X_val_t), y_val_t).item()
            else:
                val_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(val_loss)

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(
                f"    Epoch {epoch + 1}/{epochs} - "
                f"Train: {epoch_loss / max(n_batches, 1):.6f}, Val: {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_transformer_model(
    dataset: Dataset,
    fold_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = 100,
    patience: int = 15,
    optimize: bool = False,
    n_trials: int = 10,
    seed: Optional[int] = 42,
    save_model: bool = True,
    verbose: bool = True,
    demo_forecast_days: int = 0,
    demo_history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Walk-forward training of the Transformer on log returns."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n = len(dataset)
    if fold_indices is None:
        fold_indices = default_folds(n)

    params = dict(DEFAULT_PARAMS)
    fold_frames: List[pd.DataFrame] = []
    last_model = None
    last_x_scaler = None
    last_y_scaler = None

    for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
        train_idx = np.asarray(train_idx)[np.asarray(train_idx) < n]
        test_idx = np.asarray(test_idx)[np.asarray(test_idx) < n]
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"  Transformer fold {fold_id + 1}: skipped, empty train or test")
            continue

        if verbose:
            print(
                f"  Transformer fold {fold_id + 1}/{len(fold_indices)} - "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )

        x_scaler = fit_feature_scaler(dataset, train_idx)
        y_scaler = fit_target_scaler(dataset.y[train_idx])
        scaled = apply_feature_scaler(dataset, x_scaler)

        try:
            train_seq = build_sequences(scaled, lookback, train_idx)
            test_seq = build_sequences(scaled, lookback, test_idx)
        except ValueError as exc:
            print(f"    Skipping fold {fold_id + 1}: {exc}")
            continue

        if len(train_seq) < 2:
            print(f"    Skipping fold {fold_id + 1}: too few training windows")
            continue

        X_train = train_seq.X.astype(np.float32)
        y_train = scale_targets(y_scaler, train_seq.y).astype(np.float32)

        if optimize and not fold_frames:
            params = _run_optuna_on_training_data(
                X_train, y_train, dataset.n_features, n_trials=n_trials
            )

        model = _train_on_sequences(
            X_train,
            y_train,
            dataset.n_features,
            params,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
        )

        model.eval()
        with torch.no_grad():
            scaled_pred = (
                model(torch.tensor(test_seq.X.astype(np.float32)).to(DEVICE))
                .cpu()
                .numpy()
            )

        fold_frames.append(
            fold_predictions(
                dataset,
                fold_id,
                test_seq.row_index,
                unscale_targets(y_scaler, scaled_pred),
            )
        )
        last_model, last_x_scaler, last_y_scaler = model, x_scaler, y_scaler

    predictions = assemble_predictions(fold_frames, MODEL_NAME)

    if save_model and last_model is not None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(
            {
                "model_state_dict": last_model.state_dict(),
                "n_features": dataset.n_features,
                "lookback": lookback,
                "hyperparams": params,
                "feature_names": dataset.feature_names,
            },
            MODEL_PATH,
        )
        print(f"Model saved to '{MODEL_PATH}'")

    output: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "predictions": predictions,
        "model": last_model,
        "x_scaler": last_x_scaler,
        "y_scaler": last_y_scaler,
        "lookback": lookback,
        "hyperparams": params,
        "feature_names": list(dataset.feature_names),
    }

    if demo_forecast_days > 0:
        output["demo_forecast"] = _demo_forecast(
            last_model,
            last_x_scaler,
            last_y_scaler,
            dataset,
            demo_history,
            lookback,
            demo_forecast_days,
        )

    return output


def _demo_forecast(
    model, x_scaler, y_scaler, dataset: Dataset, history, lookback: int, horizon: int
) -> pd.DataFrame:
    """Recursive forecast for display. Never enters a metrics table."""
    if history is None:
        raise ValueError("demo_forecast_days requires demo_history (the raw frame)")

    feature_names = list(dataset.feature_names)
    model.eval()

    def predict_next_return(engineered: pd.DataFrame) -> float:
        window = engineered[feature_names].to_numpy(dtype=float)[-lookback:]
        if len(window) < lookback:
            raise ValueError(
                f"demo forecast needs {lookback} engineered rows, got {len(window)}"
            )
        batch = x_scaler.transform(window).reshape(1, lookback, len(feature_names))
        with torch.no_grad():
            scaled = model(torch.tensor(batch.astype(np.float32)).to(DEVICE)).cpu().numpy()
        return float(unscale_targets(y_scaler, scaled)[0])

    forecast = recursive_demo_forecast(
        history, predict_next_return, horizon, label="Predicted Close Transformer"
    )
    os.makedirs("data", exist_ok=True)
    forecast.to_csv("data/future_predictions_transformer.csv", index=False)
    return forecast
