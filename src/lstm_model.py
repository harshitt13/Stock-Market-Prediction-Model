"""Bidirectional LSTM (PyTorch), forecasting next-day log return.

Ported to the contract in REFACTOR_PLAN.md section 1. Two things changed
beyond swapping the target:

- Windows come from :func:`dataset.build_sequences`, so the target is ``y``
  from the dataset object rather than ``dataset[i + time_step, 0]``, which was
  the *scaled Close column of the feature matrix*. Predictions are therefore
  keyed by the same ``target_date`` as the tree model, and the two can be
  merged on an exact key instead of on an assumption.

- MinMaxScaler became StandardScaler. The scaler was already fitted on
  training folds only, which was correct and is kept; MinMax was still the
  wrong choice, because it maps the training range onto [0, 1] and puts every
  larger test move outside it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import Dataset, build_sequences
from model_utils import (
    apply_feature_scaler,
    assemble_predictions,
    clip_fold,
    default_folds,
    fit_feature_scaler,
    fit_target_scaler,
    fold_predictions,
    scale_targets,
    sequence_demo_forecast,
    unscale_targets,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "BiLSTM"
MODEL_PATH = "models/lstm_model.pt"
DEFAULT_LOOKBACK = 90


class BiLSTMModel(nn.Module):
    """3x BiLSTM(128->64->32) -> Dense(64) -> Dense(32) -> Dense(1)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, lookback, features) -> (batch,) scaled return."""
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


def _train_on_sequences(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 64,
    verbose: bool = True,
) -> BiLSTMModel:
    """Train with early stopping on a time-respecting tail validation split."""
    model = BiLSTMModel(n_features).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
        drop_last=len(X_tr) > batch_size,  # BatchNorm needs more than one row
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
            if len(X_val) > 1:
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


def train_lstm_model(
    dataset: Dataset,
    fold_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = 100,
    patience: int = 15,
    seed: Optional[int] = 42,
    save_model: bool = True,
    verbose: bool = True,
    demo_forecast_days: int = 0,
    demo_history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Walk-forward training of the BiLSTM on log returns.

    Both scalers are fitted on the fold's training rows only. Test windows may
    reach back into training rows for their lookback, which is not leakage:
    those are past observations at the time each forecast is made.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n = len(dataset)
    if fold_indices is None:
        fold_indices = default_folds(n)

    fold_frames: List[pd.DataFrame] = []
    last_model = None
    last_x_scaler = None
    last_y_scaler = None

    for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
        train_idx, test_idx = clip_fold(train_idx, test_idx, n)
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"  LSTM fold {fold_id + 1}: skipped, empty train or test")
            continue

        if verbose:
            print(
                f"  LSTM fold {fold_id + 1}/{len(fold_indices)} - "
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

        model = _train_on_sequences(
            train_seq.X.astype(np.float32),
            scale_targets(y_scaler, train_seq.y).astype(np.float32),
            dataset.n_features,
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
    model.eval()

    def predict_scaled_batch(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.tensor(batch.astype(np.float32)).to(DEVICE)).cpu().numpy()

    return sequence_demo_forecast(
        predict_scaled_batch,
        x_scaler,
        y_scaler,
        dataset.feature_names,
        history,
        lookback,
        horizon,
        label="Predicted Close LSTM",
        output_path="data/future_predictions_lstm.csv",
    )
