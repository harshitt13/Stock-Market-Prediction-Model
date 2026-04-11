import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from evaluate import compute_metrics, print_metrics


# ── Device Selection ─────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_feature_columns(df):
    """Return the list of feature columns (everything except Date)."""
    return [c for c in df.columns if c != 'Date']


def create_sequences(dataset, time_step=90):
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
        X.append(dataset[i:i + time_step])
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

        self.lstm1 = nn.LSTM(n_features, hidden_sizes[0], batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0] * 2)
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_sizes[0] * 2, hidden_sizes[1], batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1] * 2)
        self.drop2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(hidden_sizes[1] * 2, hidden_sizes[2], batch_first=True, bidirectional=True)
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
        # x: (batch, seq_len, features)
        out, _ = self.lstm1(x)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop1(out)

        out, _ = self.lstm2(out)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop2(out)

        out, _ = self.lstm3(out)
        out = self.bn3(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop3(out)

        # Take the last time step
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


def train_lstm_model(data, future_days=30):
    """
    Train a deep Bidirectional LSTM (PyTorch) for next-day close price prediction.

    Returns:
        dict with keys:
            'model': trained PyTorch model
            'scaler': fitted MinMaxScaler
            'features': list of feature column names
            'y_test': actual test values (inverse-scaled)
            'y_pred': predicted test values (inverse-scaled)
            'test_dates': dates for the test period
            'future_dates': predicted future dates
            'future_predictions': predicted future close prices
            'metrics': evaluation metrics dict
    """
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    features = get_feature_columns(data)  # from original (before Date was set as index)

    # Ensure 'Close' is the first feature for target extraction
    if features[0] != 'Close':
        features.remove('Close')
        features.insert(0, 'Close')

    # ── Scaling ──────────────────────────────────────────────────────────
    train_boundary = int(len(df) * 0.8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[features].iloc[:train_boundary])
    scaled_data = scaler.transform(df[features])

    # ── Sequences ────────────────────────────────────────────────────────
    time_step = 90

    X, y = create_sequences(scaled_data, time_step)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    n_features = len(features)

    # DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────
    model = BiLSTMModel(n_features).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # ── Training ─────────────────────────────────────────────────────────
    print(f"Training Bidirectional LSTM on {DEVICE} (this may take a few minutes)...")

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_state = None

    epochs = 100

    # Split training into train/val for early stopping
    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    tr_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    tr_loader = DataLoader(tr_dataset, batch_size=64, shuffle=False)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)

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

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Train Loss: {epoch_loss/len(tr_loader):.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss: {best_val_loss:.6f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Predictions ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test).to(DEVICE)
        y_pred_scaled = model(X_test_t).cpu().numpy()

    # Inverse transform: pad with zeros for non-Close features
    def inverse_close(arr):
        arr = np.asarray(arr, dtype=np.float64).flatten()
        padded = np.column_stack([arr.reshape(-1, 1), np.zeros((len(arr), n_features - 1))])
        return scaler.inverse_transform(padded)[:, 0]

    y_pred_inv = inverse_close(y_pred_scaled)
    y_test_inv = inverse_close(y_test)

    # Test dates
    test_dates = df.index[time_step + train_size: time_step + train_size + len(y_test_inv)]

    # ── Evaluate ─────────────────────────────────────────────────────────
    metrics = compute_metrics(y_test_inv, y_pred_inv, model_name="BiLSTM")
    print_metrics(metrics)

    # ── Save model ───────────────────────────────────────────────────────
    model_path = 'models/lstm_model.pt'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': n_features,
        'time_step': time_step,
    }, model_path)
    print(f"Model saved to '{model_path}'")

    # ── Future predictions (iterative walk-forward) ──────────────────────
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

    last_sequence = scaled_data[-time_step:].copy()
    current_sequence = last_sequence.reshape(1, time_step, n_features)

    future_preds_scaled = []
    model.eval()
    for _ in range(future_days):
        with torch.no_grad():
            seq_t = torch.tensor(current_sequence, dtype=torch.float32).to(DEVICE)
            pred_scaled = model(seq_t).cpu().numpy()[0]
        future_preds_scaled.append(pred_scaled)

        # Build next day: copy last day's features, update Close
        next_day = current_sequence[0, -1, :].copy()
        next_day[0] = pred_scaled  # Close is index 0

        # Slide window forward
        current_sequence = np.append(current_sequence[:, 1:, :], [[next_day]], axis=1).astype(np.float32)

    future_preds_inv = inverse_close(np.array(future_preds_scaled))

    # Save future predictions
    future_df = pd.DataFrame({
        'date': future_dates,
        'Predicted Close LSTM': future_preds_inv,
    })
    os.makedirs('data', exist_ok=True)
    future_df.to_csv('data/future_predictions_lstm.csv', index=False)
    print("Future predictions saved to 'data/future_predictions_lstm.csv'")

    return {
        'model': model,
        'scaler': scaler,
        'features': features,
        'y_test': y_test_inv,
        'y_pred': y_pred_inv,
        'test_dates': test_dates,
        'future_dates': future_dates,
        'future_predictions': future_preds_inv,
        'metrics': metrics,
    }
