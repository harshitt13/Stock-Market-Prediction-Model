import pandas as pd
import numpy as np
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import optuna

from evaluate import compute_metrics, print_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_feature_columns(df):
    return [c for c in df.columns if c != 'Date']

def create_sequences(dataset, time_step=90):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step])
        y.append(dataset[i + time_step, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model optimally
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_linear = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        
        # Take the output of the last time step for forecasting
        output = output[:, -1, :]
        return self.regressor(output).squeeze()

def train_transformer_model(data, future_days=30, optimize=False):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    features = get_feature_columns(data)
    if features[0] != 'Close':
        features.remove('Close')
        features.insert(0, 'Close')

    train_boundary = int(len(df) * 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[features].iloc[:train_boundary])
    scaled_data = scaler.transform(df[features])

    time_step = 90
    X, y = create_sequences(scaled_data, time_step)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    n_features = len(features)

    best_params = {'d_model': 64, 'nhead': 4, 'num_layers': 3, 'dim_feedforward': 128, 'dropout': 0.2, 'lr': 1e-3}

    if optimize:
        print("Optimizing Transformer hyperparameters with Optuna...")
        def objective(trial):
            d_model = trial.suggest_categorical('d_model', [32, 64, 128])
            nhead = trial.suggest_categorical('nhead', [2, 4, 8])
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dim_feedforward = trial.suggest_categorical('dim_feedforward', [64, 128, 256])
            dropout = trial.suggest_float('dropout', 0.1, 0.4)
            lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)

            model = TimeSeriesTransformer(n_features, d_model=d_model, nhead=nhead, 
                                          num_layers=num_layers, dim_feedforward=dim_feedforward, 
                                          dropout=dropout).to(DEVICE)
            criterion = nn.HuberLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            val_split = int(len(X_train) * 0.8)
            X_tr_t = torch.tensor(X_train[:val_split]).to(DEVICE)
            y_tr_t = torch.tensor(y_train[:val_split]).to(DEVICE)
            X_val_t = torch.tensor(X_train[val_split:]).to(DEVICE)
            y_val_t = torch.tensor(y_train[val_split:]).to(DEVICE)

            dataset = TensorDataset(X_tr_t, y_tr_t)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            for epoch in range(15):  # short training for search
                model.train()
                for bx, by in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)
        best_params = study.best_params
        print(f"Best params found: {best_params}")

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    model = TimeSeriesTransformer(n_features, d_model=best_params['d_model'], nhead=best_params['nhead'], 
                                  num_layers=best_params['num_layers'], dim_feedforward=best_params['dim_feedforward'], 
                                  dropout=best_params['dropout']).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    print(f"Training Time-Series Transformer on {DEVICE}...")

    best_val_loss = float('inf')
    patience, patience_counter = 15, 0
    best_state = None
    epochs = 100

    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=64, shuffle=False)
    X_val_t, y_val_t = torch.tensor(X_val).to(DEVICE), torch.tensor(y_val).to(DEVICE)

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
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
             print(f"  Epoch {epoch+1}/{epochs} — Train Loss: {epoch_loss/len(tr_loader):.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test).to(DEVICE)
        y_pred_scaled = model(X_test_t).cpu().numpy()

    def inverse_close(arr):
        arr = np.asarray(arr, dtype=np.float64).flatten()
        padded = np.column_stack([arr.reshape(-1, 1), np.zeros((len(arr), n_features - 1))])
        return scaler.inverse_transform(padded)[:, 0]

    y_pred_inv = inverse_close(y_pred_scaled)
    y_test_inv = inverse_close(y_test)
    test_dates = df.index[time_step + train_size: time_step + train_size + len(y_test_inv)]

    metrics = compute_metrics(y_test_inv, y_pred_inv, model_name="Transformer")
    print_metrics(metrics)

    model_path = 'models/transformer_model.pt'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')
    last_sequence = scaled_data[-time_step:].copy()
    current_sequence = last_sequence.reshape(1, time_step, n_features)
    future_preds_scaled = []

    model.eval()
    for _ in range(future_days):
        with torch.no_grad():
            seq_t = torch.tensor(current_sequence, dtype=torch.float32).to(DEVICE)
            pred_scaled = model(seq_t).cpu().numpy()
        future_preds_scaled.append(pred_scaled)
        next_day = current_sequence[0, -1, :].copy()
        next_day[0] = pred_scaled
        current_sequence = np.append(current_sequence[:, 1:, :], [[next_day]], axis=1).astype(np.float32)

    future_preds_inv = inverse_close(np.array(future_preds_scaled))
    future_df = pd.DataFrame({'date': future_dates, 'Predicted Close Transformer': future_preds_inv})
    os.makedirs('data', exist_ok=True)
    future_df.to_csv('data/future_predictions_transformer.csv', index=False)

    return {
        'model': model, 'scaler': scaler, 'features': features,
        'y_test': y_test_inv, 'y_pred': y_pred_inv, 'test_dates': test_dates,
        'future_dates': future_dates, 'future_predictions': future_preds_inv, 'metrics': metrics
    }
