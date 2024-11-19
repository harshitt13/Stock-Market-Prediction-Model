import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import os

class AdvancedRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32]):
        super(AdvancedRegressionModel, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Dynamic hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def preprocess_data(data):
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    
    # Create lag features
    for col in data.select_dtypes(include=[np.number]).columns:
        for lag in [1, 2, 3]:
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    # Create rolling window features
    for col in data.select_dtypes(include=[np.number]).columns:
        data[f'{col}_rolling_mean_5'] = data[col].rolling(window=5).mean()
        data[f'{col}_rolling_std_5'] = data[col].rolling(window=5).std()
    
    # Drop initial rows with NaN from lag and rolling features
    data.dropna(inplace=True)
    
    return data

def feature_selection(X, y, top_k=10):
    # Use mutual information for feature selection
    mi_scores = mutual_info_regression(X, y)
    top_feature_indices = mi_scores.argsort()[-top_k:][::-1]
    return X.iloc[:, top_feature_indices]

def train_advanced_regression(file_path, save_path, epochs=2000, learning_rate=0.001):
    # Load and preprocess data
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
    
    # Enhanced preprocessing
    processed_data = preprocess_data(data)
    
    # Prepare features and target
    X = processed_data.drop(columns=['Close', 'Date'])
    y = processed_data['Close']
    
    # Feature selection
    X = feature_selection(X, y)
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # PyTorch setup
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    # Model initialization
    input_dim = X_tensor.shape[1]
    model = AdvancedRegressionModel(input_dim)
    criterion = nn.HuberLoss()  # More robust to outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
    
    # Training loop with cross-validation approach
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Learning rate adjustment
        scheduler.step(loss)
        
        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)
        
        # Periodic loss reporting
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    
    print(f'Best Model Loss: {best_loss}')
    return model

if __name__ == "__main__":
    train_advanced_regression("data/stock_data.csv", "models/linear_regression_model.pkl")