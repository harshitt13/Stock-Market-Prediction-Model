import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

def train_linear_regression_model(data):
    """
    Train and evaluate a linear regression model on the given stock data.
    """
    df = data.copy()
    
    # Select features
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'Log_Return']
    
    # We want to predict the NEXT day's Close, so we shift target backwards
    df['Target_Close'] = df['Close'].shift(-1)
    
    # Drop rows without a target (the last row)
    df.dropna(subset=['Target_Close'], inplace=True)
    
    X = df[features]
    y = df['Target_Close']

    # Chronological Split (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on Test Set
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Linear Regression - Test MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

    # Plot
    os.makedirs('images', exist_ok=True)
    plt.figure(figsize=(10, 6))
    # Plot sequence rather than scatter
    plt.plot(df['Date'].iloc[train_size:].values, y_test.values, label='Actual Next Day Close')
    plt.plot(df['Date'].iloc[train_size:].values, y_pred, label='Predicted Close', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices using Linear Regression')
    plt.legend()
    plt.savefig('images/lr_actual_vs_predicted.png')
    # plt.show() # Disabled to prevent blocking terminal

    # Save
    model_path = 'models/linear_regression_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Predict Future 30 business days Walk-Forward
    last_date = pd.to_datetime(data['Date']).max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
    
    current_features = data[features].iloc[-1].copy()
    future_predictions = []
    
    for _ in range(30):
        # Predict next day
        pred = model.predict(current_features.values.reshape(1, -1))[0]
        future_predictions.append(pred)
        
        # Simple walk-forward logic: assume all other features remain constant or change slightly 
        # For simplicity, we only update 'Close' to the new prediction
        current_features['Close'] = pred

    future_df = pd.DataFrame({
        'date': future_dates,
        'Predicted Close LR': future_predictions
    })
    
    # Save future predictions
    os.makedirs('data', exist_ok=True)
    future_df.to_csv('data/future_predictions_lr.csv', index=False)
    
    print("Future stock price predictions saved to 'data/future_predictions_lr.csv'")
