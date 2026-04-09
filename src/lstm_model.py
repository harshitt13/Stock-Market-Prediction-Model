import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm_model(data):
    """
    Train and evaluate an LSTM model on the given stock data.
    """
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Select features
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'Log_Return']
    
    # Chronological Split for Scaling boundary
    train_size_raw = int(len(df) * 0.8)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler strictly on training boundary to prevent leakage
    scaler.fit(df[features].iloc[:train_size_raw])
    scaled_data = scaler.transform(df[features])

    # Prepare the data for LSTM (sequence prediction)
    # We use past 60 days (time_step) to predict the 61st day's Close
    time_step = 60
    
    def create_dataset(dataset, time_step=1):
        X, y = [], []
        # Target is dataset[i + time_step, 0] assuming 'Close' is at index 0
        for i in range(len(dataset) - time_step):
            a = dataset[i:(i + time_step), :]
            X.append(a)
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, time_step)

    # Split the sequence data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    print("Training LSTM... this may take a moment.")
    model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=30, callbacks=[early_stop], verbose=1)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actuals
    # We need to pad the predictions with zeros for the other features to inverse transform correctly
    y_pred_padded = np.concatenate((y_pred, np.zeros((y_pred.shape[0], len(features) - 1))), axis=1)
    y_pred_inv = scaler.inverse_transform(y_pred_padded)[:, 0]
    
    y_test_padded = np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features) - 1))), axis=1)
    y_test_inv = scaler.inverse_transform(y_test_padded)[:, 0]

    # Evaluation
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    print(f'LSTM - Test MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

    # Plot the results
    os.makedirs('images', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Actual Close Prices')
    plt.plot(df.index[-len(y_pred_inv):], y_pred_inv, label='Predicted Close Prices', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices using LSTM')
    plt.legend()
    plt.savefig('images/lstm_actual_vs_predicted.png')
    # plt.show() # Disabled to prevent blocking terminal

    # Save the trained model
    model_path = 'models/lstm_model.keras'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # Predict future 30 business days iteratively
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
    
    # Grab the last 60 days of scaled data
    last_60_days = scaled_data[-time_step:]
    current_sequence = last_60_days.copy().reshape(1, time_step, len(features))
    
    future_predictions_scaled = []
    for _ in range(30):
        # Predict the next day
        pred_scaled = model.predict(current_sequence, verbose=0)[0][0]
        future_predictions_scaled.append(pred_scaled)
        
        # We assume other features remain the same as the last known day, except the predicted 'Close'
        next_day_features = current_sequence[0, -1, :].copy()
        next_day_features[0] = pred_scaled  # Index 0 is 'Close'
        
        # Append next day to the sequence, remove the oldest day
        current_sequence = np.append(current_sequence[:, 1:, :], [[next_day_features]], axis=1)

    future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
    
    # Inverse transform future predictions
    future_preds_padded = np.concatenate((future_predictions_scaled, np.zeros((future_predictions_scaled.shape[0], len(features) - 1))), axis=1)
    future_preds_inv = scaler.inverse_transform(future_preds_padded)[:, 0]

    # Save future predictions to a CSV file
    future_df = pd.DataFrame({
        'date': future_dates,
        'Predicted Close LSTM': future_preds_inv
    })
    
    os.makedirs('data', exist_ok=True)
    future_df.to_csv('data/future_predictions_lstm.csv', index=False)

    print("Future stock price predictions saved to 'data/future_predictions_lstm.csv'")
