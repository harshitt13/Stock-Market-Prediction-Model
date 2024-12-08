import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import matplotlib.pyplot as plt
from datetime import timedelta

# Load the dataset
data = pd.read_csv('data/AAPL_stock_data.csv')
if data.empty:
    raise FileNotFoundError("The file 'AAPL_stock_prediction.csv' was not found or is empty.")

# Select features
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'EPS', 'Revenue', 'ROE', 'P/E']
data = data[features]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predicting the 'Close' price
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], len(features) - 1))), axis=1))[:, 0]

# Future prediction
future_days = 30
last_sequence = scaled_data[-seq_length:]
future_predictions = []

for _ in range(future_days):
    next_pred = model.predict(last_sequence[np.newaxis, :, :])[0, 0]
    future_predictions.append(next_pred)
    next_sequence = np.append(last_sequence[1:], [[next_pred] + [0] * (len(features) - 1)], axis=0)

future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((future_days, len(features) - 1))), axis=1))[:, 0]

# Extend the date range for future predictions
last_date = pd.to_datetime(data.index[-1])
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

# Save future predictions to CSV
future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions})
future_predictions_df.to_csv('data/future_predictions.csv', index=False)

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size + seq_length:], data['Close'][train_size + seq_length:], color='blue', label='Actual Stock Price')
plt.plot(data.index[train_size + seq_length:], predictions, color='red', label='Predicted Stock Price')
plt.plot(future_dates, future_predictions, color='green', label='Future Predictions')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# Save the plot
if not os.path.exists('image'):
    os.makedirs('image')
plt.savefig('image/AAPL_stock_price_prediction.png')
plt.show()