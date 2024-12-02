import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Hyperparameters
layers = [50, 100, 150]  # Number of units in LSTM layers
epochs = [10, 20, 50]  # Number of epochs for training
batch_sizes = [1, 5, 10]  # Batch sizes to test

# Load the data
data_path = 'data/stock_data.csv'
df = pd.read_csv(data_path)

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df['Close'].values
data = data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Function to build model
def build_model(units, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model

# Backtracking approach to test different hyperparameters
best_model = None
best_mse = float('inf')
best_params = None

for unit in layers:
    for epoch in epochs:
        for batch_size in batch_sizes:
            model = build_model(unit, epoch, batch_size)
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            
            # Reshape y_test for inverse transformation
            y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate MSE
            mse = mean_squared_error(y_test_scaled, predictions)
            
            # If the current model performs better, save it
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = (unit, epoch, batch_size)
                print(f"New best model with params: {best_params} and MSE: {best_mse}")

def save_plot(y_test_scaled, predictions):
    # Plot Stock Price Prediction Using LSTM
    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction Using LSTM')
    plt.plot(df.index[-len(y_test_scaled):], y_test_scaled, label='Actual Price')
    plt.plot(df.index[-len(y_test_scaled):], predictions, label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Save the plot
    plt.savefig('images/lstm_predictions.png')
    plt.close()

# Save the best model
best_model.save('models/lstm_model.h5')
print("Best model saved.")
