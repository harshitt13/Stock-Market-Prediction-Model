import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the data
data_path = 'data/stock_data.csv'
df = pd.read_csv(data_path)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the dataset
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

# Function to build and train the model
def build_model(units, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model

# Hyperparameter tuning
layers = [50, 100, 150]
epochs = [10, 20, 50]
batch_sizes = [1, 5, 10]

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
            
            # Save the best model
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = (unit, epoch, batch_size)
                print(f"New best model with params: {best_params} and MSE: {best_mse}")

# Save the best model and scaler
if not os.path.exists('models'):
    os.makedirs('models')
best_model.save('models/lstm_model.h5')
np.save('models/scaler.npy', scaler)
print("Best model and scaler saved.")

# Plot predictions
def save_plot(y_test_scaled, predictions):
    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction Using LSTM')
    plt.plot(y_test_scaled, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('images/lstm_predictions.png')
    plt.close()

# Load the model and scaler
def load_lstm_model(model_path, scaler_path):
    model = load_model(model_path)
    scaler = np.load(scaler_path, allow_pickle=True).item()
    return model, scaler

# Predict using the loaded model
def predict_lstm(model, scaler, input_data):
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = np.reshape(input_data_scaled, (1, input_data_scaled.shape[0], 1))
    prediction_scaled = model.predict(input_data_scaled)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction

# Function to forecast future stock prices using the model
def forecast_future(model, scaler, last_known_data, forecast_days):
    """
    Generate future forecasts using the trained LSTM model.
    """
    model.eval()
    forecasted_prices = []
    current_input = np.array(last_known_data)

    for _ in range(forecast_days):
        # Reshape the current input to 2D before transforming
        input_data_scaled = scaler.transform(current_input.reshape(-1, 1))

        # Convert to PyTorch tensor and reshape for LSTM input
        input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).view(1, -1, 1)

        # Generate the prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Inverse transform the scaled prediction
        predicted_price = scaler.inverse_transform(prediction.numpy())[0][0]
        forecasted_prices.append(predicted_price)

        # Update current input by appending the predicted price
        current_input = np.append(current_input[1:], predicted_price)

    return forecasted_prices

# Example of loading the model and making predictions
model_path = 'models/lstm_model.h5'
scaler_path = 'models/scaler.npy'

try:
    # Load the model and scaler
    model, scaler = load_lstm_model(model_path, scaler_path)
    print("Model and scaler loaded successfully.")
    
    # Example prediction
    input_data = np.array([[1500]])  # Replace with actual stock price data
    prediction = predict_lstm(model, scaler, input_data)
    print("Predicted Stock Price:", prediction)
    
    # Forecast future stock prices
    last_known_data = df['Close'].values[-60:].reshape(-1, 1)  # Last 60 days of stock prices
    forecast_days = 10  # Forecast the next 10 days

    forecasted_prices = forecast_future(model, scaler, last_known_data, forecast_days)
    print("Forecasted Stock Prices for the next 10 days:", forecasted_prices)
    
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the paths are correct.")

# Save predictions
predictions = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
save_plot(y_test_scaled, predictions)
