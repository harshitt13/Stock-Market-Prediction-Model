import pandas as pd
import matplotlib.pyplot as plt
from lstm_model import load_lstm_model, forecast_future
from linear_regression_model import train_linear_regression, predict_linear_regression

# Load the dataset
data_path = "../data/AAPL.csv"  # Update the path to your dataset if necessary
data = pd.read_csv(data_path)

# Extract the closing prices
close_prices = data["Close"].values

# ---------------- LSTM Model Prediction ---------------- #
# Load the trained LSTM model and scaler
model, scaler = load_lstm_model()

# Define the number of future days to forecast
forecast_days = 30

# Use the last known data for LSTM prediction
last_known_data = close_prices[-60:]  # Using the last 60 days as input

# Generate future predictions using LSTM
lstm_forecasted_prices = forecast_future(model, scaler, last_known_data, forecast_days)

# ---------------- Linear Regression Prediction ---------------- #
# Prepare data for Linear Regression
train_features, train_labels, test_features = train_linear_regression(data)

# Generate future predictions using Linear Regression
linear_regression_forecasted_prices = predict_linear_regression(test_features)

# ---------------- Visualization ---------------- #
# Combine actual and forecasted data for visualization
forecast_start_index = len(close_prices)

plt.figure(figsize=(12, 6))
plt.plot(range(len(close_prices)), close_prices, label="Actual Prices", color="blue")

# Plot LSTM predictions
plt.plot(range(forecast_start_index, forecast_start_index + forecast_days),
    lstm_forecasted_prices, label="LSTM Forecast", color="orange")

# Plot Linear Regression predictions
plt.plot(range(forecast_start_index, forecast_start_index + len(linear_regression_forecasted_prices)),
    linear_regression_forecasted_prices, label="Linear Regression Forecast", color="green")

plt.axvline(x=forecast_start_index - 1, linestyle="--", color="gray", label="Forecast Start")
plt.legend()
plt.title("Stock Price Prediction with LSTM and Linear Regression")
plt.xlabel("Time (days)")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# ---------------- Save Predictions ---------------- #
# Save LSTM predictions to a CSV file
lstm_forecasted_df = pd.DataFrame({
    "Day": range(1, forecast_days + 1),
    "LSTM Forecasted Price": lstm_forecasted_prices
})
lstm_forecasted_df.to_csv("../data/lstm_forecasted_prices.csv", index=False)
print("LSTM forecasted prices saved to 'lstm_forecasted_prices.csv'.")

# Save Linear Regression predictions to a CSV file
linear_forecasted_df = pd.DataFrame({
    "Day": range(1, len(linear_regression_forecasted_prices) + 1),
    "Linear Regression Forecasted Price": linear_regression_forecasted_prices
})
linear_forecasted_df.to_csv("../data/linear_regression_forecasted_prices.csv", index=False)
print("Linear Regression forecasted prices saved to 'linear_regression_forecasted_prices.csv'.")
