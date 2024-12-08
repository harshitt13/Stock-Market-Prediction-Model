import pandas as pd
import matplotlib.pyplot as plt
from lstm_model import load_lstm_model, forecast_future
from linear_regression_model import train_linear_regression, predict_linear_regression

# Define paths
data_path = "../data/AAPL.csv"
lstm_model_path = "../models/lstm_model.pth"
lstm_scaler_path = "../models/lstm_scaler.pkl"
linear_model_path = "../models/linear_regression.pkl"

# Load the dataset
try:
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
except FileNotFoundError:
    print(f"Error: Dataset file not found at {data_path}. Please check the file path.")
    exit()

# Extract the closing prices
close_prices = data["Close"].values

# ---------------- LSTM Model Prediction ---------------- #
try:
    # Load the trained LSTM model and scaler
    input_size = 5  # Example number of features
    hidden_size = 50
    num_layers = 2
    output_size = 1

    model, scaler = load_lstm_model(
        lstm_model_path, lstm_scaler_path, input_size, hidden_size, num_layers, output_size
    )

    # Define the number of future days to forecast
    forecast_days = 30

    # Use the last 60 days as input for LSTM
    last_known_data = data.iloc[-60:, 1:].values  # Adjust based on your features
    if last_known_data.shape[0] < 60:
        raise ValueError("Insufficient data for LSTM input. Ensure at least 60 days of historical data.")

    # Generate future predictions using LSTM
    lstm_forecasted_prices = forecast_future(model, scaler, last_known_data, forecast_days)
except Exception as e:
    print(f"Error during LSTM prediction: {e}")
    lstm_forecasted_prices = []

# ---------------- Linear Regression Prediction ---------------- #
try:
    # Train the linear regression model and save it
    linear_model = train_linear_regression(data_path, linear_model_path)

    # Use the trained model for prediction
    X, _ = data.iloc[-forecast_days:, 1:].values, data["Close"].iloc[-forecast_days:].values
    linear_regression_forecasted_prices = predict_linear_regression(linear_model, X)
except Exception as e:
    print(f"Error during Linear Regression prediction: {e}")
    linear_regression_forecasted_prices = []

# ---------------- Visualization ---------------- #
try:
    # Combine actual and forecasted data for visualization
    forecast_start_index = len(close_prices)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(close_prices)), close_prices, label="Actual Prices", color="blue")

    if lstm_forecasted_prices:
        plt.plot(
            range(forecast_start_index, forecast_start_index + forecast_days),
            lstm_forecasted_prices,
            label="LSTM Forecast",
            color="orange"
        )

    if linear_regression_forecasted_prices:
        plt.plot(
            range(forecast_start_index, forecast_start_index + len(linear_regression_forecasted_prices)),
            linear_regression_forecasted_prices,
            label="Linear Regression Forecast",
            color="green"
        )

    plt.axvline(x=forecast_start_index - 1, linestyle="--", color="gray", label="Forecast Start")
    plt.legend()
    plt.title("Stock Price Prediction with LSTM and Linear Regression")
    plt.xlabel("Time (days)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error during visualization: {e}")

# ---------------- Save Predictions ---------------- #
try:
    # Save LSTM predictions to a CSV file
    if lstm_forecasted_prices:
        lstm_forecasted_df = pd.DataFrame({
            "Day": range(1, forecast_days + 1),
            "LSTM Forecasted Price": lstm_forecasted_prices
        })
        lstm_forecasted_df.to_csv("../data/lstm_forecasted_prices.csv", index=False)
        print("LSTM forecasted prices saved to 'lstm_forecasted_prices.csv'.")

    # Save Linear Regression predictions to a CSV file
    if linear_regression_forecasted_prices:
        linear_forecasted_df = pd.DataFrame({
            "Day": range(1, len(linear_regression_forecasted_prices) + 1),
            "Linear Regression Forecasted Price": linear_regression_forecasted_prices
        })
        linear_forecasted_df.to_csv("../data/linear_regression_forecasted_prices.csv", index=False)
        print("Linear Regression forecasted prices saved to 'linear_regression_forecasted_prices.csv'.")
except Exception as e:
    print(f"Error saving predictions: {e}")
