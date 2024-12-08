import os
import pandas as pd
from lstm_model import train_lstm_model, forecast_future, load_lstm_model
from linear_regression_model import train_linear_regression, predict_linear_regression
import matplotlib.pyplot as plt

# Define paths for data and models
data_path = "data/AAPL_stock_data.csv"  # Update with your actual path
lstm_model_path = "models/lstm_model.pth"
lstm_scaler_path = "models/lstm_scaler.pkl"
linear_regression_model_path = "models/linear_regression.pkl"
forecast_days = 30  # Number of days to forecast

# Check if models directory exists, if not create it
if not os.path.exists("models"):
    os.makedirs("models")

# Load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}. Please check the file path.")
        return None

# Train LSTM model
def train_lstm_model_wrapper(data):
    model, scaler = train_lstm_model(data, lstm_model_path, lstm_scaler_path)
    return model, scaler

# Train Linear Regression model
def train_linear_regression_wrapper(data):
    model = train_linear_regression(data, linear_regression_model_path)
    return model

# Make predictions using LSTM and Linear Regression models
def make_predictions(data, model, scaler, forecast_days):
    # Prepare data for LSTM prediction (last 60 days)
    close_prices = data["Close"].values
    last_known_data = close_prices[-60:].reshape(1, -1)  # Using last 60 days as input
    
    # LSTM forecast
    lstm_forecast = forecast_future(model, scaler, last_known_data, forecast_days)

    # Prepare data for Linear Regression prediction
    train_features, train_labels = train_linear_regression(data)
    
    # Linear Regression forecast
    linear_regression_forecast = predict_linear_regression(model, train_features)

    return lstm_forecast, linear_regression_forecast

# Plot the results
def plot_results(actual_prices, lstm_forecast, linear_regression_forecast, forecast_days):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(actual_prices)), actual_prices, label="Actual Prices", color="blue")

    # Plot LSTM predictions
    plt.plot(range(len(actual_prices), len(actual_prices) + forecast_days),
        lstm_forecast, label="LSTM Forecast", color="orange")

    # Plot Linear Regression predictions
    plt.plot(range(len(actual_prices), len(actual_prices) + len(linear_regression_forecast)),
        linear_regression_forecast, label="Linear Regression Forecast", color="green")

    plt.axvline(x=len(actual_prices) - 1, linestyle="--", color="gray", label="Forecast Start")
    plt.legend()
    plt.title("Stock Price Prediction with LSTM and Linear Regression")
    plt.xlabel("Time (days)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

# Save forecasted prices to CSV
def save_forecasts(lstm_forecast, linear_regression_forecast, forecast_days):
    # Save LSTM forecasted prices to CSV
    lstm_forecast_df = pd.DataFrame({
        "Day": range(1, forecast_days + 1),
        "LSTM Forecasted Price": lstm_forecast
    })
    lstm_forecast_df.to_csv("../data/lstm_forecasted_prices.csv", index=False)
    print("LSTM forecasted prices saved to 'lstm_forecasted_prices.csv'.")

    # Save Linear Regression forecasted prices to CSV
    linear_forecast_df = pd.DataFrame({
        "Day": range(1, len(linear_regression_forecast) + 1),
        "Linear Regression Forecasted Price": linear_regression_forecast
    })
    linear_forecast_df.to_csv("../data/linear_regression_forecasted_prices.csv", index=False)
    print("Linear Regression forecasted prices saved to 'linear_regression_forecasted_prices.csv'.")

def main():
    # Load and preprocess the dataset
    data = load_data(data_path)
    if data is None:
        return

    # Train models
    print("Training LSTM model...")
    lstm_model, lstm_scaler = train_lstm_model_wrapper(data)

    print("Training Linear Regression model...")
    linear_regression_model = train_linear_regression_wrapper(data)

    # Make predictions
    print("Making predictions...")
    lstm_forecast, linear_regression_forecast = make_predictions(data, lstm_model, lstm_scaler, forecast_days)

    # Plot the results
    print("Plotting the results...")
    plot_results(data['Close'].values, lstm_forecast, linear_regression_forecast, forecast_days)

    # Save the forecasts
    print("Saving forecasted data...")
    save_forecasts(lstm_forecast, linear_regression_forecast, forecast_days)

if __name__ == "__main__":
    main()
