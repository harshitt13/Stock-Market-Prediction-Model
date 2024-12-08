import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

from fetch_data import fetch_stock_data, preprocess_data
from linear_regression_model import load_linear_regression_model, predict_linear_regression
from lstm_model import load_lstm_model, predict_lstm

def prepare_future_input(data, sequence_length=60, future_days=30):
    """
    Prepare input sequences for future prediction
    
    Args:
        data (numpy.ndarray): Historical price data
        sequence_length (int): Number of past time steps
        future_days (int): Number of days to predict
    
    Returns:
        numpy.ndarray: Input sequences for prediction
    """
    # Use the last complete sequence as input
    last_sequence = data[-sequence_length:]
    
    # Reshape for model input
    last_sequence = last_sequence.reshape(1, sequence_length, 1)
    
    return last_sequence

def predict_future_stock_prices(ticker='AAPL', future_days=30):
    """
    Predict future stock prices using multiple models
    
    Args:
        ticker (str): Stock ticker symbol
        future_days (int): Number of days to predict
    
    Returns:
        dict: Predictions from different models
    """
    # Fetch stock data
    stock_data = fetch_stock_data(ticker)
    
    if stock_data is None:
        print("Failed to fetch stock data.")
        return None
    
    # Preprocess data
    processed_data = preprocess_data(stock_data)
    X, y = processed_data['X'], processed_data['y']
    close_scaler = processed_data['scalers']['Close']
    original_data = processed_data['original_data']
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Load pre-trained models
    lr_model = load_linear_regression_model(X_train.shape[1])
    lstm_model = load_lstm_model()
    
    # Prepare future input
    future_input = prepare_future_input(X)
    
    # Predict future prices
    lr_predictions = predict_linear_regression(lr_model, future_input)
    lstm_predictions = predict_lstm(lstm_model, future_input)
    
    # Inverse transform predictions
    lr_future_prices = close_scaler.inverse_transform(lr_predictions)
    lstm_future_prices = close_scaler.inverse_transform(lstm_predictions)
    
    # Generate future dates
    last_date = original_data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
    
    # Visualize predictions
    plt.figure(figsize=(12, 6))
    plt.plot(original_data.index, original_data['Close'], label='Historical Prices')
    
    # Plot Linear Regression predictions
    plt.plot(future_dates, lr_future_prices, 'r--', label='Linear Regression Prediction')
    
    # Plot LSTM predictions
    plt.plot(future_dates, lstm_future_prices, 'g--', label='LSTM Prediction')
    
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/stock_price_predictions.png')
    plt.close()
    
    # Print predictions
    print("\nLinear Regression Future Predictions:")
    for date, price in zip(future_dates, lr_future_prices):
        print(f"{date.date()}: ${price[0]:.2f}")
    
    print("\nLSTM Future Predictions:")
    for date, price in zip(future_dates, lstm_future_prices):
        print(f"{date.date()}: ${price[0]:.2f}")
    
    return {
        'linear_regression': {
            'dates': future_dates,
            'prices': lr_future_prices
        },
        'lstm': {
            'dates': future_dates,
            'prices': lstm_future_prices
        }
    }

def evaluate_models(y_true, lr_predictions, lstm_predictions):
    """
    Evaluate model performance using Mean Absolute Error (MAE)
    
    Args:
        y_true (numpy.ndarray): True target values
        lr_predictions (numpy.ndarray): Linear Regression predictions
        lstm_predictions (numpy.ndarray): LSTM predictions
    
    Returns:
        dict: Model performance metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    lr_mae = mean_absolute_error(y_true, lr_predictions)
    lstm_mae = mean_absolute_error(y_true, lstm_predictions)
    
    lr_mse = mean_squared_error(y_true, lr_predictions)
    lstm_mse = mean_squared_error(y_true, lstm_predictions)
    
    print("\nModel Performance Evaluation:")
    print("Linear Regression:")
    print(f"  Mean Absolute Error: {lr_mae:.4f}")
    print(f"  Mean Squared Error: {lr_mse:.4f}")
    
    print("\nLSTM:")
    print(f"  Mean Absolute Error: {lstm_mae:.4f}")
    print(f"  Mean Squared Error: {lstm_mse:.4f}")
    
    return {
        'linear_regression': {
            'mae': lr_mae,
            'mse': lr_mse
        },
        'lstm': {
            'mae': lstm_mae,
            'mse': lstm_mse
        }
    }

def main(ticker='AAPL', future_days=30):
    """
    Main function to run stock price prediction
    
    Args:
        ticker (str): Stock ticker symbol
        future_days (int): Number of days to predict
    """
    # Load preprocessed data
    X = np.load('data/processed_X.npy')
    y = np.load('data/processed_y.npy')
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Load pre-trained models
    lr_model = load_linear_regression_model(X_train.shape[1])
    lstm_model = load_lstm_model()
    
    # Make predictions
    lr_test_predictions = predict_linear_regression(lr_model, X_test)
    lstm_test_predictions = predict_lstm(lstm_model, X_test)
    
    # Evaluate models
    evaluate_models(y_test, lr_test_predictions, lstm_test_predictions)
    
    # Predict future prices
    future_predictions = predict_future_stock_prices(ticker, future_days)
    
    return future_predictions

if __name__ == "__main__":
    main()