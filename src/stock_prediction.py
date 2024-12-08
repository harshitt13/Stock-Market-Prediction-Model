import os
from fetch_data import fetch_stock_data
from linear_regression_model import train_linear_regression_model
from lstm_model import train_lstm_model

def main():
    # Parameters
    ticker_symbol = 'AAPL' #input("Enter a Stock Ticker Symbol (e.g., AAPl): ")  # Example: Apple Inc.
    start_date = '2010-01-01' #input("Enter the start date (YYYY-MM-DD): ")  # Start date
    end_date = None  # Use None for the current date (optional)

    # Fetch stock data
    stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)
    
    if stock_data is not None:
        # Train and evaluate Linear Regression model
        print("Training and evaluating Linear Regression model...")
        train_linear_regression_model(stock_data)

        # Train and evaluate LSTM model
        print("Training and evaluating LSTM model...")
        train_lstm_model(stock_data)

if __name__ == "__main__":
    main()
