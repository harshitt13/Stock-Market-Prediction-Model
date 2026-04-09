import os
import pandas as pd
from fetch_data import fetch_stock_data
from linear_regression_model import train_linear_regression_model
from lstm_model import train_lstm_model

def combine_predictions():
    # Paths to the prediction CSV files
    lr_predictions_path = 'data/future_predictions_lr.csv'
    lstm_predictions_path = 'data/future_predictions_lstm.csv'

    # Check if both files exist
    if os.path.exists(lr_predictions_path) and os.path.exists(lstm_predictions_path):
        lr_predictions = pd.read_csv(lr_predictions_path, header=0)
        lstm_predictions = pd.read_csv(lstm_predictions_path, header=0)

        # Merge on date
        combined_predictions = pd.merge(lr_predictions, lstm_predictions, how='inner', on='date')
        
        # Add a simple hybrid column
        combined_predictions['Predicted Close Hybrid (Avg)'] = (combined_predictions['Predicted Close LR'] + combined_predictions['Predicted Close LSTM']) / 2.0

        combined_predictions.sort_values(by='date', inplace=True)
        os.makedirs('data', exist_ok=True)
        
        combined_predictions_path = 'data/combined_predictions.csv'
        combined_predictions.to_csv(combined_predictions_path, index=False)
        print(f"Combined predictions saved to '{combined_predictions_path}'")
        print(combined_predictions.head())
    else:
        print("One or both prediction files do not exist.")

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

        # Combine predictions
        print("Combining predictions...")
        combine_predictions()

if __name__ == "__main__":
    main()
