import os
import torch
import pandas as pd
from lstm_model import predict_lstm, load_lstm_model
from fetch_data import fetch_stock_data
from sklearn.linear_model import LinearRegression
import numpy as np

def load_historical_stock_data(ticker_symbol, start_date, end_date=None):
    """
    Wrapper to fetch and load stock data into a DataFrame.
    
    Args:
        ticker_symbol (str): Stock ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
    
    Returns:
        pd.DataFrame: A DataFrame with historical stock data and additional metrics.
    """
    # Fetch stock data
    fetch_stock_data(ticker_symbol, start_date, end_date)
    
    # Load the saved CSV file
    data_file_path = os.path.join('data', 'stock_data.csv')
    if os.path.exists(data_file_path):
        stock_data = pd.read_csv(data_file_path)
        print("Stock data loaded successfully.")
        return stock_data
    else:
        raise FileNotFoundError(f"Stock data file not found at {data_file_path}")

def preprocess_data(stock_data):
    """
    Preprocess the stock data for model input.
    
    Args:
        stock_data (pd.DataFrame): Raw stock data.
    
    Returns:
        np.array: Processed data for Linear Regression.
        torch.Tensor: Processed data for LSTM.
    """
    # Select relevant features (e.g., Close price, Revenue, EPS, etc.)
    features = ['Close', 'Revenue', 'EPS', 'ROE', 'P/E']
    for feature in features:
        if feature not in stock_data.columns:
            raise ValueError(f"Feature {feature} is missing in the stock data.")
    
    processed_data = stock_data[features].fillna(0)  # Replace missing values with 0

    # Split data into Linear Regression (NumPy) and LSTM (PyTorch)
    lr_data = processed_data.values
    lstm_data = torch.tensor(processed_data.values, dtype=torch.float32)
    return lr_data, lstm_data

def predict_with_linear_regression(stock_data):
    """
    Use Linear Regression to predict stock prices.
    
    Args:
        stock_data (pd.DataFrame): Historical stock data.
    
    Returns:
        np.array: Predicted stock prices.
    """
    # Extract features and target variable
    features = ['Revenue', 'EPS', 'ROE', 'P/E']  # Independent variables
    target = 'Close'  # Dependent variable

    if target not in stock_data.columns:
        raise ValueError(f"Target column {target} is missing in the stock data.")

    X = stock_data[features].fillna(0).values  # Replace NaN values with 0
    y = stock_data[target].fillna(0).values

    # Train-test split (last 20% for testing)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize and train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions on the entire dataset
    predictions = lr_model.predict(X)
    print("Linear Regression model trained successfully.")
    return predictions

def main():
    """
    Main function to load data, preprocess it, and predict stock prices using Linear Regression and LSTM models.
    """
    # Define parameters
    ticker_symbol = 'AAPL'  # Example: Apple Inc.
    start_date = '2014-01-01'
    end_date = '2024-01-01'  # Optional: Leave as None to use the current date
    lstm_model_path = 'models/lstm_model.h5'  # Path to the saved LSTM model

    try:
        # Step 1: Load historical stock data
        stock_data = load_historical_stock_data(ticker_symbol, start_date, end_date)

        # Step 2: Preprocess the data
        lr_data, lstm_data = preprocess_data(stock_data)

        # Step 3: Predict with Linear Regression
        lr_predictions = predict_with_linear_regression(stock_data)
        stock_data['Linear Regression Predicted Price'] = lr_predictions

        # Step 4: Load the trained LSTM model
        lstm_model = load_lstm_model(lstm_model_path)

        # Step 5: Predict with LSTM
        lstm_predictions = predict_lstm(lstm_model, lstm_data)
        stock_data['LSTM Predicted Price'] = lstm_predictions.detach().numpy()

        # Display results
        print(stock_data[['Date', 'Close', 'Linear Regression Predicted Price', 'LSTM Predicted Price']])

        # Save the predictions to a CSV file
        output_file = os.path.join('data', 'predicted_stock_data.csv')
        stock_data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
