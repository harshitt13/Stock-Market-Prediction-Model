# Description: Main script to train and evaluate Linear Regression and LSTM models for stock price prediction.

# Libraries Used
import os
import pandas as pd
import torch
import joblib
from src.fetch_data import fetch_and_save_stock_data
from src.linear_regression_model import LinearRegressionModel
from src.lstm_model import LSTMModel
from src.stock_prediction import evaluate_models

# Constants
TICKER = 'AAPL'  # Stock ticker symbol
START_DATE = '2020-01-01'  # Start date for fetching stock data
END_DATE = '2023-01-01'  # End date for fetching stock data
DATA_PATH = 'data/stock_data.csv'  # Path to save fetched stock data
LR_MODEL_PATH = 'models/linear_regression_model.pkl'  # Path to save/load Linear Regression model
LSTM_MODEL_PATH = 'models/lstm_model.pth'  # Path to save/load LSTM model

def fetch_data_if_needed() -> None:
# Fetch stock data if it does not already exist.
    if not os.path.exists(DATA_PATH):
        print(f"Fetching data for {TICKER}...")
        fetch_and_save_stock_data(TICKER, START_DATE, END_DATE)

def load_stock_data() -> pd.DataFrame:
# Load stock data from CSV file.
    print(f"Loading stock data from {DATA_PATH}...")
    return pd.read_csv(DATA_PATH)

def load_or_train_lr_model(stock_data: pd.DataFrame) -> LinearRegressionModel:
# Load Linear Regression model if it exists, otherwise return None.
    if not os.path.exists(LR_MODEL_PATH):
        print(f"No Linear Regression model found at {LR_MODEL_PATH}")
        return None
    else:
        print(f"Loading Linear Regression model from {LR_MODEL_PATH}...")
        return joblib.load(LR_MODEL_PATH)

def load_or_train_lstm_model(stock_data: pd.DataFrame) -> LSTMModel:
# Load LSTM model if it exists, otherwise return None.
    lstm_model = LSTMModel(stock_data)
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"No LSTM model found at {LSTM_MODEL_PATH}")
        return None
    else:
        print(f"Loading LSTM model from {LSTM_MODEL_PATH}...")
        lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH))
    return lstm_model

def main() -> None:
# Main function to fetch data, train/load models, and evaluate them.
    try:
        fetch_data_if_needed()
        stock_data = load_stock_data()
        lr_model = load_or_train_lr_model(stock_data)
        lstm_model = load_or_train_lstm_model(stock_data)
        print("Evaluating models...")
        evaluate_models(lr_model, lstm_model, stock_data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
