import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_stock_data(ticker_symbol, start_date, end_date=None):
    """
    Fetch stock data and financial metrics using yfinance.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format (defaults to current date)

    Returns:
        pd.DataFrame: A DataFrame containing the stock data with financial metrics.
    """
    # Validate and set the end date
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}...")

    # Initialize yfinance ticker object
    ticker = yf.Ticker(ticker_symbol)

    try:
        # Fetch historical market data
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data found for {ticker_symbol} in the given date range.")

        # Reset index to make Date a column
        df.reset_index(inplace=True)

        # Drop unnecessary columns if they exist
        unnecessary_cols = ['Dividends', 'Stock Splits']
        df.drop(columns=[col for col in unnecessary_cols if col in df.columns], inplace=True)

        # Fetch additional financial metrics
        financial_metrics = {
            'EPS': ticker.info.get('trailingEps', None),
            'Revenue': ticker.info.get('totalRevenue', None),
            'ROE': ticker.info.get('returnOnEquity', None),
            'P/E': ticker.info.get('trailingPE', None)
        }

        # Add financial metrics to the DataFrame
        for metric, value in financial_metrics.items():
            df[metric] = value

        # Reorder columns
        price_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        price_cols = [col for col in price_cols if col in df.columns]  # Ensure they exist in the DataFrame
        final_cols = ['Date'] + price_cols + list(financial_metrics.keys())
        df = df[final_cols]

        # Save to CSV
        save_to_csv(df, ticker_symbol)

        print(f"Data fetched successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    except ValueError as ve:
        print(f"Validation error: {ve}")
    except Exception as e:
        print(f"Unexpected error while fetching data for {ticker_symbol}: {e}")

    return None

def save_to_csv(df, ticker_symbol):
    """
    Save the DataFrame to a CSV file in the 'data' directory.

    Args:
        df (pd.DataFrame): The DataFrame to save
        ticker_symbol (str): Stock ticker symbol to use in the filename
    """
    try:
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)

        filename = f"{ticker_symbol}_stock_data.csv"
        filepath = os.path.join(data_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"Stock data saved to {filepath}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# Example usage
if __name__ == "__main__":
    ticker_symbol = 'AAPL'  # Example: Apple Inc.
    start_date = '2014-01-01'  # Start date
    end_date = None  # Use None for the current date (optional)

    fetch_stock_data(ticker_symbol, start_date, end_date)
