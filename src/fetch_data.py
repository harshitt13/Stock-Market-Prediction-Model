import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_stock_data(ticker_symbol, start_date, end_date=None):
    """
    Fetch stock data and financial metrics using yfinance
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format (defaults to current date)
    """

    # Set end date to current date if not specified
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize yfinance ticker object
    ticker = yf.Ticker(ticker_symbol)
    
    # Fetch historical market data
    df = ticker.history(start=start_date, end=end_date)
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Remove Dividends and Stock Splits columns
    columns_to_drop = ['Dividends', 'Stock Splits']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    try:
        # Get earnings per share
        eps = ticker.info.get('trailingEps', None)
        
        # Get revenue (trailing 12 months)
        revenue = ticker.info.get('totalRevenue', None)
        
        # Get return on equity
        roe = ticker.info.get('returnOnEquity', None)
        
        # Get P/E ratio
        pe_ratio = ticker.info.get('trailingPE', None)
        
        # Add financial metrics to the dataframe
        df['EPS'] = eps
        df['Revenue'] = revenue
        df['ROE'] = roe
        df['P/E'] = pe_ratio
        
    except Exception as e:
        print(f"Error fetching financial metrics: {e}")
    
    # Reorder columns to match desired format
    price_cols = [col for col in ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'] 
                if col in df.columns]
    
    final_cols = ['Date'] + price_cols + ['EPS', 'Revenue', 'ROE', 'P/E']
    df = df[final_cols]
    
    # Save to CSV
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filepath = os.path.join(data_dir, "stock_data.csv")
    df.to_csv(filepath, index=False)
    
    print(f"Data fetched: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Stock data saved to {filepath}")

# Example usage
if __name__ == "__main__":
    ticker_symbol = 'AAPL'
    start_date = '2014-01-01' 
    fetch_stock_data(ticker_symbol, start_date)
