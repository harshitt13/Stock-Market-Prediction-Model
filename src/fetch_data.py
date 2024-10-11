import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Reset the index to make 'Date' a column
    stock_data.reset_index(inplace=True)
    
    # Save to CSV
    stock_data.to_csv('data/stock_data.csv', index=False)
    
    print(f"Stock data for {ticker} from {start_date} to {end_date} saved to 'data/stock_data.csv'.")

# Example usage
fetch_stock_data('AAPL', '2020-01-01', '2023-01-01')
