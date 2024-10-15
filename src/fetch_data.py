import yfinance as yf
import pandas as pd
import os
from typing import Optional

def fetch_and_save_stock_data(ticker: str, start_date: str, end_date: str, filename: str = "stock_data.csv") -> None:
    try:
# Fetch stock price data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)
        stock_data['Year'] = stock_data['Date'].dt.year  # Add Year column

# Fetch financial data
        ticker_obj = yf.Ticker(ticker)

# Get financials (Revenue, Net Income, etc.)
        financials = ticker_obj.financials
        if (financials is not None):
            financials = financials.T
            financials.reset_index(inplace=True)
            financials.rename(columns={'index': 'Date'}, inplace=True)
            financials['Date'] = pd.to_datetime(financials['Date'])
            financials['Year'] = financials['Date'].dt.year  # Add Year column

# Get income statement (for EPS)
        income_stmt = ticker_obj.income_stmt
        if (income_stmt is not None):
            income_stmt = income_stmt.T
            income_stmt.reset_index(inplace=True)
            income_stmt.rename(columns={'index': 'Date'}, inplace=True)
            income_stmt['Date'] = pd.to_datetime(income_stmt['Date'])
            income_stmt['Year'] = income_stmt['Date'].dt.year  # Add Year column

# Get other metrics (ROE, P/E)
        info = ticker_obj.info
        roe: Optional[float] = info.get('returnOnEquity')
        pe_ratio: Optional[float] = info.get('trailingPE')

# Merge dataframes
# Merge with income statement (for EPS)
        if (income_stmt is not None):
            stock_data = pd.merge(stock_data, income_stmt[['Year', 'Net Income']], on='Year', how='left')
            stock_data.rename(columns={'Net Income': 'EPS'}, inplace=True)
        else:
            stock_data['EPS'] = None

# Merge with financials (for Revenue)
        if (financials is not None):
            stock_data = pd.merge(stock_data, financials[['Year', 'Total Revenue']], on='Year', how='left')
            stock_data.rename(columns={'Total Revenue': 'Revenue'}, inplace=True)
        else:
            stock_data['Revenue'] = None

# Add ROE and P/E columns
        stock_data['ROE'] = roe
        stock_data['P/E'] = pe_ratio

        # 5. Forward fill missing values
        stock_data.fillna(method='ffill', inplace=True)  
        stock_data.drop(columns=['Year'], inplace=True)  # Remove Year column

# Save to CSV
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filepath = os.path.join(data_dir, filename)
        stock_data.to_csv(filepath, index=False)
        print(f"Stock data saved to {filepath}")

    except Exception as e:
        print(f"Error fetching or saving stock data: {e}")

if __name__ == "__main__":
    ticker_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    fetch_and_save_stock_data(ticker_symbol, start_date, end_date)
