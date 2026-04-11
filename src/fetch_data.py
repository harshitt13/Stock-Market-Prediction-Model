import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime


def compute_rsi(series, window=14):
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, window=14):
    """Compute Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def fetch_stock_data(ticker_symbol, start_date, end_date=None):
    """
    Fetch stock data from Yahoo Finance and engineer a rich feature set
    including multiple technical indicators for hybrid model consumption.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format (defaults to current date)

    Returns:
        pd.DataFrame: DataFrame with OHLCV data and engineered features.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}...")

    ticker = yf.Ticker(ticker_symbol)

    try:
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data found for {ticker_symbol} in the given date range.")

        df = df.reset_index()

        # Drop unnecessary columns
        df = df.drop(columns=[col for col in ['Dividends', 'Stock Splits'] if col in df.columns])

        # ── Core Price Features ──────────────────────────────────────────

        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # ── Momentum Indicators ──────────────────────────────────────────

        # MACD (12, 26, 9)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI (14)
        df['RSI_14'] = compute_rsi(df['Close'], window=14)

        # ── Volatility Indicators ────────────────────────────────────────

        # Bollinger Bands (20, 2)
        bb_sma = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = bb_sma + 2 * bb_std
        df['BB_Lower'] = bb_sma - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_sma  # Normalized width

        # Average True Range (14)
        df['ATR_14'] = compute_atr(df['High'], df['Low'], df['Close'], window=14)

        # Rolling Volatility (20-day std of log returns)
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()

        # ── Volume Indicators ────────────────────────────────────────────

        # On-Balance Volume
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV'] = obv

        # Volume Moving Average ratio (current volume vs 20-day avg)
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

        # ── Lagged / Auto-regressive Features ────────────────────────────

        df['Close_Lag_1'] = df['Close'].shift(1)
        df['Close_Lag_3'] = df['Close'].shift(3)
        df['Close_Lag_5'] = df['Close'].shift(5)

        # Price change ratios
        df['Return_1d'] = df['Close'].pct_change(1)
        df['Return_5d'] = df['Close'].pct_change(5)

        # ── Calendar Features ────────────────────────────────────────────

        df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek  # 0=Mon, 4=Fri

        # ── Cleanup ──────────────────────────────────────────────────────

        # Drop rows with NaN from rolling calculations
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Save to CSV
        save_to_csv(df, ticker_symbol)

        print(f"Data fetched successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        print(f"Features: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None


def save_to_csv(df, ticker_symbol):
    """
    Save the DataFrame to a CSV file in the 'data' directory.

    Args:
        df (pd.DataFrame): The DataFrame to save
        ticker_symbol (str): Stock ticker symbol to use in the filename
    """
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    filename = "stock_data.csv"
    filepath = os.path.join(data_dir, filename)

    df.to_csv(filepath, index=False)
    print(f"Stock data saved to {filepath}")


# Example usage
if __name__ == "__main__":
    ticker_symbol = 'AAPL'
    start_date = '2010-01-01'
    end_date = None

    fetch_stock_data(ticker_symbol, start_date, end_date)
