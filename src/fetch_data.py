"""Data acquisition and leak-free feature engineering.

The contract (REFACTOR_PLAN.md section 1) is that every value on row ``t`` is
computable from information available at the **close** of trading day ``t``.
Concretely that forbids centred rolling windows, backward fill, and any global
fit. ``tests/test_leakage.py`` enforces it.

Features are also kept scale free. Price levels let a tree learn the identity
function and push a MinMax-scaled test set outside ``[0, 1]``, so levels are
replaced by ratios, returns and bounded oscillators.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

# Guard for ratios whose denominator can legitimately collapse to zero
# (a flat bar, a zero-width Bollinger band).
EPS = 1e-8

MACRO_SYMBOLS = {'^GSPC': 'SP500', '^VIX': 'VIX', '^TNX': 'TNX_Yield'}

# Raw columns carried through the engineered frame. These are inputs, not
# features: the dataset builder needs Close to form the target and ``close_t``,
# and the macro levels have to survive so ``engineer_features`` can be re-run.
RAW_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
               'SP500', 'VIX', 'TNX_Yield']

#: The only raw columns permitted to appear in FEATURE_COLUMNS as levels.
#:
#: Every other entry in RAW_COLUMNS is a price or volume level and is banned by
#: default, which is the point: adding a new raw column bans it automatically,
#: and permitting it requires an explicit edit here with a reason. Levels are
#: what let a tree learn the identity function, and what put a scaled test set
#: outside the training range on any trending series.
#:
#: VIX and TNX_Yield are exceptions on their merits: both are bounded and
#: mean-reverting over the sample, and the meta-learner needs the VIX *level*
#: to gate on volatility regime. 'Date' is not a feature at all.
LEVEL_FEATURES_ALLOWED = ('VIX', 'TNX_Yield')

# Non-stationary intermediates from the previous feature set. They are computed
# as locals now; this list only exists so re-engineering an old CSV cannot
# smuggle them back in.
LEGACY_LEVEL_COLUMNS = [
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Lower', 'ATR_14', 'OBV',
    'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_5',
]

# Features derivable from OHLCV alone.
BASE_FEATURE_COLUMNS = [
    # Intraday price shape
    'hl_range', 'oc_ret', 'clv',
    # Trend: distance from a moving average, as a fraction of price
    'close_to_sma20', 'close_to_sma50', 'close_to_ema12', 'close_to_ema26',
    # Momentum
    'macd_norm', 'macd_signal_norm', 'macd_hist_norm', 'RSI_14',
    # Volatility
    'bb_position', 'BB_Width', 'atr_pct', 'Volatility_20',
    # Returns
    'Log_Return', 'Return_1d', 'Return_5d', 'Return_10d', 'Return_21d',
    # Volume
    'obv_change', 'Volume_Ratio', 'log_volume_change',
    # Calendar
    'DayOfWeek',
]

# Features that additionally require the macro joins.
MACRO_FEATURE_COLUMNS = [
    'SP500_Return_1d', 'SP500_Return_5d', 'Corr_SP500_20',
    # VIX and TNX_Yield stay as levels deliberately: both are bounded and
    # mean-reverting over the sample, and the meta-learner needs the VIX level
    # to gate on volatility regime.
    'VIX', 'VIX_Change',
    'TNX_Yield', 'TNX_change',
]

#: The feature set. Exhaustive and explicit; never infer it by excluding
#: columns, which is how ``Close`` leaked in.
FEATURE_COLUMNS: list[str] = BASE_FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS


def compute_rsi(series, window=14):
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], 100.0).fillna(50.0)


def compute_atr(high, low, close, window=14):
    """Compute Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def available_feature_columns(df: pd.DataFrame) -> list[str]:
    """``FEATURE_COLUMNS`` restricted to those present on ``df``.

    An explicit whitelist intersection. Callers must never select features by
    dropping known-bad columns instead.
    """
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw OHLCV (plus optional macro) frame into the feature set.

    Every engineered column is a function of rows ``<= t`` only. Warm-up rows
    are left as NaN rather than filled, so the caller drops them explicitly.
    """
    df = df.copy()

    # Forward fill only. ``bfill`` would pull a future observation backwards in
    # time, which is literal lookahead.
    for name in MACRO_SYMBOLS.values():
        if name in df.columns:
            df[name] = df[name].ffill()

    close, high, low, open_ = df['Close'], df['High'], df['Low'], df['Open']

    # Intraday price shape
    df['hl_range'] = (high - low) / close
    df['oc_ret'] = np.log(close / open_)
    df['clv'] = (close - low) / (high - low + EPS)

    # Trend
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    df['close_to_sma20'] = close / sma_20 - 1
    df['close_to_sma50'] = close / sma_50 - 1
    df['close_to_ema12'] = close / ema_12 - 1
    df['close_to_ema26'] = close / ema_26 - 1

    # Momentum
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_norm'] = macd / close
    df['macd_signal_norm'] = macd_signal / close
    df['macd_hist_norm'] = (macd - macd_signal) / close

    df['RSI_14'] = compute_rsi(close, window=14)

    # Volatility
    bb_std = close.rolling(window=20).std()
    bb_upper = sma_20 + 2 * bb_std
    bb_lower = sma_20 - 2 * bb_std
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + EPS)
    df['BB_Width'] = (bb_upper - bb_lower) / sma_20
    df['atr_pct'] = compute_atr(high, low, close, window=14) / close

    df['Log_Return'] = np.log(close / close.shift(1))
    df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()

    # Volume
    if 'Volume' in df.columns:
        volume = df['Volume']
        avg_volume = volume.rolling(window=20).mean()
        # Raw OBV is an unbounded cumulative sum; its change, scaled by recent
        # turnover, is not.
        obv = (np.sign(close.diff()) * volume).fillna(0.0).cumsum()
        df['obv_change'] = obv.diff() / avg_volume
        df['Volume_Ratio'] = volume / avg_volume
        df['log_volume_change'] = np.log(volume.replace(0, np.nan)).diff()
    else:
        df['obv_change'] = 0.0
        df['Volume_Ratio'] = 1.0
        df['log_volume_change'] = 0.0

    # Multi-horizon returns
    for horizon in (1, 5, 10, 21):
        df[f'Return_{horizon}d'] = close.pct_change(horizon)

    # Macro
    if 'SP500' in df.columns:
        df['SP500_Return_1d'] = df['SP500'].pct_change(1)
        df['SP500_Return_5d'] = df['SP500'].pct_change(5)
        df['Corr_SP500_20'] = close.rolling(window=20).corr(df['SP500'])
    if 'VIX' in df.columns:
        df['VIX_Change'] = df['VIX'].diff()
    if 'TNX_Yield' in df.columns:
        df['TNX_change'] = df['TNX_Yield'].diff()

    # Calendar
    if 'Date' in df.columns:
        df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    elif isinstance(df.index, pd.DatetimeIndex):
        df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop(columns=[c for c in LEGACY_LEVEL_COLUMNS if c in df.columns])

    # Raw inputs first, then the features. VIX and TNX_Yield are both, hence
    # the de-duplication.
    keep = [c for c in RAW_COLUMNS if c in df.columns]
    keep += [c for c in available_feature_columns(df) if c not in keep]
    return df[keep]


def _get_ticker(symbol: str):
    """Return a yfinance Ticker, importing yfinance lazily.

    The import lives here rather than at module scope so that importing this
    module -- which every test does -- neither requires yfinance to be
    installed nor performs any network setup. Tests monkeypatch this function
    to serve the recorded fixture, which is what makes them genuinely offline
    rather than merely not-hitting-the-wire-today.
    """
    import yfinance as yf

    return yf.Ticker(symbol)


def fetch_stock_data(ticker_symbol, start_date, end_date=None):
    """
    Fetch stock data from Yahoo Finance and engineer the leak-free feature set.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format (defaults to current date)

    Returns:
        pd.DataFrame | None: Raw OHLCV plus ``FEATURE_COLUMNS``, or None on error.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}...")
    ticker = _get_ticker(ticker_symbol)

    try:
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data found for {ticker_symbol} in the given date range.")

        df = df.reset_index()
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        df['Date'] = df['Date'].dt.normalize()

        # Fetch macroeconomic indicators and merge them on Date
        print("Fetching macroeconomic indicators...")

        df.set_index('Date', inplace=True)
        for sym, name in MACRO_SYMBOLS.items():
            try:
                m_df = _get_ticker(sym).history(start=start_date, end=end_date)[['Close']]
                if not m_df.empty:
                    m_df.rename(columns={'Close': name}, inplace=True)
                    if m_df.index.tz is not None:
                        m_df.index = m_df.index.tz_localize(None)
                    m_df.index = m_df.index.normalize()
                    df = df.join(m_df, how='left')
                    # ffill only; the leading gap is dropped below.
                    df[name] = df[name].ffill()
            except Exception as e:
                print(f"Warning: Failed to fetch macro {sym}: {e}")

        df.reset_index(inplace=True)
        df = df.drop(columns=[col for col in ['Dividends', 'Stock Splits'] if col in df.columns])

        df = engineer_features(df)

        # Drop the rolling-window warm-up and the leading macro gap.
        df = df.dropna().reset_index(drop=True)

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            print(f"Warning: missing feature columns: {missing}")

        save_to_csv(df)
        print(f"Data fetched successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        print(f"Features: {available_feature_columns(df)}")
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None


def save_to_csv(df):
    """
    Save the DataFrame to a CSV file in the 'data' directory.

    Args:
        df (pd.DataFrame): The DataFrame to save
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
