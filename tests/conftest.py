"""Shared test fixtures.

The offline AAPL fixture used across the suite: raw OHLCV plus macro levels,
2018-2021, no network access. Previously each test module carried its own copy
of the path and loader.
"""

from pathlib import Path

import pandas as pd

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aapl_raw.csv"


def load_fixture() -> pd.DataFrame:
    """Raw AAPL OHLCV plus macro levels. No network access."""
    return pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])
