"""Tests for the data layer.

Offline by construction. ``fetch_data._get_ticker`` is replaced with the
recorded fixture, and ``yfinance`` is never imported at all: the module holds
no top-level import of it, so these tests neither require it to be installed
nor perform any network setup. Nothing here touches the repository's ``data/``
directory either.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fetch_data
from fetch_data import (
    BASE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    LEGACY_LEVEL_COLUMNS,
    RAW_COLUMNS,
    available_feature_columns,
    engineer_features,
    fetch_stock_data,
    save_to_csv,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aapl_raw.csv"


def load_fixture() -> pd.DataFrame:
    """Raw AAPL OHLCV plus macro levels. No network access."""
    return pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])


class _FakeTicker:
    """Serves the recorded fixture in the shape ``yf.Ticker`` returns."""

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None):
        raw = load_fixture()
        index = pd.DatetimeIndex(raw["Date"]).tz_localize("America/New_York")
        index.name = "Date"

        if self.symbol in fetch_data.MACRO_SYMBOLS:
            name = fetch_data.MACRO_SYMBOLS[self.symbol]
            return pd.DataFrame({"Close": raw[name].to_numpy()}, index=index)

        frame = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        frame.index = index
        frame["Dividends"] = 0.0
        frame["Stock Splits"] = 0.0
        return frame


class _EmptyTicker(_FakeTicker):
    def history(self, start=None, end=None):
        return pd.DataFrame()


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Serve the fixture instead of yfinance, and keep writes in tmp_path."""
    monkeypatch.setattr(fetch_data, "_get_ticker", _FakeTicker)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_fetch_stock_data_returns_features_and_saves_csv(offline):
    df = fetch_stock_data("AAPL", "2018-01-01", "2022-01-01")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert (offline / "data" / "stock_data.csv").exists()

    assert not df.columns.duplicated().any()
    assert set(df.columns) == set(RAW_COLUMNS) | set(FEATURE_COLUMNS)
    assert not df.isna().any().any(), "warm-up rows should have been dropped"
    assert df["Date"].is_monotonic_increasing
    assert df["Date"].dt.tz is None


def test_fetch_stock_data_invalid_ticker(offline, monkeypatch):
    monkeypatch.setattr(fetch_data, "_get_ticker", _EmptyTicker)
    assert fetch_stock_data("INVALID_TICKER_XYZ", "2018-01-01", "2022-01-01") is None


def test_save_to_csv(offline):
    sample_df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Close": [150.0, 152.5],
            "Volume": [1000000, 1200000],
        }
    )
    save_to_csv(sample_df)

    saved_df = pd.read_csv(offline / "data" / "stock_data.csv")
    pd.testing.assert_frame_equal(sample_df, saved_df, check_dtype=False)


def test_fetch_stock_data_is_defined_once():
    """It used to be defined twice, the first shadowed and silently dead."""
    text = Path(fetch_data.__file__).read_text(encoding="utf-8")
    assert text.count("def fetch_stock_data(") == 1


def test_engineered_frame_drops_non_stationary_levels():
    engineered = engineer_features(load_fixture())
    for col in LEGACY_LEVEL_COLUMNS:
        assert col not in engineered.columns


def test_engineer_features_survives_missing_macro_columns():
    ohlcv_only = load_fixture()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    engineered = engineer_features(ohlcv_only)

    assert available_feature_columns(engineered) == BASE_FEATURE_COLUMNS


def test_macro_gaps_are_forward_filled_never_backward_filled():
    raw = load_fixture()
    gap = range(300, 305)
    last_known = raw["VIX"].iloc[299]
    first_after_gap = raw["VIX"].iloc[305]
    raw.loc[raw.index[gap], "VIX"] = np.nan

    engineered = engineer_features(raw)

    assert (engineered["VIX"].iloc[gap] == last_known).all()
    # A bfill would have used the first post-gap observation instead.
    assert engineered["VIX"].iloc[300] != first_after_gap


def test_module_never_imports_yfinance_at_import_time():
    """The offline guarantee, asserted rather than assumed.

    A top-level ``import yfinance`` would make every test in this repository
    depend on the package being installed, and on whatever network setup it
    performs on import. The import lives inside ``_get_ticker`` instead.
    """
    import subprocess
    import sys

    repo = Path(fetch_data.__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); import fetch_data; "
            "print('yfinance' in sys.modules)",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "importing fetch_data pulled in yfinance; move the import back inside "
        "_get_ticker"
    )
