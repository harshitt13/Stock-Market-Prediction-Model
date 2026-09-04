"""The most important test in this repository.

Everything downstream assumes the row contract from REFACTOR_PLAN.md section 1:
a feature on row ``t`` uses only information available at the close of trading
day ``t``. Leakage does not crash a pipeline, it just makes the numbers look
good, so it has to be caught here rather than in a results table.

These tests are offline. They read ``tests/fixtures/aapl_raw.csv``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fetch_data import (
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    available_feature_columns,
    engineer_features,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aapl_raw.csv"

# Spread across the sample so a window-length-dependent bug cannot hide in one
# quiet stretch.
TRUNCATION_POINTS = [200, 400, 600, 800]

# A feature this correlated with tomorrow's return is a bug, not a discovery.
MAX_ABS_TARGET_CORRELATION = 0.5


def load_fixture() -> pd.DataFrame:
    """Raw AAPL OHLCV plus macro levels. No network access."""
    return pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def engineered() -> pd.DataFrame:
    return engineer_features(load_fixture())


def test_fixture_is_present_and_long_enough():
    raw = load_fixture()
    assert list(raw.columns) == RAW_COLUMNS
    assert len(raw) > max(TRUNCATION_POINTS)


def test_every_declared_feature_is_produced(engineered):
    missing = [col for col in FEATURE_COLUMNS if col not in engineered.columns]
    assert not missing, f"engineer_features did not produce: {missing}"
    assert available_feature_columns(engineered) == FEATURE_COLUMNS


def test_features_do_not_depend_on_future_rows(engineered):
    """
    Feature values at row t must be identical whether computed on the full
    history or on history truncated at t. Any centred rolling window, any
    bfill, any global fit will fail this.
    """
    for t in TRUNCATION_POINTS:
        truncated = engineer_features(load_fixture().iloc[: t + 1].copy())
        for col in FEATURE_COLUMNS:
            assert np.isclose(
                engineered[col].iloc[t], truncated[col].iloc[t], equal_nan=True
            ), f"{col} at row {t} changed when future rows were removed"


def test_no_feature_is_suspiciously_correlated_with_the_target(engineered):
    """Correlate every feature against the *return* target, not the price.

    Against price levels a trending feature scores ~0.99 and tells you nothing.
    Against next-day returns anything above 0.5 means the target leaked in.
    """
    df = engineered.dropna().reset_index(drop=True)
    y = np.log(df["Close"].shift(-1) / df["Close"])
    valid = y.notna()

    for col in FEATURE_COLUMNS:
        corr = df.loc[valid, col].corr(y[valid])
        assert np.isfinite(corr), f"{col} has no variance, correlation undefined"
        assert abs(corr) < MAX_ABS_TARGET_CORRELATION, (
            f"{col} correlates {corr:.3f} with the next-day return; "
            "that is a leak, not a signal"
        )


def test_feature_set_contains_no_price_levels():
    """Guards the reason ``Close`` leaked in: features chosen by exclusion.

    Trees cannot extrapolate past their training target range and MinMax
    scaling of a trending series puts every test value outside [0, 1], so a
    level in FEATURE_COLUMNS is a silent failure either way.
    """
    banned = set(RAW_COLUMNS) - {"VIX", "TNX_Yield"}
    leaked = banned.intersection(FEATURE_COLUMNS)
    assert not leaked, f"price/level columns used as features: {sorted(leaked)}"


def test_engineer_features_does_not_mutate_its_input():
    """Truncated re-computation is only meaningful on an untouched input."""
    raw = load_fixture()
    before = raw.copy()
    engineer_features(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_data_layer_contains_no_backward_fill():
    """``bfill`` pulls a future observation backwards in time."""
    source = (Path(__file__).parents[1] / "src" / "fetch_data.py").read_text(
        encoding="utf-8"
    )
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "bfill" not in code, "bfill found in src/fetch_data.py"
