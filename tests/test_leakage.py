"""The most important test in this repository.

Everything downstream assumes the row contract from REFACTOR_PLAN.md section 1:
a feature on row ``t`` uses only information available at the close of trading
day ``t``. Leakage does not crash a pipeline, it just makes the numbers look
good, so it has to be caught here rather than in a results table.

The truncation check has a subtlety worth stating, because it was measured
rather than assumed. It compares the value **at row t** between the full and
the truncated history. A backward fill therefore only shows up if row ``t``
itself is missing: ``ffill()`` runs first and covers every interior gap
identically in both windows, so ``bfill`` can alter only *leading* NaNs. That
is why :data:`TRUNCATION_POINTS` includes points inside the leading region of
the gapped fixture below, and why the bfill check is behavioural rather than a
scan of the source for the string "bfill".

``tests/test_leakage_mutations.py`` verifies that this check actually fails
when leakage is injected.

These tests are offline. They read ``tests/fixtures/aapl_raw.csv``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fetch_data import (
    FEATURE_COLUMNS,
    LEVEL_FEATURES_ALLOWED,
    MACRO_SYMBOLS,
    RAW_COLUMNS,
    available_feature_columns,
    engineer_features,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aapl_raw.csv"

# Spread across the sample so a window-length-dependent bug cannot hide in one
# quiet stretch.
TRUNCATION_POINTS = [200, 400, 600, 800]

#: Rows inside the leading macro gap. A backward fill is invisible anywhere
#: else, so without these the bfill check has no power.
LEADING_GAP_LENGTH = 6
LEADING_TRUNCATION_POINTS = [1, 3, 5]

# A feature this correlated with tomorrow's return is a bug, not a discovery.
MAX_ABS_TARGET_CORRELATION = 0.5


def load_fixture() -> pd.DataFrame:
    """Raw AAPL OHLCV plus macro levels. No network access."""
    return pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])


def fixture_with_leading_macro_gap(length: int = LEADING_GAP_LENGTH) -> pd.DataFrame:
    """The fixture with the macro series starting later than the stock.

    This is the real-world shape that makes a backward fill dangerous: the
    macro history begins after the price history, so a bfill reaches forward
    to a value that did not exist yet on those days.
    """
    raw = load_fixture()
    for name in MACRO_SYMBOLS.values():
        raw.loc[: length - 1, name] = np.nan
    return raw


def truncation_violations(engineer, raw, points, columns):
    """Columns whose value at row t changes when rows after t are removed.

    The shared body of the leakage check, so the mutation suite can reuse the
    exact comparison rather than an approximation of it.
    """
    full = engineer(raw.copy())
    violations = []
    for t in points:
        truncated = engineer(raw.iloc[: t + 1].copy())
        for col in columns:
            if col not in full.columns or col not in truncated.columns:
                continue
            a, b = full[col].iloc[t], truncated[col].iloc[t]
            if not np.isclose(a, b, equal_nan=True):
                violations.append((col, t, float(a), float(b)))
    return violations


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
    violations = truncation_violations(
        engineer_features, load_fixture(), TRUNCATION_POINTS, FEATURE_COLUMNS
    )
    assert not violations, (
        f"{len(violations)} feature values changed when future rows were "
        f"removed: {violations[:10]}"
    )


# ---------------------------------------------------------------------------
# Backward fill, checked behaviourally
# ---------------------------------------------------------------------------


def _bfill_leaky_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """The exact bug from commit 3933168's fetch_stock_data.

    Master applied ``ffill().bfill()`` to the macro columns during the join,
    before engineer_features ever ran, so the leak was upstream of the function
    under test. Reproducing it here as a wrapper puts it back in the path the
    truncation check actually covers.
    """
    df = df.copy()
    for name in MACRO_SYMBOLS.values():
        if name in df.columns:
            df[name] = df[name].ffill().bfill()
    return engineer_features(df)


def test_leading_macro_gap_is_not_filled_from_the_future():
    """The real assertion behind "no bfill": rows before the first macro
    observation must stay unknown, not borrow a later value."""
    raw = fixture_with_leading_macro_gap()
    engineered = engineer_features(raw)

    for name in MACRO_SYMBOLS.values():
        leading = engineered[name].iloc[:LEADING_GAP_LENGTH]
        assert leading.isna().all(), (
            f"{name} was filled on rows before its first observation; "
            "that value did not exist yet"
        )


def test_truncation_check_catches_a_backward_fill():
    """Guards the guard.

    If this ever passes, the truncation check has lost its power over backward
    fill and the no-bfill guarantee is worthless.
    """
    raw = fixture_with_leading_macro_gap()
    points = LEADING_TRUNCATION_POINTS + TRUNCATION_POINTS

    leaked = truncation_violations(_bfill_leaky_engineer, raw, points, FEATURE_COLUMNS)
    assert leaked, (
        "a backward fill on the macro columns went undetected; the truncation "
        "check no longer constrains it"
    )
    assert {col for col, *_ in leaked} & {"VIX", "TNX_Yield", "VIX_Change", "TNX_change"}


def test_real_pipeline_survives_the_gapped_fixture():
    """And the actual implementation passes the same check that catches bfill."""
    violations = truncation_violations(
        engineer_features,
        fixture_with_leading_macro_gap(),
        LEADING_TRUNCATION_POINTS + TRUNCATION_POINTS,
        FEATURE_COLUMNS,
    )
    assert not violations, violations[:10]


# ---------------------------------------------------------------------------
# Target correlation and the level ban
# ---------------------------------------------------------------------------


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
    """Every raw column is banned as a feature unless explicitly permitted.

    Derived from RAW_COLUMNS rather than a hand-written list, so adding a new
    raw column bans it by default and permitting it requires an edit to
    LEVEL_FEATURES_ALLOWED with a stated reason.
    """
    banned = set(RAW_COLUMNS) - set(LEVEL_FEATURES_ALLOWED) - {"Date"}
    leaked = banned.intersection(FEATURE_COLUMNS)
    assert not leaked, (
        f"raw level columns used as features: {sorted(leaked)}. "
        "If one genuinely belongs, add it to LEVEL_FEATURES_ALLOWED with a "
        "reason; do not special-case it here."
    )


def test_every_allowed_level_is_actually_used_and_justified():
    """The allowlist must not accumulate dead entries."""
    for name in LEVEL_FEATURES_ALLOWED:
        assert name in RAW_COLUMNS, f"{name} is not a raw column"
        assert name in FEATURE_COLUMNS, (
            f"{name} is permitted as a level feature but is not in "
            "FEATURE_COLUMNS; remove it from LEVEL_FEATURES_ALLOWED"
        )


def test_date_is_never_a_feature():
    assert "Date" not in FEATURE_COLUMNS


def test_engineer_features_does_not_mutate_its_input():
    """Truncated re-computation is only meaningful on an untouched input."""
    raw = load_fixture()
    before = raw.copy()
    engineer_features(raw)
    pd.testing.assert_frame_equal(raw, before)
