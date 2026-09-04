"""Tests for the shared dataset builder (REFACTOR_PLAN.md section 3).

The row contract is checked here numerically rather than by inspection,
because an off-by-one in the target is the single easiest way to produce
excellent-looking results from a broken pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset import Dataset, build_dataset, build_sequences
from fetch_data import FEATURE_COLUMNS, engineer_features

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aapl_raw.csv"
LOOKBACK = 30


def load_fixture() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def engineered() -> pd.DataFrame:
    return engineer_features(load_fixture()).dropna().reset_index(drop=True)


@pytest.fixture(scope="module")
def ds(engineered) -> Dataset:
    return build_dataset(engineered)


def test_target_is_the_next_day_log_return(ds, engineered):
    close = engineered["Close"].to_numpy(float)
    expected = np.log(close[1:] / close[:-1])
    np.testing.assert_allclose(ds.y, expected)


def test_keys_follow_the_row_contract(ds, engineered):
    dates = engineered["Date"].to_numpy("datetime64[ns]")
    close = engineered["Close"].to_numpy(float)

    # feature_date[t] is day t, target_date[t] is day t+1, close_t[t] is Close[t]
    np.testing.assert_array_equal(ds.feature_date, dates[:-1])
    np.testing.assert_array_equal(ds.target_date, dates[1:])
    np.testing.assert_allclose(ds.close_t, close[:-1])


def test_final_row_is_dropped_because_its_target_is_unknown(ds, engineered):
    assert len(ds) == len(engineered) - 1
    assert ds.feature_date[-1] == engineered["Date"].iloc[-2].to_datetime64()


def test_price_reconstruction_recovers_the_next_close(ds, engineered):
    """close_hat[t+1] = close_t[t] * exp(y[t]), exactly, for the realised y."""
    from contracts import reconstruct_close

    reconstructed = reconstruct_close(ds.close_t, ds.y)
    np.testing.assert_allclose(reconstructed, engineered["Close"].to_numpy(float)[1:])


def test_X_is_exactly_feature_columns_in_order(ds):
    assert ds.feature_names == FEATURE_COLUMNS
    assert ds.X.shape == (len(ds), len(FEATURE_COLUMNS))


def test_no_price_level_reaches_the_feature_matrix(ds):
    for banned in ("Close", "Open", "High", "Low", "Volume", "SP500"):
        assert banned not in ds.feature_names


def test_build_dataset_engineers_features_when_given_a_raw_frame():
    ds_from_raw = build_dataset(load_fixture())
    assert ds_from_raw.feature_names == FEATURE_COLUMNS
    assert np.isfinite(ds_from_raw.X).all()
    assert np.isfinite(ds_from_raw.y).all()


def test_build_dataset_rejects_a_frame_without_close():
    with pytest.raises(ValueError, match="Date.*Close|Close"):
        build_dataset(pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"])}))


def test_build_dataset_sorts_by_date(engineered):
    shuffled = engineered.sample(frac=1.0, random_state=0)
    ds_shuffled = build_dataset(shuffled)
    ordered = build_dataset(engineered)
    np.testing.assert_array_equal(ds_shuffled.target_date, ordered.target_date)
    np.testing.assert_allclose(ds_shuffled.y, ordered.y)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------


def test_sequences_are_keyed_like_the_parent_dataset(ds):
    fold = np.arange(500, 600)
    seq = build_sequences(ds, LOOKBACK, fold)

    np.testing.assert_array_equal(seq.target_date, ds.target_date[fold])
    np.testing.assert_array_equal(seq.close_t, ds.close_t[fold])
    np.testing.assert_allclose(seq.y, ds.y[fold])


def test_sequence_window_ends_at_the_feature_date(ds):
    """Window j must be rows i-lookback+1 .. i of X, with i the keyed row."""
    fold = np.arange(400, 410)
    seq = build_sequences(ds, LOOKBACK, fold)

    assert seq.X.shape == (len(fold), LOOKBACK, ds.n_features)
    for j, i in enumerate(seq.row_index):
        np.testing.assert_allclose(seq.X[j], ds.X[i - LOOKBACK + 1 : i + 1])
        np.testing.assert_allclose(seq.X[j, -1], ds.X[i])


def test_sequences_never_reach_forward(ds):
    """The last row of a window is the keyed feature_date, never later."""
    seq = build_sequences(ds, LOOKBACK, np.arange(300, 320))
    for j, i in enumerate(seq.row_index):
        assert seq.feature_date[j] == ds.feature_date[i]
        # The window's newest row is X[i]; anything at i+1 must be absent.
        assert not np.allclose(seq.X[j, -1], ds.X[i + 1])


def test_a_test_fold_keeps_its_full_length(ds):
    """Windows reach back into earlier data, which is not leakage."""
    fold = np.arange(700, 760)
    seq = build_sequences(ds, LOOKBACK, fold)
    assert len(seq) == len(fold)


def test_early_rows_without_enough_history_are_dropped(ds):
    fold = np.arange(0, 50)
    seq = build_sequences(ds, LOOKBACK, fold)
    assert len(seq) == 50 - (LOOKBACK - 1)
    assert seq.row_index[0] == LOOKBACK - 1


def test_build_sequences_rejects_an_impossible_lookback(ds):
    with pytest.raises(ValueError, match="predecessors"):
        build_sequences(ds, lookback=100, fold=np.arange(0, 10))


def test_build_sequences_rejects_out_of_range_folds(ds):
    with pytest.raises(ValueError, match="out of range"):
        build_sequences(ds, LOOKBACK, np.array([len(ds)]))


def test_subset_keeps_keys_aligned(ds):
    idx = np.arange(100, 200)
    sub = ds.subset(idx)
    assert len(sub) == 100
    np.testing.assert_array_equal(sub.target_date, ds.target_date[idx])
    np.testing.assert_allclose(sub.X, ds.X[idx])
