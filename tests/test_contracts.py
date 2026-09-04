"""Tests for the prediction contract (REFACTOR_PLAN.md section 1).

validate_predictions exists to fail loudly, so most of these assert that it
actually rejects the malformed frame rather than passing it through.
"""

import numpy as np
import pandas as pd
import pytest

from contracts import (
    PREDICTION_COLUMNS,
    ContractViolation,
    empty_predictions,
    make_predictions,
    reconstruct_close,
    validate_predictions,
)


def good_frame(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return make_predictions(
        target_date=pd.date_range("2020-01-01", periods=n, freq="B"),
        fold_id=np.zeros(n, dtype=int),
        close_t=np.full(n, 100.0),
        y_true=rng.normal(0, 0.01, n),
        y_pred=rng.normal(0, 0.01, n),
    )


def test_a_well_formed_frame_passes_and_is_returned_unchanged():
    df = good_frame()
    out = validate_predictions(df)
    pd.testing.assert_frame_equal(out, df)


def test_make_predictions_produces_the_exact_schema():
    df = good_frame()
    assert list(df.columns) == PREDICTION_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["target_date"])
    assert df["fold_id"].dtype.kind in "iu"
    for col in ("close_t", "y_true", "y_pred"):
        assert df[col].dtype.kind == "f"


def test_empty_frame_has_the_right_dtypes_but_fails_validation():
    empty = empty_predictions()
    assert list(empty.columns) == PREDICTION_COLUMNS
    with pytest.raises(ContractViolation, match="no rows"):
        validate_predictions(empty)


def test_rejects_a_missing_column():
    df = good_frame().drop(columns=["close_t"])
    with pytest.raises(ContractViolation, match="close_t"):
        validate_predictions(df)


def test_rejects_an_extra_column():
    df = good_frame()
    df["y_pred_price"] = 1.0
    with pytest.raises(ContractViolation, match="unexpected"):
        validate_predictions(df)


def test_rejects_columns_in_the_wrong_order():
    df = good_frame()[["fold_id", "target_date", "close_t", "y_true", "y_pred"]]
    with pytest.raises(ContractViolation, match="exactly"):
        validate_predictions(df)


def test_rejects_a_non_datetime_target_date():
    df = good_frame()
    df["target_date"] = df["target_date"].astype(str)
    with pytest.raises(ContractViolation, match="datetime64"):
        validate_predictions(df)


def test_rejects_a_float_fold_id():
    df = good_frame()
    df["fold_id"] = df["fold_id"].astype(float)
    with pytest.raises(ContractViolation, match="fold_id must be an integer"):
        validate_predictions(df)


def test_rejects_nans():
    df = good_frame()
    df.loc[2, "y_pred"] = np.nan
    with pytest.raises(ContractViolation, match="NaN"):
        validate_predictions(df)


def test_rejects_infinities():
    df = good_frame()
    df.loc[1, "y_pred"] = np.inf
    with pytest.raises(ContractViolation, match="finite"):
        validate_predictions(df)


def test_rejects_duplicate_target_dates():
    """Overlapping walk-forward test folds would produce this."""
    df = good_frame()
    df.loc[3, "target_date"] = df.loc[2, "target_date"]
    with pytest.raises(ContractViolation, match="unique"):
        validate_predictions(df)


def test_rejects_unsorted_target_dates():
    df = good_frame().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ContractViolation, match="sorted"):
        validate_predictions(df)


def test_rejects_non_positive_close():
    df = good_frame()
    df.loc[0, "close_t"] = 0.0
    with pytest.raises(ContractViolation, match="positive"):
        validate_predictions(df)


def test_make_predictions_sorts_by_target_date():
    n = 4
    df = make_predictions(
        target_date=pd.to_datetime(
            ["2020-01-04", "2020-01-01", "2020-01-03", "2020-01-02"]
        ),
        fold_id=np.arange(n),
        close_t=np.full(n, 10.0),
        y_true=np.zeros(n),
        y_pred=np.zeros(n),
    )
    validate_predictions(df)
    assert df["target_date"].is_monotonic_increasing


def test_reconstruct_close_inverts_the_log_return():
    close_t = np.array([100.0, 250.0, 33.5])
    y = np.array([0.01, -0.02, 0.0])
    np.testing.assert_allclose(reconstruct_close(close_t, y), close_t * np.exp(y))


def test_contract_violation_is_an_assertion_error():
    """So a bare `pytest.raises(AssertionError)` in a model test still works."""
    assert issubclass(ContractViolation, AssertionError)
