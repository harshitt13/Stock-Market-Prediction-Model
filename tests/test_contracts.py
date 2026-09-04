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


# ---------------------------------------------------------------------------
# The common evaluation window
# ---------------------------------------------------------------------------


def windowed_frame(dates, fold_ids, seed=0):
    rng = np.random.default_rng(seed)
    n = len(dates)
    return make_predictions(
        target_date=dates,
        fold_id=np.asarray(fold_ids, dtype=int),
        close_t=np.full(n, 100.0),
        y_true=rng.normal(0, 0.01, n),
        y_pred=rng.normal(0, 0.01, n),
    )


def walk_forward_models(n_folds=9, fold_size=10, meta_starts_at=2):
    """A base model over every fold and a meta over the later folds only."""
    dates = pd.date_range("2020-01-01", periods=n_folds * fold_size, freq="B")
    fold_ids = np.repeat(np.arange(n_folds), fold_size)
    base = windowed_frame(dates, fold_ids, seed=1)
    keep = fold_ids >= meta_starts_at
    meta = windowed_frame(dates[keep], fold_ids[keep], seed=2)
    return {"base": base, "meta": meta}


class TestCommonEvaluationWindow:
    def test_is_the_intersection_across_all_models(self):
        from contracts import common_evaluation_window

        models = walk_forward_models()
        window = common_evaluation_window(models)
        assert window.n == len(models["meta"])
        assert window.start == models["meta"]["target_date"].iloc[0]
        assert window.end == models["meta"]["target_date"].iloc[-1]

    def test_reports_what_each_model_gave_up(self):
        from contracts import common_evaluation_window

        window = common_evaluation_window(walk_forward_models())
        assert window.dropped_by_model["meta"] == 0
        assert window.dropped_by_model["base"] == 20  # two folds of ten

    def test_reference_is_the_widest_model(self):
        from contracts import common_evaluation_window

        assert common_evaluation_window(walk_forward_models()).reference_model == "base"

    def test_rejects_a_disjoint_set(self):
        from contracts import common_evaluation_window

        a = windowed_frame(pd.date_range("2020-01-01", periods=10, freq="B"), np.zeros(10))
        b = windowed_frame(pd.date_range("2021-01-01", periods=10, freq="B"), np.zeros(10))
        with pytest.raises(ContractViolation, match="share no forecast days"):
            common_evaluation_window({"a": a, "b": b})

    def test_rejects_a_window_with_a_hole(self):
        """A gap in the middle is a bug, not a windowing decision."""
        from contracts import common_evaluation_window

        models = walk_forward_models()
        holed = models["meta"].drop(index=range(5, 10)).reset_index(drop=True)
        with pytest.raises(ContractViolation, match="not contiguous"):
            common_evaluation_window({"base": models["base"], "meta": holed})

    def test_contiguity_check_can_be_waived(self):
        from contracts import common_evaluation_window

        models = walk_forward_models()
        holed = models["meta"].drop(index=range(5, 10)).reset_index(drop=True)
        window = common_evaluation_window(
            {"base": models["base"], "meta": holed}, require_contiguous=False
        )
        assert window.n == len(holed)

    def test_describe_names_the_range(self):
        from contracts import common_evaluation_window

        window = common_evaluation_window(walk_forward_models())
        text = window.describe()
        assert text.startswith(f"{window.n} forecast days")
        assert f"{window.start:%Y-%m-%d} to {window.end:%Y-%m-%d}" in text


class TestRestrictToWindow:
    def test_every_model_ends_up_on_identical_days(self):
        from contracts import common_evaluation_window, restrict_all

        models = walk_forward_models()
        window = common_evaluation_window(models)
        restricted = restrict_all(models, window)

        date_sets = [set(df["target_date"]) for df in restricted.values()]
        assert all(s == date_sets[0] for s in date_sets)
        assert all(len(df) == window.n for df in restricted.values())

    def test_restricted_frames_still_satisfy_the_contract(self):
        from contracts import common_evaluation_window, restrict_all

        models = walk_forward_models()
        for df in restrict_all(models, common_evaluation_window(models)).values():
            validate_predictions(df)

    def test_alignment_becomes_an_exact_merge(self):
        from contracts import align_predictions, common_evaluation_window, restrict_all

        models = walk_forward_models()
        restricted = restrict_all(models, common_evaluation_window(models))
        wide = align_predictions(restricted, require_identical=True)
        assert len(wide) == len(restricted["meta"])

    def test_a_model_missing_window_days_fails_loudly(self):
        from contracts import common_evaluation_window, restrict_to_window

        models = walk_forward_models()
        window = common_evaluation_window(models)
        short = models["meta"].iloc[:-3]
        with pytest.raises(ContractViolation, match="does not cover"):
            restrict_to_window(short, window, "short")

    def test_y_true_is_untouched_by_restriction(self):
        from contracts import common_evaluation_window, restrict_to_window

        models = walk_forward_models()
        window = common_evaluation_window(models)
        restricted = restrict_to_window(models["base"], window, "base")
        merged = restricted.merge(
            models["base"], on="target_date", suffixes=("", "_orig")
        )
        np.testing.assert_allclose(merged["y_true"], merged["y_true_orig"])
