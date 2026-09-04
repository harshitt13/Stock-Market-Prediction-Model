"""Tests for the meta-ensemble (REFACTOR_PLAN.md section 7).

The two bugs section 7 exists to fix are both silent: they produce plausible
numbers rather than crashes. So they are tested directly -- no meta-prediction
may come from a model that saw its own row, and the fold structure must be
respected rather than approximated by an 80/20 split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts import make_predictions, validate_predictions
from dataset import build_dataset
from fetch_data import engineer_features
from meta_ensemble import (
    MIN_TRAIN_FOLDS,
    build_meta_frame,
    compare_vix_gating,
    coverage_report,
    fit_stacked_meta,
    fold_respecting_intervals,
    meta_feature_columns,
    prediction_intervals,
    residual_quantiles,
    scale_for_horizon,
    vix_tercile_weights,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aapl_raw.csv"
N_FOLDS = 6
FOLD_SIZE = 60


@pytest.fixture(scope="module")
def ds():
    raw = pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])
    return build_dataset(engineer_features(raw).dropna().reset_index(drop=True))


@pytest.fixture(scope="module")
def folds(ds):
    start = len(ds) - N_FOLDS * FOLD_SIZE
    return [
        (np.arange(0, start + i * FOLD_SIZE),
         np.arange(start + i * FOLD_SIZE, start + (i + 1) * FOLD_SIZE))
        for i in range(N_FOLDS)
    ]


@pytest.fixture(scope="module")
def base_results(ds, folds):
    """Three synthetic base models with different, known skill levels."""
    rng = np.random.default_rng(0)
    test_idx = np.concatenate([test for _, test in folds])
    fold_id = np.concatenate(
        [np.full(len(test), i) for i, (_, test) in enumerate(folds)]
    )
    y_true = ds.y[test_idx]

    def model(signal_weight, noise):
        y_pred = signal_weight * y_true + rng.normal(0, noise, len(y_true))
        return {
            "predictions": make_predictions(
                target_date=ds.target_date[test_idx],
                fold_id=fold_id,
                close_t=ds.close_t[test_idx],
                y_true=y_true,
                y_pred=y_pred,
            )
        }

    return {
        "tree": model(0.30, 0.010),
        "lstm": model(0.15, 0.015),
        "transformer": model(0.05, 0.020),
    }


@pytest.fixture(scope="module")
def meta_frame(base_results, ds):
    return build_meta_frame(base_results, ds)


@pytest.fixture(scope="module")
def y_train_by_fold(ds, folds):
    return {i: ds.y[train] for i, (train, _) in enumerate(folds)}


# ---------------------------------------------------------------------------
# The meta frame
# ---------------------------------------------------------------------------


class TestBuildMetaFrame:
    def test_has_one_column_per_base_model(self, meta_frame, base_results):
        for name in base_results:
            assert f"pred_{name}" in meta_frame.columns

    def test_is_sorted_and_unique_by_target_date(self, meta_frame):
        assert meta_frame["target_date"].is_monotonic_increasing
        assert not meta_frame["target_date"].duplicated().any()

    def test_vix_is_the_level_known_at_feature_date(self, meta_frame, ds):
        """Not the level on the day being forecast, which would be lookahead."""
        vix_idx = ds.feature_names.index("VIX")
        lookup = dict(zip(ds.target_date, ds.X[:, vix_idx]))
        expected = np.array(
            [lookup[d.to_datetime64()] for d in meta_frame["target_date"]]
        )
        np.testing.assert_allclose(meta_frame["VIX"], expected)

    def test_vix_can_be_excluded(self, base_results, ds):
        frame = build_meta_frame(base_results, ds, include_vix=False)
        assert "VIX" not in frame.columns

    def test_feature_columns_respect_the_vix_switch(self, meta_frame):
        assert "VIX" in meta_feature_columns(meta_frame, use_vix=True)
        assert "VIX" not in meta_feature_columns(meta_frame, use_vix=False)


# ---------------------------------------------------------------------------
# 7.1 / 7.2 Out-of-fold stacking
# ---------------------------------------------------------------------------


class TestFoldRespectingStacking:
    def test_early_folds_produce_no_meta_prediction(self, meta_frame):
        """Correct, and stated rather than hidden."""
        result = fit_stacked_meta(meta_frame, min_train_folds=2)
        assert result["skipped_folds"] == [0, 1]
        assert set(result["predictions"]["fold_id"]) == set(range(2, N_FOLDS))

    def test_no_row_is_predicted_by_a_model_that_saw_it(self, meta_frame):
        """The 7.1 bug: predictions covering rows the meta-model was fitted on.

        Each fold's meta-model is fitted only on strictly earlier folds, so
        every returned row's fold_id exceeds every training fold_id.
        """
        result = fit_stacked_meta(meta_frame, min_train_folds=2)
        for _, row in result["fold_weights"].iterrows():
            trained_on = meta_frame[meta_frame["fold_id"] < row["fold_id"]]
            assert len(trained_on) == row["n_train"]
            assert trained_on["fold_id"].max() < row["fold_id"]

    def test_training_set_grows_with_each_fold(self, meta_frame):
        result = fit_stacked_meta(meta_frame, min_train_folds=2)
        n_train = result["fold_weights"]["n_train"].to_numpy()
        assert (np.diff(n_train) > 0).all()

    def test_result_satisfies_the_contract(self, meta_frame):
        result = fit_stacked_meta(meta_frame, min_train_folds=2)
        validate_predictions(result["predictions"])

    def test_y_true_is_untouched_by_stacking(self, meta_frame):
        result = fit_stacked_meta(meta_frame, min_train_folds=2)
        merged = result["predictions"].merge(
            meta_frame[["target_date", "y_true"]], on="target_date", suffixes=("", "_src")
        )
        np.testing.assert_allclose(merged["y_true"], merged["y_true_src"])

    def test_reports_a_weight_per_feature_per_fold(self, meta_frame):
        result = fit_stacked_meta(meta_frame, min_train_folds=2)
        for feature in result["features"]:
            assert feature in result["fold_weights"].columns

    def test_refuses_when_there_are_too_few_folds(self, meta_frame):
        with pytest.raises(RuntimeError, match="no fold had enough prior folds"):
            fit_stacked_meta(meta_frame, min_train_folds=N_FOLDS + 1)

    def test_learns_to_favour_the_better_base_model(self, meta_frame):
        """tree carries the most signal by construction, so it should get the
        largest standardised weight."""
        result = fit_stacked_meta(meta_frame, use_vix=False, min_train_folds=2)
        final = result["fold_weights"].iloc[-1]
        weights = {
            name: abs(final[f"pred_{name}"]) for name in ("tree", "lstm", "transformer")
        }
        assert max(weights, key=weights.get) == "tree"


# ---------------------------------------------------------------------------
# 7.3 VIX gating
# ---------------------------------------------------------------------------


class TestVixGating:
    def test_compares_with_and_without_and_runs_a_dm_test(
        self, meta_frame, y_train_by_fold
    ):
        comparison = compare_vix_gating(
            meta_frame, min_train_folds=2, y_train_by_fold=y_train_by_fold
        )
        assert comparison["with_vix"]["use_vix"] is True
        assert comparison["without_vix"]["use_vix"] is False
        assert "dm_statistic" in comparison["diebold_mariano"]
        assert "delta_directional_accuracy" in comparison
        assert "delta_r2_oos" in comparison

    def test_both_sides_forecast_the_same_days(self, meta_frame, y_train_by_fold):
        comparison = compare_vix_gating(
            meta_frame, min_train_folds=2, y_train_by_fold=y_train_by_fold
        )
        assert set(comparison["with_vix"]["predictions"]["target_date"]) == set(
            comparison["without_vix"]["predictions"]["target_date"]
        )

    def test_without_vix_really_excludes_it(self, meta_frame):
        result = fit_stacked_meta(meta_frame, use_vix=False, min_train_folds=2)
        assert "VIX" not in result["features"]


class TestVixTerciles:
    def test_reports_a_row_per_tercile(self, meta_frame):
        table = vix_tercile_weights(meta_frame, min_train_folds=2)
        assert list(table.index) == ["low", "mid", "high"]

    def test_terciles_partition_the_days(self, meta_frame):
        table = vix_tercile_weights(meta_frame, min_train_folds=2)
        assert table["n"].sum() == len(meta_frame)

    def test_tercile_ranges_do_not_overlap(self, meta_frame):
        table = vix_tercile_weights(meta_frame, min_train_folds=2)
        fitted = table[table["fitted"]]
        assert fitted["vix_max"].iloc[0] <= fitted["vix_min"].iloc[1]

    def test_reports_a_weight_per_base_model(self, meta_frame, base_results):
        table = vix_tercile_weights(meta_frame, min_train_folds=2)
        for name in base_results:
            assert f"pred_{name}" in table.columns

    def test_requires_vix(self, base_results, ds):
        frame = build_meta_frame(base_results, ds, include_vix=False)
        with pytest.raises(ValueError, match="no VIX column"):
            vix_tercile_weights(frame)


# ---------------------------------------------------------------------------
# 7.4 Confidence intervals
# ---------------------------------------------------------------------------


class TestResidualQuantiles:
    def test_brackets_the_residual_distribution(self):
        rng = np.random.default_rng(0)
        n = 2000
        residuals = rng.normal(0, 0.02, n)
        predictions = make_predictions(
            target_date=pd.date_range("2020-01-01", periods=n, freq="B"),
            fold_id=np.zeros(n, dtype=int),
            close_t=np.full(n, 100.0),
            y_true=residuals,
            y_pred=np.zeros(n),
        )
        low, high = residual_quantiles(predictions, level=0.95)
        assert low == pytest.approx(-0.039, abs=0.005)
        assert high == pytest.approx(0.039, abs=0.005)


class TestHorizonScaling:
    def test_width_grows_as_sqrt_h(self):
        low, high = scale_for_horizon(-0.02, 0.02, [1, 4, 9, 16])
        widths = high - low
        np.testing.assert_allclose(widths / widths[0], [1.0, 2.0, 3.0, 4.0])

    def test_a_flat_interval_would_understate_thirty_days(self):
        """The previous version applied one-step quantiles flat across 30 days."""
        _, high = scale_for_horizon(-0.02, 0.02, [30])
        assert high[0] == pytest.approx(0.02 * np.sqrt(30))
        assert high[0] / 0.02 == pytest.approx(5.48, abs=0.01)

    def test_rejects_a_horizon_below_one(self):
        with pytest.raises(ValueError, match="horizons must be"):
            scale_for_horizon(-0.01, 0.01, [0])

    def test_intervals_bracket_the_prediction(self):
        lower, upper = prediction_intervals([0.01, -0.01], -0.02, 0.02)
        np.testing.assert_allclose(lower, [-0.01, -0.03])
        np.testing.assert_allclose(upper, [0.03, 0.01])


class TestCoverage:
    def test_perfect_coverage(self):
        report = coverage_report(np.zeros(100), np.full(100, -1.0), np.full(100, 1.0))
        assert report["empirical_coverage"] == 1.0
        assert report["verdict"] == "over-covered"

    def test_zero_coverage_is_flagged_as_under_covered(self):
        report = coverage_report(np.zeros(100), np.full(100, 1.0), np.full(100, 2.0))
        assert report["empirical_coverage"] == 0.0
        assert report["verdict"] == "UNDER-COVERED"

    def test_a_correctly_specified_interval_is_well_calibrated(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 0.02, 4000)
        half = 1.959964 * 0.02
        report = coverage_report(y, y * 0 - half, y * 0 + half, nominal_level=0.95)
        assert report["verdict"] == "well calibrated"
        assert report["empirical_coverage"] == pytest.approx(0.95, abs=0.02)


class TestFoldRespectingIntervals:
    def test_quantiles_come_from_earlier_folds_only(self, meta_frame):
        stacked = fit_stacked_meta(meta_frame, min_train_folds=2)
        calibration = fold_respecting_intervals(stacked["predictions"])

        covered = set(calibration["intervals"]["fold_id"])
        # The earliest meta fold has no prior meta residuals to calibrate on.
        assert min(stacked["predictions"]["fold_id"]) not in covered

    def test_reports_coverage_per_fold_and_overall(self, meta_frame):
        stacked = fit_stacked_meta(meta_frame, min_train_folds=2)
        calibration = fold_respecting_intervals(stacked["predictions"])
        assert not calibration["per_fold"].empty
        assert 0.0 <= calibration["overall"]["empirical_coverage"] <= 1.0
        assert calibration["overall"]["nominal_level"] == 0.95

    def test_handles_too_little_history_without_crashing(self):
        n = 10
        predictions = make_predictions(
            target_date=pd.date_range("2020-01-01", periods=n, freq="B"),
            fold_id=np.zeros(n, dtype=int),
            close_t=np.full(n, 100.0),
            y_true=np.zeros(n),
            y_pred=np.zeros(n),
        )
        calibration = fold_respecting_intervals(predictions)
        assert calibration["overall"] is None
