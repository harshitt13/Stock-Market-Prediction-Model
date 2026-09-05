"""Tests for the baselines (REFACTOR_PLAN.md section 6)."""


import numpy as np
import pandas as pd
import pytest

from baselines import (
    BASELINES,
    ar1_baseline,
    arima_baseline,
    historical_mean_baseline,
    random_sign_baseline,
    run_all_baselines,
)
from contracts import ContractViolation, align_predictions, validate_predictions
from dataset import build_dataset
from evaluate import (
    UnseededBenchmark, directional_accuracy, evaluate_predictions, expanding_mean_benchmark,
    per_fold_metrics, r2_oos,
)
from fetch_data import engineer_features
from conftest import FIXTURE_PATH


@pytest.fixture(scope="module")
def ds():
    raw = pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])
    return build_dataset(engineer_features(raw).dropna().reset_index(drop=True))


@pytest.fixture(scope="module")
def folds(ds):
    return [
        (np.arange(0, 500), np.arange(500, 600)),
        (np.arange(0, 600), np.arange(600, 700)),
        (np.arange(0, 700), np.arange(700, len(ds))),
    ]


@pytest.fixture(scope="module")
def all_baselines(ds, folds):
    return run_all_baselines(ds, folds, seed=1)


class TestEveryBaselineSatisfiesTheContract:
    @pytest.fixture(params=list(BASELINES))
    def one(self, request, all_baselines):
        return all_baselines[request.param]

    def test_valid_standard_frame(self, one):
        validate_predictions(one["predictions"])

    def test_covers_every_test_row(self, one, folds):
        expected = sum(len(test) for _, test in folds)
        assert len(one["predictions"]) == expected

    def test_every_fold_contributes(self, one, folds):
        assert set(one["predictions"]["fold_id"]) == set(range(len(folds)))

    def test_y_true_is_the_dataset_target(self, one, ds, folds):
        predictions = one["predictions"]
        lookup = dict(zip(ds.target_date, ds.y))
        expected = np.array(
            [lookup[d.to_datetime64()] for d in predictions["target_date"]]
        )
        np.testing.assert_allclose(predictions["y_true"], expected)


def test_all_baselines_forecast_identical_days(all_baselines):
    """Section 6: identical target_date values, so alignment is a merge."""
    date_sets = [set(r["predictions"]["target_date"]) for r in all_baselines.values()]
    assert all(d == date_sets[0] for d in date_sets)


def test_align_predictions_is_a_trivial_merge(all_baselines):
    wide = align_predictions(all_baselines)
    assert len(wide) == len(next(iter(all_baselines.values()))["predictions"])
    for name in all_baselines:
        assert f"y_pred_{name}" in wide.columns
    assert wide["target_date"].is_monotonic_increasing


def test_align_predictions_rejects_mismatched_days(all_baselines):
    truncated = dict(all_baselines)
    short = truncated["zero_return"]["predictions"].iloc[:-5]
    truncated["zero_return"] = {"predictions": short}
    with pytest.raises(ContractViolation, match="target_date set differs"):
        align_predictions(truncated)


def test_align_predictions_can_fall_back_to_the_intersection(all_baselines):
    truncated = dict(all_baselines)
    short = truncated["zero_return"]["predictions"].iloc[:-5]
    truncated["zero_return"] = {"predictions": short}
    wide = align_predictions(truncated, require_identical=False)
    assert len(wide) == len(short)


class TestZeroReturn:
    def test_predicts_exactly_zero(self, all_baselines):
        assert (all_baselines["zero_return"]["predictions"]["y_pred"] == 0.0).all()

    def test_is_the_benchmark_models_must_beat(self, all_baselines, ds, folds):
        """Its R2_OOS against the historical mean is close to zero, which is
        what makes it the reference point rather than a strawman."""
        y_train_by_fold = {i: ds.y[train] for i, (train, _) in enumerate(folds)}
        result = evaluate_predictions(
            all_baselines["zero_return"]["predictions"],
            "Zero return",
            y_train_by_fold=y_train_by_fold,
        )
        assert abs(result["pooled"]["r2_oos"]) < 0.05

    def test_an_unseeded_benchmark_is_refused(self, all_baselines, ds, folds):
        """Why y_train_by_fold is not optional, and is now not omittable.

        This fixture's test period opens on the March 2020 crash. With no
        training history the expanding mean's first forecasts are one- and
        two-observation means of double-digit moves, the benchmark's SSE
        inflates, and the zero-return baseline scores +0.087 instead of the
        correct -0.005. That bias was found, fixed and flagged once, and it
        recurred at an aggregation call site that never read the flag. The
        evaluator now refuses to run without the seeds; the cold number is
        reproduced here by building the cold benchmark by hand, so the size
        of what the refusal prevents stays on record.
        """
        predictions = all_baselines["zero_return"]["predictions"]
        y_train_by_fold = {i: ds.y[train] for i, (train, _) in enumerate(folds)}

        with pytest.raises(TypeError):
            evaluate_predictions(predictions, "omitted", validate=False)  # type: ignore[call-arg]
        with pytest.raises(UnseededBenchmark):
            evaluate_predictions(predictions, "none", validate=False, y_train_by_fold=None)
        with pytest.raises(UnseededBenchmark):
            partial = {k: v for k, v in y_train_by_fold.items() if k != 0}
            evaluate_predictions(predictions, "partial", validate=False, y_train_by_fold=partial)
        with pytest.raises(UnseededBenchmark):
            aggregate_frame = predictions.copy()
            per_fold_metrics(aggregate_frame, y_train_by_fold=None)

        seeded = evaluate_predictions(predictions, "seeded", y_train_by_fold=y_train_by_fold)
        y_true = predictions["y_true"].to_numpy(float)
        cold_benchmark = np.concatenate([
            expanding_mean_benchmark(block["y_true"].to_numpy(float))
            for _, block in predictions.groupby("fold_id", sort=True)
        ])
        cold_r2 = r2_oos(y_true, predictions["y_pred"].to_numpy(float), cold_benchmark)
        assert cold_r2 > seeded["pooled"]["r2_oos"] + 0.05
        assert abs(seeded["pooled"]["r2_oos"]) < 0.05

    def test_directional_accuracy_is_the_down_day_rate(self, all_baselines):
        """Predicting zero is predicting "down" under the sign convention."""
        predictions = all_baselines["zero_return"]["predictions"]
        result = directional_accuracy(predictions["y_true"], predictions["y_pred"])
        assert result["predicted_rate_up"] == 0.0


class TestHistoricalMean:
    def test_uses_only_prior_returns(self, ds, folds):
        result = historical_mean_baseline(ds, folds)
        predictions = result["predictions"]
        first_fold = predictions[predictions["fold_id"] == 0]
        # The first test prediction is the training mean, exactly.
        assert first_fold["y_pred"].iloc[0] == pytest.approx(ds.y[folds[0][0]].mean())

    def test_predictions_are_small_and_stable(self, all_baselines):
        y_pred = all_baselines["historical_mean"]["predictions"]["y_pred"]
        assert np.abs(y_pred).max() < 0.01
        assert y_pred.std() < 0.001


class TestAr1:
    def test_recovers_a_planted_autocorrelation(self, ds, folds):
        """On real returns AR(1) is nearly flat, so test the mechanism on a
        dataset whose target genuinely is an AR(1)."""
        from dataclasses import replace

        rng = np.random.default_rng(0)
        y = np.zeros(len(ds))
        for i in range(1, len(y)):
            y[i] = 0.6 * y[i - 1] + rng.normal(0, 0.01)
        planted = replace(ds, y=y)

        result = ar1_baseline(planted, folds)
        predictions = result["predictions"]
        correlation = np.corrcoef(predictions["y_true"], predictions["y_pred"])[0, 1]
        assert correlation > 0.5

    def test_on_real_returns_it_is_nearly_flat(self, all_baselines):
        y_pred = all_baselines["ar1"]["predictions"]["y_pred"]
        assert np.abs(y_pred).max() < 0.05


class TestArima:
    def test_is_fitted_on_returns_not_prices(self, all_baselines):
        """A model fitted on prices would predict values near 100, not 0."""
        y_pred = all_baselines["arima"]["predictions"]["y_pred"]
        assert np.abs(y_pred).max() < 0.1

    def test_default_order_has_no_differencing(self, ds, folds):
        """Returns are already differenced; d=1 would over-difference."""
        import inspect

        signature = inspect.signature(arima_baseline)
        assert signature.parameters["order"].default == (5, 0, 0)


class TestRandomSign:
    def test_is_reproducible_given_a_seed(self, ds, folds):
        a = random_sign_baseline(ds, folds, seed=7)["predictions"]
        b = random_sign_baseline(ds, folds, seed=7)["predictions"]
        np.testing.assert_allclose(a["y_pred"], b["y_pred"])

    def test_lands_near_fifty_percent_directional_accuracy(self, all_baselines):
        predictions = all_baselines["random_sign"]["predictions"]
        result = directional_accuracy(predictions["y_true"], predictions["y_pred"])
        assert 0.35 < result["directional_accuracy"] < 0.65

    def test_only_the_sign_carries_information(self, all_baselines):
        y_pred = all_baselines["random_sign"]["predictions"]["y_pred"].to_numpy()
        assert len(np.unique(np.abs(np.round(y_pred, 10)))) <= len(
            set(all_baselines["random_sign"]["predictions"]["fold_id"])
        )


def test_unknown_baseline_is_rejected(ds, folds):
    with pytest.raises(KeyError, match="unknown baseline"):
        run_all_baselines(ds, folds, include=["not_a_baseline"])
