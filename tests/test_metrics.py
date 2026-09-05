"""Tests for the return-space metrics (REFACTOR_PLAN.md section 4)."""

import numpy as np
import pandas as pd
import pytest

from contracts import ContractViolation, make_predictions
from evaluate import (
    UnseededBenchmark,
    DIRECTION_THRESHOLD,
    compare_evaluations,
    diebold_mariano,
    directional_accuracy,
    dm_table,
    evaluate_predictions,
    expanding_mean_benchmark,
    per_fold_metrics,
    pesaran_timmermann,
    price_space_metrics,
    r2_oos,
    return_error_metrics,
    summarize_across_folds,
)


def seeds_for(preds, rng=None, n: int = 200):
    """Training returns for every fold of a synthetic frame: the seed the
    R2_OOS benchmark now requires. Scale matches the synthetic returns."""
    rng = rng if rng is not None else np.random.default_rng(0)
    return {int(k): rng.normal(0.0, 0.02, n) for k in preds["fold_id"].unique()}


def predictions_frame(y_true, y_pred, fold_id=None, close=100.0) -> pd.DataFrame:
    n = len(y_true)
    return make_predictions(
        target_date=pd.date_range("2020-01-01", periods=n, freq="B"),
        fold_id=np.zeros(n, dtype=int) if fold_id is None else np.asarray(fold_id),
        close_t=np.full(n, close),
        y_true=np.asarray(y_true, dtype=float),
        y_pred=np.asarray(y_pred, dtype=float),
    )


# ---------------------------------------------------------------------------
# 4.1 Directional accuracy
# ---------------------------------------------------------------------------


class TestDirectionalAccuracy:
    def test_perfect_sign_prediction(self):
        y_true = np.array([0.02, -0.02, 0.03, -0.01])
        result = directional_accuracy(y_true, y_true * 0.5)
        assert result["directional_accuracy"] == 1.0
        assert result["n_evaluated"] == 4

    def test_always_wrong_sign(self):
        y_true = np.array([0.02, -0.02, 0.03, -0.015])
        result = directional_accuracy(y_true, -y_true)
        assert result["directional_accuracy"] == 0.0

    def test_magnitude_is_irrelevant_only_sign_matters(self):
        """A tiny positive prediction is still a positive prediction."""
        y_true = np.array([0.05, 0.05])
        big = directional_accuracy(y_true, np.array([9.9, 9.9]))
        tiny = directional_accuracy(y_true, np.array([1e-9, 1e-9]))
        assert big["directional_accuracy"] == tiny["directional_accuracy"] == 1.0

    def test_near_flat_days_are_excluded_and_reported(self):
        y_true = np.array([0.02, 0.0001, -0.0002, -0.03])
        result = directional_accuracy(y_true, y_true, threshold=DIRECTION_THRESHOLD)
        assert result["n_evaluated"] == 2
        assert result["n_excluded"] == 2
        assert result["excluded_fraction"] == 0.5

    def test_all_days_flat_gives_nan_not_a_crash(self):
        result = directional_accuracy(np.zeros(5), np.zeros(5))
        assert np.isnan(result["directional_accuracy"])
        assert result["n_evaluated"] == 0

    def test_exactly_zero_prediction_counts_as_down(self):
        result = directional_accuracy(np.array([-0.02]), np.array([0.0]))
        assert result["directional_accuracy"] == 1.0

    def test_confusion_matrix_is_two_by_two(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 0.02, 200)
        result = directional_accuracy(y_true, rng.normal(0, 0.02, 200))
        assert result["confusion_matrix"].shape == (2, 2)


def test_directional_accuracy_has_no_fold_boundary_artefact():
    """The bug section 4.1 exists to remove.

    The old definition differenced the concatenated prediction series, so
    joining two folds invented one extra observation straddling the boundary.
    Sign-based accuracy is per-row, so pooling folds is exactly the sum of the
    folds and nothing appears at the seam.
    """
    rng = np.random.default_rng(7)
    y_true = rng.normal(0, 0.02, 120)
    y_pred = rng.normal(0, 0.02, 120)
    fold_id = np.repeat([0, 1, 2], 40)

    preds = predictions_frame(y_true, y_pred, fold_id)
    pooled = directional_accuracy(y_true, y_pred)
    folds = per_fold_metrics(preds, y_train_by_fold=seeds_for(preds))

    assert pooled["n_evaluated"] == int(folds["n_evaluated"].sum())
    n_correct_pooled = pooled["directional_accuracy"] * pooled["n_evaluated"]
    n_correct_folds = (folds["directional_accuracy"] * folds["n_evaluated"]).sum()
    assert n_correct_pooled == pytest.approx(n_correct_folds)


# ---------------------------------------------------------------------------
# 4.2 Pesaran-Timmermann
# ---------------------------------------------------------------------------


class TestPesaranTimmermann:
    def test_independent_predictions_are_not_significant(self):
        rng = np.random.default_rng(1)
        result = pesaran_timmermann(rng.normal(0, 0.02, 500), rng.normal(0, 0.02, 500))
        assert result["pt_p_value"] > 0.05

    def test_perfect_prediction_is_highly_significant(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(0, 0.02, 300)
        result = pesaran_timmermann(y_true, y_true)
        assert result["pt_statistic"] > 5
        assert result["pt_p_value"] < 1e-6

    def test_high_hit_rate_on_a_one_sided_series_is_not_skill(self):
        """Always predicting up on a series that is 90% up scores 90% and
        should still fail the test, because it beats nothing."""
        rng = np.random.default_rng(3)
        y_true = np.abs(rng.normal(0, 0.02, 400))
        y_true[::10] *= -1  # 90% up
        y_pred = np.full(400, 0.01)  # always up

        naive_accuracy = directional_accuracy(y_true, y_pred)["directional_accuracy"]
        assert naive_accuracy > 0.85

        result = pesaran_timmermann(y_true, y_pred)
        # A constant predictor makes the statistic degenerate, which is the
        # honest answer: there is no directional information to test.
        assert np.isnan(result["pt_statistic"])

    def test_too_few_observations(self):
        result = pesaran_timmermann(np.array([0.01]), np.array([0.01]))
        assert np.isnan(result["pt_statistic"])


# ---------------------------------------------------------------------------
# 4.2 R2_OOS and the expanding-mean benchmark
# ---------------------------------------------------------------------------


class TestExpandingMeanBenchmark:
    def test_uses_only_data_before_each_point(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            expanding_mean_benchmark(y), [0.0, 1.0, 1.5, 2.0]
        )

    def test_training_history_seeds_the_mean(self):
        np.testing.assert_allclose(
            expanding_mean_benchmark(np.array([1.0, 3.0]), y_train=np.array([0.0, 0.0])),
            [0.0, 1.0 / 3.0],
        )

    def test_is_never_the_test_set_mean(self):
        """The test-set mean would use the future and be unbeatable."""
        y = np.array([0.05, -0.05, 0.05, -0.05])
        benchmark = expanding_mean_benchmark(y)
        assert not np.allclose(benchmark, y.mean())

    def test_final_value_excludes_the_final_observation(self):
        y = np.array([1.0, 1.0, 100.0])
        assert expanding_mean_benchmark(y)[-1] == 1.0


class TestR2Oos:
    def test_matching_the_benchmark_scores_zero(self):
        y_true = np.array([0.01, -0.02, 0.03])
        benchmark = expanding_mean_benchmark(y_true)
        assert r2_oos(y_true, benchmark, benchmark) == pytest.approx(0.0)

    def test_a_perfect_model_scores_one(self):
        y_true = np.array([0.01, -0.02, 0.03])
        assert r2_oos(y_true, y_true, expanding_mean_benchmark(y_true)) == 1.0

    def test_can_legitimately_be_negative(self):
        """Worse than the historical mean. The common case for daily returns."""
        y_true = np.array([0.01, -0.02, 0.03, -0.01])
        benchmark = expanding_mean_benchmark(y_true)
        assert r2_oos(y_true, -y_true * 5, benchmark) < 0


# ---------------------------------------------------------------------------
# 4.2 / 4.3 Error magnitude
# ---------------------------------------------------------------------------


def test_return_errors_are_reported_in_basis_points():
    y_true = np.array([0.0010, -0.0010])
    y_pred = np.zeros(2)
    metrics = return_error_metrics(y_true, y_pred)
    assert metrics["rmse_bps"] == pytest.approx(10.0)
    assert metrics["mae_bps"] == pytest.approx(10.0)


def test_price_metrics_never_include_r_squared():
    """Never put price R2 in the paper."""
    metrics = price_space_metrics(np.full(3, 100.0), np.zeros(3), np.zeros(3))
    assert set(metrics) == {"price_rmse", "price_mape_pct"}
    assert metrics["price_rmse"] == 0.0


def test_price_metrics_reconstruct_from_close_t():
    close_t = np.array([100.0, 200.0])
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([np.log(1.01), np.log(1.01)])
    metrics = price_space_metrics(close_t, y_true, y_pred)
    assert metrics["price_mape_pct"] == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 4.5 Diebold-Mariano
# ---------------------------------------------------------------------------


class TestDieboldMariano:
    def test_a_clearly_better_model_gives_a_negative_significant_statistic(self):
        rng = np.random.default_rng(4)
        errors_a = rng.normal(0, 0.001, 400)
        errors_b = rng.normal(0, 0.010, 400)
        result = diebold_mariano(errors_a, errors_b)
        assert result["dm_statistic"] < 0
        assert result["dm_p_value"] < 0.01
        assert result["favours"] == "A"

    def test_indistinguishable_models_are_not_significant(self):
        rng = np.random.default_rng(5)
        result = diebold_mariano(rng.normal(0, 0.01, 500), rng.normal(0, 0.01, 500))
        assert result["dm_p_value"] > 0.05

    def test_identical_errors_are_degenerate_not_significant(self):
        errors = np.random.default_rng(6).normal(0, 0.01, 100)
        result = diebold_mariano(errors, errors)
        assert np.isnan(result["dm_statistic"])

    def test_newey_west_lag_widens_the_standard_error(self):
        """Serially correlated loss differentials must not look more
        significant than they are.

        Uses a persistent AR(1) so the loss differential really is
        autocorrelated; on iid errors the lag terms are just noise and the
        correction is entitled to move the statistic either way.
        """
        rng = np.random.default_rng(8)
        n, rho = 400, 0.95
        errors_b = np.zeros(n)
        for i in range(1, n):
            errors_b[i] = rho * errors_b[i - 1] + rng.normal(0, 0.01)
        errors_a = errors_b * 0.5

        stats_by_lag = [
            abs(diebold_mariano(errors_a, errors_b, h=h)["dm_statistic"])
            for h in (1, 2, 5, 10)
        ]
        assert stats_by_lag == sorted(stats_by_lag, reverse=True)

    def test_misaligned_series_is_an_error(self):
        with pytest.raises(ValueError, match="same length"):
            diebold_mariano(np.zeros(10), np.zeros(11))

    def test_absolute_loss_is_supported(self):
        rng = np.random.default_rng(9)
        result = diebold_mariano(
            rng.normal(0, 0.001, 300), rng.normal(0, 0.01, 300), loss="absolute"
        )
        assert result["dm_statistic"] < 0

    def test_unknown_loss_is_rejected(self):
        with pytest.raises(ValueError, match="squared"):
            diebold_mariano(np.zeros(5), np.zeros(5), loss="huber")


def test_dm_table_merges_on_target_date():
    rng = np.random.default_rng(10)
    y_true = rng.normal(0, 0.02, 100)
    good = predictions_frame(y_true, y_true * 0.9)
    bad = predictions_frame(y_true, rng.normal(0, 0.02, 100))

    table = dm_table({"good": good, "zero": bad}, reference="zero")
    assert list(table.index) == ["good"]
    assert table.loc["good", "n shared days"] == 100


# ---------------------------------------------------------------------------
# 4.4 Per-fold statistics
# ---------------------------------------------------------------------------


class TestPerFoldStatistics:
    def test_one_row_per_fold(self):
        rng = np.random.default_rng(11)
        preds = predictions_frame(
            rng.normal(0, 0.02, 90), rng.normal(0, 0.02, 90), np.repeat([0, 1, 2], 30)
        )
        folds = per_fold_metrics(preds, y_train_by_fold=seeds_for(preds))
        assert list(folds["fold_id"]) == [0, 1, 2]
        assert (folds["n"] == 30).all()

    def test_summary_reports_mean_and_std_across_folds(self):
        rng = np.random.default_rng(12)
        preds = predictions_frame(
            rng.normal(0, 0.02, 80), rng.normal(0, 0.02, 80), np.repeat([0, 1], 40)
        )
        summary = summarize_across_folds(per_fold_metrics(preds, y_train_by_fold=seeds_for(preds)))
        assert summary["n_folds"] == 2
        assert "directional_accuracy_mean" in summary
        assert "directional_accuracy_std" in summary
        assert "r2_oos_std" in summary

    def test_a_single_fold_has_zero_std_not_nan(self):
        rng = np.random.default_rng(13)
        preds = predictions_frame(rng.normal(0, 0.02, 40), rng.normal(0, 0.02, 40))
        summary = summarize_across_folds(per_fold_metrics(preds, y_train_by_fold=seeds_for(preds)))
        assert summary["directional_accuracy_std"] == 0.0


# ---------------------------------------------------------------------------
# Whole-model evaluation
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    def test_returns_pooled_per_fold_and_summary(self):
        rng = np.random.default_rng(14)
        preds = predictions_frame(
            rng.normal(0, 0.02, 60), rng.normal(0, 0.02, 60), np.repeat([0, 1], 30)
        )
        result = evaluate_predictions(preds, "Test Model", y_train_by_fold=seeds_for(preds))

        assert result["model"] == "Test Model"
        assert result["n_predictions"] == 60
        assert set(result) >= {"pooled", "per_fold", "summary"}
        assert len(result["per_fold"]) == 2
        assert "pt_statistic" in result["pooled"]

    def test_enforces_the_prediction_contract(self):
        rng = np.random.default_rng(15)
        preds = predictions_frame(rng.normal(0, 0.02, 20), rng.normal(0, 0.02, 20))
        preds.loc[3, "y_pred"] = np.nan
        with pytest.raises(ContractViolation):
            evaluate_predictions(preds, "Broken", y_train_by_fold=seeds_for(preds))

    def test_refuses_to_run_without_training_returns(self):
        """The benchmark seed is structural, not a flag: omitting it is a
        TypeError at the call site, and passing None or an incomplete mapping
        raises before any metric is computed."""
        rng = np.random.default_rng(18)
        preds = predictions_frame(rng.normal(0, 0.02, 60), rng.normal(0, 0.02, 60), np.repeat([0, 1], 30))
        with pytest.raises(TypeError):
            evaluate_predictions(preds, "Omitted")  # type: ignore[call-arg]
        with pytest.raises(UnseededBenchmark):
            evaluate_predictions(preds, "None", y_train_by_fold=None)
        with pytest.raises(UnseededBenchmark):
            evaluate_predictions(preds, "Partial", y_train_by_fold={0: rng.normal(0, 0.02, 50)})
        with pytest.raises(UnseededBenchmark):
            evaluate_predictions(preds, "Empty", y_train_by_fold={0: np.array([]), 1: np.array([])})
        assert "unseeded_benchmark" not in evaluate_predictions(preds, "ok", y_train_by_fold=seeds_for(preds))

    def test_zero_return_baseline_scores_r2_oos_near_zero(self):
        """The benchmark every model has to beat, evaluated against itself."""
        rng = np.random.default_rng(16)
        y_true = rng.normal(0.0002, 0.015, 500)
        preds = predictions_frame(y_true, np.zeros(500), np.repeat([0, 1], 250))
        result = evaluate_predictions(preds, "Zero return", y_train_by_fold=seeds_for(preds, rng))
        assert abs(result["pooled"]["r2_oos"]) < 0.02

    def test_comparison_table_labels_price_metrics_as_secondary(self, capsys):
        rng = np.random.default_rng(17)
        preds = predictions_frame(rng.normal(0, 0.02, 50), rng.normal(0, 0.02, 50))
        table = compare_evaluations([evaluate_predictions(preds, "A", y_train_by_fold=seeds_for(preds))])

        assert "Price RMSE [2nd]" in table.columns
        assert "R2_OOS" in table.columns
        assert not any("Price R2" in c for c in table.columns)
        assert "do not measure forecasting skill" in capsys.readouterr().out
