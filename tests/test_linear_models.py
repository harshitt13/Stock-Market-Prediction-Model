"""The two linear comparators keep the contract and record their fits."""

import numpy as np
import pytest

from conftest import load_fixture
from contracts import validate_predictions
from dataset import build_dataset
from linear_models import (
    LOGISTIC_NAME, RIDGE_ALPHAS, RIDGE_NAME, run_linear_comparators, train_logistic_model, train_ridge_model,
)
from walk_forward import WalkForwardSplitter


@pytest.fixture(scope="module")
def ds():
    return build_dataset(load_fixture())


@pytest.fixture(scope="module")
def folds(ds):
    return WalkForwardSplitter(400, 60, 60).split(len(ds))


@pytest.fixture(scope="module")
def results(ds, folds):
    return run_linear_comparators(ds, folds, seed=0)


class TestContract:
    def test_both_models_emit_the_standard_frame(self, results):
        assert set(results) == {"ridge", "logistic"}
        for result in results.values():
            validate_predictions(result["predictions"], name=result["model_name"])

    def test_forecast_days_match_the_tree_and_baselines(self, results, ds, folds):
        expected = np.concatenate([ds.target_date[test] for _, test in folds])
        for result in results.values():
            assert np.array_equal(result["predictions"]["target_date"].to_numpy(), expected)

    def test_names(self, results):
        assert results["ridge"]["model_name"] == RIDGE_NAME
        assert results["logistic"]["model_name"] == LOGISTIC_NAME


class TestRidge:
    def test_records_the_selected_penalty_per_fold(self, results, folds):
        fits = results["ridge"]["fold_fits"]
        assert len(fits) == len(folds)
        assert set(fits["alpha"]).issubset(set(RIDGE_ALPHAS))
        assert (fits["n_train"].diff().dropna() > 0).all()

    def test_is_deterministic(self, ds, folds):
        a = train_ridge_model(ds, folds)["predictions"]["y_pred"].to_numpy()
        b = train_ridge_model(ds, folds)["predictions"]["y_pred"].to_numpy()
        assert np.array_equal(a, b)

    def test_records_dispersion_relative_to_training(self, results):
        fits = results["ridge"]["fold_fits"]
        assert "pred_sd_over_train_sd" in fits.columns
        assert (fits["pred_sd_over_train_sd"] >= 0).all()


class TestLogistic:
    def test_output_is_confidence_at_training_scale(self, results, ds, folds):
        preds = results["logistic"]["predictions"]
        for fold_id, (train, _) in enumerate(folds):
            scale = float(np.std(ds.y[train]))
            block = preds[preds["fold_id"] == fold_id]["y_pred"].to_numpy()
            assert np.all(np.abs(block) <= scale + 1e-12)

    def test_records_the_selected_inverse_penalty(self, results, folds):
        fits = results["logistic"]["fold_fits"]
        assert len(fits) == len(folds)
        assert "C" in fits.columns and "p_up_mean" in fits.columns

    def test_single_class_training_fold_predicts_the_majority_sign(self, ds):
        n = 80
        folds = [(np.arange(0, 60), np.arange(60, n))]
        from dataclasses import replace

        constant = replace(ds, y=np.abs(ds.y[:n]) + 1e-4, X=ds.X[:n], target_date=ds.target_date[:n],
                           feature_date=ds.feature_date[:n], close_t=ds.close_t[:n])
        result = train_logistic_model(constant, folds)
        assert (result["predictions"]["y_pred"] > 0).all()
        assert bool(result["fold_fits"]["single_class"].iloc[0])
