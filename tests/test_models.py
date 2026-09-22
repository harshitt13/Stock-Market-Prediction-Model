"""Tests for the three model modules (REFACTOR_PLAN.md section 5).

Every model is checked against the same contract, and the cross-model
alignment test below is the one that would have caught the original
off-by-one: it asserts the tree and the sequence models name exactly the same
forecast days.

Kept fast deliberately -- a short lookback, few epochs, a slice of the
fixture. These assert the plumbing, not predictive performance.
"""


import numpy as np
import pandas as pd
import pytest

from contracts import PREDICTION_COLUMNS, validate_predictions
from dataset import build_dataset
from fetch_data import engineer_features
from lstm_model import train_lstm_model
from model_utils import (
    DEMO_FORECAST_CAVEAT,
    apply_feature_scaler,
    default_folds,
    fit_feature_scaler,
    fit_target_scaler,
    scale_targets,
    unscale_targets,
)
from transformer_model import train_transformer_model
from tree_model import train_tree_model
from conftest import FIXTURE_PATH


LOOKBACK = 20
EPOCHS = 3
N_ROWS = 360


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def engineered(raw) -> pd.DataFrame:
    return engineer_features(raw).dropna().reset_index(drop=True).iloc[:N_ROWS].copy()


@pytest.fixture(scope="module")
def ds(engineered):
    return build_dataset(engineered)


@pytest.fixture(scope="module")
def folds(ds):
    """Two non-overlapping test folds, expanding training window."""
    return [
        (np.arange(0, 200), np.arange(200, 260)),
        (np.arange(0, 260), np.arange(260, len(ds))),
    ]


@pytest.fixture(scope="module")
def tree_result(ds, folds):
    return train_tree_model(ds, folds, save_model=False)


@pytest.fixture(scope="module")
def lstm_result(ds, folds):
    return train_lstm_model(
        ds, folds, lookback=LOOKBACK, epochs=EPOCHS, save_model=False, verbose=False
    )


@pytest.fixture(scope="module")
def transformer_result(ds, folds):
    return train_transformer_model(
        ds, folds, lookback=LOOKBACK, epochs=EPOCHS, save_model=False, verbose=False
    )


# ---------------------------------------------------------------------------
# The contract, for every model
# ---------------------------------------------------------------------------


@pytest.fixture(params=["tree_result", "lstm_result", "transformer_result"])
def any_result(request):
    return request.getfixturevalue(request.param)


class TestEveryModelSatisfiesTheContract:
    def test_returns_a_valid_standard_frame(self, any_result):
        predictions = any_result["predictions"]
        assert list(predictions.columns) == PREDICTION_COLUMNS
        validate_predictions(predictions)

    def test_predictions_are_keyed_by_target_date_not_feature_date(
        self, any_result, ds, folds
    ):
        predictions = any_result["predictions"]
        expected = set(ds.target_date[np.concatenate([f[1] for f in folds])])
        assert set(predictions["target_date"]) <= expected
        # and never a feature_date that is not also a target_date
        assert not set(predictions["target_date"]) & set(ds.feature_date[:1])

    def test_y_true_matches_the_dataset_target(self, any_result, ds):
        predictions = any_result["predictions"]
        lookup = dict(zip(ds.target_date, ds.y))
        expected = np.array([lookup[d.to_datetime64()] for d in predictions["target_date"]])
        np.testing.assert_allclose(predictions["y_true"], expected)

    def test_close_t_matches_the_dataset(self, any_result, ds):
        predictions = any_result["predictions"]
        lookup = dict(zip(ds.target_date, ds.close_t))
        expected = np.array(
            [lookup[d.to_datetime64()] for d in predictions["target_date"]]
        )
        np.testing.assert_allclose(predictions["close_t"], expected)

    def test_predictions_are_returns_not_prices(self, any_result):
        """A model that quietly went back to price space is caught here."""
        y_pred = any_result["predictions"]["y_pred"].to_numpy()
        assert np.abs(y_pred).max() < 1.0, "predictions look like prices, not returns"

    def test_every_fold_contributes(self, any_result, folds):
        assert set(any_result["predictions"]["fold_id"]) == set(range(len(folds)))

    def test_no_duplicate_forecast_days_across_folds(self, any_result):
        predictions = any_result["predictions"]
        assert not predictions["target_date"].duplicated().any()


def test_all_three_models_forecast_exactly_the_same_days(
    tree_result, lstm_result, transformer_result
):
    """The test that kills the original off-by-one.

    A sequence model consuming a 20-day window and a tree consuming one row
    must still name the same forecast days, because both key on target_date.
    Under the old code the tree and the sequence models were offset by the
    lookback and the misalignment was invisible.
    """
    days = {
        name: set(result["predictions"]["target_date"])
        for name, result in [
            ("tree", tree_result),
            ("lstm", lstm_result),
            ("transformer", transformer_result),
        ]
    }
    assert days["tree"] == days["lstm"] == days["transformer"]


def test_merging_two_models_on_target_date_loses_nothing(tree_result, lstm_result):
    merged = tree_result["predictions"].merge(
        lstm_result["predictions"], on="target_date", how="inner", suffixes=("_a", "_b")
    )
    assert len(merged) == len(tree_result["predictions"])
    # Same day, same realised return, whichever model reported it.
    np.testing.assert_allclose(merged["y_true_a"], merged["y_true_b"])


# ---------------------------------------------------------------------------
# Model-specific behaviour
# ---------------------------------------------------------------------------


class TestTreeModel:
    def test_reports_feature_importances_per_feature(self, tree_result, ds):
        assert len(tree_result["feature_importances"]) == ds.n_features
        assert len(tree_result["mean_feature_importances"]) == ds.n_features

    def test_has_no_scaler(self, tree_result):
        """Standardising inputs does nothing for a tree; it should be gone."""
        assert "scaler" not in tree_result

    def test_falls_back_to_a_single_chronological_split(self, ds):
        result = train_tree_model(ds, fold_indices=None, save_model=False)
        assert set(result["predictions"]["fold_id"]) == {0}
        validate_predictions(result["predictions"])


class TestSequenceModels:
    def test_lstm_test_fold_keeps_full_length(self, lstm_result, folds):
        """Windows reach back into training rows, so nothing is lost."""
        n_test = sum(len(test) for _, test in folds)
        assert len(lstm_result["predictions"]) == n_test

    def test_transformer_test_fold_keeps_full_length(self, transformer_result, folds):
        n_test = sum(len(test) for _, test in folds)
        assert len(transformer_result["predictions"]) == n_test

    def test_scalers_are_standard_not_minmax(self, lstm_result, transformer_result):
        from sklearn.preprocessing import StandardScaler

        assert isinstance(lstm_result["x_scaler"], StandardScaler)
        assert isinstance(transformer_result["x_scaler"], StandardScaler)


# ---------------------------------------------------------------------------
# Fold scaling
# ---------------------------------------------------------------------------


class TestFoldScaling:
    def test_feature_scaler_sees_only_training_rows(self, ds):
        train_idx = np.arange(0, 200)
        scaler = fit_feature_scaler(ds, train_idx)
        np.testing.assert_allclose(scaler.mean_, ds.X[train_idx].mean(axis=0))

    def test_scaling_leaves_the_keys_untouched(self, ds):
        scaler = fit_feature_scaler(ds, np.arange(0, 200))
        scaled = apply_feature_scaler(ds, scaler)
        np.testing.assert_array_equal(scaled.target_date, ds.target_date)
        np.testing.assert_allclose(scaled.y, ds.y)
        np.testing.assert_allclose(scaled.close_t, ds.close_t)

    def test_target_scaling_round_trips(self, ds):
        scaler = fit_target_scaler(ds.y[:200])
        np.testing.assert_allclose(
            unscale_targets(scaler, scale_targets(scaler, ds.y)), ds.y, atol=1e-12
        )

    def test_target_scaler_fitted_on_train_only(self, ds):
        scaler = fit_target_scaler(ds.y[:200])
        assert scaler.mean_[0] == pytest.approx(ds.y[:200].mean())


def test_default_folds_is_one_chronological_split():
    folds = default_folds(100)
    assert len(folds) == 1
    train, test = folds[0]
    assert train.max() < test.min()


# ---------------------------------------------------------------------------
# The demo forecast
# ---------------------------------------------------------------------------


class TestDemoForecast:
    def test_is_off_by_default(self, tree_result):
        assert "demo_forecast" not in tree_result

    def test_carries_the_caveat_and_stays_out_of_the_metrics_frame(
        self, ds, folds, engineered, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = train_tree_model(
            ds, folds, save_model=False, demo_forecast_days=3, demo_history=engineered
        )
        forecast = result["demo_forecast"]

        assert len(forecast) == 3
        assert (forecast["caveat"] == DEMO_FORECAST_CAVEAT).all()
        assert "ILLUSTRATIVE ONLY" in DEMO_FORECAST_CAVEAT
        # It must not have leaked into the evaluated predictions.
        assert set(forecast["date"]).isdisjoint(set(result["predictions"]["target_date"]))
        validate_predictions(result["predictions"])

    def test_reconstructs_price_from_the_predicted_return(
        self, ds, folds, engineered, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = train_tree_model(
            ds, folds, save_model=False, demo_forecast_days=2, demo_history=engineered
        )
        forecast = result["demo_forecast"]
        last_close = engineered["Close"].iloc[-1]
        expected = last_close * np.exp(forecast["predicted_log_return"].iloc[0])
        assert forecast["Predicted Close Tree"].iloc[0] == pytest.approx(expected)

    def test_requires_history(self, ds, folds):
        with pytest.raises(ValueError, match="demo_history"):
            train_tree_model(ds, folds, save_model=False, demo_forecast_days=2)
