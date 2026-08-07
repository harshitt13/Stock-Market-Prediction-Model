"""
Tests for the updated evaluate module (classification-primary metrics).
"""

import numpy as np
import pytest
from evaluate import (
    compute_metrics,
    compute_classification_metrics,
    compute_regression_metrics,
    derive_direction_labels,
)


class TestDeriveDirectionLabels:
    def test_clear_up_down(self):
        """Test with clear up and down movements."""
        prices = np.array([100.0, 101.0, 99.0, 102.0])
        labels, mask = derive_direction_labels(prices, threshold=0.001)
        # Returns: diff=[1, -2, 3], returns=[0.01, -0.0198, 0.0303]
        assert labels[0] == 1   # up
        assert labels[1] == 0   # down
        assert labels[2] == 1   # up
        assert mask.all()       # all above threshold

    def test_sub_threshold_masked(self):
        """Test that sub-threshold moves are masked."""
        prices = np.array([100.0, 100.05, 100.10])  # ~0.05% moves
        labels, mask = derive_direction_labels(prices, threshold=0.001)
        # 0.05% < 0.1% threshold
        assert not mask.all()   # some should be masked

    def test_single_price(self):
        """No direction can be derived from a single price."""
        prices = np.array([100.0])
        labels, mask = derive_direction_labels(prices)
        assert len(labels) == 0


class TestComputeClassificationMetrics:
    def test_perfect_prediction(self):
        """Metrics when prediction perfectly matches actual direction."""
        y_true = np.array([100, 102, 101, 104, 103])
        y_pred = np.array([100, 102, 101, 104, 103])  # identical

        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["Directional Accuracy (%)"] == 100.0

    def test_opposite_prediction(self):
        """Metrics when prediction is always wrong direction."""
        y_true = np.array([100, 102, 104, 106])  # always up
        y_pred = np.array([100, 98, 96, 94])       # always down

        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["Directional Accuracy (%)"] == 0.0

    def test_short_series(self):
        """Handle series too short for direction."""
        metrics = compute_classification_metrics(
            np.array([100.0]), np.array([101.0])
        )
        assert np.isnan(metrics["Directional Accuracy (%)"])

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 2x2."""
        y_true = np.array([100, 102, 101, 104, 103, 105])
        y_pred = np.array([100, 103, 102, 103, 104, 106])
        metrics = compute_classification_metrics(y_true, y_pred)
        cm = metrics["Confusion Matrix"]
        assert cm is not None
        assert cm.shape == (2, 2)


class TestComputeRegressionMetrics:
    def test_perfect_prediction(self):
        """Metrics when prediction is perfect."""
        y = np.array([100.0, 200.0, 300.0])
        metrics = compute_regression_metrics(y, y)
        assert metrics["RMSE"] == 0.0
        assert metrics["MAE"] == 0.0
        assert metrics["R2"] == 1.0

    def test_known_error(self):
        """Test with a known constant error."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([101.0, 201.0, 301.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["MAE"] == 1.0
        assert metrics["RMSE"] == 1.0


class TestComputeMetrics:
    def test_combined_metrics_keys(self):
        """Test that combined metrics contain both classification and regression keys."""
        y_true = np.array([100, 102, 101, 104, 103, 105])
        y_pred = np.array([100, 103, 102, 103, 104, 106])
        metrics = compute_metrics(y_true, y_pred)

        # Classification keys
        assert "Directional Accuracy (%)" in metrics
        assert "F1 Score" in metrics
        assert "Precision" in metrics
        assert "Recall" in metrics
        assert "Confusion Matrix" in metrics

        # Regression keys
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "MAPE (%)" in metrics
        assert "R2" in metrics

    def test_model_name_propagated(self):
        """Test model name is set correctly."""
        metrics = compute_metrics(
            np.array([100, 101, 102]),
            np.array([100, 101, 102]),
            model_name="TestModel",
        )
        assert metrics["Model"] == "TestModel"
