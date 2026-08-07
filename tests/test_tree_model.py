"""
Tests for the tree ensemble model (updated for walk-forward API).
"""

import unittest
import os
import numpy as np
import pandas as pd
import joblib
from tree_model import train_tree_model


class TestTrainTreeModel(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with enough data for walk-forward
        np.random.seed(42)
        n = 200
        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        close = np.cumsum(np.random.randn(n)) + 150

        self.test_data = pd.DataFrame(
            {
                "Date": dates,
                "Close": close,
                "High": close + np.abs(np.random.randn(n)),
                "Low": close - np.abs(np.random.randn(n)),
                "Open": close + np.random.randn(n) * 0.5,
                "Volume": np.random.randint(500000, 2000000, n),
                "SMA_20": pd.Series(close).rolling(20).mean().bfill(),
                "SMA_50": pd.Series(close).rolling(50).mean().bfill(),
                "RSI_14": 50 + np.random.randn(n) * 10,
                "Log_Return": np.random.randn(n) * 0.01,
                "DayOfWeek": dates.dayofweek,
            }
        )

        self.model_dir = "models"
        self.data_dir = "data"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, "tree_ensemble_model.pkl")
        self.predictions_path = os.path.join(
            self.data_dir, "future_predictions_tree.csv"
        )

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.predictions_path):
            os.remove(self.predictions_path)

    def test_train_tree_model_default_split(self):
        """Test with default (no fold_indices → single 80/20 split)."""
        result = train_tree_model(self.test_data, future_days=5)

        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.predictions_path))
        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIn("future_predictions", result)
        self.assertEqual(len(result["future_predictions"]), 5)

    def test_train_tree_model_with_folds(self):
        """Test with explicit walk-forward fold indices."""
        n = len(self.test_data) - 1  # -1 because target shifts
        fold_indices = [
            (np.arange(0, 100), np.arange(100, 150)),
            (np.arange(0, 150), np.arange(150, n)),
        ]
        result = train_tree_model(
            self.test_data, future_days=5, fold_indices=fold_indices
        )

        self.assertIn("y_test", result)
        self.assertIn("y_pred", result)
        self.assertEqual(len(result["y_test"]), len(result["y_pred"]))
        self.assertIn("per_fold_metrics", result)

    def test_classification_metrics_in_output(self):
        """Test that classification metrics are present in output."""
        result = train_tree_model(self.test_data, future_days=5)
        metrics = result["metrics"]
        self.assertIn("Directional Accuracy (%)", metrics)
        self.assertIn("F1 Score", metrics)


if __name__ == "__main__":
    unittest.main()
