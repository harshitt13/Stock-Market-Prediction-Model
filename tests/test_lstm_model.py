"""
Tests for the BiLSTM model (updated for walk-forward API).
"""

import unittest
import os
import numpy as np
import pandas as pd
import torch
from lstm_model import train_lstm_model


class TestTrainLSTMModel(unittest.TestCase):
    def setUp(self):
        # Need enough data for time_step=90 + train + test
        np.random.seed(42)
        n = 250
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
            }
        )

        self.model_dir = "models"
        self.data_dir = "data"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, "lstm_model.pt")
        self.predictions_path = os.path.join(
            self.data_dir, "future_predictions_lstm.csv"
        )

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.predictions_path):
            os.remove(self.predictions_path)

    def test_train_lstm_model_default(self):
        """Test with default split (backward compatible)."""
        result = train_lstm_model(self.test_data, future_days=5)

        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.predictions_path))
        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIn("future_predictions", result)
        self.assertEqual(len(result["future_predictions"]), 5)

    def test_train_lstm_with_folds(self):
        """Test with explicit walk-forward folds."""
        n = len(self.test_data)
        fold_indices = [
            (np.arange(0, 150), np.arange(150, 200)),
            (np.arange(0, 200), np.arange(200, n)),
        ]
        result = train_lstm_model(
            self.test_data, future_days=5, fold_indices=fold_indices
        )

        self.assertIn("y_test", result)
        self.assertIn("y_pred", result)
        self.assertIn("per_fold_metrics", result)

    def test_classification_metrics_present(self):
        """Test that classification metrics exist in output."""
        result = train_lstm_model(self.test_data, future_days=5)
        metrics = result["metrics"]
        self.assertIn("Directional Accuracy (%)", metrics)
        self.assertIn("F1 Score", metrics)


if __name__ == "__main__":
    unittest.main()
