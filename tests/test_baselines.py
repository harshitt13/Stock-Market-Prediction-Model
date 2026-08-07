"""
Tests for baseline models (Naive and ARIMA).
"""

import numpy as np
import pytest
from baselines import naive_baseline, arima_baseline


class TestNaiveBaseline:
    def test_predict_yesterday(self):
        """Naive baseline should predict close[T] = close[T-1]."""
        prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
        dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03",
                          "2024-01-04", "2024-01-05"])
        train_idx = np.array([0, 1, 2])
        test_idx = np.array([3, 4])

        result = naive_baseline(prices, dates, train_idx, test_idx)

        # Predicted for index 3 should be price at index 2
        assert result["y_pred"][0] == prices[2]
        # Predicted for index 4 should be price at index 3
        assert result["y_pred"][1] == prices[3]
        assert len(result["y_true"]) == 2
        assert result["y_true"][0] == prices[3]
        assert result["y_true"][1] == prices[4]

    def test_metrics_present(self):
        """Test that metrics dict is returned."""
        prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0])
        dates = np.arange(6).astype(str)
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5])

        result = naive_baseline(prices, dates, train_idx, test_idx)
        assert "metrics" in result
        assert result["metrics"]["Model"] == "Naive (Zero-Change)"


class TestARIMABaseline:
    def test_runs_without_error(self):
        """ARIMA should produce predictions without crashing."""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200)) + 100
        dates = np.arange(200).astype(str)
        train_idx = np.arange(150)
        test_idx = np.arange(150, 200)

        result = arima_baseline(prices, dates, train_idx, test_idx)

        assert len(result["y_pred"]) == 50
        assert len(result["y_true"]) == 50
        assert "metrics" in result

    def test_fallback_on_bad_data(self):
        """ARIMA should fall back gracefully on bad data."""
        prices = np.array([1.0] * 20 + [2.0] * 10)  # constant then jump
        dates = np.arange(30).astype(str)
        train_idx = np.arange(20)
        test_idx = np.arange(20, 30)

        # Should not raise, may fall back to naive
        result = arima_baseline(prices, dates, train_idx, test_idx)
        assert len(result["y_pred"]) == 10

    def test_metrics_model_name(self):
        """Test that ARIMA metrics have correct model name."""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100)) + 100
        dates = np.arange(100).astype(str)
        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)

        result = arima_baseline(prices, dates, train_idx, test_idx)
        assert result["metrics"]["Model"] == "ARIMA (5,1,0)"
