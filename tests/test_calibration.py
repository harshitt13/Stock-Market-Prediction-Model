"""
Tests for confidence interval calibration.
"""

import numpy as np
import pytest
from calibration import (
    compute_volatility_adjusted_ci,
    calibrate_confidence_interval,
)


class TestComputeVolatilityAdjustedCI:
    def test_basic_ci(self):
        """Test that CI bounds are symmetric around predictions."""
        preds = np.array([100.0, 200.0, 300.0])
        lower, upper, dynamic_std = compute_volatility_adjusted_ci(
            preds, residual_std=10.0, vix_current=20.0,
            vix_historical_mean=20.0, confidence_level=0.95,
        )
        assert len(lower) == 3
        assert len(upper) == 3
        # Symmetric
        np.testing.assert_array_almost_equal(
            preds - lower, upper - preds, decimal=5
        )
        # Width should be approximately 1.96 * 10
        expected_half_width = 1.96 * 10.0
        np.testing.assert_almost_equal(upper[0] - preds[0], expected_half_width, decimal=1)

    def test_high_volatility_widens(self):
        """CI should be wider when VIX is above historical mean."""
        preds = np.array([100.0])
        _, upper_normal, _ = compute_volatility_adjusted_ci(
            preds, residual_std=10.0, vix_current=20.0,
            vix_historical_mean=20.0,
        )
        _, upper_high, _ = compute_volatility_adjusted_ci(
            preds, residual_std=10.0, vix_current=40.0,
            vix_historical_mean=20.0,
        )
        width_normal = upper_normal[0] - preds[0]
        width_high = upper_high[0] - preds[0]
        assert width_high > width_normal

    def test_low_volatility_floor(self):
        """CI should not shrink below base residual_std (multiplier floor = 1.0)."""
        preds = np.array([100.0])
        _, upper_low, _ = compute_volatility_adjusted_ci(
            preds, residual_std=10.0, vix_current=10.0,
            vix_historical_mean=20.0,
        )
        _, upper_normal, _ = compute_volatility_adjusted_ci(
            preds, residual_std=10.0, vix_current=20.0,
            vix_historical_mean=20.0,
        )
        # Floor multiplier = max(1.0, 10/20=0.5) = 1.0
        width_low = upper_low[0] - preds[0]
        width_normal = upper_normal[0] - preds[0]
        np.testing.assert_almost_equal(width_low, width_normal)


class TestCalibrateConfidenceInterval:
    def test_perfect_coverage(self):
        """All values inside interval → 100% coverage."""
        y_true = np.array([100.0, 200.0, 300.0])
        lower = np.array([90.0, 190.0, 290.0])
        upper = np.array([110.0, 210.0, 310.0])

        # nominal=0.97 so 100% coverage (3pp diff) is within ±5pp tolerance
        result = calibrate_confidence_interval(y_true, lower, upper, nominal_level=0.97)
        assert result["empirical_coverage"] == 1.0
        assert result["n_inside"] == 3
        assert result["is_well_calibrated"]

    def test_over_covered_large_sample(self):
        """100% coverage with many samples → OVER-COVERED."""
        n = 100
        y_true = np.arange(n, dtype=float)
        lower = y_true - 10
        upper = y_true + 10

        result = calibrate_confidence_interval(y_true, lower, upper, nominal_level=0.50)
        assert result["empirical_coverage"] == 1.0
        assert result["flag"] == "⚠️ OVER-COVERED"

    def test_zero_coverage(self):
        """No values inside interval → 0% coverage."""
        y_true = np.array([100.0, 200.0, 300.0])
        lower = np.array([200.0, 300.0, 400.0])
        upper = np.array([210.0, 310.0, 410.0])

        result = calibrate_confidence_interval(y_true, lower, upper, nominal_level=0.95)
        assert result["empirical_coverage"] == 0.0
        assert result["flag"] == "⚠️ UNDER-COVERED"

    def test_well_calibrated(self):
        """Test coverage within tolerance is flagged as OK."""
        # 95 out of 100 inside → 95% coverage
        np.random.seed(42)
        y_true = np.random.randn(100) * 10 + 100
        lower = y_true - 50
        upper = y_true + 50
        # Force 5 outside
        lower[:5] = y_true[:5] + 1
        upper[:5] = y_true[:5] + 2

        result = calibrate_confidence_interval(y_true, lower, upper, nominal_level=0.95)
        assert result["empirical_coverage"] == 0.95
        assert result["is_well_calibrated"]
        assert result["flag"] == "✅ OK"

    def test_edge_boundary(self):
        """Test that values exactly on bounds count as inside."""
        y_true = np.array([100.0])
        lower = np.array([100.0])
        upper = np.array([100.0])

        result = calibrate_confidence_interval(y_true, lower, upper)
        assert result["empirical_coverage"] == 1.0
