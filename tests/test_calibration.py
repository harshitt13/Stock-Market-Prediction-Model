"""Tests for src/calibration.py.

These were stale: they asserted an older API (a scalar ``residual_std``, and
emoji status flags) that the empirical-quantile rewrite replaced without the
tests being updated. They are brought in line with the module as it stands.

Note that the pipeline no longer calls this module. REFACTOR_PLAN.md section
7.4 replaced the volatility-multiplier approach with fold-respecting residual
quantiles and sqrt(h) horizon scaling, which live in ``meta_ensemble`` and are
covered by ``test_meta_ensemble.py``. This file keeps the module honest for as
long as it stays in the repository.
"""

import numpy as np
import pytest

from calibration import calibrate_confidence_interval, compute_volatility_adjusted_ci


def symmetric_residuals(scale: float = 10.0, n: int = 4001) -> np.ndarray:
    """Residuals whose 2.5th/97.5th percentiles are about +/- 1.96 * scale."""
    return np.linspace(-1, 1, n) * scale * 1.96 / 0.95


class TestComputeVolatilityAdjustedCI:
    def test_bounds_come_from_empirical_residual_quantiles(self):
        preds = np.array([100.0, 200.0, 300.0])
        residuals = symmetric_residuals()

        lower, upper, offsets = compute_volatility_adjusted_ci(
            preds, residuals, vix_current=20.0, vix_historical_mean=20.0,
            confidence_level=0.95,
        )

        assert len(lower) == len(upper) == 3
        np.testing.assert_allclose(offsets[0], np.percentile(residuals, 2.5))
        np.testing.assert_allclose(offsets[1], np.percentile(residuals, 97.5))
        # Symmetric residuals give symmetric bounds.
        np.testing.assert_allclose(preds - lower, upper - preds, rtol=1e-6)

    def test_high_volatility_widens_the_interval(self):
        preds = np.array([100.0])
        residuals = symmetric_residuals()

        _, normal, _ = compute_volatility_adjusted_ci(preds, residuals, 20.0, 20.0)
        _, high, _ = compute_volatility_adjusted_ci(preds, residuals, 40.0, 20.0)
        assert high[0] - preds[0] > normal[0] - preds[0]

    def test_low_volatility_narrows_but_is_floored(self):
        """The multiplier is clipped to [0.5, 2.0]."""
        preds = np.array([100.0])
        residuals = symmetric_residuals()

        _, low_vix, _ = compute_volatility_adjusted_ci(preds, residuals, 10.0, 20.0)
        _, normal, _ = compute_volatility_adjusted_ci(preds, residuals, 20.0, 20.0)
        _, extreme, _ = compute_volatility_adjusted_ci(preds, residuals, 1.0, 20.0)

        assert low_vix[0] - preds[0] < normal[0] - preds[0]
        # 10/20 = 0.5 is already the floor, so 1/20 cannot narrow it further.
        np.testing.assert_allclose(extreme[0], low_vix[0])

    def test_multiplier_is_capped_above(self):
        preds = np.array([100.0])
        residuals = symmetric_residuals()

        _, doubled, _ = compute_volatility_adjusted_ci(preds, residuals, 40.0, 20.0)
        _, extreme, _ = compute_volatility_adjusted_ci(preds, residuals, 400.0, 20.0)
        np.testing.assert_allclose(extreme[0], doubled[0])

    def test_asymmetric_residuals_give_asymmetric_bounds(self):
        """The point of using quantiles rather than a Gaussian multiplier."""
        preds = np.array([100.0])
        skewed = np.concatenate([np.linspace(-5, 0, 900), np.linspace(0, 50, 100)])

        lower, upper, _ = compute_volatility_adjusted_ci(preds, skewed, 20.0, 20.0)
        assert (upper[0] - preds[0]) > (preds[0] - lower[0])


class TestCalibrateConfidenceInterval:
    def test_full_coverage_on_a_small_sample(self):
        result = calibrate_confidence_interval(
            np.array([100.0, 200.0, 300.0]),
            np.array([90.0, 190.0, 290.0]),
            np.array([110.0, 210.0, 310.0]),
            nominal_level=0.97,
        )
        assert result["empirical_coverage"] == 1.0
        assert result["n_inside"] == 3

    def test_full_coverage_is_flagged_over_covered(self):
        y_true = np.arange(100, dtype=float)
        result = calibrate_confidence_interval(
            y_true, y_true - 10, y_true + 10, nominal_level=0.50
        )
        assert result["empirical_coverage"] == 1.0
        assert result["flag"] == "[!] OVER-COVERED"
        assert not result["is_well_calibrated"]

    def test_zero_coverage_is_flagged_under_covered(self):
        result = calibrate_confidence_interval(
            np.array([100.0, 200.0, 300.0]),
            np.array([200.0, 300.0, 400.0]),
            np.array([210.0, 310.0, 410.0]),
        )
        assert result["empirical_coverage"] == 0.0
        assert result["flag"] == "[!] UNDER-COVERED"

    def test_coverage_inside_tolerance_is_well_calibrated(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(100, 10, 100)
        lower, upper = y_true - 50, y_true + 50
        # Push exactly five observations outside.
        lower[:5] = y_true[:5] + 1
        upper[:5] = y_true[:5] + 2

        result = calibrate_confidence_interval(y_true, lower, upper, nominal_level=0.95)
        assert result["empirical_coverage"] == 0.95
        assert result["is_well_calibrated"]
        assert result["flag"] == "[OK] WELL-CALIBRATED"

    def test_values_exactly_on_the_bound_count_as_inside(self):
        result = calibrate_confidence_interval(
            np.array([100.0]), np.array([100.0]), np.array([100.0])
        )
        assert result["empirical_coverage"] == 1.0

    def test_residual_count_is_reported_when_given(self):
        y_true = np.zeros(10)
        result = calibrate_confidence_interval(
            y_true, y_true - 1, y_true + 1, residuals=np.zeros(37)
        )
        assert result["n_residuals"] == 37

    def test_residual_count_is_na_when_omitted(self):
        y_true = np.zeros(10)
        result = calibrate_confidence_interval(y_true, y_true - 1, y_true + 1)
        assert result["n_residuals"] == "N/A"


def test_report_prints_the_status_flag(capsys):
    from calibration import print_calibration_report

    y_true = np.arange(100, dtype=float)
    report = calibrate_confidence_interval(
        y_true, y_true - 10, y_true + 10, nominal_level=0.50
    )
    print_calibration_report(report)
    assert "OVER-COVERED" in capsys.readouterr().out
