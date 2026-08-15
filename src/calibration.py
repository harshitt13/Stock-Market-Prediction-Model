"""
Confidence interval calibration module.

Provides functions to:
1. Compute volatility-adjusted confidence intervals.
2. Validate that the empirical coverage matches the nominal level (e.g. 95 %).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


def compute_volatility_adjusted_ci(
    predictions: np.ndarray,
    residuals: np.ndarray,
    vix_current: float,
    vix_historical_mean: float,
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Compute a volatility-adjusted confidence interval around predictions using empirical quantiles.

    Parameters
    ----------
    predictions : array
        Point predictions.
    residuals : array
        Array of residuals from OOF meta-learner evaluation.
    vix_current : float
        Current VIX level.
    vix_historical_mean : float
        Historical average VIX.
    confidence_level : float
        Nominal confidence level (default 0.95).

    Returns
    -------
    lower, upper : arrays
        Lower and upper bounds of the interval.
    empirical_offsets : Tuple[float, float]
        The computed empirical (lower_offset, upper_offset).
    """
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100

    empirical_lower_offset = np.percentile(residuals, lower_percentile)
    empirical_upper_offset = np.percentile(residuals, upper_percentile)

    # Let the multiplier shrink below 1.0 in low volatility, but bounded
    volatility_multiplier = np.clip(vix_current / vix_historical_mean, 0.5, 2.0)
    
    lower = predictions + empirical_lower_offset * volatility_multiplier
    upper = predictions + empirical_upper_offset * volatility_multiplier

    return lower, upper, (empirical_lower_offset, empirical_upper_offset)


def calibrate_confidence_interval(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_level: float = 0.95,
    residuals: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Validate empirical coverage of the confidence interval.

    Parameters
    ----------
    y_true : array
        Actual values over the test period.
    lower, upper : arrays
        Predicted interval bounds.
    nominal_level : float
        The intended confidence level (e.g. 0.95).
    residuals : Optional array
        The array of OOF residuals used to build the intervals.

    Returns
    -------
    dict with coverage metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    inside = (y_true >= lower) & (y_true <= upper)
    n_inside = int(inside.sum())
    n_total = len(y_true)
    coverage = n_inside / n_total if n_total > 0 else 0.0

    # Tolerance: [0.90, 0.97] is considered OK based on 128 samples noise
    is_well_calibrated = (0.90 <= coverage <= 0.97)

    if coverage < 0.90:
        flag = "[!] UNDER-COVERED"
    elif coverage > 0.97:
        flag = "[!] OVER-COVERED"
    else:
        flag = "[OK] WELL-CALIBRATED"

    n_residuals = len(residuals) if residuals is not None else "N/A"

    return {
        "empirical_coverage": round(coverage, 4),
        "nominal_level": nominal_level,
        "n_inside": n_inside,
        "n_total": n_total,
        "n_residuals": n_residuals,
        "is_well_calibrated": is_well_calibrated,
        "flag": flag,
    }


def print_calibration_report(cal: Dict[str, Any]):
    """Pretty-print the calibration report."""
    print(f"\n{'='*60}")
    print(f"  CONFIDENCE INTERVAL CALIBRATION REPORT")
    print(f"{'='*60}")
    print(f"  Nominal Level       : {cal['nominal_level']*100:.1f}%")
    print(f"  Empirical Coverage  : {cal['empirical_coverage']*100:.2f}%")
    print(f"  OOF Residuals (n)   : {cal.get('n_residuals', 'N/A')}")
    print(f"  Samples Inside      : {cal['n_inside']} / {cal['n_total']}")
    print(f"  Status              : {cal['flag']}")
    if not cal["is_well_calibrated"]:
        diff = (cal["empirical_coverage"] - cal["nominal_level"]) * 100
        direction = "higher" if diff > 0 else "lower"
        print(f"  [!] Coverage is {abs(diff):.1f}pp {direction} than nominal.")
        print(f"      The intervals may be too {'wide' if diff > 0 else 'narrow'}.")
    print(f"{'='*60}\n")
