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
    residual_std: float,
    vix_current: float,
    vix_historical_mean: float,
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a volatility-adjusted confidence interval around predictions.

    The interval width is scaled by the ratio of current VIX to its
    historical mean — wider during high-volatility regimes.

    Parameters
    ----------
    predictions : array
        Point predictions.
    residual_std : float
        Standard deviation of residuals from meta-learner training.
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
    dynamic_std : float
        The volatility-adjusted standard deviation used.
    """
    from scipy import stats
    z = stats.norm.ppf((1 + confidence_level) / 2)

    # Let the multiplier shrink below 1.0 in low volatility, but bounded
    volatility_multiplier = np.clip(vix_current / vix_historical_mean, 0.5, 2.0)
    
    # We apply a slight reduction factor (0.85) because empirical coverage was ~99.5% for a 95% target
    dynamic_std = residual_std * volatility_multiplier * 0.85

    lower = predictions - z * dynamic_std
    upper = predictions + z * dynamic_std

    return lower, upper, dynamic_std


def calibrate_confidence_interval(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_level: float = 0.95,
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

    Returns
    -------
    dict with:
        - empirical_coverage : float (0-1)
        - nominal_level : float
        - n_inside : int
        - n_total : int
        - is_well_calibrated : bool  (coverage within ±5pp of nominal)
        - flag : str  (OK / UNDER-COVERED / OVER-COVERED)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    inside = (y_true >= lower) & (y_true <= upper)
    n_inside = int(inside.sum())
    n_total = len(y_true)
    coverage = n_inside / n_total if n_total > 0 else 0.0

    # Tolerance: ±5 percentage points
    tolerance = 0.05
    is_well_calibrated = abs(coverage - nominal_level) <= tolerance

    if coverage < nominal_level - tolerance:
        flag = "[!] UNDER-COVERED"
    elif coverage > nominal_level + tolerance:
        flag = "[!] OVER-COVERED"
    else:
        flag = "[OK] WELL-CALIBRATED"

    return {
        "empirical_coverage": round(coverage, 4),
        "nominal_level": nominal_level,
        "n_inside": n_inside,
        "n_total": n_total,
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
    print(f"  Samples Inside      : {cal['n_inside']} / {cal['n_total']}")
    print(f"  Status              : {cal['flag']}")
    if not cal["is_well_calibrated"]:
        diff = (cal["empirical_coverage"] - cal["nominal_level"]) * 100
        direction = "higher" if diff > 0 else "lower"
        print(f"  ⚠️  Coverage is {abs(diff):.1f}pp {direction} than nominal.")
        print(f"      The intervals may be too {'wide' if diff > 0 else 'narrow'}.")
    print(f"{'='*60}\n")
