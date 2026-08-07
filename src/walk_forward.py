"""
Walk-Forward (Expanding Window) Cross-Validation for Time-Series.

This module provides a time-respecting validation framework that prevents
data leakage by always training on past data and testing on future data.

Scheme (expanding window):
    Fold 1: Train [0, min_train_size)              Test [min_train_size, min_train_size + test_size)
    Fold 2: Train [0, min_train_size + step_size)  Test [min_train_size + step_size, ...]
    ...
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional


class WalkForwardSplitter:
    """
    Expanding-window walk-forward splitter for time-series data.

    Parameters
    ----------
    min_train_size : int
        Minimum number of samples in the initial training window.
        Default 504 ≈ 2 years of trading days.
    test_size : int
        Number of samples in each test fold.
        Default 63 ≈ 3 months of trading days.
    step_size : int
        How far to slide the window forward between folds.
        Default 63 (same as test_size for non-overlapping test folds).
    """

    def __init__(
        self,
        min_train_size: int = 504,
        test_size: int = 63,
        step_size: int = 63,
    ):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index arrays for each fold.

        Parameters
        ----------
        n_samples : int
            Total number of samples in the dataset.

        Returns
        -------
        folds : list of (train_indices, test_indices) tuples
            Each element is a pair of 1-D numpy arrays.
        """
        folds = []
        train_end = self.min_train_size

        while train_end + self.test_size <= n_samples:
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, min(train_end + self.test_size, n_samples))
            folds.append((train_idx, test_idx))
            train_end += self.step_size

        # If no folds were created, make a single fold with whatever we have
        if not folds and n_samples > self.min_train_size:
            train_idx = np.arange(0, self.min_train_size)
            test_idx = np.arange(self.min_train_size, n_samples)
            folds.append((train_idx, test_idx))

        return folds

    def get_final_holdout(
        self, n_samples: int, holdout_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reserve a final holdout set that is never touched during walk-forward
        or Optuna tuning.  Used exclusively for meta-learner evaluation.

        Parameters
        ----------
        n_samples : int
            Total dataset size.
        holdout_size : int, optional
            Size of the holdout.  Defaults to ``self.test_size``.

        Returns
        -------
        main_idx, holdout_idx : tuple of arrays
        """
        if holdout_size is None:
            holdout_size = self.test_size
        holdout_start = n_samples - holdout_size
        main_idx = np.arange(0, holdout_start)
        holdout_idx = np.arange(holdout_start, n_samples)
        return main_idx, holdout_idx

    def __repr__(self) -> str:
        return (
            f"WalkForwardSplitter(min_train={self.min_train_size}, "
            f"test={self.test_size}, step={self.step_size})"
        )


def aggregate_fold_results(
    fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge per-fold predictions into a single set of aggregated results.

    Each element of *fold_results* must contain at least:
        - ``y_true``: array of actual values for the fold
        - ``y_pred``: array of predicted values
        - ``test_dates``: array of dates for the test fold
        - ``metrics``: dict of metric_name -> value  (optional)

    Returns
    -------
    aggregated : dict
        ``y_true``, ``y_pred``, ``test_dates`` concatenated across folds;
        ``per_fold_metrics`` list preserved for inspection.
    """
    all_y_true = []
    all_y_pred = []
    all_dates = []
    per_fold_metrics = []

    for fr in fold_results:
        all_y_true.append(np.asarray(fr["y_true"]))
        all_y_pred.append(np.asarray(fr["y_pred"]))
        all_dates.append(np.asarray(fr["test_dates"]))
        if "metrics" in fr:
            per_fold_metrics.append(fr["metrics"])

    return {
        "y_true": np.concatenate(all_y_true),
        "y_pred": np.concatenate(all_y_pred),
        "test_dates": np.concatenate(all_dates),
        "per_fold_metrics": per_fold_metrics,
    }


def print_fold_summary(folds: List[Tuple[np.ndarray, np.ndarray]], dates=None):
    """Pretty-print walk-forward fold boundaries."""
    print(f"\n{'='*60}")
    print(f"  Walk-Forward Validation - {len(folds)} Folds")
    print(f"{'='*60}")
    for i, (train_idx, test_idx) in enumerate(folds):
        if dates is not None:
            train_start = pd.Timestamp(dates[train_idx[0]]).strftime("%Y-%m-%d")
            train_end = pd.Timestamp(dates[train_idx[-1]]).strftime("%Y-%m-%d")
            test_start = pd.Timestamp(dates[test_idx[0]]).strftime("%Y-%m-%d")
            test_end = pd.Timestamp(dates[test_idx[-1]]).strftime("%Y-%m-%d")
            print(
                f"  Fold {i+1}: Train [{train_start} -> {train_end}] "
                f"({len(train_idx)} days)  |  "
                f"Test [{test_start} -> {test_end}] ({len(test_idx)} days)"
            )
        else:
            print(
                f"  Fold {i+1}: Train [0:{train_idx[-1]+1}] ({len(train_idx)})  |  "
                f"Test [{test_idx[0]}:{test_idx[-1]+1}] ({len(test_idx)})"
            )
    print(f"{'='*60}\n")
