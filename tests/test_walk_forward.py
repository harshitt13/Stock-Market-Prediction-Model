"""
Tests for the walk-forward validation engine.
"""

import numpy as np
import pytest
from walk_forward import WalkForwardSplitter, aggregate_fold_results


class TestWalkForwardSplitter:
    def test_basic_split(self):
        """Test that splitter generates correct non-overlapping folds."""
        splitter = WalkForwardSplitter(min_train_size=100, test_size=20, step_size=20)
        folds = splitter.split(n_samples=200)

        assert len(folds) > 0, "Should generate at least one fold"

        for train_idx, test_idx in folds:
            # Training always comes before test
            assert train_idx[-1] < test_idx[0], (
                "Training must end before test starts"
            )
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0, (
                "Train and test must not overlap"
            )

    def test_expanding_window(self):
        """Test that training window expands with each fold."""
        splitter = WalkForwardSplitter(min_train_size=50, test_size=10, step_size=10)
        folds = splitter.split(n_samples=100)

        assert len(folds) >= 2, "Should have multiple folds"
        # Each fold's training set should be larger than the previous
        for i in range(1, len(folds)):
            assert len(folds[i][0]) > len(folds[i - 1][0]), (
                "Expanding window: later folds should have more training data"
            )

    def test_no_future_leakage(self):
        """Test that no test index appears in any training set."""
        splitter = WalkForwardSplitter(min_train_size=30, test_size=10, step_size=10)
        folds = splitter.split(n_samples=80)

        for train_idx, test_idx in folds:
            assert np.max(train_idx) < np.min(test_idx), (
                "All training indices must be less than all test indices"
            )

    def test_test_folds_non_overlapping(self):
        """Test that test folds don't overlap (when step_size == test_size)."""
        splitter = WalkForwardSplitter(min_train_size=50, test_size=10, step_size=10)
        folds = splitter.split(n_samples=100)

        all_test_indices = []
        for _, test_idx in folds:
            all_test_indices.extend(test_idx.tolist())

        assert len(all_test_indices) == len(set(all_test_indices)), (
            "Test indices should not repeat across folds"
        )

    def test_insufficient_data(self):
        """Test behavior when data is too small for any folds."""
        splitter = WalkForwardSplitter(min_train_size=100, test_size=20, step_size=20)
        folds = splitter.split(n_samples=50)
        # Not enough data for min_train_size
        assert len(folds) == 0

    def test_single_fold_fallback(self):
        """Test that a single fold is created when just barely enough data."""
        splitter = WalkForwardSplitter(min_train_size=50, test_size=10, step_size=10)
        folds = splitter.split(n_samples=60)
        assert len(folds) == 1
        assert len(folds[0][0]) == 50
        assert len(folds[0][1]) == 10

    def test_final_holdout(self):
        """Test final holdout set separation."""
        splitter = WalkForwardSplitter(min_train_size=50, test_size=10, step_size=10)
        main_idx, holdout_idx = splitter.get_final_holdout(100, holdout_size=15)

        assert len(holdout_idx) == 15
        assert main_idx[-1] + 1 == holdout_idx[0]
        assert holdout_idx[-1] == 99

    def test_fold_indices_are_contiguous(self):
        """Test that indices within each fold are contiguous."""
        splitter = WalkForwardSplitter(min_train_size=40, test_size=10, step_size=10)
        folds = splitter.split(n_samples=100)

        for train_idx, test_idx in folds:
            assert np.all(np.diff(train_idx) == 1), "Train indices must be contiguous"
            assert np.all(np.diff(test_idx) == 1), "Test indices must be contiguous"


class TestAggregateFoldResults:
    def test_aggregation(self):
        """Test that fold results are correctly concatenated."""
        fold_results = [
            {
                "y_true": np.array([1.0, 2.0]),
                "y_pred": np.array([1.1, 2.1]),
                "test_dates": np.array(["2024-01-01", "2024-01-02"]),
                "metrics": {"RMSE": 0.1},
            },
            {
                "y_true": np.array([3.0, 4.0]),
                "y_pred": np.array([3.1, 4.1]),
                "test_dates": np.array(["2024-01-03", "2024-01-04"]),
                "metrics": {"RMSE": 0.1},
            },
        ]
        agg = aggregate_fold_results(fold_results)

        assert len(agg["y_true"]) == 4
        assert len(agg["y_pred"]) == 4
        assert len(agg["test_dates"]) == 4
        assert len(agg["per_fold_metrics"]) == 2
        np.testing.assert_array_almost_equal(
            agg["y_true"], [1.0, 2.0, 3.0, 4.0]
        )
