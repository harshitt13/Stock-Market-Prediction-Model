"""Tests for the figure module.

Figures are read-only over pipeline outputs, so the tests build every input
with the same helpers the pipeline uses -- on synthetic predictions over the
offline fixture -- and assert that each figure is written. No model trains.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import backtest_table
from conftest import load_fixture
from contracts import align_predictions, common_evaluation_window, restrict_all
from dataset import build_dataset
from evaluate import compare_evaluations, evaluate_predictions
from figures import (
    fig_alpha_beta,
    fig_calibration,
    fig_equity_curves,
    fig_feature_importance,
    fig_model_comparison,
    fig_per_fold_metric,
    fig_predicted_vs_realised,
    fig_prediction_dispersion,
    fig_walk_forward_folds,
    render_run_figures,
)
from meta_ensemble import fold_respecting_intervals
from model_utils import fold_predictions

N_FOLDS, FOLD = 4, 40


@pytest.fixture(scope="module")
def ds():
    return build_dataset(load_fixture())


@pytest.fixture(scope="module")
def folds(ds):
    start = len(ds) - N_FOLDS * FOLD
    return [(np.arange(0, start + i * FOLD), np.arange(start + i * FOLD, start + (i + 1) * FOLD))
            for i in range(N_FOLDS)]


def synthetic_model(ds, folds, signal, noise, seed):
    """Predictions with a known signal fraction, keyed like a real model."""
    rng = np.random.default_rng(seed)
    frames = []
    for fold_id, (_, test) in enumerate(folds):
        y_pred = signal * ds.y[test] + rng.normal(0, noise, len(test))
        frames.append(fold_predictions(ds, fold_id, test, y_pred))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def result(ds, folds):
    """A run_pipeline-shaped result dict, built entirely from pipeline helpers."""
    all_predictions = {
        "Tree Ensemble": synthetic_model(ds, folds, 0.3, 0.01, 1),
        "BiLSTM": synthetic_model(ds, folds, 0.0, 0.002, 2),
        "Hybrid meta (+VIX)": synthetic_model(ds, folds, 0.1, 0.008, 3),
        "Zero return": synthetic_model(ds, folds, 0.0, 0.0, 4),
        "Historical mean": synthetic_model(ds, folds, 0.0, 0.0005, 5),
    }
    # Meta covers only later folds, as in the real pipeline.
    meta = all_predictions["Hybrid meta (+VIX)"]
    all_predictions["Hybrid meta (+VIX)"] = meta[meta["fold_id"] >= 1].reset_index(drop=True)

    window = common_evaluation_window(all_predictions)
    primary = restrict_all(all_predictions, window)
    y_train = {i: ds.y[tr] for i, (tr, _) in enumerate(folds)}
    evaluations = [evaluate_predictions(p, n, y_train_by_fold=y_train) for n, p in primary.items()]

    return {
        "window": window,
        "primary_predictions": primary,
        "base_results": {
            "tree": {"model_name": "Tree Ensemble", "feature_names": ds.feature_names,
                     "mean_feature_importances": np.random.default_rng(0).random(ds.n_features)},
            "lstm": {"model_name": "BiLSTM"},
        },
        "comparison_df": compare_evaluations(evaluations),
        "aligned": align_predictions(primary),
        "economics": backtest_table(primary, benchmark_predictions=primary["Zero return"]),
        "evaluations": evaluations,
        "calibration": fold_respecting_intervals(primary["Hybrid meta (+VIX)"]),
        "dataset": ds,
        "folds": folds,
        "cost_bps": 7.5,
    }


def _is_png(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1000 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestEachFigure:
    def test_model_comparison(self, result, tmp_path):
        assert _is_png(fig_model_comparison(result["comparison_df"], "t", tmp_path / "a.png"))

    def test_predicted_vs_realised(self, result, tmp_path):
        out = fig_predicted_vs_realised(result["aligned"], ["Tree Ensemble", "BiLSTM"], "t",
                                        tmp_path / "b.png")
        assert _is_png(out)

    def test_predicted_vs_realised_skips_unknown_models(self, result, tmp_path):
        out = fig_predicted_vs_realised(result["aligned"], ["Tree Ensemble", "not a model"], "t",
                                        tmp_path / "b2.png")
        assert _is_png(out)

    def test_prediction_dispersion(self, result, tmp_path):
        out = fig_prediction_dispersion(result["aligned"], list(result["primary_predictions"]),
                                        "t", tmp_path / "c.png")
        assert _is_png(out)

    def test_equity_curves(self, result, tmp_path):
        assert _is_png(fig_equity_curves(result["primary_predictions"], "t", tmp_path / "d.png"))

    def test_alpha_beta_tolerates_na_cells(self, result, tmp_path):
        """The economics table carries 'n/a' strings; the figure must not choke."""
        assert (result["economics"] == "n/a").any().any()
        assert _is_png(fig_alpha_beta(result["economics"], "t", tmp_path / "e.png"))

    def test_per_fold_metric(self, result, tmp_path):
        assert _is_png(fig_per_fold_metric(result["evaluations"], "r2_oos", "t", tmp_path / "f.png"))

    def test_calibration(self, result, tmp_path):
        assert _is_png(fig_calibration(result["calibration"], "t", tmp_path / "g.png"))

    def test_calibration_refuses_empty(self, tmp_path):
        with pytest.raises(ValueError, match="no per-fold"):
            fig_calibration({"per_fold": pd.DataFrame()}, "t", tmp_path / "g2.png")

    def test_walk_forward_folds(self, result, tmp_path):
        ds = result["dataset"]
        out = fig_walk_forward_folds(ds.feature_date, ds.close_t, result["folds"], "t",
                                     tmp_path / "h.png")
        assert _is_png(out)

    def test_feature_importance(self, result, tmp_path):
        tree = result["base_results"]["tree"]
        out = fig_feature_importance(tree["feature_names"], tree["mean_feature_importances"],
                                     "t", tmp_path / "i.png")
        assert _is_png(out)


class TestRenderRunFigures:
    EXPECTED = {
        "model_comparison.png", "predicted_vs_realised.png", "prediction_dispersion.png",
        "equity_curves.png", "alpha_beta.png", "r2_oos_by_fold.png",
        "directional_accuracy_by_fold.png", "calibration.png", "walk_forward_folds.png",
        "feature_importance.png",
    }

    def test_writes_the_full_set(self, result, tmp_path):
        written = render_run_figures(result, tmp_path, ticker="TEST")
        assert {p.name for p in written} == self.EXPECTED
        assert all(_is_png(p) for p in written)

    def test_one_broken_input_does_not_stop_the_others(self, result, tmp_path, capsys):
        broken = dict(result)
        broken["calibration"] = {"per_fold": pd.DataFrame()}
        written = render_run_figures(broken, tmp_path, ticker="TEST")
        assert {p.name for p in written} == self.EXPECTED - {"calibration.png"}
        assert "calibration.png skipped" in capsys.readouterr().out

    def test_figures_do_not_mutate_their_inputs(self, result, tmp_path):
        before = {n: f.copy() for n, f in result["primary_predictions"].items()}
        aligned_before = result["aligned"].copy()
        render_run_figures(result, tmp_path)
        for n, f in result["primary_predictions"].items():
            pd.testing.assert_frame_equal(f, before[n])
        pd.testing.assert_frame_equal(result["aligned"], aligned_before)
