"""Tests for the experiment runner (REFACTOR_PLAN.md section 9).

Offline: the loader is stubbed with the fixture, so no test hits the network.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments
from experiments import (
    DEFAULT_SEEDS,
    DEFAULT_TICKERS,
    PREDICTION_SCHEMA,
    REGIMES,
    SURVIVORSHIP_WARNING,
    across_tickers,
    aggregate,
    load_runs,
    run_path,
    run_single,
    save_run,
    seed_variance,
    sweep,
    write_aggregates,
)
from fetch_data import engineer_features

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aapl_raw.csv"


@pytest.fixture(scope="module")
def engineered() -> pd.DataFrame:
    raw = pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])
    return engineer_features(raw).dropna().reset_index(drop=True)


@pytest.fixture
def loader(engineered):
    """Ignores the date window and always serves the fixture."""

    def load(ticker, start, end):
        return engineered.copy()

    return load


@pytest.fixture(scope="module")
def runs(engineered, tmp_path_factory):
    """A small real sweep: 2 tickers x 1 regime x 2 seeds, tree only."""
    outdir = tmp_path_factory.mktemp("results") / "predictions"

    def load(ticker, start, end):
        return engineered.copy()

    sweep(
        tickers=["AAA", "BBB"],
        regimes=["full"],
        seeds=[0, 1],
        results_dir=str(outdir),
        load_raw=load,
        min_train_size=400,
        test_size=100,
        step_size=100,
        include_models=["tree", "baselines"],
    )
    return outdir


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_at_least_thirty_tickers_across_sectors(self):
        assert len(DEFAULT_TICKERS) >= 30
        assert len(set(DEFAULT_TICKERS)) == len(DEFAULT_TICKERS)

    def test_survivorship_bias_is_stated_explicitly(self):
        """Section 9: state it if you cannot include delisted names."""
        assert "SURVIVORSHIP BIAS" in SURVIVORSHIP_WARNING
        assert "delisted" in SURVIVORSHIP_WARNING

    def test_regimes_cover_the_windows_the_plan_names(self):
        assert {"pre_2020", "crash_2020", "drawdown_2021_2022", "post_2023"} <= set(REGIMES)

    def test_at_least_five_seeds_by_default(self):
        """Single-seed neural results are not credible."""
        assert len(DEFAULT_SEEDS) >= 5

    def test_sweep_prints_the_survivorship_warning(self, loader, tmp_path, capsys):
        sweep(
            tickers=["AAA"],
            regimes=["full"],
            seeds=[0],
            results_dir=str(tmp_path / "p"),
            load_raw=loader,
            min_train_size=400,
            test_size=200,
            step_size=200,
            include_models=["baselines"],
        )
        assert "SURVIVORSHIP BIAS" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Running and persistence
# ---------------------------------------------------------------------------


class TestRunSingle:
    def test_produces_the_long_schema(self, loader):
        frame = run_single(
            "AAA", "full", 0, load_raw=loader,
            min_train_size=400, test_size=200, step_size=200,
            include_models=["tree"],
        )
        assert list(frame.columns) == PREDICTION_SCHEMA
        assert (frame["ticker"] == "AAA").all()
        assert (frame["regime"] == "full").all()
        assert (frame["seed"] == 0).all()

    def test_returns_none_when_the_window_is_too_short(self, loader):
        assert run_single(
            "AAA", "full", 0, load_raw=loader, min_train_size=100_000
        ) is None

    def test_returns_none_when_the_loader_finds_nothing(self):
        assert run_single("ZZZ", "full", 0, load_raw=lambda *a: None) is None

    def test_includes_only_the_requested_models(self, loader):
        frame = run_single(
            "AAA", "full", 0, load_raw=loader,
            min_train_size=400, test_size=200, step_size=200,
            include_models=["baselines"],
        )
        assert "Tree Ensemble" not in set(frame["model"])
        assert "Zero return" in set(frame["model"])


class TestPersistence:
    def test_round_trips_through_disk(self, tmp_path, loader):
        frame = run_single(
            "AAA", "full", 0, load_raw=loader,
            min_train_size=400, test_size=200, step_size=200,
            include_models=["baselines"],
        )
        path = save_run(frame, run_path("AAA", "full", 0, str(tmp_path)))
        reloaded = load_runs(str(tmp_path))

        assert len(reloaded) == len(frame)
        assert pd.api.types.is_datetime64_any_dtype(reloaded["target_date"])
        np.testing.assert_allclose(
            reloaded.sort_values(["model", "target_date"])["y_pred"],
            frame.sort_values(["model", "target_date"])["y_pred"],
        )

    def test_load_runs_fails_loudly_on_an_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no persisted runs"):
            load_runs(str(tmp_path))

    def test_a_second_sweep_reuses_cached_runs(self, runs, loader, capsys):
        sweep(
            tickers=["AAA"], regimes=["full"], seeds=[0],
            results_dir=str(runs), load_raw=loader,
            min_train_size=400, test_size=100, step_size=100,
            include_models=["tree", "baselines"],
        )
        assert "cached" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_aggregates_are_computed_from_disk(self, runs):
        """Section 9: never from memory."""
        per_run = aggregate(str(runs))
        assert not per_run.empty
        assert set(per_run["ticker"]) == {"AAA", "BBB"}
        assert set(per_run["seed"]) == {0, 1}

    def test_one_row_per_regime_model_ticker_seed(self, runs):
        per_run = aggregate(str(runs))
        assert not per_run.duplicated(["regime", "model", "ticker", "seed"]).any()

    def test_carries_the_primary_metrics(self, runs):
        per_run = aggregate(str(runs))
        for column in ("directional_accuracy", "r2_oos", "rmse_bps", "pt_p_value"):
            assert column in per_run.columns

    def test_seed_variance_is_reported(self, runs):
        """The spread across seeds is often larger than the gap between
        models; a model comparison that ignores it is not a finding."""
        variance = seed_variance(aggregate(str(runs)))
        assert (variance["n_seeds"] == 2).all()
        assert "da_std" in variance.columns
        assert "r2_std" in variance.columns

    def test_cross_ticker_summary(self, runs):
        summary = across_tickers(aggregate(str(runs)))
        assert (summary["n_tickers"] == 2).all()
        assert "da_frac_above_half" in summary.columns
        assert summary["da_frac_above_half"].between(0, 1).all()
        assert "r2_frac_positive" in summary.columns

    def test_zero_return_baseline_lands_near_zero_r2(self, runs):
        per_run = aggregate(str(runs))
        zero = per_run[per_run["model"] == "Zero return"]
        assert not zero.empty
        assert zero["r2_oos"].abs().max() < 0.2

    def test_write_aggregates_emits_three_files(self, runs, tmp_path):
        write_aggregates(str(runs), str(tmp_path))
        for name in ("per_run.csv", "seed_variance.csv", "across_tickers.csv"):
            assert (tmp_path / name).exists()

    def test_aggregates_can_be_recomputed_without_retraining(self, runs, tmp_path):
        """The whole point of persisting: rerun the aggregation cheaply."""
        first = write_aggregates(str(runs), str(tmp_path))["per_run.csv"]
        second = write_aggregates(str(runs), str(tmp_path))["per_run.csv"]
        pd.testing.assert_frame_equal(first, second)
