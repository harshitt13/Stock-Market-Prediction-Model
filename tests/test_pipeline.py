"""End-to-end pipeline test, offline.

Unit tests cover each module; this catches the wiring between them. It runs
the real run_pipeline with yfinance stubbed by the fixture, tiny epochs and
plots off.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fetch_data
import main
from contracts import validate_predictions
from model_utils import DEMO_FORECAST_CAVEAT

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aapl_raw.csv"


@pytest.fixture(scope="module")
def raw_fixture() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def engineered(raw_fixture) -> pd.DataFrame:
    return fetch_data.engineer_features(raw_fixture).dropna().reset_index(drop=True)


@pytest.fixture(scope="module")
def result(engineered, tmp_path_factory):
    """Run the whole pipeline once; the assertions below share it."""
    workdir = tmp_path_factory.mktemp("pipeline")
    original_cwd = Path.cwd()

    import os

    os.chdir(workdir)
    original_fetch = main.fetch_stock_data
    main.fetch_stock_data = lambda *args, **kwargs: engineered.copy()
    try:
        yield main.run_pipeline(
            ticker="AAPL",
            min_train_size=400,
            test_size=60,
            step_size=60,
            epochs=2,
            demo_forecast_days=3,
            make_plots=False,
        ), workdir
    finally:
        main.fetch_stock_data = original_fetch
        os.chdir(original_cwd)


def test_pipeline_completes(result):
    output, _ = result
    assert "error" not in output
    assert len(output["folds"]) > 3


def test_every_model_and_baseline_is_evaluated(result):
    output, _ = result
    names = {evaluation["model"] for evaluation in output["evaluations"]}
    assert {"Tree Ensemble", "BiLSTM", "Transformer"} <= names
    assert {"Zero return", "Historical mean", "Random sign"} <= names
    assert any("Hybrid meta" in n for n in names)


def test_all_predictions_satisfy_the_contract(result):
    output, _ = result
    for group in ("base_results", "baseline_results"):
        for item in output[group].values():
            validate_predictions(item["predictions"], name=item["model_name"])


def test_base_models_forecast_identical_days(result):
    output, _ = result
    date_sets = [
        set(item["predictions"]["target_date"]) for item in output["base_results"].values()
    ]
    assert all(s == date_sets[0] for s in date_sets)


def test_comparison_table_is_written_and_return_space(result):
    output, workdir = result
    assert (workdir / "data" / "model_comparison.csv").exists()

    comparison = output["comparison_df"]
    assert "R2_OOS" in comparison.columns
    assert "DA (%)" in comparison.columns
    assert not any("Price R2" in c for c in comparison.columns)
    assert all("[2nd]" in c for c in comparison.columns if c.startswith("Price"))


def test_reported_numbers_are_in_the_honest_range(result):
    """Section 10: expect the numbers to collapse, and that is correct.

    A directional accuracy far outside the low fifties, or an R2_OOS far from
    zero, on daily equity returns means something leaked.
    """
    output, _ = result
    comparison = output["comparison_df"]
    for model in ("Tree Ensemble", "BiLSTM", "Transformer"):
        assert 35.0 < comparison.loc[model, "DA (%)"] < 65.0
        assert -0.5 < comparison.loc[model, "R2_OOS"] < 0.1


def test_zero_return_baseline_is_present_and_flat(result):
    output, _ = result
    zero = output["baseline_results"]["zero_return"]["predictions"]
    assert (zero["y_pred"] == 0.0).all()


def test_diebold_mariano_table_is_produced(result):
    output, workdir = result
    dm = output["diebold_mariano"]
    assert not dm.empty
    assert "p-value" in dm.columns
    assert (workdir / "data" / "diebold_mariano.csv").exists()


def test_meta_predictions_skip_the_earliest_folds(result):
    output, _ = result
    gating = output["gating"]
    assert gating is not None
    assert gating["with_vix"]["skipped_folds"] == [0, 1]

    meta_folds = set(gating["with_vix"]["predictions"]["fold_id"])
    assert 0 not in meta_folds and 1 not in meta_folds


def test_vix_gating_is_reported_with_a_dm_test(result):
    output, _ = result
    gating = output["gating"]
    assert "dm_statistic" in gating["diebold_mariano"]
    assert isinstance(output["vix_terciles"], pd.DataFrame)
    assert list(output["vix_terciles"].index) == ["low", "mid", "high"]


def test_interval_calibration_is_reported(result):
    output, _ = result
    calibration = output["calibration"]
    assert calibration["overall"] is not None
    assert calibration["overall"]["nominal_level"] == 0.95
    assert 0.0 <= calibration["overall"]["empirical_coverage"] <= 1.0


def test_demo_forecast_is_written_and_caveated(result):
    output, workdir = result
    path = workdir / "data" / "combined_predictions.csv"
    assert path.exists()

    demo = pd.read_csv(path)
    assert len(demo) == 3
    assert (demo["caveat"] == DEMO_FORECAST_CAVEAT).all()
    assert "interval_lower_log_return" in demo.columns


def test_demo_forecast_days_are_not_in_any_metric(result):
    """It must never reach a results table."""
    output, workdir = result
    demo_dates = set(pd.read_csv(workdir / "data" / "combined_predictions.csv")["date"])
    for item in output["base_results"].values():
        scored = set(item["predictions"]["target_date"].astype(str))
        assert not (demo_dates & scored)


def test_aligned_predictions_are_written(result):
    output, workdir = result
    assert (workdir / "data" / "aligned_predictions.csv").exists()
    assert "y_true" in output["aligned"].columns


def test_cli_exposes_the_demo_forecast_flag():
    parser = main.build_parser()
    args = parser.parse_args(["--demo-forecast", "10"])
    assert args.demo_forecast == 10
    assert parser.parse_args([]).demo_forecast == 0


def test_backtest_table_is_produced(result):
    """Section 8: directional accuracy does not pay for lunch."""
    output, workdir = result
    economics = output["economics"]
    assert (workdir / "data" / "backtest.csv").exists()
    assert "Buy and hold" in economics.index
    for column in ("Sharpe gross", "Sharpe net", "Max drawdown", "Breakeven cost (bps)"):
        assert column in economics.columns


def test_costs_are_actually_charged(result):
    """Net return is always at or below gross; costs are non-negative.

    Note the Sharpe ratio is NOT monotone in cost: subtracting a varying cost
    series changes the numerator and the denominator, so net Sharpe can exceed
    gross Sharpe. Only the return comparison is guaranteed.
    """
    output, _ = result
    economics = output["economics"].drop(index="Buy and hold")
    for name, row in economics.iterrows():
        if row["Ann. turnover"] > 0:
            assert row["Sharpe net"] != row["Sharpe gross"], name


def test_a_strategy_is_only_a_finding_if_it_beats_buy_and_hold(result):
    """Section 8: report the comparison either way."""
    output, _ = result
    economics = output["economics"]
    assert "Beats B&H net" in economics.columns
    assert economics.loc["Buy and hold", "Ann. turnover"] == 0.0


def test_cli_exposes_the_cost_flag():
    parser = main.build_parser()
    assert parser.parse_args(["--cost-bps", "10"]).cost_bps == 10.0
