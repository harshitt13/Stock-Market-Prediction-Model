"""Tests for the economic evaluation (REFACTOR_PLAN.md section 8)."""

import numpy as np
import pandas as pd
import pytest

from backtest import (
    DEFAULT_COST_BPS,
    LONG_FLAT,
    LONG_SHORT,
    backtest,
    backtest_table,
    breakeven_cost_bps,
    positions_from_predictions,
)
from contracts import make_predictions


def frame(y_true, y_pred, close=100.0) -> pd.DataFrame:
    n = len(y_true)
    return make_predictions(
        target_date=pd.date_range("2020-01-01", periods=n, freq="B"),
        fold_id=np.zeros(n, dtype=int),
        close_t=np.full(n, close),
        y_true=np.asarray(y_true, dtype=float),
        y_pred=np.asarray(y_pred, dtype=float),
    )


@pytest.fixture
def noisy():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0.0003, 0.015, 750)
    y_pred = 0.1 * y_true + rng.normal(0, 0.01, 750)
    return frame(y_true, y_pred)


class TestPositions:
    def test_long_flat_is_zero_or_one(self):
        positions = positions_from_predictions([0.01, -0.01, 0.0], LONG_FLAT)
        np.testing.assert_array_equal(positions, [1.0, 0.0, 0.0])

    def test_long_short_is_plus_or_minus_one(self):
        positions = positions_from_predictions([0.01, -0.01, 0.0], LONG_SHORT)
        np.testing.assert_array_equal(positions, [1.0, -1.0, -1.0])

    def test_only_the_sign_matters(self):
        small = positions_from_predictions([1e-9], LONG_FLAT)
        large = positions_from_predictions([9.9], LONG_FLAT)
        np.testing.assert_array_equal(small, large)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="mode must be"):
            positions_from_predictions([0.01], "market_neutral")


class TestBacktestMechanics:
    def test_log_returns_are_converted_to_simple_returns(self):
        """A position times a log return is not a portfolio return."""
        y = np.log(1.10)  # a genuine +10% day
        result = backtest(frame([y], [0.01]), cost_bps=0.0)
        assert result["gross"]["total_return"] == pytest.approx(0.10)

    def test_perfect_foresight_makes_money(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(0, 0.02, 500)
        result = backtest(frame(y_true, y_true), mode=LONG_SHORT, cost_bps=0.0)
        assert result["gross"]["sharpe"] > 5
        assert result["gross"]["total_return"] > 0

    def test_inverted_forecast_loses_money(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(0, 0.02, 500)
        result = backtest(frame(y_true, -y_true), mode=LONG_SHORT, cost_bps=0.0)
        assert result["gross"]["sharpe"] < -5

    def test_always_long_reproduces_buy_and_hold(self, noisy):
        always_long = frame(noisy["y_true"], np.ones(len(noisy)))
        result = backtest(always_long, mode=LONG_FLAT, cost_bps=0.0)
        assert result["gross"]["sharpe"] == pytest.approx(
            result["buy_and_hold"]["sharpe"]
        )

    def test_costs_only_ever_reduce_returns(self, noisy):
        """The guaranteed invariant is on returns, not on Sharpe.

        Sharpe is a ratio: subtracting a varying cost series moves the
        numerator and the denominator, so net Sharpe can legitimately come out
        above gross. Only the return comparison holds by construction.
        """
        result = backtest(noisy, cost_bps=DEFAULT_COST_BPS)
        assert result["net"]["total_return"] <= result["gross"]["total_return"]
        assert (result["daily"]["net_return"] <= result["daily"]["gross_return"]).all()

    def test_higher_costs_hurt_more(self, noisy):
        cheap = backtest(noisy, cost_bps=1.0)["net"]["total_return"]
        dear = backtest(noisy, cost_bps=50.0)["net"]["total_return"]
        assert dear < cheap

    def test_zero_cost_leaves_net_equal_to_gross(self, noisy):
        result = backtest(noisy, cost_bps=0.0)
        assert result["net"]["total_return"] == pytest.approx(
            result["gross"]["total_return"]
        )

    def test_a_never_trading_strategy_has_no_turnover(self):
        result = backtest(frame([0.01, 0.02, -0.01], [-1.0, -1.0, -1.0]), LONG_FLAT)
        assert result["net"]["annualised_turnover"] == 0.0

    def test_turnover_counts_each_position_change(self):
        # flat, long, flat, long -> entering trades on days 2 and 4
        result = backtest(frame([0.01] * 4, [-1.0, 1.0, -1.0, 1.0]), LONG_FLAT)
        np.testing.assert_array_equal(result["daily"]["turnover"], [0, 1, 1, 1])

    def test_max_drawdown_is_negative_or_zero(self, noisy):
        result = backtest(noisy)
        assert result["net"]["max_drawdown"] <= 0.0

    def test_daily_frame_is_keyed_by_target_date(self, noisy):
        result = backtest(noisy)
        np.testing.assert_array_equal(
            result["daily"]["target_date"], noisy["target_date"]
        )


class TestBreakevenCost:
    def test_is_the_cost_that_zeroes_the_edge(self):
        """At the break-even cost the net mean return must be zero."""
        rng = np.random.default_rng(3)
        y_true = rng.normal(0, 0.02, 400)
        predictions = frame(y_true, 0.2 * y_true + rng.normal(0, 0.01, 400))

        gross = backtest(predictions, cost_bps=0.0)
        breakeven = gross["breakeven_cost_bps"]

        at_breakeven = backtest(predictions, cost_bps=breakeven)
        assert np.mean(at_breakeven["daily"]["net_return"]) == pytest.approx(0, abs=1e-12)

    def test_a_strategy_that_never_trades_has_no_finite_breakeven(self):
        result = backtest(frame([0.01, 0.01], [-1.0, -1.0]))
        assert not np.isfinite(result["breakeven_cost_bps"])

    def test_direct_computation_matches(self):
        gross = np.array([0.001, 0.001, 0.001])
        turnover = np.array([1.0, 1.0, 1.0])
        assert breakeven_cost_bps(gross, turnover) == pytest.approx(20.0)


class TestReporting:
    def test_table_includes_buy_and_hold_row(self, noisy):
        table = backtest_table({"model": noisy})
        assert "Buy and hold" in table.index
        assert "Breakeven cost (bps)" in table.columns

    def test_table_flags_whether_the_strategy_beats_holding(self, noisy):
        table = backtest_table({"model": noisy})
        assert table.loc["model", "Beats B&H net"] in (True, False)

    def test_losing_to_buy_and_hold_is_reported_not_hidden(self, capsys):
        """Section 8: say so plainly."""
        from backtest import print_backtest_report

        rng = np.random.default_rng(4)
        y_true = rng.normal(0.001, 0.02, 400)
        losing = frame(y_true, -y_true)
        print_backtest_report(backtest(losing), "Losing model")
        assert "loses to buy-and-hold" in capsys.readouterr().out

    def test_realistic_default_cost(self):
        assert 5.0 <= DEFAULT_COST_BPS <= 10.0


class TestOneEvaluationWindow:
    def test_mismatched_day_sets_are_rejected(self):
        """The bug this guards: buy-and-hold used to be measured on whichever
        model came first in the dict, which on a walk-forward run meant the
        base models' full range while the meta covered only later folds."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 0.02, 100)
        full = frame(y, rng.normal(0, 0.02, 100))
        short = full.iloc[30:].reset_index(drop=True)

        with pytest.raises(ValueError, match="different evaluation windows"):
            backtest_table({"full": full, "short": short})

    def test_buy_and_hold_uses_the_named_benchmark(self):
        from backtest import backtest

        rng = np.random.default_rng(1)
        y = rng.normal(0, 0.02, 200)
        a = frame(y, rng.normal(0, 0.02, 200))
        b = frame(y, rng.normal(0, 0.02, 200))

        table = backtest_table({"a": a, "b": b}, benchmark_predictions=b)
        expected = backtest(b)["buy_and_hold"]
        assert table.loc["Buy and hold", "Total return net"] == pytest.approx(
            round(expected["total_return"], 4)
        )

    def test_buy_and_hold_is_identical_whichever_model_names_it(self):
        """On one window every model carries the same realised returns."""
        rng = np.random.default_rng(2)
        y = rng.normal(0, 0.02, 200)
        a = frame(y, rng.normal(0, 0.02, 200))
        b = frame(y, rng.normal(0, 0.02, 200))

        from_a = backtest_table({"a": a, "b": b}, benchmark_predictions=a)
        from_b = backtest_table({"a": a, "b": b}, benchmark_predictions=b)
        pd.testing.assert_series_equal(
            from_a.loc["Buy and hold"], from_b.loc["Buy and hold"]
        )


class TestZeroReturnRow:
    """The benchmark every model must beat should still have a readable row."""

    @pytest.fixture
    def zero_table(self):
        rng = np.random.default_rng(3)
        y = rng.normal(0.0005, 0.02, 300)
        return backtest_table({"Zero return": frame(y, np.zeros(300))})

    def test_never_trading_strategy_reports_zeros_and_na(self, zero_table):
        from backtest import NOT_APPLICABLE

        row = zero_table.loc["Zero return"]
        assert row["Sharpe gross"] == NOT_APPLICABLE
        assert row["Sharpe net"] == NOT_APPLICABLE
        assert row["Breakeven cost (bps)"] == NOT_APPLICABLE
        assert row["Total return net"] == 0.0
        assert row["Max drawdown"] == 0.0
        assert row["Ann. turnover"] == 0.0

    def test_no_nan_appears_in_the_table(self, zero_table):
        assert not zero_table.isna().any().any()

    def test_a_trading_strategy_still_gets_numeric_sharpe(self):
        rng = np.random.default_rng(4)
        y = rng.normal(0, 0.02, 300)
        table = backtest_table({"m": frame(y, rng.normal(0, 0.02, 300))})
        assert isinstance(table.loc["m", "Sharpe net"], float)
