"""Economic evaluation of a directional forecast.

REFACTOR_PLAN.md section 8. Directional accuracy does not pay for lunch: a
model can be right 53% of the time and still lose money, because the days it
gets wrong are larger than the days it gets right, or because it trades so
often that costs eat the edge.

The strategy is deliberately the simplest thing that follows from the forecast
-- take the predicted sign -- so that what is being measured is the forecast
and not a trading overlay.

Everything here is computed on the standard result frame, so the backtest runs
on exactly the days the model was scored on.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from contracts import validate_predictions

#: Round-trip cost in basis points for liquid US equities. A full buy-and-sell
#: cycle costs this much; a single side costs half.
DEFAULT_COST_BPS = 7.5

TRADING_DAYS = 252

BPS = 1e4

LONG_FLAT = "long_flat"
LONG_SHORT = "long_short"


def positions_from_predictions(y_pred, mode: str = LONG_FLAT) -> np.ndarray:
    """Target position for each forecast day, from the predicted sign.

    The position for a day is set by the forecast made at the previous close,
    so it is known before that day's return is realised.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if mode == LONG_FLAT:
        return (y_pred > 0).astype(float)
    if mode == LONG_SHORT:
        return np.where(y_pred > 0, 1.0, -1.0)
    raise ValueError(f"mode must be {LONG_FLAT!r} or {LONG_SHORT!r}, got {mode!r}")


def _max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough fall of the equity curve, as a fraction."""
    running_peak = np.maximum.accumulate(equity)
    return float(np.min(equity / running_peak - 1.0))


def _sharpe(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio, zero risk-free rate.

    A zero risk-free rate makes this a raw reward-to-variability figure. It is
    the standard simplification for a daily equity strategy comparison and it
    is applied identically to the strategy and to buy-and-hold, so the
    comparison is fair even if the level is not a true Sharpe.
    """
    if len(returns) < 2:
        return float("nan")
    sigma = float(np.std(returns, ddof=1))
    if sigma == 0:
        return float("nan")
    return float(np.mean(returns) / sigma * np.sqrt(periods_per_year))


def _summarise(
    strategy_returns: np.ndarray,
    turnover: np.ndarray,
    periods_per_year: int,
) -> Dict[str, float]:
    equity = np.cumprod(1.0 + strategy_returns)
    n = len(strategy_returns)
    years = n / periods_per_year if periods_per_year else float("nan")

    return {
        "total_return": float(equity[-1] - 1.0) if n else float("nan"),
        "annualised_return": (
            float(equity[-1] ** (1.0 / years) - 1.0) if n and years > 0 and equity[-1] > 0
            else float("nan")
        ),
        "annualised_volatility": (
            float(np.std(strategy_returns, ddof=1) * np.sqrt(periods_per_year))
            if n > 1 else float("nan")
        ),
        "sharpe": _sharpe(strategy_returns, periods_per_year),
        "max_drawdown": _max_drawdown(equity) if n else float("nan"),
        "annualised_turnover": float(np.mean(turnover) * periods_per_year) if n else 0.0,
        "n_days": n,
    }


def breakeven_cost_bps(
    gross_returns: np.ndarray, turnover: np.ndarray
) -> float:
    """Round-trip cost, in bps, at which the strategy stops being profitable.

    Net return per day is ``gross - turnover * cost_one_way``. Setting the mean
    to zero and solving for the round-trip cost (twice the one-way cost) gives
    ``2e4 * mean(gross) / mean(turnover)``.

    Reported because "profitable at 5bps" and "profitable at 500bps" are very
    different claims, and only one of them survives contact with a real desk.
    """
    mean_turnover = float(np.mean(turnover))
    if mean_turnover <= 0:
        return float("inf") if float(np.mean(gross_returns)) > 0 else float("nan")
    return float(2e4 * np.mean(gross_returns) / mean_turnover)


def market_adjusted(
    strategy_returns, market_returns, periods_per_year: int = TRADING_DAYS
) -> Dict[str, float]:
    """OLS of a strategy's returns on the market's: ``r_s = a + b*r_m + e``.

    A long/flat strategy driven by a near-constant forecast is long almost
    every day, so it inherits the market's Sharpe ratio wholesale. Raw Sharpe
    cannot tell that apart from skill; alpha and beta can. Beta near 1 with
    alpha near 0 means the strategy *is* the market.

    The t-statistic uses plain OLS standard errors. Daily strategy residuals
    carry little serial correlation, but this is not a HAC estimator and the
    t-stat should be read as indicative rather than exact.
    """
    y = np.asarray(strategy_returns, dtype=float)
    x = np.asarray(market_returns, dtype=float)
    n = len(y)

    empty = {
        "alpha_daily": float("nan"),
        "alpha_annualised": float("nan"),
        "beta": float("nan"),
        "t_alpha": float("nan"),
        "p_alpha": float("nan"),
        "r_squared": float("nan"),
    }
    if n < 3 or np.std(x) == 0:
        return empty

    design = np.column_stack([np.ones(n), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha, beta = float(coefficients[0]), float(coefficients[1])

    residuals = y - design @ coefficients
    dof = n - 2
    sigma2 = float(residuals @ residuals) / dof
    if sigma2 <= 0:
        # A strategy that replicates the market exactly, or never trades.
        return {**empty, "alpha_daily": alpha, "beta": beta,
                "alpha_annualised": alpha * periods_per_year}

    standard_errors = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * sigma2)
    t_alpha = alpha / standard_errors[0]
    total_variance = float(np.var(y))

    return {
        "alpha_daily": alpha,
        "alpha_annualised": alpha * periods_per_year,
        "beta": beta,
        "t_alpha": float(t_alpha),
        "p_alpha": float(2 * stats.t.sf(abs(t_alpha), dof)),
        "r_squared": (
            1.0 - float(np.var(residuals)) / total_variance
            if total_variance > 0 else float("nan")
        ),
    }


def prediction_diagnostics(predictions: pd.DataFrame) -> Dict[str, float]:
    """Is the model actually varying its forecast, or emitting a constant?

    A model whose predicted returns have far less dispersion than the realised
    ones is not forecasting; it is picking a level. Combined with a long/flat
    rule that turns any positive number into a full position, that produces a
    near-constant long book whose Sharpe is the market's.
    """
    y_pred = predictions["y_pred"].to_numpy(float)
    y_true = predictions["y_true"].to_numpy(float)
    signs = y_pred > 0

    std_pred = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0
    std_true = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float("nan")

    if std_pred > 0 and std_true > 0:
        correlation = float(np.corrcoef(y_pred, y_true)[0, 1])
        dof = len(y_pred) - 2
        if dof > 0 and abs(correlation) < 1.0:
            t = correlation * np.sqrt(dof / (1.0 - correlation**2))
            p_value = float(2 * stats.t.sf(abs(t), dof))
        else:
            p_value = float("nan")
    else:
        correlation, p_value = float("nan"), float("nan")

    return {
        "n": len(y_pred),
        "mean_pred_bps": float(np.mean(y_pred)) * BPS,
        "std_pred_bps": std_pred * BPS,
        "std_true_bps": std_true * BPS,
        "std_ratio": std_pred / std_true if std_true else float("nan"),
        "frac_positive": float(signs.mean()),
        "sign_flip_rate": (
            float(np.mean(signs[1:] != signs[:-1])) if len(signs) > 1 else 0.0
        ),
        "corr_pred_true": correlation,
        "corr_p_value": p_value,
    }


def backtest(
    predictions: pd.DataFrame,
    mode: str = LONG_FLAT,
    cost_bps: float = DEFAULT_COST_BPS,
    periods_per_year: int = TRADING_DAYS,
    validate: bool = True,
) -> Dict[str, Any]:
    """Run the sign-following strategy over a standard result frame.

    Returns gross and net figures, buy-and-hold for comparison, and the
    break-even cost. Log returns are converted to simple returns first, because
    a position times a log return is not a portfolio return.
    """
    if validate:
        validate_predictions(predictions, name=f"backtest ({mode})")

    predictions = predictions.sort_values("target_date").reset_index(drop=True)

    # A log return compounds; a portfolio return does not. Convert first.
    asset_returns = np.expm1(predictions["y_true"].to_numpy(float))
    positions = positions_from_predictions(predictions["y_pred"], mode)

    # Turnover on day i is the position change entering that day; the first
    # trade is entered from flat.
    previous = np.concatenate([[0.0], positions[:-1]])
    turnover = np.abs(positions - previous)

    one_way_cost = cost_bps / 2.0 / 1e4
    gross_returns = positions * asset_returns
    net_returns = gross_returns - turnover * one_way_cost

    gross = _summarise(gross_returns, turnover, periods_per_year)
    net = _summarise(net_returns, turnover, periods_per_year)
    hold = _summarise(asset_returns, np.zeros(len(asset_returns)), periods_per_year)

    return {
        "mode": mode,
        "cost_bps": cost_bps,
        "gross": gross,
        "net": net,
        "buy_and_hold": hold,
        "breakeven_cost_bps": breakeven_cost_bps(gross_returns, turnover),
        "beats_buy_and_hold_net": bool(net["sharpe"] > hold["sharpe"]),
        # Fraction of days holding the asset. A long/flat rule turns any
        # positive forecast into a full position, so this is usually the
        # single most informative number about what the strategy is doing.
        "average_exposure": float(np.mean(positions)),
        "market_adjusted": market_adjusted(net_returns, asset_returns, periods_per_year),
        "predictions_diagnostics": prediction_diagnostics(predictions),
        "daily": pd.DataFrame(
            {
                "target_date": predictions["target_date"],
                "position": positions,
                "turnover": turnover,
                "asset_return": asset_returns,
                "gross_return": gross_returns,
                "net_return": net_returns,
            }
        ),
    }


#: Shown instead of NaN for a strategy that never takes a position. Its return
#: series is identically zero, so the Sharpe ratio is 0/0 -- undefined, not
#: bad. The zero-return baseline is the case that matters: predicting no move
#: every day gives nothing to trade on.
NOT_APPLICABLE = "n/a"

#: Below this annualised turnover the break-even cost divides by ~zero and the
#: result is a meaningless six-figure number. A strategy that barely trades has
#: no meaningful cost threshold.
MIN_TURNOVER_FOR_BREAKEVEN = 5.0


def _round(value, digits):
    return NOT_APPLICABLE if not np.isfinite(value) else round(float(value), digits)


def _sharpe_cell(value: float):
    return NOT_APPLICABLE if not np.isfinite(value) else round(value, 3)


def backtest_table(
    results_by_model: Dict[str, pd.DataFrame],
    mode: str = LONG_FLAT,
    cost_bps: float = DEFAULT_COST_BPS,
    benchmark_predictions: Optional[pd.DataFrame] = None,
    require_identical_days: bool = True,
) -> pd.DataFrame:
    """Backtest every model and tabulate, with buy-and-hold on the same days.

    ``require_identical_days`` asserts every model covers exactly the same
    forecast days. Without it the comparison silently crosses evaluation
    windows: buy-and-hold used to be measured on whichever model happened to be
    first in the dict, which on a walk-forward run meant the base models' full
    range while the meta-learner covered only the later folds.

    ``benchmark_predictions`` names the frame buy-and-hold is computed from.
    When every model shares the same days it makes no difference which is used
    -- they all carry the same realised returns -- but passing it explicitly
    documents the intent.
    """
    frames = {
        name: (item["predictions"] if isinstance(item, dict) else item)
        for name, item in results_by_model.items()
    }
    if not frames:
        return pd.DataFrame()

    date_sets = {name: set(df["target_date"]) for name, df in frames.items()}
    reference_name, reference_dates = next(iter(date_sets.items()))
    if require_identical_days:
        for name, dates in date_sets.items():
            if dates != reference_dates:
                raise ValueError(
                    f"{name!r} covers {len(dates)} days but {reference_name!r} "
                    f"covers {len(reference_dates)}. Backtesting them in one "
                    "table would compare different evaluation windows. Restrict "
                    "to a common window first (contracts.restrict_all)."
                )

    rows = []
    for name, predictions in frames.items():
        result = backtest(predictions, mode=mode, cost_bps=cost_bps)
        turnover = result["net"]["annualised_turnover"]
        adjusted = result["market_adjusted"]
        rows.append(
            {
                "Model": name,
                # PRIMARY: market-adjusted. Raw Sharpe cannot separate skill
                # from being long the market every day.
                "Alpha ann.": _round(adjusted["alpha_annualised"], 4),
                "t(alpha)": _round(adjusted["t_alpha"], 2),
                "Beta": _round(adjusted["beta"], 3),
                "Exposure": _round(result["average_exposure"], 3),
                # SECONDARY: raw risk/return.
                "Sharpe gross": _sharpe_cell(result["gross"]["sharpe"]),
                "Sharpe net": _sharpe_cell(result["net"]["sharpe"]),
                "Ann. return net": round(result["net"]["annualised_return"], 4),
                "Total return net": round(result["net"]["total_return"], 4),
                "Max drawdown": round(result["net"]["max_drawdown"], 4),
                "Ann. turnover": round(turnover, 1),
                "Breakeven cost (bps)": (
                    NOT_APPLICABLE
                    if turnover < MIN_TURNOVER_FOR_BREAKEVEN
                    or not np.isfinite(result["breakeven_cost_bps"])
                    else round(result["breakeven_cost_bps"], 1)
                ),
                "Beats B&H net": result["beats_buy_and_hold_net"],
            }
        )

    table = pd.DataFrame(rows).set_index("Model")

    benchmark = (
        benchmark_predictions if benchmark_predictions is not None
        else frames[reference_name]
    )
    hold = backtest(benchmark, mode=mode, cost_bps=cost_bps)["buy_and_hold"]
    table.loc["Buy and hold"] = {
        "Alpha ann.": 0.0,   # the benchmark has no alpha against itself
        "t(alpha)": NOT_APPLICABLE,
        "Beta": 1.0,
        "Exposure": 1.0,
        "Sharpe gross": _sharpe_cell(hold["sharpe"]),
        "Sharpe net": _sharpe_cell(hold["sharpe"]),
        "Ann. return net": round(hold["annualised_return"], 4),
        "Total return net": round(hold["total_return"], 4),
        "Max drawdown": round(hold["max_drawdown"], 4),
        "Ann. turnover": 0.0,
        "Breakeven cost (bps)": NOT_APPLICABLE,
        "Beats B&H net": True,
    }
    return table


def print_backtest_report(result: Dict[str, Any], model_name: str = "Model") -> None:
    """Print one strategy's economics, buy-and-hold alongside."""
    gross, net, hold = result["gross"], result["net"], result["buy_and_hold"]

    print(f"\n  {'-' * 18} BACKTEST: {model_name} {'-' * 18}")
    print(f"  Strategy: {result['mode']}, costs {result['cost_bps']:.1f} bps round-trip")
    print(f"  {'':<22}{'gross':>10}{'net':>10}{'buy & hold':>13}")
    for label, key in [
        ("Sharpe", "sharpe"),
        ("Annualised return", "annualised_return"),
        ("Annualised vol", "annualised_volatility"),
        ("Max drawdown", "max_drawdown"),
    ]:
        print(
            f"  {label:<22}{gross[key]:>10.3f}{net[key]:>10.3f}{hold[key]:>13.3f}"
        )
    print(f"  {'Annualised turnover':<22}{net['annualised_turnover']:>10.1f}")
    print(f"  {'Break-even cost (bps)':<22}{result['breakeven_cost_bps']:>10.1f}")

    if not result["beats_buy_and_hold_net"]:
        print(
            "  After costs this strategy loses to buy-and-hold. Stated plainly: "
            "that is a publishable finding, not a failure to hide."
        )
