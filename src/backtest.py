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

from contracts import validate_predictions

#: Round-trip cost in basis points for liquid US equities. A full buy-and-sell
#: cycle costs this much; a single side costs half.
DEFAULT_COST_BPS = 7.5

TRADING_DAYS = 252

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


def backtest_table(
    results_by_model: Dict[str, pd.DataFrame],
    mode: str = LONG_FLAT,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Backtest every model on the same days and tabulate."""
    rows = []
    for name, item in results_by_model.items():
        predictions = item["predictions"] if isinstance(item, dict) else item
        result = backtest(predictions, mode=mode, cost_bps=cost_bps)
        rows.append(
            {
                "Model": name,
                "Sharpe gross": round(result["gross"]["sharpe"], 3),
                "Sharpe net": round(result["net"]["sharpe"], 3),
                "Ann. return net": round(result["net"]["annualised_return"], 4),
                "Max drawdown": round(result["net"]["max_drawdown"], 4),
                "Ann. turnover": round(result["net"]["annualised_turnover"], 1),
                "Breakeven cost (bps)": round(result["breakeven_cost_bps"], 1),
                "Beats B&H net": result["beats_buy_and_hold_net"],
            }
        )

    table = pd.DataFrame(rows).set_index("Model")
    if rows:
        first = next(iter(results_by_model.values()))
        predictions = first["predictions"] if isinstance(first, dict) else first
        hold = backtest(predictions, mode=mode, cost_bps=cost_bps)["buy_and_hold"]
        table.loc["Buy and hold"] = {
            "Sharpe gross": round(hold["sharpe"], 3),
            "Sharpe net": round(hold["sharpe"], 3),
            "Ann. return net": round(hold["annualised_return"], 4),
            "Max drawdown": round(hold["max_drawdown"], 4),
            "Ann. turnover": 0.0,
            "Breakeven cost (bps)": float("nan"),
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
