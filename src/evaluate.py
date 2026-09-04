"""Evaluation metrics, computed in return space.

REFACTOR_PLAN.md section 4. The primary metrics all operate on log returns.
Price-space numbers are reported too, but as clearly-labelled secondary
diagnostics, because the naive baseline achieves essentially the same values
on all of them and they therefore measure the price series rather than any
forecasting skill.

Two things the previous version got wrong and this one fixes:

1. Directional accuracy had two incompatible definitions. The old
   ``derive_direction_labels`` took ``np.diff`` of the *prediction series*,
   which measures whether prediction t+1 exceeds prediction t, and produced
   garbage at every fold boundary because folds were concatenated before
   differencing. In return space the correct definition is trivial:
   ``sign(y_pred)`` against ``sign(y_true)``, and the fold-boundary problem
   disappears entirely.

2. Metrics were computed on all folds pooled into one array. Here every metric
   is computed per fold and reported as mean +/- std across folds, which is
   what gives the error bars a reviewer expects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from contracts import reconstruct_close, validate_predictions

#: Near-flat days are excluded from directional metrics. In return space this
#: is 0.1% -- a day whose realised move is smaller than this carries no
#: directional information worth scoring.
DIRECTION_THRESHOLD = 0.001

BPS = 1e4  # one basis point is 1e-4, so a return times BPS is in bps


# ---------------------------------------------------------------------------
# Directional metrics
# ---------------------------------------------------------------------------


def direction_labels(y: np.ndarray) -> np.ndarray:
    """1 for a positive return, 0 otherwise. Exactly zero counts as down."""
    return (np.asarray(y, dtype=float) > 0).astype(int)


def directional_accuracy(
    y_true, y_pred, threshold: float = DIRECTION_THRESHOLD
) -> Dict[str, Any]:
    """Directional accuracy of ``sign(y_pred)`` against ``sign(y_true)``.

    Days with ``abs(y_true) <= threshold`` are excluded as near-flat. The
    fraction excluded is reported alongside, because an accuracy figure means
    nothing without knowing how much of the sample it was measured on.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    keep = np.abs(y_true) > threshold
    n_total = len(y_true)
    n_kept = int(keep.sum())

    result = {
        "n_total": n_total,
        "n_evaluated": n_kept,
        "n_excluded": n_total - n_kept,
        "excluded_fraction": (n_total - n_kept) / n_total if n_total else float("nan"),
        "threshold": threshold,
    }

    if n_kept == 0:
        result.update(
            {
                "directional_accuracy": float("nan"),
                "majority_class_rate": float("nan"),
                "da_minus_majority": float("nan"),
                "f1_macro": float("nan"),
                "precision_macro": float("nan"),
                "recall_macro": float("nan"),
                "confusion_matrix": None,
                "base_rate_up": float("nan"),
                "predicted_rate_up": float("nan"),
            }
        )
        return result

    actual = direction_labels(y_true[keep])
    predicted = direction_labels(y_pred[keep])

    result.update(
        {
            "directional_accuracy": float((actual == predicted).mean()),
            "f1_macro": float(f1_score(actual, predicted, average="macro", zero_division=0)),
            "precision_macro": float(
                precision_score(actual, predicted, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(actual, predicted, average="macro", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]),
            "base_rate_up": float(actual.mean()),
            "predicted_rate_up": float(predicted.mean()),
            # Always predicting the more common direction scores this much.
            # Directional accuracy is only informative as a delta against it:
            # 55% on a sample that is 55% up is worth nothing.
            "majority_class_rate": float(max(actual.mean(), 1.0 - actual.mean())),
            "da_minus_majority": float(
                (actual == predicted).mean() - max(actual.mean(), 1.0 - actual.mean())
            ),
        }
    )
    return result


def pesaran_timmermann(y_true, y_pred) -> Dict[str, float]:
    """Pesaran-Timmermann test of directional predictive ability.

    Tests the null that the predicted and realised directions are
    independent, so that the observed hit rate is what you would get by
    chance given the two marginal rates. A high hit rate on a series that is
    up 55% of the time is not evidence of skill; this is the statistic that
    says so.

    Returns the standard-normal statistic and a one-sided p-value (the
    alternative of interest is that accuracy *beats* independence).

    Pesaran & Timmermann (1992), JBES 10(4), 461-465.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    if n < 2:
        return {"pt_statistic": float("nan"), "pt_p_value": float("nan"), "n": n}

    x = direction_labels(y_true)  # realised up/down
    z = direction_labels(y_pred)  # predicted up/down

    p_hat = float((x == z).mean())  # observed hit rate
    p_x = float(x.mean())
    p_z = float(z.mean())

    # Hit rate expected if the two directions were independent
    p_star = p_x * p_z + (1 - p_x) * (1 - p_z)

    var_p_hat = p_star * (1 - p_star) / n
    var_p_star = (
        (2 * p_z - 1) ** 2 * p_x * (1 - p_x) / n
        + (2 * p_x - 1) ** 2 * p_z * (1 - p_z) / n
        + 4 * p_x * p_z * (1 - p_x) * (1 - p_z) / n**2
    )

    denom = var_p_hat - var_p_star
    if denom <= 0:
        # Degenerate: one of the direction series is constant.
        return {
            "pt_statistic": float("nan"),
            "pt_p_value": float("nan"),
            "hit_rate": p_hat,
            "hit_rate_under_independence": p_star,
            "n": n,
        }

    statistic = (p_hat - p_star) / np.sqrt(denom)
    return {
        "pt_statistic": float(statistic),
        "pt_p_value": float(stats.norm.sf(statistic)),  # one-sided
        "hit_rate": p_hat,
        "hit_rate_under_independence": p_star,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Out-of-sample R-squared (Campbell-Thompson)
# ---------------------------------------------------------------------------


def expanding_mean_benchmark(y_test, y_train=None) -> np.ndarray:
    """The historical-mean forecast, expanding and strictly backward-looking.

    ``benchmark[i]`` is the mean of every return observed before test point
    ``i``: the training returns, plus the test returns already realised. It is
    emphatically **not** the test-set mean, which would use the future and make
    the benchmark unbeatable for the wrong reason.

    With no training history the first forecast is 0.0, which is the natural
    prior for a daily equity return.
    """
    y_test = np.asarray(y_test, dtype=float)
    history = np.asarray([] if y_train is None else y_train, dtype=float)

    # Cumulative sum/count of everything strictly before each test point.
    prior_sum = history.sum() + np.concatenate([[0.0], np.cumsum(y_test)[:-1]])
    prior_n = len(history) + np.arange(len(y_test))

    with np.errstate(invalid="ignore", divide="ignore"):
        benchmark = np.where(prior_n > 0, prior_sum / np.maximum(prior_n, 1), 0.0)
    return benchmark


def r2_oos(y_true, y_pred, benchmark) -> float:
    """Campbell-Thompson out-of-sample R-squared.

    ``1 - SSE_model / SSE_benchmark``. Legitimately negative when the model
    forecasts worse than the historical mean, which for daily equity returns
    is the common case and not a bug.
    """
    y_true = np.asarray(y_true, dtype=float)
    sse_model = float(np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2))
    sse_bench = float(np.sum((y_true - np.asarray(benchmark, dtype=float)) ** 2))
    if sse_bench == 0:
        return float("nan")
    return 1.0 - sse_model / sse_bench


# ---------------------------------------------------------------------------
# Error magnitude
# ---------------------------------------------------------------------------


def return_error_metrics(y_true, y_pred) -> Dict[str, float]:
    """RMSE and MAE of returns, in basis points."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    return {
        "rmse_bps": float(np.sqrt(np.mean(errors**2)) * BPS),
        "mae_bps": float(np.mean(np.abs(errors)) * BPS),
    }


def price_space_metrics(close_t, y_true, y_pred) -> Dict[str, float]:
    """SECONDARY. Price MAPE and RMSE on reconstructed closes.

    Reported for continuity with the previous results table only. The naive
    baseline achieves essentially the same values, so these numbers describe
    the price series, not forecasting skill. There is deliberately no price
    R-squared here: on a trending series it is ~0.99 for any model and is the
    single most misleading number this project used to report.
    """
    actual = reconstruct_close(close_t, y_true)
    predicted = reconstruct_close(close_t, y_pred)
    errors = actual - predicted
    return {
        "price_rmse": float(np.sqrt(np.mean(errors**2))),
        "price_mape_pct": float(np.mean(np.abs(errors / actual)) * 100),
    }


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


def diebold_mariano(errors_a, errors_b, h: int = 1, loss: str = "squared") -> Dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy, Newey-West SE.

    ``errors_a`` and ``errors_b`` are forecast errors (``y_true - y_pred``) from
    two models on the *same* target days. A negative statistic favours model A.

    "Model A had lower RMSE than model B" without this test is not a finding:
    the difference has to be large relative to its own serial-correlation-
    robust standard error before it means anything.

    Newey-West truncation lag is ``h - 1``, the usual choice for an h-step
    forecast, with Bartlett weights.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if errors_a.shape != errors_b.shape:
        raise ValueError(
            f"error series must be the same length and aligned on target_date; "
            f"got {errors_a.shape} and {errors_b.shape}"
        )

    if loss == "squared":
        d = errors_a**2 - errors_b**2
    elif loss == "absolute":
        d = np.abs(errors_a) - np.abs(errors_b)
    else:
        raise ValueError(f"loss must be 'squared' or 'absolute', got {loss!r}")

    n = len(d)
    if n < 3:
        return {"dm_statistic": float("nan"), "dm_p_value": float("nan"), "n": n}

    d_bar = float(d.mean())
    demeaned = d - d_bar

    # Newey-West long-run variance with Bartlett weights, lag h-1.
    gamma_0 = float(np.mean(demeaned**2))
    long_run_var = gamma_0
    for lag in range(1, h):
        gamma_k = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
        long_run_var += 2 * (1 - lag / h) * gamma_k

    if long_run_var <= 0:
        return {
            "dm_statistic": float("nan"),
            "dm_p_value": float("nan"),
            "mean_loss_differential": d_bar,
            "n": n,
        }

    statistic = d_bar / np.sqrt(long_run_var / n)
    return {
        "dm_statistic": float(statistic),
        "dm_p_value": float(2 * stats.norm.sf(abs(statistic))),  # two-sided
        "mean_loss_differential": d_bar,
        "favours": "A" if statistic < 0 else "B",
        "n": n,
    }


# ---------------------------------------------------------------------------
# Whole-model evaluation over the standard result frame
# ---------------------------------------------------------------------------

#: Metrics summarised as mean +/- std across folds.
FOLD_METRIC_KEYS = [
    "directional_accuracy",
    "da_minus_majority",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "excluded_fraction",
    "r2_oos",
    "rmse_bps",
    "mae_bps",
    "price_rmse",
    "price_mape_pct",
]


def _metrics_for_slice(
    close_t, y_true, y_pred, threshold: float, y_train=None, benchmark=None
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out.update(directional_accuracy(y_true, y_pred, threshold))
    out.update(return_error_metrics(y_true, y_pred))
    out.update(price_space_metrics(close_t, y_true, y_pred))
    if benchmark is None:
        benchmark = expanding_mean_benchmark(y_true, y_train)
    out["r2_oos"] = r2_oos(y_true, y_pred, benchmark)
    return out


def per_fold_metrics(
    predictions: pd.DataFrame,
    threshold: float = DIRECTION_THRESHOLD,
    y_train_by_fold: Optional[Dict[int, np.ndarray]] = None,
) -> pd.DataFrame:
    """One row of metrics per fold.

    Folds are evaluated separately rather than pooled, so that the summary can
    carry a standard deviation across folds instead of a single pooled number
    that hides which fold did the work.
    """
    rows = []
    for fold_id, block in predictions.groupby("fold_id", sort=True):
        y_train = None if y_train_by_fold is None else y_train_by_fold.get(int(fold_id))
        row = {"fold_id": int(fold_id), "n": len(block)}
        metrics = _metrics_for_slice(
            block["close_t"].to_numpy(float),
            block["y_true"].to_numpy(float),
            block["y_pred"].to_numpy(float),
            threshold,
            y_train,
        )
        row.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_folds(fold_metrics: pd.DataFrame) -> Dict[str, float]:
    """``mean`` and ``std`` for each fold-level metric, plus the fold count."""
    summary: Dict[str, float] = {"n_folds": int(len(fold_metrics))}
    for key in FOLD_METRIC_KEYS:
        if key not in fold_metrics.columns:
            continue
        values = fold_metrics[key].to_numpy(dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values)) if len(values) else float("nan")
        summary[f"{key}_std"] = (
            float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        )
    return summary


def evaluate_predictions(
    predictions: pd.DataFrame,
    model_name: str = "Model",
    threshold: float = DIRECTION_THRESHOLD,
    y_train_by_fold: Optional[Dict[int, np.ndarray]] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """Evaluate a standard result frame.

    Returns both the pooled metrics (over all target days, for the headline
    table) and the per-fold metrics summarised as mean +/- std (for the error
    bars). The pooled directional numbers are safe to compute here precisely
    because the definition is ``sign(y_pred)`` -- there is no differencing, so
    concatenating folds introduces no boundary artefacts.

    ``y_train_by_fold`` maps ``fold_id`` to that fold's training returns and
    should be supplied whenever it is available. Without it the R2_OOS
    benchmark starts cold, and its first few values are one- and
    two-observation means. On a test period that opens with a volatile stretch
    those are terrible forecasts, the benchmark's SSE inflates, and *every*
    model scores an R2_OOS that is too high. On the AAPL fixture, seeding moves
    the zero-return baseline from +0.087 to -0.005. The result carries
    ``unseeded_benchmark`` so a caller cannot miss it.
    """
    if validate:
        validate_predictions(predictions, name=model_name)

    predictions = predictions.reset_index(drop=True)
    fold_metrics = per_fold_metrics(predictions, threshold, y_train_by_fold)

    close_t = predictions["close_t"].to_numpy(float)
    y_true = predictions["y_true"].to_numpy(float)
    y_pred = predictions["y_pred"].to_numpy(float)

    # The pooled benchmark is built per fold, each seeded with its own training
    # returns, then reassembled in target_date order. Building it over the
    # pooled series instead would restart the expanding mean from nothing.
    benchmark = np.empty(len(predictions), dtype=float)
    for fold_id, block in predictions.groupby("fold_id", sort=True):
        y_train = None if y_train_by_fold is None else y_train_by_fold.get(int(fold_id))
        benchmark[block.index.to_numpy()] = expanding_mean_benchmark(
            block["y_true"].to_numpy(float), y_train
        )

    pooled = _metrics_for_slice(
        close_t, y_true, y_pred, threshold, benchmark=benchmark
    )
    pooled.update(pesaran_timmermann(y_true, y_pred))

    return {
        "model": model_name,
        "pooled": pooled,
        "per_fold": fold_metrics,
        "summary": summarize_across_folds(fold_metrics),
        "n_predictions": len(predictions),
        "unseeded_benchmark": y_train_by_fold is None,
        "first_target_date": predictions["target_date"].iloc[0],
        "last_target_date": predictions["target_date"].iloc[-1],
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

PRICE_METRICS_CAVEAT = (
    "Price metrics are SECONDARY. The zero-return baseline achieves "
    "essentially the same values; they do not measure forecasting skill."
)


def print_evaluation(result: Dict[str, Any]) -> None:
    """Print one model's evaluation, primary metrics first."""
    pooled = result["pooled"]
    summary = result["summary"]

    print(f"\n{'=' * 68}")
    print(f"  {result['model']} - {result['n_predictions']} forecast days, "
          f"{summary['n_folds']} folds")
    print(f"{'=' * 68}")

    print(f"\n  {'-' * 18} PRIMARY (return space) {'-' * 18}")
    da = pooled["directional_accuracy"]
    print(f"  Directional accuracy   : {da * 100:6.2f}%  pooled")
    print(
        f"                           {summary['directional_accuracy_mean'] * 100:6.2f}% "
        f"+/- {summary['directional_accuracy_std'] * 100:.2f}  across folds"
    )
    print(
        f"    days excluded        : {pooled['excluded_fraction'] * 100:6.2f}% "
        f"(|y_true| <= {pooled['threshold']:.4f})"
    )
    print(
        f"  Majority-class rate    : {pooled['majority_class_rate'] * 100:6.2f}%  "
        f"(always predicting the more common direction)"
    )
    print(f"  DA - majority          : {pooled['da_minus_majority'] * 100:+6.2f} pp")
    pt_stat = pooled.get("pt_statistic", float("nan"))
    pt_p = pooled.get("pt_p_value", float("nan"))
    print(f"  Pesaran-Timmermann     : {pt_stat:6.3f}  (p = {pt_p:.4f}, one-sided)")
    print(f"  F1 macro               : {pooled['f1_macro']:6.4f}")
    print(f"  Precision / Recall     : {pooled['precision_macro']:.4f} / "
          f"{pooled['recall_macro']:.4f}")
    print(
        f"  R2_OOS (Campbell-Thom.): {pooled['r2_oos']:+7.5f}  pooled;  "
        f"{summary['r2_oos_mean']:+.5f} +/- {summary['r2_oos_std']:.5f} across folds"
    )
    if result.get("unseeded_benchmark"):
        print(
            "    WARNING: benchmark not seeded with training returns, so its "
            "first values are one- and two-observation means. R2_OOS is "
            "biased upward; pass y_train_by_fold."
        )
    print(f"  RMSE / MAE             : {pooled['rmse_bps']:6.1f} / "
          f"{pooled['mae_bps']:.1f} bps")

    cm = pooled.get("confusion_matrix")
    if cm is not None:
        print("\n  Confusion matrix (actual \\ predicted):")
        print("                   Pred DOWN   Pred UP")
        print(f"    Actual DOWN  | {cm[0, 0]:9d} | {cm[0, 1]:7d}")
        print(f"    Actual UP    | {cm[1, 0]:9d} | {cm[1, 1]:7d}")

    print(f"\n  {'-' * 18} SECONDARY (price space) {'-' * 17}")
    print(f"  Price RMSE             : {pooled['price_rmse']:.4f}")
    print(f"  Price MAPE             : {pooled['price_mape_pct']:.4f}%")
    print(f"  NOTE: {PRICE_METRICS_CAVEAT}")


def compare_evaluations(
    results: Sequence[Dict[str, Any]], save_path: Optional[str] = None
) -> pd.DataFrame:
    """Build the comparison table across models, with per-fold error bars."""
    rows = []
    for result in results:
        pooled, summary = result["pooled"], result["summary"]
        rows.append(
            {
                "Model": result["model"],
                "DA (%)": round(pooled["directional_accuracy"] * 100, 2),
                "Majority (%)": round(pooled["majority_class_rate"] * 100, 2),
                "DA - majority": round(pooled["da_minus_majority"] * 100, 2),
                "DA std (%)": round(summary["directional_accuracy_std"] * 100, 2),
                "PT stat": round(pooled.get("pt_statistic", float("nan")), 3),
                "PT p": round(pooled.get("pt_p_value", float("nan")), 4),
                "F1 macro": round(pooled["f1_macro"], 4),
                "R2_OOS": round(pooled["r2_oos"], 5),
                "R2_OOS std": round(summary["r2_oos_std"], 5),
                "RMSE (bps)": round(pooled["rmse_bps"], 1),
                "MAE (bps)": round(pooled["mae_bps"], 1),
                "Excluded (%)": round(pooled["excluded_fraction"] * 100, 1),
                "Price RMSE [2nd]": round(pooled["price_rmse"], 4),
                "Price MAPE (%) [2nd]": round(pooled["price_mape_pct"], 4),
            }
        )

    df = pd.DataFrame(rows).set_index("Model")

    print(f"\n{'=' * 100}")
    print("  MODEL COMPARISON - primary metrics in return space")
    print(f"{'=' * 100}")
    print(df.to_string())
    print(f"\n  [2nd] {PRICE_METRICS_CAVEAT}")
    print(f"{'=' * 100}\n")

    if save_path:
        df.to_csv(save_path)
        print(f"Comparison table saved to '{save_path}'")

    return df


def dm_table(
    results_by_model: Dict[str, pd.DataFrame],
    reference: str,
    h: int = 1,
) -> pd.DataFrame:
    """Diebold-Mariano of every model against ``reference``, on shared days.

    The two frames are merged on ``target_date`` first, so the test never
    compares errors from different days.
    """
    if reference not in results_by_model:
        raise KeyError(f"reference model {reference!r} not among {list(results_by_model)}")

    ref = results_by_model[reference]
    rows = []
    for name, preds in results_by_model.items():
        if name == reference:
            continue
        merged = preds.merge(
            ref, on="target_date", suffixes=("_model", "_ref"), how="inner"
        )
        if merged.empty:
            continue
        test = diebold_mariano(
            merged["y_true_model"] - merged["y_pred_model"],
            merged["y_true_ref"] - merged["y_pred_ref"],
            h=h,
        )
        rows.append(
            {
                "Model": name,
                "vs": reference,
                "DM stat": round(test["dm_statistic"], 3),
                "p-value": round(test["dm_p_value"], 4),
                "Favours": {"A": name, "B": reference}.get(test.get("favours", "")),
                "n shared days": test["n"],
            }
        )
    return pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()
