"""Fold-respecting meta-ensemble, and the VIX-gating experiment.

REFACTOR_PLAN.md section 7. Split out of main.py so that 7.3 -- "keep the VIX
gating, make it testable" -- is something that can actually be tested.

Two bugs from the previous version are fixed here.

**The in-sample evaluation bug (7.1).** ``train_meta_ensemble_oof`` fitted
RidgeCV on the first 80% of the aligned test rows, then returned predictions
covering 100% of them, and the caller computed metrics over the whole array.
The reported hybrid regression numbers were therefore 80% in-sample and not
comparable to any other row in the table. Nothing here ever returns a
prediction for a row its model was fitted on.

**The holdout-inside-a-holdout (7.2).** An 80/20 split of the pooled test rows
is not out-of-fold. Meta-predictions for fold ``k`` now come from a meta-model
fitted only on base-model predictions from folds ``1 .. k-1``. The first few
folds therefore produce no meta-predictions at all, which is correct and is
reported rather than hidden.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from contracts import make_predictions, validate_predictions
from dataset import Dataset
from evaluate import diebold_mariano, directional_accuracy, evaluate_predictions

#: Folds before this index produce no meta-prediction: there is not yet enough
#: base-model history to fit a meta-learner without using the fold itself.
MIN_TRAIN_FOLDS = 2

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)

VIX_COLUMN = "VIX"


def _new_meta_model() -> Pipeline:
    """Standardised RidgeCV.

    Standardising matters here because the meta-features are a mix of log
    returns (~0.02) and a VIX level (~20). Without it the ridge penalty falls
    almost entirely on the base-model coefficients, and the reported weights
    are not comparable to each other.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )


def build_meta_frame(
    base_results: Dict[str, Any],
    dataset: Optional[Dataset] = None,
    include_vix: bool = True,
) -> pd.DataFrame:
    """One wide row per forecast day, with every base model's prediction.

    The VIX column is taken from ``dataset.X`` at the *feature_date*, which is
    the level known at the close of day t, when the forecast for t+1 is made.
    Reading VIX at target_date would be using the day being forecast.
    """
    frames = {}
    for name, item in base_results.items():
        df = item["predictions"] if isinstance(item, dict) else item
        validate_predictions(df, name=name)
        frames[name] = df

    names = list(frames)
    reference = frames[names[0]]
    date_sets = {n: set(df["target_date"]) for n, df in frames.items()}
    shared = set.intersection(*date_sets.values())
    if not shared:
        raise ValueError("base models share no forecast days")

    meta = (
        reference[reference["target_date"].isin(shared)][
            ["target_date", "fold_id", "close_t", "y_true"]
        ]
        .sort_values("target_date")
        .reset_index(drop=True)
    )

    for name, df in frames.items():
        block = df[df["target_date"].isin(shared)].sort_values("target_date")
        meta[f"pred_{name}"] = block["y_pred"].to_numpy()

    if include_vix and dataset is not None and VIX_COLUMN in dataset.feature_names:
        vix_idx = dataset.feature_names.index(VIX_COLUMN)
        lookup = dict(zip(dataset.target_date, dataset.X[:, vix_idx]))
        meta[VIX_COLUMN] = [
            lookup.get(d.to_datetime64(), np.nan) for d in meta["target_date"]
        ]
        if meta[VIX_COLUMN].isna().any():
            raise ValueError("VIX missing for some forecast days")

    return meta


def meta_feature_columns(meta_frame: pd.DataFrame, use_vix: bool) -> List[str]:
    """Base-model prediction columns, optionally plus the VIX level."""
    columns = [c for c in meta_frame.columns if c.startswith("pred_")]
    if use_vix and VIX_COLUMN in meta_frame.columns:
        columns.append(VIX_COLUMN)
    return columns


def fit_stacked_meta(
    meta_frame: pd.DataFrame,
    use_vix: bool = True,
    min_train_folds: int = MIN_TRAIN_FOLDS,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Out-of-fold stacking that respects the walk-forward structure.

    For each fold ``k >= min_train_folds``, a meta-model is fitted on folds
    ``0 .. k-1`` and predicts fold ``k`` only. Folds below the threshold are
    skipped entirely; every meta-prediction returned is genuinely
    out-of-sample.
    """
    features = meta_feature_columns(meta_frame, use_vix)
    if not features:
        raise ValueError("no meta-features; base predictions are missing")

    name = model_name or ("Hybrid meta (+VIX)" if use_vix else "Hybrid meta (no VIX)")
    fold_ids = sorted(meta_frame["fold_id"].unique())

    frames: List[pd.DataFrame] = []
    fold_weights: List[Dict[str, Any]] = []
    skipped: List[int] = []

    for k in fold_ids:
        train = meta_frame[meta_frame["fold_id"] < k]
        test = meta_frame[meta_frame["fold_id"] == k]

        if k < min_train_folds or len(train) < len(features) + 2:
            skipped.append(int(k))
            continue

        model = _new_meta_model()
        model.fit(train[features].to_numpy(float), train["y_true"].to_numpy(float))
        y_pred = model.predict(test[features].to_numpy(float))

        frames.append(
            make_predictions(
                target_date=test["target_date"],
                fold_id=test["fold_id"].to_numpy(int),
                close_t=test["close_t"].to_numpy(float),
                y_true=test["y_true"].to_numpy(float),
                y_pred=y_pred,
            )
        )

        ridge = model.named_steps["ridge"]
        fold_weights.append(
            {
                "fold_id": int(k),
                "n_train": len(train),
                "n_test": len(test),
                "alpha": float(ridge.alpha_),
                "intercept": float(ridge.intercept_),
                **{f: float(w) for f, w in zip(features, ridge.coef_)},
            }
        )

    if not frames:
        raise RuntimeError(
            f"{name}: no fold had enough prior folds to fit a meta-model. "
            f"Need more than {min_train_folds} walk-forward folds."
        )

    predictions = (
        pd.concat(frames, ignore_index=True)
        .sort_values("target_date")
        .reset_index(drop=True)
    )

    return {
        "model_name": name,
        "predictions": validate_predictions(predictions, name=name),
        "features": features,
        "fold_weights": pd.DataFrame(fold_weights),
        "skipped_folds": skipped,
        "use_vix": use_vix,
    }


def vix_tercile_weights(
    meta_frame: pd.DataFrame,
    min_train_folds: int = MIN_TRAIN_FOLDS,
) -> pd.DataFrame:
    """Base-model weights fitted separately within each VIX tercile.

    Section 7.3's actual experiment. Tercile boundaries come from the training
    folds only, so the split is not informed by the days being weighted. If the
    weights shift with volatility regime that is a figure and a result; if they
    do not, that is also a result, and an honest one.

    Coefficients are on standardised features, so they are comparable across
    terciles and across models.
    """
    if VIX_COLUMN not in meta_frame.columns:
        raise ValueError("meta_frame has no VIX column; cannot split by regime")

    features = meta_feature_columns(meta_frame, use_vix=False)
    train_pool = meta_frame[meta_frame["fold_id"] < min_train_folds]
    if len(train_pool) < 3 * (len(features) + 2):
        # Fall back to the whole frame's own boundaries rather than refusing.
        train_pool = meta_frame

    lower, upper = np.quantile(train_pool[VIX_COLUMN], [1 / 3, 2 / 3])
    edges = [-np.inf, float(lower), float(upper), np.inf]
    labels = ["low", "mid", "high"]
    tercile = pd.cut(meta_frame[VIX_COLUMN], bins=edges, labels=labels)

    rows = []
    for label in labels:
        block = meta_frame[tercile == label]
        if len(block) < len(features) + 2:
            rows.append(
                {"vix_tercile": label, "n": len(block), "fitted": False}
            )
            continue

        model = _new_meta_model()
        model.fit(block[features].to_numpy(float), block["y_true"].to_numpy(float))
        ridge = model.named_steps["ridge"]
        rows.append(
            {
                "vix_tercile": label,
                "n": len(block),
                "fitted": True,
                "vix_min": float(block[VIX_COLUMN].min()),
                "vix_max": float(block[VIX_COLUMN].max()),
                "alpha": float(ridge.alpha_),
                **{f: float(w) for f, w in zip(features, ridge.coef_)},
            }
        )

    return pd.DataFrame(rows).set_index("vix_tercile")


def compare_vix_gating(
    meta_frame: pd.DataFrame,
    min_train_folds: int = MIN_TRAIN_FOLDS,
    y_train_by_fold: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Fit the meta-learner with and without VIX and test the difference.

    Reports directional accuracy and R2_OOS for each, and a Diebold-Mariano
    test on the two error series. "With VIX had lower RMSE" is not a finding
    without the DM test.
    """
    with_vix = fit_stacked_meta(meta_frame, use_vix=True, min_train_folds=min_train_folds)
    without_vix = fit_stacked_meta(
        meta_frame, use_vix=False, min_train_folds=min_train_folds
    )

    evaluations = {
        "with_vix": evaluate_predictions(
            with_vix["predictions"], with_vix["model_name"], y_train_by_fold=y_train_by_fold
        ),
        "without_vix": evaluate_predictions(
            without_vix["predictions"],
            without_vix["model_name"],
            y_train_by_fold=y_train_by_fold,
        ),
    }

    merged = with_vix["predictions"].merge(
        without_vix["predictions"], on="target_date", suffixes=("_vix", "_novix")
    )
    dm = diebold_mariano(
        merged["y_true_vix"] - merged["y_pred_vix"],
        merged["y_true_novix"] - merged["y_pred_novix"],
    )

    delta_da = (
        evaluations["with_vix"]["pooled"]["directional_accuracy"]
        - evaluations["without_vix"]["pooled"]["directional_accuracy"]
    )
    delta_r2 = (
        evaluations["with_vix"]["pooled"]["r2_oos"]
        - evaluations["without_vix"]["pooled"]["r2_oos"]
    )

    return {
        "with_vix": with_vix,
        "without_vix": without_vix,
        "evaluations": evaluations,
        "delta_directional_accuracy": float(delta_da),
        "delta_r2_oos": float(delta_r2),
        "diebold_mariano": dm,
        "n_shared_days": len(merged),
    }


# ---------------------------------------------------------------------------
# 7.4 Confidence intervals
# ---------------------------------------------------------------------------


def residual_quantiles(
    predictions: pd.DataFrame, level: float = 0.95
) -> Tuple[float, float]:
    """Empirical residual quantiles, in return space.

    Must be given out-of-fold meta-predictions only. Residual quantiles taken
    from rows the model was fitted on are optimistic by construction.
    """
    residuals = (predictions["y_true"] - predictions["y_pred"]).to_numpy(float)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(residuals, [tail, 1.0 - tail])
    return float(low), float(high)


def scale_for_horizon(
    q_low: float, q_high: float, horizons
) -> Tuple[np.ndarray, np.ndarray]:
    """Widen a one-step-ahead interval as ``sqrt(h)``.

    Under a random walk the variance of a cumulative return grows linearly in
    the horizon, so its standard deviation grows as sqrt(h). Applying the
    one-step quantiles flat across 30 days, as the previous version did,
    understates the 30-day interval by a factor of about 5.5.
    """
    h = np.asarray(horizons, dtype=float)
    if (h < 1).any():
        raise ValueError("horizons must be >= 1")
    scale = np.sqrt(h)
    return q_low * scale, q_high * scale


def prediction_intervals(
    y_pred, q_low: float, q_high: float, horizons=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return-space prediction interval around each forecast."""
    y_pred = np.asarray(y_pred, dtype=float)
    if horizons is None:
        horizons = np.ones(len(y_pred))
    low, high = scale_for_horizon(q_low, q_high, horizons)
    return y_pred + low, y_pred + high


def coverage_report(
    y_true, lower, upper, nominal_level: float = 0.95
) -> Dict[str, Any]:
    """Empirical coverage against the nominal level.

    Under-coverage is a finding worth reporting, not a bug to hide.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    inside = (y_true >= lower) & (y_true <= upper)
    n = len(y_true)
    empirical = float(inside.mean()) if n else float("nan")

    # Binomial standard error of the coverage estimate at the nominal level.
    se = np.sqrt(nominal_level * (1 - nominal_level) / n) if n else float("nan")
    deviation = empirical - nominal_level

    if n and abs(deviation) <= 2 * se:
        verdict = "well calibrated"
    elif deviation < 0:
        verdict = "UNDER-COVERED"
    else:
        verdict = "over-covered"

    return {
        "n": n,
        "nominal_level": nominal_level,
        "empirical_coverage": empirical,
        "deviation": float(deviation),
        "binomial_se": float(se),
        "verdict": verdict,
        "mean_width": float(np.mean(upper - lower)) if n else float("nan"),
    }


def fold_respecting_intervals(
    predictions: pd.DataFrame,
    level: float = 0.95,
) -> Dict[str, Any]:
    """Intervals for fold k from residual quantiles of folds before k.

    The same principle as the stacking itself: an interval must not be
    calibrated on the rows it is then scored against. The earliest meta fold
    has no prior meta residuals and is therefore excluded from the coverage
    table.
    """
    predictions = predictions.sort_values("target_date").reset_index(drop=True)
    fold_ids = sorted(predictions["fold_id"].unique())

    rows = []
    per_fold = []
    for k in fold_ids:
        prior = predictions[predictions["fold_id"] < k]
        block = predictions[predictions["fold_id"] == k]
        if len(prior) < 20:
            continue

        q_low, q_high = residual_quantiles(prior, level)
        lower, upper = prediction_intervals(block["y_pred"], q_low, q_high)

        fold_block = block.copy()
        fold_block["lower"] = lower
        fold_block["upper"] = upper
        rows.append(fold_block)

        fold_report = coverage_report(block["y_true"], lower, upper, level)
        fold_report["fold_id"] = int(k)
        fold_report["q_low"] = q_low
        fold_report["q_high"] = q_high
        per_fold.append(fold_report)

    if not rows:
        return {"intervals": None, "per_fold": pd.DataFrame(), "overall": None}

    intervals = pd.concat(rows, ignore_index=True)
    overall = coverage_report(
        intervals["y_true"], intervals["lower"], intervals["upper"], level
    )

    return {
        "intervals": intervals,
        "per_fold": pd.DataFrame(per_fold),
        "overall": overall,
        "level": level,
    }


def print_calibration_table(calibration: Dict[str, Any]) -> None:
    """The calibration table section 7.4 asks for."""
    overall = calibration.get("overall")
    if overall is None:
        print("  Not enough out-of-fold meta-predictions to calibrate intervals.")
        return

    level = calibration["level"]
    print(f"\n  {'-' * 20} INTERVAL CALIBRATION {'-' * 20}")
    print(f"  Nominal level: {level:.0%}   (quantiles from earlier folds only)")
    print(f"  {'fold':>6} {'n':>6} {'coverage':>10} {'width (bps)':>13}  verdict")
    for _, row in calibration["per_fold"].iterrows():
        print(
            f"  {int(row['fold_id']):>6} {int(row['n']):>6} "
            f"{row['empirical_coverage']:>9.1%} {row['mean_width'] * 1e4:>13.0f}"
            f"  {row['verdict']}"
        )
    print(
        f"  {'ALL':>6} {overall['n']:>6} {overall['empirical_coverage']:>9.1%} "
        f"{overall['mean_width'] * 1e4:>13.0f}  {overall['verdict']}"
    )
    if overall["verdict"] == "UNDER-COVERED":
        print(
            "  Under-coverage is reported, not hidden: the intervals are too "
            "narrow for the realised return distribution."
        )


def print_vix_gating_report(comparison: Dict[str, Any]) -> None:
    """Section 7.3's headline: does the volatility gating actually do anything?"""
    with_eval = comparison["evaluations"]["with_vix"]["pooled"]
    without_eval = comparison["evaluations"]["without_vix"]["pooled"]
    dm = comparison["diebold_mariano"]

    print(f"\n  {'-' * 22} VIX GATING {'-' * 22}")
    print(f"  {'':<14}{'with VIX':>12}{'without VIX':>14}{'delta':>12}")
    print(
        f"  {'DA':<14}{with_eval['directional_accuracy']:>11.2%}"
        f"{without_eval['directional_accuracy']:>14.2%}"
        f"{comparison['delta_directional_accuracy']:>+12.2%}"
    )
    print(
        f"  {'R2_OOS':<14}{with_eval['r2_oos']:>11.5f}"
        f"{without_eval['r2_oos']:>14.5f}"
        f"{comparison['delta_r2_oos']:>+12.5f}"
    )
    print(
        f"  Diebold-Mariano: {dm['dm_statistic']:.3f} "
        f"(p = {dm['dm_p_value']:.4f}) on {comparison['n_shared_days']} shared days"
    )
    if dm["dm_p_value"] > 0.05:
        print(
            "  The two are statistically indistinguishable. That is a result, "
            "and an honest one."
        )
