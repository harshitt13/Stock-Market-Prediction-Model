# Refactor Plan: Leak-Free Return Forecasting Pipeline

This document is the authoritative spec for refactoring the Hybrid Stock Market
Prediction Model from price-level regression to return forecasting with a
leak-free, publication-defensible evaluation protocol.

Work through the phases in order. Do not skip ahead. Each phase ends with a
commit and a green test suite.

---

## 0. Ground rules

**Branch:** do all of this on `refactor/returns-pipeline`. Keep `master` as the
"before" state so you can show a before/after comparison in the paper.

**Snapshot first.** Before changing anything, run the current pipeline on AAPL
and save `data/model_comparison.csv` to `docs/baseline_before_refactor.csv`.
This is evidence for the paper's "evaluation protocol matters" argument.

**Commit per phase.** Small commits with the phase number in the message.

**Tests are not optional.** Every phase adds tests. The leakage test in Phase 1
is the single most important piece of code in this repository.

---

## 1. The core contract

Everything below depends on one convention. Write it down, enforce it, and never
deviate.

### Row semantics

For a dataset row indexed `t`:

| Quantity | Definition |
|---|---|
| `feature_date[t]` | Trading day `t`. All features use information available at the **close** of day `t`. |
| `target_date[t]` | Trading day `t+1`. The day being forecast. |
| `close_t[t]` | `Close[t]`, the last observed price. Known at forecast time. |
| `y[t]` | `log(Close[t+1] / Close[t])`, the next-day log return. **This is the target.** |

### Prediction keying

Every model module returns predictions keyed by **`target_date`**, never by
`feature_date`. This is what kills the current off-by-one between the tree model
and the sequence models.

### Price reconstruction

Price is a **display quantity only**. Never a metric target.

```
close_hat[t+1] = close_t[t] * exp(y_hat[t])
```

No metric in the primary results table is computed on reconstructed prices.

### Standard result object

Every model, including baselines, returns a tidy DataFrame with exactly these
columns:

```
target_date   datetime64[ns]   the day being forecast
fold_id       int              which walk-forward fold produced this row
close_t       float            last observed close (day t)
y_true        float            realised log return
y_pred        float            predicted log return
```

Add a `src/contracts.py` with a `validate_predictions(df)` function that asserts
this schema, asserts `target_date` is unique and sorted, and asserts no NaNs.
Call it at the end of every model's train function. Fail loudly.

---

## 2. Phase 1: Data layer

**Files:** `src/fetch_data.py`, new `tests/test_leakage.py`

### 2.1 Delete the duplicate function

`fetch_stock_data` is defined twice (roughly lines 26 and 176). The first
definition is dead code shadowed by the second. Delete the first one.

### 2.2 Remove backward fill

`df[name].ffill().bfill()` on the macro columns pulls future observations
backward in time. This is literal lookahead. Same for
`df_meta["Current_Close"].ffill().bfill()` in `main.py`.

Use `ffill()` only, then `dropna()` on the leading rows. Do this everywhere.
Grep the whole repo for `bfill` and remove every occurrence.

### 2.3 Make features stationary

Price-level features are the reason the current model learns the identity
function. Trees cannot extrapolate past their training target range, and
MinMaxScaler on a trending series pushes every test value outside `[0, 1]`.
Replace levels with scale-free transforms:

| Drop | Replace with |
|---|---|
| `Close`, `Open`, `High`, `Low` | `hl_range = (High - Low) / Close`, `oc_ret = log(Close/Open)`, `clv = (Close - Low) / (High - Low + eps)` |
| `SMA_20`, `SMA_50` | `close_to_sma20 = Close/SMA_20 - 1`, `close_to_sma50 = Close/SMA_50 - 1` |
| `EMA_12`, `EMA_26` | `close_to_ema12`, `close_to_ema26` (same form) |
| `MACD`, `MACD_Signal`, `MACD_Hist` | divide each by `Close` |
| `BB_Upper`, `BB_Lower` | `bb_position = (Close - BB_Lower) / (BB_Upper - BB_Lower + eps)` |
| `ATR_14` | `atr_pct = ATR_14 / Close` |
| `OBV` | `obv_change = OBV.diff() / Volume.rolling(20).mean()` (raw OBV is an unbounded cumulative sum) |
| `Volume` | keep `Volume_Ratio` only; add `log_volume_change = log(Volume).diff()` |
| `Close_Lag_1/3/5` | drop; you already have `Return_1d`, `Return_5d`. Add `Return_10d`, `Return_21d` |
| `SP500` level | `SP500_Return_1d` (already present), plus `SP500_Return_5d` |
| `TNX_Yield` level | `TNX_change = TNX_Yield.diff()` and keep the level, since yields are bounded and mean-reverting over the sample |

**Keep as-is:** `RSI_14`, `BB_Width`, `Volatility_20`, `Log_Return`, `VIX`,
`VIX_Change`, `Corr_SP500_20`, `DayOfWeek`.

`VIX` stays as a level deliberately. It is mean-reverting and bounded, and the
meta-learner needs the level to gate on volatility regime.

Add a module-level constant `FEATURE_COLUMNS: list[str]` listing exactly the
features used. Never again infer the feature set by excluding columns, which is
how `Close` leaked in.

### 2.4 The leakage test

This is the most important test in the repo. Add `tests/test_leakage.py`:

```python
def test_features_do_not_depend_on_future_rows():
    """
    Feature values at row t must be identical whether computed on the full
    history or on history truncated at t. Any centred rolling window, any
    bfill, any global fit will fail this.
    """
    full = engineer_features(load_fixture())
    for t in [200, 400, 600, 800]:
        truncated = engineer_features(load_fixture().iloc[: t + 1].copy())
        for col in FEATURE_COLUMNS:
            assert np.isclose(
                full[col].iloc[t], truncated[col].iloc[t], equal_nan=True
            ), f"{col} at row {t} changed when future rows were removed"
```

Add a second test asserting `abs(corr(feature, y)) < 0.5` for every feature
against the **return** target. Any feature correlating above that with next-day
returns is a bug, not a discovery.

### 2.5 Offline test fixture

Tests currently hit Yahoo Finance, which makes them slow and flaky. Save one
ticker's raw OHLCV plus macro columns to `tests/fixtures/aapl_raw.csv` and have
`load_fixture()` read it. No test should require network access.

**Phase 1 done when:** duplicate function gone, no `bfill` anywhere,
`FEATURE_COLUMNS` explicit, leakage test passing, tests run offline.

---

## 3. Phase 2: Shared dataset builder

**New file:** `src/dataset.py`

One function, used by every model, so the target and alignment are defined
exactly once:

```python
def build_dataset(raw: pd.DataFrame) -> Dataset:
    """
    Returns a Dataset with:
      X            (n, d) float array of FEATURE_COLUMNS at feature_date
      y            (n,)   log return from close_t to the next close
      feature_date (n,)   datetime
      target_date  (n,)   datetime, = next trading day
      close_t      (n,)   float, last observed close
    The final row is dropped because its target is unknown.
    """
```

Every model module imports this. No model computes its own target ever again.

For the sequence models, add `build_sequences(dataset, lookback, fold)` that
returns windows plus the same `target_date` and `close_t` arrays, so the LSTM
and Transformer key their outputs identically to the tree model.

---

## 4. Phase 3: Metrics

**File:** `src/evaluate.py` (substantial rewrite)

### 4.1 Fix directional accuracy

There are currently two incompatible definitions. `derive_direction_labels`
takes `np.diff` of the prediction series, which measures whether prediction
`t+1` exceeds prediction `t`, and produces garbage at every fold boundary
because folds are concatenated before differencing. The classification
meta-learner uses the correct definition. Standardise on:

```
predicted_direction = sign(y_pred)          # sign of the predicted return
actual_direction    = sign(y_true)
```

Working in return space makes this trivially correct and removes the fold
boundary problem entirely. Keep the `|y_true| > threshold` filter for
near-flat days, but report the fraction of days excluded.

### 4.2 Primary metrics (return space)

- **Directional accuracy**, with the fraction of days excluded by the threshold.
- **Pesaran-Timmermann statistic** for whether directional accuracy beats
  independence. Implement it directly, it is about fifteen lines. Report the
  statistic and p-value.
- **F1 macro**, precision, recall, confusion matrix.
- **R²_OOS (Campbell-Thompson)**: `1 - SSE_model / SSE_benchmark`, where the
  benchmark is the expanding-window historical mean return computed only from
  data before each prediction. Not the test-set mean. This is the standard in
  return predictability literature and can legitimately be negative.
- **RMSE and MAE of returns**, in basis points.

### 4.3 Secondary metrics (price space, clearly labelled)

Report price MAPE and RMSE, but the table must carry an explicit note that the
naive baseline achieves essentially the same values and that these numbers do
not measure forecasting skill. Never put price R² in the paper.

### 4.4 Per-fold statistics

Stop pooling all folds into one array before computing metrics. Compute each
metric per fold, then report `mean ± std` across folds. This gives you error
bars for free and is what reviewers expect.

### 4.5 Diebold-Mariano

Add `diebold_mariano(errors_a, errors_b, h=1)` with Newey-West standard errors.
Every model comparison in the paper needs this. "Model A had lower RMSE than
model B" without a DM test is not a finding.

---

## 5. Phase 4: Model modules

**Files:** `src/tree_model.py`, `src/lstm_model.py`, `src/transformer_model.py`

Port all three to the contract. For each:

1. Consume `build_dataset` output. Delete all local target construction.
2. Predict `y` (log return). No inverse price transform inside the model.
3. Return the standard result DataFrame keyed by `target_date`.
4. Fit scalers on training folds only. The LSTM and Transformer already do this
   correctly, keep it. Switch `MinMaxScaler` to `StandardScaler`, since returns
   are roughly symmetric and unbounded and MinMax is a bad fit.
5. Drop the `StandardScaler` in the tree model. It does nothing for trees.

**Sequence target fix:** `create_sequences` currently uses
`dataset[i + time_step, 0]` as the target, which is the scaled `Close` column.
It must now take `y` from the dataset object, not a column of the feature matrix.

**Recursive multi-step forecasting:** the current loop sets
`High = Low = Close = pred`, which destroys ATR and Bollinger features, then
re-engineers the full dataframe every iteration. Compounding 30 steps of this is
indefensible in a paper.

Decision: **keep it as a demo feature, exclude it from all scientific claims.**
Move it behind a `--demo-forecast` flag, and state in the README that multi-step
forecasts are illustrative only. If you later want multi-horizon results,
train separate direct models for h = 1, 5, 21 rather than recursing.

---

## 6. Phase 5: Baselines

**File:** `src/baselines.py`

Rebuild against the same contract so the comparison is exact rather than
approximately-the-same-distribution.

- **Zero-return baseline:** `y_pred = 0` for all rows. In return space, this is
  what the naive price baseline actually is, and it is the benchmark your models
  must beat.
- **Historical mean:** expanding-window mean return.
- **AR(1) on returns**, and **ARIMA** fitted on log returns rather than prices.
- **Random-sign baseline:** useful sanity check for directional accuracy.

All four must produce the standard result DataFrame with identical
`target_date` values to the models, so `align_test_results` becomes a trivial
merge with an assertion that the date sets are equal.

---

## 7. Phase 6: Meta-ensemble

**File:** `src/main.py`

### 7.1 Fix the in-sample evaluation bug

`train_meta_ensemble_oof` fits RidgeCV on the first 80%, then returns
`meta_pred_all` covering 100% of the rows, and `run_pipeline` computes metrics
over the whole array. The reported hybrid regression numbers are therefore 80%
in-sample and not comparable to any other row in the table. The classification
path slices correctly; the regression path does not.

### 7.2 Do proper out-of-fold stacking

The 80/20 split is not out-of-fold, it is a holdout inside a holdout. Replace
with fold-respecting stacking:

> Meta-predictions for fold `k` come from a meta-model fitted only on base-model
> predictions from folds `1 .. k-1`.

This means the first two or three folds produce no meta-predictions, which is
correct and should be stated in the paper. Every meta-prediction you report is
then genuinely out-of-sample.

### 7.3 Keep the VIX gating, make it testable

The volatility-conditional weighting is the most interesting idea in the
project and is likely your actual paper contribution. Make it a first-class
experiment:

- Fit meta-learner **with** VIX and **without** VIX.
- Report the difference in directional accuracy and R²_OOS, with a DM test.
- Additionally, split test days into VIX terciles and report per-tercile base
  model weights. If the weights genuinely shift with volatility regime, that is
  a figure and a result. If they do not, that is also a result, and an honest
  one.

### 7.4 Confidence intervals

Compute residual quantiles on out-of-fold meta-predictions only. Scale interval
width with horizon (`~ sqrt(h)`) rather than applying one-step-ahead quantiles
flat across 30 days. Report empirical coverage against the nominal 95% level as
a calibration table. Under-coverage is a finding worth reporting, not a bug to
hide.

---

## 8. Phase 7: Experiment runner

**New file:** `src/experiments.py`

One stock is an anecdote. Build a runner that sweeps:

- **Tickers:** 30 to 100 names across sectors. Include survivors and delisted
  names if you can, otherwise state the survivorship bias explicitly.
- **Regimes:** pre-2020, the 2020 crash, 2021 to 2022 drawdown, 2023 onward.
- **Models:** each base model, the ensemble with and without VIX, all baselines.

Persist raw per-fold, per-ticker predictions to `results/` as parquet. Compute
aggregates from those files, never from memory. You will re-run the aggregation
many times while writing the paper and you do not want to re-train for it.

Add a `--seed` argument and run at least five seeds for the neural models.
Report seed variance. Single-seed neural results are not credible.

---

## 9. Phase 8: Economic evaluation

**New file:** `src/backtest.py`

Directional accuracy does not pay for lunch. Add a simple long/flat or
long/short strategy driven by predicted sign, and report:

- Sharpe ratio, gross and net of costs
- Maximum drawdown
- Turnover
- Break-even transaction cost, meaning the cost level at which the strategy
  stops being profitable

Use realistic costs (5 to 10 basis points round-trip for liquid US equities).
Compare against buy-and-hold. If the strategy loses to buy-and-hold after
costs, say so plainly. That is a publishable finding and reviewers respect it.

---

## 10. Expected outcome

After Phase 4 your numbers will collapse. Expect:

- R²_OOS somewhere between -0.01 and +0.01
- Directional accuracy between 50% and 54%
- Most models statistically indistinguishable from the zero-return baseline

**This is correct.** The previous MAPE of 0.89% and R² of 0.975 were the naive
baseline wearing a transformer costume. Honest daily equity return forecasting
looks like this, and every credible paper in the field reports numbers in this
range. A capstone that demonstrates a correct evaluation protocol and reports
modest honest results is worth considerably more than one claiming 99% accuracy.

Before you re-run, update the README. The line claiming "state-of-the-art
precision: ~99.11%" should be the first thing deleted.

---

## 11. Suggested Claude Code workflow

Add a `CLAUDE.md` at the repo root containing sections 1 (the contract) and 2.4
(the leakage test), so the agent has the invariants in context on every run.

Then work one phase per session:

```
Read REFACTOR_PLAN.md. Implement Phase 1 only. Do not touch any file
outside src/fetch_data.py and tests/. When done, run pytest and show me
the diff before committing.
```

Resist letting it do multiple phases at once. The contract in section 1 is easy
to violate silently, and the failures are the kind that produce good-looking
numbers rather than crashes, which is exactly what got the project here.
