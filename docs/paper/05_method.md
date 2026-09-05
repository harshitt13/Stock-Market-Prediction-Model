# 5. Method

Every number in this section is read from a committed file in the repository
at commit `783828c` (results tagged `v1.0-results` at `3fa7f8b`), and the file
is named where the number appears. Where a fact is not recorded in any
committed artefact it is marked **[unsourced]** and left as a gap rather than
filled in.

## 5.1 Data and universe

**Source.** Daily open, high, low, close and volume from Yahoo Finance through
the `yfinance` client, joined to three daily macro series from the same
source: the S&P 500 index (`^GSPC`), the CBOE VIX (`^VIX`) and the 10-year
Treasury yield (`^TNX`) (`src/fetch_data.py`, `MACRO_SYMBOLS`,
`fetch_stock_data`). The macro series are forward-filled only; a backward fill
would move a future observation into the past and is excluded by test (§5.3).

**The headline series.** The single-ticker results (Table 1) are computed on a
frozen file, `docs/frozen_aapl_raw.csv`: AAPL, 4144 trading days from
2010-03-16 to 2026-09-03, 34 columns (raw inputs plus the previous feature
set, which is discarded and recomputed on load), SHA-256
`213808b8e187e49a6a0406e5d217428b1ded57848409d42a723945b950a57ac5`
(README §9). Freezing the file is what makes the before/after comparison
in §6.1 a comparison of code alone: both protocols read the same bytes.
Feature engineering (§5.3) consumes 49 leading rows for the longest rolling
window and one trailing row whose next-day target does not exist, leaving
4094 rows of 31 features, the first dated 2010-05-25 (computed from the frozen
file with `src/dataset.py`, `build_dataset`).

**The universe.** The cross-sectional evaluation uses thirty US large-cap
equities in six sector groups, listed in `src/experiments.py`
(`DEFAULT_TICKERS`):

| Group | Tickers |
|---|---|
| Technology | AAPL, MSFT, NVDA, AVGO, CRM, ORCL |
| Communication services | GOOGL, META, NFLX, DIS |
| Consumer | AMZN, TSLA, HD, MCD, NKE, PG, KO, WMT |
| Financials | JPM, BAC, GS, BRK-B |
| Health care | JNJ, UNH, PFE, MRK |
| Industrials, energy, utilities | CAT, BA, XOM, NEE |

Each was fetched from 2010-01-01 with an open end date
(`src/fetch_universe.py`, `--start` default; `src/experiments.py`,
`REGIMES["full"]`). The fetch cache itself is not committed (`results/raw/`
is gitignored), so **[unsourced]**: the per-ticker raw row counts and end
dates of the 29 non-AAPL series, and the calendar date on which the sweep's
cache was fetched, are not recoverable from a committed artefact. What is
committed is every prediction the sweep made (`results/predictions/`, 595,224
rows). From those, the first forecast day is 2014-03-19 for 28 of the 30
tickers; the two later listings start later and have fewer folds, META from
2016-08-03 with 10 folds and TSLA from 2014-09-11 with 11. Last forecast days
run from 2025-09-18 to 2026-08-12, the spread being a consequence of the
start-anchored fold grid (§5.4) rather than of differing data ends.

**Survivorship bias, stated.** All thirty names trade today. Firms that were
delisted, acquired or went bankrupt between 2010 and 2026 are absent, which
biases every cross-sectional aggregate upward: the surviving universe
outperformed the universe that existed at the time. The sweep prints this
warning on every run (`src/experiments.py`, `SURVIVORSHIP_WARNING`; README
§8.2). Correcting it needs point-in-time index constituents including
delisted names, which this study does not have.

**The 5-versus-30 split, stated.** The thirty-ticker evaluation covers the
tree ensemble and the five baselines only. The two neural architectures and
the stacked meta-learner were run on five of the thirty, spanning five of the
six sector groups: AAPL, JNJ, JPM, WMT and XOM. The committed predictions
show this directly: five tickers carry all ten models and twenty-five carry
six (`results/predictions/`). **[unsourced]**: the reason for the
restriction and the basis for choosing those five are not recorded in any
committed artefact. The compute context is: a full run of all models on one
ticker takes about a hundred minutes on the machine used (README §9,
"Runtime"), and the default quarterly configuration would take 11 to 45
hours per run (README §8.3); that is context, not a cited decision. Every
claim in §6.2 about "30 tickers" is therefore a claim about the
tree ensemble; the neural null is established on five tickers, and on one of
them across five seeds (§5.4).

## 5.2 The row contract

Everything downstream depends on one convention, reproduced from
`REFACTOR_PLAN.md` §1 and enforced in code by `src/contracts.py`. For a
dataset row indexed *t*:

| Quantity | Definition |
|---|---|
| `feature_date[t]` | Trading day *t*. Every feature uses only information available at the **close** of day *t*. |
| `target_date[t]` | Trading day *t+1*, the day being forecast. |
| `close_t[t]` | `Close[t]`, the last observed price, known at forecast time. |
| `y[t]` | `log(Close[t+1] / Close[t])`, the next-day log return. **This is the target.** |

Three consequences follow.

*Predictions are keyed by `target_date`, never by `feature_date`.* A tree
consumes one row and a sequence model consumes a 90-day window ending at the
same row; keying both outputs by the day they forecast is what removed the
off-by-one between them that the previous pipeline carried. A test asserts
that all three base models forecast exactly the same set of days (README §6).

*Price is a display quantity only.* Where a price is wanted it is
reconstructed as `close_hat[t+1] = close_t[t] · exp(y_hat[t])`, and no primary
metric is computed on a reconstructed price (`REFACTOR_PLAN.md` §1; the
price-space columns in the results tables are labelled secondary).

*Every model returns the same object.* A frame with exactly the columns
`target_date, fold_id, close_t, y_true, y_pred`. `validate_predictions`
asserts the schema and dtypes, rejects NaN and infinite values, and requires
`target_date` to be unique and sorted; it is called at the end of every train
function, including every baseline, and fails loudly (`src/contracts.py`).
The final row of every series is dropped at dataset construction because its
target does not exist (`src/dataset.py`, `build_dataset`).

## 5.3 Features

Thirty-one features, all computed in `src/fetch_data.py` (`engineer_features`)
from information at or before the close of day *t*, and all scale-free: no
price level, no volume level and no cumulative sum enters the feature matrix.
The set is an explicit whitelist (`FEATURE_COLUMNS`); every raw column is
banned as a feature unless named in `LEVEL_FEATURES_ALLOWED`, which contains
exactly `VIX` and `TNX_Yield`, both bounded and mean-reverting over the
sample, the VIX level being needed by the meta-learner's gating variant
(§5.5). A new raw column is therefore banned by default. Fourteen
level-valued intermediates from the previous feature set (`SMA_20`, `EMA_12`,
`MACD`, `BB_Upper`, `OBV`, lagged closes, and so on) are computed as locals and
dropped, and a frozen file that still carries them has them stripped on load
(`LEGACY_LEVEL_COLUMNS`).

The transformations, with *C, O, H, L, V* the close, open, high, low and
volume on day *t*, *ε* = 10⁻⁸, and all windows in trading days:

| # | Feature | Definition | Window |
|---|---|---|---|
| 1 | `hl_range` | (H − L) / C | 1 |
| 2 | `oc_ret` | log(C / O) | 1 |
| 3 | `clv` | (C − L) / (H − L + ε) | 1 |
| 4 | `close_to_sma20` | C / SMA₂₀(C) − 1 | 20 |
| 5 | `close_to_sma50` | C / SMA₅₀(C) − 1 | 50 |
| 6 | `close_to_ema12` | C / EMA₁₂(C) − 1 | 12 |
| 7 | `close_to_ema26` | C / EMA₂₆(C) − 1 | 26 |
| 8 | `macd_norm` | (EMA₁₂ − EMA₂₆) / C | 26 |
| 9 | `macd_signal_norm` | EMA₉(MACD) / C | 26 + 9 |
| 10 | `macd_hist_norm` | (MACD − EMA₉(MACD)) / C | 26 + 9 |
| 11 | `RSI_14` | 100 − 100 / (1 + mean gain / mean loss), simple 14-day means; NaN → 50, ∞ → 100 | 14 |
| 12 | `bb_position` | (C − BB_lower) / (BB_upper − BB_lower + ε), bands SMA₂₀ ± 2·SD₂₀ | 20 |
| 13 | `BB_Width` | (BB_upper − BB_lower) / SMA₂₀ | 20 |
| 14 | `atr_pct` | ATR₁₄ / C, true range = max(H − L, \|H − C₋₁\|, \|L − C₋₁\|) | 14 |
| 15 | `Volatility_20` | SD₂₀ of `Log_Return` | 20 |
| 16 | `Log_Return` | log(C / C₋₁) | 1 |
| 17 | `Return_1d` | C / C₋₁ − 1 | 1 |
| 18 | `Return_5d` | C / C₋₅ − 1 | 5 |
| 19 | `Return_10d` | C / C₋₁₀ − 1 | 10 |
| 20 | `Return_21d` | C / C₋₂₁ − 1 | 21 |
| 21 | `obv_change` | ΔOBV / mean₂₀(V), OBV = Σ sign(ΔC)·V | 20 |
| 22 | `Volume_Ratio` | V / mean₂₀(V) | 20 |
| 23 | `log_volume_change` | Δ log V | 1 |
| 24 | `DayOfWeek` | weekday of day *t*, 0–4 | — |
| 25 | `SP500_Return_1d` | S&P 500 one-day simple return | 1 |
| 26 | `SP500_Return_5d` | S&P 500 five-day simple return | 5 |
| 27 | `Corr_SP500_20` | 20-day rolling correlation of C with the S&P 500 level | 20 |
| 28 | `VIX` | VIX close, as a level (allowed) | — |
| 29 | `VIX_Change` | ΔVIX | 1 |
| 30 | `TNX_Yield` | 10-year yield, as a level (allowed) | — |
| 31 | `TNX_change` | ΔTNX | 1 |

Rows 1–24 need only the ticker's own OHLCV; 25–31 need the macro joins
(`BASE_FEATURE_COLUMNS`, `MACRO_FEATURE_COLUMNS`). Warm-up rows are left as
NaN and dropped explicitly, never filled; the 50-day moving average sets the
warm-up at 49 rows. Feature 27 is a correlation of levels, not of returns; it
is reported as implemented.

**Leakage controls.** The contract is tested rather than assumed
(`tests/test_leakage.py`, `tests/test_leakage_mutations.py`; summarised in
README §7). The central test is truncation invariance: every feature at row
*t* must be identical whether computed on the full history or on the history
truncated at *t*, checked at rows 200, 400, 600 and 800 of the test fixture
and, for a fixture whose macro series begins after its price series, at rows
1, 3 and 5, where a backward fill is the only thing that can differ
(`TRUNCATION_POINTS`, `LEADING_TRUNCATION_POINTS`). The test's own power is
demonstrated by mutation: six known leaks are injected into a copy of the
feature code and each must be caught: a macro backward fill, a centred
20-day moving average, a centred rolling volume, a volatility fitted on the
whole series, a direct one-row-ahead read, and an RSI z-scored on the full
sample (`tests/test_leakage_mutations.py`, `MUTATIONS`). A further test bounds
the absolute correlation of any feature with the next-day return at 0.5, on
the grounds that anything higher means the target has leaked in
(`MAX_ABS_TARGET_CORRELATION`); the observed maximum is 0.205, for
`SP500_Return_1d` (README §7). The fixture on which these tests run is 1008
raw rows of AAPL, 2018-01-02 to 2021-12-31 (`tests/fixtures/aapl_raw.csv`),
so the suite is offline; the data client is imported lazily and a subprocess
test asserts it is absent after the data layer loads. The suite has 331 tests
(README §7).

## 5.4 Walk-forward evaluation

**The splitter.** Expanding-window walk-forward (`src/walk_forward.py`,
`WalkForwardSplitter`): fold *k* trains on rows [0, *m* + *k·s*) and tests on
the next *n* rows, with *m* the minimum training size, *n* the test size and
*s* the step. Test folds never overlap and training always precedes testing.

**The headline configuration.** *m* = 1008, *n* = 252, *s* = 252, which on the
4094-row AAPL dataset gives twelve folds, each a year of forecasts
(`README` §9, the reproduction command; `results/headline/AAPL__frozen__seed42.parquet`).
The boundaries, from the committed predictions and the frozen file:

| Fold | Training rows | Training span | Test span |
|---|---|---|---|
| 0 | 1008 | 2010-05-25 to 2014-05-27 | 2014-05-29 to 2015-05-28 |
| 1 | 1260 | to 2015-05-27 | 2015-05-29 to 2016-05-26 |
| 2 | 1512 | to 2016-05-25 | 2016-05-27 to 2017-05-26 |
| 3 | 1764 | to 2017-05-25 | 2017-05-30 to 2018-05-29 |
| 4 | 2016 | to 2018-05-25 | 2018-05-30 to 2019-05-30 |
| 5 | 2268 | to 2019-05-29 | 2019-05-31 to 2020-05-29 |
| 6 | 2520 | to 2020-05-28 | 2020-06-01 to 2021-05-28 |
| 7 | 2772 | to 2021-05-27 | 2021-06-01 to 2022-05-27 |
| 8 | 3024 | to 2022-05-26 | 2022-05-31 to 2023-05-31 |
| 9 | 3276 | to 2023-05-30 | 2023-06-01 to 2024-05-31 |
| 10 | 3528 | to 2024-05-30 | 2024-06-03 to 2025-06-04 |
| 11 | 3780 | to 2025-06-03 | 2025-06-05 to 2026-06-05 |

Each test fold has 252 days; 62 rows after the last fold are unused. The
thirty-ticker sweep uses the same *m, n, s* on each ticker's own series
(`results/fold_diagnostics.csv`: `n_train` = 1008 at fold 0 and `n_test` = 252
throughout; 357 fold-ticker rows, being 28 × 12 + 11 + 10).

**The common evaluation window.** The meta-learner for fold *k* is fitted on
the out-of-fold predictions of folds 0 to *k*−1, so it produces nothing for
the first two folds (`src/meta_ensemble.py`, `MIN_TRAIN_FOLDS` = 2). Every
primary comparison is therefore computed on the **intersection** of all
models' forecast days, including the meta's: folds 2 to 11, 2520 days,
2016-05-27 to 2026-06-05 (README §3.1; the meta rows in the headline parquet
number exactly 2520). `common_evaluation_window` asserts the intersection is
non-empty and contiguous, since a hole would mean a model is missing days the
others have (`src/contracts.py`). Scoring the base models on the two extra
folds the meta never saw is one of the protocol artefacts measured in §4; on
the fixture, those folds were the 2020 crash (README §5.1).

**Limitation: the annual refit.** A 252-day test fold with a 252-day step
means every model is refitted once per fold, once per year, and forecasts for
up to twelve months without seeing new data. A model trading in May 2026 was
last fitted on data ending in June 2025. Real deployment would refit monthly
or daily. This was a compute concession: a full run of one ticker costs
about a hundred minutes under this configuration (README §9), the default
quarterly configuration 11 to 45 hours (README §8.3), and a monthly refit
would multiply the fold count by twelve again. The direction of the
bias is against the models. The distribution-shift analysis in §6.4 shows a
third of folds train on a materially different return distribution from the
one they are scored on, and a staler model suffers more from that; the annual
refit therefore plausibly accounts for part of the poor performance, and the
limitation is stated here rather than in §8 alone (README §8.1).

**The fixture configuration.** The mechanism experiments reported in §4
(the cold-start benchmark and the short-window artefacts) were run on the
test fixture with *m* = 400, *n* = 60, *s* = 60 (`analysis/ridge_grid_and_tree.py`),
a quarterly refit that gives nine folds on the fixture's 958 dataset rows and
a 420-day common window for the meta (computed from `tests/fixtures/aapl_raw.csv`
with `src/dataset.py` and `src/walk_forward.py`; the 420-day window is the
one named in `analysis/alpha_correction_and_window.py`). The fixture is for
demonstrating mechanism, not for producing results; no headline number comes
from it.

**Seeds.** The headline run and the sweep use seed 42 throughout
(`src/main.py` default; the `seed` column of every committed prediction).
The seed study reran the two neural architectures on AAPL for seeds 0 to 4
under the same 1008/252/252 configuration (`seed_study.py`;
`results/seeds/`, BiLSTM and Transformer only, 12 folds each). The seed study
ran on the sweep's AAPL fetch cache rather than the frozen file, so its fold
grid opens 49 trading days earlier than Table 1's (2014-03-19 against
2014-05-29; computed from `results/predictions/` and the frozen file by
`analysis/make_figures.py`, `grid_offset`).

## 5.5 Models

No hyperparameter was tuned against a test fold. The values below are the
ones in the committed code; the transformer carries an optional Optuna search
over its own training data that is off by default and was not used in any
reported run (`src/main.py`, `--optimize` default false; `src/experiments.py`,
`run_single` takes the default). Every model is fitted from scratch on each
fold's training rows.

**Tree ensemble** (`src/tree_model.py`, `_build_tree_ensemble`). An
equal-weight average (`VotingRegressor`) of a gradient-boosted regressor
(`XGBRegressor`: 500 trees, depth 6, learning rate 0.05, row subsample 0.8,
column subsample 0.8, L1 penalty 0.1, L2 penalty 1.0) and a random forest
(`RandomForestRegressor`: 300 trees, depth 12, minimum 5 samples to split and
3 per leaf, √*d* features per split). Inputs are not scaled: trees split on
rank and a scaler changes nothing.

**Bidirectional LSTM** (`src/lstm_model.py`). Input windows of 90 days
(`DEFAULT_LOOKBACK`), each row the 31 features, standardised by a
`StandardScaler` fitted on the fold's training rows only; the target is
standardised the same way and predictions are mapped back
(`src/model_utils.py`, `fit_feature_scaler`, `fit_target_scaler`). A test
window may reach back into training rows for its lookback, which is reaching
backwards and is not leakage (`src/dataset.py`, `build_sequences`). Three
stacked bidirectional LSTM layers of 128, 64 and 32 units per direction, each
followed by batch normalisation and dropout 0.3; the final time step feeds a
head of 64 and 32 units with ReLU and dropout 0.2, then one output. Training:
Huber loss (δ = 1), Adam at 10⁻³, learning rate halved on a plateau of five
epochs down to 10⁻⁶, batch 64, gradient norm clipped at 1.0, up to 100
epochs with early stopping after 15 epochs without improvement on a
validation set that is the time-ordered last 10% of the training windows.

**Transformer** (`src/transformer_model.py`). The same windows, scalers and
training loop. A linear projection of the 31 features to *d* = 64, sinusoidal
positional encoding, three encoder layers with 4 heads, feed-forward width
128 and dropout 0.2; the representation at the last position feeds a head of
32 units with ReLU and dropout, then one output (`DEFAULT_PARAMS`,
`TimeSeriesTransformer`).

**Two linear comparators** (`src/linear_models.py`). To span the capacity
range from linear to Transformer with something at its bottom that has
almost no capacity to misbuild, two well-regularised linear models run on
the same folds and the same contract as every other model, in the headline
run and on all thirty sweep tickers. `Ridge (returns)` standardises the 31
features on the fold's training rows and fits `RidgeCV` over the
meta-learner's penalty grid, 10⁻³ to 10⁶, to the next-day log return; its
selected penalty and the ratio of its prediction spread to the training
return spread are recorded per fold (`results/linear_fits_headline.csv`,
`results/linear_fits_sweep.csv`; `analysis/run_linear_comparators.py`).
`Logistic (direction)` fits an L2 logistic regression to the sign of the
return with the inverse penalty chosen from 10⁻⁴ to 10⁴ by a
time-respecting inner cross-validation on the training rows
(`TimeSeriesSplit`, five splits); because the contract needs a return-scale
number, its output is (2*p* − 1) times the training fold's return standard
deviation, so the sign is the class decision and the magnitude is
confidence at the training scale. Its R²_OOS is reported for completeness
and is not a return forecast. Neither comparator feeds the meta-learner,
whose inputs remain the three base models, so that no reported
meta-learner number changed when they were added.

**Stacked meta-learner with VIX gating** (`src/meta_ensemble.py`). For fold
*k* ≥ 2, a ridge regression is fitted on the out-of-fold predictions that the
three base models made for folds 0 to *k*−1, with the realised return as the
target, and predicts fold *k* only, so every meta-prediction is genuinely
out-of-sample (`fit_stacked_meta`). Inputs are standardised and the penalty
is chosen by `RidgeCV` over the grid 10⁻³ to 10⁶ in decades (`RIDGE_ALPHAS`),
with the estimator's default leave-one-out selection. Two variants are
reported: with the VIX level as a fourth input (`Hybrid meta (+VIX)`) and
without it (`Hybrid meta (no VIX)`). "Gating" is tested, not assumed: the two
variants are compared by a Diebold-Mariano test on their error series
(`compare_vix_gating`), and base-model weights refitted within VIX terciles are
reported as a diagnostic (`vix_tercile_weights`). The penalty `RidgeCV`
selected in each fold of the headline run is committed
(`results/AAPL__frozen__seed42_meta_fits.csv`, written by
`analysis/persist_meta_fits.py`, which refits both variants from the stored
base predictions and asserts the refit reproduces the parquet's meta
predictions exactly). Over the ten fitted folds the +VIX variant chose 10³
three times, 10⁴ four times and 10⁶, the top of the grid, three times; the
no-VIX variant chose 10³ six times, 10⁴ three times and 10⁶ once. The only
tuned component in the pipeline therefore chose heavy shrinkage in every
fold and, in four fold-fits, the heaviest the grid allowed.

**Software.** The pipeline is Python with pandas, NumPy, scikit-learn,
XGBoost, PyTorch, statsmodels, SciPy and Optuna (`requirements.txt`, the
unpinned install list). The exact versions in the environment that produced
the committed results are recorded in `requirements-lock.txt`, a `pip freeze`
of that environment.

## 5.6 Baselines

Five baselines, on the same folds, the same contract and the same evaluation
window as the models (`src/baselines.py`). They are ordered from the one that
predicts nothing to the one that predicts noise.

| Baseline | Forecast for day *t*+1 | Fitted on |
|---|---|---|
| Zero return | 0 | nothing |
| Historical mean | expanding mean of every return observed before the day, seeded with the fold's training returns and extended with realised test returns as they arrive | training fold, then expanding |
| AR(1) returns | *c* + *φ·y[t−1]*, with *y[t−1]* the return from close *t*−1 to close *t*, known at the close of day *t* | OLS on the training fold |
| ARIMA(5, 0, 0) returns | one-step-ahead forecast from an AR(5) on log returns; realised test returns are appended with the coefficients frozen at their training values | maximum likelihood on the training fold (statsmodels) |
| Random sign | ±1 with equal probability, scaled by the training fold's return standard deviation, seed 42 | nothing |

The zero-return baseline is the benchmark, not a strawman: it is what the
previous protocol's "naive" price baseline was in return space, and for daily
equity returns it is hard to beat. The historical mean is exactly the
Campbell-Thompson benchmark of §5.7, so its R²_OOS is zero by construction and
it is the model that "is the market" in the economic tests. ARIMA is fitted
with *d* = 0 because returns are already differenced; the previous pipeline's
(5, 1, 0) was differencing a price series.

## 5.7 Metrics

All primary metrics are computed in return space on the common evaluation
window (`src/evaluate.py`, `src/backtest.py`). Price-space error metrics are
computed on reconstructed prices, reported in the tables as secondary and
labelled as such; on a trending series they are near-perfect for every model
and measure the price series, not the forecast (`src/evaluate.py`,
`PRICE_METRICS_CAVEAT`).

**Direction.** Directional accuracy is the fraction of days on which
sign(*ŷ*) = sign(*y*), computed after excluding near-flat days with
|*y*| ≤ 10 bps (`DIRECTION_THRESHOLD` = 0.001); the excluded fraction is
reported beside every accuracy figure because an accuracy on a different
denominator is a different number (7.0% of the AAPL common window,
`docs/baseline_after_refactor.csv`). Accuracy is compared not with 50% but
with the **majority-class rate**, max(*p*, 1 − *p*) for *p* the fraction of
up days among the kept rows, which is what "always predict the common class"
scores; the reported quantity is DA minus majority, in percentage points
(`directional_accuracy`). Significance of direction is the Pesaran-Timmermann
(1992) test, whose null is independence of predicted and realised signs
(`pesaran_timmermann`).

**Magnitude.** Out-of-sample R² follows Campbell and Thompson (2008):
R²_OOS = 1 − SSE_model / SSE_benchmark, where the benchmark for each day is
the expanding mean of all returns before it. The benchmark is **seeded with
the fold's training returns** and extended with realised test returns
(`expanding_mean_benchmark`); a benchmark that restarts from nothing at each
fold's first day is the cold-start artefact measured in §4. The training
returns are a required argument of the evaluation function: omitting them
is an error at the call site, and a mapping that lacks any fold raises
before a metric is computed (`evaluate_predictions`,
`require_training_returns`; §4.2 records why this is structural rather than
a flag). R²_OOS is legitimately negative when the
model is worse than the mean. RMSE and MAE are reported in basis points of
log return. Every metric is also computed per fold, and the standard
deviation across folds is reported beside the pooled value
(`per_fold_metrics`, `summarize_across_folds`).

A note on aggregation. Pooled R²_OOS is a ratio of sums of squared errors
over every test day, so each fold enters it in proportion to its
variance, and the statistic inherits the fragility of the
highest-variance fold: moving one fold boundary through a market crash
changes the pooled value by more than the effects under study (§6.2,
§8.3). The across-fold mean and median weight every fold equally. We
therefore report the across-fold median of R²_OOS as the headline
magnitude, with the across-fold mean and standard deviation and the
pooled value beside it, computed from the committed predictions by
`analysis/fold_aggregates.py` (`results/fold_aggregates_headline.csv`,
`results/fold_aggregates_sweep.csv`). Signs, counts and corrected
p-values, on which the paper's claims rest, are reported under both. The
same fragility appears wherever a statistic is pooled over folds: the grid
analysis of §8.3, where one fold moved pooled R²_OOS by 0.09, and the one
directional hit that survives a within-model Holm correction in §6.2, whose
pooled edge of +2.1 points sits over a fold median of −0.7. We treat the
three as one recurring finding.

**Equal predictive accuracy.** Diebold-Mariano (1995) on squared errors,
each model against the zero-return baseline on exactly the days the two
share (`dm_table`; the reference is `Zero return`, `src/main.py`). The
long-run variance uses Newey-West weights with truncation lag *h* − 1, which
for the one-step horizon used here is lag 0, so the statistic reduces to the
mean loss differential over its plain standard error (`diebold_mariano`).
The VIX-gating comparison of §5.5 uses the same test between the two
meta-learner variants.

**Economics.** Each model is traded as a long/flat rule, long the next day
when *ŷ* > 0 and flat otherwise, and compared with buy-and-hold on the same
days (`positions_from_predictions`, `backtest`). Log returns are converted to
simple returns before compounding. Transaction cost is 7.5 basis points per
round trip, charged as half that per unit of turnover, turnover being the
absolute change in position (`DEFAULT_COST_BPS`); a stress grid re-runs the
tree at 0, 7.5 and 10 bps with and without a one-day execution lag, the lag
shifting each position one day later (`stress_alpha`, `execution_lag`;
`analysis/alpha_correction_and_window.py`). Reported quantities: net and
gross annualised Sharpe ratio, zero risk-free rate, √252 scaling, sample
standard deviation (`_sharpe`); average exposure, the fraction of days long;
annualised turnover; and the break-even round-trip cost
2·10⁴·mean(gross return)/mean(turnover), shown only when annualised turnover
is at least 5 because below that it divides by nothing
(`breakeven_cost_bps`, `MIN_TURNOVER_FOR_BREAKEVEN`). Skill is separated
from exposure by regressing the strategy's daily returns on buy-and-hold's,
*r*ₛ = α + β·*r*ₘ + *e*; α is annualised by 252 and its t-statistic uses plain
OLS standard errors, not a HAC estimator, and should be read as indicative
(`market_adjusted`). Because eleven strategies are tested for alpha on the same
asset (buy-and-hold is the market and the never-trading zero-return rule has
no alpha to test; README §3.4), the p-values are corrected by the
Holm-Bonferroni step-down
procedure, which controls the family-wise error rate without assuming
independence (`holm_bonferroni`).

**Distribution shift.** For every fold of every ticker, a two-sample
Kolmogorov-Smirnov test between the training and the test return
distributions, its 5% rejection flag, and the fraction of test returns
outside the training range (`src/experiments.py`, `fold_return_diagnostics`;
`results/fold_diagnostics.csv`, 357 rows). This is a diagnostic of how much
any model could have learned that still applied, and it is used in §6.4 to
explain the magnitude of the error, not its sign.

### Artefacts cited in this section

| File | What it sources here |
|---|---|
| `docs/frozen_aapl_raw.csv` | the headline series: rows, span, columns; row counts after engineering (with `src/dataset.py`) |
| `results/headline/AAPL__frozen__seed42.parquet` | the twelve headline fold boundaries; the 2520-day meta window |
| `results/predictions/` | sweep coverage: 30 tickers, which carry ten models and which six, first and last forecast days, fold counts |
| `results/fold_diagnostics.csv` | sweep fold sizes; the 357 fold-ticker pairs |
| `results/seeds/` | the seed study's models, seeds and folds |
| `results/AAPL__frozen__seed42_meta_fits.csv` | the ridge penalty, intercept and coefficients the meta-learner fitted in each fold |
| `requirements-lock.txt` | library versions of the environment behind the committed results |
| `tests/fixtures/aapl_raw.csv` | the fixture: rows, span; nine folds and 420 days under 400/60/60 |
| `src/fetch_data.py`, `src/dataset.py`, `src/contracts.py`, `src/walk_forward.py` | data source, features, contract, splitter |
| `src/tree_model.py`, `src/lstm_model.py`, `src/transformer_model.py`, `src/meta_ensemble.py`, `src/model_utils.py`, `src/baselines.py` | every model and baseline setting |
| `src/evaluate.py`, `src/backtest.py`, `src/experiments.py` | every metric definition |
| `src/main.py`, `src/fetch_universe.py`, `seed_study.py`, `analysis/ridge_grid_and_tree.py`, `analysis/alpha_correction_and_window.py`, `analysis/make_figures.py`, `analysis/persist_meta_fits.py` | defaults, fetch window, seed study, fixture configuration, grid offset, meta-fit persistence |
| `tests/test_leakage.py`, `tests/test_leakage_mutations.py` | truncation points, correlation bound, the six injected leaks |
| `README.md` §3.1, §6, §7, §8, §9 | common window dates, contract test, observed maximum correlation, test count, limitations, SHA-256 and runtime |
| `REFACTOR_PLAN.md` §1 | the row contract, reproduced |
