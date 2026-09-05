# 4. Protocol artifacts

A protocol artifact is an evaluation choice that produces a good-looking
number without a forecast behind it. None of the five below crashes a
pipeline. Each produced plausible output in ours, and each was found by
noticing that a trivial baseline scored in a way it could not honestly score.
This section treats them one at a time, and for each covers three things in
order: the mechanism, the magnitude measured in our own pipeline with the
artifact cited, and how a reader detects it in their own work. It closes
with a checklist.

Every number is read from a committed file in the repository at the results commit recorded in §9.1 (results tagged `v1.0-results` at `3fa7f8b`) or from the message
of a commit in its history, and the source is named beside it. Anything not
so recorded is marked **[unsourced]**.

| # | Artifact | Measured effect | Where |
|---|---|---|---|
| 4.1 | Price-level target with the previous close as a feature | R² 0.9627 → R²_OOS −0.0003, same model, same data, same folds | `docs/baseline_before_refactor.csv`, `docs/baseline_after_refactor.csv` |
| 4.2 | Cold-start R²_OOS benchmark | zero-return baseline +0.087 vs −0.005 (fixture); +0.0349 vs −0.0006 (30 tickers); 1 of 30 tickers moved from apparently positive to 0 of 30 | commit `db46cb3`, `tests/test_baselines.py`; recomputed from `results/predictions/` against `results/per_ticker_model.csv` |
| 4.3 | Mismatched evaluation windows across models | base models lost 1.9 to 2.3 points of directional accuracy; their lead over the meta-learner narrowed from about 5 points to 2.3 to 3.5 | commit `710d90c`; README §5.1, §12 |
| 4.4 | Short evaluation window | the same VIX-gating test: DM +2.32, p = 0.0203 on 420 days; DM −0.595, p = 0.552 on 2520 days; opposite signs | `analysis/alpha_correction_and_window.py`; README §12 |
| 4.5 | Single-seed neural results | the project's only significant directional result appears in 1 of 5 seeds; the Transformer's R²_OOS changes sign across seeds | `results/seeds/`; README §3.3; commit `3fa7f8b` |

## 4.1 A price-level target with the previous close as a feature

**Mechanism.** Ask a model to predict tomorrow's closing price and give it
today's closing price as an input. The cheapest function that fits is the
identity: output the input. On a trending price series the variance of the
target is almost entirely the variance of the level, which the identity
reproduces, so R² on the price target measures the price series rather than
the forecast, and a mean absolute percentage error of one or two percent is
what a random walk looks like at a daily horizon. The number that exposes
this is the naive "tomorrow equals today" baseline: under this protocol it is
the identity function computed exactly, so it scores at least as well as any
model that has merely approximated the identity. A protocol that reports
R² on price levels without that baseline cannot tell a forecaster from a
copy.

**Magnitude.** The previous version of this pipeline predicted the next-day
close from a feature set that included the close, moving averages of the
close, Bollinger bands and lagged closes (the intermediates now listed in
`src/fetch_data.py`, `LEGACY_LEVEL_COLUMNS`). Its comparison table was
frozen as `docs/baseline_before_refactor.csv`, produced at commit `3933168`
on the frozen file `docs/frozen_aapl_raw.csv` with the twelve-fold
configuration of §5.4 (commit `d26ad59`). The refactored pipeline was then
run on the same file with the same folds, giving
`docs/baseline_after_refactor.csv`.

| Model | Before: R² on price | Before: MAPE (%) | After: R²_OOS on return |
|---|---|---|---|
| Hybrid meta (regressor; "+VIX" after) | **0.9627** | 9.89 | **−0.00029** |
| Tree ensemble | 0.9482 | 9.27 | −0.02037 |
| Transformer | 0.8562 | 17.59 | +0.00102 |
| BiLSTM | 0.4488 | 40.48 | −0.00499 |
| ARIMA (5,1,0) before; (5,0,0) on returns after | 0.8876 | 16.30 | −0.00753 |
| Naive zero-change before; zero return after | **0.9992** | **1.23** | −0.00276 |

The "after" column is the pooled R²_OOS. For the same meta-learner the
across-fold median is +0.0011 and the across-fold mean −0.0022 ±
0.0109 (§6.1; `results/fold_aggregates_headline.csv`); under no
aggregation is it distinguishable from zero, and the contrast with 0.9627
is the same under all three.

The naive baseline beat every model on both of the old protocol's metrics,
and the protocol could not show it because the table had no row that said
"this is what doing nothing scores". The best model's R² of 0.9627 became
R²_OOS of −0.00029 with nothing changed but the target and the evaluation:
same bytes in, same fold boundaries, same hyperparameters. The old table's
directional accuracy column (46 to 52%) was itself an artifact of a second
kind, computed by differencing the concatenated prediction series and so
inventing an observation at every fold boundary (README §12, item 2); it is
not used here.

**Detection.**

- Put a last-value baseline in every table, under the same metric as the
  models. If it sits at or near the top, the metric is measuring the series.
  In this pipeline the zero-return baseline is a first-class model with the
  same contract as every other (`src/baselines.py`,
  `zero_return_baseline`).
- Move the target to return space and score against a benchmark that is
  itself a forecast (§4.2). The price-space metrics survive only as a
  labelled secondary table (`src/evaluate.py`, `PRICE_METRICS_CAVEAT`).
- Bound the correlation between any feature and the target. Here no feature
  may correlate above 0.5 in absolute value with the next-day return, on the
  grounds that anything higher means the target has leaked in; the observed
  maximum is 0.205 (`tests/test_leakage.py`,
  `MAX_ABS_TARGET_CORRELATION`; README §7).
- Ban levels by default. Every raw column is excluded from the feature set
  unless it is named in an explicit allow-list with a reason
  (`src/fetch_data.py`, `LEVEL_FEATURES_ALLOWED`;
  `tests/test_leakage.py`, `test_feature_set_contains_no_price_levels`).

## 4.2 The cold-start R²_OOS benchmark

This artifact gets an extended treatment because of what happened after it
was fixed.

**Mechanism.** Out-of-sample R² (Campbell and Thompson, 2008) is
1 − SSE_model / SSE_benchmark, with the benchmark the expanding mean of all
returns observed before each forecast day. The benchmark is a forecast too,
and it has to be given its history. If the expanding mean is started from
nothing at the first day of a test period, its first values are means of
one and two observations. When that period opens on a volatile stretch,
those are forecasts of double-digit daily moves, the benchmark's squared
error inflates, and the ratio falls for every model identically. The
symptom is that a forecast of zero, which can only be about as good as the
historical mean, scores clearly better than it.

The specification was right. `REFACTOR_PLAN.md` §4.2 defines the benchmark
as "the expanding-window historical mean return computed only from data
before each prediction. Not the test-set mean." The implementation dropped
the training history: it built one expanding mean over the pooled test
series with no seed.

**Magnitude, first occurrence.** The bug was found while writing the
baseline tests in Phase 5. On the test fixture (1008 raw rows of AAPL,
2018 to 2021, whose test period opens on the March 2020 crash), the
zero-return baseline scored R²_OOS = **+0.087** with the cold benchmark and
**−0.005** with the benchmark seeded from each fold's training returns
(commit `db46cb3`; `tests/test_baselines.py`,
`test_an_unseeded_benchmark_inflates_r2_oos`, whose docstring records both
numbers). Every model would have been flattered by the same amount. The fix
assembled the pooled benchmark per fold, each fold seeded with its own
training returns (`src/evaluate.py`, `evaluate_predictions`,
`expanding_mean_benchmark`), and added three defences: the result carries a
boolean `unseeded_benchmark`; the report printer warns when it is true
("R2_OOS is biased upward; pass y_train_by_fold", `print_evaluation`); and a
regression test asserts that on the fixture the cold benchmark is more than
0.05 too optimistic and the seeded one is within 0.05 of zero.

**The recurrence.** The 30-ticker sweep persists every prediction to
`results/predictions/` and recomputes its aggregates from disk. The
aggregation function, `aggregate` in `src/experiments.py`, calls
`evaluate_predictions(predictions, model_name=..., validate=False)` and
passes no training returns, because a long parquet of predictions does not
carry them. The result dictionaries it received each had
`unseeded_benchmark = True`. Nothing on that path reads the flag: the
printer that carries the warning is not called during aggregation. The
sweep's entry point calls `write_aggregates` at the end of every run, which
calls `aggregate` and writes the cold numbers out. The bug therefore came
back, in a codebase built specifically to prevent it, by people who had
already found it once, with a fix, a flag, a warning and a regression test
in place.

The regression test did not catch it, and could not have. It tests
`evaluate_predictions` on the fixture with and without seeds, and that
function behaved exactly as tested. The new call site simply did not pass
the seeds, and no test asserted that every call site does. What caught it
was a sanity check on the output: the zero-return baseline scored +0.035 on
all thirty tickers, and a forecast of zero cannot beat the historical mean
by three and a half percent (`analysis/cross_ticker_sweep.py`, comment at
the training-returns rebuild; README §12, item 4).

**Magnitude, recurrence, recomputed from committed files.** The cold path
is still callable, so its output can be regenerated from the committed
predictions alone and set beside the committed seeded table:

| | Cold benchmark (`experiments.aggregate` on `results/predictions/`) | Seeded (`results/per_ticker_model.csv`) |
|---|---|---|
| Zero-return baseline, mean R²_OOS over 30 tickers | **+0.0349** | **−0.0006** |
| Tree ensemble, tickers with R²_OOS > 0 | **1 of 30** (JNJ, +0.0001) | **0 of 30** (JNJ: −0.0533; maximum −0.0318) |

The seeded numbers come from the analysis path, which rebuilds each
ticker's dataset from the raw cache to recover its training returns before
evaluating (`analysis/cross_ticker_sweep.py`). The cold path above is the
code as it stood when the recurrence was found; it can be checked out at
commit `a2995de` and run, which is how the table was produced.

**The fix was structural.** After the recurrence we stopped relying on
care. The training-returns argument of the evaluator is now required and
keyword-only, so omitting it is a `TypeError` at the call site before
anything runs; passing `None`, or a mapping that lacks any fold present in
the predictions, raises `UnseededBenchmark` before any metric is computed;
the `unseeded_benchmark` flag and its printed warning no longer exist,
because a flag that nothing reads is not a defence (`src/evaluate.py`,
`require_training_returns`). The aggregation path that recurred now takes a
required loader for the raw data and rebuilds every fold's training returns
from the run's own test windows (`src/experiments.py`, `aggregate`,
`training_returns_for_run`); the cold call is no longer expressible. The
regression test that missed the recurrence was replaced by one that asserts
the refusal in all three forms and reproduces the +0.087 by building the
cold benchmark by hand, so the size of what the refusal prevents stays on
record (`tests/test_baselines.py`, `test_an_unseeded_benchmark_is_refused`;
`tests/test_metrics.py`, `test_refuses_to_run_without_training_returns`).

**What this shows.** Care is demonstrably insufficient. A correct
specification, a fix by people who understood the failure, a flag on every
result, a printed warning and a passing regression test did not prevent a
recurrence three phases later, because the recurrence happened at a call
site none of those things covered. What worked was a baseline whose correct
score is known a priori: a forecast of zero must score about zero against
the historical mean, and when it scored +0.035 the number could not be
believed. Trivial baselines are not a courtesy to the reader; they are the
only check that operates on the output rather than on the code, and so the
only one that survives a new call site.

**Detection.**

- Run a zero forecast through the same R²_OOS code as the models. Its
  score against the historical mean must be near zero or slightly
  negative. A do-nothing forecast at +0.03 means the benchmark is wrong,
  whatever the tests say.
- Seed the benchmark from the training data of the fold, and make the seed
  argument impossible to omit at every call site: raise, rather than flag,
  when it is missing. This pipeline flagged, the flag was not enough, and
  it now raises.
- Print the benchmark's first few values for each fold. A one-observation
  mean of a crash-day return is visible to the eye.
- Recompute aggregates from persisted predictions, never from memory, so
  that a cold aggregate can be regenerated and compared, as in the table
  above (`REFACTOR_PLAN.md` §9; `src/experiments.py`, `load_runs`).

## 4.3 Mismatched evaluation windows across models

**Mechanism.** A stacked meta-learner that is fitted only on earlier folds'
out-of-fold predictions cannot forecast the earliest folds, so it covers
fewer days than its base models. Scoring each model over the days it
happens to cover compares different periods. If the days only the base
models cover are unusual, the comparison is biased, in either direction,
by however unusual they are. Nothing in the arithmetic flags it: every
model gets a number, and the table looks complete.

**Magnitude.** On the AAPL fixture with the nine-fold configuration of
§5.4, the meta-learner covered 420 of the 540 days the base models did. The
excluded folds ran from 2019-10-16 to 2020-04-07, the 2020 crash, with
3.22% daily realised volatility against 1.92% in the retained window and a
13.8% single-day move. The base models were being scored across the most
volatile stretch in the sample while the meta-learner never saw it.
Restricting every model to the common window cost the base models 1.9 to
2.3 percentage points of directional accuracy and narrowed their apparent
lead over the meta-learner from about 5 points to 2.3 to 3.5. The bias was
real and it favoured the base models (commit `710d90c`; README §5.1, §12
item 6). The same commit found a second instance in the economics: the
buy-and-hold benchmark had been computed over whichever model came first in
the dictionary, the base models' full range, so the meta-learner's rows were
compared with a buy-and-hold measured over two folds it was never allowed to
trade.

Every primary comparison now runs on the intersection of all models'
forecast days, including the meta's (`src/contracts.py`,
`common_evaluation_window`, which asserts the intersection is non-empty and
contiguous, and `restrict_all`). On the headline run that is folds 2 to 11,
2520 days (§5.4). The economics table refuses to compare strategies over
different day sets (`src/backtest.py`, `backtest_table`,
`require_identical_days`) and names the frame buy-and-hold is computed from
(`benchmark_predictions`). Models that cover more days are still reported
over their full range, in a separate table labelled secondary and never
mixed into the primary one (`data/model_comparison_full_range.csv`; README
§11).

**Detection.**

- Before any cross-model table, assert that the set of forecast days is
  identical across models. If it is not, restrict all models to the
  intersection and report what was excluded.
- Report the volatility of the excluded days beside the volatility of the
  retained ones. A gap the size of the one above is the artifact.
- Compute the market benchmark on exactly the days each strategy could
  trade. A benchmark measured over a range a strategy could not enter is a
  different benchmark.
- Keep the full-range numbers, labelled, in a table of their own. Deleting
  them hides the size of the effect; mixing them in reintroduces it.

## 4.4 Short evaluation windows

**Mechanism.** A test statistic computed on a short window of one ticker
has enough sampling variance that sign and significance both depend on the
window. At the 5% level a null effect reaches significance one time in
twenty by construction, and from inside a single window there is no way to
tell which time this is. The artifact is not the short window itself but
reporting its p-value as if it were a property of the method.

**Magnitude.** The pipeline asks whether adding the VIX level to the
meta-learner's inputs changes its forecasts, and answers with a
Diebold-Mariano test between the two variants' error series. The same test,
same code, same statistic, on two windows:

| Window | Days | DM statistic | p | Favours | Significant at 5% |
|---|---|---|---|---|---|
| Fixture, 2018 to 2021 | 420 | **+2.321** | **0.0203** | no VIX | yes |
| Full span, 2016 to 2026 | 2520 | **−0.595** | **0.5519** | with VIX | no |

(`analysis/alpha_correction_and_window.py`, the window-artifact table;
README §12, item 5.) The short window says VIX gating significantly hurts;
the long one says the two variants are indistinguishable and, if anything,
VIX helps. The signs are opposite. The project's own earlier reading of the
short-window result is on record: a commit interpreting the ridge grid
found that "adding VIX makes the meta want roughly ten times more
regularisation ... which agrees with the DM test favouring −VIX at p=0.02"
(commit `d26ad59`). That agreement was between two readings of the same 420
days, and it did not survive the full span.

**Detection.**

- Report the window length, in days, beside every p-value, and the number
  of tickers it covers.
- Re-run the test on the longest window available before reporting a sign.
  If the sign changes, report both and claim neither.
- Treat a single-ticker finding on fewer than a few years of daily data as
  a hypothesis for a longer window, not a result.
- Keep a separate short fixture for testing mechanism and refuse to report
  numbers from it. This pipeline's fixture exists for exactly that
  (`analysis/alpha_correction_and_window.py`: "the fixture is for testing
  mechanism, not for producing results").

## 4.5 Single-seed neural results

**Mechanism.** Neural training is a stochastic procedure: weight
initialisation, dropout masks and, with shuffling, batch order all depend
on the seed. A result at one seed is one draw from a distribution over
seeds, and a comparison between two architectures at one seed each is a
comparison between two draws. When the seed spread is as large as the gap
between architectures, the ranking is not identifiable from one seed, and
any single significant result among several seeds is what chance produces.

**Magnitude.** Five seeds each of the BiLSTM and the Transformer on AAPL
under the headline fold configuration (`results/seeds/`; 12 folds; run on
the sweep's fetch cache, see §5.4 for the grid offset). Two of the
project's apparent positive findings dissolve:

- **The Transformer's R²_OOS changes sign across seeds**, from −0.0106 to
  +0.0010 (README §3.3; commit `3fa7f8b`). The Transformer's +0.00102
  (`docs/baseline_after_refactor.csv`), the only positive pooled R²_OOS
  among the original ten models of Table 1 (the ridge comparator added
  later scores +0.0012, §6.1), is seed 42 landing on one side of a
  distribution that straddles zero.
- **BiLSTM seed 4 reports a Pesaran-Timmermann p of 0.0137** with
  directional accuracy 0.43 points above its majority class, the only
  neural configuration in this project to beat its majority class with a
  significant directional statistic (the linear comparators' directional
  hits are treated in §6.2). Its other four seeds give p = 0.329,
  0.521, 0.708 and 0.893; the five Transformer seeds give 0.201 to 0.696
  (recomputed from `results/seeds/` with `src/evaluate.py`; README §3.3).
  The seed study ran ten tests, two architectures by five seeds, so the
  expected number of hits at the 5% level is 0.5 and the probability of at
  least one is 1 − 0.95¹⁰ = 0.401. This is the expected chance hit.

The ratio of the architecture gap to the seed standard deviation is 1.66 on
R²_OOS and 1.69 on directional accuracy; under a null in which the two
architectures are identical, the 90% interval of that ratio at five seeds
is [0.67, 1.77], obtained by simulation, and both observed values fall
inside it (README §3.3). The seed variance and the architecture difference
are the same order of magnitude, and the paper accordingly withdraws any
ranking of one architecture over the other.

**Detection.**

- Never report a neural comparison at one seed. Report the range or the
  standard deviation across seeds beside every point estimate, as Table 1
  does for its BiLSTM and Transformer rows (README §3.1).
- Count the tests. With *k* seeds and *m* architectures there are *k·m*
  chances at a 5% hit; report the expected number and the probability of at
  least one.
- Do not rank architectures whose seed clouds overlap. State that the
  ordering is not identifiable at the number of seeds run.
- Say which results are single-seed. Here the thirty-ticker sweep is seed
  42 only, and the hybrid rows of Table 1 are marked single-seed because
  their spread was never measured (README §3.1, §8.3).

## 4.6 A detection checklist

Each item is a check on the output, not on the code, and each is one that
this pipeline either failed once or was built to enforce. The artifact it
targets is in brackets.

1. **A do-nothing forecast in every table, under the same metric.** A
   zero-return or last-value baseline that scores well means the metric is
   measuring the series or the benchmark is wrong. [4.1, 4.2]
2. **R²_OOS of the zero forecast near zero or negative.** If it is
   positive by more than sampling noise, the benchmark started cold. [4.2]
3. **Benchmark seed mandatory, not flagged.** A missing training history
   should raise at every call site. A flag that nothing reads is not a
   defence; this pipeline's flag was read by nothing on the path that
   recurred, and has been replaced by a required argument. [4.2]
4. **Aggregates recomputed from persisted predictions**, so a suspect
   number can be regenerated and compared, as in the cold-versus-seeded
   table above. [4.2]
5. **No feature correlated with the target above a stated bound**, and no
   level-valued feature without a named reason. [4.1]
6. **Identical forecast-day sets across models before any comparison**,
   asserted, with the excluded days' volatility reported. [4.3]
7. **The market benchmark computed on the days each strategy could
   trade.** [4.3]
8. **Window length and ticker count beside every p-value**, and every
   sign re-tested on the longest available window. [4.4]
9. **Seed ranges beside every neural point estimate**, the number of tests
   counted, and no ranking where the seed clouds overlap. [4.5]
10. **Every number traceable to a committed file.** Where this section
    could not trace one it says so; the only items it could not trace are
    marked **[unsourced]** in §5, and there are none in this section.
