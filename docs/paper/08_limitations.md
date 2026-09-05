# 8. Limitations

We state these at length because the credibility of a null result rests on
how completely its boundaries are drawn. Every number below was read from
a committed file in the repository at commit `ead69ec` (results tagged
`v1.0-results` at `3fa7f8b`), or from the message of a commit in its
history, and the source is named beside it. Anything not so recorded is
marked **[unsourced]**.

## 8.1 Scope of the question

**One horizon.** We forecast the next trading day's log return and nothing
else (`REFACTOR_PLAN.md` §1; §5.2). The daily horizon has the lowest
signal-to-noise ratio of any at which return predictability has been
claimed, and our result says nothing about weekly, monthly or annual
horizons, where a substantial literature finds predictability from
variables we did not use.

**One market, one asset class, one country.** Thirty US large-cap equities
(`src/experiments.py`, `DEFAULT_TICKERS`). No other geography, no other
asset class, no small caps, no cross-section. These are among the most
heavily traded securities in the world, and a null at the daily horizon on
them is the least surprising null available.

**One feature family.** Thirty-one features derived from price, volume and
three macro series (§5.3). No fundamentals, no order flow, no earnings
data, no text, no options-implied quantities. One feature, `Corr_SP500_20`,
is a rolling correlation of price levels rather than of returns, as
implemented and reported (§5.3). The absence of predictability in these
features is not evidence about features we did not use.

**Three architectures, one configuration each, is not "deep learning".** A
tree ensemble, a bidirectional LSTM and an encoder-only Transformer, each
at a single hyperparameter setting (§5.5). The transformer's hyperparameter
search exists in the code and was disabled for every reported run
(`src/main.py`, `--optimize`; README §8.3). The neural models used a 90-day
lookback that we did not vary. A negative result at this scale is evidence
about these runs, not about a model class.

## 8.2 Data

**Survivorship bias.** All thirty tickers trade today. Companies delisted,
acquired or bankrupted between 2010 and 2026 are absent from the universe,
so every cross-sectional aggregate in §6.2 and §6.3 is biased upward: the
surviving universe outperformed the universe that existed at the time. The
sweep prints this warning on every run (`src/experiments.py`,
`SURVIVORSHIP_WARNING`). Correcting it requires point-in-time index
constituents including delisted names, which we did not have. We note that
the direction of this bias is toward finding profitable strategies, and we
found none.

**One data vendor, one vintage.** Prices came from Yahoo Finance through
the `yfinance` client's `Ticker.history` method with library defaults
(`src/fetch_data.py`, `fetch_stock_data`). **[unsourced]**: the code does
not set the price-adjustment option, so whether closes are adjusted for
dividends as well as splits is a default of the library version recorded
in `requirements-lock.txt`, and we did not verify it against the vendor's
documentation. Adjusted histories are revised after the fact: the frozen
AAPL file differed by 342 bytes from a refetch made days later (commit
`d26ad59`). The headline result is therefore reproducible from the
committed file but not from a fresh download.

**The sweep's raw cache is not committed.** `results/raw/` is gitignored
(README §9). The thirty-ticker predictions are committed, but the per-ticker
raw spans and the fetch date are not recoverable from the repository
(§5.1), and any analysis that needs training returns for the sweep grid
(the KS regression of §6.4, the seed figure's benchmark) needs the cache
regenerated, which fetches a new vintage.

**Macro series were forward-filled.** Days on which the VIX or the yield
was missing carry the previous value (`src/fetch_data.py`,
`engineer_features`). This is the leak-free choice, but it means a stale
macro value on those days.

## 8.3 Evaluation design

**Annual refit.** Every model was refitted once per fold, and a fold is
252 trading days, so a model forecast for up to twelve months without
seeing new data; one trading in May 2026 was last fitted on data ending in
June 2025 (§5.4, fold table). This was a compute concession: a full run of
one ticker took about a hundred minutes under this configuration, and the
default quarterly configuration would have taken 11 to 45 hours per run
(README §8.3, §9). Real deployment would refit monthly or daily. The
direction of the bias is against the models, and our own analysis suggests
it is material: a third of folds trained on a return distribution that
differed detectably from the one they were scored on (§6.4), and a staler
model suffers more from that. Part of the poor performance we report may be
staleness rather than absence of signal.

**Two folds are never scored in the primary tables.** The common window
drops folds 0 and 1, 504 forecast days from 2014-05-29 to 2016-05-26, from
every primary comparison, because the meta-learner cannot forecast them
(§5.4). We chose the common window over the alternative of scoring models
on different days (§4.3), but the cost is that a sixth of the walk-forward
period is reported only in the secondary full-range table.

**Single seed for the sweep.** The thirty-ticker results are seed 42 only.
Only AAPL has a five-seed study, and only for the two neural architectures
(`results/seeds/`). Table 1's hybrid rows are single-seed with an
unmeasured spread, since the meta-learner inherits its inputs' seed
variance and adds its own (§6.1, note b). The seed study itself is five
seeds, and we treat its ratio as an order-of-magnitude comparison with the
interval reported (§6.5), not as a precise estimate.

**The 5-versus-30 split.** The thirty-ticker evaluation covers the tree
ensemble and the five baselines only; the neural architectures and the
meta-learner ran on AAPL, JNJ, JPM, WMT and XOM (§5.1;
`results/predictions/`). Every claim of the form "across thirty tickers"
is a claim about the tree. The neural null rests on five tickers and, for
seed variance, on one. **[unsourced]**: the reason for the restriction and
the basis for choosing those five are not recorded in any committed
artifact; the runtime figures above are context, not a cited decision.

**The seed-variance figure sits on a different fold grid.** The seed study
ran on the sweep's AAPL fetch cache, not the frozen file, and the two have
different warm-ups, so its fold grid opens 49 trading days earlier than
Table 1's: 2014-03-19 against 2014-05-29. The figure states this in its
title, and the offset is computed from committed files and asserted in the
test suite (`analysis/make_figures.py`, `grid_offset`;
`tests/test_headline_provenance.py`). The seed ranges annotated on Table 1
therefore describe a run whose fold boundaries are seven weeks away from
the one they annotate. We did not rerun the five seeds on the frozen grid;
that would cost five further hour-long runs, and the offset is stated rather
than hidden.

**Pooled R²_OOS is sensitive to where the fold boundaries fall, and the
sensitivity is concentrated in one fold.** The same tree ensemble on AAPL,
same code, same seed, scored pooled R²_OOS of −0.025 over all twelve folds
on the frozen grid and −0.121 on the sweep grid. We separated the two
things that differ between those runs, the data file and the fold grid, by
running the tree four ways (`analysis/grid_vs_data.py`;
`results/grid_vs_data.csv`):

| Run | Data | Fold grid (fold 0 opens) | Pooled | Fold mean | Fold median |
|---|---|---|---|---|---|
| A | frozen CSV | frozen (2014-05-29) | −0.0251 | −0.0204 | −0.0087 |
| B | frozen CSV | sweep (2014-03-19) | −0.1109 | −0.0749 | −0.0376 |
| C | sweep cache | frozen (2014-05-29) | −0.0244 | −0.0200 | −0.0226 |
| D | sweep cache | sweep (2014-03-19) | −0.1208 | −0.0788 | −0.0401 |

A and D reproduced the committed runs to 7e-18 and
1e-17. The grid moved the pooled value by 0.086 (A to B) and
0.096 (C to D); the data moved it by 0.0007 (A to C) and 0.010 (B to
D). The data effect stayed under 0.01 on the pooled value and the fold mean,
and reached 0.014 on the fold median, which reorders small values
easily. The data difference itself is floating-point noise: on the 4,144
shared dates the two files' closes differ by at most 7.6e-05 in absolute
terms and 9.4e-07 relatively, on 2922 rows beyond 10⁻⁶ absolute and
none beyond 10⁻⁶ relative; volume and the three macro series are identical
(`results/grid_vs_data_rawdiff.csv`). A fifth run, the sweep cache with its
49 extra early rows removed on the frozen grid, scored −0.0269 pooled, so
neither the vendor's revisions nor the extra history matters.

What matters is one fold. On the sweep grid, fold 6's training data ends on
2020-03-19, within days of the March 2020 low, and its test window opens the
next day on the rebound; that fold scored −0.554 on the frozen data (B) against
−0.030 for the frozen grid's fold 6 (A), whose boundary falls on 2020-05-28
after the recovery had begun. Fold 6 carried 80% of the summed per-fold
gap between A and B. Because pooled R²_OOS is a ratio of summed squared
errors, the highest-variance year dominates it, and a 49-day shift in where
that year is cut moves the whole number (§5.7).

To turn the two-point comparison into an error bar we ran the tree on the
frozen CSV with every fold boundary moved earlier by 0, 12, 25, 37, 49 and
63 trading days, twelve folds each time (`analysis/grid_offset_sweep.py`;
`results/grid_offset_sweep.csv`):

| Days earlier | Training rows, fold 0 | Fold 0 opens | Pooled | Fold mean ± sd | Fold median | Worst fold |
|---|---|---|---|---|---|---|
|   0 | 1008 | 2014-05-29 | −0.0251 | −0.0204 ± 0.036 | −0.0087 | 4 |
|  12 | 996 | 2014-05-12 | −0.0255 | −0.0201 ± 0.035 | −0.0131 | 4 |
|  25 | 983 | 2014-04-23 | −0.0270 | −0.0296 ± 0.043 | −0.0332 | 4 |
|  37 | 971 | 2014-04-04 | −0.0378 | −0.0355 ± 0.049 | −0.0360 | 6 |
|  49 | 959 | 2014-03-19 | −0.1109 | −0.0749 ± 0.154 | −0.0376 | 6 |
|  63 | 945 | 2014-02-27 | −0.0214 | −0.0305 ± 0.042 | −0.0308 | 5 |

Over the six offsets the pooled value ranged over 0.089, the fold mean over
0.055 and the fold median over 0.029. The 49-day offset is a
spike, not a trend: at 37 and 63 days the pooled value was −0.038 and
−0.021, and the worst fold moved back off the crash. Four later offsets
(12 to 49 days) gave pooled values between −0.025 and −0.023 (same
file). This is why we report the fold median as the headline magnitude
(§5.7, §6.1): a reader should take any pooled R²_OOS in this paper as
uncertain by about ±0.05 to the choice of fold boundaries, the fold mean by
about ±0.03 and the fold median by about ±0.015, and should rest on signs,
counts and corrected p-values rather than on point values. This is one
instance of a recurring finding: pooled statistics inherit their extreme
folds. The aggregation table of §5.7 and Table 1 is the second (the headline
magnitude was switched to the fold median for it), and the one directional
hit that survives a within-model Holm correction in §6.2 is the third, with
a pooled edge of +2.1 points over a fold median of −0.7.

**The cross-ticker pooled magnitude is grid-conditional; the null is not.**
Re-running the tree on the five tickers that carried every model, each on
its own sweep cache under the frozen grid as well as its sweep grid
(`analysis/grid_conditional_tickers.py`; `results/grid_conditional_tickers.csv`):

| Ticker | Pooled, sweep grid | Pooled, frozen grid | Median, sweep grid | Median, frozen grid | Worst fold (sweep / frozen) |
|---|---|---|---|---|---|
| AAPL | −0.1208 | −0.0244 | −0.0401 | −0.0226 | 6 / 4 |
| JNJ | −0.0533 | −0.0240 | −0.0360 | −0.0330 | 6 / 7 |
| JPM | −0.2018 | −0.0335 | −0.0566 | −0.0327 | 6 / 7 |
| WMT | −0.0318 | −0.0310 | −0.0196 | −0.0270 | 1 / 8 |
| XOM | −0.2042 | −0.0471 | −0.0140 | −0.0339 | 6 / 6 |

The five-ticker mean pooled R²_OOS was −0.122 on the sweep grid and −0.032 on
the frozen grid; the mean fold median was −0.033 and −0.030. The pooled
mean of Table 2 is therefore a property of the sweep grid, and §6.2 states it
as such. No ticker had positive R²_OOS under either grid by either
aggregation (0 of 5 and 0 of 5 pooled; 0 of 5 and 0 of 5 by median), so the
0-of-30 null does not depend on the grid. On the sweep grid the worst fold
was fold 6 on 25 of the 30 tickers (§6.2).

**Two artifact magnitudes were measured on the fixture.** The mismatched
window (§4.3) and short window (§4.4) effects were measured on the
1008-row, nine-fold test fixture, not on the headline configuration
(`tests/fixtures/aapl_raw.csv`; commit `710d90c`;
`analysis/alpha_correction_and_window.py`). The fixture exists to
demonstrate mechanism, and we report those magnitudes as such.

**The cold aggregation path existed until commit `a2995de`.**
`experiments.aggregate` evaluated without training returns, and its output
was unseeded (§4.2). No number in this paper was taken from it. It has
since been closed structurally: the evaluator refuses to run without the
seeds and the aggregator rebuilds them from each run's own windows. We
record here that the closure came after the results were produced, so the
committed sweep aggregates were computed by the analysis path, not by the
repaired aggregator; the two agree because both seed the benchmark the
same way.

**The ridge grid may be censored at the top.** The meta-learner's
cross-validation selected the grid's ceiling of 10⁶ in four of twenty
fold-fits (`results/AAPL__frozen__seed42_meta_fits.csv`). The grid had
already been widened once for the same symptom (commit `d26ad59`). We did
not widen it again, and we read the ceiling hits as the meta-learner asking
to ignore its inputs rather than as a tuning problem; a reader may
disagree.

## 8.4 Statistical procedures

**Alpha standard errors are plain OLS.** The t-statistic on alpha uses
ordinary least-squares standard errors, not a heteroskedasticity- and
autocorrelation-consistent estimator, and the code says it should be read
as indicative (`src/backtest.py`, `market_adjusted`). Daily strategy
residuals carry little serial correlation, but we did not test that
assumption.

**The Diebold-Mariano test at a one-step horizon uses no autocorrelation
correction.** The Newey-West truncation lag is h − 1, which is zero here,
so the statistic is the mean loss differential over its plain standard
error (`src/evaluate.py`, `diebold_mariano`).

**Multiple-testing correction was applied within families, not across the
paper.** Holm-Bonferroni was applied across the eleven strategies on AAPL and
across the thirty tickers per test (§6.2, §6.3). It was not applied across
the union of every test this paper reports. The seed study's ten tests
were counted (§6.5), but not folded into the others.

**The seed-ratio null was simulated under normality.** The 90% interval of
[0.67, 1.77] assumed two iid normal samples of five (README §3.3). The true
seed distribution of a neural model's R²_OOS need not be normal.

**The distribution-shift diagnostic is marginal.** The KS test compares the
marginal distributions of training and test returns and ignores the
features (`src/experiments.py`, `fold_return_diagnostics`). It explained
11.6% of the variance in per-fold R²_OOS and 1.2% in directional accuracy
(§6.4); we do not claim it as the mechanism for the absence of directional
signal. **[unsourced]**: the Mann-Whitney p of 0.0003 for shifted against
unshifted folds appears in the outline and is computed by the analysis
script, but its output is not persisted and the series cannot be rebuilt
from committed files, so we did not cite it.

**Directional accuracy excludes near-flat days.** Days with a realised
return within 10 basis points of zero were excluded, 7.0% of the AAPL
common window (`docs/baseline_after_refactor.csv`). The excluded fraction
was identical across models, so comparisons within a table are fair, but
the reported accuracies are on that denominator and not on all days.

## 8.5 Economics

**A flat cost, no impact, no borrow.** Transaction cost was 7.5 basis
points per round trip and nothing else (`src/backtest.py`,
`DEFAULT_COST_BPS`; README §8.3). There was no market-impact model and no
borrowing cost, the latter moot because every reported strategy was
long/flat. The stress grid varied cost only between 0 and 10 basis points.

**Long/flat only.** A long/short rule is implemented but was not the
reported configuration (README §8.3). Every economic result is for a
strategy that can only hold or not hold the asset.

**Execution at the close, same day.** The base case assumes a position
taken at the close on which the forecast is made. A one-day lag reversed
the sign of the tree's alpha (§6.3), so the economics are sensitive to
an assumption we could not test beyond that one step.

## 8.6 The before/after comparison

The "before" table (`docs/baseline_before_refactor.csv`) was produced by
the previous pipeline on a price target with a different feature set,
including level-valued features, and its directional accuracy was computed
by a definition that invented an observation at every fold boundary (§4.1;
README §12). The before/after comparison of §4.1 is therefore a comparison
of two protocols on identical data, not a controlled ablation of one
factor. We could not isolate the effect of the target alone from the effect
of the features alone without a run that was never made.

## 8.7 Reproducibility limits

**The random forest is reproducible only to about 10⁻¹⁶.** Its prediction
is averaged over trees in parallel, and thread scheduling changes the
summation order; two runs of the same seed on the same data differed by at
most 1.0 × 10⁻¹⁶ in one prediction, with every downstream table
byte-identical (commit `148f064`). Bit-for-bit reproduction of the
prediction file is therefore not guaranteed; reproduction of every reported
digit is.

**Timing figures are one machine.** Runtimes are CPU-only on a single
machine and are given to the nearest ten minutes (README §9); the
environment is recorded in `requirements-lock.txt`.

**The test suite verifies the contract, not the models.** The 331 tests
(`README` §7; `python -m pytest --collect-only`) check leakage, the row
contract, the metrics and the provenance of the committed artifacts. They
do not and cannot check that a model is well specified.
