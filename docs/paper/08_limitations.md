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

**Results are sensitive to that grid and to the data vintage.** The same
tree ensemble on AAPL, same code, same seed, scored R²_OOS of −0.025 over
all twelve folds on the frozen grid (recomputed from
`results/headline/AAPL__frozen__seed42.parquet` and `docs/frozen_aapl_raw.csv`)
and −0.121 on the sweep grid (`results/per_ticker_model.csv`), with
directional accuracy minus majority of −1.14 against −2.31 points. The two
runs differ by a 49-row shift in fold boundaries and by a refetch. We do
not know how much of the difference is attributable to each, and we did
not run the experiment that would separate them. A reader should take the
magnitude of any single R²_OOS in this paper as uncertain by at least that
much, which is another reason the paper's claims rest on signs, counts and
corrected p-values rather than on point values.

**Two artifact magnitudes were measured on the fixture.** The mismatched
window (§4.3) and short window (§4.4) effects were measured on the
1008-row, nine-fold test fixture, not on the headline configuration
(`tests/fixtures/aapl_raw.csv`; commit `710d90c`;
`analysis/alpha_correction_and_window.py`). The fixture exists to
demonstrate mechanism, and we report those magnitudes as such.

**The cold aggregation path is still present.** `experiments.aggregate`
evaluates without training returns and its output is unseeded (§4.2). No
number in this paper is taken from it, but a reader running
`python src/experiments.py --aggregate-only` will obtain cold numbers. We
left it unchanged rather than alter evaluation code during writing, and we
record it here.

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
paper.** Holm-Bonferroni was applied across the nine strategies on AAPL and
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
