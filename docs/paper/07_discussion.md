# 7. Discussion

Every number in this section was read from a committed file in the
repository at commit `ead69ec` (results tagged `v1.0-results` at
`3fa7f8b`), or recomputed from committed predictions with the committed
evaluation code, and the source is named beside it. Anything not so
recorded is marked **[unsourced]**.

## 7.1 What the null result is and is not

We found that, under a leak-free walk-forward protocol with correctly seeded
benchmarks, three architectures and two linear comparators trained on
thirty-one scale-free technical and macro features did not forecast
next-day log returns on thirty US large caps: no ticker had positive
out-of-sample R² for the tree ensemble, while the two comparators scored
within 0.017 of zero on every ticker by fold median, positive on about
half, which is what a forecast of the unconditional mean scores (§6.2); no
model beat the majority class on the headline ticker (§6.1),
nothing survived multiple-testing correction, and the two neural
architectures were not distinguishable from each other across seeds (§6.5).
We also found that the performance the previous version of this same
pipeline had reported, an R² of 0.9627 on the price target, was traceable
to specific evaluation choices, each of which we reproduced and measured
(§4).

We are careful about what that supports. It does not show that markets are
efficient, that equity returns are unpredictable, or that deep learning
cannot work in finance (README §2). We tested one horizon, one asset class,
one country, one feature family, three architectures at one configuration
each, and one seed for most runs. The claim is narrow: with these features,
this horizon, this universe and this protocol, the models did not beat
trivial baselines, and the protocol was strict enough that had they done so
the result would have been credible.

## 7.2 The detection checklist

The five artifacts of §4 share a signature: each produced a plausible
number rather than a crash, and each was exposed by a baseline whose
correct score was known in advance. We collected the checks in §4.6 and
restate the principle here because it is the paper's most transferable
result. A check on the output survives a new call site; a check on the
code does not. The cold-start benchmark of §4.2 recurred in a codebase that
had a correct specification, a fix, a flag on every result, a printed
warning and a passing regression test, because the recurrence happened at a
call site none of those things covered (`src/experiments.py`, `aggregate`).
What caught it was a forecast of zero scoring +0.035 against the historical
mean on all thirty tickers, which it cannot honestly do. We recommend that
every table in this literature carry a do-nothing row under the same metric
as the models, that the out-of-sample R² of that row be reported, and that
a benchmark seed be mandatory at every call site rather than flagged when
absent.

## 7.3 Why the architectures converged to the unconditional mean

The three model families were fitted by minimising a squared-error-type loss
(Huber for the neural models, squared error for the trees and the ridge
meta-learner; §5.5) on a target that is mostly noise. The population
minimiser of such a loss is the conditional mean of the return given the
features. Where the features carry little conditional information, that
conditional mean is close to the unconditional one, and where the training
distribution differs from the test distribution, as it did on a third of
folds (§6.4), even a conditional mean that was learned collapses towards a
constant when evaluated out of sample. A model that has converged to the
unconditional mean forecasts a near-constant, slightly positive number on
every day, is long almost every day under a long/flat rule, and inherits
the market's Sharpe ratio without having forecast anything.

We observed this independently in all three families, and we can quantify
each from committed files.

- **The sequence models emitted near-constants.** On the headline run,
  the range of the BiLSTM's predictions within each test fold was between
  0.5% and 6.6% of the range of its training targets, and the Transformer's
  between 0.6% and 8.2% (recomputed for all twelve folds from
  `results/headline/AAPL__frozen__seed42.parquet` and
  `docs/frozen_aapl_raw.csv`). Their prediction standard deviations were 9
  and 10 basis points against 182 realised (`docs/figures/aapl_prediction_dispersion.png`).
- **The tree's predictions were compressed to a fraction of its training
  range.** The same ratio for the tree ensemble was between 5.7% and 19.7%
  across the twelve folds, mean 12.0% (same recomputation). Trees cannot
  extrapolate beyond their leaves, and with heavy regularisation the leaves
  themselves average over wide neighbourhoods of a noisy target.
- **The meta-learner regularised itself towards its intercept.** The only
  tuned component in the pipeline, the ridge penalty chosen by leave-one-out
  cross-validation over a grid from 10⁻³ to 10⁶, never selected a value
  below 10³ in any fold, and selected the grid's ceiling of 10⁶ in four of
  the twenty fold-fits (`results/AAPL__frozen__seed42_meta_fits.csv`). Its
  prediction spread was 4 to 5% of realised (README §4.2). Given the choice,
  the meta-learner's own cross-validation preferred to ignore its inputs.
- **A standalone ridge regression on the same features collapsed the same
  way, and so did a logistic classifier.** We added two well-regularised
  linear comparators at the bottom of the capacity range (§5.5). On the
  headline run the ridge chose the grid's ceiling of 10⁶ in 6 of its
  twelve folds, where its prediction spread was near zero relative to the
  training returns and the sum of its absolute coefficients was about
  10⁻⁵ against an intercept of about 10⁻³; in the remaining folds it chose
  10³ to 10⁴ and spread 4 to 12% of the training spread
  (`results/linear_fits_headline.csv`). Across the thirty sweep tickers,
  357 fits, it never chose a penalty below 10³, chose the ceiling in
  95, and its median prediction spread was 4.2% of the training spread
  (`results/linear_fits_sweep.csv`). The logistic direction classifier chose
  the smallest inverse penalty on its grid, 10⁻⁴, in 207 of 357 sweep fits
  and 10 of 12 headline fits, and its fitted probabilities had a standard
  deviation of 0.017 around the base rate of about 0.52: a near-constant
  vote for "up". A model with almost no capacity to misbuild, given the
  choice by its own cross-validation, chose to predict the mean.

These are four independent mechanisms pointing at the same conclusion:
there was no learnable conditional signal in these features at this
horizon, rather than an optimisation failure that a better optimiser would
have repaired; the linear comparators have no optimiser to fail, and they
reached the same place. We add the symmetric point that the burden of
proof is not ours alone. Papers claiming success at this horizon rarely
report a zero-return baseline at all, and §4 shows what happens to a result
when one is added.

## 7.4 The economic result, read correctly

The eight significant alphas in the sweep were all negative, and the
mechanism was the tree's underexposure to a rising market: a mean predicted
return of −1.48 basis points against +6.00 realised, out of the market on
about 47% of days, at beta 0.44 to 0.65 (§6.3). We read this as a
demonstration that the economic test had power, not as evidence that the
models were harmful; a strategy that was flat on random days would show the
same shortfall. The one Sharpe ratio above buy-and-hold, the tree's 1.166
against 1.045 on AAPL, dissolved under the four checks of §6.3 and was not
a like-for-like comparison in the first place, since it set a beta-0.66,
61%-exposed position against a fully exposed one.

## 7.5 What would change the conclusion

The result is bounded by its design, and we can say what would move it.

- **Longer horizons.** The literature that finds return predictability
  finds it at monthly to annual horizons, where the signal-to-noise ratio of
  the target is higher. Nothing here speaks to those horizons.
- **Cross-sectional rather than time-series prediction.** Ranking many
  assets at one date is a different problem from forecasting one asset
  through time, and the evidence for the former is stronger. We tested only
  the latter.
- **Different information.** Our features were price, volume and three
  macro series. Valuation ratios, order flow, earnings revisions, text and
  options-implied measures were absent, and any of them could carry
  conditional information these features do not.
- **Intraday data**, where microstructure effects give short-horizon
  predictability that daily closes average away.
- **More architectures, tuned.** We ran three families at one configuration
  each with the hyperparameter search disabled (§5.5). A tuned model might
  do better; we did not test that, and we note that tuning against the test
  fold is itself one of the artifacts we documented.
- **Other markets.** Thirty US large caps, all current constituents, are the
  most efficiently priced securities in the world, and survivorship biases
  our aggregates upward (§8).
- **Refitting more often.** Our models forecast for up to a year without
  seeing new data (§5.4). A monthly or daily refit would remove a
  disadvantage that our own distribution-shift analysis suggests is real.

A positive result under any of these changes would not contradict this
paper. It would be a different experiment, and we would ask of it only what
we asked of ours: the row for doing nothing, the seeded benchmark, the
common window, the window length beside the p-value, and the seed range
beside the point estimate.
