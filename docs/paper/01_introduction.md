# 1. Introduction

Of the ten primary studies of deep learning for equity prediction that we
survey in §2, nine forecast a price level rather than a return, none
reports a naive or persistence baseline at the level we could examine, and
none reports a significance test. The reported numbers are correspondingly
high: coefficients of determination of 0.98 on next-day closing prices,
mean absolute percentage errors of two to three percent. Our own earlier
pipeline was one of these. Predicting Apple's next-day close from a feature
set that included today's close, it reported R² = 0.9627, and described
itself as achieving state-of-the-art precision.

The gap in that protocol is not that the models were leaky in the sense
usually meant. Today's close is a legitimate input at forecast time. The gap
is that the protocol had no row that said what doing nothing scores. When we
added one, the naive "tomorrow equals today" forecast scored R² = 0.9992 and
a mean absolute percentage error of 1.23% on the same data, beating every
model on both metrics. R² on a trending price series measures the price
series, and a persistence forecast is the identity function computed
exactly. The number was real arithmetic on a target that could not
discriminate a forecaster from a copy.

We rebuilt the pipeline around a return target and a leak-free walk-forward
protocol, and then did two things the original protocol could not. First,
we ran the same models on the same frozen file, the same twelve folds and
the same hyperparameters under both protocols, so that the difference
between what each reported is attributable to the evaluation alone: the
best model's R² of 0.9627 became an out-of-sample R² of −0.0003 against the
historical mean (§4.1, §6.1). Second, we took the protocol to thirty US
large caps with the multiple-testing treatment that a thirty-ticker table
requires, added two well-regularised linear comparators so that the models
span the capacity range from ridge regression to a Transformer, reran the
neural architectures across five seeds, and tested the mechanism we
propose for the failure rather than asserting it (§5, §6).

What we found is a null result with a positive mechanism and one nuance.
The tree ensemble had positive out-of-sample R² on none of the thirty
tickers under any aggregation, two directional hits against 1.5 expected by
chance, and eight significant alphas that were all negative, the
signature of a model underexposed to a rising market rather than of skill;
nothing survived Holm-Bonferroni correction (§6.2, §6.3). The nuance came
from the linear comparators. Their out-of-sample R² sat within 0.018 of
zero on every ticker, positive on about half, which is what a forecast of
the unconditional mean scores; but they registered more directional hits
at the uncorrected 5% level than a binomial count expects, eleven of the
eighty-six defined tests across the three models against 4.3. That excess
was a positive association in every case, significant before correction
under the correlation we measured for the test statistics themselves and
not under the higher correlation a reader might assume for the returns; it
survived no multiple-testing correction across the ninety tests, it
converted to directional accuracy above the majority class on four of the
eleven tickers by fold median, and it converted to a corrected alpha on
none (§6.2). We report it as a weak, non-robust directional tilt, and we
report both nulls. A third of walk-forward folds trained on a return
distribution that differed detectably from the one they were scored on,
and that shift explained 11.6% of the variance in per-fold R²_OOS and 1.2%
of the variance in directional accuracy: the magnitude of the loss, not the
absence of signal (§6.4). Across five seeds, the difference between the two
neural architectures was the same size as the seed noise, so no ranking
between them is identifiable at one seed (§6.5). And in every model family
we observed the same convergence to the unconditional mean, from sequence
models whose test-fold predictions spanned under a tenth of their training
range to a ridge regression whose own cross-validation chose the largest
penalty on its grid (§7.3).

Along the way we isolated five evaluation choices that each produced a
plausible number without a forecast behind it, measured each in our own
pipeline, and found that none of them is leakage under the taxonomy of
Kapoor and Narayanan (2023); two, the construction of the benchmark and the
alignment of comparisons across models, are to our knowledge
uncharacterised (§4, §2.4). One of them, the cold-started benchmark,
recurred after we had found it, fixed it, flagged it and written a
regression test for it, and was caught the second time by a trivial
baseline scoring what it could not honestly score. We take that as the
paper's most transferable lesson: a check on the output survives a new call
site, and a check on the code does not (§4.2, §7.2).

We make six contributions, listed in §3 and summarised here: a controlled
before/after on identical data; five isolated and measured protocol
artifacts, none of them leakage and two uncharacterised; a thirty-ticker
null with correct multiple-testing treatment together with the weak
positive association that survives none of it; a distribution-shift
mechanism tested rather than asserted, with its ceiling stated; seed
variance that renders architecture comparisons unidentifiable; and an
open, tested artifact with 341 tests, mutation-tested leakage detection,
frozen data and a tagged results commit.

We are explicit about what we do not claim. We do not claim that markets
are efficient, that equity returns are unpredictable, or that deep learning
cannot work in finance. We tested three architectures and two linear
comparators, at one configuration each, on one horizon (the next trading
day), in one market (US large caps), for the most part at one seed; the
thirty-ticker evaluation covers the tree ensemble, the five baselines and
the two comparators, while the two neural architectures and the
meta-learner ran on five tickers. A large literature finds predictability
at longer horizons, in the cross-section and from information we did not
use, and nothing here speaks to it. Our claim is narrow: with these
features, this horizon, this universe and this protocol, the models did
not beat trivial baselines, and the protocol was strict enough that had
they done so the result would have been credible.

The rest of the paper is organised as follows. §2 reviews the three
literatures we draw on and positions this work against the closest prior
study. §3 lists the contributions. §4 treats the five protocol artifacts,
each with its mechanism, its measured magnitude and how to detect it, and
closes with a checklist. §5 gives the data, the row contract, the features,
the walk-forward design, the models, the baselines and the metrics, and
states two limitations in place. §6 reports the results. §7 discusses what
the null does and does not support and why the architectures converged.
§8 states the limitations at length. §9 is the reproducibility statement.
