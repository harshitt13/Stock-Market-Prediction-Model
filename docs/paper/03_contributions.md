# 3. Contributions

*Venue note (not part of the paper). Workshop formats such as ICAIF and the
NeurIPS/ICML finance workshops fold contributions into the introduction;
journals such as the International Journal of Forecasting keep a separate
list. §1 ends with a one-paragraph version of this list that mirrors its
numbering, so for a workshop submission this section is deleted and §1's
paragraph stands; for a journal submission both stay. Every number below is
established in §4 to §9 and sourced there.*

We make six contributions, in descending order of strength.

1. **A controlled before/after on identical data.** The same models,
   trained on the same frozen file (`docs/frozen_aapl_raw.csv`), the same
   twelve walk-forward folds and the same hyperparameters, scored R² = 0.9627
   on a price-level target under the conventional protocol and R²_OOS =
   −0.0003 on the return target under the corrected one (pooled; +0.0011 by
   fold median, −0.0022 ± 0.0109 by fold mean). Only the target and the
   evaluation changed. The naive zero-change forecast scored R² = 0.9992 on
   the price target and beat every model (§4.1, §6.1).

2. **Five protocol artifacts, each isolated and measured in our own
   pipeline.** A price-level target with the previous close as a feature; a
   cold-started out-of-sample benchmark (+0.087 against −0.005 on the
   fixture, +0.0349 against −0.0006 across thirty tickers, one ticker moved
   from apparently positive to none); evaluation windows that differed
   across models (1.9 to 2.3 points of directional accuracy); a short
   evaluation window (the same test at p = 0.0203 on 420 days and p = 0.552
   on 2520, with opposite signs); and single-seed neural results (the one
   significant directional result among five seeds, and an R²_OOS that
   changed sign across them). None of the five is leakage under Kapoor and
   Narayanan's taxonomy, and two, the benchmark construction and the
   alignment of comparisons, are to our knowledge uncharacterised (§4, §2.4).
   The cold benchmark recurred after it had been fixed, flagged and
   regression-tested, and was caught by a trivial baseline rather than by
   the test; the fix is now structural (§4.2).

3. **A cross-ticker null with correct multiple-testing treatment, and the
   weak positive directional association that survives none of it.** Across
   thirty US large caps the tree ensemble had positive R²_OOS on no ticker
   under pooled, fold-mean or fold-median aggregation, two directional hits
   against 1.5 expected, and eight alpha hits that were all negative;
   nothing survived Holm-Bonferroni. Two linear comparators added at the
   bottom of the capacity range scored within 0.018 of zero by fold median
   on every ticker and registered more directional hits than chance,
   significant before correction under the correlation we measured for the
   test statistics and not under the correlation a reader would assume;
   none survived correction across the ninety tests, eight of the eleven
   beat their majority class pooled and four did by fold median, and none
   produced a corrected alpha (§6.2, §6.3).

4. **A distribution-shift mechanism, tested rather than asserted, with its
   ceiling stated.** A third of the 357 fold-ticker pairs rejected a
   train-test Kolmogorov-Smirnov test at 5%, 6.7 times the null rate;
   per-fold R²_OOS fell with the KS statistic (t = −6.84); and the fit
   explained 11.6% of the variance in R²_OOS and 1.2% of the variance in
   directional accuracy. Shift explains the magnitude of the loss, not the
   absence of directional signal (§6.4). A related finding is that pooled
   statistics inherit their extreme folds: one fold boundary on the March
   2020 low moved pooled R²_OOS by 0.09, which is why the headline magnitude
   is the fold median (§5.7, §8.3).

5. **Seed variance that renders architecture comparisons unidentifiable.**
   Over five seeds each, the ratio of the BiLSTM-Transformer gap to the seed
   standard deviation was 1.66 on R²_OOS and 1.69 on directional accuracy,
   inside the 90% interval [0.67, 1.77] expected if the two architectures
   were identical; the Transformer's R²_OOS changed sign across seeds, and
   the study's one significant directional result was the expected chance
   hit from ten tests (§6.5).

6. **An open, tested artifact.** 341 tests, all offline; a truncation-
   invariance leakage check whose power is demonstrated by six injected
   leaks that it must catch; a frozen input with a published hash; every
   prediction behind every table committed, with the results tagged
   `v1.0-results` and the environment frozen in a lock file; and Table 1
   reproducible by one command with no network access (§5.3, §9).
