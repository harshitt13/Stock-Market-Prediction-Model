# Abstract

The same models, trained on the same frozen data with the same walk-forward
folds and hyperparameters, scored R² = 0.9627 on a next-day price target
under a conventional evaluation protocol and R²_OOS = −0.0003 on the
next-day return under a leak-free one; a naive zero-change forecast scored
0.9992 on the former. We isolated and measured five evaluation choices that
produce such numbers without a forecast behind them; none is leakage under
Kapoor and Narayanan's taxonomy. Across thirty US equities the tree ensemble
had positive R²_OOS on no ticker under any aggregation and eight
significant alphas that were all negative; nothing survived Holm-Bonferroni
correction. Two linear comparators scored within 0.018 of zero on every
ticker and showed a weak positive directional association on a few,
significant before correction under the measured null, surviving no
multiple-testing correction, converting to directional accuracy above the
majority class on four tickers by fold median and to a corrected alpha on
none. Train-test distribution shift, present in a third of folds, explained
11.6% of the variance in R²_OOS and 1.2% in directional accuracy. Across
five seeds, architecture differences were the size of seed noise. Code,
frozen data, 341 tests and every prediction are released.
