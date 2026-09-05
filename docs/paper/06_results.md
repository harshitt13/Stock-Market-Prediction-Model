# 6. Results

Every number in this section was read from a committed file in the
repository at commit `ead69ec` (results tagged `v1.0-results` at
`3fa7f8b`) or recomputed from committed predictions with the committed
evaluation code, and the source is named beside it. Where a number could not
be traced we marked it **[unsourced]** rather than filling it in.

## 6.1 The before/after on identical data

Table 1 reports the headline configuration of §5.4: AAPL on the frozen file
`docs/frozen_aapl_raw.csv`, twelve expanding folds, and every model scored
on the common evaluation window of 2520 forecast days from 2016-05-27 to
2026-06-05 (`docs/baseline_after_refactor.csv`; README §3.1). Directional
accuracy excludes the 7.0% of days on which the realised return was within
10 basis points of zero, identically for every row. For the magnitude we
report three aggregations of the same per-fold R²_OOS series. The pooled
value is a ratio of summed squared errors and is dominated by the fold with
the largest errors (§5.7); the across-fold median is our headline; the mean
and its standard deviation show the spread across folds.

**Table 1.** AAPL, return target, common window, seed 42, including the two
linear comparators of §5.5. The headline
magnitude is the across-fold median of R²_OOS over the ten common-window
folds; the across-fold mean and standard deviation and the pooled value
are beside it (`results/fold_aggregates_headline.csv`; §5.7 on why).

| Model | DA (%) | Majority (%) | DA − majority (pp) | PT p | R²_OOS, fold median | R²_OOS, fold mean ± sd | R²_OOS, pooled | RMSE (bps) |
|---|---|---|---|---|---|---|---|---|
| Tree Ensemble | 52.82 | 54.01 | −1.19 | 0.0586 | **−0.0081** | −0.0152 ± 0.0326 | −0.02037 | 183.4 |
| BiLSTM ᵃ | 52.47 | 54.01 | −1.54 | 0.7159 | **−0.0009** | −0.0043 ± 0.0113 | −0.00499 | 182.0 |
| Transformer ᵃ | 52.60 | 54.01 | −1.41 | 0.7065 | **−0.0003** | +0.0011 ± 0.0074 | +0.00102 | 181.5 |
| Ridge (returns) | 53.28 | 54.01 | −0.73 | 0.2698 | **0.0000** | +0.0004 ± 0.0033 | +0.00124 | 181.5 |
| Logistic (direction) | 53.88 | 54.01 | −0.13 | 0.5217 | **−0.0004** | −0.0008 ± 0.0027 | −0.00027 | 181.6 |
| Hybrid meta (+VIX) ᵇ | 52.65 | 54.01 | −1.37 | 0.7480 | **+0.0011** | −0.0022 ± 0.0109 | −0.00029 | 181.6 |
| Hybrid meta (no VIX) ᵇ | 52.30 | 54.01 | −1.71 | 0.5926 | **+0.0010** | −0.0016 ± 0.0120 | −0.00197 | 181.8 |
| Zero return | 45.99 | 54.01 | −8.02 | — | **−0.0023** | −0.0037 ± 0.0063 | −0.00276 | 181.8 |
| Historical mean | 54.01 | 54.01 | 0.00 | — | **0.0000** | 0.0000 ± 0.0000 | 0.00000 | 181.6 |
| AR(1) returns | 53.71 | 54.01 | −0.30 | 0.2355 | **−0.0027** | −0.0028 ± 0.0063 | −0.00347 | 181.9 |
| ARIMA(5,0,0) returns | 52.05 | 54.01 | −1.96 | 0.8962 | **−0.0082** | −0.0066 ± 0.0101 | −0.00753 | 182.3 |
| Random sign | 50.26 | 54.01 | −3.75 | 0.2745 | **−0.8630** | −1.1199 ± 0.5787 | −0.89480 | 249.9 |

ᵃ Single seed drawn from a distribution measured over five seeds (§6.5):
across seeds the BiLSTM's directional accuracy ranged from 51.46 to 53.91%
and its R²_OOS from −0.0102 to −0.0032; the Transformer's from 52.27 to
52.95% and from −0.0106 to +0.0010 (README §3.1). ᵇ Single seed, spread not
measured; the meta-learner inherits its inputs' seed variance and adds its
own.

We draw three observations from the table. First, no model beat the
majority class. A rule that always predicted "up" scored 54.01% on this
window, and the closest model was the logistic direction classifier at
53.88%, whose forecasts were near-constant votes for "up" (§7.3);
directional accuracy measured against 50% would have reported every row as a
success. Second, no model's
R²_OOS was distinguishable from zero under any aggregation. Under the pooled
aggregation the ridge comparator's +0.00124 and the Transformer's
+0.00102 were the only positive entries, each inside a fold standard
deviation of 0.0033 and 0.0074. Under the fold
median the two meta-learner variants were the largest entries above zero, at
+0.0011 and +0.0010, against a fold standard deviation of
0.0109 and 0.0120, with the ridge at +0.00004 and the
Transformer at −0.0003. Which model sits a thousandth above zero depends on how
the folds are aggregated, and every such entry is inside a seed range or a
fold standard deviation that crosses zero. The historical mean's zero is by
construction, since it is the benchmark. Third, no Diebold-Mariano test against the zero-return baseline
was significant on the 2520 shared days. Recomputed from the committed
headline predictions (`results/headline/AAPL__frozen__seed42.parquet`,
`src/evaluate.py`, `dm_table`), the smallest p-value among the models was
0.174 (Logistic (direction), DM = −1.36) and among the baselines 0.174 (historical
mean); the random-sign baseline was rejected at DM = 23.8, p < 0.0001, in
favour of the zero forecast, which is the sanity check behaving.

The same models under the previous protocol, on the same file and the same
twelve folds, are in `docs/baseline_before_refactor.csv` (run at commit
`3933168`; commit `d26ad59`). We reproduce the comparison in §4.1 and repeat
only its headline here: the best model's R² on the price target was 0.9627
against 0.9992 for the naive zero-change baseline, and its R²_OOS on the
return target is +0.0011 by fold median, −0.0022 ± 0.0109 by fold mean and
−0.00029 pooled. The "before" columns report mean absolute
percentage error and R² computed on a different target with the previous
close available as a feature; they are not like-for-like with the "after"
columns and were never intended to be. They are what the old protocol
claimed.

## 6.2 The null across thirty tickers

We ran the tree ensemble and the five baselines on thirty US large caps
under the same fold configuration and seed (§5.1; `results/predictions/`),
and evaluated each ticker over all its folds with the benchmark seeded from
its own training returns (`results/per_ticker_model.csv`; produced by
`analysis/cross_ticker_sweep.py`). All statistics below were recomputed from
that table.

**Table 2.** Tree ensemble across 30 tickers, 12 folds each (10 for META,
11 for TSLA), seed 42.

| | mean | median | sd | min | max |
|---|---|---|---|---|---|
| DA − majority (pp) | −2.17 | −2.41 | 1.33 | −4.16 | +0.15 |
| R²_OOS, pooled | −0.116 | −0.096 | 0.074 | −0.343 | −0.032 |
| R²_OOS, fold mean | −0.081 | −0.078 | 0.031 | −0.150 | −0.035 |
| **R²_OOS, fold median** | **−0.047** | −0.042 | 0.021 | −0.107 | −0.014 |
| PT p-value | 0.440 | 0.408 | 0.311 | 0.021 | 0.932 |
| t(alpha) | −0.84 | −0.64 | 1.12 | −2.64 | +0.99 |
| beta | 0.54 | 0.53 | 0.07 | 0.44 | 0.65 |
| exposure | 0.54 | 0.52 | 0.05 | 0.45 | 0.63 |

No ticker had positive R²_OOS under any aggregation: 0 of 30 pooled (best
−0.032), 0 of 30 by fold mean (best −0.035) and 0 of 30 by fold median
(best −0.014) (`results/fold_aggregates_sweep.csv`). The fold on which each
ticker scored worst was the same fold, fold 6, on 25 of 30 tickers, with a mean
R²_OOS of −0.481 on that fold alone; on the sweep grid its test window opens on
2020-03-20, the day after the March 2020 low (§8.3). No ticker had
t(alpha) above 1.96 (0 of 30). Three tickers beat their majority class, by
at most 0.15 percentage points (META, PFE, TSLA). Two tickers reached a
Pesaran-Timmermann p below 0.05 (AVGO and META) against 1.5 expected by
chance under the null; the probability of at least two such hits in thirty
independent tests at the 5% level is 0.446. Eight tickers reached an alpha
p below 0.05 against 1.5 expected, which has probability 8.5 × 10⁻⁵ under
the null; we return to those eight in §6.3. After Holm-Bonferroni correction
across the thirty tickers, neither test retained a single hit (0 of 30 for
both; `src/backtest.py`, `holm_bonferroni`).

AAPL, the ticker of Table 1, was not cherry-picked: on the sweep it ranked
15th of 30 on directional accuracy minus majority (−2.31 points), 22nd of 30
on pooled R²_OOS (−0.121) and 14th of 30 on fold-median R²_OOS
(−0.0401), so it was mid-pack.

**The two linear comparators across the thirty tickers.** We ran the ridge
regression and the logistic direction classifier of §5.5 on every ticker under
the sweep grid (`results/comparators/`; statistics recomputed from
`results/per_ticker_model.csv` and `results/fold_aggregates_sweep.csv`; the
tree's row is repeated for comparison).

| Model | DA − majority, mean (pp) | tickers > 0 | pooled R²_OOS, mean | tickers > 0 | fold-median R²_OOS, mean | tickers > 0 | PT p < 0.05 (chance: 1.5) | Holm, within model | alpha p < 0.05, t < −1.96 / t > +1.96 | Holm, within model |
|---|---|---|---|---|---|---|---|---|---|---|
| Tree Ensemble | −2.17 | 3 | −0.1157 | 0 | −0.0473 | 0 | 2 (p = 0.446) | 0 | 8 / 0 | 0 |
| Ridge (returns) | −0.48 | 9 | −0.0052 | 12 | −0.0011 | 18 | 5 (p = 0.016) | 0 | 1 / 1 | 0 |
| Logistic (direction) | −0.28 | 7 | +0.0006 | 18 | +0.0004 | 18 | 4 (p = 0.061) | 1 | 0 / 2 | 0 |

Two things differ from the tree. First, the comparators' R²_OOS is not
uniformly negative: it lies within 0.017 of zero on every ticker by fold
median (pooled: within 0.01 for the logistic on all thirty and for the ridge
on twenty-nine, the exception at −0.099), positive on about half and
negative on the other half. That is what a forecast of the unconditional mean
scores by construction: the benchmark is the historical mean, the comparators
predict something within a few basis points of it (§7.3), and the sign of the
difference across tickers is a coin flip. The tree's 0 of 30 is a statement about
a model that moved away from the mean and lost; the comparators' 12 and
18 of 30 is a statement about models that did not move. Second, the comparators registered more directional hits at the uncorrected 5%
level than a binomial count expects: the ridge 5 of its 28 defined tests
(AVGO, HD, MCD, NEE, XOM), the logistic 4 of 28 (JNJ, MSFT, NKE, UNH), and 11
of the 86 defined direction tests across the three models against 4.3
expected. Four comparator runs have no direction test at all, because the
model predicted "up" on every day of the ticker's test period (ridge on DIS, TSLA; logistic on GOOGL, HD);
on those the comparator is the always-up rule, scores exactly the majority
rate, and is counted as a non-hit. A binomial reference treats the tests as
independent, and they are not: the thirty tickers are US equities over the
same sixteen years on the same fold grid, and the three models are
near-constant predictors fitted to identical data. We therefore replaced the
binomial with a simulated null of equicorrelated test statistics, one-sided
at 5% as the pipeline's PT p-value is, in the same way that the expected
maximum |t| of §6.3 already treats its correlated strategies
(`analysis/correlated_hit_null.py`; `results/hit_count_null.csv`;
`results/cross_ticker_correlation.csv`).

| Family | Observed hits | ρ = 0 (independent) | ρ = 0.3 | ρ = 0.5 | ρ as measured |
|---|---|---|---|---|---|
| Ridge, 28 tickers | 5 | P = 0.012 | 0.084 | 0.099 | 0.024 (ρ = 0.04) |
| Logistic, 28 tickers | 4 | 0.049 | 0.123 | 0.128 | 0.062 (ρ = 0.02) |
| All three models, 90 positions, ρ_model = 0.8 | 11 | 0.033 | 0.126 | 0.134 | 0.051 (ρ_ticker = 0.03) |
| All three models, 90 positions, ρ_model as measured (0.20) | 11 | 0.008 | 0.099 | 0.131 | 0.017 |

The pool is simulated on the full 30 × 3 grid with the four undefined
positions counted as non-hits, which can only raise the expected count and
so works against significance. Under independence the expected count is
4.5 with standard deviation 2.1; at ρ_ticker = 0.3 and ρ_model = 0.8
it is still 4.5 but the standard deviation is 6.2, which is why
the same eleven hits go from a one-in-two-hundred event to a one-in-eight
one. Which correlation describes our design is an empirical question, and
we measured it rather than assumed it. The daily realised returns of the
thirty tickers correlate 0.36 on average and their signs 0.21, inside the 0.3 to
0.5 that is typical of US equities. But the quantity the hit count is made
of is the PT statistic, and a moving-block bootstrap over calendar days (400
replicates, 21-day blocks, every statistic recomputed on each replicate)
puts the correlation of the statistics themselves at only 0.02 to 0.04 across
tickers and 0.20 across models within a ticker. The reason is in the
statistic: Pesaran-Timmermann compares the hit rate with the rate implied by
the two marginal sign frequencies, and a market-wide up day raises both the
hit rate and the implied rate together, so the common market component that
correlates the returns is largely netted out of the statistic. The measured
correlations are therefore the ones that describe the design, and under
them the ridge's 5 of 28 has probability 0.024, the logistic's 4 of 28 has
0.062, and the pooled 11 has 0.017 at the measured model correlation and
0.051 at 0.8. Under the return-level correlation of 0.3 to 0.5 that a reader
might reasonably assume instead, every figure is 0.08 or above. We report
both. On the measured null the ridge's five and the pooled eleven are events
at the 5% level, roughly one in forty and one in sixty, and the pooled count
reaches 0.051 only if the model correlation is taken to be 0.8 rather than
the 0.20 we measured; on the assumed null neither is close to significance.
Holm-Bonferroni across the ninety leaves no directional hit and no alpha hit
under either reading, and no single ridge ticker survives Holm within its
own family.

Which way the hits point. The pipeline's PT p-value is one-sided in the
upper tail, so every one of the eleven is by construction a positive
association between predicted and realised direction; the question is
whether the excess is one-sided or the tail of a symmetric spread. It is
one-sided. Of the 86 defined statistics, 57 are positive, 11 exceed
+1.645 and 0 fall below −1.645 against about 4.3 expected in each tail,
and their mean is +0.46 (`results/mean_pt_null.csv`). Under the same
correlated null that mean has probability 0.003 at the measured correlations
and 0.19 at 0.3 and 0.8: the same pattern as the count, a shift that is real
on the measured null and invisible on the assumed one. This is the opposite
sign from the tree's eight alpha hits, which were all negative and which we
read as systematic underexposure (§6.3). But positive association is not
the same as beating the majority class. Of the eleven, 8 have pooled
directional accuracy above their majority rate and 4 do by fold median;
the others (AVGO tree, MSFT logistic, NEE ridge) have a significant PT statistic and still lose to the always-up
rule, because the association lives in the model's rare "down" calls, and a
model that is long on 83% of days pays for every wrong "down" against a market
that rose on 53% of them. The count therefore means a weak positive
association on a few tickers, too small to convert into directional
accuracy above the majority class on most of them and into a corrected
alpha on any (the comparators' uncorrected alpha hits are two positive for
the logistic and one of each sign for the ridge, none surviving Holm).

The one within-model survivor of Holm is the logistic classifier on NKE:
directional accuracy 52.92% against a majority rate of 50.78%, +2.14 points
pooled over its twelve folds, PT p = 0.00075. Its fold-median directional edge
on the same ticker is −0.66 points: six of the twelve folds are negative, and
the pooled figure is carried by the three largest positive folds (+4.7, +4.2
and +3.8 points against a worst of −4.3). Its alpha (t = +2.43, p = 0.015) does
not survive Holm either. This is the third instance in this paper of a pooled
statistic inheriting its extreme folds, after the pooled R²_OOS that one fold
on the 2020 low moved by 0.09 (§8.3) and the aggregation table on which we
switched the headline magnitude to the fold median (§5.7, Table 1); we treat
the three as one finding, and report NKE's fold median beside its pooled
figure for that reason. The comparators' mean directional edge was negative
on both (−0.48 and −0.28 points), and they held the asset on
82% and 83% of days, so the economic picture is the historical mean's, not
the tree's. We read the uncorrected excess as a weak directional tilt on a
few tickers that is borderline under the correlation we measured, absent
under the correlation a reader would assume, and gone under correction, and
we report the count and the survivor rather than either alone.

The pooled mean of Table 2 is grid-conditional. The sweep's AAPL run sits
on a fold grid 49 trading days earlier than Table 1's, and its pooled
R²_OOS over the same twelve folds was −0.121 against −0.025 on the
frozen grid; we separated the causes in §8.3 and found the fold grid
responsible and the data irrelevant, with 80% of the gap in the one fold
whose boundary falls on the March 2020 low. Re-running the tree on the five
full-model tickers under the frozen grid moved their mean pooled R²_OOS from
−0.122 to −0.032 while their mean fold median moved only from −0.033 to
−0.030 (`results/grid_conditional_tickers.csv`). We therefore state the
pooled mean of −0.116 as a property of this grid, report the fold-median
mean of −0.047 as the magnitude, and note that the counts do not depend on
the choice: no ticker was positive under either grid by either aggregation.

## 6.3 The negative alpha is a power demonstration

The eight alpha hits of §6.2 all had t below −1.96 and none above (8 of 8
negative; `results/per_ticker_model.csv`). This was not the tree finding
skill in either direction. It was the tree losing to the market, and the
mechanism is visible in its own predictions. Recomputed from the committed
sweep predictions (`results/predictions/`), the tree's mean predicted return
across the thirty tickers was −1.48 basis points a day against a realised
mean of +6.00; its mean prediction was negative on 20 of 30 tickers; the
standard deviation of its predictions was on average 0.324 of the standard
deviation of realised returns (range 0.20 to 0.50); and it predicted a
positive return, and so held the asset, on 53.5% of days (range 45 to 63%).
A long/flat rule that is out of the market roughly half the time, at beta
0.44 to 0.65, during a period in which these thirty names rose, earns less
than the market by construction. That is a shortfall, and the alpha test
detected it at this sample size on eight of thirty tickers with the correct
sign.

We use this to pre-empt the objection that our economic tests were
underpowered. The same test, on the same data, detected a real and
replicating effect: the models' systematic underexposure to a rising market.
Had a model produced a positive alpha of comparable size, the test would
have found that too.

We meet the one number that points the other way before a reader finds it.
On AAPL the tree's net Sharpe ratio was 1.166 against 1.045 for buy-and-hold
(Table 3; `docs/figures/aapl_equity_curves.png`; every value in this
paragraph was recomputed from `results/headline/AAPL__frozen__seed42.parquet`
with `src/backtest.py` and agrees with README §3.4). That gap was exposure
and noise, not skill, and the same table showed it in four steps. The
market-adjusted alpha behind the Sharpe was +7.4% a year with t(alpha) =
+1.71, p = 0.087; Holm-corrected over the eleven strategies tested, p = 0.873
(the count is taken from the economics table, `results/aapl_economics_holm.csv`,
and rose from nine when the two linear comparators were added). Under a pure
null with eleven correlated tests (ρ = 0.8) the expected maximum |t| was 1.44 and
P(max|t| ≥ 1.71) = 0.29; with eleven independent tests, 1.92 and 0.63
(`analysis/alpha_correction_and_window.py`; README §3.4). A t of 1.71 is what
no alpha looks like when one looks eleven times. The ridge comparator's net
Sharpe of 1.115 also exceeded buy-and-hold's 1.045, on an alpha of +3.9%
a year with t = +1.23, p = 0.219, Holm p = 1.000: the same reading. Raising the
round-trip cost from 7.5 to 10 basis points took it to t = +1.43, p = 0.152.
A one-day execution lag flipped the alpha negative at zero cost (−1.8% a
year, t = −0.41) and to −5.4% at 7.5 basis points. As for the Sharpe ratio
itself: it compared a position with beta 0.663 and 61% exposure against one
with beta 1 and 100%, so the gap was not a like-for-like measure of skill.
The market-adjusted alpha was, and that alpha survived neither a
multiple-testing correction, nor a 2.5 basis-point cost increase, nor a
single day's delay.

**Table 3.** AAPL economics on the common window, long/flat on the predicted
sign, 7.5 bps round trip (`results/aapl_economics_holm.csv`; Holm over the
eleven strategies with a p-value).

| Strategy | Alpha (ann.) | t(alpha) | p, Holm | Beta | Exposure | Sharpe net |
|---|---|---|---|---|---|---|
| Tree Ensemble | +0.0740 | +1.71 | 0.873 | 0.663 | 0.608 | 1.166 |
| Hybrid meta (+VIX) | −0.0419 | −1.88 | 0.667 | 0.937 | 0.888 | 0.861 |
| Transformer | +0.0156 | +0.39 | 1.000 | 0.742 | 0.845 | 0.963 |
| Ridge (returns) | +0.0390 | +1.23 | 1.000 | 0.860 | 0.846 | 1.115 |
| Logistic (direction) | −0.0162 | −1.05 | 1.000 | 0.971 | 0.985 | 0.973 |
| BiLSTM | −0.0257 | −0.93 | 1.000 | 0.899 | 0.848 | 0.896 |
| Historical mean | −0.0000 | −1.00 | 1.000 | 1.000 | 1.000 | 1.045 |
| Buy and hold | 0 | — | — | 1.000 | 1.000 | 1.045 |

The tree's alpha under stress (same source):

| lag (days) | cost (bps) | alpha (ann.) | t | p |
|---|---|---|---|---|
| 0 | 0.0 | +0.1101 | +2.55 | 0.011 |
| 0 | 7.5 | +0.0740 | +1.71 | 0.087 |
| 0 | 10.0 | +0.0619 | +1.43 | 0.152 |
| 1 | 0.0 | −0.0179 | −0.41 | 0.682 |
| 1 | 7.5 | −0.0539 | −1.24 | 0.217 |
| 1 | 10.0 | −0.0660 | −1.51 | 0.131 |

## 6.4 Distribution shift explains the magnitude of the error

For every fold of every ticker we compared the training and test return
distributions with a two-sample Kolmogorov-Smirnov test
(`results/fold_diagnostics.csv`, 357 fold-ticker pairs). The test rejected
at the 5% level on 119 of 357 folds, 33.3%, against 5.0% expected under no
shift, a factor of 6.7 (recomputed from the file). The shift was one of
shape rather than support: on average only 0.20% of a fold's test returns
fell outside its training range (same file, `test_outside_train_range`).

We then regressed the tree's per-fold R²_OOS on the fold's KS statistic
across all 357 pairs (`results/shift_tests.csv`; the per-fold series it is
fitted on is `results/fold_r2_oos.csv`; README §3.5;
`docs/figures/shift_vs_r2_oos.png`; both files are written by
`analysis/distribution_shift_and_exposure.py`, which requires the gitignored
raw cache to rebuild each fold's training returns):

| Outcome | Slope | t | p | R² |
|---|---|---|---|---|
| R²_OOS on KS | −1.576 | −6.84 | < 0.0001 | 0.116 |
| DA (%) on KS | −11.65 | −2.09 | 0.038 | 0.012 |

Folds on which the test rejected had mean R²_OOS of −0.1419 (n = 119)
against −0.0505 (n = 238) on folds where it did not, a factor of 2.8; Welch's
t was −4.10, p = 0.0001, and a Mann-Whitney test, which does not assume that
per-fold R²_OOS is normal, gave U = 10818, p = 0.0003
(`results/shift_tests.csv`; README §3.5). The same comparison on directional
accuracy gave 49.69% against 50.82%, Welch p = 0.0056 and Mann-Whitney
p = 0.011 (same file).

We report the ceiling honestly. Shift explained 11.6% of the variance in
per-fold R²_OOS and 1.2% of the variance in directional accuracy. It
explains how much the models lost on a bad fold, not why they had no
directional signal on a good one. We do not claim more than that.

## 6.5 Seed variance is the same order as the architecture difference

We reran the BiLSTM and the Transformer on AAPL for seeds 0 to 4 under the
headline fold configuration (`results/seeds/`; §5.4 records that this study
sits on the sweep's fetch grid). Our statistic was the mean absolute gap
between the two architectures divided by the mean standard deviation across
seeds. Its distribution under a null in which the two architectures are
identical, five seeds each, was obtained by simulation (400,000 trials of
two iid normal samples at n = 5) rather than from a closed form: median
1.22, 90% interval [0.67, 1.77] (README §3.3).

| Metric | Observed ratio | p (one-tailed) | Inside the 90% interval |
|---|---|---|---|
| R²_OOS | 1.66 | 0.083 | yes |
| Directional accuracy | 1.69 | 0.072 | yes |
| sd of predicted returns | 0.62 | 0.036 | no (lower tail) |

We did not report point ratios as findings. On the two metrics that matter
the observed ratios fell inside the interval expected if the architectures
were identical, so neither the Transformer-over-BiLSTM ordering nor the
reverse was identifiable at one seed, and we withdrew the ranking that an
earlier draft had made. The 0.62 on prediction dispersion fell below the
interval, which would suggest that reseeding moved that quantity more than
switching architecture did; we do not claim it, since it is one comparison
at n = 5 among three reported and is not corrected for that.

Two of the project's apparent positive findings dissolved across seeds.
The Transformer's R²_OOS changed sign, from −0.0106 to +0.0010, so Table 1's
only positive entry was seed 42 landing on one side of a distribution that
straddles zero. And BiLSTM seed 4 reported a Pesaran-Timmermann p of 0.0137
with directional accuracy 0.43 points above its majority class, the only
neural configuration in the project to do so (the linear comparators'
directional hits are treated in §6.2); its other four seeds gave p = 0.329,
0.521, 0.708 and 0.893, and the five Transformer seeds gave 0.201 to 0.696
(recomputed from `results/seeds/` with `src/evaluate.py`). The study ran ten
tests, and the probability of at least one hit at the 5% level among ten is
1 − 0.95¹⁰ = 0.401. We report it as the expected chance hit.
