# Paper Outline

Working title: **Protocol Artifacts in Deep Learning for Stock Return
Prediction: A Reproducible Null Result Across 30 Equities**

---

## 0. What the paper claims

One sentence: *When daily equity return forecasting is evaluated under a
leak-free walk-forward protocol with correct benchmarks, deep architectures
show no forecastability across 30 US equities, and the performance commonly
reported in this literature is traceable to a small number of specific,
nameable evaluation choices, each of which we reproduce and quantify.*

This is a negative result with a positive mechanism. The negative result alone
would be weak, since "our models did not work" is unpublishable. What makes it
a paper is that you can name, reproduce and measure the artifacts, and you have
a distribution-shift mechanism that explains the magnitude of the failure.

**What you must not claim:** that stock returns are unpredictable, that
markets are efficient, or that deep learning cannot work in finance. You
tested three architectures, one horizon, one asset class, one country,
one seed for most runs. Say exactly that.

---

## 1. Venue

Realistic targets, in order of fit:

1. **A workshop paper.** ICAIF (ACM International Conference on AI in Finance)
   has a workshop track; ML4Finance-style workshops at NeurIPS/ICML take
   reproducibility and negative results. Best fit: the contribution is
   methodological, the scope is bounded, and workshops actively want
   replication-and-protocol work.
2. **A reproducibility venue.** ReScience C, or the MLRC (ML Reproducibility
   Challenge). Your artifact quality is the strongest thing you have and these
   venues weight it heavily.
3. **An applied journal.** *Journal of Forecasting*, *International Journal of
   Forecasting*, *Quantitative Finance*. Longer review cycles; IJF in
   particular has a tradition of publishing forecast-evaluation critiques.
4. **arXiv preprint** regardless, on submission day. Cite the frozen commit.

For a final-year capstone, target 1 or 2. Do not target a top-tier main
conference; the scope is too narrow and it will waste six months.

---

## 2. Structure

### Abstract (150-200 words)
Lead with the headline number. Same ticker, same data, same folds:
R² = 0.9627 under the conventional protocol, R²_OOS = −0.0003 under a
corrected one. Then: 30 tickers, 0/30 positive R²_OOS, nothing survives
multiple-testing correction, distribution shift in 33% of folds explains
the magnitude, seed variance exceeds architecture differences.

### 1. Introduction
- The claim pattern in the literature: high R², low MAPE, directional
  accuracy in the high 50s or 60s on daily equity prices.
- The gap: these are usually reported on price levels with the previous
  close available as a feature, against no trivial baseline.
- Contributions, stated as a numbered list (see §3 below).
- Explicit non-claims (see §0).

### 2. Related work
Three strands:
- Deep learning for stock prediction (LSTM/Transformer/hybrid papers). Be
  specific and fair: cite the actual papers making the high-accuracy claims,
  describe their protocols accurately, do not strawman.
- Forecast evaluation methodology: Diebold-Mariano (1995), Pesaran-Timmermann
  (1992), Campbell-Thompson (2008) for R²_OOS, Welch-Goyal (2008) on
  out-of-sample return predictability, Bailey-Lopez de Prado on backtest
  overfitting.
- Reproducibility and leakage in applied ML: Kapoor-Narayanan (2023) on
  leakage in ML-based science is the closest antecedent and you should
  position against it explicitly. Your paper is essentially their argument
  instantiated and quantified in one domain.

### 3. Contributions
State these as the numbered list. They are, in descending strength:

1. **A quantified before/after on identical data.** R² 0.9627 to R²_OOS
   −0.0003, same ticker, same frozen CSV, same 12 folds, same
   hyperparameters. Only the target and evaluation protocol change.
2. **Five reproducible protocol artifacts**, each isolated and measured
   (see §4).
3. **A universal null across 30 sector-spread equities** with correct
   multiple-testing treatment.
4. **A distribution-shift mechanism** linking train/test divergence to
   degradation, with the link tested rather than asserted.
5. **A seed-variance result** showing architecture comparisons are not
   identifiable at one seed.
6. **An open, tested artifact**: 331 tests, offline fixtures, mutation-tested
   leakage detection, frozen data for the headline figure.

### 4. Protocol artifacts (this is the core section)

One subsection each. For every artifact: the mechanism, the magnitude in your
own pipeline, and how to detect it.

| # | Artifact | Measured effect |
|---|---|---|
| 1 | Price-level target with the previous close as a feature | R² 0.9627 vs R²_OOS −0.0003 |
| 2 | Cold-start R²_OOS benchmark | Zero-return baseline +0.087 vs −0.005 (fixture); +0.0349 vs −0.0006 (30-ticker sweep); moved 1/30 tickers from apparently-positive to 0/30 |
| 3 | Mismatched evaluation windows across models | Base models lost 1.9-2.3pp DA when restricted to the meta's window; apparent BiLSTM/Transformer advantage over the meta halved |
| 4 | Short evaluation window | VIX gating: DM +2.32 (p=0.0203) on 420 days vs −0.595 (p=0.552) on 2520 days. Opposite signs, same code |
| 5 | Single-seed neural results | The project's only significant PT result (p=0.0137) appears in 1 of 5 seeds; Transformer R²_OOS changes sign across seeds |

**Artifact 2 deserves its own paragraph and is the strongest rhetorical point
in the paper.** You found this bug in Phase 5, fixed it, wrote a regression
test, and it recurred in the aggregation path because that path did not pass
the training returns. It came back *in a codebase built specifically to prevent
it, by people who already knew the failure mode.* That is a much stronger claim
than "protocol errors inflate results." Report the recurrence honestly,
including that it was caught by a sanity check (a zero-return baseline cannot
beat the historical mean by 3.5%) rather than by the regression test.

Frame artifacts as a **detection checklist** in the discussion, so the section
gives readers something actionable rather than only a critique.

### 5. Method
- Data: 30 US equities, sector spread, 2010-2026, Yahoo Finance. State
  survivorship bias explicitly.
- The row contract: features at close of day t, target log(C[t+1]/C[t]),
  predictions keyed by target date. Reproduce §1 of REFACTOR_PLAN.md.
- 31 scale-free features; the transformations table.
- Walk-forward: 12 folds, min-train 1008, test 252, step 252. **State the
  annual-refit limitation explicitly**: a model forecasts for up to 12 months
  without refitting. Justify as a compute concession and note the 9-fold
  quarterly config used for the fixture experiments.
- Models: XGBoost+RF ensemble, BiLSTM, Transformer, fold-respecting stacked
  meta-learner with VIX gating.
- Baselines: zero-return, historical mean, AR(1), ARIMA, random sign.
- Metrics: DA against the majority-class rate (not 50%), Pesaran-Timmermann,
  Campbell-Thompson R²_OOS with a fold-seeded benchmark, Diebold-Mariano,
  market-adjusted alpha with Holm correction.

### 6. Results

**6.1 The before/after.** Table 1, the headline. Note the before column
reports MAPE and price R² computed on a different target and is not
like-for-like; it is what the old protocol *claimed*.

**6.2 The null across 30 tickers.** 0/30 positive R²_OOS. 2/30 PT hits vs 1.5
expected (p=0.446). 8/30 alpha hits, all negative. Nothing survives Holm.
AAPL is mid-pack, so the headline ticker was not cherry-picked.

**6.3 The negative alpha is a power demonstration.** Eight significant hits,
all negative, P(≥8) < 0.001, with a clean mechanism: mean predicted return
−1.5bps against +6.0bps realised, prediction dispersion 32% of realised,
exposure ~50% in a 16-year bull market. Use this to preempt the
underpowered-tests objection: the economic test *did* detect a real,
replicating effect at this sample size. It was the model losing to the market.

Meet the one number that points the other way before a reader finds it. On
AAPL the tree's net Sharpe is 1.166 against buy-and-hold's 1.045 (Table 3,
equity-curve figure). That gap is exposure and noise, not skill, and the same
table shows it in four steps. The market-adjusted alpha behind the Sharpe is
+7.4% a year with t(alpha) = +1.71, p = 0.087; Holm-corrected over the nine
strategies tested, p = 0.698. Under a pure null with nine correlated tests
(rho = 0.8) the expected maximum |t| is 1.40 and P(max|t| >= 1.71) = 0.27;
with nine independent tests, 1.84 and 0.56. A t of 1.71 is what no alpha looks
like when you look nine times. Raising the round-trip cost from 7.5 to 10 bps
takes it to t = +1.43, p = 0.152. A one-day execution lag flips the alpha
negative at zero cost (-1.8% a year, t = -0.41) and to -5.4% at 7.5 bps. As
for the Sharpe itself: it compares a position with beta 0.663 and 61%
exposure against one with beta 1 and 100%, so the gap is not a like-for-like
measure of skill. The market-adjusted alpha is, and that alpha does not
survive a multiple-testing correction, a 2.5 bps cost increase, or a single
day's delay.

**6.4 Distribution shift.** 33.3% of 357 folds reject KS at 5%, 6.7× the null
rate. Shifted folds are 2.8× worse in R²_OOS (−0.142 vs −0.051, Welch
p=0.0001, MW p=0.0003). Regression R²_OOS ~ KS: t = −6.84, p < 0.0001.
**Report the ceiling honestly**: KS explains 11.6% of R²_OOS variance and 1.2%
of DA variance. Shift explains the *magnitude* of the error, not the absence
of directional signal. Do not overclaim this; the honest version is more
credible and a reviewer will find the R² anyway.

**6.5 Seed variance.** Seed noise and architecture difference are the same
order of magnitude. Do not report point ratios; at n=5 a true ratio of 1.0
gives a 90% interval of [0.67, 1.77]. Report that neither the
Transformer/BiLSTM ordering nor the sign of Transformer R²_OOS is identifiable
at one seed, and that the study's one significant PT result is the expected
chance hit from 10 tests (P(≥1) = 0.401).

### 7. Discussion
- The detection checklist.
- What would change the conclusion: longer horizons, cross-sectional rather
  than time-series prediction, intraday data, alternative data, more
  architectures, non-US markets.
- Why these architectures converge to the unconditional mean: MSE regression
  on a target that is mostly noise has the conditional mean as its optimum;
  under distribution shift the conditional mean collapses to the
  unconditional one. Your three independent observations of this (BiLSTM
  emitting near-constants, tree predictions compressed to 6-20% of training
  range, meta ridge shrinking to its intercept) are the evidence.

### 8. Limitations
Be aggressive here; it buys credibility. One horizon. One market.
Survivorship bias. Annual refit. Single seed for the sweep. No intraday, no
alternative data, no cross-sectional models. Three architectures is not
"deep learning."

### 9. Reproducibility statement
Commit hash, frozen CSV, test count, how to reproduce Table 1 in one command.

---

## 3. Figures

1. **Before/after bar pair.** The single most important figure. R² 0.9627
   next to R²_OOS −0.0003, with the axis break made honest.
2. **Cross-ticker R²_OOS distribution**, 30 points, zero line marked, all
   below it.
3. **KS vs R²_OOS scatter**, 357 fold-ticker points with the fitted line.
   Include the R² = 0.116 in the caption so the ceiling is visible.
4. **Seed spread vs architecture gap**, 5 seeds x 2 architectures, showing
   the intervals overlapping.
5. **Predicted vs realised return dispersion**, showing the 32% compression.

---

## 4. Writing order

Do not write section 1 first. Write in this order:

1. Section 6 (Results). The tables exist; turn them into prose.
2. Section 5 (Method). Mostly a rewrite of REFACTOR_PLAN.md §1.
3. Section 4 (Artifacts). The core contribution; needs the most care.
4. Section 7 (Discussion) and 8 (Limitations).
5. Section 2 (Related work). Do this properly with a real literature search;
   it is the section most likely to sink you in review if the citations are
   thin or the characterisations of prior work are unfair.
6. Section 1 (Introduction) and the abstract, last, once you know what the
   paper actually says.

---

## 5. The review objections you will get, and your answers

**"You just built bad models."** Answer: possibly, but the null holds across
30 tickers, five baselines, and both neural architectures; the tree's
predictions are compressed to 6-20% of their training range and the meta's
own cross-validation regularises it to a constant. Three independent
mechanisms all indicate no learnable signal, not poor optimisation. Also:
the burden is symmetric, and papers claiming success rarely report a
zero-return baseline at all.

**"Your hyperparameters were not tuned."** Answer: correct, deliberately.
Tuning against the test set is one of the artifacts we are documenting.
Report that the meta's ridge penalty was CV-selected and landed at 100-1000,
i.e. the only tuned component chose maximal shrinkage.

**"Daily horizon is known to be hard."** Answer: agreed, and that is the
point. The literature we cite makes strong claims *at this horizon*. State
which papers, at which horizon.

**"33% KS rejection could be sample-size artifact."** Answer: the null rate
is 5% by construction of the test; you observe 6.7×. Also report the KS
rejection rate under a stationary bootstrap of the same data as a
sanity check if a reviewer pushes.

**"n=5 seeds is too few."** Answer: agreed, and stated. The claim is an
order-of-magnitude comparison, not a precise ratio, and the interval is
reported.

---

## 6. Immediate next steps

1. Push. Then tag the commit that produced the results: `git tag v1.0-results`.
2. Apply the three corrections from this turn (seed ranges on the headline
   table, order-of-magnitude framing, withdraw the neural model ranking).
3. Rewrite the README results section to match reality. The
   "state-of-the-art precision: ~99.11%" line must go before anyone external
   sees the repo.
4. Generate the five figures.
5. Start writing section 6.

The experiments are done. Everything from here is writing.
