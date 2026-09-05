# Section 2: Related Work

**DRAFT. Read the verification notes at the bottom before submitting.**

Claims marked `[V]` rest on sources I read in full or on publisher-level
metadata. Claims marked `[VERIFY]` require you to read the paper. Do not
remove a `[VERIFY]` marker without reading the source.

---

## 2. Related Work

Our work sits at the intersection of three literatures: deep learning applied
to equity price prediction, the forecast-evaluation methodology developed in
econometrics, and the recent body of work documenting leakage and
reproducibility failures in machine-learning-based science. We review each in
turn and then state our position relative to the closest prior work.

### 2.1 Deep learning for equity price prediction

The application of recurrent and attention-based architectures to equity
forecasting has produced a large literature over the past decade. `[V]` Fischer
and Krauss (2018), writing in the *European Journal of Operational Research*,
established the reference point for this line of work, reporting that long
short-term memory networks generate more accurate directional forecasts of
S&P 500 constituent returns than momentum-based and memory-free benchmarks.
`[V]` Wang et al. (2022) extended the approach to attention-based models,
applying a deep Transformer to stock market index prediction with back-testing
on major indices, and their paper has become the corresponding reference point
for Transformer-based work.

We emphasise at the outset that our critique is not directed at these papers.
Both forecast returns or directions rather than price levels, both evaluate on
broad cross-sections rather than single instruments, and both report
comparisons against non-trivial benchmarks. `[VERIFY: confirm each of these
three claims by reading the methods sections. If any is wrong, this paragraph
must change, and it is the paragraph a reviewer is most likely to check.]`

The subsequent literature is large and heterogeneous. Architectural variants
have proliferated, including convolutional-recurrent hybrids `[VERIFY: Lu et
al. 2020, 2021]`, decomposition-based pipelines `[VERIFY: the VMD-TMFG-LSTM
paper in Journal of Big Data, 2025; Yañez et al. 2024]`, optimisation-augmented
LSTMs `[VERIFY: Gülmez 2023]`, sentiment-augmented models `[VERIFY: Jin et al.
2020; Ouf et al. 2024]`, and incremental-learning Transformers `[VERIFY:
IL-ETransformer, 2025]`. Surveys of the field `[VERIFY: Sezer et al.; Thakkar
and Chaudhari 2021; Shah et al. 2022]` document hundreds of such
contributions.

Reported performance in this literature is frequently very high. `[V]` A 2025
review in *Archives of Computational Methods in Engineering* opens by
observing that machine learning techniques are now used to predict stock
prices with high accuracy, a framing that reflects the prevailing view rather
than an outlier position.

Three properties of this literature motivate our study. `[VERIFY: this is the
central empirical claim of Section 2 and it must be supported by your protocol
table, not asserted. Fill the table in before finalising these three items, and
give the counts.]`

First, a substantial fraction of these studies forecast price levels rather
than returns. `[VERIFY: give the count, n of m papers surveyed.]` As an
illustrative case, `[V]` one recent study forecasts the closing prices of
Apple, Google, Microsoft and Amazon using data sourced from Yahoo Finance,
reporting a mean absolute percentage error of 2.72 against ARIMA as the
comparison model. `[VERIFY: confirm whether the previous close is available in
the feature window, and whether any naive or persistence baseline is reported.
If the answer is yes and no respectively, this paper is the cleanest available
illustration of the artifact we describe in Section 4.1, and it is on
substantially the same data as our own experiments.]`

Second, trivial baselines are rarely reported. `[VERIFY: count how many of the
papers you read report a naive, persistence, zero-return, or majority-class
baseline. Our expectation is that this number is small, but the claim is only
worth making with the count attached.]`

Third, significance testing is uncommon. `[VERIFY: count how many report
Diebold-Mariano, Pesaran-Timmermann, confidence intervals, or any
multiple-testing correction.]` This mirrors a finding reported outside finance:
`[V]` in a reproducibility study of civil war prediction, Kapoor and Narayanan
found that nine of the twelve papers for which complete code and data were
available included no significance tests or uncertainty quantification for
classifier performance comparison.

### 2.2 Forecast evaluation methodology

The statistical apparatus required to evaluate forecasts rigorously is
long established. `[VERIFY citation details for all of 2.2]` Diebold and
Mariano (1995) introduced the standard test for comparing predictive accuracy
between two forecasts. Pesaran and Timmermann (1992) provided a non-parametric
test of directional predictive performance that accounts for the marginal
frequencies of predicted and realised signs, a correction that matters
precisely when a predictor is close to constant. Campbell and Thompson (2008)
formalised the out-of-sample R² against an expanding historical-mean
benchmark, which has since become the standard measure of return
predictability. Harvey, Liu and Zhu (2016) argued for substantially raised
significance thresholds in cross-sectional asset pricing given the volume of
hypotheses tested, and Bailey et al. (2014) documented the effect of backtest
overfitting on out-of-sample performance.

Two results in this literature bear directly on ours. Welch and Goyal (2008)
evaluated a broad set of equity premium predictors out of sample and found
that most fail to outperform the historical average, a result that reframed
the return-predictability literature around the difficulty of beating a
trivial benchmark. `[VERIFY]` Our finding is the deep-learning-era analogue of
theirs: the architectures we evaluate do not outperform the historical mean at
the daily horizon, and the metric by which we establish this is the one
Campbell and Thompson introduced in direct response to Welch and Goyal.

This apparatus is standard in the forecasting and empirical asset pricing
literatures. It is not standard in the deep-learning stock prediction
literature we survey in Section 2.1, and that gap is the subject of our paper.

### 2.3 Leakage and reproducibility in machine-learning-based science

`[V]` Kapoor and Narayanan (2023) surveyed twenty-two papers across seventeen
scientific fields and found that data leakage collectively affected at least
294 papers, in some cases producing wildly overoptimistic conclusions. They
introduced a taxonomy of eight leakage types organised under three headings:
absence of clean train-test separation (L1, comprising no test set, L1.1;
pre-processing across the split, L1.2; feature selection across the split,
L1.3; and duplicate records, L1.4), use of illegitimate features (L2), and a
test set not drawn from the distribution of scientific interest (L3, comprising
temporal leakage, L3.1; non-independence between training and test samples,
L3.2; and sampling bias in the test distribution, L3.3). They proposed model
info sheets, comprising twenty-one questions, as a preventive instrument.

`[V]` Their case study is the closest precedent for the form of our result. In
civil war prediction, complex models were believed to substantially outperform
logistic regression. They identified errors in four of twelve papers, and in
exactly the four that claimed superiority for complex models. When those errors
were corrected, the complex models no longer performed substantively better
than decades-old logistic regression in any case. `[V]` They further situate
this alongside comparable findings for children's life outcomes and recidivism
prediction, and argue for research that identifies and communicates the limits
to prediction in a given domain.

`[V]` Within finance, Prata et al. (2024), in *Artificial Intelligence Review*,
conducted the closest study to ours. They evaluated fifteen state-of-the-art
deep learning models for stock price trend prediction on limit order book data
using an open-source framework, and reported that all models exhibit a
significant performance drop when exposed to new data, raising questions about
their real-world applicability.

Correct practice is not absent from the recent stock-prediction literature.
`[VERIFY: read both of these before finalising this paragraph. It is the
paragraph that protects you from the charge of attacking a straw man.]` At
least two 2026 studies adopt leakage-aware protocols explicitly. One reports
that a standalone LSTM and a CNN-LSTM both failed to outperform a simpler
feed-forward network under strictly backward-looking normalisation and an
expanding walk-forward gap, attributing the result to the application of
high-capacity models to noisy data, which is consistent with the convergence
behaviour we document in Section 7.3. Another, evaluating a ten-day
cross-sectional outperformance task across five international indices under
embargo and staleness controls, reports that a deep LSTM remains the strongest
model under leakage-safe settings and that performance degrades when those
controls are relaxed.

We regard the latter result as compatible with ours rather than contradictory,
for two reasons. `[VERIFY: this argument is only sound if the two task
descriptions are as stated. Read the paper.]` The task differs in horizon,
being ten days rather than one, and in structure, being a cross-sectional
outperformance classification across index constituents rather than an
absolute next-day return forecast for an individual instrument. Positive
results in cross-sectional and longer-horizon settings are well supported
elsewhere `[VERIFY: Gu, Kelly and Xiu 2020]`. Our claim is specific to the
daily single-name horizon, and we state it as such.

### 2.4 Position of this work

Our contribution differs from the three literatures above in the following
respects.

Relative to Section 2.1, we do not propose an architecture. We evaluate
established ones under a protocol designed to eliminate the specific failure
modes we identify, and we report the difference between what the conventional
protocol reports and what the corrected protocol reports on identical data.

Relative to Section 2.2, we do not introduce a statistical method. We apply
existing tests to a literature in which they are seldom applied, and we
quantify how much of the reported performance in that literature they remove.

Relative to Section 2.3, our position requires more care, and it is the
sharpest way to state our contribution. Of the five protocol artifacts we
document in Section 4, only one maps onto Kapoor and Narayanan's eight-type
taxonomy without qualification.

`[VERIFY: the following mapping is my reading of their taxonomy applied to your
five artifacts. Check every line of it yourself. If you disagree with any of
these placements, change it, because this paragraph is your novelty claim and
you must be able to defend it.]`

- Backward fill of macroeconomic series, which we identify and remove in our
  data layer, is a clear instance of temporal leakage (L3.1), and the
  associated global rescaling would fall under pre-processing across the split
  (L1.2). These are covered by the existing taxonomy.
- Our first artifact, a price-level target evaluated while the previous close
  is available as a feature, is **not leakage under their definition**. The
  previous close is legitimately available at forecast time, so the feature is
  not illegitimate in the sense of L2, and there is no contamination of the
  test set. The overoptimism arises instead from a target and metric
  specification under which a trivial persistence forecast attains a high
  coefficient of determination. Kapoor and Narayanan record metric choice as a
  distinct category of reporting failure in their survey, separate from the
  leakage taxonomy, and our artifact belongs to that category rather than to
  leakage proper.
- Our second artifact, a cold-started out-of-sample benchmark, falls outside
  the taxonomy entirely. The error resides in the construction of the
  reference quantity rather than in the model, the features, or the test
  distribution. To our knowledge this failure mode has not previously been
  characterised.
- Our third artifact, comparison of models evaluated over non-identical test
  windows, is adjacent to sampling bias in the test distribution (L3.3) but is
  distinct from it. Each model's test set may be individually representative
  while the comparison between them remains invalid.
- Our fourth and fifth artifacts, sensitivity to evaluation-window length and
  to random seed, are properties of the reported estimate rather than of the
  model or its data, and are likewise outside the taxonomy.

We therefore position this work as an extension of Kapoor and Narayanan's
framework rather than a further illustration of it. Four of the five failure
modes we document produce the same overoptimism as leakage while arising from
error classes their taxonomy does not cover, and we suggest that a complete
account of reproducibility failure in predictive science must address the
construction of benchmarks, the alignment of comparisons, and the stability of
reported estimates alongside the integrity of the train-test split.

Relative to Prata et al. (2024), we differ in data, task and contribution. They
evaluate limit order book models on an intraday trend-classification task and
measure generalisation failure across datasets. We evaluate daily OHLCV models
on an absolute return-regression task and attribute the gap between reported
and corrected performance to five separately measured causes.

---

## Verification notes

### What I actually read

**Read in full:** Kapoor and Narayanan (2023), *Patterns* 4(9), 100804,
including the complete taxonomy, the civil war case study, Finding 2 on
significance testing, and the discussion section. Every `[V]` claim in Section
2.3 about this paper is from the full text.

**Read at abstract or publisher-metadata level only:** Fischer and Krauss
(2018), Wang et al. (2022), Prata et al. (2024), the 2025 *Archives of
Computational Methods in Engineering* review, arXiv 2505.05325, and the two
2026 ScienceDirect papers. The characterisations above reflect what those
sources state about themselves. They are almost certainly accurate at that
level of generality, but I could not verify protocol details.

**Not read at all:** every paper cited in Section 2.1 as an architectural
variant. Those citations came from other papers' reference lists.

### What you must do before this section is submittable

1. **Fill the protocol table.** The three numbered claims in Section 2.1 are
   currently assertions. They become evidence only when you attach counts. This
   is the single most important remaining task and no one else can do it.

2. **Verify the Fischer and Krauss paragraph.** I claim their paper forecasts
   returns or directions, evaluates on a cross-section, and reports non-trivial
   benchmarks. If any of that is wrong, you are misrepresenting the field's
   most-cited paper in your opening, which is the worst place to be wrong.

3. **Read both 2026 papers.** Section 2.3's penultimate paragraph and Section
   2.4's compatibility argument both depend on task descriptions I took from
   abstracts. If the cross-sectional or horizon characterisation is inaccurate,
   the argument collapses and a reviewer will find it.

4. **Check the taxonomy mapping yourself.** It is my reading, not theirs. It is
   also your strongest novelty claim, which means it is the paragraph most
   likely to be challenged. Read their Section on the taxonomy and satisfy
   yourself that four of your five artifacts genuinely fall outside it.

5. **Verify every citation** against the publisher page. Volume, issue, pages,
   year, DOI.

6. **Decide whether to name papers in Section 2.1.** I have left the
   architectural variants as bracketed placeholders rather than naming them in
   prose, because naming a paper as an example of poor practice without having
   read it is not acceptable. Name them only after you have read them.

### Length

Roughly 1,400 words as drafted, which is appropriate for a journal submission.
For an eight-page workshop paper you will need to compress to about 700. Cut
Section 2.1's architectural survey to two sentences and keep the taxonomy
mapping intact, since that is the part that carries your contribution.
