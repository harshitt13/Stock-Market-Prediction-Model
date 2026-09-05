# 2. Related Work

Our work sits at the intersection of three literatures: deep learning
applied to equity price prediction, the forecast-evaluation methodology
developed in econometrics, and the recent body of work documenting leakage
and reproducibility failures in machine-learning-based science. We review
each in turn and then state our position relative to the closest prior work.
Every characterisation below was checked against the source at the level
recorded in the verification log that follows the section; the two points
that could not be checked are marked `[VERIFY]` in the text.

## 2.1 Deep learning for equity price prediction

The application of recurrent and attention-based architectures to equity
forecasting has produced a large literature over the past decade. Fischer
and Krauss (2018) established the reference point for this line of work:
they deployed long short-term memory networks to predict out-of-sample
directional movements of the constituent stocks of the S&P 500 from 1992 to
2015, and reported that the networks outperformed memory-free classifiers, a
random forest, a deep neural network and a logistic regression, and a
rules-based short-term reversal strategy. Wang et al. (2022) became the
corresponding reference point for attention-based work, applying a deep
Transformer to the prediction of major stock market indices, the CSI 300,
the S&P 500, the Hang Seng Index and the Nikkei 225, and comparing it with
convolutional and recurrent baselines by error metrics and by back-tested
trading.

We emphasise at the outset that our critique is not directed at these
papers. Fischer and Krauss forecast direction rather than price level,
evaluate on a broad cross-section rather than a single instrument, and
report comparisons against non-trivial benchmarks. Wang et al. evaluate
across four indices and report back-tested trading results alongside error
metrics; their target is the index series itself `[VERIFY: we could not
retrieve the abstract or methods; secondary descriptions characterise the
task as index-level prediction evaluated by error metrics and back-testing.
Confirm the target from the paper before submission]`. Our critique
concerns a protocol pattern, price-level targets scored by fit statistics
without a trivial baseline or a significance test, and we document below
how common that pattern is.

The subsequent literature is large and heterogeneous. Architectural variants
have proliferated: convolutional-recurrent hybrids (Lu et al., 2020, 2021),
decomposition-based pipelines (Jin et al., 2020; Zhang et al., 2025),
sentiment-augmented models (Jin et al., 2020; Ouf et al., 2024) and
incremental-learning Transformers (Qian, 2025). Surveys of the field (Sezer
et al., 2020; Thakkar and Chaudhari, 2021; Shah et al., 2022) document
hundreds of such contributions. Reported performance is frequently very
high. A 2025 review in *Archives of Computational Methods in Engineering*
opens by observing that machine learning techniques are now used to predict
stock prices with high accuracy (Shafiei Hafshejani and Mansouri, 2025), a
framing that reflects the prevailing view rather than an outlier position.

Three properties of this literature motivate our study. Rather than assert
them, we tabulate them for the ten primary studies cited in this section,
recording for each the prediction target, whether a naive, persistence,
zero-return or majority-class baseline is reported, whether any
significance test or uncertainty quantification is reported, and the level
at which we were able to check (full text, abstract, or secondary
description). Absence from an abstract is not proof of absence from a
paper, so the baseline and significance counts are reported at the level
examined and should be read as what the papers foreground.

| Study | Target | Naive or persistence baseline | Significance test or uncertainty | Checked at |
|---|---|---|---|---|
| Fischer and Krauss (2018) | direction (S&P 500 constituents) | none; benchmarks are RF, DNN, LR and a reversal strategy | none reported | abstract |
| Wang et al. (2022) | index series `[VERIFY]` | none reported | none reported | secondary |
| Lu et al. (2020) | next-day closing price, Shanghai Composite | none reported | none reported | abstract |
| Lu et al. (2021) | next-day closing price, Shanghai Composite | none reported | none reported | abstract |
| Jin et al. (2020) | closing price | none reported | none reported | abstract |
| Ouf et al. (2024) | price (Apple, Google, Tesla) | none reported | none reported | abstract |
| Zhang et al. (2025) | price (one Shanghai stock) | none reported; ARIMA and six networks | none reported | abstract |
| Qian (2025) | closing price (two Chinese indices, three stocks) | none, anywhere in the paper | none; 20-run averages without error bounds | full text |
| Chaudhary (2025) | closing price (Apple, Google, Microsoft, Amazon) | none; ARIMA only | none; confidence intervals deferred to future work | full text |
| Tejas et al. (2026) | multi-step price | not reported in abstract | not reported in abstract | abstract |

First, price levels are the dominant target: nine of the ten studies
forecast a price or index level, and one forecasts direction. As the
cleanest illustration, Chaudhary (2025) forecasts the closing prices of
Apple, Google, Microsoft and Amazon from Yahoo Finance data, using a sliding
window of sixty days of normalised prices, moving averages and sentiment
scores to predict the sixty-first day's close, and reports a mean absolute
percentage error of 2.72% for Apple against ARIMA as the sole comparison.
The previous close is in the feature window, no persistence baseline is
reported, and no significance test is reported. This is the artifact of
§4.1 in its simplest form, on substantially the same data as our own
experiments. Lu et al. (2021) report a coefficient of determination of
0.9804 for next-day closing prices of the Shanghai Composite Index; our own
pre-refactor pipeline reported 0.9627 on the same kind of target, and a
naive zero-change forecast scored 0.9992 on it (§4.1).

Second, trivial baselines are rarely reported. None of the ten studies
reports a naive, persistence, zero-return or majority-class baseline at the
level we examined, including both full texts. The comparison models are
other learners: ARIMA, other networks, or in Fischer and Krauss's case a
rules-based strategy.

Third, significance testing is uncommon. None of the ten reports a
significance test or an uncertainty quantification at the level examined;
of the two full texts, one presents averages over twenty runs without error
bounds and the other names confidence intervals as future work. This
mirrors a finding reported outside finance: in a reproducibility study of
civil war prediction, Kapoor and Narayanan (2023) found that nine of the
twelve papers for which complete code and data were available included no
significance tests or uncertainty quantification for classifier performance
comparison.

## 2.2 Forecast evaluation methodology

The statistical apparatus required to evaluate forecasts rigorously is long
established. Diebold and Mariano (1995) introduced the standard test for
comparing the predictive accuracy of two forecasts. Pesaran and Timmermann
(1992) provided a non-parametric test of directional predictive performance
that accounts for the marginal frequencies of predicted and realised signs,
a correction that matters precisely when a predictor is close to constant.
Campbell and Thompson (2008) formalised the out-of-sample R² against an
expanding historical-mean benchmark, which has since become the standard
measure of return predictability. Harvey, Liu and Zhu (2016) argued for
substantially raised significance thresholds in cross-sectional asset
pricing given the volume of hypotheses tested, and Bailey et al. (2014)
documented the effect of backtest overfitting on out-of-sample performance.

Two results in this literature bear directly on ours. Welch and Goyal
(2008) evaluated a broad set of equity premium predictors out of sample and
found that most fail to outperform the historical average, a result that
reframed the return-predictability literature around the difficulty of
beating a trivial benchmark. Our finding is the deep-learning-era analogue
of theirs: the architectures we evaluate do not outperform the historical
mean at the daily horizon, and the metric by which we establish this is the
one Campbell and Thompson proposed in direct response to Welch and Goyal.

This apparatus is standard in the forecasting and empirical asset pricing
literatures. It is not standard in the deep-learning stock prediction
literature surveyed in §2.1, and that gap is the subject of our paper.

## 2.3 Leakage and reproducibility in machine-learning-based science

Kapoor and Narayanan (2023) surveyed twenty-two papers that identify
pitfalls in the adoption of machine learning across seventeen scientific
fields and found that data leakage collectively affected at least 294
papers, in some cases producing wildly overoptimistic conclusions. They
introduced a taxonomy of eight leakage types organised under three headings:
lack of clean separation between training and test data (L1, comprising no
test set, L1.1; pre-processing on training and test set, L1.2; feature
selection on training and test set, L1.3; and duplicates, L1.4); use of
features that are not legitimate (L2); and a test set not drawn from the
distribution of scientific interest (L3, comprising temporal leakage, L3.1;
nonindependence between training and test samples, L3.2; and sampling bias
in the test distribution, L3.3). Their survey records, separately from
leakage, further categories of pitfall, among them metric choice, a mismatch
between the evaluation metric and the scientific question, which four of
the surveyed papers highlighted. They proposed model info sheets, twenty-one
questions organised by the taxonomy, as a preventive instrument.

Their case study is the closest precedent for the form of our result. In
civil war prediction, complex models were believed to substantially
outperform logistic regression. From a systematic search yielding 124
papers they examined the twelve that predicted civil war with a train-test
split and shared complete code and data, and found errors in four, exactly
the four that claimed superior performance for complex models. When the
errors were corrected, the complex models performed no better than the
logistic regression baseline in each case except one, where the gap in area
under the curve fell from 0.14 to 0.01; and this against logistic
regressions that had been specified as explanatory models rather than tuned
for prediction. They situate this alongside comparable findings for
children's life outcomes and recidivism, and recommend research that
identifies and communicates the limits to prediction in a given domain.

Within finance, Prata et al. (2024) conducted the closest study to ours.
They evaluated fifteen state-of-the-art deep learning models for stock price
trend prediction on limit order book data through an open-source framework,
and reported that all models exhibit a significant performance drop when
exposed to new data, raising questions about their real-world
applicability.

Correct practice is not absent from the recent stock-prediction literature.
Tejas et al. (2026) normalise their inputs with strictly backward-looking
rolling z-scores and protect against leakage with an expanding walk-forward
gap, and report under that protocol that a standalone LSTM and a CNN-LSTM
both fail to outperform their best feed-forward network. We read this as
consistent with the convergence behaviour we document in §7.3, though we
have not verified the authors' own account of the mechanism. Positive
results for machine learning in equity returns are well supported in the
cross-section and at longer horizons (Gu, Kelly and Xiu, 2020); our claim
is specific to the daily single-name horizon, and we state it as such.

## 2.4 Position of this work

Our contribution differs from the three literatures above in the following
respects.

Relative to §2.1, we do not propose an architecture. We evaluate established
ones under a protocol designed to eliminate the specific failure modes we
identify, and we report the difference between what the conventional
protocol reports and what the corrected protocol reports on identical data.

Relative to §2.2, we do not introduce a statistical method. We apply
existing tests to a literature in which they are seldom applied, and we
quantify how much of the reported performance in that literature they
remove.

Relative to §2.3, our position requires more care, and it is the sharpest
way to state our contribution. The leakage we found in our own data layer
and removed, a backward fill of macroeconomic series and a global rescaling,
maps cleanly onto their taxonomy as temporal leakage (L3.1) and
pre-processing across the split (L1.2), and neither is among the five
artifacts of §4, because neither produced a headline number. None of the
five artifacts is leakage under their definition:

- The first, a price-level target evaluated while the previous close is
  available as a feature, involves no illegitimate feature (the previous
  close is available at forecast time) and no contamination of the test
  set. The overoptimism arises from a target and metric specification under
  which a trivial persistence forecast attains a high coefficient of
  determination. Kapoor and Narayanan record metric choice as a category of
  pitfall separate from leakage; this artifact belongs there.
- The second, a cold-started out-of-sample benchmark, falls outside their
  framework entirely. The error resides in the construction of the reference
  quantity rather than in the model, the features or the test distribution.
  To our knowledge this failure mode has not previously been characterised.
- The third, comparison of models evaluated over non-identical test
  windows, is adjacent to sampling bias in the test distribution (L3.3) but
  distinct from it: each model's test set may be individually representative
  while the comparison between them remains invalid. We know of no prior
  characterisation of it either.
- The fourth and fifth, sensitivity of a reported estimate to the
  evaluation-window length and to the random seed, are failures of
  uncertainty quantification. Kapoor and Narayanan document the absence of
  uncertainty quantification as an empirical finding of their case study,
  but it lies outside the leakage taxonomy, and we measure two specific
  mechanisms by which its absence inflates a reported result.

We therefore position this work as an extension of Kapoor and Narayanan's
framework rather than a further illustration of it. All five failure modes
we document produce the same overoptimism as leakage while arising from
error classes outside the leakage taxonomy; two of them, the benchmark
construction and the alignment of comparisons, are to our knowledge
uncharacterised anywhere; and we suggest that a complete account of
reproducibility failure in predictive science must address the construction
of benchmarks, the alignment of comparisons and the stability of reported
estimates alongside the integrity of the train-test split.

Relative to Prata et al. (2024), we differ in data, task and contribution.
They evaluate limit order book models on an intraday trend-classification
task and measure generalisation failure across datasets. We evaluate daily
OHLCV models on an absolute return-regression task and attribute the gap
between reported and corrected performance to five separately measured
causes.

## References

- Bailey, D. H., Borwein, J. M., López de Prado, M., and Zhu, Q. J. (2014).
  Pseudo-mathematics and financial charlatanism: The effects of backtest
  overfitting on out-of-sample performance. *Notices of the American
  Mathematical Society*, 61(5), 458–471. doi:10.1090/noti1105
- Campbell, J. Y., and Thompson, S. B. (2008). Predicting excess stock
  returns out of sample: Can anything beat the historical average? *Review
  of Financial Studies*, 21(4), 1509–1531. doi:10.1093/rfs/hhm055
- Chaudhary, R. (2025). Advanced stock market prediction using long
  short-term memory networks: A comprehensive deep learning framework.
  arXiv:2505.05325.
- Diebold, F. X., and Mariano, R. S. (1995). Comparing predictive accuracy.
  *Journal of Business & Economic Statistics*, 13(3), 253–263.
  doi:10.1080/07350015.1995.10524599
- Fischer, T., and Krauss, C. (2018). Deep learning with long short-term
  memory networks for financial market predictions. *European Journal of
  Operational Research*, 270(2), 654–669. doi:10.1016/j.ejor.2017.11.054
- Gu, S., Kelly, B., and Xiu, D. (2020). Empirical asset pricing via
  machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
  doi:10.1093/rfs/hhaa009
- Harvey, C. R., Liu, Y., and Zhu, H. (2016). ... and the cross-section of
  expected returns. *Review of Financial Studies*, 29(1), 5–68.
  doi:10.1093/rfs/hhv059
- Jin, Z., Yang, Y., and Liu, Y. (2020). Stock closing price prediction
  based on sentiment analysis and LSTM. *Neural Computing and
  Applications*, 32, 9713–9729. doi:10.1007/s00521-019-04504-2
- Kapoor, S., and Narayanan, A. (2023). Leakage and the reproducibility
  crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
  doi:10.1016/j.patter.2023.100804
- Lu, W., Li, J., Li, Y., Sun, A., and Wang, J. (2020). A CNN-LSTM-based
  model to forecast stock prices. *Complexity*, 2020, 6622927.
  doi:10.1155/2020/6622927
- Lu, W., Li, J., Wang, J., and Qin, L. (2021). A CNN-BiLSTM-AM method for
  stock price prediction. *Neural Computing and Applications*, 33(10),
  4741–4753. doi:10.1007/s00521-020-05532-z
- Ouf, S., et al. (2024). A deep learning-based LSTM for stock price
  prediction using Twitter sentiment analysis. *International Journal of
  Advanced Computer Science and Applications*, 15(12).
- Pesaran, M. H., and Timmermann, A. (1992). A simple nonparametric test of
  predictive performance. *Journal of Business & Economic Statistics*,
  10(4), 461–465. doi:10.1080/07350015.1992.10509922
- Prata, M., Masi, G., Berti, L., Arrigoni, V., Coletta, A., Cannistraci,
  I., Vyetrenko, S., Velardi, P., and Bartolini, N. (2024). LOB-based deep
  learning models for stock price trend prediction: A benchmark study.
  *Artificial Intelligence Review*, 57, 116. doi:10.1007/s10462-024-10715-4
- Qian, Y. (2025). An enhanced Transformer framework with incremental
  learning for online stock price prediction. *PLOS ONE*, 20(1), e0316955.
  doi:10.1371/journal.pone.0316955
- Sezer, O. B., Gudelek, M. U., and Ozbayoglu, A. M. (2020). Financial time
  series forecasting with deep learning: A systematic literature review:
  2005–2019. *Applied Soft Computing*, 90, 106181.
  doi:10.1016/j.asoc.2020.106181
- Shafiei Hafshejani, M., and Mansouri, N. (2025). Enhancing stock market
  prediction with LSTM: A review of recent developments and comparative
  analysis. *Archives of Computational Methods in Engineering*, 33(5),
  6177–6274. doi:10.1007/s11831-025-10370-0
- Shah, J., Vaidya, D., and Shah, M. (2022). A comprehensive review on
  multiple hybrid deep learning approaches for stock prediction.
  *Intelligent Systems with Applications*, 16, 200111.
  doi:10.1016/j.iswa.2022.200111
- Tejas, I. P., Goyal, A. K., Mahadeva, R., and Sarda, V. (2026).
  Multi-step stock price prediction using optimized neural network
  architectures: A comprehensive analysis with feature engineering, dynamic
  configuration selection, and hybrid deep learning validation. *Array*,
  30, 100851. doi:10.1016/j.array.2026.100851
- Thakkar, A., and Chaudhari, K. (2021). A comprehensive survey on deep
  neural networks for stock market: The need, challenges, and future
  directions. *Expert Systems with Applications*, 177, 114800.
  doi:10.1016/j.eswa.2021.114800
- Wang, C., Chen, Y., Zhang, S., and Zhang, Q. (2022). Stock market index
  prediction using deep Transformer model. *Expert Systems with
  Applications*, 208, 118128. doi:10.1016/j.eswa.2022.118128
- Welch, I., and Goyal, A. (2008). A comprehensive look at the empirical
  performance of equity premium prediction. *Review of Financial Studies*,
  21(4), 1455–1508. doi:10.1093/rfs/hhm014
- Zhang, Z., Liu, Q., Hu, Y., and Liu, H. (2025). Multi-feature stock price
  prediction by LSTM networks based on VMD and TMFG. *Journal of Big
  Data*, 12, 74. doi:10.1186/s40537-025-01127-4

---

## Verification log (not part of the paper)

Checked on 2026-09-06 against publisher metadata (Crossref), open full
texts, or the sources named. "Full text" means the paper was read;
"abstract" means the publisher's abstract or its verbatim reproduction was
read; "secondary" means the characterisation rests on how other papers or
indexes describe the work.

**Read in full.** Kapoor and Narayanan (2023), from the accepted manuscript:
every number in §2.3 (22 reviews, 17 fields, 294 papers; 124 papers found,
12 examined, errors in 4; "no better ... except Wang", AUC gap 0.14 to 0.01;
9 of 12 with no significance testing; 21 questions; the eight types and the
separate metric-choice category) is quoted from it. Qian (2025), full text
via PMC. Chaudhary (2025), full text via arXiv.

**Abstract.** Fischer and Krauss (2018); Lu et al. (2020); Lu et al. (2021);
Jin et al. (2020); Zhang et al. (2025); Ouf et al. (2024); Prata et al.
(2024); Tejas et al. (2026); Shafiei Hafshejani and Mansouri (2025). All
seven §2.2 citations and the three surveys verified for volume, issue,
pages and DOI at Crossref.

**Secondary only.** Wang et al. (2022): the abstract could not be retrieved
from the publisher or any index; the description of indices, comparison
models and back-testing comes from citing papers. The target-type claim
carries a `[VERIFY]` in the text and the table.

**Corrections to the draft.**
1. Fischer and Krauss's benchmarks are memory-free classifiers and a
   rules-based short-term reversal strategy, not "momentum-based"
   benchmarks.
2. The draft's "both forecast returns or directions rather than price
   levels" held for Fischer and Krauss only; Wang et al. is described as
   index-level prediction. The paragraph now says what each does.
3. Kapoor and Narayanan's corrected complex models performed "no better in
   each case except Wang", not "in any case"; the sentence now carries the
   exception.
4. The draft's novelty claim ("only one of five maps onto the taxonomy;
   four of five are outside it") was inconsistent with its own bullets. The
   position is now: none of the five is leakage; the leakage found (bfill,
   rescaling) is not among the five; artifact 1 is Kapoor and Narayanan's
   separate metric-choice category; artifacts 4 and 5 are failures of
   uncertainty quantification, which they document as a finding but which
   is outside the taxonomy; artifacts 2 and 3 are uncharacterised anywhere.

**Removed.**
- Gülmez (2023), *Expert Systems with Applications* 227, 120346: the
  publisher's page carries a retraction notice ("loss of confidence in the
  results and conclusions"). Not cited.
- "Yañez et al. 2024": no such paper could be identified from the
  reference given. The decomposition strand is covered by Jin et al. (2020)
  and Zhang et al. (2025).
- The second 2026 study (a ten-day cross-sectional outperformance task on
  five indices under embargo and staleness controls, with a deep LSTM
  strongest under leakage-safe settings): it could not be located from the
  description, and the compatibility argument that depended on it is
  withdrawn. If you have the reference, the paragraph in the draft can be
  reinstated once the task description is verified against the paper.
- The clause attributing Tejas et al.'s result to "high-capacity models on
  noisy data": their abstract states the result, not that mechanism.

**Remaining for the author.** Only the two `[VERIFY]` markers on Wang et
al.'s target. Everything else in the section is verified at the level
stated above; the baseline and significance counts in the table are
explicitly counts at the level examined, and the text says so.

**Length.** About 1,900 words of section text including the table, against
the draft's 1,400; the table is what turned three assertions into evidence.
For an eight-page workshop paper, cut the architectural-survey sentence and
the Lu et al. (2021) sentence, and keep the table and §2.4 intact.
