# 9. Reproducibility statement

Every number in this paper was read from a committed file or recomputed
from committed predictions with the committed evaluation code, and each
section names its sources beside its numbers. This section records where
the artifacts are, how each table is regenerated, and what is and is not
reproducible bit for bit. Anything not so recorded is marked
**[unsourced]**.

## 9.1 Commits

| What | Where |
|---|---|
| Repository | `https://github.com/harshitt13/Stock-Market-Prediction-Model`, branch `refactor/returns-pipeline` |
| Results tag | `v1.0-results`, commit `3fa7f8b` (the seed study and the analysis scripts) |
| The 30-ticker sweep | commit `baeca9f` |
| The headline predictions and figures | commit `5e80afb` |
| The code state this text was written against | commit `ead69ec` |
| The pre-refactor run behind the "before" table | commit `3933168` (commit `d26ad59` records the run) |

The paper's section files live under `docs/paper/`, each committed
separately; their commit hashes are in the repository log.

## 9.2 The frozen input

Table 1 and the before/after comparison were computed on one committed
file, and both protocols read the same bytes.

| | |
|---|---|
| File | `docs/frozen_aapl_raw.csv` |
| SHA-256 | `213808b8e187e49a6a0406e5d217428b1ded57848409d42a723945b950a57ac5` (README §9; verified at the time of writing) |
| Shape | 4144 rows × 34 columns, AAPL, 2010-03-16 to 2026-09-03 |
| Provenance | the raw frame the pre-refactor run fetched, taken from that run rather than refetched; a refetch made days later differed by 342 bytes (commit `d26ad59`) |

## 9.3 Table 1 in one command

```bash
git clone https://github.com/harshitt13/Stock-Market-Prediction-Model.git
cd Stock-Market-Prediction-Model
git checkout refactor/returns-pipeline
pip install -r requirements-lock.txt
python src/main.py --ticker AAPL \
    --raw-csv docs/frozen_aapl_raw.csv \
    --min-train 1008 --test-size 252 --step-size 252 --no-plots
```

No network access is needed: the `--raw-csv` flag loads the frozen file
instead of fetching. The run writes `data/model_comparison.csv`, which is
Table 1, together with the full-range, Diebold-Mariano, economics and
aligned-prediction tables (README §9, §11). It took about a hundred minutes
on a single CPU-only machine (README §9), and 103 minutes when we re-ran it
through the experiment runner to persist its predictions (commit
`5e80afb`).

## 9.4 Environment

`requirements.txt` is the unpinned install list; `requirements-lock.txt` is
a `pip freeze` of the virtual environment that produced every committed
result. The versions that matter for the numbers are torch 2.13.0, xgboost
3.4.0, scikit-learn 1.9.0, pandas 3.0.5 and numpy 2.5.1
(`requirements-lock.txt`). The machine was Windows 11, CPU only.
**[unsourced]**: the CPU model and core count are not recorded in any
committed file; the sweep log that would show them is gitignored.

## 9.5 Committed results

Every table in §6 is regenerated from files under `results/` and `docs/`
without retraining, by the scripts under `analysis/` (README §9;
`analysis/README.md`).

| Artifact | Contents | Feeds |
|---|---|---|
| `results/headline/AAPL__frozen__seed42.parquet` | 29,232 per-fold predictions of the headline run, ten models, twelve folds | Table 1 checks, Table 3, the AAPL figures |
| `results/AAPL__frozen__seed42_meta_fits.csv` | the ridge penalty, intercept and coefficients the meta-learner fitted in each of its ten folds, both variants | §5.5, §7.3 |
| `results/predictions/` | 595,224 predictions of the 30-ticker sweep | §6.2, §6.3, the cross-ticker figure |
| `results/per_ticker_model.csv` | 200 rows of per-ticker, per-model metrics with the seeded benchmark | Table 2, §6.2 |
| `results/fold_diagnostics.csv` | 357 fold-ticker rows of train/test return statistics and the KS test | §6.4 |
| `results/fold_aggregates_headline.csv`, `results/fold_aggregates_sweep.csv` | pooled, across-fold mean, sd and median R²_OOS for every headline model and every sweep ticker | Table 1, Table 2, §5.7 |
| `results/grid_vs_data.csv`, `results/grid_vs_data_rawdiff.csv` | the 2×2 of data against fold grid for the AAPL tree, and the sweep cache against the frozen CSV column by column | §8.3 |
| `results/grid_offset_sweep.csv` | the AAPL tree at ten fold-grid offsets on the frozen CSV | §8.3 |
| `results/grid_conditional_tickers.csv` | the tree on five tickers under the sweep grid and the frozen grid | §6.2, §8.3 |
| `results/seeds/` | 30,240 predictions, BiLSTM and Transformer, seeds 0 to 4 | §6.5 |
| `docs/baseline_before_refactor.csv`, `docs/baseline_after_refactor.csv` | the two comparison tables on the frozen file | §4.1, §6.1 |
| `docs/figures/` | thirteen figures | §4, §6 |
| `tests/fixtures/aapl_raw.csv` | 1008 raw rows of AAPL, 2018 to 2021 | the offline test suite; §4.3, §4.4 |

Two cross-checks tie the headline parquet to Table 1, and both run in the
test suite (`tests/test_headline_provenance.py`; `analysis/make_figures.py`,
`headline_frames`): its fold-0 realised returns equal the frozen dataset's,
and the comparison table rebuilt from it equals
`docs/baseline_after_refactor.csv` to the printed digits. A third check ties
it to the pipeline: every model's prediction agreed with a `main.py` run of
the same configuration on every common-window day to 1.0 × 10⁻¹⁶ (commit
`5e80afb`). A fourth ties the meta-fits file to the parquet: refitting both
meta variants from the stored base predictions reproduced the parquet's
meta predictions with zero difference (`analysis/persist_meta_fits.py`).

## 9.6 Regenerating everything else

```bash
python src/fetch_universe.py                          # 30-ticker raw cache, ~3 min, network
python src/experiments.py --parallel --workers 6      # the sweep, ~270 min
python seed_study.py                                  # 5 seeds on AAPL, ~240 min
python analysis/freeze_headline_predictions.py        # headline parquet, ~100 min, no network
python analysis/persist_meta_fits.py                  # meta fits from the parquet, seconds
python analysis/cross_ticker_sweep.py                 # section 6.2
python analysis/alpha_correction_and_window.py        # section 6.3
python analysis/distribution_shift_and_exposure.py    # section 6.4
python analysis/seed_variance.py                      # section 6.5
python analysis/fold_aggregates.py                    # across-fold aggregates, seconds
python analysis/grid_offset_sweep.py                  # grid-offset error bar, ~6 min, no cache
python analysis/grid_vs_data.py                       # data vs grid 2x2, ~3 min, needs results/raw/
python analysis/grid_conditional_tickers.py           # five tickers, two grids, ~6 min, needs results/raw/
python analysis/make_figures.py                       # the thirteen figures
python -m pytest                                      # 331 tests
```

Timings are from README §9 and commit messages (`baeca9f`: 270.2 minutes
wall on six workers; `3fa7f8b`: 240 minutes on five workers). The analysis
scripts that need each fold's training returns on the sweep grid
(`cross_ticker_sweep.py`, `distribution_shift_and_exposure.py`, the sweep
half of `fold_aggregates.py`, `grid_vs_data.py`, `grid_conditional_tickers.py`,
and the seed and KS figures in `make_figures.py`) require `results/raw/`,
which is gitignored (86 MB) and regenerated by the first command; the others
run from committed files alone.

## 9.7 What is not reproducible bit for bit

- **A fresh download is a different vintage.** Adjusted price histories
  are revised; the frozen file differed from a refetch by 342 bytes
  (commit `d26ad59`). The sweep's raw cache is not committed, so a
  regenerated cache will differ from the one behind `results/predictions/`
  by whatever the vendor has revised since; the committed predictions
  themselves do not change.
- **The random forest's averaging order.** Two runs of the same seed on the
  same data differed by at most 1.0 × 10⁻¹⁶ in one prediction, with every
  downstream table byte-identical (commit `148f064`). Every reported digit
  reproduces; the prediction file may not, at the last bit.
- **The cold aggregation path.** `python src/experiments.py --aggregate-only`
  runs `experiments.aggregate`, which evaluates without training returns
  (§4.2). Its output files are not committed and no reported number comes
  from them; the seeded per-ticker table is produced by
  `analysis/cross_ticker_sweep.py`.

## 9.8 Tests

The suite has 331 tests (`python -m pytest --collect-only`; README §7),
all offline. It covers the row contract, truncation invariance of every
feature, six injected leaks that the leakage check must catch, the seeded
benchmark, the common window, the metrics, the figures, and the provenance
of the headline parquet. It does not test model quality, and it cannot.
