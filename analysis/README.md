# Analysis scripts

One-shot analyses run against the persisted results, not part of the pipeline.
Each reads from `results/` and prints a report; none of them retrain anything,
so they are cheap to re-run while writing up.

| script | what it answers |
|---|---|
| `cross_ticker_sweep.py` | Is AAPL's null result universal across 30 tickers? Distributions of DA-majority, R2_OOS, PT p-values and alpha t-stats, with Holm correction and a binomial check on how many tickers would "break" the null by chance. |
| `distribution_shift_and_exposure.py` | Does train/test return-distribution shift predict degradation? Regresses per-fold R2_OOS and DA on the fold's KS statistic. Also decomposes the tree's ~50% market exposure. |
| `seed_variance.py` | How large is neural seed noise next to the gap between architectures? Five seeds of BiLSTM and Transformer on AAPL. |
| `alpha_correction_and_window.py` | Holm-corrected alpha p-values, the tree's alpha under transaction cost and execution lag, and the fixture-vs-full-span window artefact. |
| `ridge_grid_and_tree.py` | Where the widened ridge penalty lands per fold, and why the tree's predictions are confined to a fraction of its training target range. |

Prerequisites: `results/predictions/` and `results/seeds/` are committed;
`results/raw/` is a gitignored cache, regenerate with
`python src/fetch_universe.py`.

Every number these produce is one ticker-universe, one fold configuration and
(except `seed_variance.py`) one seed. Read them as diagnostics, not results.
