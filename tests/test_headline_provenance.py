"""The committed headline parquet is the run README Table 1 was computed from.

results/headline/AAPL__frozen__seed42.parquet feeds every AAPL figure. This
pins, on committed files only and without training, that its fold grid is the
frozen CSV's and that the comparison table rebuilt from it equals
docs/baseline_after_refactor.csv, which is Table 1. Both checks live in
make_figures.headline_frames(); a figure cannot be drawn from a parquet that
fails them, and this test fails the suite for the same reason.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))

from make_figures import HEADLINE_DIR, TABLE_1, grid_offset, headline_frames  # noqa: E402


def test_headline_parquet_is_the_table_1_run():
    assert (HEADLINE_DIR / "AAPL__frozen__seed42.parquet").exists()
    assert TABLE_1.exists()

    ds, folds, y_train, primary, window, evaluations, comparison = headline_frames()

    assert len(folds) == 12
    assert set(y_train) == set(range(12))
    # Common window: folds 2..11, the first two produce no meta prediction.
    assert all(len(frame) == 2520 for frame in primary.values())
    assert len(comparison) == len(primary) == 10
    assert str(ds.target_date[folds[0][1][0]])[:10] == "2014-05-29"


def test_sweep_grid_offset_is_stated_from_committed_files():
    o = grid_offset()
    assert o["frozen_open"] == "2014-05-29"
    assert o["sweep_open"] == "2014-03-19"
    assert o["rows"] == 49
