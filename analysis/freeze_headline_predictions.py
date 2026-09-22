"""Persist the per-fold predictions of the AAPL headline run.

Runs exactly the README section 3.1 configuration -- docs/frozen_aapl_raw.csv,
12 expanding folds (min_train 1008, test 252, step 252), seed 42, 100 epochs --
through experiments.run_single, and writes the long prediction frame to
results/headline/AAPL__frozen__seed42.parquet. analysis/make_figures.py builds
the AAPL figures from that file, so their fold boundaries are Table 1's, not
the 30-ticker sweep's (whose AAPL fetch cache sits 49 rows earlier).

The file lives outside results/predictions/ so the sweep aggregation, which
globs that directory, never counts it as a 31st run.

    python analysis/freeze_headline_predictions.py     # about an hour on CPU
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from experiments import run_single, save_run  # noqa: E402

FROZEN_CSV = REPO / "docs" / "frozen_aapl_raw.csv"
OUT = REPO / "results" / "headline" / "AAPL__frozen__seed42.parquet"
FOLD_CFG = {"min_train_size": 1008, "test_size": 252, "step_size": 252}
SEED = 42


def frozen_loader(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """The frozen file is the whole history; start/end are ignored, exactly as
    main.py --raw-csv ignores --start."""
    return pd.read_csv(FROZEN_CSV, parse_dates=["Date"])


def main() -> None:
    t0 = time.time()
    print(f"headline run: {FROZEN_CSV.relative_to(REPO)}, folds {FOLD_CFG}, seed {SEED}")
    frame = run_single("AAPL", "full", SEED, load_raw=frozen_loader, **FOLD_CFG)
    if frame is None:
        raise SystemExit("run_single returned no predictions")
    frame["regime"] = "frozen"
    path = save_run(frame, str(OUT))
    print(f"\nwrote {Path(path).relative_to(REPO)}  ({len(frame)} rows, "
          f"{(time.time() - t0) / 60:.0f} min)")
    fold0 = frame[frame["fold_id"] == 0]["target_date"].min()
    print(f"fold-0 test window opens {str(fold0)[:10]}")
    print(frame.groupby("model").size().rename("rows").to_string())


if __name__ == "__main__":
    main()
