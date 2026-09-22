"""Fetch the sweep universe once, sequentially, and cache it to disk.

Fetching from inside a process pool would hammer Yahoo with 30 tickers times
four series each from several workers at once, and a rate-limited failure looks
like a missing ticker rather than an error. Fetch once here, then let the sweep
read the cached files.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Sequence


from experiments import DEFAULT_TICKERS, SURVIVORSHIP_WARNING

RAW_DIR = os.path.join("results", "raw")


def raw_path(ticker: str, raw_dir: str = RAW_DIR) -> str:
    """Where one ticker's cached raw CSV lives."""
    return os.path.join(raw_dir, f"{ticker.replace('/', '-')}.csv")


def fetch_universe(
    tickers: Sequence[str] = tuple(DEFAULT_TICKERS),
    start: str = "2010-01-01",
    end: Optional[str] = None,
    raw_dir: str = RAW_DIR,
    overwrite: bool = False,
    pause: float = 0.5,
) -> Dict[str, str]:
    """Fetch each ticker once and cache the engineered frame to CSV.

    Returns a mapping of ticker to cached path. Tickers that fail are reported
    and omitted; a sweep over 30 names should not die because one delisted.
    """
    from fetch_data import fetch_stock_data

    os.makedirs(raw_dir, exist_ok=True)
    print(f"\n{SURVIVORSHIP_WARNING}\n")

    cached: Dict[str, str] = {}
    failures: List[str] = []

    for i, ticker in enumerate(tickers, 1):
        path = raw_path(ticker, raw_dir)
        if os.path.exists(path) and not overwrite:
            print(f"  [{i:>2}/{len(tickers)}] {ticker:<6} cached")
            cached[ticker] = path
            continue

        print(f"  [{i:>2}/{len(tickers)}] {ticker:<6} fetching...", end=" ", flush=True)
        try:
            frame = fetch_stock_data(ticker, start, end)
        except Exception as exc:  # pragma: no cover - network
            frame = None
            print(f"raised {type(exc).__name__}: {exc}")

        if frame is None or frame.empty:
            failures.append(ticker)
            print("FAILED")
        else:
            frame.to_csv(path, index=False)
            cached[ticker] = path
            print(f"{len(frame)} rows -> {path}")

        time.sleep(pause)

    print(f"\nFetched {len(cached)}/{len(tickers)} tickers into {raw_dir!r}")
    if failures:
        print(f"FAILED, omitted from the sweep: {failures}")
    return cached


def main() -> None:
    """CLI: fetch and cache the ticker universe, sequentially."""
    parser = argparse.ArgumentParser(description="Cache the sweep universe")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fetch_universe(args.tickers, args.start, raw_dir=args.raw_dir,
                   overwrite=args.overwrite)


if __name__ == "__main__":
    main()
