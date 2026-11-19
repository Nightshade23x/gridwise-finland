"""Safely retry a missing day's data and merge only newly available observations."""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .fingrid import CONSUMPTION_ID, FingridClient, normalize_consumption, parse_utc_day
from .ingest import DEFAULT_OUTPUT, ROOT


def merge_new(existing: pd.DataFrame, retry: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Preserve existing values and add only newly obtained timestamps."""
    if existing["start_time_utc"].duplicated().any():
        raise ValueError("Existing CSV has duplicated timestamps; inspect before merging")
    additions = retry.loc[~retry.start_time_utc.isin(existing.start_time_utc)].copy()
    if additions.empty:
        return existing, 0
    merged = pd.concat([existing, additions], ignore_index=True)
    merged = merged.sort_values("start_time_utc").reset_index(drop=True)
    return merged, len(additions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry a day and backfill missing Fingrid intervals")
    parser.add_argument("--date", required=True, help="UTC day to retry (YYYY-MM-DD)")
    parser.add_argument("--file", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.file.is_file():
        raise SystemExit(f"Existing dataset not found: {args.file}")
    start = parse_utc_day(args.date)
    end = start + timedelta(days=1)
    existing = pd.read_csv(args.file, parse_dates=["start_time_utc", "end_time_utc"])
    if existing.empty:
        raise SystemExit("Existing dataset is empty; run ingest first")
    load_dotenv(ROOT / ".env")
    client = FingridClient(os.getenv("FINGRID_API_KEY", ""))
    print(f"Retrying Fingrid dataset {CONSUMPTION_ID} for {args.date} (UTC)")
    raw = client.fetch_window(CONSUMPTION_ID, start, end)
    retry = normalize_consumption(raw, start, end)
    merged, count = merge_new(existing, retry)
    print(f"API returned {len(retry)} unique rows; newly recovered: {count}")
    if count == 0:
        print("Original CSV unchanged. Missing values may be absent at source.")
        return
    # Atomic replacement avoids leaving a partially written CSV.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix=".backfill-", dir=args.file.parent, delete=False, newline="", encoding="utf-8") as tmp:
            tmp_path = Path(tmp.name)
            merged.to_csv(tmp, index=False)
        os.replace(tmp_path, args.file)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
    print(f"Saved {len(merged):,} total rows to {args.file}")


if __name__ == "__main__":
    main()
