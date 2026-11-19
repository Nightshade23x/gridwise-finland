"""Download historical Finland electricity consumption."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .fingrid import CONSUMPTION_ID, FingridClient, normalize_consumption, parse_utc_day, windows

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "fingrid_consumption_124.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Fingrid electricity demand as a clean CSV")
    parser.add_argument("--start", default="2026-10-01", help="First UTC day, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-11-01", help="Last UTC day, exclusive (YYYY-MM-DD)")
    parser.add_argument("--chunk-days", type=int, default=7, help="Days per API query (default: 7)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    start, end = parse_utc_day(args.start), parse_utc_day(args.end)
    spans = list(windows(start, end, args.chunk_days))
    load_dotenv(ROOT / ".env")
    key = os.getenv("FINGRID_API_KEY", "")
    client = FingridClient(key)

    records = []
    for idx, (a, b) in enumerate(spans, 1):
        print(f"[{idx}/{len(spans)}] Fetching {a.date()} to {b.date()} (UTC)", flush=True)
        batch = client.fetch_window(CONSUMPTION_ID, a, b)
        print(f"  Received {len(batch)} observations", flush=True)
        records.extend(batch)

    frame = normalize_consumption(records, start, end)
    if frame.empty:
        raise RuntimeError("No observations returned; output file not modified")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame):,} unique observations to {args.output}")
    print(f"Period: {frame.start_time_utc.min()} through {frame.start_time_utc.max()}")
    print(f"Demand mean: {frame.demand_mw.mean():,.1f} MW")
    print(f"Null values: {int(frame.isna().sum().sum())}")
    deltas = frame.start_time_utc.diff().dropna()
    gaps = int((deltas.dt.total_seconds() > 15 * 60).sum())
    print(f"Gaps longer than 15 minutes: {gaps} (review before ML)")


if __name__ == "__main__":
    main()
