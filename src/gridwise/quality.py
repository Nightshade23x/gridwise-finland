"""Audit completeness of electricity measurements on a UTC 15-minute grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .ingest import DEFAULT_OUTPUT, ROOT

DEFAULT_REPORT = ROOT / "data" / "quality" / "consumption_124_report.json"


def audit_consumption(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Audit observed timestamps against the expected half-open UTC range."""
    if end <= start:
        raise ValueError("End timestamp must follow start timestamp")
    required = {"start_time_utc", "demand_mw"}
    if not required.issubset(frame):
        raise ValueError(f"Missing CSV columns: {sorted(required - set(frame.columns))}")
    times = pd.to_datetime(frame["start_time_utc"], utc=True, errors="raise")
    demand = pd.to_numeric(frame["demand_mw"], errors="coerce")
    expected = pd.date_range(start=start, end=end, freq="15min", inclusive="left", tz="UTC")
    in_range = (times >= start) & (times < end)
    actual = pd.DatetimeIndex(times.loc[in_range].drop_duplicates())
    missing = expected.difference(actual)
    unexpected = actual.difference(expected)
    return {
        "start_utc": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "expected_intervals": int(len(expected)),
        "observed_rows_in_range": int(in_range.sum()),
        "unique_timestamps_in_range": int(len(actual)),
        "completeness_percent": round(100 * (len(expected) - len(missing)) / len(expected), 4),
        "duplicate_timestamps": int(times.duplicated().sum()),
        "null_demand_values": int(demand.isna().sum()),
        "unexpected_timestamps": [t.isoformat() for t in unexpected],
        "missing_timestamps": [t.isoformat() for t in missing],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit 15-minute UTC consumption data")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", help="First UTC day, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", help="Last UTC day, exclusive (YYYY-MM-DD)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if frame.empty:
        raise SystemExit("Cannot audit empty CSV")
    stamps = pd.to_datetime(frame["start_time_utc"], utc=True, errors="raise")
    start = pd.Timestamp(args.start, tz="UTC") if args.start else stamps.min().floor("D")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else stamps.max().ceil("D")
    result = audit_consumption(frame, start, end)
    print(f"Expected intervals: {result['expected_intervals']:,}")
    print(f"Observed rows: {result['observed_rows_in_range']:,}")
    print(f"Completeness: {result['completeness_percent']:.4f}%")
    print(f"Missing intervals: {len(result['missing_timestamps'])}")
    for stamp in result["missing_timestamps"][:15]:
        print(f"  {stamp}")
    if len(result["missing_timestamps"]) > 15:
        print("  ... see full report JSON")
    print(f"Duplicate timestamps: {result['duplicate_timestamps']}")
    print(f"Null demand values: {result['null_demand_values']}")
    print(f"Unexpected/off-grid timestamps: {len(result['unexpected_timestamps'])}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
