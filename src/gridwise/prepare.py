"""Prepare reproducible 15-minute UTC demand observations for model training.

Raw Fingrid CSVs are never changed. Missing targets remain missing.
"""
from __future__ import annotations

from datetime import timedelta

import argparse
import json
from pathlib import Path

import pandas as pd

from .ingest import ROOT

FREQ = timedelta(minutes=15)
REQUIRED = {"start_time_utc", "end_time_utc", "demand_mw"}


def build_grid(
    raw: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_correction_seconds: float = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """Align observed demand onto a UTC grid, tracking small safe timestamp corrections.

    A near-grid time is corrected only if the 15-minute interval has the
    expected duration and there is no exact observation on that grid slot.
    Unknown intervals remain NaN; we do not impute demand here.
    """
    if not REQUIRED.issubset(raw.columns):
        raise ValueError(f"Missing required columns: {sorted(REQUIRED - set(raw.columns))}")
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("Start/end must be timezone-aware and start must precede end")
    if max_correction_seconds < 0 or max_correction_seconds > 60:
        raise ValueError("max_correction_seconds must be between 0 and 60")

    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    raw = raw.copy()
    raw["start_time_utc"] = pd.to_datetime(raw["start_time_utc"], utc=True, errors="raise")
    raw["end_time_utc"] = pd.to_datetime(raw["end_time_utc"], utc=True, errors="raise")
    raw["demand_mw"] = pd.to_numeric(raw["demand_mw"], errors="raise")
    if raw["demand_mw"].isna().any():
        raise ValueError("Raw demand values contain nulls; audit before preparing")
    if raw["start_time_utc"].duplicated().any():
        duplicates = raw.loc[raw["start_time_utc"].duplicated(keep=False)]
        conflict = duplicates.groupby("start_time_utc")["demand_mw"].nunique().gt(1)
        if conflict.any():
            raise ValueError(f"Conflicting duplicate source timestamps: {list(conflict[conflict].index[:5])}")
        raw = raw.drop_duplicates("start_time_utc", keep="first")

    raw = raw.loc[(raw["start_time_utc"] >= start) & (raw["start_time_utc"] < end)].copy()
    expected = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    if not len(expected) or (end - start) % FREQ != timedelta(0):
        raise ValueError("Requested UTC range must be a multiple of 15 minutes")

    original_times = pd.DatetimeIndex(raw["start_time_utc"])
    grid_times = original_times.round("15min")
    offsets = abs((original_times - grid_times).total_seconds())
    durations = raw["end_time_utc"] - raw["start_time_utc"]

    # Exact-grid timestamps take priority over near-grid corrections.
    exact = offsets == 0
    valid_exact = raw.loc[exact & (durations == FREQ)].copy()
    exact_set = set(valid_exact["start_time_utc"])
    used = set(exact_set)
    corrections: list[dict] = []
    excluded: list[dict] = []

    for original, nearest, offset, duration, row in zip(
        original_times, grid_times, offsets, durations, raw.itertuples(index=False), strict=True
    ):
        if offset == 0 and duration == FREQ:
            continue
        if (
            offset <= max_correction_seconds
            and duration == FREQ
            and nearest in expected
            and nearest not in used
        ):
            used.add(nearest)
            corrections.append({
                "start_time_utc": nearest,
                "demand_mw": row.demand_mw,
                "original_timestamp": original.isoformat(),
            })
        else:
            excluded.append({
                "original_timestamp": original.isoformat(),
                "nearest_grid_timestamp": nearest.isoformat(),
                "offset_seconds": float(offset),
                "duration_seconds": float(duration.total_seconds()),
            })

    observed = valid_exact[["start_time_utc", "demand_mw"]].copy()
    observed["is_time_corrected"] = False
    if corrections:
        repaired = pd.DataFrame(corrections)
        repaired["is_time_corrected"] = True
        observed = pd.concat([
            observed,
            repaired[["start_time_utc", "demand_mw", "is_time_corrected"]],
        ], ignore_index=True)

    observed = observed.set_index("start_time_utc").sort_index()
    if observed.index.has_duplicates:
        raise ValueError("Conflicting aligned timestamps; inspect raw input")

    clean = observed.reindex(expected)
    clean.index.name = "timestamp_utc"
    clean["is_observed"] = clean["demand_mw"].notna()
    clean["is_time_corrected"] = clean["is_time_corrected"].eq(True)
    local = clean.index.tz_convert("Europe/Helsinki")
    clean["local_date"] = local.strftime("%Y-%m-%d")
    clean["local_hour"] = local.hour
    clean["local_minute"] = local.minute
    clean["local_day_of_week"] = local.dayofweek
    clean["local_month"] = local.month
    clean["local_utc_offset_hours"] = [int(t.utcoffset().total_seconds() // 3600) for t in local]
    clean = clean.reset_index()

    report = {
        "start_utc": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "expected_intervals": int(len(expected)),
        "observed_intervals": int(clean["is_observed"].sum()),
        "missing_intervals": int((~clean["is_observed"]).sum()),
        "completeness_percent": round(float(100 * clean["is_observed"].mean()), 4),
        "corrected_timestamp_count": len(corrections),
        "corrections": [
            {"original_timestamp": c["original_timestamp"], "aligned_timestamp": c["start_time_utc"].isoformat()}
            for c in corrections
        ],
        "excluded_offgrid_or_invalid_count": len(excluded),
        "excluded_rows": excluded[:30],
        "missing_timestamp_sample": [t.isoformat() for t in expected.difference(pd.DatetimeIndex(observed.index))[:20]],
        "notes": "Raw files unchanged. demand_mw remains null when no source observation is available.",
    }
    return clean, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create aligned 15-minute demand dataset; never impute missing targets")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="One or more Fingrid raw CSV files")
    parser.add_argument("--start", required=True, help="Inclusive UTC day (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Exclusive UTC day (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/consumption_15min.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "data/quality/prepared_report.json")
    parser.add_argument("--repair-offset-seconds", type=float, default=5.0)
    args = parser.parse_args()
    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    clean, report = build_grid(
        raw, pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC"), args.repair_offset_seconds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Expected intervals: {report['expected_intervals']:,}")
    print(f"Observed intervals: {report['observed_intervals']:,}")
    print(f"Missing targets: {report['missing_intervals']:,}")
    print(f"Completeness: {report['completeness_percent']:.4f}%")
    print(f"Safely corrected timestamps: {report['corrected_timestamp_count']}")
    print(f"Excluded off-grid/invalid rows: {report['excluded_offgrid_or_invalid_count']}")
    print(f"Saved prepared CSV: {args.output}")
    print(f"Saved audit report: {args.report}")


if __name__ == "__main__":
    main()
