"""Incrementally refresh recent Fingrid measured demand (not a forecast).

Separate from historical training/backtest files. A process lock and atomic
file replacement protect persisted observations when the network fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .fingrid import CONSUMPTION_ID, FingridClient, normalize_consumption, windows

ROOT = Path(__file__).resolve().parents[2]
LIVE_CSV = Path("data/live/consumption_recent.csv")
STATUS_JSON = Path("data/live/refresh_status.json")
INIT_START = datetime(2026, 11, 1, tzinfo=timezone.utc)
SCHEMA = ("timestamp_utc", "demand_mw")


def utc_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Timestamp must be timezone aware")
    return value.astimezone(timezone.utc)


def fetch_cutoff(now: datetime, buffer_minutes: int) -> datetime:
    """Only request completed quarters, with a configurable safety buffer."""
    if buffer_minutes < 0:
        raise ValueError("Buffer must be nonnegative")
    timestamp = pd.Timestamp(utc_time(now) - timedelta(minutes=buffer_minutes))
    return timestamp.floor("15min").to_pydatetime()


def read_measurements(path: Path) -> pd.DataFrame:
    """Validate the entire saved store before merging anything."""
    if not path.exists():
        return pd.DataFrame(columns=list(SCHEMA))
    frame = pd.read_csv(path)
    if list(frame.columns) != list(SCHEMA):
        raise ValueError("Unexpected recent demand CSV schema")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    frame["demand_mw"] = pd.to_numeric(frame["demand_mw"], errors="raise")
    stamps = frame["timestamp_utc"]
    if (stamps.isna().any() or stamps.duplicated().any() or
        (stamps.dt.minute % 15 != 0).any() or (stamps.dt.second != 0).any() or
        (stamps.dt.microsecond != 0).any() or (stamps.dt.nanosecond != 0).any() or
        (~np.isfinite(frame["demand_mw"])).any() or (frame["demand_mw"] < 0).any()):
        raise ValueError("Recent demand CSV contains duplicate, invalid or off-grid measurements")
    return frame.sort_values("timestamp_utc").reset_index(drop=True)


def valid_new_rows(records: list[dict], start: datetime, end: datetime) -> tuple[pd.DataFrame, int]:
    """Reject off-grid/partial readings; never fabricate or interpolate values."""
    frame = normalize_consumption(records, start, end)
    if frame.empty:
        return pd.DataFrame(columns=list(SCHEMA)), 0
    start_time = frame["start_time_utc"]
    duration = (frame["end_time_utc"] - start_time).dt.total_seconds()
    valid = (
        (start_time.dt.minute % 15 == 0) &
        (start_time.dt.second == 0) &
        (start_time.dt.microsecond == 0) &
        (start_time.dt.nanosecond == 0) &
        (duration == 900) &
        (frame["end_time_utc"] <= end) &
        np.isfinite(frame["demand_mw"]) &
        (frame["demand_mw"] >= 0)
    )
    return (
        frame.loc[valid, ["start_time_utc", "demand_mw"]]
        .rename(columns={"start_time_utc": "timestamp_utc"})
        .reset_index(drop=True),
        int((~valid).sum()),
    )


def merge_measurements(old: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Keep original values at colliding timestamps; recover absent timestamps."""
    old_series = old.set_index("timestamp_utc")["demand_mw"].astype(float)
    incoming_series = incoming.set_index("timestamp_utc")["demand_mw"].astype(float)
    overlap = incoming_series.index.intersection(old_series.index)
    conflict = int((~np.isclose(
        incoming_series.loc[overlap].to_numpy(),
        old_series.loc[overlap].to_numpy(), atol=1e-6, rtol=0,
    )).sum())
    added_series = incoming_series.loc[~incoming_series.index.isin(old_series.index)]
    combined = (pd.concat([old_series, added_series]) if len(old_series) and len(added_series)
                else (added_series if old_series.empty else old_series)).sort_index()
    frame = combined.rename("demand_mw").rename_axis("timestamp_utc").reset_index()
    return frame, int(len(added_series)), conflict


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".csv.tmp", prefix=".gridwise-", dir=path.parent,
                                         encoding="utf-8", newline="", delete=False) as f:
            tmp = f.name
            frame.to_csv(f, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json.tmp", prefix=".gridwise-", dir=path.parent,
                                         encoding="utf-8", delete=False) as f:
            tmp = f.name
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def read_status(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def recent_completeness(frame: pd.DataFrame, end: datetime, *, since: datetime, days: int = 7) -> dict:
    start = max(since, end - timedelta(days=days))
    expected = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    observed = pd.DatetimeIndex(frame.loc[
        (frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] < end), "timestamp_utc"
    ])
    missing = expected.difference(observed)
    return {
        "start_utc": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "expected_slots": int(len(expected)),
        "available_slots": int(len(expected) - len(missing)),
        "missing_slots": int(len(missing)),
        "completeness_percent": round(100 * (1 - len(missing) / len(expected)), 2) if len(expected) else None,
    }


def run_refresh(
    root: Path,
    client: FingridClient,
    now: datetime,
    *,
    initial_start: datetime = INIT_START,
    overlap_days: int = 2,
    buffer_minutes: int = 60,
    chunk_days: int = 7,
) -> dict:
    """Run one refresh. client is injectable for deterministic offline tests."""
    if overlap_days < 1 or chunk_days < 1:
        raise ValueError("overlap_days and chunk_days must be >= 1")
    initial_start = utc_time(initial_start)
    end = fetch_cutoff(now, buffer_minutes)
    csv_path = root / LIVE_CSV
    status_path = root / STATUS_JSON
    lock_path = root / "data/live/refresh.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # EXCL prevents two scheduled jobs from writing concurrently.
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Another refresh may be running; check data/live/refresh.lock") from exc
    try:
        os.close(fd)
        prior = read_status(status_path)
        attempt = utc_time(now).isoformat()
        try:
            old = read_measurements(csv_path)
            if not old.empty:
                last = old["timestamp_utc"].max().to_pydatetime()
                start = max(initial_start, (last - timedelta(days=overlap_days)).replace(
                    hour=0, minute=0, second=0, microsecond=0,
                ))
            else:
                start = initial_start
            if start >= end:
                raise ValueError("No completed 15-minute period available yet; try later")
            records: list[dict] = []
            for a, b in windows(start, end, chunk_days=chunk_days):
                print(f"Fetching {a.isoformat()} to {b.isoformat()} (exclusive)", flush=True)
                records.extend(client.fetch_window(CONSUMPTION_ID, a, b))
            newer, rejected = valid_new_rows(records, start, end)
            if newer.empty and old.empty:
                raise RuntimeError("No valid Fingrid measurements returned; existing files unchanged")
            merged, new_count, conflicts = merge_measurements(old, newer)
            if new_count or not csv_path.exists():
                atomic_csv(csv_path, merged)
            latest = merged["timestamp_utc"].max().isoformat() if not merged.empty else None
            summary = {
                "status": "ok",
                "dataset_id": CONSUMPTION_ID,
                "source": "https://data.fingrid.fi/en/datasets/124",
                "last_attempt_utc": attempt,
                "last_success_utc": attempt,
                "latest_measurement_utc": latest,
                "latest_demand_mw": float(merged["demand_mw"].iloc[-1]) if not merged.empty else None,
                "requested_start_utc": start.isoformat(),
                "requested_end_utc_exclusive": end.isoformat(),
                "received_rows": len(records),
                "valid_rows_received": len(newer),
                "added_rows": new_count,
                "existing_values_preserved_on_conflict": conflicts,
                "rejected_offgrid_or_invalid_rows": rejected,
                "stored_rows": len(merged),
                "last_7_days": recent_completeness(merged, end, since=initial_start),
                "error": None,
            }
            atomic_json(status_path, summary)
            return summary
        except Exception as exc:
            failure = {**prior,
                "status": "failed",
                "dataset_id": CONSUMPTION_ID,
                "last_attempt_utc": attempt,
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
            atomic_json(status_path, failure)
            raise
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh recent measured Fingrid demand without changing ML history")
    parser.add_argument("--initial-start", default="2026-11-01", help="UTC date used for the first refresh")
    parser.add_argument("--buffer-minutes", type=int, default=60,
                        help="Safety buffer for data availability (default 60)")
    parser.add_argument("--overlap-days", type=int, default=2,
                        help="Recheck the last N days for previously missing records (default 2)")
    args = parser.parse_args()
    root = Path(os.getenv("GRIDWISE_PROJECT_ROOT", str(ROOT))).resolve()
    load_dotenv(root / ".env")
    key = os.getenv("FINGRID_API_KEY", "")
    if not key:
        parser.error("FINGRID_API_KEY is required; place it in the project's .env")
    client = FingridClient(key)
    report = run_refresh(
        root, client, datetime.now(timezone.utc),
        initial_start=datetime.strptime(args.initial_start, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        overlap_days=args.overlap_days,
        buffer_minutes=args.buffer_minutes,
    )
    print(f"Refresh successful: {report['added_rows']} new measurements; "
          f"{report['stored_rows']} total; latest {report['latest_measurement_utc']}")
    print(f"Source gaps in last 7 days: {report['last_7_days']['missing_slots']}")


if __name__ == "__main__":
    main()
