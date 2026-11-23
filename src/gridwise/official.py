"""Capture real-time Fingrid forecast *snapshots*, then score them fairly.

Dataset 166's public API returns forecast values without trustworthy historical
issuance identifiers. We preserve only observations actually fetched now and
record their local capture time. Never reconstruct past official issuances from
an API query made after the fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .fingrid import FingridClient
from .refresh import ROOT, LIVE_CSV, atomic_json, read_measurements
from .forward import FORECAST_JSON
from .verify import archive_forecast, _canonical, _score, _stamp, validate_forecast, HORIZON_BUCKETS

OFFICIAL_DATASET_ID = 166
SNAPSHOT_DIR = Path("data/live/official_snapshots")
BENCHMARK_REPORT = Path("data/live/official_benchmark.json")


def validate_official_records(records: list[dict], target_times: pd.DatetimeIndex,
                              captured_at: pd.Timestamp) -> dict[str, float]:
    """Accept unique, future, exact 15-minute official forecast target periods."""
    valid_targets = set(pd.DatetimeIndex(target_times))
    matched: dict[str, float] = {}
    for record in records:
        if not isinstance(record, dict) or any(k not in record for k in ("datasetId", "startTime", "endTime", "value")):
            raise ValueError("Unexpected Fingrid 166 record format")
        if int(record["datasetId"]) != OFFICIAL_DATASET_ID:
            raise ValueError("Unexpected official dataset ID")
        start = _stamp(record["startTime"])
        end = _stamp(record["endTime"])
        if (end - start != pd.Timedelta(minutes=15) or start.minute % 15 != 0 or
            start.second or start.microsecond or start.nanosecond):
            raise ValueError("Invalid official 15-minute period")
        # A forecast obtained after target start is not a valid pre-target prediction.
        if start not in valid_targets or start <= captured_at:
            continue
        value = float(record["value"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid official forecast value")
        key = start.isoformat()
        if key in matched and not np.isclose(matched[key], value, atol=1e-9, rtol=0):
            raise ValueError("Conflicting official revisions in one captured response")
        matched[key] = value
    return matched


def snapshot_path(root: Path, forecast_id: str) -> Path:
    """Exactly one saved official snapshot per GridWise issue."""
    return root / SNAPSHOT_DIR / (forecast_id + ".json")


def snapshot_id(payload: dict) -> str:
    content = _canonical(payload)
    return hashlib.sha256(content).hexdigest()


def capture_snapshot(root: Path, *, now: pd.Timestamp | None = None,
                     client: FingridClient | None = None,
                     max_issue_age_minutes: int = 30) -> dict:
    """Capture official values presently returned for a recent GridWise issuance.

    Never claim the official values were generated at GridWise's issuance time.
    """
    root = root.resolve()
    captured = pd.Timestamp.now(tz="UTC") if now is None else _stamp(str(now))
    latest_path = root / FORECAST_JSON
    if not latest_path.is_file():
        raise FileNotFoundError("Issue GridWise forecast first: python -m gridwise.forward")
    obj = json.loads(latest_path.read_text(encoding="utf-8"))
    ours = validate_forecast(obj)
    issue = _stamp(obj["issued_at_utc"])
    age = (captured - issue).total_seconds() / 60
    if age < 0 or age > max_issue_age_minutes:
        raise ValueError("Latest GridWise forecast is too old (or issued in future); rerun gridwise.forward before capture")
    archived = archive_forecast(root, obj)
    out = snapshot_path(root, archived.stem)
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        validate_snapshot(old)
        if old.get("gridwise_forecast_id") != archived.stem:
            raise ValueError("Existing snapshot belongs to different forecast")
        return old
    if client is None:
        load_dotenv(root / ".env")
        client = FingridClient(os.environ.get("FINGRID_API_KEY", ""))
    start = ours["timestamp_utc"].iloc[0].to_pydatetime()
    end = (ours["timestamp_utc"].iloc[-1] + pd.Timedelta(minutes=15)).to_pydatetime()
    # One API capture at the current time; old target data cannot stand in as
    # a historical snapshot, even if a past date can be queried via the API.
    records = client.fetch_window(OFFICIAL_DATASET_ID, start, end)
    # Conservative: record capture completion and reject targets already begun.
    # Explicit test times remain deterministic for unit tests.
    if now is None:
        captured = pd.Timestamp.now(tz="UTC")
    values = validate_official_records(records, pd.DatetimeIndex(ours["timestamp_utc"]), captured)
    if not values:
        raise ValueError("Fingrid 166 returned no eligible future targets; nothing archived")
    payload = {
        "schema_version": 1,
        "kind": "fingrid_166_capture",
        "gridwise_forecast_id": archived.stem,
        "gridwise_issued_at_utc": issue.isoformat(),
        "captured_at_utc": captured.isoformat(),
        "official_dataset_id": OFFICIAL_DATASET_ID,
        "source_issuance_time_verified": False,
        "points": [{"timestamp_utc": stamp, "official_mw": mw}
                   for stamp, mw in sorted(values.items())],
        "note": "Captured API response, not a source-certified original issuance. Only future target start times retained."
    }
    validate_snapshot(payload)
    atomic_json(out, payload)
    return payload


def validate_snapshot(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("kind") != "fingrid_166_capture":
        raise ValueError("Unknown official snapshot")
    if payload.get("official_dataset_id") != OFFICIAL_DATASET_ID or payload.get("source_issuance_time_verified") is not False:
        raise ValueError("Invalid official snapshot provenance")
    issue, captured = _stamp(payload["gridwise_issued_at_utc"]), _stamp(payload["captured_at_utc"])
    if captured < issue:
        raise ValueError("Capture cannot precede GridWise issuance")
    points = payload.get("points")
    if not isinstance(points, list) or not (1 <= len(points) <= 96):
        raise ValueError("Official snapshot must contain 1–96 target points")
    index = pd.DatetimeIndex([_stamp(p["timestamp_utc"]) for p in points])
    values = np.array([p["official_mw"] for p in points], dtype=float)
    if (index.has_duplicates or not index.is_monotonic_increasing or
        (index <= captured).any() or (index.minute % 15 != 0).any() or
        (index.second != 0).any() or (index.microsecond != 0).any() or
        not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("Invalid official forecast target grid or values")
    return pd.DataFrame({"timestamp_utc": index, "official_mw": values})


def benchmark_one(official: dict, ours: dict, measured: pd.DataFrame,
                  *, now: pd.Timestamp, buffer_minutes: int = 60) -> dict:
    if buffer_minutes < 0:
        raise ValueError("Buffer must not be negative")
    now = _stamp(str(now))
    recorded = validate_snapshot(official)
    gridwise = validate_forecast(ours)
    if _stamp(official["gridwise_issued_at_utc"]) != _stamp(ours["issued_at_utc"]):
        raise ValueError("Official snapshot refers to a different GridWise issue")
    measured = measured[["timestamp_utc", "demand_mw"]].copy()
    measured["timestamp_utc"] = pd.to_datetime(measured["timestamp_utc"], utc=True)
    if measured["timestamp_utc"].isna().any() or measured["timestamp_utc"].duplicated().any():
        raise ValueError("Invalid or duplicate measurement times")
    measured["demand_mw"] = pd.to_numeric(measured["demand_mw"], errors="raise")
    if not np.isfinite(measured["demand_mw"]).all() or (measured["demand_mw"] < 0).any():
        raise ValueError("Invalid measured demand")
    paired = gridwise.merge(recorded, on="timestamp_utc", how="inner", validate="one_to_one")
    paired = paired.merge(measured, on="timestamp_utc", how="left", validate="one_to_one")
    cutoff = now - pd.Timedelta(minutes=buffer_minutes)
    eligible = paired["timestamp_utc"] <= cutoff
    valid = paired.loc[eligible & paired["demand_mw"].notna()]
    observed = valid["demand_mw"].to_numpy()
    model = _score(observed, valid["predicted_mw"].to_numpy())
    historical = _score(observed, valid["baseline_mw"].to_numpy())
    fingrid = _score(observed, valid["official_mw"].to_numpy())
    compared = len(valid)
    all_snap = len(paired)
    status = ("pending" if eligible.sum() == 0 else
              "complete" if compared == all_snap else
              "partial" if eligible.sum() < all_snap else "incomplete_source_data")
    edge = None if not compared or fingrid["mae_mw"] == 0 else round(
        100 * (fingrid["mae_mw"] - model["mae_mw"]) / fingrid["mae_mw"], 2)
    horizons = []
    for lo, hi, label in HORIZON_BUCKETS:
        subset = paired[(paired["target_step"] >= lo) & (paired["target_step"] < hi)]
        matured = subset["timestamp_utc"] <= cutoff
        rows = subset.loc[matured & subset["demand_mw"].notna()]
        o = rows["demand_mw"].to_numpy()
        horizons.append({"range": label, "eligible_targets": int(matured.sum()),
                         "matched_targets": len(rows),
                         "model_mae_mw": _score(o, rows["predicted_mw"].to_numpy())["mae_mw"],
                         "official_mae_mw": _score(o, rows["official_mw"].to_numpy())["mae_mw"]})
    return {
        "gridwise_forecast_id": official["gridwise_forecast_id"],
        "gridwise_issued_at_utc": ours["issued_at_utc"],
        "fingrid_captured_at_utc": official["captured_at_utc"],
        "status": status,
        "official_future_targets": all_snap,
        "eligible_targets": int(eligible.sum()),
        "matched_targets": compared,
        "model": model, "reference_baseline": historical, "fingrid": fingrid,
        "model_mae_improvement_vs_fingrid_percent": edge,
        "by_horizon": horizons,
    }


def run_benchmark(root: Path, *, now: pd.Timestamp | None = None, buffer_minutes: int = 60) -> dict:
    root = root.resolve()
    clock = pd.Timestamp.now(tz="UTC") if now is None else _stamp(str(now))
    if buffer_minutes < 0:
        raise ValueError("Negative measurement availability buffer")
    snapshot_dir = root / SNAPSHOT_DIR
    from .verify import archived_forecasts
    forecast_lookup = dict(archived_forecasts(root))
    results: list[dict] = []
    if snapshot_dir.exists():
        for path in sorted(snapshot_dir.glob("*.json")):
            snap = json.loads(path.read_text(encoding="utf-8"))
            validate_snapshot(snap)
            fid = snap["gridwise_forecast_id"]
            if path.name != f"{fid}.json":
                raise ValueError("Snapshot file name does not match associated issuance")
            if fid not in forecast_lookup:
                raise ValueError("Official snapshot is missing its matching archived GridWise forecast")
            if not (root / LIVE_CSV).exists():
                raise FileNotFoundError("Refresh measured demand first")
            measured = read_measurements(root / LIVE_CSV)
            results.append(benchmark_one(snap, forecast_lookup[fid], measured, now=clock, buffer_minutes=buffer_minutes))
    results.sort(key=lambda v: (v["gridwise_issued_at_utc"], v["gridwise_forecast_id"]), reverse=True)
    report = {
        "schema_version": 1, "status": "ok" if results else "no_snapshots",
        "verified_at_utc": clock.isoformat(), "availability_buffer_minutes": buffer_minutes,
        "snapshot_count": len(results), "comparisons": results,
        "note": "Fingrid 166 snapshots are captured API responses, not verified source issuance history; model and official scores use the same eligible future targets and measured readings."
    }
    atomic_json(root / BENCHMARK_REPORT, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Capture and benchmark Fingrid 166 against issued GridWise 24-hour forecasts")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture", help="Fetch and archive official forecast soon after GridWise issuance")
    c.add_argument("--max-issue-age-minutes", type=int, default=30)
    v = sub.add_parser("verify", help="Score captured official forecasts against newly refreshed measurements")
    v.add_argument("--availability-buffer-minutes", type=int, default=60)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "capture":
        obj = capture_snapshot(args.root, max_issue_age_minutes=args.max_issue_age_minutes)
        print(f"Captured Fingrid dataset 166 at {obj['captured_at_utc']}")
        print(f"Archived {len(obj['points'])}/96 targets that were future at API capture time")
        print("Capture is not a source-certified official issuance. No historical forecast reconstruction performed.")
    else:
        obj = run_benchmark(args.root, buffer_minutes=args.availability_buffer_minutes)
        print(f"Saved {obj['snapshot_count']} snapshot comparisons to {args.root / BENCHMARK_REPORT}")
        for entry in obj["comparisons"]:
            print(f"  {entry['gridwise_issued_at_utc']}: {entry['status']}; "
                  f"{entry['matched_targets']}/{entry['official_future_targets']} measured")
            if entry["matched_targets"]:
                print(f"    GridWise MAE {entry['model']['mae_mw']:.1f} MW; Fingrid MAE {entry['fingrid']['mae_mw']:.1f} MW")


if __name__ == "__main__":
    main()
