"""Phase 22: point-in-time shadow issuances and later, paired verification.

Run `python -m gridwise.level_shift_shadow issue` to archive an *unchanged*
reference and its level-shift-corrected alternative. `verify` scores existing
snapshots against subsequent measurements; neither command alters operational
GridWise, Fingrid captures, or the public historical demo.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .forward import combined_demand, utc_timestamp
from .level_shift import (BUFFER_MINUTES, OFFSET_CAP_MW, ANCHOR_HOURS,
                          REPORT as TUNING_REPORT, live_preview)
from .refresh import ROOT, LIVE_CSV, atomic_json, read_measurements
from .shadow_report_schema import validate_report

ARCHIVE = Path("data/live/level_shift_shadow_archive")
REPORT = Path("data/live/level_shift_shadow_verification.json")
STEPS = 96
BUCKETS = ((0, 24, "0–6 h"), (24, 48, "6–12 h"),
           (48, 72, "12–18 h"), (72, 96, "18–24 h"))


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def selected_parameters(root: Path) -> tuple[dict, str]:
    """Freeze a pre-calibrated strategy; never reselect on today's outcomes."""
    path = root / TUNING_REPORT
    if not path.is_file():
        raise FileNotFoundError("Phase 21 selection missing; run python -m gridwise.level_shift first")
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema_version") != 1 or doc.get("status") != "evaluated":
        raise ValueError("Invalid Phase 21 selection report")
    p = doc["selected_parameters"]
    if (p.get("baseline") not in ("lag_48h", "lag_7d", "blend_50_50") or
            float(p.get("alpha", -1)) not in (0, .25, .5, .75, 1.) or
            float(p.get("decay_hours", -1)) not in (6., 24.) or
            float(p.get("offset_cap_mw", -1)) != OFFSET_CAP_MW or
            int(p.get("anchor_hours", -1)) != ANCHOR_HOURS or
            int(p.get("measurement_buffer_minutes", -1)) != BUFFER_MINUTES or
            doc["selection_period"].get("start_utc", "")[:7] != "2026-01" or
            doc["selection_period"].get("end_exclusive_utc", "")[:7] != "2026-03"):
        raise ValueError("Phase 21 selected parameters differ from calibrated strategy")
    return {"baseline": p["baseline"], "alpha": float(p["alpha"]),
            "decay_hours": float(p["decay_hours"]),
            "offset_cap_mw": float(p["offset_cap_mw"]),
            "anchor_hours": int(p["anchor_hours"]),
            "measurement_buffer_minutes": int(p["measurement_buffer_minutes"])}, hashlib.sha256(raw).hexdigest()


def validate_snapshot(snapshot: dict) -> pd.DataFrame:
    """Reject fabricated, malformed, nonfuture or non-grid predictions."""
    if (not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1 or
            snapshot.get("kind") != "level_shift_shadow" or snapshot.get("status") != "issued"):
        raise ValueError("Invalid shadow snapshot metadata")
    issue = utc_timestamp(snapshot["issued_at_utc"])
    source = utc_timestamp(snapshot["source_latest_observation_utc"])
    if source > issue - timedelta(minutes=BUFFER_MINUTES):
        raise ValueError("Source observations violate the availability buffer")
    if snapshot.get("resolution_minutes") != 15 or snapshot.get("horizon_steps") != STEPS:
        raise ValueError("Shadow snapshot has invalid grid metadata")
    params = snapshot["parameters"]
    if (params.get("baseline") not in ("lag_48h", "lag_7d", "blend_50_50") or
            not (0 <= float(params["alpha"]) <= 1) or float(params["decay_hours"]) <= 0 or
            float(params.get("offset_cap_mw", -1)) != OFFSET_CAP_MW or
            int(params.get("measurement_buffer_minutes", -1)) != BUFFER_MINUTES or
            int(params.get("anchor_hours", -1)) != ANCHOR_HOURS or
            not isinstance(snapshot.get("calibration_report_sha256"), str) or
            len(snapshot["calibration_report_sha256"]) != 64):
        raise ValueError("Invalid shadow calibration provenance")
    points = snapshot.get("points")
    if not isinstance(points, list) or len(points) != STEPS:
        raise ValueError("Exactly 96 shadow targets required")
    stamps = pd.DatetimeIndex([utc_timestamp(p["timestamp_utc"]) for p in points])
    if (stamps.has_duplicates or stamps[0] <= issue or
            not (stamps[1:] - stamps[:-1] == timedelta(minutes=15)).all() or
            (stamps.minute % 15 != 0).any() or (stamps.second != 0).any() or
            (stamps.microsecond != 0).any()):
        raise ValueError("Shadow targets must be strictly future 15-minute intervals")
    expected_first = issue.floor("15min") + timedelta(minutes=15)
    if stamps[0] != expected_first:
        raise ValueError("Shadow timeline must start with the first future quarter hour")
    arr = np.array([[p["uncorrected_mw"], p["corrected_mw"]] for p in points], dtype=float)
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("Shadow predictions must be finite and nonnegative")
    return pd.DataFrame({"timestamp_utc": stamps,
                         "uncorrected_mw": arr[:, 0], "corrected_mw": arr[:, 1]})


def build_snapshot(demand: pd.DataFrame, issue_time, parameters: dict,
                   calibration_sha256: str) -> dict:
    issue = utc_timestamp(issue_time)
    preview = live_preview(demand, issue, parameters)
    doc = {"schema_version": 1, "kind": "level_shift_shadow", "status": "issued",
           "issued_at_utc": issue.isoformat(),
           "source_latest_observation_utc": preview["source_latest_observation_utc"],
           "resolution_minutes": 15, "horizon_steps": STEPS,
           "parameters": parameters, "calibration_report_sha256": calibration_sha256,
           "anchor_count": preview["anchor_count"], "offset_mw": preview["offset_mw"],
           "points": [{"timestamp_utc": item["timestamp_utc"],
                       "uncorrected_mw": item["uncorrected_mw"],
                       "corrected_mw": item["candidate_mw"]}
                      for item in preview["points"]],
           "note": "Research-only point-in-time shadow issue; no changes to operational forecasts."}
    validate_snapshot(doc)
    return doc


def archive_snapshot(root: Path, payload: dict) -> Path:
    """Content-addressed creation; never overwrite an earlier issuance."""
    validate_snapshot(payload)
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    issued = utc_timestamp(payload["issued_at_utc"])
    basename = f"{issued.strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:20]}.json"
    path = root / ARCHIVE / basename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if _canonical(json.loads(path.read_text(encoding="utf-8"))) != _canonical(payload):
            raise ValueError("Archived issuance content conflicts with its filename")
        return path
    with os.fdopen(fd, "wb") as file:
        file.write(_canonical(payload))
        file.flush()
        os.fsync(file.fileno())
    return path


def archived_snapshots(root: Path) -> list[tuple[str, dict]]:
    snapshots = []
    for path in sorted((root / ARCHIVE).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_snapshot(payload)
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        issued = utc_timestamp(payload["issued_at_utc"])
        name = f"{issued.strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:20]}.json"
        if path.name != name:
            raise ValueError(f"Shadow archive content changed: {path.name}")
        snapshots.append((digest, payload))
    return snapshots


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    if len(actual) == 0:
        return {"paired_rows": 0, "mae_mw": None, "rmse_mw": None, "bias_mw": None}
    error = predicted - actual
    return {"paired_rows": int(len(error)),
            "mae_mw": round(float(np.mean(np.abs(error))), 3),
            "rmse_mw": round(float(np.sqrt(np.mean(error ** 2))), 3),
            "bias_mw": round(float(np.mean(error)), 3)}


def verify_snapshot(digest: str, snapshot: dict, measured: pd.DataFrame,
                    *, as_of, buffer_minutes: int = BUFFER_MINUTES) -> dict:
    if buffer_minutes < 0:
        raise ValueError("Availability buffer must be nonnegative")
    now = utc_timestamp(as_of)
    targets = validate_snapshot(snapshot)
    if not {"timestamp_utc", "demand_mw"}.issubset(measured.columns):
        raise ValueError("Missing demand measurements")
    actual = measured[["timestamp_utc", "demand_mw"]].copy()
    actual["timestamp_utc"] = pd.to_datetime(actual["timestamp_utc"], utc=True, errors="raise")
    actual["demand_mw"] = pd.to_numeric(actual["demand_mw"], errors="raise")
    if (actual["timestamp_utc"].isna().any() or actual["timestamp_utc"].duplicated().any() or
            (actual["demand_mw"].notna() &
             (~np.isfinite(actual["demand_mw"]) | (actual["demand_mw"] < 0))).any()):
        raise ValueError("Invalid measured demand values")
    joined = targets.merge(actual, how="left", on="timestamp_utc", validate="one_to_one")
    # Interval-start timestamp: wait 15 minutes for period completion plus source buffer.
    cutoff = now - timedelta(minutes=15 + buffer_minutes)

    def paired_scores(group: pd.DataFrame) -> tuple[dict, dict, int]:
        eligible = group["timestamp_utc"].le(cutoff)
        paired = group.loc[eligible & group["demand_mw"].notna()]
        truth = paired["demand_mw"].to_numpy(dtype=float)
        return (score(truth, paired["uncorrected_mw"].to_numpy(dtype=float)),
                score(truth, paired["corrected_mw"].to_numpy(dtype=float)), int(eligible.sum()))

    original, corrected, eligible = paired_scores(joined)
    matched = original["paired_rows"]
    status = ("pending" if eligible == 0 else
              "complete" if matched == STEPS else
              "partial" if eligible < STEPS else "incomplete_source_data")
    buckets = []
    for start, stop, label in BUCKETS:
        o, c, e = paired_scores(joined.iloc[start:stop])
        buckets.append({"range": label, "eligible_targets": e,
                        "matched_targets": o["paired_rows"], "total_targets": stop - start,
                        "uncorrected": o, "corrected": c})
    reduction = None if original["mae_mw"] is None or not original["mae_mw"] else round(
        100 * (original["mae_mw"] - corrected["mae_mw"]) / original["mae_mw"], 2)
    return {"forecast_id": digest[:20], "snapshot_sha256": digest,
            "issued_at_utc": utc_timestamp(snapshot["issued_at_utc"]).isoformat(),
            "source_latest_observation_utc": snapshot["source_latest_observation_utc"],
            "reference": snapshot["parameters"]["baseline"],
            "status": status, "total_targets": STEPS,
            "eligible_targets": eligible, "matched_targets": matched,
            "missing_eligible_targets": eligible - matched,
            "uncorrected": original, "corrected": corrected,
            "mae_reduction_percent": reduction, "by_horizon": buckets}


def run_verification(root: Path, *, as_of=None, buffer_minutes: int = BUFFER_MINUTES) -> dict:
    root = root.resolve()
    now = pd.Timestamp.now(tz="UTC") if as_of is None else utc_timestamp(as_of)
    snapshots = archived_snapshots(root)
    if snapshots and not (root / LIVE_CSV).is_file():
        raise FileNotFoundError("Recent demand unavailable; run python -m gridwise.refresh first")
    measured = (read_measurements(root / LIVE_CSV) if snapshots else
                pd.DataFrame(columns=["timestamp_utc", "demand_mw"]))
    results = [verify_snapshot(digest, snap, measured, as_of=now, buffer_minutes=buffer_minutes)
               for digest, snap in snapshots]
    results.sort(key=lambda x: (x["issued_at_utc"], x["forecast_id"]), reverse=True)
    output = {"schema_version": 1, "status": "ok" if results else "no_issuances",
              "verified_at_utc": now.isoformat(), "availability_buffer_minutes": buffer_minutes,
              "issuance_count": len(results), "issuances": results,
              "notes": ["Shadow-only: no change to issued GridWise forecasts or official captures.",
                        "Both strategies use exactly the same measured target timestamps.",
                        "A target is eligible only after its quarter-hour ends plus the availability buffer.",
                        "Repeated or overlapping issuances are not independent evaluations.",
                        "The first complete 24h forecast needs later published measurements."]}
    validate_report(output)
    atomic_json(root / REPORT, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue or verify research-only level-shift shadow forecasts")
    parser.add_argument("action", choices=("issue", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "issue":
        params, digest = selected_parameters(root)
        doc = build_snapshot(combined_demand(root), pd.Timestamp.now(tz="UTC"), params, digest)
        saved = archive_snapshot(root, doc)
        print(f"Issued 96 paired shadow targets at {doc['issued_at_utc']}")
        print(f"Source observation: {doc['source_latest_observation_utc']}; "
              f"anchor observations: {doc['anchor_count']}; offset: {doc['offset_mw']:+.1f} MW")
        print(f"Saved immutable snapshot: {saved}")
        print("Research-only candidate. Existing forward forecasts and Fingrid archives unchanged.")
    else:
        result = run_verification(root)
        print(f"Archived shadow issuances: {result['issuance_count']} | verified {result['verified_at_utc']}")
        for row in result["issuances"]:
            print(f"  {row['issued_at_utc']} | {row['status']} | {row['matched_targets']}/96 measured")
            if row["matched_targets"]:
                print(f"    Original MAE: {row['uncorrected']['mae_mw']:.1f} MW; "
                      f"Corrected MAE: {row['corrected']['mae_mw']:.1f} MW; "
                      f"change: {row['mae_reduction_percent']!s}%")
        print(f"Saved: {root / REPORT}")
        print("No download, prediction issuance, or model retraining occurred during verification.")


if __name__ == "__main__":
    main()
