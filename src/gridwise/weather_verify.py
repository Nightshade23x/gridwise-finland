"""Phase 17: verify archived weather-aware forward forecasts against later measurements.

No new forecast, network call, or retraining occurs here. Each 96-point
issuance is evaluated independently using a shared set of real observations.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .refresh import ROOT, LIVE_CSV, atomic_json, read_measurements
from .weather_forward import ARCHIVE, REPORT as LATEST_WEATHER
from .forward import HORIZON_STEPS

OUTPUT = Path("data/live/weather_forecast_verification.json")
BUFFER_MINUTES = 60
BUCKETS = ((0, 24, "0–6 h"), (24, 48, "6–12 h"),
           (48, 72, "12–18 h"), (72, 96, "18–24 h"))
MODELS = ("demand_weather", "demand_only", "baseline", "preferred")
PREDICTION_KEYS = {"demand_weather": "demand_weather_mw",
                   "demand_only": "demand_only_mw", "baseline": "baseline_mw",
                   "preferred": "preferred_mw"}
VALID_BASELINES = {"lag_48h", "lag_7d", "blend_50_50"}


def _timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamps must contain an explicit timezone")
    return stamp.tz_convert("UTC")


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_snapshot(payload: dict) -> pd.DataFrame:
    """Validate issuer, 96 unique future targets and all candidate columns."""
    if not isinstance(payload, dict) or payload.get("kind") != "weather_forward_candidate" or payload.get("status") != "ok":
        raise ValueError("Not an issued weather-forward candidate")
    if payload.get("schema_version") != 1 or payload.get("resolution_minutes") != 15:
        raise ValueError("Unsupported candidate schema or resolution")
    issued = _timestamp(payload["issued_at_utc"])
    raw = payload.get("points")
    if not isinstance(raw, list) or len(raw) != HORIZON_STEPS:
        raise ValueError("Weather candidate must contain exactly 96 targets")
    stamps = pd.DatetimeIndex([_timestamp(item["timestamp_utc"]) for item in raw])
    if (stamps.has_duplicates or stamps[0] <= issued or
            not (stamps[1:] - stamps[:-1] == timedelta(minutes=15)).all() or
            (stamps.minute % 15 != 0).any() or (stamps.second != 0).any() or
            (stamps.microsecond != 0).any() or (stamps.nanosecond != 0).any()):
        raise ValueError("Invalid, non-future or non-quarter-hour forecast timestamps")
    if _timestamp(payload["first_target_utc"]) != stamps[0] or _timestamp(payload["last_target_utc"]) != stamps[-1]:
        raise ValueError("Candidate target metadata disagrees with points")
    choice = payload.get("preferred_strategy")
    baseline = payload.get("baseline_name")
    if baseline not in VALID_BASELINES or choice not in {"demand_only", "demand_weather", baseline}:
        raise ValueError("Unknown preferred strategy or historical baseline")
    arr = np.asarray([[point[PREDICTION_KEYS[name]] for name in MODELS] for point in raw], dtype=float)
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("Predictions must be finite and nonnegative")
    expected = arr[:, {"demand_weather": 0, "demand_only": 1}.get(choice, 2)]
    if not np.allclose(arr[:, 3], expected, rtol=0, atol=0.002):
        raise ValueError("Preferred prediction differs from validation-selected strategy")
    return pd.DataFrame({"timestamp_utc": stamps, **{
        PREDICTION_KEYS[name]: arr[:, i] for i, name in enumerate(MODELS)
    }})


def archived_candidates(root: Path) -> list[tuple[str, dict]]:
    """Read existing Phase 16 snapshots; never write or overwrite an issuance."""
    folder = root / ARCHIVE
    result = []
    for path in sorted(folder.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        validate_snapshot(doc)
        issued = _timestamp(doc["issued_at_utc"])
        if path.name != issued.strftime("%Y%m%dT%H%M%S%fZ.json"):
            raise ValueError(f"Snapshot filename does not match its issuance: {path.name}")
        fingerprint = hashlib.sha256(_canonical(doc)).hexdigest()
        result.append((fingerprint, doc))
    # Earlier Phase 16 versions can have a latest artifact without an archive.
    # Do not silently import it as point-in-time evidence: require the archive.
    return result


def _metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    if not len(actual):
        return {"paired_rows": 0, "mae_mw": None, "rmse_mw": None,
                "bias_mw": None, "p90_absolute_error_mw": None}
    diff = forecast - actual
    return {"paired_rows": int(len(diff)),
            "mae_mw": round(float(np.abs(diff).mean()), 3),
            "rmse_mw": round(float(np.sqrt(np.mean(diff * diff))), 3),
            "bias_mw": round(float(diff.mean()), 3),
            "p90_absolute_error_mw": round(float(np.percentile(np.abs(diff), 90)), 3)}


def _reduction(target_mae, comparator_mae):
    if target_mae is None or comparator_mae is None or comparator_mae <= 0:
        return None
    return round(100.0 * (comparator_mae - target_mae) / comparator_mae, 2)


def verify_candidate(fingerprint: str, payload: dict, measured: pd.DataFrame,
                     *, as_of: pd.Timestamp, buffer_minutes: int = BUFFER_MINUTES) -> dict:
    if buffer_minutes < 0:
        raise ValueError("Availability buffer cannot be negative")
    now = _timestamp(as_of)
    targets = validate_snapshot(payload)
    if not set(("timestamp_utc", "demand_mw")).issubset(measured.columns):
        raise ValueError("Missing measured-demand columns")
    data = measured[["timestamp_utc", "demand_mw"]].copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="raise")
    data["demand_mw"] = pd.to_numeric(data["demand_mw"], errors="raise")
    if (data["timestamp_utc"].isna().any() or data["timestamp_utc"].duplicated().any() or
            (data["demand_mw"].notna() & (~np.isfinite(data["demand_mw"]) | (data["demand_mw"] < 0))).any()):
        raise ValueError("Invalid actual demand measurements")
    frame = targets.merge(data, on="timestamp_utc", how="left", validate="one_to_one")
    # Target timestamps are interval STARTS; require the interval to have ended,
    # plus the conservative source-availability buffer.
    cutoff = now - timedelta(minutes=(15 + buffer_minutes))
    mature = frame["timestamp_utc"] <= cutoff
    matched = mature & frame["demand_mw"].notna()

    def group_scores(group: pd.DataFrame) -> dict:
        paired = group.loc[group["timestamp_utc"].le(cutoff) & group["demand_mw"].notna()]
        actual = paired["demand_mw"].to_numpy(dtype=float)
        return {name: _metrics(actual, paired[key].to_numpy(dtype=float))
                for name, key in PREDICTION_KEYS.items()}

    metrics = group_scores(frame)
    n_eligible, n_matched = int(mature.sum()), int(matched.sum())
    status = ("pending" if n_eligible == 0 else
              "complete" if n_matched == HORIZON_STEPS else
              "partial" if n_eligible < HORIZON_STEPS else "incomplete_source_data")
    bands = []
    for begin, end, label in BUCKETS:
        sub = frame.iloc[begin:end]
        sub_eligible = sub["timestamp_utc"].le(cutoff)
        n = int((sub_eligible & sub["demand_mw"].notna()).sum())
        bands.append({"range": label, "total_targets": end - begin,
                      "eligible_targets": int(sub_eligible.sum()), "matched_targets": n,
                      "metrics": group_scores(sub)})
    return {"forecast_id": fingerprint[:20], "snapshot_sha256": fingerprint,
            "issued_at_utc": _timestamp(payload["issued_at_utc"]).isoformat(),
            "first_target_utc": targets["timestamp_utc"].iloc[0].isoformat(),
            "last_target_utc": targets["timestamp_utc"].iloc[-1].isoformat(),
            "preferred_strategy": payload["preferred_strategy"],
            "historical_baseline": payload["baseline_name"],
            "status": status, "total_targets": HORIZON_STEPS,
            "eligible_targets": n_eligible, "matched_targets": n_matched,
            "missing_eligible_targets": n_eligible - n_matched,
            "metrics": metrics,
            "weather_mae_reduction_vs_demand_only_pct": _reduction(metrics["demand_weather"]["mae_mw"], metrics["demand_only"]["mae_mw"]),
            "weather_mae_reduction_vs_baseline_pct": _reduction(metrics["demand_weather"]["mae_mw"], metrics["baseline"]["mae_mw"]),
            "by_horizon": bands}


def run_verification(root: Path, *, as_of: pd.Timestamp | None = None,
                     buffer_minutes: int = BUFFER_MINUTES) -> dict:
    root = root.resolve()
    now = pd.Timestamp.now(tz="UTC") if as_of is None else _timestamp(as_of)
    snapshots = archived_candidates(root)
    if snapshots and not (root / LIVE_CSV).is_file():
        raise FileNotFoundError("Recent measured data unavailable; run python -m gridwise.refresh first")
    actual = (read_measurements(root / LIVE_CSV) if snapshots else
              pd.DataFrame(columns=["timestamp_utc", "demand_mw"]))
    results = [verify_candidate(digest, doc, actual, as_of=now, buffer_minutes=buffer_minutes)
               for digest, doc in snapshots]
    results.sort(key=lambda entry: (entry["issued_at_utc"], entry["forecast_id"]), reverse=True)
    report = {
        "schema_version": 1, "status": "ok" if results else "no_forecasts",
        "verified_at_utc": now.isoformat(),
        "availability_buffer_minutes": buffer_minutes,
        "forecast_count": len(results), "forecasts": results,
        "notes": [
            "Weather candidates are experimental and remain separate from official Fingrid captures.",
            "All prediction methods are compared on identical observed quarter-hour targets.",
            "A quarter-hour target matures only after its end plus the availability buffer.",
            "Each archived issuance is scored separately; overlapping issuances are not pooled.",
            "Snapshots are local files with SHA-256 identifiers, not independently timestamp-certified evidence.",
            "Partial results describe only available measurements, not the complete 24-hour horizon.",
        ],
    }
    atomic_json(root / OUTPUT, report)
    return report


def main():
    p = argparse.ArgumentParser(description="Verify archived weather candidate forecasts against later measured demand")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--availability-buffer-minutes", type=int, default=BUFFER_MINUTES)
    args = p.parse_args()
    report = run_verification(args.root, buffer_minutes=args.availability_buffer_minutes)
    print(f"Archived weather issuances: {report['forecast_count']} | verified {report['verified_at_utc']}")
    for item in report["forecasts"]:
        scores = item["metrics"]
        print(f"  {item['issued_at_utc']} | {item['status']} | {item['matched_targets']}/96 measured")
        if item["matched_targets"]:
            print("    Demand+weather: {0:.1f} MW | Demand-only: {1:.1f} MW | Baseline: {2:.1f} MW".format(
                scores["demand_weather"]["mae_mw"], scores["demand_only"]["mae_mw"], scores["baseline"]["mae_mw"]))
    print(f"Saved: {args.root.resolve() / OUTPUT}")
    print("This command only verifies existing snapshots. It never downloads or issues forecasts.")


if __name__ == "__main__":
    main()
