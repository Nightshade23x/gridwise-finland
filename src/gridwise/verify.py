"""Phase 13: independently score archived forward forecasts against measured demand.

No model training, network calls, or modification of predictions occurs here.
The worker only evaluates exact UTC quarter-hour matches from Phase 11 data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .forward import FORECAST_JSON, HORIZON_STEPS, INTERVAL
from .refresh import ROOT, LIVE_CSV, atomic_json, read_measurements

ARCHIVE_DIR = Path("data/live/forecast_archive")
VERIFICATION_JSON = Path("data/live/forecast_verification.json")
DEFAULT_BUFFER_MINUTES = 60
HORIZON_BUCKETS = ((0, 24, "0–6 h"), (24, 48, "6–12 h"),
                   (48, 72, "12–18 h"), (72, 96, "18–24 h"))


def _stamp(text: str) -> pd.Timestamp:
    stamp = pd.Timestamp(text)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError("Forecast timestamps must include a timezone")
    return stamp.tz_convert("UTC")


def validate_forecast(payload: dict) -> pd.DataFrame:
    """Reject corrupt or retrospectively dated forecast artifacts."""
    if not isinstance(payload, dict) or payload.get("kind") != "experimental_forward_forecast" or payload.get("status") != "ok":
        raise ValueError("Unrecognized forward-forecast artifact")
    if payload.get("resolution_minutes") != 15:
        raise ValueError("Only 15-minute forecast artifacts are supported")
    issued = _stamp(payload["issued_at_utc"])
    points = payload.get("points")
    if not isinstance(points, list) or len(points) != HORIZON_STEPS:
        raise ValueError("Forecast must contain 96 target points")
    stamps = pd.DatetimeIndex([_stamp(p["timestamp_utc"]) for p in points])
    if (stamps[0] <= issued or stamps.has_duplicates or
        not (stamps[1:] - stamps[:-1] == INTERVAL).all() or
        (stamps.minute % 15 != 0).any() or (stamps.second != 0).any() or
        (stamps.microsecond != 0).any()):
        raise ValueError("Forecast targets must be future, unique, and quarter-hour aligned")
    if payload.get("first_target_utc") and _stamp(payload["first_target_utc"]) != stamps[0]:
        raise ValueError("First target metadata does not match forecast")
    if payload.get("last_target_utc") and _stamp(payload["last_target_utc"]) != stamps[-1]:
        raise ValueError("Last target metadata does not match forecast")
    pred = np.asarray([p["predicted_mw"] for p in points], dtype=float)
    base = np.asarray([p["baseline_mw"] for p in points], dtype=float)
    if not np.isfinite(pred).all() or not np.isfinite(base).all() or (pred < 0).any() or (base < 0).any():
        raise ValueError("Forecast and reference values must be finite and nonnegative")
    return pd.DataFrame({"timestamp_utc": stamps, "predicted_mw": pred, "baseline_mw": base,
                         "target_step": np.arange(HORIZON_STEPS)})


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def archive_forecast(root: Path, payload: dict) -> Path:
    """Content-addressed snapshot; repeated writes of identical predictions are idempotent.

    Never overwrite another forecast snapshot. The archive is local evidence,
    not a tamper-proof external timestamping service.
    """
    validate_forecast(payload)
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    issued = _stamp(payload["issued_at_utc"])
    basename = f"{issued.strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:20]}.json"
    path = root / ARCHIVE_DIR / basename
    if path.is_file():
        archived = json.loads(path.read_text(encoding="utf-8"))
        if _canonical(archived) != _canonical(payload):
            raise ValueError("Archive hash collision or changed snapshot")
    else:
        atomic_json(path, payload)
    return path


def archived_forecasts(root: Path) -> list[tuple[str, dict]]:
    records = []
    folder = root / ARCHIVE_DIR
    for path in sorted(folder.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        validate_forecast(obj)
        digest = hashlib.sha256(_canonical(obj)).hexdigest()
        issued = _stamp(obj["issued_at_utc"])
        expected = f"{issued.strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:20]}.json"
        if path.name != expected:
            raise ValueError(f"Archived forecast content no longer matches its identifier: {path.name}")
        records.append((path.stem, obj))
    return records


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    if len(actual) == 0:
        return {"paired_rows": 0, "mae_mw": None, "rmse_mw": None,
                "bias_mw": None, "p90_absolute_error_mw": None}
    error = predicted - actual
    return {"paired_rows": len(error),
            "mae_mw": round(float(np.mean(np.abs(error))), 3),
            "rmse_mw": round(float(np.sqrt(np.mean(error ** 2))), 3),
            "bias_mw": round(float(np.mean(error)), 3),
            "p90_absolute_error_mw": round(float(np.percentile(np.abs(error), 90)), 3)}


def verify_one(forecast_id: str, payload: dict, measured: pd.DataFrame,
               *, as_of: pd.Timestamp, buffer_minutes: int = DEFAULT_BUFFER_MINUTES) -> dict:
    """Score only observations eligible as of verification time; never fill gaps."""
    if buffer_minutes < 0:
        raise ValueError("buffer_minutes must be nonnegative")
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("Verification time must be timezone-aware")
    as_of = as_of.tz_convert("UTC")
    targets = validate_forecast(payload)
    measured = measured[["timestamp_utc", "demand_mw"]].copy()
    measured["timestamp_utc"] = pd.to_datetime(measured["timestamp_utc"], utc=True)
    if measured["timestamp_utc"].isna().any() or measured["timestamp_utc"].duplicated().any():
        raise ValueError("Measured timestamps must be unique and valid")
    measured["demand_mw"] = pd.to_numeric(measured["demand_mw"], errors="raise")
    if (~np.isfinite(measured["demand_mw"])).any() or (measured["demand_mw"] < 0).any():
        raise ValueError("Invalid measured demand")
    joined = targets.merge(measured, on="timestamp_utc", how="left", validate="one_to_one")
    cutoff = as_of - pd.Timedelta(minutes=buffer_minutes)
    mature = joined["timestamp_utc"] <= cutoff
    paired = mature & joined["demand_mw"].notna()
    valid = joined.loc[paired]
    actual = valid["demand_mw"].to_numpy(dtype=float)
    model = _score(actual, valid["predicted_mw"].to_numpy(dtype=float))
    baseline = _score(actual, valid["baseline_mw"].to_numpy(dtype=float))
    reduction = None if not baseline["mae_mw"] or model["mae_mw"] is None else round(
        100.0 * (baseline["mae_mw"] - model["mae_mw"]) / baseline["mae_mw"], 2)
    n_mature, n_paired = int(mature.sum()), int(paired.sum())
    status = ("pending" if n_mature == 0 else
              "complete" if n_paired == HORIZON_STEPS else
              "partial" if n_mature < HORIZON_STEPS else "incomplete_source_data")
    by_horizon = []
    for start, end, label in HORIZON_BUCKETS:
        bucket = joined.iloc[start:end]
        slot_mature = bucket["timestamp_utc"] <= cutoff
        good = bucket.loc[slot_mature & bucket["demand_mw"].notna()]
        observations = good["demand_mw"].to_numpy(dtype=float)
        by_horizon.append({"range": label, "matched": len(good),
                           "eligible": int(slot_mature.sum()), "slots": end - start,
                           "model_mae_mw": _score(observations, good["predicted_mw"].to_numpy(dtype=float))["mae_mw"],
                           "baseline_mae_mw": _score(observations, good["baseline_mw"].to_numpy(dtype=float))["mae_mw"]})
    return {
        "forecast_id": forecast_id,
        "issued_at_utc": _stamp(payload["issued_at_utc"]).isoformat(),
        "first_target_utc": targets["timestamp_utc"].iloc[0].isoformat(),
        "last_target_utc": targets["timestamp_utc"].iloc[-1].isoformat(),
        "status": status,
        "total_targets": HORIZON_STEPS,
        "eligible_targets": n_mature,
        "matched_targets": n_paired,
        "missing_eligible_targets": n_mature - n_paired,
        "model": model,
        "baseline": baseline,
        "mae_reduction_percent": reduction,
        "by_horizon": by_horizon,
    }


def run_verification(root: Path, *, as_of: pd.Timestamp | None = None,
                     buffer_minutes: int = DEFAULT_BUFFER_MINUTES) -> dict:
    """Import Phase 12's existing latest snapshot once, then score the entire archive."""
    root = root.resolve()
    time = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if time.tzinfo is None:
        raise ValueError("Verification time must be timezone-aware")
    time = time.tz_convert("UTC")
    latest = root / FORECAST_JSON
    if latest.exists():
        # Preserve the pre-Phase-13 forecast. This is a locally imported
        # artifact, not independently authenticated historical issuance.
        obj = json.loads(latest.read_text(encoding="utf-8"))
        dest = archive_forecast(root, obj)
    archives = archived_forecasts(root)
    if not archives:
        report = {"schema_version": 1, "status": "no_forecasts", "verified_at_utc": time.isoformat(),
                  "availability_buffer_minutes": buffer_minutes, "forecast_count": 0,
                  "forecasts": [], "notes": ["Issue a forecast with python -m gridwise.forward first."]}
    else:
        if not (root / LIVE_CSV).is_file():
            raise FileNotFoundError("Recent measurements unavailable; run python -m gridwise.refresh first")
        measured = read_measurements(root / LIVE_CSV)
        results = [verify_one(identifier, payload, measured, as_of=time, buffer_minutes=buffer_minutes)
                   for identifier, payload in archives]
        results.sort(key=lambda r: (r["issued_at_utc"], r["forecast_id"]), reverse=True)
        report = {
            "schema_version": 1, "status": "ok", "verified_at_utc": time.isoformat(),
            "availability_buffer_minutes": buffer_minutes, "forecast_count": len(results),
            "forecasts": results,
            "notes": [
                "Each issuance is scored separately; repeated forecasts of overlapping targets are not pooled.",
                "Errors include only source measurements already present and old enough for the availability buffer.",
                "Imported Phase 12 latest forecasts are local artifacts, not externally timestamp-authenticated evidence.",
                "Partial scores are preliminary and must not be compared with full-horizon scores as if equivalent.",
            ],
        }
    atomic_json(root / VERIFICATION_JSON, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Verify previously issued forward forecasts against updated demand")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--availability-buffer-minutes", type=int, default=DEFAULT_BUFFER_MINUTES)
    args = p.parse_args()
    result = run_verification(args.root, buffer_minutes=args.availability_buffer_minutes)
    if not result["forecasts"]:
        print("No issued forward forecasts available yet.")
    for forecast in result["forecasts"]:
        name = forecast["issued_at_utc"]
        print(f"Issued {name} | {forecast['status']} | matched {forecast['matched_targets']}/96; "
              f"eligible {forecast['eligible_targets']}/96")
        if forecast["matched_targets"]:
            print(f"  LightGBM MAE {forecast['model']['mae_mw']:.1f} MW; "
                  f"reference MAE {forecast['baseline']['mae_mw']:.1f} MW; "
                  f"bias {forecast['model']['bias_mw']:+.1f} MW")
    print(f"Saved verification report: {args.root / VERIFICATION_JSON}")
    print("These are measured forward errors, not the historical Phase 5 backtest scores.")


if __name__ == "__main__":
    main()
