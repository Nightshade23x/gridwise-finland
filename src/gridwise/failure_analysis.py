"""Phase 20: read-only investigation of *issued* forecasts and later observations.

No model fitting, web requests, data imputation, or mutation of snapshots. Each
issuance is scored separately. Forecasts with overlapping targets are NOT
independent experiments and their errors are never pooled into one headline.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .refresh import ROOT, LIVE_CSV, atomic_csv, atomic_json, read_measurements
from .verify import archived_forecasts, validate_forecast
from .weather_verify import archived_candidates, validate_snapshot

OUTPUT = Path("data/live/failure_analysis.json")
ROWS_CSV = Path("data/model_outputs/failure_analysis_targets.csv")
METHODS = {
    "forward": {"lightgbm": "predicted_mw", "baseline": "baseline_mw"},
    "weather": {"demand_weather": "demand_weather_mw", "demand_only": "demand_only_mw", "baseline": "baseline_mw"},
}
HORIZONS = ((0, 24, "0–6 h"), (24, 48, "6–12 h"), (48, 72, "12–18 h"), (72, 96, "18–24 h"))


def _stamp(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError("A timezone-aware timestamp is required")
    return t.tz_convert("UTC")


def _stats(frame: pd.DataFrame, column: str) -> dict:
    if frame.empty:
        return {"count": 0, "mae_mw": None, "rmse_mw": None, "bias_mw": None,
                "p90_absolute_error_mw": None, "overprediction_pct": None}
    diff = (frame[column] - frame["demand_mw"]).to_numpy(dtype=float)
    return {"count": int(len(diff)), "mae_mw": round(float(np.abs(diff).mean()), 2),
            "rmse_mw": round(float(np.sqrt(np.mean(diff * diff))), 2),
            "bias_mw": round(float(np.mean(diff)), 2),
            "p90_absolute_error_mw": round(float(np.percentile(np.abs(diff), 90)), 2),
            "overprediction_pct": round(float(100 * np.mean(diff > 0)), 1)}


def diagnose_issuance(kind: str, identifier: str, payload: dict, measured: pd.DataFrame,
                      *, as_of: pd.Timestamp, buffer_minutes: int = 60,
                      top_n: int = 5) -> tuple[dict, pd.DataFrame]:
    """Score one archived issuance on exact, mature, measured target timestamps."""
    if kind not in METHODS or buffer_minutes < 0 or top_n < 1:
        raise ValueError("Unknown forecast type or invalid diagnostics arguments")
    issue = _stamp(payload["issued_at_utc"])
    now = _stamp(as_of)
    targets = validate_forecast(payload) if kind == "forward" else validate_snapshot(payload)
    columns = METHODS[kind]
    if not {"timestamp_utc", "demand_mw"}.issubset(measured.columns):
        raise ValueError("Measured demand is missing required columns")
    source = measured[["timestamp_utc", "demand_mw"]].copy()
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True, errors="raise")
    source["demand_mw"] = pd.to_numeric(source["demand_mw"], errors="raise")
    if (source["timestamp_utc"].isna().any() or source["timestamp_utc"].duplicated().any() or
            source["demand_mw"].isna().any() or
            not np.isfinite(source["demand_mw"]).all() or (source["demand_mw"] < 0).any()):
        raise ValueError("Measured observations contain invalid data")
    joined = targets.merge(source, on="timestamp_utc", how="left", validate="one_to_one")
    # Each timestamp is the START of a 15-minute measurement. Wait for the
    # interval to END, then apply the user-configured publication safety buffer.
    cutoff = now - timedelta(minutes=(15 + buffer_minutes))
    matured = joined["timestamp_utc"].le(cutoff)
    matched = matured & joined["demand_mw"].notna()
    paired = joined.loc[matched].copy()
    paired["target_step"] = paired.index
    paired["horizon_hour"] = paired["target_step"] // 4
    paired["utc_hour"] = paired["timestamp_utc"].dt.hour
    paired["finland_hour"] = paired["timestamp_utc"].dt.tz_convert("Europe/Helsinki").dt.hour
    paired["forecast_kind"] = kind
    paired["forecast_id"] = identifier
    paired["issued_at_utc"] = issue.isoformat()
    for name, col in columns.items():
        paired[f"{name}_error_mw"] = paired[col] - paired["demand_mw"]
    metrics = {name: _stats(paired, col) for name, col in columns.items()}
    summary = []
    for begin, end, label in HORIZONS:
        sub = paired.loc[paired["target_step"].between(begin, end - 1)]
        summary.append({"range": label, "matched": len(sub),
                        "eligible": int(matured.iloc[begin:end].sum()), "total": end - begin,
                        "mae_by_method": {name: _stats(sub, col)["mae_mw"] for name, col in columns.items()},
                        "bias_by_method": {name: _stats(sub, col)["bias_mw"] for name, col in columns.items()}})
    by_utc_hour = []
    for hour in range(24):
        sub = paired.loc[paired["utc_hour"] == hour]
        if not sub.empty:
            by_utc_hour.append({"utc_hour": hour, "matched": len(sub),
                                "mae_by_method": {n: _stats(sub, c)["mae_mw"] for n, c in columns.items()},
                                "bias_by_method": {n: _stats(sub, c)["bias_mw"] for n, c in columns.items()}})
    by_finland_hour = []
    for hour in range(24):
        sub = paired.loc[paired["finland_hour"] == hour]
        if not sub.empty:
            by_finland_hour.append({"finland_hour": hour, "matched": len(sub),
                                    "mae_by_method": {n: _stats(sub, c)["mae_mw"] for n, c in columns.items()},
                                    "bias_by_method": {n: _stats(sub, c)["bias_mw"] for n, c in columns.items()}})
    # Demand levels are descriptive, within THIS issuance; do not use them as
    # known-at-issuance features or infer a general seasonal relationship.
    by_demand_level = []
    if len(paired) >= 8:
        mid = float(paired["demand_mw"].median())
        for label, sub in (("at_or_below_median", paired.loc[paired["demand_mw"] <= mid]),
                           ("above_median", paired.loc[paired["demand_mw"] > mid])):
            by_demand_level.append({"group": label, "matched": len(sub),
                                    "threshold_mw": round(mid, 2),
                                    "mae_by_method": {n: _stats(sub, c)["mae_mw"] for n, c in columns.items()}})
    focus = "lightgbm" if kind == "forward" else "demand_weather"
    if not paired.empty:
        worst = paired.assign(_abs=paired[f"{focus}_error_mw"].abs()).nlargest(top_n, "_abs")
        worst_errors = [{"timestamp_utc": t.isoformat(), "target_step": int(step),
                         "actual_mw": round(float(y), 2), "prediction_mw": round(float(p), 2),
                         "signed_error_mw": round(float(p-y), 2)}
                        for t, step, y, p in zip(worst["timestamp_utc"], worst["target_step"],
                                                worst["demand_mw"], worst[columns[focus]])]
    else:
        worst_errors = []
    coverage = {"eligible": int(matured.sum()), "matched": int(matched.sum()),
                "missing_eligible": int((matured & ~matched).sum()), "total": 96}
    status = ("pending" if coverage["eligible"] == 0 else
              "complete" if coverage["matched"] == 96 else
              "partial" if coverage["eligible"] < 96 else "incomplete_source_data")
    flags = []
    m = metrics[focus]
    if m["count"] >= 16 and m["mae_mw"] and abs(m["bias_mw"]) >= 0.8*m["mae_mw"]:
        direction = "overprediction" if m["bias_mw"] > 0 else "underprediction"
        flags.append(f"Persistent signed {direction} in this window; examine level shift and feature alignment.")
    if coverage["matched"] < 96:
        flags.append("Incomplete horizon: do not compare directly with full 24-hour scores.")
    if kind == "weather" and m["mae_mw"] is not None and metrics["demand_only"]["mae_mw"] is not None:
        if m["mae_mw"] > metrics["demand_only"]["mae_mw"]:
            flags.append("Weather candidate underperformed demand-only on these observed targets; a single window cannot establish why.")
    # Hash exact target+method values to identify literally duplicated curves,
    # not merely overlapping 24-hour issuance windows.
    key = [[point["timestamp_utc"].isoformat(), *[float(point[c]) for c in columns.values()]]
           for _, point in targets.iterrows()]
    curve_hash = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
    result = {"kind": kind, "forecast_id": identifier,
              "issued_at_utc": issue.isoformat(), "first_target_utc": targets["timestamp_utc"].iloc[0].isoformat(),
              "last_target_utc": targets["timestamp_utc"].iloc[-1].isoformat(),
              "status": status, "coverage": coverage, "methods": metrics,
              "focus_method": focus, "by_horizon": summary, "by_utc_hour": by_utc_hour,
              "by_finland_hour": by_finland_hour,
              "by_actual_demand_level": by_demand_level, "largest_errors": worst_errors,
              "flags": flags, "curve_sha256": curve_hash, "identical_prediction_to": None}
    return result, paired


def run_analysis(root: Path, *, as_of: pd.Timestamp | None = None,
                 buffer_minutes: int = 60, top_n: int = 5) -> dict:
    """Read archived inputs; atomically write diagnostics only (not forecasts)."""
    root = Path(root).resolve()
    if buffer_minutes < 0 or top_n < 1:
        raise ValueError("Invalid buffer or top error count")
    time = pd.Timestamp.now(tz="UTC") if as_of is None else _stamp(as_of)
    forwards = archived_forecasts(root)
    weather = archived_candidates(root)
    all_forecasts = [("forward", key, doc) for key, doc in forwards] + [
        ("weather", key[:20], doc) for key, doc in weather]
    if all_forecasts and not (root / LIVE_CSV).exists():
        raise FileNotFoundError("Measured data missing; run python -m gridwise.refresh first")
    measured = read_measurements(root / LIVE_CSV) if all_forecasts else pd.DataFrame(columns=["timestamp_utc", "demand_mw"])
    output = [diagnose_issuance(kind, key, doc, measured, as_of=time,
                               buffer_minutes=buffer_minutes, top_n=top_n)
              for kind, key, doc in all_forecasts]
    records = [x[0] for x in output]
    records.sort(key=lambda x: (x["issued_at_utc"], x["forecast_id"]), reverse=True)
    # Across issuances, mark identical curves rather than double-count them.
    seen = {}
    for record in records:
        curve = (record["kind"], record["curve_sha256"])
        if curve in seen:
            record["identical_prediction_to"] = seen[curve]
            record["flags"].append("Identical prediction curve to another archived issuance; not an independent experiment.")
        else:
            seen[curve] = record["forecast_id"]
    paired = pd.concat([x[1] for x in output], ignore_index=True) if output else pd.DataFrame()
    # Only diagnostic rows are exported. The full measured-demand store stays private.
    if not paired.empty:
        cols = ["forecast_kind", "forecast_id", "issued_at_utc", "timestamp_utc", "target_step",
                "horizon_hour", "utc_hour", "demand_mw"]
        cols += [col for col in paired.columns if col.endswith("_error_mw")]
        csv_frame = paired.reindex(columns=cols)
    else:
        csv_frame = pd.DataFrame(columns=["forecast_kind", "forecast_id", "issued_at_utc",
                                          "timestamp_utc", "target_step", "horizon_hour",
                                          "utc_hour", "demand_mw"])
    report = {"schema_version": 1, "status": "ok" if records else "no_forecasts",
              "analyzed_at_utc": time.isoformat(), "availability_buffer_minutes": buffer_minutes,
              "issuance_count": len(records), "issuances": records,
              "notes": ["Each issuance is scored separately; overlapping timestamps are not pooled.",
                        "Signed bias means predicted minus observed demand (MW).",
                        "Targets are eligible after their 15-minute measurement interval ends plus the buffer.",
                        "These descriptive diagnostics cannot identify causes of forecast failure.",
                        "Historical backtest results are not mixed with forward issuance results."]}
    atomic_json(root / OUTPUT, report)
    atomic_csv(root / ROWS_CSV, csv_frame)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze archived GridWise forward forecast failures")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--availability-buffer-minutes", type=int, default=60)
    parser.add_argument("--top-errors", type=int, default=5)
    args = parser.parse_args()
    report = run_analysis(args.root, buffer_minutes=args.availability_buffer_minutes, top_n=args.top_errors)
    print(f"Analyzed {report['issuance_count']} forecast issuances (one row per issuance, never pooled).")
    for item in report["issuances"]:
        m = item["methods"][item["focus_method"]]
        score = "pending" if m["mae_mw"] is None else f"MAE {m['mae_mw']:.1f} MW; bias {m['bias_mw']:+.1f} MW"
        print(f"  {item['kind']:7} {item['issued_at_utc']} | {item['status']} "
              f"{item['coverage']['matched']}/96 | {score}")
        if item["identical_prediction_to"]:
            print("    Identical prediction curve to another issuance (not independent).")
    print(f"Saved report: {args.root.resolve() / OUTPUT}")
    print(f"Saved paired error rows: {args.root.resolve() / ROWS_CSV}")
    print("Diagnostics are descriptive only; no archive or forecast was modified.")


if __name__ == "__main__":
    main()
