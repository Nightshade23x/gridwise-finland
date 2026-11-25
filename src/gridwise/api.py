"""Read-only API for historical backtests, recent measurements, and experimental forward predictions.

No forecasting or external requests occur inside the web-server process.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
import json
from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = Path(__file__).resolve().parent / "dashboard"

DATA_FILES = {
    "history": Path("data/processed/consumption_2024_2026.csv"),
    "test": Path("data/model_outputs/lightgbm_predictions_2026.csv"),
    "baselines": Path("data/model_outputs/baseline_predictions_2026.csv"),
    "recent": Path("data/live/consumption_recent.csv"),
}

COLUMNS = {
    "history": ("demand_mw",),
    "recent": ("demand_mw",),
    "test": ("actual_mw", "model_mw", "selected_baseline_mw"),
}


@lru_cache(maxsize=8)
def _read_csv(path: str, modified_ns: int) -> pd.DataFrame:
    """Cache CSV by its modification time; fresh outputs refresh automatically."""
    frame = pd.read_csv(path)
    if "timestamp_utc" not in frame.columns:
        raise ValueError(f"Missing timestamp_utc in {Path(path).name}")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if frame["timestamp_utc"].isna().any() or frame["timestamp_utc"].duplicated().any():
        raise ValueError(f"Invalid or duplicate timestamps in {Path(path).name}")
    return frame.sort_values("timestamp_utc").set_index("timestamp_utc")


def _file_updated_utc(path: Path) -> str | None:
    """Timestamp of our LOCAL saved artifact, not source publication or issue time."""
    if not path.is_file():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _load(root: Path, key: str) -> pd.DataFrame:
    path = root / DATA_FILES[key]
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Missing {DATA_FILES[key].as_posix()}. Run the existing GridWise preparation/forecast command first.",
        )
    try:
        frame = _read_csv(str(path), path.stat().st_mtime_ns)
        required = set(COLUMNS[key]) if key in COLUMNS else set()
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns {sorted(required - set(frame.columns))}")
        return frame
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=503, detail=f"Invalid {path.name}: {exc}") from exc


def _mean_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    truth = pd.to_numeric(actual, errors="coerce")
    forecast = pd.to_numeric(predicted, errors="coerce")
    good = truth.notna() & forecast.notna() & np.isfinite(truth) & np.isfinite(forecast)
    if not good.any():
        return {"mae_mw": None, "rmse_mw": None, "paired_rows": 0}
    diff = (truth[good] - forecast[good]).to_numpy(dtype=float)
    return {
        "mae_mw": round(float(np.mean(np.abs(diff))), 3),
        "rmse_mw": round(float(np.sqrt(np.mean(diff ** 2))), 3),
        "paired_rows": int(good.sum()),
    }


def _all_metrics(root: Path, test: pd.DataFrame) -> tuple[dict, int]:
    """Evaluate each method on identical rows, when baseline CSV is present."""
    joined = test[["actual_mw", "model_mw", "selected_baseline_mw"]].copy()
    methods = {"lightgbm": "model_mw", "selected_baseline": "selected_baseline_mw"}
    if (root / DATA_FILES["baselines"]).exists():
        old = _load(root, "baselines")
        extra = ["pred_yesterday_mw", "pred_last_week_mw"]
        if set(extra).issubset(old.columns):
            joined = joined.join(old[extra], how="left")
            joined["blend_50_50_mw"] = (joined[extra[0]] + joined[extra[1]]) / 2
            methods.update({
                "yesterday": extra[0],
                "last_week": extra[1],
                "blend_50_50": "blend_50_50_mw",
            })
    numerical = joined[["actual_mw", *methods.values()]].apply(pd.to_numeric, errors="coerce")
    common = numerical.replace([np.inf, -np.inf], np.nan).dropna()
    if common.empty:
        return {}, 0
    scores = {name: _mean_metrics(common["actual_mw"], common[col]) for name, col in methods.items()}
    return scores, len(common)


def create_app(root: Path | None = None) -> FastAPI:
    project_root = Path(root) if root is not None else Path(os.environ.get("GRIDWISE_PROJECT_ROOT", str(ROOT)))
    api = FastAPI(title="GridWise Finland", version="0.2.0", description="Historical backtests, recent measurements, and experimental forward-looking demand forecast.")
    api.mount("/static", StaticFiles(directory=str(DASHBOARD)), name="static")

    @api.get("/", include_in_schema=False)
    def index():
        return FileResponse(DASHBOARD / "index.html")

    @api.get("/api/health")
    def health():
        paths = {key: (project_root / name).is_file() for key, name in DATA_FILES.items()}
        return {"status": "ready" if paths["history"] and paths["test"] else "missing_data", "files": paths}

    @api.get("/api/deployment-info")
    def deployment_info():
        """Describe whether this installation is a static public demo or local tool."""
        if os.environ.get("GRIDWISE_DEMO_MODE") != "1":
            return {"mode": "local", "live_updates_enabled": False,
                    "message": "Local GridWise installation; data updates require separate CLI workers."}
        path = project_root / "demo_manifest.json"
        if not path.is_file():
            raise HTTPException(503, "Public demo manifest is missing")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (manifest.get("schema_version") != 1 or
                    manifest.get("kind") != "static_historical_demo" or
                    not isinstance(manifest.get("generated_at_utc"), str)):
                raise ValueError("Invalid manifest")
            return {"mode": "demo", "live_updates_enabled": False,
                    "snapshot_generated_at_utc": manifest["generated_at_utc"],
                    "source": manifest.get("source"),
                    "message": "Archived historical research demo, not a live or periodically refreshed forecast."}
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(503, "Public demo manifest is invalid") from exc

    @api.get("/api/refresh-status")
    def refresh_status():
        """Status is read-only; the API never triggers private-key downloads."""
        path = project_root / "data/live/refresh_status.json"
        if not path.is_file():
            return {"status": "not_configured", "latest_measurement_utc": None, "latest_demand_mw": None,
                    "last_success_utc": None, "last_attempt_utc": None,
                    "stored_rows": 0, "error_present": False}
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            fields = ("status", "latest_measurement_utc", "latest_demand_mw", "last_success_utc", "last_attempt_utc",
                      "stored_rows", "added_rows", "last_7_days")
            return {**{key: content.get(key) for key in fields}, "error_present": bool(content.get("error"))}
        except (ValueError, OSError) as exc:
            raise HTTPException(503, "Refresh status file is unreadable") from exc

    @api.get("/api/forward-forecast")
    def forward_forecast():
        """Serve an independently generated forward artifact. No keys, training or network here."""
        path = project_root / "data/live/forward_forecast_24h.json"
        if not path.is_file():
            return {"status": "not_generated", "points": [],
                    "message": "Run python -m gridwise.forward after refreshing measured demand"}
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            points = obj["points"]
            if obj.get("kind") != "experimental_forward_forecast" or obj.get("status") != "ok":
                raise ValueError("Invalid forecast metadata")
            if len(points) != 96 or int(obj.get("resolution_minutes", 0)) != 15:
                raise ValueError("Expected 96 fifteen-minute targets")
            stamps = pd.to_datetime([p["timestamp_utc"] for p in points], utc=True)
            if stamps.hasnans or stamps.has_duplicates or not (stamps[1:] - stamps[:-1] == timedelta(minutes=15)).all():
                raise ValueError("Forecast time grid is invalid")
            values = np.asarray([p["predicted_mw"] for p in points], dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid forecast predictions")
            # Optional Phase 15 validation choice; older snapshots still work.
            choice = obj.get("preferred_strategy")
            if choice is not None:
                if choice not in ("lightgbm", obj.get("selected_reference_baseline")):
                    raise ValueError("Invalid preferred strategy")
                preferred = np.asarray([p["preferred_mw"] for p in points], dtype=float)
                expected = (values if choice == "lightgbm" else
                            np.asarray([p["baseline_mw"] for p in points], dtype=float))
                if not np.isfinite(preferred).all() or not np.allclose(preferred, expected, rtol=0, atol=0.002):
                    raise ValueError("Preferred predictions do not match selected method")
            # Only expose safe metadata, never any credentials or filesystem paths.
            fields = ("kind", "status", "issued_at_utc", "first_target_utc", "last_target_utc",
                      "source_latest_measurement_utc", "source_age_hours_at_issue",
                      "resolution_minutes", "forecast_horizon_hours", "model",
                      "selected_reference_baseline", "preferred_strategy", "preferred_validation_mae_mw", "assumptions")
            return {**{key: obj.get(key) for key in fields}, "points": points}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(503, f"Forward forecast artifact is invalid: {exc}") from exc

    @api.get("/api/weather-model-comparison")
    def weather_model_comparison():
        """A read-only, separately issued experimental weather-candidate artifact."""
        path = project_root / "data/live/weather_model_comparison_24h.json"
        if not path.is_file():
            return {"status":"not_generated", "points":[],
                    "message":"Run python -m gridwise.weather_previous then python -m gridwise.weather_forward"}
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("kind") != "weather_forward_candidate" or doc.get("status") != "ok":
                raise ValueError("Invalid candidate metadata")
            points = doc["points"]
            if len(points) != 96 or int(doc.get("resolution_minutes", 0)) != 15:
                raise ValueError("Expected 96 fifteen-minute candidate targets")
            stamps = pd.DatetimeIndex(pd.to_datetime([r["timestamp_utc"] for r in points], utc=True))
            if (stamps.hasnans or stamps.has_duplicates or
                    not (stamps[1:] - stamps[:-1] == timedelta(minutes=15)).all()):
                raise ValueError("Invalid candidate time grid")
            issued = pd.Timestamp(doc["issued_at_utc"])
            if issued.tzinfo is None or not issued.tz_convert("UTC") < stamps[0]:
                raise ValueError("Candidate must be issued before its targets")
            values = np.asarray([[r[k] for k in
                ("demand_only_mw","demand_weather_mw","baseline_mw","preferred_mw")]
                for r in points], dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid candidate predictions")
            chosen = doc.get("preferred_strategy")
            if chosen not in ("demand_only", "demand_weather", doc.get("baseline_name")):
                raise ValueError("Unknown validation preference")
            chosen_col = {"demand_only":0,"demand_weather":1}.get(chosen,2)
            if not np.allclose(values[:,3], values[:,chosen_col], atol=0.002, rtol=0):
                raise ValueError("Preferred candidate does not match selected prediction")
            fields = ("kind", "status", "issued_at_utc", "first_target_utc", "last_target_utc",
                      "source_latest_measurement_utc", "weather_source",
                      "weather_source_vintage_certified", "weather_city_count", "resolution_minutes",
                      "forecast_horizon_hours", "validation_mae_mw", "preferred_strategy",
                      "baseline_name", "train_rows", "validation_rows", "notes")
            return {**{key:doc.get(key) for key in fields}, "points":points}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(503, f"Weather-candidate artifact is invalid: {exc}") from exc

    @api.get("/api/weather-forecast-verification")
    def weather_forecast_verification(limit: int = Query(20, ge=1, le=100)):
        """Read precomputed Phase 17 weather-forecast accuracy; no API calls or training."""
        path = project_root / "data/live/weather_forecast_verification.json"
        if not path.is_file():
            return {"status": "not_verified", "forecast_count": 0, "forecasts": [],
                    "message": "Refresh measured demand, then run python -m gridwise.weather_verify"}
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            records = report["forecasts"]
            if (report.get("schema_version") != 1 or report.get("status") not in ("ok", "no_forecasts")
                    or not isinstance(records, list) or report.get("forecast_count") != len(records)):
                raise ValueError("Malformed weather verification metadata")
            methods = ("demand_weather", "demand_only", "baseline", "preferred")
            for item in records:
                if item.get("status") not in ("pending", "partial", "complete", "incomplete_source_data"):
                    raise ValueError("Invalid comparison status")
                if not (0 <= item["matched_targets"] <= item["eligible_targets"] <= 96):
                    raise ValueError("Invalid coverage counts")
                if not isinstance(item.get("by_horizon"), list) or len(item["by_horizon"]) != 4:
                    raise ValueError("Invalid horizon breakdown")
                if not isinstance(item.get("snapshot_sha256"), str) or len(item["snapshot_sha256"]) != 64:
                    raise ValueError("Invalid snapshot identifier")
                groups = [item["metrics"], *(band["metrics"] for band in item["by_horizon"])]
                for group in groups:
                    if set(group) != set(methods):
                        raise ValueError("Missing candidate metrics")
                    paired = {name: group[name]["paired_rows"] for name in methods}
                    if len(set(paired.values())) != 1:
                        raise ValueError("Candidates must use identical measured timestamps")
            keys = ("forecast_id", "issued_at_utc", "first_target_utc", "last_target_utc",
                    "preferred_strategy", "historical_baseline", "status", "total_targets",
                    "eligible_targets", "matched_targets", "missing_eligible_targets",
                    "metrics", "weather_mae_reduction_vs_demand_only_pct",
                    "weather_mae_reduction_vs_baseline_pct", "by_horizon")
            return {"schema_version": 1, "status": report["status"],
                    "verified_at_utc": report.get("verified_at_utc"),
                    "availability_buffer_minutes": report.get("availability_buffer_minutes"),
                    "forecast_count": report["forecast_count"],
                    "forecasts": [{key: item.get(key) for key in keys} for item in records[:limit]],
                    "notes": report.get("notes", [])}
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise HTTPException(503, "Weather verification report is invalid") from exc

    @api.get("/api/forecast-verification")
    def forecast_verification(limit: int = Query(20, ge=1, le=100)):
        """Read saved forward-verification evidence; never compute it in the API."""
        path = project_root / "data/live/forecast_verification.json"
        if not path.is_file():
            return {"status": "not_verified", "forecast_count": 0, "forecasts": [],
                    "message": "Refresh measured demand, then run python -m gridwise.verify"}
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            records = report["forecasts"]
            if report.get("schema_version") != 1 or not isinstance(records, list):
                raise ValueError("Unsupported verification report")
            if report.get("status") not in ("ok", "no_forecasts"):
                raise ValueError("Invalid verification status")
            for entry in records:
                if not isinstance(entry, dict) or entry.get("status") not in (
                    "pending", "partial", "complete", "incomplete_source_data"):
                    raise ValueError("Invalid forecast verification record")
                if not (0 <= entry["matched_targets"] <= entry["eligible_targets"] <= 96):
                    raise ValueError("Invalid coverage counts")
                if not isinstance(entry.get("by_horizon"), list) or len(entry["by_horizon"]) != 4:
                    raise ValueError("Missing horizon breakdown")
            # Expose only reviewed result fields; never expose raw file paths or secrets.
            top_fields = ("schema_version", "status", "verified_at_utc", "availability_buffer_minutes", "forecast_count")
            entry_fields = ("forecast_id", "issued_at_utc", "first_target_utc", "last_target_utc", "status",
                            "total_targets", "eligible_targets", "matched_targets", "missing_eligible_targets",
                            "model", "baseline", "mae_reduction_percent", "by_horizon")
            return {**{k: report.get(k) for k in top_fields},
                    "forecasts": [{k: entry.get(k) for k in entry_fields} for entry in records[:limit]],
                    "notes": report.get("notes", [])}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise HTTPException(503, "Forward verification artifact is invalid") from exc

    @api.get("/api/official-benchmark")
    def official_benchmark(limit: int = Query(20, ge=1, le=100)):
        """Read a previously saved Phase 14 comparison; never fetch official forecasts live."""
        path = project_root / "data/live/official_benchmark.json"
        if not path.is_file():
            return {"status": "no_snapshots", "snapshot_count": 0, "comparisons": [],
                    "message": "Capture a new Fingrid forecast alongside GridWise using python -m gridwise.official capture"}
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            results = report["comparisons"]
            if report.get("schema_version") != 1 or report.get("status") not in ("ok", "no_snapshots") or not isinstance(results, list):
                raise ValueError("Invalid official benchmark report")
            for item in results:
                if item.get("status") not in ("pending", "partial", "complete", "incomplete_source_data"):
                    raise ValueError("Invalid comparison status")
                if not (0 <= item["matched_targets"] <= item["eligible_targets"] <= item["official_future_targets"] <= 96):
                    raise ValueError("Invalid matched observation counts")
            fields = ("gridwise_forecast_id", "gridwise_issued_at_utc", "fingrid_captured_at_utc", "status",
                      "official_future_targets", "eligible_targets", "matched_targets", "model", "fingrid",
                      "reference_baseline", "model_mae_improvement_vs_fingrid_percent", "by_horizon")
            return {"schema_version": 1, "status": report["status"], "verified_at_utc": report.get("verified_at_utc"),
                    "snapshot_count": report.get("snapshot_count", 0),
                    "comparisons": [{k: item.get(k) for k in fields} for item in results[:limit]],
                    "note": report.get("note", "")}
        except (ValueError, KeyError, OSError, TypeError) as exc:
            raise HTTPException(503, "Official benchmark report is invalid") from exc

    @api.get("/api/summary")
    def summary():
        history = _load(project_root, "history")
        test = _load(project_root, "test")
        measured = pd.to_numeric(history["demand_mw"], errors="coerce").notna()
        scores, common = _all_metrics(project_root, test)
        return {
            "mode": "historical_backtest",
            "live_forecast": False,
            "history": {
                "start_utc": history.index.min().isoformat(),
                "end_utc": history.index.max().isoformat(),
                "intervals": len(history),
                "observed_intervals": int(measured.sum()),
                "completeness_percent": round(100 * float(measured.mean()), 2),
            },
            "saved_artifact_times_utc": {
                "history_file_modified": _file_updated_utc(project_root / DATA_FILES["history"]),
                "backtest_file_modified": _file_updated_utc(project_root / DATA_FILES["test"]),
            },
            "evaluation": {
                "start_utc": test.index.min().isoformat(),
                "end_utc": test.index.max().isoformat(),
                "paired_rows": common,
                "metrics": scores,
            },
        }

    @api.get("/api/series")
    def series(
        kind: Literal["history", "test", "recent"] = "test",
        start: date = Query(..., description="Inclusive UTC date, YYYY-MM-DD"),
        end: date = Query(..., description="Exclusive UTC date, YYYY-MM-DD"),
        interval: Literal["15min", "hour", "day"] = "hour",
    ):
        if end <= start:
            raise HTTPException(422, "End date must be after start date")
        frame = _load(project_root, kind)
        left = pd.Timestamp(start.isoformat(), tz="UTC")
        right = pd.Timestamp(end.isoformat(), tz="UTC")
        values = frame.loc[(frame.index >= left) & (frame.index < right), list(COLUMNS[kind])].copy()
        source_modified = _file_updated_utc(project_root / DATA_FILES[kind])
        if values.empty:
            return {"kind": kind, "interval": interval, "points": [], "count": 0,
                    "source_file_modified_utc": source_modified}
        values = values.apply(pd.to_numeric, errors="coerce")
        if interval != "15min":
            values = values.resample("1h" if interval == "hour" else "1D").mean()
        if len(values) > 3000:
            raise HTTPException(422, "Too many points. Select a shorter date range or aggregate by hour/day.")
        # Return null, not NaN, for source-data gaps.
        points = [
            {"timestamp_utc": timestamp.isoformat(), **{
                col: None if pd.isna(value) else round(float(value), 3)
                for col, value in row.items()
            }}
            for timestamp, row in values.iterrows()
        ]
        return {"kind": kind, "interval": interval, "count": len(points),
                "source_file_modified_utc": source_modified, "points": points}

    @api.get("/api/insights")
    def insights(
        kind: Literal["history", "test", "recent"] = "test",
        start: date = Query(..., description="Inclusive UTC date, YYYY-MM-DD"),
        end: date = Query(..., description="Exclusive UTC date, YYYY-MM-DD"),
    ):
        """Explain the selected window using unaggregated 15-minute observations.

        This is deliberately descriptive: an observed relationship is not a
        causal explanation or evidence of future forecasting performance.
        """
        if end <= start:
            raise HTTPException(422, "End date must be after start date")
        frame = _load(project_root, kind)
        left = pd.Timestamp(start.isoformat(), tz="UTC")
        right = pd.Timestamp(end.isoformat(), tz="UTC")
        selection = frame.loc[(frame.index >= left) & (frame.index < right)].copy()
        context = {
            "kind": kind,
            "start_utc": left.isoformat(),
            "end_utc_exclusive": right.isoformat(),
            "resolution": "15min",
            "total_intervals": len(selection),
        }
        if selection.empty:
            return {
                **context,
                "status": "no_data",
                "headline": "No observations in this period",
                "metrics": [],
                "details": ["Select dates within the available historical data or test period."],
                "implication": "No conclusions can be drawn from an empty period.",
            }

        def fmt_mw(value: float) -> str:
            return f"{value:,.1f} MW"

        def fmt_stamp(value: pd.Timestamp) -> str:
            return value.strftime("%Y-%m-%d %H:%M UTC")

        if kind in ("history", "recent"):
            demand = pd.to_numeric(selection["demand_mw"], errors="coerce")
            demand = demand.replace([np.inf, -np.inf], np.nan)
            valid = demand.dropna()
            missing = int(demand.isna().sum())
            if valid.empty:
                return {
                    **context, "status": "no_measurements",
                    "headline": "No measured demand values in this period",
                    "metrics": [],
                    "details": [f"{missing:,} time slots have no valid source measurement."],
                    "implication": "Demand cannot be summarized without observed measurements.",
                }
            mean = float(valid.mean())
            low = float(valid.min())
            high = float(valid.max())
            return {
                **context, "status": "ok",
                "headline": f"Average measured demand: {fmt_mw(mean)}",
                "metrics": [
                    {"label": "Average demand", "value": fmt_mw(mean)},
                    {"label": "Highest demand", "value": fmt_mw(high)},
                    {"label": "Data coverage", "value": f"{len(valid) / len(selection) * 100:.1f}%"},
                ],
                "details": [
                    f"Highest recorded demand was {fmt_mw(high)} at {fmt_stamp(valid.idxmax())}.",
                    f"Lowest recorded demand was {fmt_mw(low)} at {fmt_stamp(valid.idxmin())}; the observed range was {fmt_mw(high - low)}.",
                    f"{len(valid):,} of {len(selection):,} available 15-minute slots contain measured demand; {missing:,} are missing or invalid.",
                ],
                "implication": "These describe measured demand only. Fresh measurements are not forward-looking forecasts." if kind == "recent" else "These describe historical demand variation, not the cause of changes or a forecast of future demand.",
            }

        values = selection[["actual_mw", "model_mw", "selected_baseline_mw"]].apply(pd.to_numeric, errors="coerce")
        paired = values.replace([np.inf, -np.inf], np.nan).dropna()
        if paired.empty:
            return {
                **context, "status": "no_pairs",
                "headline": "No comparable forecasts in this period",
                "metrics": [],
                "details": ["No 15-minute timestamps have all three values: actual, LightGBM, and selected baseline."],
                "implication": "A forecast comparison needs overlapping measured and predicted values.",
            }
        error = paired["model_mw"] - paired["actual_mw"]
        baseline_error = paired["selected_baseline_mw"] - paired["actual_mw"]
        model_mae = float(error.abs().mean())
        baseline_mae = float(baseline_error.abs().mean())
        delta = baseline_mae - model_mae
        improvement = 100 * delta / baseline_mae if baseline_mae > 0 else None
        if delta > 0.05:
            verdict = f"LightGBM had {fmt_mw(delta)} lower MAE than the baseline in this period"
        elif delta < -0.05:
            verdict = f"The baseline had {fmt_mw(-delta)} lower MAE than LightGBM in this period"
        else:
            verdict = "LightGBM and the baseline had similar MAE in this period"
        bias = float(error.mean())
        bias_text = (
            f"mean overprediction of {fmt_mw(bias)}" if bias > 0.05 else
            f"mean underprediction of {fmt_mw(-bias)}" if bias < -0.05 else
            "nearly zero net average bias"
        )
        worst = error.abs()
        return {
            **context,
            "status": "ok",
            "paired_intervals": len(paired),
            "headline": verdict,
            "metrics": [
                {"label": "LightGBM MAE", "value": fmt_mw(model_mae)},
                {"label": "Baseline MAE", "value": fmt_mw(baseline_mae)},
                {"label": "MAE difference", "value": f"{improvement:+.1f}%" if improvement is not None else "n/a"},
            ],
            "details": [
                f"Comparison uses {len(paired):,} common 15-minute timestamps (of {len(selection):,} available); chart aggregation does not affect these error metrics.",
                f"LightGBM showed {bias_text} across this window. Positive and negative errors can cancel.",
                f"The largest absolute LightGBM error was {fmt_mw(float(worst.max()))} at {fmt_stamp(worst.idxmax())}.",
            ],
            "implication": "The result describes this historical backtest window only. It does not establish a cause or demonstrate performance on future live forecasts.",
        }

    return api


app = create_app()
