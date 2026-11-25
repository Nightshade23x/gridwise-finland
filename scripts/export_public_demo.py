"""Export a small, read-only portfolio demo from locally measured public data.

Does not copy private API credentials, historical raw downloads, trained model
weights, refresh status, or issued forecast archives. Input data must already
exist from the standard GridWise workflow.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

SOURCES = {
    "history": Path("data/processed/consumption_2024_2026.csv"),
    "test": Path("data/model_outputs/lightgbm_predictions_2026.csv"),
    "baselines": Path("data/model_outputs/baseline_predictions_2026.csv"),
}

# Small winter examples plus the full July-October historical test window.
WINDOWS = (
    ("2024-01-01", "2024-01-08"),
    ("2025-01-01", "2025-01-08"),
    ("2026-07-01", "2026-11-01"),
)
TEST_START = pd.Timestamp("2026-07-01", tz="UTC")
TEST_END = pd.Timestamp("2026-11-01", tz="UTC")


def export_demo(root: Path, dest: Path) -> dict:
    """Copy only approved historical CSV subsets, never private/runtime files."""
    root = Path(root).resolve()
    dest = Path(dest).resolve()
    frames = {}
    for kind, relative in SOURCES.items():
        src = root / relative
        if not src.is_file():
            raise FileNotFoundError(f"Missing {relative}; generate the existing GridWise outputs first")
        frame = pd.read_csv(src)
        needed = {"timestamp_utc"}
        needed |= {"history": {"demand_mw"},
                   "test": {"actual_mw", "model_mw", "selected_baseline_mw"},
                   "baselines": {"pred_yesterday_mw", "pred_last_week_mw"}}[kind]
        if not needed.issubset(frame.columns):
            raise ValueError(f"Missing expected columns in {relative}: {sorted(needed-set(frame.columns))}")
        ts = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
        if ts.isna().any() or ts.duplicated().any():
            raise ValueError(f"Invalid or duplicate timestamp in {relative}")
        if kind == "history":
            allowed = pd.Series(False, index=frame.index)
            for start, end in WINDOWS:
                allowed |= (ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(end, tz="UTC"))
        else:
            allowed = (ts >= TEST_START) & (ts < TEST_END)
        selected = frame.loc[allowed, sorted(needed - {"timestamp_utc"})].copy()
        selected.insert(0, "timestamp_utc", ts.loc[allowed].map(lambda value: value.isoformat()))
        selected = selected.sort_values("timestamp_utc").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"No eligible rows in {relative}")
        frames[kind] = selected
    # The full available test set is kept, rather than cherry-picking good weeks.
    if len(frames["test"]) < 1000:
        raise ValueError("Too few test predictions: do not publish a misleading partial benchmark")
    outfiles = {
        "history": Path("data/processed/consumption_2024_2026.csv"),
        "test": Path("data/model_outputs/lightgbm_predictions_2026.csv"),
        "baselines": Path("data/model_outputs/baseline_predictions_2026.csv"),
    }
    for kind, relative in outfiles.items():
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[kind].to_csv(path, index=False)
    manifest = {
        "schema_version": 1,
        "kind": "static_historical_demo",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Fingrid Open Data, consumption dataset 124; weather-enhanced model predictions are retrospective research outputs",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "history_windows_utc": [{"start": a, "end_exclusive": b} for a,b in WINDOWS],
        "test_window_utc": {"start": TEST_START.isoformat(), "end_exclusive": TEST_END.isoformat()},
        "row_counts": {key: len(value) for key, value in frames.items()},
        "refresh_enabled": False,
        "forward_forecast_included": False,
        "caution": "Static research snapshot; not a live forecast or proof of future accuracy.",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("demo_bundle"))
    args = parser.parse_args()
    result = export_demo(args.root, args.output)
    print("Exported public historical demo with actual data, no API key or live forecast:")
    for key, rows in result["row_counts"].items():
        print(f"  {key}: {rows:,} rows")
    print(f"Bundle: {args.output.resolve()}")
    print("Review bundle contents before committing them to your public GitHub repository.")


if __name__ == "__main__":
    main()
