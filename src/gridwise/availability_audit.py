"""Audit 24-hour-ahead demand persistence under assumed publication delays.

This *does not* prove any particular Fingrid delay. It stress-tests the
assumption that the value at exactly t - 24h is already available at forecast
issuance. All variants are evaluated on identical non-null timestamps.
"""
from __future__ import annotations

from datetime import timedelta

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ingest import ROOT

STEPS_PER_DAY = 96


def audit_availability(
    prepared: pd.DataFrame,
    start: str = "2026-07-01",
    end: str = "2026-11-01",
    buffers_minutes: tuple[int, ...] = (0, 15, 30, 60, 120),
) -> tuple[dict, pd.DataFrame]:
    """Evaluate lagged-demand baselines with availability buffers at 15m steps.

    All buffers use a common set of target timestamps. The last available
    observation under a buffer of B minutes is at t - 24h - B.
    """
    if not buffers_minutes or len(set(buffers_minutes)) != len(buffers_minutes):
        raise ValueError("Supply a nonempty list of distinct buffers")
    if any(not isinstance(b, int) or b < 0 or b % 15 for b in buffers_minutes):
        raise ValueError("Buffers must be nonnegative multiples of 15 minutes")
    needed = {"timestamp_utc", "demand_mw"}
    if not needed.issubset(prepared.columns):
        raise ValueError(f"Missing columns: {sorted(needed - set(prepared.columns))}")
    df = prepared[["timestamp_utc", "demand_mw"]].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="raise")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    idx = pd.DatetimeIndex(df["timestamp_utc"])
    if len(df) < STEPS_PER_DAY + max(buffers_minutes) // 15 + 1:
        raise ValueError("Not enough observations for selected buffers")
    if idx.has_duplicates or not (idx[1:] - idx[:-1] == timedelta(minutes=15)).all():
        raise ValueError("Expected unique continuous 15-minute UTC grid")
    begin, finish = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if not begin < finish:
        raise ValueError("Start must precede end")
    output_cols = []
    for mins in buffers_minutes:
        col = f"pred_buffer_{mins}m_mw"
        df[col] = df["demand_mw"].shift(STEPS_PER_DAY + mins // 15)
        output_cols.append(col)
    evaluation = df.loc[(df["timestamp_utc"] >= begin) & (df["timestamp_utc"] < finish)]
    paired = evaluation.loc[evaluation[["demand_mw", *output_cols]].notna().all(axis=1)].copy()
    if paired.empty:
        raise ValueError("No common valid timestamps for selected date range and buffers")
    outcome = {}
    for mins, col in zip(buffers_minutes, output_cols):
        delta = paired[col].to_numpy(dtype=float) - paired["demand_mw"].to_numpy(dtype=float)
        outcome[str(mins)] = {
            "mae_mw": round(float(np.mean(np.abs(delta))), 3),
            "rmse_mw": round(float(np.sqrt(np.mean(delta ** 2))), 3),
        }
    report = {
        "purpose": "Sensitivity analysis of yesterday-persistence predictions to assumed observation publication delays",
        "target_horizon_hours": 24,
        "assumption": "A buffer of B minutes permits demand measurements only at or before target timestamp minus 24h minus B; this is not a measurement of actual publication latency.",
        "evaluation_start_utc": begin.isoformat(),
        "evaluation_end_exclusive_utc": finish.isoformat(),
        "requested_intervals": len(evaluation),
        "common_paired_intervals": len(paired),
        "buffers_minutes": list(buffers_minutes),
        "metrics": outcome,
        "caution": "Historical target and input values can have revisions; this is not a point-in-time replay of actual API vintages. The previously reported LightGBM score does not account for these buffers.",
    }
    return report, paired


def plot(report: dict, filename: Path) -> None:
    times = report["buffers_minutes"]
    values = [report["metrics"][str(b)]["mae_mw"] for b in times]
    fig, ax = plt.subplots(figsize=(7.6, 4))
    ax.plot(times, values, marker="o")
    ax.set(title="Yesterday-persistence sensitivity to observation availability",
           xlabel="Assumed publication buffer (minutes)", ylabel="MAE (MW)")
    ax.set_xticks(times)
    ax.grid(alpha=0.25)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit demand-forecast availability assumptions")
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/consumption_2024_2026.csv")
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-11-01")
    parser.add_argument("--buffers", type=int, nargs="+", default=[0, 15, 30, 60, 120])
    parser.add_argument("--report", type=Path, default=ROOT / "reports/availability_sensitivity_2026.json")
    parser.add_argument("--plot", type=Path, default=ROOT / "data/model_outputs/availability_sensitivity_2026.png")
    args = parser.parse_args()
    report, _ = audit_availability(pd.read_csv(args.input), args.start, args.end, tuple(args.buffers))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(report, args.plot)
    print(f"Evaluated {report['common_paired_intervals']:,} identical target timestamps")
    for b in report["buffers_minutes"]:
        m = report["metrics"][str(b)]
        print(f"  Buffer {b:>3} min | MAE {m['mae_mw']:>8.3f} MW | RMSE {m['rmse_mw']:>8.3f} MW")
    print(f"Saved report: {args.report}")
    print(f"Saved plot: {args.plot}")
    print("Note: This is a sensitivity analysis, not a verified publication-latency measurement.")


if __name__ == "__main__":
    main()
