"""Phase 21: issuance-safe, model-free level-shift calibration diagnostic.

Hypothesis: 48-hour/7-day demand persistence can lag sudden demand-level
changes. Compare uncorrected references with a capped, decaying correction
computed solely from measurements at least one hour old at each simulated
issuance. This is a RESEARCH CANDIDATE: do not rewrite issued forecasts.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .forward import BASELINES, INTERVAL, combined_demand, feature_table, utc_timestamp
from .refresh import ROOT, atomic_json

HISTORY = Path("data/processed/consumption_2024_2026.csv")
REPORT = Path("data/live/level_shift_backtest.json")
CHART = Path("data/model_outputs/level_shift_backtest.png")
PREVIEW = Path("data/live/level_shift_preview.json")
CAL_START = pd.Timestamp("2026-01-01T00:00:00Z")
CAL_END = pd.Timestamp("2026-03-31T00:00:00Z")
CHECK_START = pd.Timestamp("2026-04-01T00:00:00Z")
CHECK_END = pd.Timestamp("2026-06-30T00:00:00Z")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DECAYS = (6.0, 24.0)
OFFSET_CAP_MW = 1500.0
ANCHOR_HOURS = 6
BUFFER_MINUTES = 60
MIN_ANCHOR_ROWS = 16
MIN_TARGET_ROWS = 80


def prepared_history(root: Path) -> pd.DataFrame:
    path = root / HISTORY
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared historical demand: {path}")
    data = pd.read_csv(path, usecols=["timestamp_utc", "demand_mw"])
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)
    return data


def _valid(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)


def _origin_window(features: pd.DataFrame, issue: pd.Timestamp) -> tuple[pd.DataFrame | None, str | None]:
    """Build ONE issue's shared evaluation rows; no future values form features."""
    t = features["timestamp_utc"]
    anchor = features.loc[(t > issue - timedelta(hours=ANCHOR_HOURS)) &
                          (t <= issue - timedelta(minutes=BUFFER_MINUTES))].copy()
    targets = features.loc[(t > issue) & (t <= issue + timedelta(hours=24))].copy()
    if len(targets) != 96:
        return None, "incomplete_target_grid"
    if len(anchor) < MIN_ANCHOR_ROWS:
        return None, "insufficient_anchor_grid"
    if len(anchor.loc[_valid(anchor, ["demand_mw", *BASELINES])]) < MIN_ANCHOR_ROWS:
        return None, "missing_anchor_observations"
    target_good = targets.loc[_valid(targets, ["demand_mw", *BASELINES])].copy()
    if len(target_good) < MIN_TARGET_ROWS:
        return None, "missing_target_observations"
    good_anchor = anchor.loc[_valid(anchor, ["demand_mw", *BASELINES])]
    target_good["issue_utc"] = issue
    target_good["lead_hours"] = (target_good["timestamp_utc"] - issue).dt.total_seconds() / 3600
    target_good["origin_anchor_count"] = len(good_anchor)
    for baseline in BASELINES:
        # Same-input comparisons: anchor observations are known at issue-60m.
        offset = float(np.median(good_anchor["demand_mw"] - good_anchor[baseline]))
        target_good[f"offset_{baseline}_mw"] = float(np.clip(offset, -OFFSET_CAP_MW, OFFSET_CAP_MW))
    # Every baseline predictor for the latest target uses t-48h or older.
    assert target_good["timestamp_utc"].max() - timedelta(hours=48) <= issue - timedelta(hours=24)
    assert good_anchor["timestamp_utc"].max() <= issue - timedelta(minutes=BUFFER_MINUTES)
    return target_good, None


def backtest_windows(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                     *, origin_hour: int = 12) -> tuple[pd.DataFrame, dict]:
    """Simulate non-overlapping daily 24h issuances, with common target rows."""
    if not 0 <= origin_hour < 24:
        raise ValueError("origin_hour must be in [0,23]")
    begin, stop = utc_timestamp(start), utc_timestamp(end)
    if begin >= stop:
        raise ValueError("End must be later than start")
    features = feature_table(data).set_index("timestamp_utc", drop=False)
    origins = pd.date_range(begin.normalize() + timedelta(hours=origin_hour),
                            stop.normalize(), freq="D", tz="UTC", inclusive="left")
    windows, skipped = [], {}
    for issue in origins:
        window, reason = _origin_window(features, issue)
        if window is None:
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            windows.append(window[["timestamp_utc", "issue_utc", "lead_hours", "demand_mw",
                                   *BASELINES, *[f"offset_{b}_mw" for b in BASELINES],
                                   "origin_anchor_count"]])
    if not windows:
        raise ValueError("No eligible daily issuance windows; inspect historical coverage")
    result = pd.concat(windows, ignore_index=True)
    return result, {"attempted_issuances": len(origins), "valid_issuances": len(windows),
                    "excluded_issuances": skipped, "paired_target_rows": len(result),
                    "origin_hour_utc": origin_hour}


def prediction(rows: pd.DataFrame, baseline: str, alpha: float, decay_hours: float) -> np.ndarray:
    if baseline not in BASELINES or not (0 <= alpha <= 1) or decay_hours <= 0:
        raise ValueError("Invalid baseline, correction strength, or decay")
    old = rows[baseline].to_numpy(dtype=float)
    offset = rows[f"offset_{baseline}_mw"].to_numpy(dtype=float)
    lead = rows["lead_hours"].to_numpy(dtype=float)
    if not (np.isfinite(old).all() and np.isfinite(offset).all() and
            np.isfinite(lead).all() and (lead > 0).all() and (lead <= 24).all()):
        raise ValueError("Unusable reference, correction offset or lead")
    return np.maximum(0, old + alpha * offset * np.exp(-lead / decay_hours))


def metrics(actual, predicted) -> dict:
    truth = np.asarray(actual, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != estimate.shape or not len(truth) or not (
            np.isfinite(truth).all() and np.isfinite(estimate).all()):
        raise ValueError("Metrics require finite, matching nonempty arrays")
    err = estimate - truth
    return {"mae_mw": round(float(np.mean(np.abs(err))), 3),
            "rmse_mw": round(float(np.sqrt(np.mean(err ** 2))), 3),
            "bias_mw": round(float(np.mean(err)), 3),
            "paired_targets": len(truth)}


def _parameter_grid():
    for baseline in BASELINES:
        yield baseline, 0.0, 24.0
        for alpha in ALPHAS[1:]:
            for decay in DECAYS:
                yield baseline, alpha, decay


def evaluate(data: pd.DataFrame) -> dict:
    """Select Jan-Mar 2026; independently check Apr-Jun on chronologically later origins.

    Both periods were previously studied by other GridWise experiments, so
    this is a development diagnostic, NOT a fresh independent test.
    """
    cal, cal_coverage = backtest_windows(data, CAL_START, CAL_END)
    check, check_coverage = backtest_windows(data, CHECK_START, CHECK_END)
    ground = cal["demand_mw"].to_numpy(dtype=float)
    scored = []
    for baseline, alpha, decay in _parameter_grid():
        m = metrics(ground, prediction(cal, baseline, alpha, decay))
        scored.append((m["mae_mw"], alpha, decay, baseline, m))
    selected = min(scored, key=lambda x: (x[0], x[1], x[2], x[3]))
    _, alpha, decay, baseline, cal_metrics = selected
    actual_check = check["demand_mw"].to_numpy(dtype=float)
    candidate = metrics(actual_check, prediction(check, baseline, alpha, decay))
    plain = metrics(actual_check, prediction(check, baseline, 0, decay))
    other = {name: metrics(actual_check, prediction(check, name, 0, 24))
             for name in BASELINES}
    month_summaries = []
    for month, group in check.groupby(check["issue_utc"].dt.strftime("%Y-%m"), sort=True):
        month_summaries.append({"month": month, "issuances": group["issue_utc"].nunique(),
                                "candidate": metrics(group["demand_mw"], prediction(group, baseline, alpha, decay)),
                                "uncorrected_reference": metrics(group["demand_mw"], group[baseline])})
    delta = round(100 * (plain["mae_mw"]-candidate["mae_mw"])/plain["mae_mw"], 2) if plain["mae_mw"] else None
    return {"schema_version": 1, "kind": "issue_time_level_shift_diagnostic",
            "status": "evaluated", "evaluated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "selection_period": {"start_utc": CAL_START.isoformat(), "end_exclusive_utc": CAL_END.isoformat(),
                                 "coverage": cal_coverage, "selected_mae": cal_metrics},
            "later_period": {"start_utc": CHECK_START.isoformat(), "end_exclusive_utc": CHECK_END.isoformat(),
                             "coverage": check_coverage, "candidate": candidate,
                             "uncorrected_reference": plain, "other_uncorrected_references": other,
                             "mae_change_percent": delta, "by_month": month_summaries},
            "selected_parameters": {"baseline": baseline, "alpha": alpha, "decay_hours": decay,
                                    "offset_cap_mw": OFFSET_CAP_MW, "anchor_hours": ANCHOR_HOURS,
                                    "measurement_buffer_minutes": BUFFER_MINUTES},
            "notes": ["Model-free diagnostic of demand level shifts; no LightGBM retraining.",
                      "Daily UTC 12:00 origins simulate 96 predictions for the following 24h.",
                      "Only observed values at least 60 minutes old can influence each correction.",
                      "Correction strength and decay were chosen using Jan-Mar only.",
                      "Apr-Jun is a later-period check, but has been examined in earlier GridWise development.",
                      "All strategies use identical measured target timestamps; missing targets are never filled.",
                      "This experiment does NOT change issued GridWise forecasts, verifiers, or the public demo."]}


def save_figure(report: dict, path: Path) -> None:
    """Generate a compact comparison for the development dashboard; no web CDN."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    after = report["later_period"]
    methods = ["Uncorrected reference", "Capped correction"]
    maes = [after["uncorrected_reference"]["mae_mw"], after["candidate"]["mae_mw"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(methods, maes)
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Phase 21: chronological later-period diagnostic (Apr–Jun 2026)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def live_preview(data: pd.DataFrame, issue_time, parameters: dict, *, max_age_hours: float = 6.0) -> dict:
    """Optional *separate* current-window proposal; never rewrite issued archives."""
    issue = utc_timestamp(issue_time)
    if max_age_hours <= 0:
        raise ValueError("max_age_hours must be positive")
    first = issue.floor("15min") + INTERVAL
    f = feature_table(data)
    target = pd.date_range(first, periods=96, freq=INTERVAL, tz="UTC")
    latest_allowed = issue - timedelta(minutes=BUFFER_MINUTES)
    known = f.loc[(f["timestamp_utc"] <= latest_allowed) & f["demand_mw"].notna()]
    if known.empty or (issue - known["timestamp_utc"].max()).total_seconds()/3600 > max_age_hours:
        raise ValueError("Recent observations too old; run gridwise.refresh first")
    # A source file may include observations newer than the issue. The preview
    # will NEVER use them to construct the calibration offset.
    baseline = parameters["baseline"]
    alpha, decay = float(parameters["alpha"]), float(parameters["decay_hours"])
    anchor = known.loc[(known["timestamp_utc"] > issue - timedelta(hours=ANCHOR_HOURS)) &
                       _valid(known, ["demand_mw", baseline])]
    if len(anchor) < MIN_ANCHOR_ROWS:
        raise ValueError("Not enough issue-time anchor readings")
    # For the future, compute lagged features from an explicitly masked demand
    # series, rather than using actual targets in the input, even by accident.
    masked = data[["timestamp_utc", "demand_mw"]].copy()
    masked["timestamp_utc"] = pd.to_datetime(masked["timestamp_utc"], utc=True)
    masked.loc[masked["timestamp_utc"] > latest_allowed, "demand_mw"] = np.nan
    full = pd.date_range(masked["timestamp_utc"].min(), target[-1], freq=INTERVAL, tz="UTC")
    masked = masked.set_index("timestamp_utc").reindex(full).rename_axis("timestamp_utc").reset_index()
    future = feature_table(masked).set_index("timestamp_utc").loc[target].copy()
    if not _valid(future, [baseline]).all():
        raise ValueError("Required 48h/7d reference history missing for future targets")
    offset = float(np.clip(np.median(anchor["demand_mw"]-anchor[baseline]), -OFFSET_CAP_MW, OFFSET_CAP_MW))
    future[f"offset_{baseline}_mw"] = offset
    future["lead_hours"] = (future.index - issue).total_seconds()/3600
    corrected = prediction(future, baseline, alpha, decay)
    return {"kind": "level_shift_preview", "status": "research_only",
            "issued_at_utc": issue.isoformat(), "reference": baseline,
            "source_latest_observation_utc": known["timestamp_utc"].max().isoformat(),
            "anchor_count": len(anchor), "offset_mw": round(offset, 3),
            "parameters": parameters, "points": [
                {"timestamp_utc": t.isoformat(), "uncorrected_mw": float(round(raw, 3)),
                 "candidate_mw": float(round(corr, 3))}
                for t, raw, corr in zip(target, future[baseline], corrected)]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Issuance-safe 24h demand level-shift calibration diagnostic")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--preview-now", action="store_true", help="Also issue a separate research preview from recent measurements")
    args = parser.parse_args()
    root = args.root.resolve()
    report = evaluate(prepared_history(root))
    atomic_json(root / REPORT, report)
    save_figure(report, root / CHART)
    sel = report["selected_parameters"]
    cal = report["selection_period"]["selected_mae"]
    later = report["later_period"]
    print(f"Calibrated on {report['selection_period']['coverage']['valid_issuances']} daily origins (Jan-Mar 2026)")
    print(f"Chosen: {sel['baseline']}, correction alpha={sel['alpha']}, decay={sel['decay_hours']}h")
    print(f"Calibration candidate MAE: {cal['mae_mw']:.1f} MW")
    print(f"Later chronological check: {later['coverage']['valid_issuances']} origins, {later['coverage']['paired_target_rows']} common targets")
    print(f"Apr-Jun reference MAE {later['uncorrected_reference']['mae_mw']:.1f} MW; "
          f"corrected MAE {later['candidate']['mae_mw']:.1f} MW; "
          f"improvement {later['mae_change_percent']:+.2f}%")
    print(f"Saved: {root / REPORT} and {root / CHART}")
    if args.preview_now:
        preview = live_preview(combined_demand(root), pd.Timestamp.now(tz="UTC"), sel)
        atomic_json(root / PREVIEW, preview)
        print(f"Research preview saved: {root / PREVIEW}")
    print("Does not retrain models, modify existing forecast archives, or update the public demo.")


if __name__ == "__main__":
    main()
