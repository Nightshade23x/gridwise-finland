"""Retrospective availability-buffer stress test for 24h-ahead demand ML.

Refits LightGBM independently under assumed historical-demand availability
buffers; does not measure actual publication latency. The July-Oct 2026 test
period has already been inspected in earlier phases and is NOT a fresh holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .forecast_ml import build_features, model, errors, CORE_FEATURES, DEMAND_FEATURES
from .ingest import ROOT


BUFFER_DEFAULT = (0, 15, 30, 60)


def buffered_features(
    prepared: pd.DataFrame, hourly_weather: pd.DataFrame, buffer_minutes: int
) -> tuple[pd.DataFrame, list[str]]:
    """Move every historical-demand feature back by the same availability buffer.

    This also moves both trailing historical means, not just point lags.
    Target-time calendar features and weather observations lagged by 14 days
    are unaffected by the demand-publication assumption.
    """
    if not isinstance(buffer_minutes, int) or buffer_minutes < 0 or buffer_minutes % 15:
        raise ValueError("Buffer must be a nonnegative multiple of 15 minutes")
    f, weather_features = build_features(prepared, hourly_weather)
    steps = buffer_minutes // 15
    if steps:
        for feature in DEMAND_FEATURES:
            f[feature] = f[feature].shift(steps)
    demand = f["demand_mw"]
    f["baseline_yesterday_mw"] = demand.shift(96 + steps)
    f["baseline_last_week_mw"] = demand.shift(672 + steps)
    f["baseline_blend_mw"] = (
        f["baseline_yesterday_mw"] + f["baseline_last_week_mw"]
    ) / 2
    return f, weather_features


def run_stress_test(
    prepared: pd.DataFrame,
    weather: pd.DataFrame,
    buffers: tuple[int, ...] = BUFFER_DEFAULT,
    validation_start: str = "2026-01-01",
    test_start: str = "2026-07-01",
    end: str = "2026-11-01",
    estimators: int = 350,
) -> tuple[dict, pd.DataFrame]:
    """Select candidate and baseline on validation; report retrospective test.

    Apply a common row mask across all buffer settings before scoring, so
    metrics are paired and can be compared without changing target timestamps.
    """
    if not buffers or len(buffers) != len(set(buffers)):
        raise ValueError("Buffers must be distinct and nonempty")
    if any(not isinstance(b, int) or b < 0 or b % 15 for b in buffers):
        raise ValueError("Buffers must be nonnegative multiples of 15 minutes")
    if estimators < 1:
        raise ValueError("Estimators must be positive")
    trn_end = pd.Timestamp(validation_start, tz="UTC")
    val_end = pd.Timestamp(test_start, tz="UTC")
    test_end = pd.Timestamp(end, tz="UTC")
    frames = {}
    weather_cols = None
    for mins in buffers:
        f, cols = buffered_features(prepared, weather, mins)
        if weather_cols is None:
            weather_cols = cols
        elif cols != weather_cols:
            raise ValueError("Different weather feature schema by buffer")
        frames[mins] = f
    assert weather_cols is not None
    first = next(iter(frames.values()))
    max_time = first["timestamp_utc"].iloc[-1] + pd.Timedelta(minutes=15)
    if not (first["timestamp_utc"].iloc[0] < trn_end < val_end < test_end <= max_time):
        raise ValueError("Train, validation and test windows must be chronological and in range")
    full_features = CORE_FEATURES + weather_cols
    baseline_cols = ["baseline_yesterday_mw", "baseline_last_week_mw", "baseline_blend_mw"]
    necessary = ["demand_mw", *full_features, *baseline_cols]
    eligible = np.ones(len(first), dtype=bool)
    for f in frames.values():
        eligible &= f[necessary].notna().all(axis=1).to_numpy()
    # Explicitly fix the same timestamps for every candidate, baseline, and buffer.
    mask_train = (first["timestamp_utc"] < trn_end).to_numpy() & eligible
    mask_val = ((first["timestamp_utc"] >= trn_end) & (first["timestamp_utc"] < val_end)).to_numpy() & eligible
    mask_test = ((first["timestamp_utc"] >= val_end) & (first["timestamp_utc"] < test_end)).to_numpy() & eligible
    if min(mask_train.sum(), mask_val.sum(), mask_test.sum()) < 50:
        raise ValueError("Insufficient paired rows for train, validation or test")
    pretest_mask = mask_train | mask_val
    per_buffer = {}
    predictions = pd.DataFrame({
        "timestamp_utc": first.loc[mask_test, "timestamp_utc"].reset_index(drop=True),
        "actual_mw": first.loc[mask_test, "demand_mw"].reset_index(drop=True),
    })
    for mins in buffers:
        f = frames[mins]
        trn, val, test = f.loc[mask_train], f.loc[mask_val], f.loc[mask_test]
        feature_sets = {
            "calendar_and_demand": CORE_FEATURES,
            "calendar_demand_weather": full_features,
        }
        validation_models = {}
        for name, cols in feature_sets.items():
            reg = model(estimators)
            reg.fit(trn[cols], trn["demand_mw"])
            validation_models[name] = errors(val["demand_mw"], reg.predict(val[cols]))
        selected_model = min(validation_models, key=lambda k: validation_models[k]["mae_mw"])
        baseline_validation = {
            "yesterday": errors(val["demand_mw"], val["baseline_yesterday_mw"]),
            "last_week": errors(val["demand_mw"], val["baseline_last_week_mw"]),
            "blend_50_50": errors(val["demand_mw"], val["baseline_blend_mw"]),
        }
        selected_baseline = min(baseline_validation, key=lambda k: baseline_validation[k]["mae_mw"])
        final = model(estimators)
        final.fit(f.loc[pretest_mask, feature_sets[selected_model]], f.loc[pretest_mask, "demand_mw"])
        prediction = final.predict(test[feature_sets[selected_model]])
        base_test = {
            "yesterday": errors(test["demand_mw"], test["baseline_yesterday_mw"]),
            "last_week": errors(test["demand_mw"], test["baseline_last_week_mw"]),
            "blend_50_50": errors(test["demand_mw"], test["baseline_blend_mw"]),
        }
        per_buffer[str(mins)] = {
            "validation_models": validation_models,
            "selected_model": selected_model,
            "validation_baselines": baseline_validation,
            "selected_baseline": selected_baseline,
            "test_lightgbm": errors(test["demand_mw"], prediction),
            "test_baselines": base_test,
            "test_selected_baseline": base_test[selected_baseline],
        }
        predictions[f"lightgbm_buffer_{mins}m_mw"] = prediction
        predictions[f"selected_baseline_buffer_{mins}m_mw"] = test[{
            "yesterday": "baseline_yesterday_mw",
            "last_week": "baseline_last_week_mw",
            "blend_50_50": "baseline_blend_mw",
        }[selected_baseline]].to_numpy()
    result = {
        "purpose": "Retrospective stress test of 24-hour-ahead LightGBM with assumed historical-demand publication buffers",
        "warning": "Assumed buffers are not measured release times. July-Oct 2026 was already inspected in prior phases, so these results are diagnostic, not a fresh independent test. Historical values may have been revised.",
        "weather_assumption": "ERA5 observations lagged by 14 days; not an archived forecast or point-in-time weather replay.",
        "validation_start_utc": trn_end.isoformat(),
        "test_start_utc": val_end.isoformat(),
        "end_exclusive_utc": test_end.isoformat(),
        "common_training_rows": int(mask_train.sum()),
        "common_validation_rows": int(mask_val.sum()),
        "common_test_rows": int(mask_test.sum()),
        "buffers_minutes": list(buffers),
        "buffer_results": per_buffer,
    }
    return result, predictions


def save_plot(report: dict, dest: Path) -> None:
    offsets = report["buffers_minutes"]
    models = [report["buffer_results"][str(i)]["test_lightgbm"]["mae_mw"] for i in offsets]
    baselines = [report["buffer_results"][str(i)]["test_selected_baseline"]["mae_mw"] for i in offsets]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(offsets, models, marker="o", label="LightGBM (retrained)")
    ax.plot(offsets, baselines, marker="s", label="Validation-selected baseline")
    ax.set(title="Historical diagnostic: demand-publication buffer sensitivity",
           xlabel="Assumed buffer (minutes)", ylabel="MAE (MW)")
    ax.set_xticks(offsets)
    ax.grid(alpha=0.2)
    ax.legend()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrospective ML robustness to demand availability")
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/consumption_2024_2026.csv")
    parser.add_argument("--weather", type=Path, default=ROOT / "data/raw/open_meteo_temperature_2024_2026.csv")
    parser.add_argument("--buffers", type=int, nargs="+", default=list(BUFFER_DEFAULT))
    parser.add_argument("--validation-start", default="2026-01-01")
    parser.add_argument("--test-start", default="2026-07-01")
    parser.add_argument("--end", default="2026-11-01")
    parser.add_argument("--estimators", type=int, default=350)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/robust_ml_2026.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/model_outputs/robust_ml_predictions_2026.csv")
    parser.add_argument("--plot", type=Path, default=ROOT / "data/model_outputs/robust_ml_sensitivity_2026.png")
    args = parser.parse_args()
    report, output = run_stress_test(
        pd.read_csv(args.input), pd.read_csv(args.weather), tuple(args.buffers),
        args.validation_start, args.test_start, args.end, args.estimators,
    )
    for p in [args.report, args.predictions, args.plot]:
        p.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.to_csv(args.predictions, index=False)
    save_plot(report, args.plot)
    print(f"Common rows: train={report['common_training_rows']:,}, validation={report['common_validation_rows']:,}, test={report['common_test_rows']:,}")
    for mins in report["buffers_minutes"]:
        res = report["buffer_results"][str(mins)]
        print(f"Buffer {mins:>3}m | ML {res['test_lightgbm']['mae_mw']:.3f} MW | "
              f"baseline {res['test_selected_baseline']['mae_mw']:.3f} MW | "
              f"selected model={res['selected_model']}, baseline={res['selected_baseline']}")
    print(f"Saved report: {args.report}")
    print(f"Saved predictions: {args.predictions}")
    print(f"Saved plot: {args.plot}")
    print("Caution: retrospective diagnostic on previously inspected test period; not a fresh holdout.")


if __name__ == "__main__":
    main()
