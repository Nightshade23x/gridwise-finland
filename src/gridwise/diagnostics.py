"""Analyze an already-fitted GridWise LightGBM forecast without model selection.

Compares the saved held-out predictions with every historical baseline on the
SAME 15-minute timestamps. No refitting or test-based selection occurs here.
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
from lightgbm import Booster

from .baselines import COLUMNS, make_predictions
from .ingest import ROOT


def metrics(truth: pd.Series, prediction: pd.Series) -> dict:
    y = truth.to_numpy(dtype=float)
    p = prediction.to_numpy(dtype=float)
    residual = p - y
    abs_error = np.abs(residual)
    return {
        "mae_mw": round(float(abs_error.mean()), 3),
        "rmse_mw": round(float(np.sqrt(np.mean(residual ** 2))), 3),
        "bias_mw": round(float(residual.mean()), 3),
        "median_absolute_error_mw": round(float(np.median(abs_error)), 3),
        "p90_absolute_error_mw": round(float(np.percentile(abs_error, 90)), 3),
    }


def diagnose(
    prepared: pd.DataFrame,
    saved_predictions: pd.DataFrame,
    baseline_report: dict,
) -> tuple[dict, pd.DataFrame]:
    """Compare ML against every baseline on paired, nonmissing test rows.

    Reject input with mismatched ground truth or a different validation-selected
    baseline; this avoids inadvertently comparing different datasets/runs.
    """
    needed = {"timestamp_utc", "actual_mw", "model_mw", "selected_baseline_mw"}
    if not needed.issubset(saved_predictions):
        raise ValueError(f"Predictions missing {sorted(needed - set(saved_predictions))}")
    selected = baseline_report.get("selected_baseline")
    if selected not in COLUMNS:
        raise ValueError("Baseline report has no valid validation-selected baseline")

    p = saved_predictions[list(needed)].copy()
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True, errors="raise")
    if p["timestamp_utc"].duplicated().any() or p.empty:
        raise ValueError("Saved predictions must have unique timestamps and at least one row")
    b = make_predictions(prepared)
    shared = p.merge(b, on="timestamp_utc", how="left", validate="one_to_one")

    truth_present = shared["actual_mw"].notna() & shared["demand_mw"].notna()
    if not truth_present.all():
        raise ValueError("Saved predictions contain timestamps without measured demand")
    if not np.allclose(shared["actual_mw"], shared["demand_mw"], atol=0.01, rtol=0):
        raise ValueError("Saved actuals disagree with prepared input; use matching files")
    baseline_col = COLUMNS[selected]
    comparable_baseline = shared["selected_baseline_mw"].notna() & shared[baseline_col].notna()
    if not comparable_baseline.all() or not np.allclose(
        shared["selected_baseline_mw"], shared[baseline_col], atol=0.01, rtol=0
    ):
        raise ValueError("Saved selected baseline disagrees with validation-selected baseline report")
    columns = ["model_mw", *COLUMNS.values()]
    valid = shared[["actual_mw", *columns]].notna().all(axis=1)
    paired = shared.loc[valid].copy().sort_values("timestamp_utc").reset_index(drop=True)
    if paired.empty:
        raise ValueError("No complete rows across model and all baselines")

    methods = {"lightgbm": "model_mw", **COLUMNS}

    def scores(rows: pd.DataFrame) -> dict:
        return {name: metrics(rows["actual_mw"], rows[col]) for name, col in methods.items()}

    overall = scores(paired)
    monthly = {}
    for month, group in paired.groupby(paired["timestamp_utc"].dt.strftime("%Y-%m")):
        monthly[month] = {"rows": len(group), "scores": scores(group)}
    finnish_hour = paired["timestamp_utc"].dt.tz_convert("Europe/Helsinki").dt.hour
    segments = {
        "day_06_18_local": paired.loc[finnish_hour.between(6, 17)],
        "night_18_06_local": paired.loc[~finnish_hour.between(6, 17)],
    }
    by_time = {key: {"rows": len(group), "scores": scores(group)}
               for key, group in segments.items() if not group.empty}
    reference_mae = overall[selected]["mae_mw"]
    model_mae = overall["lightgbm"]["mae_mw"]
    exploratory_best = min(COLUMNS, key=lambda name: overall[name]["mae_mw"])
    report = {
        "purpose": "Post-hoc held-out error diagnostics; NOT model or baseline selection",
        "selected_baseline_from_validation": selected,
        "original_prediction_rows": len(p),
        "paired_rows_all_methods": len(paired),
        "excluded_unpaired_rows": len(p) - len(paired),
        "start_utc": paired["timestamp_utc"].iloc[0].isoformat(),
        "end_utc": paired["timestamp_utc"].iloc[-1].isoformat(),
        "overall": overall,
        "mae_improvement_vs_selected_baseline_percent": round(100 * (reference_mae - model_mae) / reference_mae, 2) if reference_mae else None,
        "best_baseline_on_test_exploratory_only": exploratory_best,
        "monthly": monthly,
        "local_time_segments": by_time,
        "important_notes": [
            "All listed errors use identical paired timestamps for every method.",
            "The selected baseline was fixed on validation; the best test baseline is descriptive only.",
            "This report does not train, retune, or select any forecasting model.",
        ],
    }
    return report, paired


def feature_gain(model_path: Path) -> list[dict]:
    """Read LightGBM split gain from a saved trained model.

    This is a model-usage diagnostic, not a causal feature explanation.
    """
    booster = Booster(model_file=str(model_path))
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    total = float(np.sum(gains))
    output = [{"feature": name, "gain": round(float(gain), 3),
               "gain_percent": round(float(100 * gain / total), 2) if total > 0 else 0.0}
              for name, gain in zip(names, gains, strict=True)]
    return sorted(output, key=lambda row: row["gain"], reverse=True)


def save_feature_plot(features: list[dict], path: Path, top_n: int = 15) -> None:
    leading = list(reversed(features[:top_n]))
    fig, ax = plt.subplots(figsize=(10, max(4, len(leading) * .38)))
    ax.barh([row["feature"] for row in leading], [row["gain_percent"] for row in leading])
    ax.set(title="LightGBM feature importance (split gain)", xlabel="Share of total gain (%)")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_monthly_plot(report: dict, path: Path) -> None:
    rows = report["monthly"]
    months = list(rows)
    names = ["lightgbm", report["selected_baseline_from_validation"],
             report["best_baseline_on_test_exploratory_only"]]
    names = list(dict.fromkeys(names))
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for name in names:
        ax.plot(months, [rows[m]["scores"][name]["mae_mw"] for m in months], marker="o", label=name)
    ax.set(title="Held-out monthly MAE on common timestamps", ylabel="MAE (MW)", xlabel="UTC month")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="GridWise held-out paired diagnostics; no retraining")
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/consumption_2024_2026.csv")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/model_outputs/lightgbm_predictions_2026.csv")
    parser.add_argument("--baseline-report", type=Path, default=ROOT / "reports/baselines_2026.json")
    parser.add_argument("--model", type=Path, default=ROOT / "data/model_outputs/lightgbm_2026.txt")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/diagnostics_2026.json")
    parser.add_argument("--feature-plot", type=Path, default=ROOT / "data/model_outputs/feature_importance_2026.png")
    parser.add_argument("--monthly-plot", type=Path, default=ROOT / "data/model_outputs/monthly_mae_2026.png")
    args = parser.parse_args()

    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    report, _ = diagnose(pd.read_csv(args.input), pd.read_csv(args.predictions), baseline)
    features = feature_gain(args.model)
    report["feature_importance_gain"] = features
    report["feature_importance_note"] = "Split gain reflects fitted-model usage, not causation."
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_feature_plot(features, args.feature_plot)
    save_monthly_plot(report, args.monthly_plot)

    print(f"Compared {report['paired_rows_all_methods']:,} paired test timestamps (of {report['original_prediction_rows']:,} saved rows)")
    for name, metric in report["overall"].items():
        print(f"  {name:12} MAE: {metric['mae_mw']:>9.3f} MW | RMSE: {metric['rmse_mw']:>9.3f} MW")
    print(f"Validation-selected baseline: {report['selected_baseline_from_validation']}")
    print(f"MAE improvement vs selected: {report['mae_improvement_vs_selected_baseline_percent']:.2f}%")
    print("Top five features by LightGBM gain:")
    for row in features[:5]:
        print(f"  {row['feature']}: {row['gain_percent']:.2f}%")
    print(f"Saved report: {args.report}")
    print(f"Saved feature plot: {args.feature_plot}")
    print(f"Saved monthly plot: {args.monthly_plot}")


if __name__ == "__main__":
    main()
