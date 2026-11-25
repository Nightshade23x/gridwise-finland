"""Fair, leakage-safe 24-hour-ahead demand forecast baselines.

Uses only previously observed demand: 24 hours or 7 days before each
prediction timestamp. No model learns from the evaluation period.
"""
from __future__ import annotations

from datetime import timedelta

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ingest import ROOT

COLUMNS = {
    'yesterday': 'pred_yesterday_mw',
    'last_week': 'pred_last_week_mw',
    'blend_50_50': 'pred_blend_mw',
}


def make_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Build 24-hour-ahead forecasts on a complete 15-minute UTC grid.

    The original target is never imputed. A missing lag yields a missing
    forecast, so we can compare methods fairly on the same valid rows.
    """
    required = {'timestamp_utc', 'demand_mw'}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing prepared-data columns: {sorted(required - set(frame.columns))}")

    result = frame[['timestamp_utc', 'demand_mw']].copy()
    result['timestamp_utc'] = pd.to_datetime(result['timestamp_utc'], utc=True)
    result['demand_mw'] = pd.to_numeric(result['demand_mw'], errors='raise')
    result = result.sort_values('timestamp_utc').reset_index(drop=True)
    idx = pd.DatetimeIndex(result['timestamp_utc'])
    if idx.has_duplicates or len(idx) < 673 or not (idx[1:] - idx[:-1] == timedelta(minutes=15)).all():
        raise ValueError('Prepared input must be a continuous, unique 15-minute UTC grid with at least 673 rows')

    result['pred_yesterday_mw'] = result['demand_mw'].shift(96)
    result['pred_last_week_mw'] = result['demand_mw'].shift(672)
    result['pred_blend_mw'] = (result['pred_yesterday_mw'] + result['pred_last_week_mw']) / 2
    return result


def evaluate(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    """Calculate fair comparison metrics on common valid target/prediction rows."""
    sample = frame.loc[(frame['timestamp_utc'] >= start) & (frame['timestamp_utc'] < end)].copy()
    if sample.empty:
        raise ValueError(f'No observations in evaluation range {start} through {end}')
    valid = sample[['demand_mw', *COLUMNS.values()]].notna().all(axis=1)
    shared = sample.loc[valid].copy()
    if shared.empty:
        raise ValueError('No complete forecast pairs: check gaps or chosen date ranges')
    observed = shared['demand_mw'].to_numpy(dtype=float)
    metrics = {}
    for name, col in COLUMNS.items():
        errors = shared[col].to_numpy(dtype=float) - observed
        metrics[name] = {
            'mae_mw': round(float(np.abs(errors).mean()), 3),
            'rmse_mw': round(float(np.sqrt((errors ** 2).mean())), 3),
        }
    return {
        'start_utc': start.isoformat(),
        'end_utc_exclusive': end.isoformat(),
        'total_intervals': len(sample),
        'paired_evaluation_intervals': len(shared),
        'excluded_due_to_missing_target_or_lag': int(len(sample) - len(shared)),
        'metrics': metrics,
    }, shared


def benchmark(
    prepared: pd.DataFrame,
    validation_start: str = '2026-01-01',
    test_start: str = '2026-07-01',
    end: str = '2026-11-01',
) -> tuple[dict, pd.DataFrame]:
    predictions = make_predictions(prepared)
    val_start = pd.Timestamp(validation_start, tz='UTC')
    test_start_ts = pd.Timestamp(test_start, tz='UTC')
    final_end = pd.Timestamp(end, tz='UTC')
    first = predictions['timestamp_utc'].iloc[0]
    data_end = predictions['timestamp_utc'].iloc[-1] + timedelta(minutes=15)
    if not first < val_start < test_start_ts < final_end <= data_end:
        raise ValueError('Expected data to span: historical context < validation < test < end')

    validation, _ = evaluate(predictions, val_start, test_start_ts)
    # Select using VALIDATION only; the held-out test is untouched for selection.
    winner = min(COLUMNS, key=lambda key: validation['metrics'][key]['mae_mw'])
    test, test_rows = evaluate(predictions, test_start_ts, final_end)
    test_rows['selected_prediction_mw'] = test_rows[COLUMNS[winner]]
    report = {
        'task': 'Forecast demand exactly 24 hours before each target 15-minute timestamp',
        'selection_rule': 'Lowest validation MAE on shared available timestamps',
        'baseline_definitions': {
            'yesterday': 'Demand at t - 96 intervals (24 hours earlier)',
            'last_week': 'Demand at t - 672 intervals (7 days earlier)',
            'blend_50_50': 'Equal average of yesterday and last-week baselines',
        },
        'validation': validation,
        'selected_baseline': winner,
        'test': test,
        'note': 'All baseline predictors rely only on historical demand; test used once after selection.',
    }
    return report, test_rows


def save_comparison_plot(test_rows: pd.DataFrame, path: Path) -> None:
    first = test_rows['timestamp_utc'].iloc[0]
    week = test_rows.loc[test_rows['timestamp_utc'] < first + timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(week['timestamp_utc'], week['demand_mw'], label='Actual', linewidth=1.6)
    ax.plot(week['timestamp_utc'], week['selected_prediction_mw'], label='Selected baseline', linewidth=1.2)
    ax.set(title='24-hour-ahead demand baseline vs actual (first test week)', ylabel='Demand (MW)', xlabel='UTC time')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=155, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate leakage-safe 24-hour-ahead demand baselines')
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/consumption_2024_2026.csv')
    parser.add_argument('--validation-start', default='2026-01-01')
    parser.add_argument('--test-start', default='2026-07-01')
    parser.add_argument('--end', default='2026-11-01')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/baselines_2026.json')
    parser.add_argument('--predictions', type=Path, default=ROOT / 'data/model_outputs/baseline_predictions_2026.csv')
    parser.add_argument('--plot', type=Path, default=ROOT / 'data/model_outputs/baseline_first_week.png')
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    report, test_rows = benchmark(frame, args.validation_start, args.test_start, args.end)
    for path in [args.report, args.predictions, args.plot]:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    test_rows.to_csv(args.predictions, index=False)
    save_comparison_plot(test_rows, args.plot)

    print('24-hour-ahead baseline evaluation (MW)')
    print(f"Validation: {report['validation']['paired_evaluation_intervals']:,} paired observations")
    for name, row in report['validation']['metrics'].items():
        print(f"  {name:12} MAE: {row['mae_mw']:>9.3f} | RMSE: {row['rmse_mw']:>9.3f}")
    print(f"Selected by validation: {report['selected_baseline']}")
    print(f"Held-out test: {report['test']['paired_evaluation_intervals']:,} paired observations")
    for name, row in report['test']['metrics'].items():
        print(f"  {name:12} MAE: {row['mae_mw']:>9.3f} | RMSE: {row['rmse_mw']:>9.3f}")
    print(f'Saved report: {args.report}')
    print(f'Saved predictions: {args.predictions}')
    print(f'Saved plot: {args.plot}')


if __name__ == '__main__':
    main()
