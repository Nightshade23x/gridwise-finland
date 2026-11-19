import numpy as np
import pandas as pd
import pytest

from gridwise.baselines import benchmark, evaluate, make_predictions


def frame(days=45):
    ts = pd.date_range('2025-12-01', periods=96 * days, freq='15min', tz='UTC')
    # Synthetic daily and weekly structure, with an initial trend.
    values = 9000 + np.sin(2 * np.pi * np.arange(len(ts)) / 96) * 800 + np.arange(len(ts)) * 0.1
    return pd.DataFrame({'timestamp_utc': ts, 'demand_mw': values})


def test_day_and_week_are_exact_historical_lags():
    prepared = frame()
    pred = make_predictions(prepared)
    assert pred.loc[700, 'pred_yesterday_mw'] == prepared.loc[604, 'demand_mw']
    assert pred.loc[700, 'pred_last_week_mw'] == prepared.loc[28, 'demand_mw']
    assert pd.isna(pred.loc[671, 'pred_last_week_mw'])


def test_missing_values_are_not_filled_and_rows_are_paired():
    prepared = frame()
    prepared.loc[700, 'demand_mw'] = np.nan
    pred = make_predictions(prepared)
    assert pd.isna(pred.loc[796, 'pred_yesterday_mw'])
    assert pd.isna(pred.loc[1372, 'pred_last_week_mw'])
    report, _ = evaluate(pred, pd.Timestamp('2025-12-10', tz='UTC'), pd.Timestamp('2025-12-20', tz='UTC'))
    assert report['excluded_due_to_missing_target_or_lag'] >= 1


def test_future_target_changes_do_not_change_earlier_prediction():
    prepared = frame()
    original = make_predictions(prepared)
    changed = prepared.copy()
    changed.loc[900, 'demand_mw'] = 999999.0
    later = make_predictions(changed)
    for col in ['pred_yesterday_mw', 'pred_last_week_mw', 'pred_blend_mw']:
        assert original.loc[900, col] == later.loc[900, col]


def test_grid_is_required():
    prepared = frame()
    prepared = prepared.drop(index=20)
    with pytest.raises(ValueError, match='continuous'):
        make_predictions(prepared)


def test_benchmark_selects_from_validation_and_keeps_test_separate():
    prepared = frame()
    report, test = benchmark(prepared, '2025-12-15', '2026-01-01', '2026-01-15')
    assert report['selected_baseline'] in {'yesterday', 'last_week', 'blend_50_50'}
    assert report['validation']['end_utc_exclusive'] == '2026-01-01T00:00:00+00:00'
    assert test['timestamp_utc'].min() == pd.Timestamp('2026-01-01', tz='UTC')
    assert (test['selected_prediction_mw'] == test[{'yesterday': 'pred_yesterday_mw', 'last_week': 'pred_last_week_mw', 'blend_50_50': 'pred_blend_mw'}[report['selected_baseline']]]).all()
