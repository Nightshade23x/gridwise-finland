"""Finland local calendar selection and DST regression tests, no network required."""
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.failure_analysis import diagnose_issuance


def _client(tmp_path: Path, utc_start: str, utc_end: str):
    (tmp_path / 'data/processed').mkdir(parents=True)
    (tmp_path / 'data/model_outputs').mkdir(parents=True)
    times = pd.date_range(utc_start, utc_end, freq='15min', inclusive='left', tz='UTC')
    pd.DataFrame({'timestamp_utc': times, 'demand_mw': 123.0}).to_csv(
        tmp_path / 'data/processed/consumption_2024_2026.csv', index=False
    )
    pd.DataFrame({'timestamp_utc': times, 'actual_mw': 123.0, 'model_mw': 130.0,
                  'selected_baseline_mw': 125.0}).to_csv(
        tmp_path / 'data/model_outputs/lightgbm_predictions_2026.csv', index=False
    )
    return TestClient(create_app(tmp_path))


@pytest.mark.parametrize('local_day,utc_left,utc_right,count', [
    ('2026-07-01', '2026-06-30T21:00:00+00:00', '2026-07-01T21:00:00+00:00', 96),
    ('2026-03-29', '2026-03-28T22:00:00+00:00', '2026-03-29T21:00:00+00:00', 92),
    ('2026-10-25', '2026-10-24T21:00:00+00:00', '2026-10-25T22:00:00+00:00', 100),
])
def test_finland_date_range_dst(tmp_path, local_day, utc_left, utc_right, count):
    c = _client(tmp_path, utc_left, utc_right)
    end = (pd.Timestamp(local_day) + pd.Timedelta(days=1)).date().isoformat()
    query = {'kind':'history', 'start':local_day, 'end':end,
             'interval':'15min', 'tz':'Europe/Helsinki'}
    r = c.get('/api/series', params=query)
    assert r.status_code == 200, r.text
    assert r.json()['count'] == count
    assert r.json()['points'][0]['timestamp_utc'] == pd.Timestamp(utc_left).isoformat()
    insights = c.get('/api/insights', params=query)
    assert insights.status_code == 200, insights.text
    assert insights.json()['total_intervals'] == count
    assert insights.json()['start_utc'] == pd.Timestamp(utc_left).isoformat()
    assert insights.json()['end_utc_exclusive'] == pd.Timestamp(utc_right).isoformat()


def test_local_daily_mean_uses_finnish_midnight(tmp_path):
    c = _client(tmp_path, '2026-06-30T20:00Z', '2026-07-02T22:00Z')
    r = c.get('/api/series', params={'kind':'history','start':'2026-07-01','end':'2026-07-03',
                                      'interval':'day','tz':'Europe/Helsinki'})
    assert r.status_code == 200
    assert r.json()['count'] == 2
    assert [p['timestamp_utc'] for p in r.json()['points']] == [
        '2026-06-30T21:00:00+00:00','2026-07-01T21:00:00+00:00']


def test_utc_api_remains_backward_compatible(tmp_path):
    c = _client(tmp_path, '2026-06-30T20:00Z', '2026-07-02T22:00Z')
    r = c.get('/api/series', params={'kind':'history','start':'2026-07-01',
                                      'end':'2026-07-02','interval':'day'})
    assert r.status_code == 200
    assert r.json()['points'][0]['timestamp_utc'] == '2026-07-01T00:00:00+00:00'
    assert c.get('/api/series', params={'kind':'history','start':'2026-07-01',
                                        'end':'2026-07-02','tz':'Asia/Tokyo'}).status_code == 422


def test_dashboard_requests_finland_tz_and_preserves_utc_timestamp(tmp_path):
    c = _client(tmp_path, '2026-06-30T20:00Z','2026-07-02T22:00Z')
    js = c.get('/static/app.js').text
    html = c.get('/').text
    assert 'tz:FINLAND_TZ' in js
    assert 'Europe/Helsinki' in js
    assert 'timestamp_finland' in js and 'timestamp_utc' in js
    assert 'Finland time · EET/EEST' in html


