"""Phase 16: no network, provenance gating and immutable candidate coverage."""
import copy
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.forward import FEATURES
from gridwise.weather_previous import VARIABLE, COLUMNS, decode_hourly, retrieve
from gridwise.weather_forward import _align_weather, experiment, save_report, REPORT, ARCHIVE


def mock_doc(stamps=None, values=None):
    stamps = pd.date_range("2026-11-24",periods=24,freq="h",tz="UTC") if stamps is None else stamps
    return {"utc_offset_seconds":0,"hourly_units":{VARIABLE:"°C"},
            "hourly":{"time":[t.strftime('%Y-%m-%dT%H:%M') for t in stamps],
                      VARIABLE:[4.0]*len(stamps) if values is None else values}}


class FakeResponse:
    status_code=200
    headers={}
    def __init__(self, doc): self.doc=doc
    def raise_for_status(self): pass
    def json(self): return self.doc


class FakeSession:
    def __init__(self, doc): self.doc=doc;self.calls=[]
    def get(self,url,params,timeout):
        self.calls.append((url,params,timeout))
        return FakeResponse(self.doc)


def test_previous_runs_explicit_48h_parameter():
    fake=FakeSession(mock_doc())
    result=retrieve(date(2026,11,24),date(2026,11,24),session=fake,cities=("helsinki",),pause=lambda s: None)
    assert len(result)==24
    assert result[COLUMNS["helsinki"]].eq(4).all()
    url,params,_=fake.calls[0]
    assert url.startswith("https://previous-runs-api.open-meteo.com")
    assert params["hourly"]=="temperature_2m_previous_day2"
    assert params["timezone"]=="UTC"
    assert params["start_date"]=="2026-11-24"


def test_previous_runs_missing_weather_remains_missing_not_imputed():
    doc=mock_doc(values=[3.0]*12+[None]+[4.0]*11)
    weather=decode_hourly(doc,"helsinki")
    assert np.isnan(weather[COLUMNS['helsinki']].iloc[12])


def test_reject_wrong_timezone_or_reanalysis_variable():
    doc=mock_doc();doc['utc_offset_seconds']=7200
    with pytest.raises(ValueError,match='UTC'): decode_hourly(doc,"helsinki")
    doc=mock_doc();doc['hourly_units']={"temperature_2m":"°C"}
    with pytest.raises(ValueError,match='Celsius'):decode_hourly(doc,"helsinki")


def test_reject_duplicate_weather_timestamps():
    doc=mock_doc()
    doc['hourly']['time'][2]=doc['hourly']['time'][1]
    with pytest.raises(ValueError,match='timestamp'): decode_hourly(doc,"helsinki")


def full_weather():
    idx=pd.date_range('2024-01-01','2026-11-24',freq='h',tz='UTC')
    values=5 + 12*np.cos(2*np.pi*idx.dayofyear.to_numpy()/365.25) + np.sin(np.arange(len(idx))*2*np.pi/24)
    return pd.DataFrame({"timestamp_utc":idx,**{col:values.copy() + i for i,col in enumerate(COLUMNS.values())}})


def test_weather_join_never_uses_next_hour_or_fills_missing():
    hours=pd.date_range('2026-11-24 12:00Z',periods=3,freq='h')
    weather=pd.DataFrame({'timestamp_utc':hours,**{col:[1.0,np.nan,5.0] for col in COLUMNS.values()}})
    slots=pd.date_range('2026-11-24 12:00Z',periods=12,freq='15min')
    joined=_align_weather(slots,weather)
    assert joined[COLUMNS['helsinki']].iloc[:4].eq(1).all()
    assert joined[COLUMNS['helsinki']].iloc[4:8].isna().all()
    assert joined[COLUMNS['helsinki']].iloc[8:].eq(5).all()


def test_weather_experiment_end_to_end_and_future_demand_safe():
    from test_forward import synthetic_data
    data=synthetic_data()
    weather=full_weather()
    issue=pd.Timestamp('2026-11-21 14:15Z')
    report,points,reg=experiment(data,weather,issue,estimators=8)
    assert len(points)==96
    assert report['train_rows']>50000
    assert report['validation_rows']>1000
    assert set(report['validation_mae_mw'])=={'demand_only','demand_weather','lag_48h','lag_7d','blend_50_50'}
    assert report['preferred_strategy']==min(report['validation_mae_mw'],key=report['validation_mae_mw'].get)
    chosen=({'demand_only':'demand_only_mw','demand_weather':'demand_weather_mw'}.
            get(report['preferred_strategy'],'baseline_mw'))
    assert np.allclose(points['preferred_mw'],points[chosen],atol=0.002)
    assert report['source_latest_measurement_utc']=='2026-11-21T13:00:00+00:00'
    assert points.iloc[0]['timestamp_utc']==pd.Timestamp('2026-11-21 14:30Z')
    assert points.iloc[-1]['timestamp_utc']==pd.Timestamp('2026-11-22 14:15Z')
    assert reg.booster_ is not None
    # Adding observations strictly after issuance must not change the issued curve.
    extra=pd.DataFrame({'timestamp_utc':pd.date_range('2026-11-21 14:30Z',periods=96,freq='15min'),
                        'demand_mw':[1e9]*96})
    changed=pd.concat([data,extra],ignore_index=True)
    report2,points2,_=experiment(changed,weather,issue,estimators=8)
    pd.testing.assert_frame_equal(points,points2)
    assert report2['validation_mae_mw']==report['validation_mae_mw']


def test_future_weather_gap_fails_without_writing(tmp_path):
    from test_forward import synthetic_data
    weather=full_weather()
    t=pd.Timestamp('2026-11-21 20:00Z')
    weather.loc[weather.timestamp_utc==t,COLUMNS['helsinki']]=np.nan
    with pytest.raises(ValueError,match='48h-lead forecast weather'):
        experiment(synthetic_data(),weather,pd.Timestamp('2026-11-21 14:15Z'),estimators=5)
    assert not (tmp_path/REPORT).exists()


def test_weather_snapshots_immutable_and_api_validates(tmp_path):
    from test_forward import synthetic_data
    report,points,est=experiment(synthetic_data(),full_weather(),pd.Timestamp('2026-11-21 14:15Z'),estimators=5)
    save_report(tmp_path,report,points,est)
    assert len(list((tmp_path/ARCHIVE).glob('*.json')))==1
    r=TestClient(create_app(tmp_path)).get('/api/weather-model-comparison')
    assert r.status_code==200
    assert len(r.json()['points'])==96
    save_report(tmp_path,report,points,est)
    assert len(list((tmp_path/ARCHIVE).glob('*.json')))==1
    tampered=copy.deepcopy(report)
    tampered['points'][0]['preferred_mw']+=101
    path=tmp_path/REPORT
    path.write_text(json.dumps(tampered))
    assert TestClient(create_app(tmp_path)).get('/api/weather-model-comparison').status_code==503
    # Original archive remains unchanged despite poisoned latest file.
    assert next((tmp_path/ARCHIVE).glob('*.json')).read_text() != path.read_text()


def test_absent_weather_report_explicit(tmp_path):
    data=TestClient(create_app(tmp_path)).get('/api/weather-model-comparison').json()
    assert data['status']=='not_generated'
    assert data['points']==[]
