"""Phase 22: immutable issued shadow candidates + later paired verification."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.level_shift_shadow import (
    ARCHIVE, REPORT, archive_snapshot, archived_snapshots, build_snapshot,
    run_verification, score, selected_parameters, validate_snapshot,
    verify_snapshot,
)
from gridwise.level_shift import REPORT as CALIBRATION_REPORT
from gridwise.refresh import LIVE_CSV, atomic_csv


def _source():
    t = pd.date_range('2026-01-01', '2026-02-01', freq='15min', inclusive='left', tz='UTC')
    idx = np.arange(len(t))
    return pd.DataFrame({'timestamp_utc': t,
                         'demand_mw': 10500 + 850*np.sin(2*np.pi*idx/96) + idx*.02})


def _params():
    return {'baseline':'blend_50_50', 'alpha':1.0, 'decay_hours':24.0,
            'offset_cap_mw':1500.0, 'anchor_hours':6,
            'measurement_buffer_minutes':60}


def _issue(issue='2026-01-26T12:00:00Z'):
    return build_snapshot(_source(), pd.Timestamp(issue), _params(), 'a'*64)


def _actual(doc):
    times = pd.to_datetime([p['timestamp_utc'] for p in doc['points']], utc=True)
    return pd.DataFrame({'timestamp_utc':times,
                         'demand_mw': [p['uncorrected_mw'] + 100 for p in doc['points']]})


def test_issue_contains_two_versions_and_all_targets_strictly_future():
    doc=_issue()
    f=validate_snapshot(doc)
    assert len(f)==96
    assert f.timestamp_utc.iloc[0]==pd.Timestamp('2026-01-26T12:15:00Z')
    assert f.timestamp_utc.iloc[-1]==pd.Timestamp('2026-01-27T12:00:00Z')
    assert doc['source_latest_observation_utc'] <= '2026-01-26T11:00:00+00:00'
    assert (f.uncorrected_mw!=f.corrected_mw).any()


def test_future_observations_cannot_change_shadow_predictions():
    issue=pd.Timestamp('2026-01-26T12:05:00Z')
    original=_source()
    params=_params()
    first=build_snapshot(original, issue, params, 'a'*64)
    changed=original.copy()
    changed.loc[changed.timestamp_utc > issue - pd.Timedelta(minutes=60), 'demand_mw']=99999
    second=build_snapshot(changed, issue, params, 'a'*64)
    assert first==second


def test_issue_fails_closed_for_stale_or_missing_recent_values():
    data=_source()
    issue=pd.Timestamp('2026-01-26T12:00:00Z')
    data.loc[data.timestamp_utc>issue-pd.Timedelta(hours=9),'demand_mw']=np.nan
    with pytest.raises(ValueError,match='too old'):
        build_snapshot(data,issue,_params(),'a'*64)


def test_snapshot_rejects_non_future_non_grid_or_nan_values():
    doc=_issue()
    for mutation in ('past','off_grid','nan','bad_provenance'):
        changed=copy.deepcopy(doc)
        if mutation=='past':changed['points'][0]['timestamp_utc']=doc['issued_at_utc']
        elif mutation=='off_grid':changed['points'][1]['timestamp_utc']='2026-01-26T12:31:00Z'
        elif mutation=='nan':changed['points'][1]['corrected_mw']=float('nan')
        else:changed['calibration_report_sha256']='x'
        with pytest.raises(ValueError):validate_snapshot(changed)


def test_content_addressed_archive_idempotent_and_tamper_detection(tmp_path):
    doc=_issue();a=archive_snapshot(tmp_path,doc);b=archive_snapshot(tmp_path,doc)
    assert a==b and len(list((tmp_path/ARCHIVE).glob('*.json')))==1
    assert len(archived_snapshots(tmp_path))==1
    changed=copy.deepcopy(doc);changed['points'][0]['uncorrected_mw']+=1
    a.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='content changed'):
        archived_snapshots(tmp_path)


def test_pending_and_partial_are_paired_and_do_not_fill_missing():
    doc=_issue(); actual=_actual(doc)
    no=verify_snapshot('b'*64,doc,actual,as_of=pd.Timestamp('2026-01-26T12:45Z'))
    assert no['status']=='pending' and no['matched_targets']==0
    assert no['corrected']['mae_mw'] is None
    partial=actual.iloc[0:10].copy();partial=partial.drop(index=partial.index[2])
    r=verify_snapshot('b'*64,doc,partial,as_of=pd.Timestamp('2026-01-26T16:00Z'))
    assert r['status']=='partial' and r['eligible_targets']==11 and r['matched_targets']==9
    assert r['uncorrected']['paired_rows']==r['corrected']['paired_rows']==9


def test_complete_and_source_gap_states():
    doc=_issue(); actual=_actual(doc)
    now=pd.Timestamp('2026-01-30T00:00Z')
    complete=verify_snapshot('b'*64,doc,actual,as_of=now)
    assert complete['status']=='complete' and complete['matched_targets']==96
    assert complete['uncorrected']['mae_mw']==100
    assert all(x['matched_targets']==24 for x in complete['by_horizon'])
    incomplete=verify_snapshot('b'*64,doc,actual.drop(index=actual.index[4]),as_of=now)
    assert incomplete['status']=='incomplete_source_data' and incomplete['matched_targets']==95


def test_score_reports_signed_bias_and_guard_on_zero_reference_error():
    assert score(np.array([100.,200.]), np.array([120.,180.])) == {
        'paired_rows':2,'mae_mw':20.,'rmse_mw':20.,'bias_mw':0.}
    doc=_issue(); actual=_actual(doc)
    actual['demand_mw']=[p['uncorrected_mw'] for p in doc['points']]
    scored=verify_snapshot('b'*64,doc,actual,as_of=pd.Timestamp('2026-01-28T14:00Z'))
    assert scored['mae_reduction_percent'] is None


def test_calibration_report_freeze_and_validation(tmp_path):
    path=tmp_path/CALIBRATION_REPORT;path.parent.mkdir(parents=True)
    doc={'schema_version':1,'status':'evaluated','selected_parameters':_params(),
         'selection_period':{'start_utc':'2026-01-01T00:00:00+00:00',
                             'end_exclusive_utc':'2026-03-31T00:00:00+00:00'}}
    path.write_text(json.dumps(doc))
    params,digest=selected_parameters(tmp_path)
    assert params==_params() and digest==hashlib.sha256(path.read_bytes()).hexdigest()
    doc['selected_parameters']['anchor_hours']=0
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='differ'):
        selected_parameters(tmp_path)


def test_verification_read_only_api_pending_and_measured(tmp_path,monkeypatch):
    client=TestClient(create_app(tmp_path))
    assert client.get('/api/level-shift-shadow').json()['status']=='not_run'
    doc=_issue()
    archive_snapshot(tmp_path,doc)
    measured=_actual(doc)
    path=tmp_path/LIVE_CSV;path.parent.mkdir(parents=True,exist_ok=True)
    atomic_csv(path,measured)
    result=run_verification(tmp_path,as_of=pd.Timestamp('2026-01-30T00:00Z'))
    assert result['status']=='ok' and result['issuance_count']==1
    assert result['issuances'][0]['status']=='complete'
    assert (tmp_path/REPORT).exists()
    assert client.get('/api/level-shift-shadow').json()['issuances'][0]['matched_targets']==96
    # The verifier must not alter archived prediction bytes.
    snapshot_before=(tmp_path/ARCHIVE).glob('*.json')
    archived=list(snapshot_before);raw=archived[0].read_bytes()
    run_verification(tmp_path,as_of=pd.Timestamp('2026-01-30T00:00Z'))
    assert archived[0].read_bytes()==raw
    monkeypatch.setenv('GRIDWISE_DEMO_MODE','1')
    assert client.get('/api/level-shift-shadow').json()['status']=='not_available_in_public_demo'


def test_api_rejects_corrupt_saved_report(tmp_path):
    path=tmp_path/REPORT;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('{"schema_version":1,"status":"ok","issuances":[{}],"issuance_count":1}')
    assert TestClient(create_app(tmp_path)).get('/api/level-shift-shadow').status_code==503


def test_pending_api_requires_no_ml_or_shadow_worker_import(tmp_path, monkeypatch):
    """Regression: the Docker API image uses [app], not [ml] dependencies."""
    import builtins
    from unittest.mock import patch

    doc = _issue()
    archive_snapshot(tmp_path, doc)
    path = tmp_path / LIVE_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(path, pd.DataFrame({"timestamp_utc": [], "demand_mw": []}))
    result = run_verification(tmp_path, as_of=pd.Timestamp('2026-01-26T12:45Z'))
    assert result["issuances"][0]["status"] == "pending"
    client = TestClient(create_app(tmp_path))

    real_import = builtins.__import__
    def deny_model_imports(name, *args, **kwargs):
        if name.endswith("level_shift_shadow") or name.startswith("lightgbm"):
            raise ModuleNotFoundError("Model packages are not installed in the API-only image")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=deny_model_imports):
        response = client.get('/api/level-shift-shadow')
    assert response.status_code == 200
    saved = response.json()
    assert saved['issuance_count'] == 1
    assert saved['issuances'][0]['status'] == 'pending'
    assert saved['issuances'][0]['matched_targets'] == 0


def test_schema_validator_still_rejects_nonfinite_scores():
    from gridwise.shadow_report_schema import validate_report
    item={"status":"complete", "matched_targets":96, "eligible_targets":96,
          "uncorrected":{"paired_rows":96,"mae_mw":10,"rmse_mw":12,"bias_mw":1},
          "corrected":{"paired_rows":96,"mae_mw":float('nan'),"rmse_mw":12,"bias_mw":1}}
    report={"schema_version":1,"status":"ok","issuance_count":1,"issuances":[item]}
    with pytest.raises(ValueError, match='Invalid verification score'):
        validate_report(report)
