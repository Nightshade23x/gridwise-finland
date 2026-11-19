import pandas as pd

from gridwise.backfill import merge_new
from gridwise.quality import audit_consumption


def sample_frame(timestamps):
    return pd.DataFrame({
        "dataset_id": 124,
        "start_time_utc": pd.to_datetime(timestamps, utc=True),
        "end_time_utc": pd.to_datetime(timestamps, utc=True) + pd.Timedelta(minutes=15),
        "demand_mw": [9500.0] * len(timestamps),
    })


def test_audit_finds_missing_15min_interval():
    frame = sample_frame(["2026-10-21T17:00:00Z", "2026-10-21T17:30:00Z", "2026-10-21T17:45:00Z"])
    report = audit_consumption(frame, pd.Timestamp("2026-10-21T17:00:00Z"), pd.Timestamp("2026-10-21T18:00:00Z"))
    assert report["expected_intervals"] == 4
    assert report["unique_timestamps_in_range"] == 3
    assert report["missing_timestamps"] == ["2026-10-21T17:15:00+00:00"]
    assert report["completeness_percent"] == 75.0


def test_merge_adds_only_missing_row_and_keeps_original():
    existing = sample_frame(["2026-10-21T17:00:00Z", "2026-10-21T17:30:00Z"])
    retry = sample_frame(["2026-10-21T17:00:00Z", "2026-10-21T17:15:00Z", "2026-10-21T17:30:00Z"])
    retry.loc[0, "demand_mw"] = 99999.0
    merged, added = merge_new(existing, retry)
    assert added == 1
    assert len(merged) == 3
    assert merged.iloc[0].demand_mw == 9500.0
    assert str(merged.iloc[1].start_time_utc) == "2026-10-21 17:15:00+00:00"


def test_merge_does_not_modify_existing_if_no_new_data():
    existing = sample_frame(["2026-10-21T17:00:00Z"])
    merged, added = merge_new(existing, existing.copy())
    assert added == 0
    assert merged.equals(existing)
