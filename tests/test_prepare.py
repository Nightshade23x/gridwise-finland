import pandas as pd
import pytest

from gridwise.prepare import build_grid


def sample_row(timestamp, value):
    ts = pd.Timestamp(timestamp)
    return {
        "start_time_utc": ts.isoformat(),
        "end_time_utc": (ts + pd.Timedelta(minutes=15)).isoformat(),
        "demand_mw": value,
    }


def grid(rows, start="2025-03-30 10:30:00+00:00", end="2025-03-30 11:30:00+00:00"):
    return build_grid(pd.DataFrame(rows), pd.Timestamp(start), pd.Timestamp(end))


def test_preserves_missing_targets_without_imputation():
    clean, report = grid([sample_row("2025-03-30 10:30:00+00:00", 100)])
    assert len(clean) == 4
    assert report["missing_intervals"] == 3
    assert pd.isna(clean.loc[1, "demand_mw"])
    assert bool(clean.loc[1, "is_observed"]) is False


def test_repairs_small_offset_when_exact_slot_absent():
    clean, report = grid([sample_row("2025-03-30 10:45:02+00:00", 150)])
    assert report["corrected_timestamp_count"] == 1
    record = clean.loc[clean["timestamp_utc"] == pd.Timestamp("2025-03-30 10:45:00+00:00")].iloc[0]
    assert record["demand_mw"] == 150
    assert bool(record["is_time_corrected"]) is True


def test_does_not_replace_existing_exact_with_offgrid_measurement():
    rows = [
        sample_row("2025-03-30 10:45:00+00:00", 101),
        sample_row("2025-03-30 10:45:02+00:00", 999),
    ]
    clean, report = grid(rows)
    assert report["corrected_timestamp_count"] == 0
    assert report["excluded_offgrid_or_invalid_count"] == 1
    assert clean.loc[clean["timestamp_utc"] == pd.Timestamp("2025-03-30 10:45:00+00:00"), "demand_mw"].iloc[0] == 101


def test_refuses_conflicting_duplicate_source_timestamp():
    rows = [sample_row("2025-03-30 10:45:00+00:00", 101), sample_row("2025-03-30 10:45:00+00:00", 999)]
    with pytest.raises(ValueError, match="Conflicting"):
        grid(rows)


def test_local_time_features_account_for_dst_fall_back():
    # 2025-10-26 00:30 UTC and 01:30 UTC are both 03:30 local in Finland.
    clean, _ = grid([
        sample_row("2025-10-26 00:30:00+00:00", 100),
        sample_row("2025-10-26 01:30:00+00:00", 200),
    ], "2025-10-26 00:30:00+00:00", "2025-10-26 01:45:00+00:00")
    first = clean.iloc[0]
    second = clean.iloc[4]
    assert first["local_hour"] == second["local_hour"] == 3
    assert first["local_utc_offset_hours"] == 3
    assert second["local_utc_offset_hours"] == 2
