import numpy as np
import pandas as pd
import pytest

from gridwise.availability_audit import audit_availability


def data(n=400):
    return pd.DataFrame({
        "timestamp_utc": pd.date_range("2026-06-30", periods=n, freq="15min", tz="UTC"),
        "demand_mw": np.arange(n, dtype=float),
    })


def test_expected_lags_under_each_buffer():
    _, paired = audit_availability(data(), "2026-07-01", "2026-07-03", (0, 15, 60))
    # Row at 2026-07-01 00:00 is position 96, so lag-24h is position 0.
    first = paired.iloc[0]
    # Common rows start at position 100 because max buffer = 60m.
    assert first["timestamp_utc"] == pd.Timestamp("2026-07-01 01:00", tz="UTC")
    assert first["pred_buffer_0m_mw"] == 4
    assert first["pred_buffer_15m_mw"] == 3
    assert first["pred_buffer_60m_mw"] == 0


def test_same_pairs_and_missing_observations_not_imputed():
    df = data()
    df.loc[2, "demand_mw"] = np.nan
    report, paired = audit_availability(df, "2026-07-01", "2026-07-03", (0, 15, 60))
    assert len(paired) == report["common_paired_intervals"]
    assert paired[["demand_mw", "pred_buffer_0m_mw", "pred_buffer_15m_mw", "pred_buffer_60m_mw"]].notna().all().all()
    assert len(paired) < 188


def test_predictions_do_not_use_future_targets():
    a = data()
    _, original = audit_availability(a, "2026-07-01", "2026-07-03", (0, 15))
    a.loc[120:, "demand_mw"] += 999
    _, changed = audit_availability(a, "2026-07-01", "2026-07-03", (0, 15))
    assert (original.loc[original.index < 120, ["pred_buffer_0m_mw", "pred_buffer_15m_mw"]].to_numpy() ==
            changed.loc[changed.index < 120, ["pred_buffer_0m_mw", "pred_buffer_15m_mw"]].to_numpy()).all()


def test_reject_invalid_buffers_and_grid():
    with pytest.raises(ValueError, match="multiples"):
        audit_availability(data(), buffers_minutes=(0, 10))
    df = data()
    df.loc[5, "timestamp_utc"] = pd.Timestamp("2026-06-30 01:16", tz="UTC")
    with pytest.raises(ValueError, match="15-minute UTC grid"):
        audit_availability(df)
