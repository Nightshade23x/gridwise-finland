import pandas as pd
import pytest

from gridwise.weather import date_windows, download_temperatures, fetch_hourly


class FakeResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return {"hourly": {"time": ["2026-01-01T00:00", "2026-01-01T01:00"],
                           "temperature_2m": [-5.1, -4.9]}}


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResponse()


def test_year_windows_cover_inclusive_end_without_overlap():
    bounds = list(date_windows("2024-01-01", "2025-03-02"))
    assert bounds == [("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-03-02")]


def test_fetch_era5_hourly_utc_and_units():
    client = FakeSession()
    df = fetch_hourly("helsinki", "2026-01-01", "2026-01-01", session=client)
    assert df["temp_helsinki_c"].tolist() == [-5.1, -4.9]
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2026-01-01", tz="UTC")
    assert client.calls[0][1]["models"] == "era5"
    assert client.calls[0][1]["timezone"] == "UTC"


def test_bad_city_fails_before_api_request():
    with pytest.raises(ValueError, match="Unknown city"):
        fetch_hourly("wrong", "2025-01-01", "2025-01-01", session=FakeSession())
    with pytest.raises(ValueError, match="Duplicate cities"):
        download_temperatures(["helsinki", "helsinki"], "2025-01-01", "2025-01-02")
