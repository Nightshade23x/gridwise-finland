"""Download ERA5 hourly temperatures from Open-Meteo's historical archive.

ERA5 is reanalysis rather than an archived point-in-time weather forecast.
For operational forecasts, replace this input with weather forecasts as issued.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from .ingest import ROOT

API_URL = "https://archive-api.open-meteo.com/v1/archive"
# A geographically distributed, deliberately *unweighted* set of proxies.
CITIES = {
    "helsinki": (60.1699, 24.9384),
    "turku": (60.4518, 22.2666),
    "tampere": (61.4978, 23.7610),
    "oulu": (65.0121, 25.4651),
    "rovaniemi": (66.5039, 25.7294),
}


def date_windows(start_day: str, end_day: str):
    """Yield inclusive calendar-date API windows, at most one year each."""
    current = pd.Timestamp(start_day)
    last = pd.Timestamp(end_day)
    if current.tzinfo is not None or last.tzinfo is not None or last < current:
        raise ValueError("Use YYYY-MM-DD dates with start <= end")
    while current <= last:
        end = min(current + pd.DateOffset(years=1) - pd.Timedelta(days=1), last)
        yield current.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        current = end + pd.Timedelta(days=1)


def fetch_hourly(
    city: str,
    start_day: str,
    end_day: str,
    session: requests.Session | None = None,
    sleep=time.sleep,
) -> pd.DataFrame:
    """Request ERA5 historical hourly temperatures for a single city/window."""
    if city not in CITIES:
        raise ValueError(f"Unknown city: {city}")
    lat, lon = CITIES[city]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_day,
        "end_date": end_day,
        "hourly": "temperature_2m",
        "timezone": "UTC",
        "models": "era5",
    }
    client = session or requests.Session()
    for attempt in range(5):
        response = client.get(API_URL, params=params, timeout=90)
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 4:
            header = response.headers.get("Retry-After", "")
            try:
                retry_after = max(0.0, float(header))
            except (TypeError, ValueError):
                retry_after = 0.0
            delay = max(retry_after, float(3 * (2 ** attempt)))
            print(f"  Open-Meteo HTTP {response.status_code}; retrying after {delay:g}s", flush=True)
            sleep(delay)
            continue
        response.raise_for_status()
        break
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason', 'unknown')}")
    hourly = payload.get("hourly", {})
    timestamps = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    if not timestamps or len(timestamps) != len(temps):
        raise ValueError("Open-Meteo response missing aligned hourly timestamps and temperatures")
    frame = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps, utc=True),
        f"temp_{city}_c": pd.to_numeric(pd.Series(temps), errors="coerce"),
    })
    if frame["timestamp_utc"].duplicated().any():
        raise ValueError(f"Duplicate Open-Meteo timestamps for {city}")
    if not (frame["timestamp_utc"].dt.minute.eq(0) & frame["timestamp_utc"].dt.second.eq(0)).all():
        raise ValueError("Open-Meteo returned off-hour timestamps")
    return frame


def download_temperatures(
    cities: list[str],
    start_day: str,
    end_day: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    if len(set(cities)) != len(cities):
        raise ValueError("Duplicate cities")
    for city in cities:
        if city not in CITIES:
            raise ValueError(f"Unknown city: {city}")
    all_cities = []
    for city in cities:
        chunks = []
        for a, b in date_windows(start_day, end_day):
            print(f"Fetching ERA5 {city}: {a} through {b} (UTC)", flush=True)
            chunks.append(fetch_hourly(city, a, b, session=session))
        part = pd.concat(chunks, ignore_index=True)
        if part["timestamp_utc"].duplicated().any():
            raise ValueError(f"Repeated timestamps in {city} chunks")
        all_cities.append(part.set_index("timestamp_utc"))
    combined = pd.concat(all_cities, axis=1).sort_index()
    begin = pd.Timestamp(start_day, tz="UTC")
    exclusive_end = pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(days=1)
    expected = pd.date_range(begin, exclusive_end, freq="h", inclusive="left")
    unexpected = combined.index.difference(expected)
    if len(unexpected):
        raise ValueError(f"Weather timestamps outside requested dates: {list(unexpected[:5])}")
    combined = combined.reindex(expected)
    combined.index.name = "timestamp_utc"
    return combined.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download hourly ERA5 weather proxies for Finland")
    parser.add_argument("--start", default="2024-01-01", help="Inclusive UTC calendar day")
    parser.add_argument("--end", default="2026-10-31", help="Inclusive UTC calendar day")
    parser.add_argument("--cities", nargs="+", choices=sorted(CITIES), default=list(CITIES))
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/open_meteo_temperature_2024_2026.csv")
    args = parser.parse_args()
    temperatures = download_temperatures(args.cities, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temperatures.to_csv(args.output, index=False)
    print(f"Saved {len(temperatures):,} hourly timestamps to {args.output}")
    print("Missing readings by city:", temperatures.isna().sum().drop("timestamp_utc").to_dict())


if __name__ == "__main__":
    main()
