"""Phase 15: archive *captured* Open-Meteo weather forecasts for later ML work.

These are NOT reanalysis observations, nor independently verified model issuance
snapshots. We record fetch completion time and only retain future hourly values.
No weather values are incorporated into the deployed demand model yet.
"""
from __future__ import annotations

from datetime import timedelta

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from .weather import CITIES
from .refresh import ROOT, atomic_json

API = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = Path("data/live/weather_forecast_archive")
LATEST = Path("data/live/weather_forecast_latest.json")


def parse_hourly(payload: dict, city: str) -> pd.DataFrame:
    """Reject unaligned, missing, duplicate and non-UTC forecast readings."""
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError(f"Weather API failed for {city}: {payload.get('reason', 'unknown') if isinstance(payload, dict) else 'invalid payload'}")
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Open-Meteo UTC offset must be zero")
    units = payload.get("hourly_units", {})
    if units.get("temperature_2m") != "°C":
        raise ValueError("Expected forecast temperatures in Celsius")
    hourly = payload.get("hourly", {})
    stamps, values = hourly.get("time"), hourly.get("temperature_2m")
    if not isinstance(stamps, list) or not isinstance(values, list) or not stamps or len(stamps) != len(values):
        raise ValueError("Missing or mismatched hourly weather data")
    index = pd.to_datetime(stamps, utc=True, errors="raise")
    if (index.hasnans or index.has_duplicates or not index.is_monotonic_increasing or
            (index.minute != 0).any() or (index.second != 0).any()):
        raise ValueError("Weather forecast time grid is invalid")
    temps = np.asarray(values, dtype=float)
    if not np.isfinite(temps).all() or ((temps < -90) | (temps > 65)).any():
        raise ValueError("Missing or implausible weather forecast temperatures")
    return pd.DataFrame({"timestamp_utc": index, "temperature_c": temps})


def capture_weather(root: Path, *, session: requests.Session | None = None,
                    now: pd.Timestamp | None = None, cities: tuple[str, ...] = tuple(CITIES),
                    pause=time.sleep) -> dict:
    """Capture five-city 24h weather predictions; no partial writes on failure."""
    if not cities or len(set(cities)) != len(cities) or any(city not in CITIES for city in cities):
        raise ValueError("Cities must be distinct known Finnish locations")
    client = session or requests.Session()
    by_city = {}
    for city in cities:
        lat, lon = CITIES[city]
        params = {"latitude":lat, "longitude":lon, "hourly":"temperature_2m",
                  "temperature_unit":"celsius", "timezone":"UTC", "forecast_days":3}
        response = client.get(API, params=params, timeout=45)
        response.raise_for_status()
        by_city[city] = parse_hourly(response.json(), city)
        # Modest throttle; no retrying endlessly or calling the API from the UI.
        if city != cities[-1]:
            pause(0.5)
    captured = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if captured.tzinfo is None:
        raise ValueError("Weather snapshot timestamp must be timezone-aware")
    captured = captured.tz_convert("UTC")
    # The first hour that has not begun at completion time. We do not pretend
    # that values for earlier hours were known at a past date.
    first = captured.ceil("h")
    if first <= captured:
        first += timedelta(hours=1)
    targets = pd.date_range(first, periods=24, freq="h", tz="UTC")
    result = []
    for city, rows in by_city.items():
        values = rows.set_index("timestamp_utc")["temperature_c"].reindex(targets)
        if values.isna().any():
            raise ValueError(f"Missing future hourly forecast values for {city}")
        result.append({"city":city,"latitude":CITIES[city][0],"longitude":CITIES[city][1],
                       "hourly": [{"timestamp_utc":t.isoformat(),"temperature_c":float(v)}
                                  for t,v in values.items()]})
    doc = {"schema_version":1,"kind":"captured_weather_forecast",
           "source":"Open-Meteo Forecast API; best-match model, not a fixed run",
           "captured_at_utc":captured.isoformat(),
           "source_model_run_time_verified":False,"timezone":"UTC","forecast_hours":24,
           "note":"Captured API response, not a certified weather model issuance. Do not train on future observed/reanalysis temperatures.",
           "cities":result}
    path=root/ARCHIVE/(captured.strftime("%Y%m%dT%H%M%S%fZ")+".json")
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != doc:
            raise ValueError("Conflicting weather capture at this timestamp")
    else:
        atomic_json(path,doc)
    atomic_json(root/LATEST,doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description="Capture point-in-time Open-Meteo temperature forecasts")
    parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args()
    doc=capture_weather(args.root.resolve())
    print(f"Captured hourly weather forecasts for {len(doc['cities'])} cities at {doc['captured_at_utc']}")
    print(f"Stored 24 future hourly values per city under {args.root / ARCHIVE}")
    print("These weather forecasts are archived but NOT used in the demand model yet.")

if __name__=="__main__":
    main()
