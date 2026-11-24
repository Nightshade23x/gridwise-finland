"""Phase 16: 48-hour-lead historical forecast temperatures, never ERA5 truth.

Open-Meteo's Previous Runs API returns temperature_2m_previous_day2 for the
valid hour: a weather prediction with a nominal 48-hour lead. A full next-24h
issue window therefore only requires forecast information from >24h before
issue, leaving time for weather-model publication delays.

The API does not supply independently certified issuance timestamps. Preserve
this provenance limitation in every output and do not overwrite raw sources.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
import requests

from .refresh import ROOT, atomic_csv
from .weather import CITIES

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLE = "temperature_2m_previous_day2"
DEFAULT_OUTPUT = Path("data/raw/open_meteo_forecast_48h.csv")
COLUMNS = {city: f"temp_{city}_forecast_48h_c" for city in CITIES}


def _get_json(session, url: str, params: dict, *, sleep: Callable[[float], None]) -> dict:
    for attempt in range(4):
        response = session.get(url, params=params, timeout=90)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            try:
                retry = float(response.headers.get("Retry-After", 0))
            except (TypeError, ValueError, AttributeError):
                retry = 0
            sleep(max(retry, 3 * (2 ** attempt)))
            continue
        response.raise_for_status()
        doc = response.json()
        if not isinstance(doc, dict) or doc.get("error"):
            raise ValueError(f"Open-Meteo previous-runs error: {doc.get('reason', 'invalid response') if isinstance(doc, dict) else 'invalid response'}")
        return doc
    raise RuntimeError("Previous-runs retry limit exceeded")


def decode_hourly(doc: dict, city: str) -> pd.DataFrame:
    """Validate UTC hourly previous-day2 forecast without inventing missing data."""
    if city not in CITIES:
        raise ValueError(f"Unknown city {city}")
    if doc.get("utc_offset_seconds") != 0:
        raise ValueError("Previous-runs response must be UTC")
    units = doc.get("hourly_units") or {}
    if units.get(VARIABLE) != "°C":
        raise ValueError("Previous-runs temperature must be in Celsius")
    hourly = doc.get("hourly") or {}
    times, vals = hourly.get("time"), hourly.get(VARIABLE)
    if not isinstance(times, list) or not isinstance(vals, list) or not times or len(times) != len(vals):
        raise ValueError("Missing/mismatched previous-runs times and values")
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True, errors="raise"))
    if (index.hasnans or index.has_duplicates or not index.is_monotonic_increasing or
            (index.minute != 0).any() or (index.second != 0).any() or
            (index.microsecond != 0).any()):
        raise ValueError("Invalid hourly previous-runs UTC timestamp grid")
    values = pd.to_numeric(pd.Series(vals), errors="coerce")
    bad = values.notna() & (~np.isfinite(values) | (values < -90) | (values > 65))
    if bad.any():
        raise ValueError("Implausible previous-runs forecast temperatures")
    return pd.DataFrame({"timestamp_utc": index, COLUMNS[city]: values.to_numpy(dtype=float)})


def year_windows(first: date, last: date):
    """Cap requests at 90 days to avoid oversized historical forecast responses."""
    if last < first:
        raise ValueError("end must not precede start")
    current = first
    while current <= last:
        end = min(last, current + timedelta(days=89))
        yield current, end
        current = end + timedelta(days=1)


def retrieve(
    start: date,
    end: date,
    *,
    session=None,
    cities: tuple[str, ...] = tuple(CITIES),
    pause: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Fetch hourly forecast vintages, leaving genuine missing values as NaN."""
    if not cities or len(set(cities)) != len(cities) or any(c not in CITIES for c in cities):
        raise ValueError("Cities must be distinct known locations")
    client = session or requests.Session()
    expected = pd.date_range(start, end + timedelta(days=1), freq="h", inclusive="left", tz="UTC")
    output = pd.DataFrame({"timestamp_utc": expected})
    for city in cities:
        lat, lon = CITIES[city]
        chunks = []
        for a, b in year_windows(start, end):
            print(f"Downloading 48h-lead {city} forecasts: {a} to {b}", flush=True)
            doc = _get_json(client, API, {
                "latitude": lat, "longitude": lon, "hourly": VARIABLE,
                "start_date": a.isoformat(), "end_date": b.isoformat(),
                "timezone": "UTC", "temperature_unit": "celsius",
            }, sleep=pause)
            frame = decode_hourly(doc, city)
            if ((frame.timestamp_utc < pd.Timestamp(a, tz="UTC")) |
                    (frame.timestamp_utc >= pd.Timestamp(b + timedelta(days=1), tz="UTC"))).any():
                raise ValueError("Weather API returned dates outside requested window")
            chunks.append(frame)
            pause(0.6)
        joined = pd.concat(chunks, ignore_index=True)
        if joined["timestamp_utc"].duplicated().any():
            raise ValueError("Overlapping weather response timestamps")
        output = output.merge(joined, on="timestamp_utc", how="left", validate="one_to_one")
    return output


def main():
    parser = argparse.ArgumentParser(description="Download Open-Meteo 48h-lead forecast temperatures for GridWise")
    parser.add_argument("--start", default="2024-01-01", help="First calendar day, UTC")
    parser.add_argument("--end", help="Last calendar day, UTC; default is two days after today")
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_OUTPUT)
    parser.add_argument("--cities", nargs="+", choices=sorted(CITIES), default=list(CITIES))
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)).date()
    frame = retrieve(start, end, cities=tuple(args.cities))
    atomic_csv(args.output.resolve(), frame)
    print(f"Saved {len(frame):,} hourly timestamps to {args.output}")
    print("Unavailable forecast hours by city:", frame.drop(columns="timestamp_utc").isna().sum().to_dict())
    print("Source: Open-Meteo Previous Runs API, 48h lead; NOT observed temperatures or certified issuance snapshots.")

if __name__ == "__main__":
    main()
