"""Fingrid Open Data client and basic measurement validation."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://data.fingrid.fi/api/datasets"
CONSUMPTION_ID = 124
RATE_LIMIT_SECONDS = 3.0  # Leave a safety margin above Fingrid's 2-second limit.


def parse_utc_day(day: str) -> datetime:
    """Parse an ISO calendar day as midnight UTC."""
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def windows(start: datetime, end: datetime, chunk_days: int = 7) -> Iterable[tuple[datetime, datetime]]:
    """Generate consecutive windows [start, end) for bounded requests."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Both timestamps must contain a timezone")
    if end <= start:
        raise ValueError("--end must be after --start")
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + timedelta(days=chunk_days), end)
        yield cursor, next_cursor
        cursor = next_cursor


def as_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class FingridClient:
    """Fetch a dataset with pagination; respect the documented API throttle."""

    def __init__(self, api_key: str, session: requests.Session | None = None, throttle: float = RATE_LIMIT_SECONDS):
        if not api_key or api_key == "replace_with_your_key":
            raise ValueError("Set a real FINGRID_API_KEY in your .env file")
        self.session = session or requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Accept": "application/json"})
        self.throttle = throttle
        self._last_request_at: float | None = None

    def _get(self, dataset_id: int, params: dict) -> dict:
        """Make a rate-limited request; retry temporary HTTP 429 responses."""
        max_attempts = 5
        for attempt in range(max_attempts):
            # Space consecutive requests, even if the preceding one failed.
            if self._last_request_at is not None:
                remaining = self.throttle - (time.monotonic() - self._last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_request_at = time.monotonic()

            response = self.session.get(
                f"{BASE_URL}/{dataset_id}/data", params=params, timeout=45
            )
            if response.status_code == 429 and attempt < max_attempts - 1:
                # Respect the server's Retry-After (when it is numeric).
                retry_after = response.headers.get("Retry-After", "")
                try:
                    server_delay = max(0, float(retry_after))
                except (TypeError, ValueError):
                    server_delay = 0.0
                delay = max(server_delay, 5.0 * (2 ** attempt))
                print(f"  Rate limited (429). Retrying in {delay:g} seconds...", flush=True)
                time.sleep(delay)
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"Fingrid API HTTP {response.status_code}. "
                    "Check your API key, date range, or usage limits."
                ) from exc

            payload = response.json()
            if not isinstance(payload, dict) or "data" not in payload or not isinstance(payload["data"], list):
                raise ValueError("Unexpected Fingrid response structure (missing data array)")
            return payload

        raise RuntimeError("Fingrid request did not succeed after retries")

    def fetch_window(self, dataset_id: int, start: datetime, end: datetime) -> list[dict]:
        """Collect every page for a bounded UTC date window."""
        page = 1
        rows: list[dict] = []
        while True:
            payload = self._get(dataset_id, {
                "startTime": as_iso(start),
                "endTime": as_iso(end),
                "format": "json",
                "pageSize": 1000,
                "page": page,
                "sortBy": "startTime",
                "sortOrder": "asc",
            })
            rows.extend(payload["data"])
            pagination = payload.get("pagination") or {}
            last_page = int(pagination.get("lastPage", pagination.get("totalPages", 1)))
            if len(payload["data"]) == 1000 and not pagination:
                raise ValueError("API response is full but lacks pagination metadata; refusing silent truncation")
            if page >= last_page:
                break
            page += 1
        return rows


def normalize_consumption(records: list[dict], start: datetime, end: datetime) -> pd.DataFrame:
    """Validate, deduplicate and sort interval data in UTC."""
    columns = ["dataset_id", "start_time_utc", "end_time_utc", "demand_mw"]
    if not records:
        return pd.DataFrame(columns=columns)
    data = pd.DataFrame.from_records(records)
    required = {"datasetId", "startTime", "endTime", "value"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing Fingrid fields: {sorted(required - set(data.columns))}")
    result = pd.DataFrame({
        "dataset_id": pd.to_numeric(data["datasetId"], errors="raise").astype(int),
        "start_time_utc": pd.to_datetime(data["startTime"], utc=True, errors="raise"),
        "end_time_utc": pd.to_datetime(data["endTime"], utc=True, errors="raise"),
        "demand_mw": pd.to_numeric(data["value"], errors="coerce"),
    })
    result = result.loc[(result.start_time_utc >= start) & (result.start_time_utc < end)].copy()
    if result.empty:
        return result.reset_index(drop=True)
    if not (result.dataset_id == CONSUMPTION_ID).all():
        raise ValueError("Response contained unexpected dataset ID")
    if result.demand_mw.isna().any():
        raise ValueError("Dataset contains null or invalid demand values")
    if (result.end_time_utc <= result.start_time_utc).any():
        raise ValueError("Dataset contains intervals with invalid start/end")
    result = result.sort_values("start_time_utc")
    result = result.drop_duplicates(subset=["dataset_id", "start_time_utc"], keep="last")
    return result[columns].reset_index(drop=True)
