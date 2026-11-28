# Phase 22b — Finnish local time in GridWise

The dashboard displays timestamps in `Europe/Helsinki`, using EET (UTC+2) in winter and EEST (UTC+3) in summer. This is a **presentation change**, not a re-issuance of any forecast. Timestamps in stored artifacts, CLI output, all archived forecast IDs, and JSON keys such as `timestamp_utc` stay UTC for reproducibility.

## Local chart filters and DST

The dashboard asks the API for `tz=Europe/Helsinki` when fetching `/api/series` and `/api/insights`. Date inputs now refer to **Finnish calendar dates**; the `Until` date is exclusive. Daily means are grouped by local midnight, which means the spring DST transition has 23 hours and the autumn transition has 25 hours. Forecast and observation timestamps are displayed with their EET/EEST abbreviation so the repeated autumn 03:00 hour is unambiguous.

The endpoints remain backward compatible: callers that omit the `tz` query parameter use UTC calendar dates and UTC daily bins, as before. API responses continue to return ISO-8601 `timestamp_utc`. CSV exports from the browser contain both `timestamp_finland` (human-readable local time and zone abbreviation) and the original `timestamp_utc` (machine-readable instant).

## Other dashboard sections

All issuance, capture, observation, and refresh timestamps are presented in Finnish time. Relative ages remain calculated from the underlying UTC instants. Phase 20's hourly-error diagnostic now also calculates `by_finland_hour` from actual target dates, rather than blindly shifting a UTC hour by a fixed offset. This requires rebuilding the saved analysis report **once** after installing the patch:

```powershell
.\.venv\Scripts\python.exe -m gridwise.failure_analysis
```

The existing `by_utc_hour` report field remains for compatibility with older reports. If an old report is loaded, the dashboard honestly labels its hour groups UTC until the report is rebuilt. In a DST fall-back window, two distinct local 03:00 hours are grouped under the same clock hour in the hourly diagnostic; precise individual timestamps still include EET/EEST.

No Fingrid refresh, forecast re-issuance or model retraining is required. Rebuild the local Docker dashboard after applying the patch. The public static Render demo will pick up the UI changes only after pushing the commit and redeploying.
