import pytest

from gridwise.fingrid import FingridClient, normalize_consumption, parse_utc_day, windows


def test_windows_are_contiguous():
    segments = list(windows(parse_utc_day("2026-10-01"), parse_utc_day("2026-10-11"), 7))
    assert len(segments) == 2
    assert segments[0][1] == segments[1][0]
    assert segments[-1][1] == parse_utc_day("2026-10-11")


def test_invalid_window():
    with pytest.raises(ValueError):
        list(windows(parse_utc_day("2026-10-10"), parse_utc_day("2026-10-01")))


def test_normalizes_and_deduplicates():
    record = {"datasetId": 124, "startTime": "2026-10-01T00:00:00Z", "endTime": "2026-10-01T00:15:00Z", "value": 10000.5}
    result = normalize_consumption([record, dict(record)], parse_utc_day("2026-10-01"), parse_utc_day("2026-10-02"))
    assert len(result) == 1
    assert result.iloc[0].demand_mw == 10000.5
    assert str(result.start_time_utc.dtype).startswith("datetime64[ns, UTC]")


def test_rejects_wrong_dataset():
    row = {"datasetId": 193, "startTime": "2026-10-01T00:00:00Z", "endTime": "2026-10-01T00:03:00Z", "value": 9999}
    with pytest.raises(ValueError, match="unexpected dataset"):
        normalize_consumption([row], parse_utc_day("2026-10-01"), parse_utc_day("2026-10-02"))


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params.copy()))
        page = params["page"]
        return FakeResponse({"data": [{"datasetId": 124, "startTime": "2026-10-01T00:00:00Z", "endTime": "2026-10-01T00:15:00Z", "value": page}], "pagination": {"lastPage": 2}})


def test_pagination():
    fake = FakeSession()
    client = FingridClient("test-key", session=fake, throttle=0)
    rows = client.fetch_window(124, parse_utc_day("2026-10-01"), parse_utc_day("2026-10-02"))
    assert len(rows) == 2
    assert [c[1]["page"] for c in fake.calls] == [1, 2]
    assert fake.headers["x-api-key"] == "test-key"


def test_retries_429(monkeypatch):
    import requests

    class RateLimitedResponse:
        status_code = 429
        headers = {"Retry-After": "2"}

        def raise_for_status(self):
            raise requests.HTTPError("429 Too Many Requests")

    class RetrySession:
        def __init__(self):
            self.headers = {}
            self.calls = 0

        def get(self, url, params, timeout):
            self.calls += 1
            if self.calls == 1:
                return RateLimitedResponse()
            return FakeResponse({"data": [], "pagination": {"lastPage": 1}})

    sleeps = []
    monkeypatch.setattr("gridwise.fingrid.time.sleep", sleeps.append)
    session = RetrySession()
    client = FingridClient("test-key", session=session, throttle=0)
    assert client.fetch_window(124, parse_utc_day("2026-10-01"), parse_utc_day("2026-10-02")) == []
    assert session.calls == 2
    assert sleeps == [5.0]
