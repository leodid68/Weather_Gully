"""Tests for NOAA Weather API client (async + disk cache)."""

from __future__ import annotations

import json
import os
import time

import pytest
from unittest.mock import AsyncMock, patch

from weather.noaa import (
    _cache_path,
    _read_cache,
    _write_cache,
    get_noaa_forecast,
)

# ---------------------------------------------------------------------------
# Sample location config
# ---------------------------------------------------------------------------

LOCATIONS = {
    "NYC": {"lat": 40.71, "lon": -74.01},
    "Chicago": {"lat": 41.88, "lon": -87.63},
}

# ---------------------------------------------------------------------------
# Sample NOAA API responses
# ---------------------------------------------------------------------------

SAMPLE_POINTS = {
    "properties": {
        "forecast": "https://api.weather.gov/gridpoints/OKX/33,37/forecast",
        "forecastGridData": "https://api.weather.gov/gridpoints/OKX/33,37",
    }
}

SAMPLE_FORECAST = {
    "properties": {
        "periods": [
            {
                "startTime": "2026-02-18T06:00:00-05:00",
                "temperature": 45,
                "isDaytime": True,
            },
            {
                "startTime": "2026-02-18T18:00:00-05:00",
                "temperature": 32,
                "isDaytime": False,
            },
            {
                "startTime": "2026-02-19T06:00:00-05:00",
                "temperature": 50,
                "isDaytime": True,
            },
            {
                "startTime": "2026-02-19T18:00:00-05:00",
                "temperature": 35,
                "isDaytime": False,
            },
        ]
    }
}


# ---------------------------------------------------------------------------
# Pure-function tests (sync)
# ---------------------------------------------------------------------------


def test_cache_path():
    """_cache_path formats lat/lon to 2 decimals."""
    from pathlib import Path

    p = _cache_path(Path("/tmp/test_cache"), 40.71, -74.01)
    assert p == Path("/tmp/test_cache/40.71_-74.01.json")


def test_read_cache_missing(tmp_path):
    """_read_cache returns None for non-existent file."""
    p = tmp_path / "nope.json"
    assert _read_cache(p, 900) is None


def test_read_cache_valid(tmp_path):
    """_read_cache returns data when file is fresh."""
    p = tmp_path / "data.json"
    data = {"2026-02-18": {"high": 45, "low": 32}}
    with open(p, "w") as f:
        json.dump(data, f)
    assert _read_cache(p, 900) == data


def test_read_cache_expired(tmp_path):
    """_read_cache returns None when file is older than TTL."""
    p = tmp_path / "old.json"
    with open(p, "w") as f:
        json.dump({"x": 1}, f)
    old_time = time.time() - 2000
    os.utime(p, (old_time, old_time))
    assert _read_cache(p, 900) is None


def test_read_cache_corrupt(tmp_path):
    """_read_cache returns None for invalid JSON."""
    p = tmp_path / "bad.json"
    with open(p, "w") as f:
        f.write("not json{{{")
    assert _read_cache(p, 900) is None


def test_write_cache(tmp_path):
    """_write_cache writes JSON atomically."""
    p = tmp_path / "sub" / "data.json"
    data = {"2026-02-18": {"high": 45, "low": 32}}
    _write_cache(p, data)
    assert p.exists()
    with open(p) as f:
        assert json.load(f) == data


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noaa_forecast_success(tmp_path):
    """Happy path: points + forecast calls produce {date: {high, low}}."""
    mock_fetch = AsyncMock(side_effect=[SAMPLE_POINTS, SAMPLE_FORECAST])
    with patch("weather.noaa.fetch_json", mock_fetch):
        result = await get_noaa_forecast("NYC", LOCATIONS, cache_dir=tmp_path)

    assert len(result) == 2
    assert result["2026-02-18"] == {"high": 45, "low": 32}
    assert result["2026-02-19"] == {"high": 50, "low": 35}
    assert mock_fetch.call_count == 2


@pytest.mark.asyncio
async def test_noaa_unknown_location(tmp_path):
    """Unknown location returns empty dict without HTTP calls."""
    mock_fetch = AsyncMock()
    with patch("weather.noaa.fetch_json", mock_fetch):
        result = await get_noaa_forecast("Mars", LOCATIONS, cache_dir=tmp_path)

    assert result == {}
    mock_fetch.assert_not_called()


@pytest.mark.asyncio
async def test_noaa_points_failure(tmp_path):
    """When the points endpoint fails, return empty dict."""
    mock_fetch = AsyncMock(return_value=None)
    with patch("weather.noaa.fetch_json", mock_fetch):
        result = await get_noaa_forecast("NYC", LOCATIONS, cache_dir=tmp_path)

    assert result == {}


@pytest.mark.asyncio
async def test_noaa_forecast_failure(tmp_path):
    """When the forecast endpoint fails, return empty dict."""
    mock_fetch = AsyncMock(side_effect=[SAMPLE_POINTS, None])
    with patch("weather.noaa.fetch_json", mock_fetch):
        result = await get_noaa_forecast("NYC", LOCATIONS, cache_dir=tmp_path)

    assert result == {}


@pytest.mark.asyncio
async def test_noaa_no_forecast_url(tmp_path):
    """When points response has no forecast URL, return empty dict."""
    bad_points = {"properties": {"forecastGridData": "https://..."}}
    mock_fetch = AsyncMock(return_value=bad_points)
    with patch("weather.noaa.fetch_json", mock_fetch):
        result = await get_noaa_forecast("NYC", LOCATIONS, cache_dir=tmp_path)

    assert result == {}


@pytest.mark.asyncio
async def test_noaa_cache_hit(tmp_path):
    """When a valid cache file exists, fetch_json is NOT called."""
    cp = _cache_path(tmp_path, 40.71, -74.01)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cached_data = {"2026-02-18": {"high": 45, "low": 32}}
    with open(cp, "w") as f:
        json.dump(cached_data, f)

    mock_fetch = AsyncMock()
    with patch("weather.noaa.fetch_json", mock_fetch):
        result = await get_noaa_forecast("NYC", LOCATIONS, cache_dir=tmp_path)

    assert result == cached_data
    mock_fetch.assert_not_called()


@pytest.mark.asyncio
async def test_noaa_cache_expired(tmp_path):
    """When cache file is older than TTL, fetch_json IS called."""
    cp = _cache_path(tmp_path, 40.71, -74.01)
    cp.parent.mkdir(parents=True, exist_ok=True)
    old_data = {"2026-02-17": {"high": 40, "low": 25}}
    with open(cp, "w") as f:
        json.dump(old_data, f)

    # Set mtime to 20 minutes ago (> 900s TTL)
    old_time = time.time() - 1200
    os.utime(cp, (old_time, old_time))

    mock_fetch = AsyncMock(side_effect=[SAMPLE_POINTS, SAMPLE_FORECAST])
    with patch("weather.noaa.fetch_json", mock_fetch):
        result = await get_noaa_forecast(
            "NYC", LOCATIONS, cache_dir=tmp_path, cache_ttl=900,
        )

    assert result is not None
    assert len(result) == 2
    mock_fetch.assert_called()


@pytest.mark.asyncio
async def test_noaa_cache_written_on_success(tmp_path):
    """After a successful fetch, the result is written to cache."""
    mock_fetch = AsyncMock(side_effect=[SAMPLE_POINTS, SAMPLE_FORECAST])
    with patch("weather.noaa.fetch_json", mock_fetch):
        result = await get_noaa_forecast("NYC", LOCATIONS, cache_dir=tmp_path)

    assert len(result) == 2

    # Verify cache file was written
    cp = _cache_path(tmp_path, 40.71, -74.01)
    assert cp.exists()
    with open(cp) as f:
        cached = json.load(f)
    assert cached == result


@pytest.mark.asyncio
async def test_noaa_empty_periods(tmp_path):
    """Empty periods list returns empty dict (no cache write)."""
    empty_forecast = {"properties": {"periods": []}}
    mock_fetch = AsyncMock(side_effect=[SAMPLE_POINTS, empty_forecast])
    with patch("weather.noaa.fetch_json", mock_fetch):
        result = await get_noaa_forecast("NYC", LOCATIONS, cache_dir=tmp_path)

    assert result == {}
    # No cache file should be written for empty results
    cp = _cache_path(tmp_path, 40.71, -74.01)
    assert not cp.exists()
