"""Tests for Weather Underground forecast client."""

from __future__ import annotations

import json
import os
import time

import pytest
from unittest.mock import AsyncMock, patch

from weather.wu import get_wu_forecast

# ---------------------------------------------------------------------------
# Sample TWC response
# ---------------------------------------------------------------------------

SAMPLE_WU_RESPONSE = {
    "validTimeLocal": [
        "2026-02-18T07:00:00-0500",
        "2026-02-19T07:00:00-0500",
        "2026-02-20T07:00:00-0500",
    ],
    "temperatureMax": [45, 50, 55],
    "temperatureMin": [30, 35, 40],
    "dayOfWeek": ["Wednesday", "Thursday", "Friday"],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wu_forecast_parse(tmp_path):
    """Valid TWC response is parsed into {date: {high, low}} dict."""
    with patch("weather.wu.fetch_json", AsyncMock(return_value=SAMPLE_WU_RESPONSE)):
        result = await get_wu_forecast(40.71, -74.01, "fake-key", cache_dir=tmp_path)

    assert result is not None
    assert len(result) == 3
    assert result["2026-02-18"] == {"high": 45, "low": 30}
    assert result["2026-02-19"] == {"high": 50, "low": 35}
    assert result["2026-02-20"] == {"high": 55, "low": 40}


@pytest.mark.asyncio
async def test_wu_cache_hit(tmp_path):
    """When a valid cache file exists, fetch_json is NOT called."""
    # Write a valid cache file
    from weather.wu import _cache_path

    cp = _cache_path(tmp_path, 40.71, -74.01)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cached_data = {"2026-02-18": {"high": 45, "low": 30}}
    with open(cp, "w") as f:
        json.dump(cached_data, f)

    mock_fetch = AsyncMock(return_value=SAMPLE_WU_RESPONSE)
    with patch("weather.wu.fetch_json", mock_fetch):
        result = await get_wu_forecast(40.71, -74.01, "fake-key", cache_dir=tmp_path)

    assert result == cached_data
    mock_fetch.assert_not_called()


@pytest.mark.asyncio
async def test_wu_cache_expired(tmp_path):
    """When cache file is older than TTL, fetch_json IS called."""
    from weather.wu import _cache_path

    cp = _cache_path(tmp_path, 40.71, -74.01)
    cp.parent.mkdir(parents=True, exist_ok=True)
    old_data = {"2026-02-17": {"high": 40, "low": 25}}
    with open(cp, "w") as f:
        json.dump(old_data, f)

    # Set mtime to 2 hours ago
    old_time = time.time() - 7200
    os.utime(cp, (old_time, old_time))

    mock_fetch = AsyncMock(return_value=SAMPLE_WU_RESPONSE)
    with patch("weather.wu.fetch_json", mock_fetch):
        result = await get_wu_forecast(
            40.71, -74.01, "fake-key", cache_dir=tmp_path, cache_ttl=3600,
        )

    assert result is not None
    assert len(result) == 3
    mock_fetch.assert_called_once()


@pytest.mark.asyncio
async def test_wu_api_failure(tmp_path):
    """When fetch_json returns None, get_wu_forecast returns None gracefully."""
    with patch("weather.wu.fetch_json", AsyncMock(return_value=None)):
        result = await get_wu_forecast(40.71, -74.01, "fake-key", cache_dir=tmp_path)

    assert result is None


@pytest.mark.asyncio
async def test_wu_no_api_key(tmp_path):
    """Empty api_key returns None immediately with no HTTP call."""
    mock_fetch = AsyncMock(return_value=SAMPLE_WU_RESPONSE)
    with patch("weather.wu.fetch_json", mock_fetch):
        result = await get_wu_forecast(40.71, -74.01, "", cache_dir=tmp_path)

    assert result is None
    mock_fetch.assert_not_called()
