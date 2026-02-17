"""Tests for the shared async HTTP client."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import weather.http_client as http_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status: int = 200, json_data: dict | list | None = None):
    """Build a mock aiohttp response usable as an async context manager."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data)

    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    return mock_cm, mock_resp


def _mock_session(cm):
    """Build a mock session whose .get() returns *cm*.

    Uses MagicMock because aiohttp's session.get() returns a context
    manager synchronously (not a coroutine).
    """
    session = MagicMock()
    session.get.return_value = cm
    session.post.return_value = cm
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_json_success():
    """200 response with JSON body returns parsed dict."""
    payload = {"temperature": 72.5}
    cm, _ = _mock_response(200, payload)
    session = _mock_session(cm)

    with patch.object(http_client, "get_session", AsyncMock(return_value=session)):
        result = await http_client.fetch_json("https://api.example.com/data")

    assert result == payload
    session.get.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_json_retry_on_500():
    """500 on first attempt, 200 on second — should retry and succeed."""
    # First call: 500
    cm_500, _ = _mock_response(500, None)
    # Second call: 200
    payload = {"ok": True}
    cm_200, _ = _mock_response(200, payload)

    session = MagicMock()
    session.get.side_effect = [cm_500, cm_200]

    with patch.object(http_client, "get_session", AsyncMock(return_value=session)):
        result = await http_client.fetch_json(
            "https://api.example.com/data", base_delay=0.01
        )

    assert result == payload
    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_fetch_json_timeout():
    """TimeoutError on every attempt returns None."""
    session = MagicMock()
    session.get.side_effect = TimeoutError("timed out")

    with patch.object(http_client, "get_session", AsyncMock(return_value=session)):
        result = await http_client.fetch_json(
            "https://api.example.com/slow",
            max_retries=3,
            base_delay=0.01,
        )

    assert result is None
    assert session.get.call_count == 3


@pytest.mark.asyncio
async def test_session_singleton():
    """Two consecutive calls to get_session() return the same object."""
    # Reset module state
    http_client._session = None

    with patch("aiohttp.TCPConnector"):
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_cls.return_value = mock_instance

            s1 = await http_client.get_session()
            s2 = await http_client.get_session()

    assert s1 is s2
    # Clean up
    http_client._session = None


@pytest.mark.asyncio
async def test_close_session():
    """After close_session(), next get_session() creates a new session."""
    http_client._session = None

    with patch("aiohttp.TCPConnector"):
        with patch("aiohttp.ClientSession") as mock_cls:
            inst1 = MagicMock()
            inst1.closed = False
            inst1.close = AsyncMock()

            inst2 = MagicMock()
            inst2.closed = False

            mock_cls.side_effect = [inst1, inst2]

            s1 = await http_client.get_session()
            assert s1 is inst1

            await http_client.close_session()
            assert http_client._session is None

            s2 = await http_client.get_session()
            assert s2 is inst2
            assert s1 is not s2

    # Clean up
    http_client._session = None
