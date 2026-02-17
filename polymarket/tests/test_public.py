"""Tests for polymarket.public — async PublicClient with retry."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from polymarket.public import PublicClient


def _mock_response(status_code=200, json_data=None):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.raise_for_status = MagicMock()
    return resp


class TestPublicClientRetry:

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Successful request should not retry."""
        client = PublicClient(base_delay=0.01)
        ok_resp = _mock_response(200, {"bids": [], "asks": []})

        client._http = AsyncMock()
        client._http.request.return_value = ok_resp

        result = await client.get_orderbook("token-1")
        assert result == {"bids": [], "asks": []}
        assert client._http.request.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        """429 should trigger retry and eventually succeed."""
        client = PublicClient(base_delay=0.01)

        retry_resp = _mock_response(429)
        ok_resp = _mock_response(200, {"bids": [{"price": 0.5}], "asks": []})

        client._http = AsyncMock()
        client._http.request.side_effect = [retry_resp, ok_resp]

        result = await client.get_orderbook("token-1")
        assert result["bids"][0]["price"] == 0.5
        assert client._http.request.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Timeout should trigger retry and eventually succeed."""
        client = PublicClient(base_delay=0.01)

        ok_resp = _mock_response(200, {"mid": "0.55"})

        client._http = AsyncMock()
        client._http.request.side_effect = [httpx.ReadTimeout("timeout"), ok_resp]

        result = await client.get_price("token-1")
        assert result["price"] == "0.55"
        assert client._http.request.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """After max_retries, should raise."""
        client = PublicClient(max_retries=2, base_delay=0.01)

        fail_resp = _mock_response(503)
        fail_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=fail_resp,
        )

        client._http = AsyncMock()
        client._http.request.return_value = fail_resp

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_orderbook("token-1")
