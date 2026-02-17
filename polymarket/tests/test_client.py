"""Tests for polymarket.client — async REST client with mocked HTTP."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from polymarket.client import CircuitOpenError, PolymarketClient, _CircuitBreaker

_FAKE_CREDS = {
    "apiKey": "test-api-key",
    "secret": "dGVzdC1zZWNyZXQta2V5LTMyYnl0ZXMhYWJjZGVmZ2g=",  # base64
    "passphrase": "test-passphrase",
}
_FAKE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"


def _make_client(**kwargs) -> PolymarketClient:
    """Create a client with fake creds (no L1 derivation)."""
    return PolymarketClient(private_key=_FAKE_KEY, api_creds=_FAKE_CREDS, **kwargs)


def _mock_response(status_code=200, json_data=None):
    """Create a mock httpx.Response with given status and JSON data."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = json.dumps(json_data) if json_data is not None else ""
    resp.raise_for_status = MagicMock()
    return resp


class TestClientInit:
    def test_stores_creds(self):
        client = _make_client()
        assert client._api_key == "test-api-key"
        assert client._passphrase == "test-passphrase"

    @patch("polymarket.client.derive_api_key")
    def test_auto_derives_when_no_creds(self, mock_derive):
        mock_derive.return_value = _FAKE_CREDS
        client = PolymarketClient(private_key=_FAKE_KEY)
        mock_derive.assert_called_once_with(_FAKE_KEY)
        assert client._api_key == "test-api-key"


class TestMarkets:
    @pytest.mark.asyncio
    async def test_get_markets(self):
        client = _make_client()
        mock_resp = _mock_response(200, [{"id": "market-1"}])
        client._http = AsyncMock()
        client._http.request.return_value = mock_resp

        result = await client.get_markets()
        assert result == [{"id": "market-1"}]

        call_args = client._http.request.call_args
        assert call_args[0][0] == "GET"  # method
        assert "/markets" in call_args[0][1]  # path

    @pytest.mark.asyncio
    async def test_get_markets_with_filters(self):
        client = _make_client()
        mock_resp = _mock_response(200, [])
        client._http = AsyncMock()
        client._http.request.return_value = mock_resp

        await client.get_markets(status="active", limit=10)

        call_args = client._http.request.call_args
        path = call_args[0][1]
        assert "status=active" in path
        assert "limit=10" in path

    @pytest.mark.asyncio
    async def test_get_market(self):
        client = _make_client()
        mock_resp = _mock_response(200, {"id": "cond-123", "question": "Test?"})
        client._http = AsyncMock()
        client._http.request.return_value = mock_resp

        result = await client.get_market("cond-123")
        assert result["id"] == "cond-123"


class TestOrderbook:
    @pytest.mark.asyncio
    async def test_get_orderbook(self):
        book = {"bids": [{"price": 0.60, "size": 100}], "asks": [{"price": 0.65, "size": 50}]}
        client = _make_client()
        mock_resp = _mock_response(200, book)
        client._http = AsyncMock()
        client._http.request.return_value = mock_resp

        result = await client.get_orderbook("token-abc")
        assert result["bids"][0]["price"] == 0.60
        assert result["asks"][0]["size"] == 50

    @pytest.mark.asyncio
    async def test_get_price(self):
        client = _make_client()
        mock_resp = _mock_response(200, {"bid": 0.60, "ask": 0.65, "last": 0.62})
        client._http = AsyncMock()
        client._http.request.return_value = mock_resp

        result = await client.get_price("token-abc")
        assert result["last"] == 0.62


class TestOrders:
    @pytest.mark.asyncio
    async def test_cancel_order(self):
        client = _make_client()
        mock_resp = _mock_response(200, {"status": "cancelled"})
        client._http = AsyncMock()
        client._http.request.return_value = mock_resp

        result = await client.cancel_order("order-123")
        assert result["status"] == "cancelled"

        call_args = client._http.request.call_args
        assert call_args[0][0] == "DELETE"
        assert call_args[0][1] == "/order"
        # Body contains the order ID
        body = call_args[1].get("content", b"").decode()
        assert "order-123" in body

    @pytest.mark.asyncio
    async def test_cancel_all(self):
        client = _make_client()
        mock_resp = _mock_response(200, {"cancelled": 3})
        client._http = AsyncMock()
        client._http.request.return_value = mock_resp

        result = await client.cancel_all()
        assert result["cancelled"] == 3

    @pytest.mark.asyncio
    async def test_get_open_orders(self):
        client = _make_client()
        mock_resp = _mock_response(200, [{"id": "order-1"}, {"id": "order-2"}])
        client._http = AsyncMock()
        client._http.request.return_value = mock_resp

        result = await client.get_open_orders()
        assert len(result) == 2


class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        retry_resp = _mock_response(429)
        ok_resp = _mock_response(200, {"ok": True})

        client = _make_client(base_delay=0.01)
        client._http = AsyncMock()
        client._http.request.side_effect = [retry_resp, ok_resp]

        result = await client._request("GET", "/test", auth=False)
        assert result == {"ok": True}
        assert client._http.request.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        retry_resp = _mock_response(500)
        ok_resp = _mock_response(200, {"ok": True})

        client = _make_client(base_delay=0.01)
        client._http = AsyncMock()
        client._http.request.side_effect = [retry_resp, ok_resp]

        result = await client._request("GET", "/test", auth=False)
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        ok_resp = _mock_response(200, {"ok": True})

        client = _make_client(base_delay=0.01)
        client._http = AsyncMock()
        client._http.request.side_effect = [httpx.ReadTimeout("timeout"), ok_resp]

        result = await client._request("GET", "/test", auth=False)
        assert result == {"ok": True}


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        """Circuit should open after failure_threshold consecutive failures."""
        cb = _CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        assert cb.state == _CircuitBreaker.CLOSED
        assert cb.allow_request()

        for _ in range(3):
            cb.record_failure()

        assert cb.state == _CircuitBreaker.OPEN
        assert not cb.allow_request()

    def test_closes_on_success(self):
        """Success should reset failure count and close the circuit."""
        cb = _CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb.state == _CircuitBreaker.CLOSED
        assert cb._failure_count == 0

    def test_half_open_after_timeout(self):
        """After recovery timeout, circuit should move to HALF_OPEN and allow one probe."""
        cb = _CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == _CircuitBreaker.OPEN

        # With recovery_timeout=0, next allow_request() should transition to HALF_OPEN
        assert cb.allow_request()
        assert cb.state == _CircuitBreaker.HALF_OPEN

        # Success should close it
        cb.record_success()
        assert cb.state == _CircuitBreaker.CLOSED

    @pytest.mark.asyncio
    async def test_client_raises_circuit_open(self):
        """Client should raise CircuitOpenError when breaker is open."""
        client = _make_client(base_delay=0.01)
        # Force breaker open
        client._breaker = _CircuitBreaker(failure_threshold=1, recovery_timeout=999)
        client._breaker.record_failure()
        assert client._breaker.state == _CircuitBreaker.OPEN

        with pytest.raises(CircuitOpenError):
            await client._request("GET", "/test", auth=False)


class TestContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        client = _make_client()
        client._http = AsyncMock()
        async with client as c:
            assert c._api_key == "test-api-key"
        client._http.aclose.assert_awaited_once()
