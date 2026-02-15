"""Tests for polymarket.client — REST client with mocked HTTP."""

import json
from unittest.mock import MagicMock, patch

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


class TestClientInit:
    def test_stores_creds(self):
        client = _make_client()
        assert client.api_key == "test-api-key"
        assert client.passphrase == "test-passphrase"
        client.close()

    @patch("polymarket.client.derive_api_key")
    def test_auto_derives_when_no_creds(self, mock_derive):
        mock_derive.return_value = _FAKE_CREDS
        client = PolymarketClient(private_key=_FAKE_KEY)
        mock_derive.assert_called_once_with(_FAKE_KEY)
        assert client.api_key == "test-api-key"
        client.close()


class TestMarkets:
    @patch.object(httpx.Client, "request")
    def test_get_markets(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": "market-1"}]
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        client = _make_client()
        result = client.get_markets()
        assert result == [{"id": "market-1"}]

        call_args = mock_request.call_args
        assert call_args[0][0] == "GET"  # method
        assert "/markets" in call_args[0][1]  # path
        client.close()

    @patch.object(httpx.Client, "request")
    def test_get_markets_with_filters(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        client = _make_client()
        client.get_markets(status="active", limit=10)

        call_args = mock_request.call_args
        path = call_args[0][1]
        assert "status=active" in path
        assert "limit=10" in path
        client.close()

    @patch.object(httpx.Client, "request")
    def test_get_market(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "cond-123", "question": "Test?"}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        client = _make_client()
        result = client.get_market("cond-123")
        assert result["id"] == "cond-123"
        client.close()


class TestOrderbook:
    @patch.object(httpx.Client, "request")
    def test_get_orderbook(self, mock_request):
        book = {"bids": [{"price": 0.60, "size": 100}], "asks": [{"price": 0.65, "size": 50}]}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = book
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        client = _make_client()
        result = client.get_orderbook("token-abc")
        assert result["bids"][0]["price"] == 0.60
        assert result["asks"][0]["size"] == 50
        client.close()

    @patch.object(httpx.Client, "request")
    def test_get_price(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bid": 0.60, "ask": 0.65, "last": 0.62}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        client = _make_client()
        result = client.get_price("token-abc")
        assert result["last"] == 0.62
        client.close()


class TestOrders:
    @patch.object(httpx.Client, "request")
    def test_cancel_order(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "cancelled"}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        client = _make_client()
        result = client.cancel_order("order-123")
        assert result["status"] == "cancelled"

        call_args = mock_request.call_args
        assert call_args[0][0] == "DELETE"
        assert call_args[0][1] == "/order"
        # Body contains the order ID
        body = call_args[1].get("content", b"").decode()
        assert "order-123" in body
        client.close()

    @patch.object(httpx.Client, "request")
    def test_cancel_all(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cancelled": 3}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        client = _make_client()
        result = client.cancel_all()
        assert result["cancelled"] == 3
        client.close()

    @patch.object(httpx.Client, "request")
    def test_get_open_orders(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": "order-1"}, {"id": "order-2"}]
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        client = _make_client()
        result = client.get_open_orders()
        assert len(result) == 2
        client.close()


class TestRetry:
    @patch.object(httpx.Client, "request")
    def test_retries_on_429(self, mock_request):
        retry_resp = MagicMock()
        retry_resp.status_code = 429

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}
        ok_resp.raise_for_status = MagicMock()

        mock_request.side_effect = [retry_resp, ok_resp]

        client = _make_client(base_delay=0.01)
        result = client._request("GET", "/test", auth=False)
        assert result == {"ok": True}
        assert mock_request.call_count == 2
        client.close()

    @patch.object(httpx.Client, "request")
    def test_retries_on_500(self, mock_request):
        retry_resp = MagicMock()
        retry_resp.status_code = 500

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}
        ok_resp.raise_for_status = MagicMock()

        mock_request.side_effect = [retry_resp, ok_resp]

        client = _make_client(base_delay=0.01)
        result = client._request("GET", "/test", auth=False)
        assert result == {"ok": True}
        client.close()

    @patch.object(httpx.Client, "request")
    def test_retries_on_timeout(self, mock_request):
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}
        ok_resp.raise_for_status = MagicMock()

        mock_request.side_effect = [httpx.ReadTimeout("timeout"), ok_resp]

        client = _make_client(base_delay=0.01)
        result = client._request("GET", "/test", auth=False)
        assert result == {"ok": True}
        client.close()


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

    def test_client_raises_circuit_open(self):
        """Client should raise CircuitOpenError when breaker is open."""
        client = _make_client(base_delay=0.01)
        # Force breaker open
        client._breaker = _CircuitBreaker(failure_threshold=1, recovery_timeout=999)
        client._breaker.record_failure()
        assert client._breaker.state == _CircuitBreaker.OPEN

        with pytest.raises(CircuitOpenError):
            client._request("GET", "/test", auth=False)

        client.close()


class TestContextManager:
    def test_context_manager(self):
        with _make_client() as client:
            assert client.api_key == "test-api-key"
