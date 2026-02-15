"""Tests for polymarket.public — PublicClient with retry."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from polymarket.public import PublicClient


class TestPublicClientRetry:

    def test_success_no_retry(self):
        """Successful request should not retry."""
        client = PublicClient(base_delay=0.01)
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"bids": [], "asks": []}
        ok_resp.raise_for_status = MagicMock()

        with patch.object(client._http, "request", return_value=ok_resp) as mock_req:
            result = client.get_orderbook("token-1")
            assert result == {"bids": [], "asks": []}
            assert mock_req.call_count == 1

        client.close()

    def test_retry_on_429(self):
        """429 should trigger retry and eventually succeed."""
        client = PublicClient(base_delay=0.01)

        retry_resp = MagicMock()
        retry_resp.status_code = 429

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"bids": [{"price": 0.5}], "asks": []}
        ok_resp.raise_for_status = MagicMock()

        with patch.object(client._http, "request", side_effect=[retry_resp, ok_resp]) as mock_req:
            result = client.get_orderbook("token-1")
            assert result["bids"][0]["price"] == 0.5
            assert mock_req.call_count == 2

        client.close()

    def test_retry_on_timeout(self):
        """Timeout should trigger retry and eventually succeed."""
        client = PublicClient(base_delay=0.01)

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"mid": "0.55"}
        ok_resp.raise_for_status = MagicMock()

        with patch.object(
            client._http, "request",
            side_effect=[httpx.ReadTimeout("timeout"), ok_resp],
        ) as mock_req:
            result = client.get_price("token-1")
            assert result["price"] == "0.55"
            assert mock_req.call_count == 2

        client.close()

    def test_max_retries_exhausted(self):
        """After max_retries, should raise."""
        client = PublicClient(max_retries=2, base_delay=0.01)

        fail_resp = MagicMock()
        fail_resp.status_code = 503
        fail_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=fail_resp,
        )

        with patch.object(client._http, "request", return_value=fail_resp):
            with pytest.raises(httpx.HTTPStatusError):
                client.get_orderbook("token-1")

        client.close()
