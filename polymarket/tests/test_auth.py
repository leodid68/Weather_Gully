"""Tests for polymarket.auth — HMAC signature and L2 header construction."""

import base64
import hashlib
import hmac
import time
from unittest.mock import patch

from polymarket.auth import build_hmac_signature, build_l2_headers


class TestBuildHmacSignature:
    def test_deterministic(self):
        """Same inputs produce the same signature."""
        secret = base64.urlsafe_b64encode(b"test-secret-key-32bytes!abcdefgh").decode()
        sig1 = build_hmac_signature(secret, "1700000000", "GET", "/orders")
        sig2 = build_hmac_signature(secret, "1700000000", "GET", "/orders")
        assert sig1 == sig2

    def test_different_timestamp_changes_sig(self):
        secret = base64.urlsafe_b64encode(b"test-secret-key-32bytes!abcdefgh").decode()
        sig1 = build_hmac_signature(secret, "1700000000", "GET", "/orders")
        sig2 = build_hmac_signature(secret, "1700000001", "GET", "/orders")
        assert sig1 != sig2

    def test_different_method_changes_sig(self):
        secret = base64.urlsafe_b64encode(b"test-secret-key-32bytes!abcdefgh").decode()
        sig1 = build_hmac_signature(secret, "1700000000", "GET", "/orders")
        sig2 = build_hmac_signature(secret, "1700000000", "POST", "/orders")
        assert sig1 != sig2

    def test_body_included(self):
        secret = base64.urlsafe_b64encode(b"test-secret-key-32bytes!abcdefgh").decode()
        sig1 = build_hmac_signature(secret, "1700000000", "POST", "/order", "")
        sig2 = build_hmac_signature(secret, "1700000000", "POST", "/order", '{"side":"BUY"}')
        assert sig1 != sig2

    def test_matches_manual_computation(self):
        raw_key = b"mysecretkey12345mysecretkey12345"
        secret = base64.urlsafe_b64encode(raw_key).decode()
        ts, method, path, body = "1700000000", "POST", "/order", '{"x":1}'
        message = ts + method + path + body
        expected = base64.urlsafe_b64encode(
            hmac.new(raw_key, message.encode(), hashlib.sha256).digest()
        ).decode()
        assert build_hmac_signature(secret, ts, method, path, body) == expected

    def test_empty_body_not_appended(self):
        """Empty body should not change the signature vs no body."""
        secret = base64.urlsafe_b64encode(b"test-secret-key-32bytes!abcdefgh").decode()
        sig1 = build_hmac_signature(secret, "1700000000", "GET", "/orders")
        sig2 = build_hmac_signature(secret, "1700000000", "GET", "/orders", "")
        assert sig1 == sig2


class TestBuildL2Headers:
    @patch("polymarket.auth.time")
    def test_all_headers_present(self, mock_time):
        mock_time.time.return_value = 1700000000
        secret = base64.urlsafe_b64encode(b"test-secret-key-32bytes!abcdefgh").decode()
        headers = build_l2_headers(
            api_key="my-api-key",
            secret=secret,
            passphrase="my-pass",
            address="0xMyWalletAddress",
            method="GET",
            path="/orders",
        )
        assert headers["POLY_API_KEY"] == "my-api-key"
        assert headers["POLY_PASSPHRASE"] == "my-pass"
        assert headers["POLY_TIMESTAMP"] == "1700000000"
        assert headers["POLY_ADDRESS"] == "0xMyWalletAddress"
        assert "POLY_SIGNATURE" in headers
        # POLY_NONCE should NOT be in L2 headers
        assert "POLY_NONCE" not in headers

    @patch("polymarket.auth.time")
    def test_signature_uses_timestamp(self, mock_time):
        mock_time.time.return_value = 1700000000
        secret = base64.urlsafe_b64encode(b"test-secret-key-32bytes!abcdefgh").decode()
        headers = build_l2_headers("key", secret, "pass", "0xAddr", "GET", "/orders")
        expected_sig = build_hmac_signature(secret, "1700000000", "GET", "/orders")
        assert headers["POLY_SIGNATURE"] == expected_sig

    @patch("polymarket.auth.time")
    def test_address_is_wallet_not_api_key(self, mock_time):
        mock_time.time.return_value = 1700000000
        secret = base64.urlsafe_b64encode(b"test-secret-key-32bytes!abcdefgh").decode()
        headers = build_l2_headers("my-api-key", secret, "pass", "0xWallet", "GET", "/x")
        assert headers["POLY_ADDRESS"] == "0xWallet"
        assert headers["POLY_API_KEY"] == "my-api-key"
