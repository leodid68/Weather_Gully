"""Tests for polymarket.order — order construction and EIP-712 signing."""

from unittest.mock import patch

from eth_account import Account

from polymarket.order import (
    SIDE_BUY,
    SIDE_SELL,
    SIGNATURE_TYPE_EOA,
    USDC_UNIT,
    ZERO_ADDRESS,
    build_order,
    build_signed_order,
    sign_order,
)


# Deterministic test key (DO NOT use with real funds)
_TEST_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
_TEST_ACCOUNT = Account.from_key(_TEST_KEY)
_TEST_ADDRESS = _TEST_ACCOUNT.address
_TEST_TOKEN_ID = "12345678"


class TestBuildOrder:
    def test_buy_amounts(self):
        """BUY: makerAmount = price * size * 1e6, takerAmount = size * 1e6."""
        order = build_order(
            maker=_TEST_ADDRESS,
            token_id=_TEST_TOKEN_ID,
            side="BUY",
            price=0.65,
            size=100,
        )
        assert order["makerAmount"] == round(0.65 * 100 * USDC_UNIT)
        assert order["takerAmount"] == round(100 * USDC_UNIT)

    def test_sell_amounts(self):
        """SELL: makerAmount = size * 1e6, takerAmount = price * size * 1e6."""
        order = build_order(
            maker=_TEST_ADDRESS,
            token_id=_TEST_TOKEN_ID,
            side="SELL",
            price=0.70,
            size=50,
        )
        assert order["makerAmount"] == round(50 * USDC_UNIT)
        assert order["takerAmount"] == round(0.70 * 50 * USDC_UNIT)

    def test_side_encoding(self):
        buy_order = build_order(_TEST_ADDRESS, _TEST_TOKEN_ID, "BUY", 0.5, 10)
        sell_order = build_order(_TEST_ADDRESS, _TEST_TOKEN_ID, "SELL", 0.5, 10)
        assert buy_order["side"] == SIDE_BUY
        assert sell_order["side"] == SIDE_SELL

    def test_case_insensitive_side(self):
        order = build_order(_TEST_ADDRESS, _TEST_TOKEN_ID, "buy", 0.5, 10)
        assert order["side"] == SIDE_BUY

    def test_defaults(self):
        order = build_order(_TEST_ADDRESS, _TEST_TOKEN_ID, "BUY", 0.5, 10)
        assert order["maker"] == _TEST_ADDRESS
        assert order["signer"] == _TEST_ADDRESS
        assert order["taker"] == ZERO_ADDRESS
        assert order["tokenId"] == int(_TEST_TOKEN_ID)
        assert order["expiration"] == 0
        assert order["nonce"] == 0
        assert order["feeRateBps"] == 0
        assert order["signatureType"] == SIGNATURE_TYPE_EOA

    def test_custom_params(self):
        order = build_order(
            maker=_TEST_ADDRESS,
            token_id=_TEST_TOKEN_ID,
            side="BUY",
            price=0.5,
            size=10,
            fee_rate_bps=100,
            expiration=1700000000,
            nonce=42,
            taker="0x1111111111111111111111111111111111111111",
        )
        assert order["feeRateBps"] == 100
        assert order["expiration"] == 1700000000
        assert order["nonce"] == 42
        assert order["taker"] == "0x1111111111111111111111111111111111111111"

    def test_salt_is_random(self):
        o1 = build_order(_TEST_ADDRESS, _TEST_TOKEN_ID, "BUY", 0.5, 10)
        o2 = build_order(_TEST_ADDRESS, _TEST_TOKEN_ID, "BUY", 0.5, 10)
        assert o1["salt"] != o2["salt"]


class TestSignOrder:
    @patch("polymarket.order._generate_salt", return_value=999)
    def test_returns_hex_string(self, _):
        order = build_order(_TEST_ADDRESS, _TEST_TOKEN_ID, "BUY", 0.5, 10)
        sig = sign_order(order, _TEST_KEY)
        assert isinstance(sig, str)
        # Hex string (with or without 0x prefix)
        sig_clean = sig.removeprefix("0x")
        assert len(sig_clean) >= 128  # 64 bytes min (r + s + v)
        int(sig_clean, 16)  # Must be valid hex

    @patch("polymarket.order._generate_salt", return_value=999)
    def test_deterministic(self, _):
        """Same order + key produces same signature."""
        order = build_order(_TEST_ADDRESS, _TEST_TOKEN_ID, "BUY", 0.5, 10)
        sig1 = sign_order(order, _TEST_KEY)
        sig2 = sign_order(order, _TEST_KEY)
        assert sig1 == sig2

    @patch("polymarket.order._generate_salt", return_value=999)
    def test_neg_risk_different_sig(self, _):
        """Different exchange address (neg_risk) produces different signature."""
        order = build_order(_TEST_ADDRESS, _TEST_TOKEN_ID, "BUY", 0.5, 10)
        sig_standard = sign_order(order, _TEST_KEY, neg_risk=False)
        sig_neg = sign_order(order, _TEST_KEY, neg_risk=True)
        assert sig_standard != sig_neg


class TestBuildSignedOrder:
    @patch("polymarket.order._generate_salt", return_value=999)
    def test_contains_signature(self, _):
        result = build_signed_order(
            maker=_TEST_ADDRESS,
            token_id=_TEST_TOKEN_ID,
            side="BUY",
            price=0.5,
            size=10,
            private_key=_TEST_KEY,
        )
        assert "signature" in result
        assert result["maker"] == _TEST_ADDRESS
        assert result["side"] == SIDE_BUY

    @patch("polymarket.order._generate_salt", return_value=999)
    def test_signature_matches_sign_order(self, _):
        result = build_signed_order(
            maker=_TEST_ADDRESS,
            token_id=_TEST_TOKEN_ID,
            side="BUY",
            price=0.5,
            size=10,
            private_key=_TEST_KEY,
        )
        order = {k: v for k, v in result.items() if k != "signature"}
        expected_sig = sign_order(order, _TEST_KEY)
        assert result["signature"] == expected_sig
