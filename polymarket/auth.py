"""Polymarket CLOB authentication — L1 API key derivation + L2 HMAC request signing."""

import base64
import hashlib
import hmac
import time

import httpx
from eth_account import Account
from eth_account.messages import encode_typed_data

from .constants import CLOB_BASE_URL, CHAIN_ID

# EIP-712 types for CLOB auth message
_AUTH_DOMAIN = {
    "name": "ClobAuthDomain",
    "version": "1",
    "chainId": CHAIN_ID,
}

_AUTH_TYPES = {
    "ClobAuth": [
        {"name": "address", "type": "address"},
        {"name": "timestamp", "type": "string"},
        {"name": "nonce", "type": "uint256"},
        {"name": "message", "type": "string"},
    ]
}


def _sign_l1(private_key: str, nonce: int = 0) -> tuple[Account, dict]:
    """Sign a ClobAuth EIP-712 message. Returns (account, headers)."""
    acct = Account.from_key(private_key)
    ts = str(int(time.time()))
    msg_value = {
        "address": acct.address,
        "timestamp": ts,
        "nonce": nonce,
        "message": "This message attests that I control the given wallet",
    }
    signable = encode_typed_data(
        domain_data=_AUTH_DOMAIN,
        message_types={"ClobAuth": _AUTH_TYPES["ClobAuth"]},
        message_data=msg_value,
    )
    signed = acct.sign_message(signable)
    sig_hex = "0x" + signed.signature.hex()
    headers = {
        "Content-Type": "application/json",
        "POLY_ADDRESS": acct.address,
        "POLY_SIGNATURE": sig_hex,
        "POLY_TIMESTAMP": ts,
        "POLY_NONCE": str(nonce),
    }
    return acct, headers


def derive_api_key(private_key: str, nonce: int = 0) -> dict:
    """Obtain API credentials. Tries to create first, falls back to deriving existing.

    Returns dict with keys: apiKey, secret, passphrase.
    """
    acct, headers = _sign_l1(private_key, nonce)

    # Try creating a new API key
    resp = httpx.post(
        f"{CLOB_BASE_URL}/auth/api-key",
        headers=headers,
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json()

    # If creation fails (key already exists), derive the existing one
    acct, headers = _sign_l1(private_key, nonce)
    resp = httpx.get(
        f"{CLOB_BASE_URL}/auth/derive-api-key",
        headers=headers,
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json()

    raise RuntimeError(f"Auth failed ({resp.status_code}): {resp.text}")


def build_hmac_signature(
    secret: str, timestamp: str, method: str, path: str, body: str = ""
) -> str:
    """Compute HMAC-SHA256 signature for L2 request authentication."""
    message = timestamp + method + path
    if body:
        message += body
    key = base64.urlsafe_b64decode(secret)
    sig = hmac.new(key, message.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode()


def build_l2_headers(
    api_key: str,
    secret: str,
    passphrase: str,
    address: str,
    method: str,
    path: str,
    body: str = "",
) -> dict:
    """Build the full set of L2 authentication headers for a CLOB request."""
    ts = str(int(time.time()))
    sig = build_hmac_signature(secret, ts, method, path, body)
    return {
        "POLY_ADDRESS": address,
        "POLY_SIGNATURE": sig,
        "POLY_TIMESTAMP": ts,
        "POLY_API_KEY": api_key,
        "POLY_PASSPHRASE": passphrase,
    }
