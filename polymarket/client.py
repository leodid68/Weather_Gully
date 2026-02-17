"""Polymarket CLOB REST client — sync, with connection pooling and retry."""

import json
import logging
import random
import threading
import time
from urllib.parse import urlencode

import httpx

from .auth import build_l2_headers, derive_api_key
from .constants import CLOB_BASE_URL
from .order import build_signed_order

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {429, 500, 502, 503, 504}

# Compact JSON matching the official client (critical for HMAC validation)
_json_compact = lambda obj: json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and requests are blocked."""


class _CircuitBreaker:
    """Simple circuit breaker: CLOSED → OPEN (after failures) → HALF_OPEN → CLOSED.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout: Seconds to wait before attempting a probe request.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == self.CLOSED:
                return True
            if self.state == self.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self.state = self.HALF_OPEN
                    logger.info("Circuit breaker → HALF_OPEN (probe allowed)")
                    return True
                return False
            # HALF_OPEN: allow one probe
            return True

    def record_success(self) -> None:
        with self._lock:
            if self.state == self.HALF_OPEN:
                logger.info("Circuit breaker → CLOSED (probe succeeded)")
            self._failure_count = 0
            self.state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self.state = self.OPEN
                logger.warning(
                    "Circuit breaker → OPEN after %d consecutive failures (cooldown %.0fs)",
                    self._failure_count, self.recovery_timeout,
                )


class PolymarketClient:
    """Synchronous client for the Polymarket CLOB API.

    Args:
        private_key: Ethereum private key (hex string with 0x prefix).
        api_creds: Pre-existing API credentials dict with keys apiKey, secret, passphrase.
            If None, credentials are derived automatically via L1 auth.
        base_url: CLOB API base URL.
        max_retries: Number of retries on transient HTTP errors.
        base_delay: Initial backoff delay in seconds.
    """

    def __init__(
        self,
        private_key: str,
        api_creds: dict | None = None,
        base_url: str = CLOB_BASE_URL,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self._private_key = private_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.base_delay = base_delay

        from eth_account import Account as _Acct
        self.address = _Acct.from_key(private_key).address

        if api_creds is None:
            api_creds = derive_api_key(private_key)
        self._api_key = api_creds["apiKey"]
        self._secret = api_creds["secret"]
        self._passphrase = api_creds["passphrase"]

        self._http = httpx.Client(base_url=base_url, timeout=15)
        self._breaker = _CircuitBreaker()

    def __repr__(self) -> str:
        return f"PolymarketClient(address={self.address})"

    # ------------------------------------------------------------------
    # Low-level request
    # ------------------------------------------------------------------

    def _request(
        self, method: str, path: str, body: dict | list | None = None, auth: bool = True
    ) -> dict | list:
        if not self._breaker.allow_request():
            raise CircuitOpenError(
                f"Circuit breaker OPEN — blocking {method} {path}. "
                f"Will retry after {self._breaker.recovery_timeout:.0f}s cooldown."
            )

        body_str = _json_compact(body) if body is not None else ""
        resp = None

        for attempt in range(self.max_retries + 1):
            # Build headers inside retry loop so HMAC timestamp is fresh
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "py_clob_client",
                "Accept": "*/*",
                "Connection": "keep-alive",
            }
            if auth:
                headers.update(
                    build_l2_headers(
                        self._api_key, self._secret, self._passphrase,
                        self.address, method, path, body_str,
                    )
                )

            try:
                resp = self._http.request(
                    method,
                    path,
                    content=body_str.encode() if body_str else None,
                    headers=headers,
                )
                if resp.status_code in _RETRYABLE_CODES and attempt < self.max_retries:
                    delay = self.base_delay * (2**attempt) * (0.5 + random.random())
                    logger.warning(
                        "CLOB %d on %s %s — retry %d/%d in %.1fs",
                        resp.status_code, method, path, attempt + 1, self.max_retries, delay,
                    )
                    time.sleep(delay)
                    continue
                if resp.status_code >= 500:
                    logger.error("CLOB %d %s %s: %s", resp.status_code, method, path, resp.text)
                    self._breaker.record_failure()
                elif resp.status_code >= 400:
                    logger.error("CLOB %d %s %s: %s", resp.status_code, method, path, resp.text)
                resp.raise_for_status()
                self._breaker.record_success()
                return resp.json()
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2**attempt) * (0.5 + random.random())
                    logger.warning(
                        "CLOB timeout on %s %s — retry %d/%d in %.1fs",
                        method, path, attempt + 1, self.max_retries, delay,
                    )
                    time.sleep(delay)
                    continue
                self._breaker.record_failure()
                raise

        self._breaker.record_failure()
        if resp is None:
            raise httpx.ConnectError(
                f"All {self.max_retries} retries timed out for {method} {path}"
            )
        raise httpx.HTTPStatusError(
            f"Max retries exceeded for {method} {path}",
            request=httpx.Request(method, path),
            response=resp,
        )

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def get_markets(self, **filters) -> list[dict]:
        """Fetch available markets with optional query filters."""
        path = "/markets" + (f"?{urlencode(filters)}" if filters else "")
        return self._request("GET", path, auth=False)

    def get_market(self, condition_id: str) -> dict:
        """Fetch a single market by condition ID."""
        return self._request("GET", f"/markets/{condition_id}", auth=False)

    # ------------------------------------------------------------------
    # Orderbook
    # ------------------------------------------------------------------

    def get_orderbook(self, token_id: str) -> dict:
        """Fetch the orderbook for a token. Returns {bids: [...], asks: [...]}."""
        return self._request("GET", f"/book?token_id={token_id}", auth=False)

    def get_price(self, token_id: str) -> dict:
        """Fetch current midpoint price for a token."""
        data = self._request("GET", f"/midpoint?token_id={token_id}", auth=False)
        # Normalize: /midpoint returns {"mid": "0.xx"}, callers expect {"price": ...}
        if "mid" in data and "price" not in data:
            data["price"] = data["mid"]
        return data

    def is_neg_risk(self, token_id: str) -> bool:
        """Check if a token uses the neg-risk exchange."""
        resp = self._request("GET", f"/neg-risk?token_id={token_id}", auth=False)
        return resp.get("neg_risk", False)

    def get_tick_size(self, token_id: str) -> str:
        """Fetch minimum tick size for a token from the CLOB API."""
        try:
            resp = self._request("GET", f"/tick-size?token_id={token_id}", auth=False)
            return resp.get("minimum_tick_size", "0.01")
        except Exception as exc:
            logger.warning("Could not fetch tick_size for %s: %s", token_id[:16], exc)
            return "0.01"

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def post_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        neg_risk: bool | None = None,
        order_type: str = "GTC",
        post_only: bool = False,
        **kwargs,
    ) -> dict:
        """Build, sign, and submit an order to the CLOB.

        If neg_risk is None, auto-detects from the API.
        """
        if neg_risk is None:
            neg_risk = self.is_neg_risk(token_id)
            logger.info("Auto-detected neg_risk=%s for token", neg_risk)
        tick_size = self.get_tick_size(token_id)
        signed = build_signed_order(
            maker=self.address,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            private_key=self._private_key,
            neg_risk=neg_risk,
            **kwargs,
        )
        # Convert numeric fields to strings (CLOB API requirement)
        order_payload = {
            "salt": str(signed["salt"]),
            "maker": signed["maker"],
            "signer": signed["signer"],
            "taker": signed["taker"],
            "tokenId": str(signed["tokenId"]),
            "makerAmount": str(signed["makerAmount"]),
            "takerAmount": str(signed["takerAmount"]),
            "expiration": str(signed["expiration"]),
            "nonce": str(signed["nonce"]),
            "feeRateBps": str(signed["feeRateBps"]),
            "side": "BUY" if signed["side"] == 0 else "SELL",
            "signatureType": signed["signatureType"],
            "signature": signed["signature"],
        }
        body = {
            "order": order_payload,
            "owner": self._api_key,
            "orderType": order_type,
            "postOnly": post_only,
            "tickSize": tick_size,
        }
        logger.debug("POST /order side=%s price=%s size=%s",
                     order_payload["side"], order_payload["makerAmount"], order_payload["takerAmount"])
        return self._request("POST", "/order", body=body)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order by its ID."""
        return self._request("DELETE", "/order", body={"orderID": order_id})

    def cancel_all(self) -> dict:
        """Cancel all open orders."""
        return self._request("DELETE", "/cancel-all")

    def get_open_orders(self, **filters) -> list[dict]:
        """Fetch all open orders for the authenticated user."""
        path = "/data/orders" + (f"?{urlencode(filters)}" if filters else "")
        return self._request("GET", path)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def get_trades(self, **filters) -> list[dict]:
        """Fetch trade history with optional filters."""
        path = "/data/trades" + (f"?{urlencode(filters)}" if filters else "")
        return self._request("GET", path)

    def get_order(self, order_id: str) -> dict:
        """Fetch a single order by ID to check fill status."""
        return self._request("GET", f"/data/order/{order_id}")

    def is_order_filled(self, order_id: str) -> bool:
        """Check whether an order has been fully filled."""
        try:
            order = self.get_order(order_id)
            if order.get("status") == "MATCHED":
                return True
            size_matched = order.get("size_matched")
            original_size = order.get("original_size")
            if size_matched is not None and original_size is not None:
                return abs(float(size_matched) - float(original_size)) < 1e-9
            return False
        except Exception as exc:
            logger.warning("Could not verify fill for order %s: %s", order_id, exc)
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
