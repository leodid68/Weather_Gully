"""Public (read-only) Polymarket CLOB client — async, no private key required."""

import asyncio
import logging
import random

import httpx

from .constants import CLOB_BASE_URL

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {429, 500, 502, 503, 504}


class PublicClient:
    """Lightweight async read-only client for public CLOB endpoints (no private key).

    Args:
        base_url: CLOB API base URL.
        timeout: HTTP request timeout in seconds.
        max_retries: Number of retries on transient HTTP errors.
        base_delay: Initial backoff delay in seconds.
    """

    def __init__(
        self,
        base_url: str = CLOB_BASE_URL,
        timeout: float = 15,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def _request(self, method: str, path: str) -> httpx.Response:
        """Execute an HTTP request with retry + jitter on transient errors."""
        resp = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._http.request(method, path)
                if resp.status_code in _RETRYABLE_CODES and attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt) * (0.5 + random.random())
                    logger.warning(
                        "PublicClient %d on %s %s — retry %d/%d in %.1fs",
                        resp.status_code, method, path, attempt + 1, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt) * (0.5 + random.random())
                    logger.warning(
                        "PublicClient timeout on %s %s — retry %d/%d in %.1fs",
                        method, path, attempt + 1, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        # All retries exhausted — raise last response error
        if resp is not None:
            resp.raise_for_status()
        raise httpx.ConnectError(f"All {self.max_retries} retries exhausted for {method} {path}")

    async def get_markets(self, **filters) -> list[dict]:
        limit = filters.pop("limit", None)
        from urllib.parse import urlencode
        params = urlencode(filters) if filters else ""
        path = "/sampling-markets" + (f"?{params}" if params else "")
        resp = await self._request("GET", path)
        data = resp.json()
        if isinstance(data, dict):
            items = data.get("data", data.get("markets", []))
        else:
            items = data
        if limit is not None:
            items = items[:int(limit)]
        return items

    async def get_orderbook(self, token_id: str) -> dict:
        resp = await self._request("GET", f"/book?token_id={token_id}")
        return resp.json()

    async def get_price(self, token_id: str) -> dict:
        resp = await self._request("GET", f"/midpoint?token_id={token_id}")
        data = resp.json()
        # Normalize: /midpoint returns {"mid": "0.xx"}, callers expect {"price": ...}
        if "mid" in data and "price" not in data:
            data["price"] = data["mid"]
        return data

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
