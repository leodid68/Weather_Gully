"""Public (read-only) Polymarket CLOB client — no private key required."""

import httpx

from .constants import CLOB_BASE_URL


class PublicClient:
    """Lightweight read-only client for public CLOB endpoints (no private key)."""

    def __init__(self, base_url: str = CLOB_BASE_URL, timeout: float = 15):
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def get_markets(self, **filters) -> list[dict]:
        limit = filters.pop("limit", None)
        from urllib.parse import urlencode
        params = urlencode(filters) if filters else ""
        path = "/sampling-markets" + (f"?{params}" if params else "")
        resp = self._http.get(path)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            items = data.get("data", data.get("markets", []))
        else:
            items = data
        if limit is not None:
            items = items[:int(limit)]
        return items

    def get_orderbook(self, token_id: str) -> dict:
        resp = self._http.get(f"/book?token_id={token_id}")
        resp.raise_for_status()
        return resp.json()

    def get_price(self, token_id: str) -> dict:
        resp = self._http.get(f"/midpoint?token_id={token_id}")
        resp.raise_for_status()
        data = resp.json()
        # Normalize: /midpoint returns {"mid": "0.xx"}, callers expect {"price": ...}
        if "mid" in data and "price" not in data:
            data["price"] = data["mid"]
        return data

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
