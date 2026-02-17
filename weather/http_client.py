"""Shared async HTTP client — replaces urllib across weather modules."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import ssl
from typing import Any

import aiohttp
import certifi

logger = logging.getLogger(__name__)

_session: aiohttp.ClientSession | None = None


def _make_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


_SSL_CTX = _make_ssl_context()


async def get_session() -> aiohttp.ClientSession:
    """Get or create the shared aiohttp session."""
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(ssl=_SSL_CTX, limit=20)
        _session = aiohttp.ClientSession(connector=connector)
    return _session


async def close_session() -> None:
    """Close the shared session (call at shutdown)."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


async def fetch_json(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict | list | None:
    """Async GET that returns parsed JSON, with retry and backoff.

    Returns None on failure (never raises).
    """
    session = await get_session()
    for attempt in range(max_retries):
        try:
            async with session.get(
                url, params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                if resp.status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    logger.debug("HTTP %d from %s — retry in %.1fs", resp.status, url[:80], delay)
                    await asyncio.sleep(delay)
                    continue
                logger.warning("HTTP %d from %s", resp.status, url[:80])
                return None
        except (aiohttp.ClientError, TimeoutError) as exc:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                logger.debug("Fetch error %s — retry in %.1fs: %s", url[:80], delay, exc)
                await asyncio.sleep(delay)
            else:
                logger.warning("Fetch failed after %d attempts: %s — %s", max_retries, url[:80], exc)
                return None
    return None


async def post_json(
    url: str,
    *,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: int = 15,
) -> dict | None:
    """Async POST that returns parsed JSON. Returns None on failure."""
    session = await get_session()
    try:
        async with session.post(
            url, json=body, headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status in (200, 201):
                return await resp.json(content_type=None)
            logger.warning("POST %d from %s", resp.status, url[:80])
            return None
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.warning("POST failed: %s — %s", url[:80], exc)
        return None
