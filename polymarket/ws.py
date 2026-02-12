"""Polymarket CLOB WebSocket client — real-time market data with auto-reconnect."""

import asyncio
import json
import logging
from collections.abc import Callable

import websockets

from .constants import WS_URL

logger = logging.getLogger(__name__)

_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 60.0
_PING_INTERVAL = 30


class PolymarketWS:
    """Async WebSocket client for Polymarket real-time market data.

    Args:
        on_message: Callback invoked with each parsed JSON message.
        url: WebSocket endpoint URL.
    """

    def __init__(
        self,
        on_message: Callable[[dict], None],
        url: str = WS_URL,
    ):
        self.on_message = on_message
        self.url = url
        self._ws = None
        self._subscriptions: dict[str, set[str]] = {}  # token_id -> set of channels
        self._running = False

    async def subscribe_market(self, token_id: str) -> None:
        """Subscribe to orderbook updates for a token."""
        await self._subscribe(token_id, "market")

    async def subscribe_price(self, token_id: str) -> None:
        """Subscribe to last trade price updates for a token."""
        await self._subscribe(token_id, "price")

    async def unsubscribe(self, token_id: str) -> None:
        """Unsubscribe from all channels for a token."""
        if self._ws is not None:
            channels = self._subscriptions.pop(token_id, set())
            for channel in channels:
                msg = {"type": "unsubscribe", "channel": channel, "assets_id": token_id}
                await self._ws.send(json.dumps(msg))
                logger.debug("Unsubscribed from %s for %s", channel, token_id)

    async def run(self) -> None:
        """Main loop: connect, listen, and auto-reconnect on failure."""
        self._running = True
        backoff = _INITIAL_BACKOFF

        while self._running:
            try:
                async with websockets.connect(
                    self.url, ping_interval=_PING_INTERVAL
                ) as ws:
                    self._ws = ws
                    backoff = _INITIAL_BACKOFF
                    logger.info("WebSocket connected to %s", self.url)

                    # Re-subscribe after reconnect
                    await self._resubscribe()

                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                            self.on_message(data)
                        except json.JSONDecodeError:
                            logger.warning("Non-JSON WS message: %s", raw[:200])

            except websockets.ConnectionClosed as exc:
                logger.warning("WebSocket closed: %s — reconnecting in %.1fs", exc, backoff)
            except OSError as exc:
                logger.warning("WebSocket OS error: %s — reconnecting in %.1fs", exc, backoff)

            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

        self._ws = None

    async def close(self) -> None:
        """Stop the run loop and close the connection."""
        self._running = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _subscribe(self, token_id: str, channel: str) -> None:
        self._subscriptions.setdefault(token_id, set()).add(channel)
        if self._ws is not None:
            msg = {"type": "subscribe", "channel": channel, "assets_id": token_id}
            await self._ws.send(json.dumps(msg))
            logger.debug("Subscribed to %s for %s", channel, token_id)

    async def _resubscribe(self) -> None:
        """Re-send all subscriptions (used after reconnect)."""
        for token_id, channels in self._subscriptions.items():
            for channel in channels:
                msg = {"type": "subscribe", "channel": channel, "assets_id": token_id}
                await self._ws.send(json.dumps(msg))
                logger.debug("Re-subscribed to %s for %s", channel, token_id)
