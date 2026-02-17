"""Order manager daemon — monitors pending maker orders for fills and TTL expiry."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .pending_state import PendingOrders, pending_lock
from .state import TradingState, state_lock

logger = logging.getLogger(__name__)

POLL_INTERVAL = 10  # seconds


async def reconcile_on_startup(
    clob,
    pending: PendingOrders,
    pending_path: str,
) -> int:
    """Reconcile pending_orders.json with actual CLOB open orders on startup.

    Remove stale entries (orders that no longer exist on CLOB).
    Returns number of stale entries cleaned.
    """
    try:
        open_orders = await clob.get_open_orders()
    except Exception as exc:
        logger.warning("Failed to fetch open orders for reconciliation: %s", exc)
        return 0

    open_ids = {o.get("id") or o.get("orderID") for o in open_orders}
    cleaned = 0

    with pending_lock(pending_path):
        pending.load()
        stale = [o for o in pending.orders if o["order_id"] not in open_ids]
        for o in stale:
            logger.info("Reconcile: removing stale pending order %s", o["order_id"])
            pending.remove(o["order_id"])
            cleaned += 1
        if cleaned:
            pending.save()

    return cleaned


async def poll_once(
    clob,
    pending: PendingOrders,
    pending_path: str,
    state: TradingState | None,
    state_path: str,
) -> tuple[int, int, int]:
    """Single poll iteration. Returns (fills, cancels, errors)."""
    fills = 0
    cancels = 0
    errors = 0
    now = datetime.now(timezone.utc)

    with pending_lock(pending_path):
        pending.load()
        to_remove: list[str] = []
        to_record: list[dict] = []

        for order in list(pending.orders):
            order_id = order["order_id"]

            # 1. TTL check
            submitted = datetime.fromisoformat(order["submitted_at"])
            elapsed = (now - submitted).total_seconds()
            if elapsed > order.get("ttl_seconds", 900):
                logger.info("TTL expired for order %s (%.0fs)", order_id, elapsed)
                try:
                    await clob.cancel_order(order_id)
                except Exception as exc:
                    logger.debug("Cancel failed (may already be gone): %s", exc)
                to_remove.append(order_id)
                cancels += 1
                continue

            # 2. Check fill status
            try:
                status_resp = await clob.get_order(order_id)
            except Exception as exc:
                logger.debug("Get order failed for %s: %s", order_id, exc)
                errors += 1
                continue

            status = status_resp.get("status", "")
            size_matched = float(status_resp.get("size_matched", 0))
            original_size = float(status_resp.get("original_size") or order.get("size", 0))

            if status == "MATCHED" or (original_size > 0 and size_matched >= original_size * 0.99):
                # Fully filled
                logger.info("Order %s FILLED: %.1f shares @ $%.2f",
                            order_id, size_matched, order["price"])
                to_record.append({
                    "market_id": order["market_id"],
                    "outcome_name": order["outcome_name"],
                    "side": order["side"],
                    "cost_basis": order["price"],
                    "shares": size_matched or order["size"],
                    "location": order.get("location", ""),
                    "forecast_date": order.get("forecast_date", ""),
                    "event_id": order.get("event_id", ""),
                    "forecast_temp": order.get("forecast_temp"),
                    "metric": order.get("metric", "high"),
                })
                to_remove.append(order_id)
                fills += 1

            elif status == "CANCELLED":
                logger.info("Order %s cancelled externally", order_id)
                to_remove.append(order_id)
                cancels += 1

        # Apply removals
        for oid in to_remove:
            pending.remove(oid)
        pending.save()

    # Record fills in trading state (outside pending lock to avoid deadlock)
    if to_record:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with state_lock(state_path):
            state_data = TradingState.load(state_path)
            for rec in to_record:
                state_data.record_trade(**rec)
                # Record P&L: maker fill = money spent
                amount = rec.get("cost_basis", 0) * rec.get("shares", 0)
                state_data.record_daily_pnl(today_str, -amount)
                # Correlation guard: register event → market mapping
                event_id = rec.get("event_id", "")
                if event_id:
                    state_data.record_event_position(event_id, rec["market_id"])
                state_data.record_position_opened(today_str)
            state_data.save(state_path)

    return fills, cancels, errors


async def run_manager(
    clob,
    config_dir: str = "",
    poll_interval: int = POLL_INTERVAL,
) -> None:
    """Main daemon loop. Runs until KeyboardInterrupt."""
    config_dir = config_dir or str(Path(__file__).parent)
    config = Config.load(config_dir)

    pending_path = str(Path(config_dir) / "pending_orders.json")
    state_path = str(Path(config_dir) / config.state_file)

    pending = PendingOrders(pending_path)

    logger.info("Order manager starting — poll every %ds", poll_interval)

    # Startup reconciliation
    cleaned = await reconcile_on_startup(clob, pending, pending_path)
    if cleaned:
        logger.info("Reconciliation: cleaned %d stale entries", cleaned)

    while True:
        try:
            fills, cancels, errors = await poll_once(clob, pending, pending_path, None, state_path)
            if fills or cancels:
                logger.info("Poll: %d fills, %d cancels, %d errors", fills, cancels, errors)
        except Exception as exc:
            logger.error("Poll error: %s", exc)

        await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_dir = str(Path(__file__).parent)
    config = Config.load(config_dir)
    creds = config.load_api_creds(config_dir)
    if not creds:
        print("No creds.json found — cannot start order manager", file=sys.stderr)
        sys.exit(1)

    from .bridge import CLOBWeatherBridge
    bridge = CLOBWeatherBridge(config=config, api_creds=creds)
    asyncio.run(run_manager(bridge.clob, config_dir=config_dir))
