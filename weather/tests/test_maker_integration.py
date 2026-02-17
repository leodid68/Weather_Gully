"""Integration tests for the full maker order lifecycle."""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
import pytest
from weather.pending_state import PendingOrders, pending_lock
from weather.order_manager import poll_once, reconcile_on_startup


def _make_order(**overrides):
    base = {
        "order_id": "ox1", "market_id": "m1", "token_id": "t1",
        "side": "yes", "price": 0.10, "size": 20.0, "amount_usd": 2.0,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "ttl_seconds": 900,
        "location": "NYC", "outcome_name": "41F or below",
        "forecast_date": "2026-02-18", "prob": 0.25,
    }
    base.update(overrides)
    return base


class TestFullMakerCycle:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_post_fill_record(self, tmp_path):
        """Full cycle: post maker → fill → record in state."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")

        # 1. Strategy posts maker order
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()
        assert len(po) == 1

        # 2. Order manager detects fill
        mock_clob = AsyncMock()
        mock_clob.get_order.return_value = {
            "status": "MATCHED", "size_matched": 20.0, "original_size": 20.0,
        }

        fills, _, _ = await poll_once(mock_clob, po, pending_path, None, state_path)
        assert fills == 1

        # 3. Pending cleaned
        po.load()
        assert len(po) == 0

    def test_pending_blocks_duplicate(self, tmp_path):
        """Market already in pending → has_market returns True → strategy should skip."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(market_id="m1"))
        po.save()

        assert po.has_market("m1") is True
        assert po.has_market("m2") is False

    @pytest.mark.asyncio(loop_scope="function")
    async def test_ttl_frees_capital(self, tmp_path):
        """TTL expired → order cancelled, exposure freed."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
        po.add(_make_order(order_id="ox1", submitted_at=old_time, ttl_seconds=900, amount_usd=5.0))
        po.save()

        initial_exposure = po.total_exposure()
        assert initial_exposure == pytest.approx(5.0)

        mock_clob = AsyncMock()
        _, cancels, _ = await poll_once(mock_clob, po, pending_path, None, state_path)
        assert cancels == 1

        po.load()
        assert po.total_exposure() == pytest.approx(0.0)

    def test_pending_exposure_in_budget(self, tmp_path):
        """Pending orders counted in effective exposure."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(amount_usd=3.0))
        po.add(_make_order(order_id="ox2", market_id="m2", amount_usd=4.0))

        assert po.total_exposure() == pytest.approx(7.0)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_reconcile_after_crash(self, tmp_path):
        """After crash, reconcile removes orders not on CLOB."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1", amount_usd=2.0))
        po.add(_make_order(order_id="ox2", market_id="m2", amount_usd=3.0))
        po.add(_make_order(order_id="ox3", market_id="m3", amount_usd=4.0))
        po.save()

        mock_clob = AsyncMock()
        # Only ox2 is still open on CLOB
        mock_clob.get_open_orders.return_value = [{"id": "ox2"}]

        cleaned = await reconcile_on_startup(mock_clob, po, pending_path)
        assert cleaned == 2

        po.load()
        assert len(po) == 1
        assert po.orders[0]["order_id"] == "ox2"
        assert po.total_exposure() == pytest.approx(3.0)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_multiple_orders_mixed_status(self, tmp_path):
        """Multiple orders: one filled, one expired, one live."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()

        # Order 1: will be filled
        po.add(_make_order(order_id="filled1", market_id="m1"))
        # Order 2: will be expired
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
        po.add(_make_order(order_id="expired1", market_id="m2", submitted_at=old_time, ttl_seconds=900))
        # Order 3: still live
        po.add(_make_order(order_id="live1", market_id="m3"))
        po.save()

        mock_clob = AsyncMock()

        async def get_order_side_effect(order_id):
            if order_id == "filled1":
                return {"status": "MATCHED", "size_matched": 20.0, "original_size": 20.0}
            elif order_id == "live1":
                return {"status": "LIVE", "size_matched": 0, "original_size": 20}
            return {"status": "LIVE", "size_matched": 0, "original_size": 20}

        mock_clob.get_order.side_effect = get_order_side_effect

        fills, cancels, errors = await poll_once(mock_clob, po, pending_path, None, state_path)
        assert fills == 1
        assert cancels == 1  # expired

        po.load()
        assert len(po) == 1
        assert po.orders[0]["order_id"] == "live1"
