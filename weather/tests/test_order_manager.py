"""Tests for order manager — fill detection, TTL, reconciliation."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import pytest
from weather.pending_state import PendingOrders
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


class TestPollOnce:
    def test_fill_detected(self, tmp_path):
        """Filled order → remove from pending."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_order.return_value = {
            "status": "MATCHED", "size_matched": 20.0, "original_size": 20.0,
        }

        fills, cancels, errors = poll_once(mock_clob, po, pending_path, None, state_path)
        assert fills == 1
        assert cancels == 0
        po.load()
        assert len(po) == 0

    def test_ttl_expired(self, tmp_path):
        """Expired order → cancel + remove."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
        po.add(_make_order(order_id="ox1", submitted_at=old_time, ttl_seconds=900))
        po.save()

        mock_clob = MagicMock()
        fills, cancels, errors = poll_once(mock_clob, po, pending_path, None, state_path)
        assert cancels == 1
        mock_clob.cancel_order.assert_called_once_with("ox1")

    def test_cancelled_externally(self, tmp_path):
        """Externally cancelled → cleanup."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_order.return_value = {"status": "CANCELLED", "size_matched": 0}

        fills, cancels, errors = poll_once(mock_clob, po, pending_path, None, state_path)
        assert cancels == 1
        po.load()
        assert len(po) == 0

    def test_still_live_no_change(self, tmp_path):
        """Live order, not expired → no action."""
        pending_path = str(tmp_path / "pending.json")
        state_path = str(tmp_path / "state.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_order.return_value = {"status": "LIVE", "size_matched": 0, "original_size": 20}

        fills, cancels, errors = poll_once(mock_clob, po, pending_path, None, state_path)
        assert fills == 0 and cancels == 0
        po.load()
        assert len(po) == 1


class TestReconciliation:
    def test_removes_stale(self, tmp_path):
        """Stale entries not in CLOB open orders → removed."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.add(_make_order(order_id="ox2", market_id="m2"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_open_orders.return_value = [{"id": "ox2"}]

        cleaned = reconcile_on_startup(mock_clob, po, pending_path)
        assert cleaned == 1
        po.load()
        assert len(po) == 1
        assert po.orders[0]["order_id"] == "ox2"

    def test_no_stale(self, tmp_path):
        """All entries match CLOB → nothing removed."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_open_orders.return_value = [{"id": "ox1"}]

        cleaned = reconcile_on_startup(mock_clob, po, pending_path)
        assert cleaned == 0

    def test_reconcile_with_api_error(self, tmp_path):
        """API error during reconciliation → no changes."""
        pending_path = str(tmp_path / "pending.json")
        po = PendingOrders(pending_path)
        po.load()
        po.add(_make_order(order_id="ox1"))
        po.save()

        mock_clob = MagicMock()
        mock_clob.get_open_orders.side_effect = Exception("API down")

        cleaned = reconcile_on_startup(mock_clob, po, pending_path)
        assert cleaned == 0
        po.load()
        assert len(po) == 1
