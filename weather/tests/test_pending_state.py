"""Tests for pending maker orders state management."""

import json
import os
import pytest
from weather.pending_state import PendingOrders, pending_lock


@pytest.fixture
def tmp_path_file(tmp_path):
    return str(tmp_path / "pending_orders.json")


def _sample_order(**overrides):
    base = {
        "order_id": "ox1", "market_id": "m1", "token_id": "t1",
        "side": "yes", "price": 0.10, "size": 20.0, "amount_usd": 2.0,
        "submitted_at": "2026-02-17T10:00:00Z", "ttl_seconds": 900,
        "location": "NYC", "outcome_name": "41F or below",
        "forecast_date": "2026-02-18", "prob": 0.25,
    }
    base.update(overrides)
    return base


class TestPendingOrders:
    def test_add_and_len(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        assert len(po) == 0
        po.add(_sample_order())
        assert len(po) == 1

    def test_save_load_roundtrip(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order())
        po.save()
        po2 = PendingOrders(tmp_path_file)
        po2.load()
        assert len(po2) == 1
        assert po2.orders[0]["order_id"] == "ox1"

    def test_remove_existing(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order(order_id="a"))
        po.add(_sample_order(order_id="b"))
        removed = po.remove("a")
        assert removed is not None
        assert removed["order_id"] == "a"
        assert len(po) == 1

    def test_remove_nonexistent(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        assert po.remove("nope") is None

    def test_has_market(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order(market_id="m1"))
        assert po.has_market("m1") is True
        assert po.has_market("m2") is False

    def test_get_by_market(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order(market_id="m1"))
        result = po.get_by_market("m1")
        assert result is not None
        assert result["market_id"] == "m1"
        assert po.get_by_market("m2") is None

    def test_total_exposure(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order(amount_usd=2.0))
        po.add(_sample_order(order_id="ox2", market_id="m2", amount_usd=3.5))
        assert po.total_exposure() == pytest.approx(5.5)

    def test_atomic_save(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        po.add(_sample_order())
        po.save()
        with open(tmp_path_file) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_load_missing_file(self, tmp_path_file):
        po = PendingOrders(tmp_path_file)
        po.load()
        assert len(po) == 0

    def test_load_corrupt_file(self, tmp_path_file):
        with open(tmp_path_file, "w") as f:
            f.write("{invalid json")
        po = PendingOrders(tmp_path_file)
        po.load()
        assert len(po) == 0


class TestPendingLock:
    def test_lock_acquire_release(self, tmp_path_file):
        with pending_lock(tmp_path_file):
            pass

    def test_lock_creates_lockfile(self, tmp_path_file):
        with pending_lock(tmp_path_file):
            assert os.path.exists(tmp_path_file + ".lock")
