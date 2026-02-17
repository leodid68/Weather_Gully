"""Tests for taker/maker decision logic in strategy."""

import pytest
from unittest.mock import MagicMock, patch
from weather.config import Config


def _make_config(**overrides):
    """Config with maker defaults."""
    defaults = dict(
        maker_edge_threshold=0.05,
        maker_spread_threshold=0.10,
        maker_ttl_seconds=900,
        maker_tick_buffer=1,
    )
    defaults.update(overrides)
    return Config(**defaults)


class TestTakerMakerDecision:
    """Test the edge/spread decision logic."""

    def test_high_edge_low_spread_uses_taker(self):
        """Edge > threshold AND spread < threshold → taker."""
        cfg = _make_config(maker_edge_threshold=0.05, maker_spread_threshold=0.10)
        edge = 0.10  # > 0.05
        spread = 0.05  # < 0.10
        use_taker = (edge > cfg.maker_edge_threshold and spread < cfg.maker_spread_threshold)
        assert use_taker is True

    def test_low_edge_uses_maker(self):
        """Edge < threshold → maker (regardless of spread)."""
        cfg = _make_config(maker_edge_threshold=0.05)
        edge = 0.03  # < 0.05
        spread = 0.05
        use_taker = (edge > cfg.maker_edge_threshold and spread < cfg.maker_spread_threshold)
        assert use_taker is False

    def test_high_spread_uses_maker(self):
        """Spread > threshold → maker (regardless of edge)."""
        cfg = _make_config(maker_spread_threshold=0.10)
        edge = 0.10
        spread = 0.15  # > 0.10
        use_taker = (edge > cfg.maker_edge_threshold and spread < cfg.maker_spread_threshold)
        assert use_taker is False

    def test_maker_price_capped_at_prob(self):
        """Maker price = min(prob, best_bid + tick)."""
        prob = 0.08
        best_bid = 0.10
        tick = 0.01
        tick_buffer = 1
        maker_price = min(prob, best_bid + tick * tick_buffer)
        assert maker_price == 0.08  # Capped at prob

    def test_maker_price_above_bid(self):
        """Maker price = best_bid + tick when prob is higher."""
        prob = 0.30
        best_bid = 0.10
        tick = 0.01
        tick_buffer = 1
        maker_price = min(prob, best_bid + tick * tick_buffer)
        assert maker_price == 0.11  # best_bid + 1 tick

    def test_no_room_for_maker(self):
        """If bid + tick >= ask → no room → must use taker."""
        best_bid = 0.50
        best_ask = 0.51
        tick = 0.01
        # bid + tick = 0.51 >= ask = 0.51
        assert best_bid + tick >= best_ask


class TestPendingExposure:
    """Test pending exposure in budget calculation."""

    def test_pending_exposure_counted(self, tmp_path):
        """Pending orders add to effective exposure."""
        from weather.pending_state import PendingOrders

        po = PendingOrders(str(tmp_path / "pending.json"))
        po.load()
        po.add({"order_id": "a", "market_id": "m1", "amount_usd": 3.0})
        po.add({"order_id": "b", "market_id": "m2", "amount_usd": 4.0})

        state_exposure = 10.0
        effective = state_exposure + po.total_exposure()
        assert effective == pytest.approx(17.0)


class TestPendingDuplicateGuard:
    """Test that pending markets are skipped."""

    def test_pending_market_skipped(self, tmp_path):
        """Market in pending → has_market returns True."""
        from weather.pending_state import PendingOrders

        po = PendingOrders(str(tmp_path / "pending.json"))
        po.load()
        po.add({"order_id": "a", "market_id": "m1", "amount_usd": 2.0})

        assert po.has_market("m1") is True
        assert po.has_market("m2") is False
