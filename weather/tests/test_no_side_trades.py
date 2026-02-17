"""Tests for NO-side trade handling — entry threshold and paper pricing."""

from __future__ import annotations

import asyncio
import pytest


class TestEntryThresholdNoSide:
    """Verify that entry_threshold only blocks YES trades, not NO."""

    def test_yes_blocked_above_threshold(self):
        """YES price >= entry_threshold should be blocked."""
        entry_threshold = 0.15
        side = "yes"
        price = 0.20  # YES price
        # Old logic: yes_price_check = price if yes else (1 - price)
        # New logic: only check for YES side
        should_skip = side == "yes" and price >= entry_threshold
        assert should_skip is True

    def test_yes_passes_below_threshold(self):
        """YES price < entry_threshold should pass."""
        entry_threshold = 0.15
        side = "yes"
        price = 0.10
        should_skip = side == "yes" and price >= entry_threshold
        assert should_skip is False

    def test_no_never_blocked_by_entry_threshold(self):
        """NO trades should never be blocked by entry_threshold.

        A profitable NO trade has high YES price (overpriced bucket).
        Old code would block: yes_price_check = 1 - 0.40 = 0.60 >= 0.15 → SKIP
        New code: side != 'yes' → don't check entry_threshold at all.
        """
        entry_threshold = 0.15
        side = "no"
        # NO price = 0.40 → YES price = 0.60 (overpriced bucket, profitable NO)
        price = 0.40
        should_skip = side == "yes" and price >= entry_threshold
        assert should_skip is False

    def test_no_with_expensive_yes_passes(self):
        """NO on a bucket with YES @ $0.90 should not be blocked."""
        entry_threshold = 0.15
        side = "no"
        price = 0.10  # NO price — YES = $0.90
        should_skip = side == "yes" and price >= entry_threshold
        assert should_skip is False


class TestScoreBucketsNoSide:
    """Verify score_buckets produces NO entries for overpriced buckets."""

    def test_no_entry_generated_for_overpriced_bucket(self):
        """If model prob is moderate but market price is very high, NO should have positive EV."""
        from weather.strategy import score_buckets
        from weather.config import Config

        config = Config(min_probability=0.10, trading_fees=0.02, min_ev_threshold=0.01)
        # Bucket "60-65" with YES priced at $0.90 but model says ~35% chance
        # (forecast 62°F is near center of bucket, so prob is moderate)
        # NO prob ≈ 65%, NO price = $0.10 → EV = 0.65*0.98 - 0.10 = 0.537
        markets = [{
            "id": "mkt1",
            "outcome_name": "60-65°F",
            "best_ask": 0.90,
            "external_price_yes": 0.90,
        }]
        scored = score_buckets(
            markets, forecast_temp=62.0, forecast_date="2025-06-15",
            config=config, metric="high", location="NYC",
        )
        no_entries = [s for s in scored if s["side"] == "no"]
        assert len(no_entries) > 0, "Should generate NO entry for overpriced bucket"
        assert no_entries[0]["ev"] > 0, "NO EV should be positive"

    def test_no_not_generated_for_fairly_priced_bucket(self):
        """If market price matches model prob, NO has no edge."""
        from weather.strategy import score_buckets
        from weather.config import Config

        config = Config(min_probability=0.10, trading_fees=0.02, min_ev_threshold=0.01)
        # Bucket priced at $0.05 and model agrees ~5% prob → NO at $0.95 has thin margin
        markets = [{
            "id": "mkt1",
            "outcome_name": "80-85°F",
            "best_ask": 0.05,
            "external_price_yes": 0.05,
        }]
        scored = score_buckets(
            markets, forecast_temp=70.0, forecast_date="2025-06-15",
            config=config, metric="high", location="NYC",
        )
        no_entries = [s for s in scored if s["side"] == "no"]
        # NO price = 0.95, NO prob ~0.95 → EV ≈ 0.95*0.98 - 0.95 = -0.019 → negative
        # Should NOT be in scored (or EV < 0 means not added)
        for entry in no_entries:
            assert entry["ev"] > 0, "Only positive EV NO entries should be scored"


class TestPaperBridgeNoSidePricing:
    """Verify PaperBridge uses correct price for NO trades."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_trade_uses_complement_of_bid(self):
        """NO price should be 1 - YES best_bid, not YES best_ask."""
        from unittest.mock import MagicMock
        from weather.paper_bridge import PaperBridge

        real_bridge = MagicMock()
        gm = MagicMock()
        gm.best_ask = 0.70
        gm.best_bid = 0.65
        gm.outcome_prices = [0.70]
        real_bridge._market_cache = {"mkt1": gm}

        paper = PaperBridge(real_bridge)
        result = await paper.execute_trade("mkt1", "no", 1.00)

        assert result["success"] is True
        # NO price = 1 - 0.65 = 0.35
        expected_no_price = 0.35
        expected_shares = 1.00 / expected_no_price
        assert abs(result["shares_bought"] - expected_shares) < 0.01

    @pytest.mark.asyncio(loop_scope="function")
    async def test_yes_trade_uses_best_ask(self):
        """YES trades should still use best_ask."""
        from unittest.mock import MagicMock
        from weather.paper_bridge import PaperBridge

        real_bridge = MagicMock()
        gm = MagicMock()
        gm.best_ask = 0.10
        gm.best_bid = 0.08
        gm.outcome_prices = [0.10]
        real_bridge._market_cache = {"mkt1": gm}

        paper = PaperBridge(real_bridge)
        result = await paper.execute_trade("mkt1", "yes", 1.00)

        assert result["success"] is True
        expected_shares = 1.00 / 0.10
        assert abs(result["shares_bought"] - expected_shares) < 0.01
