"""Integration tests for execution, low temp, and portfolio features."""
import json
from weather.config import Config
from weather.sizing import compute_position_size
from weather.bridge import compute_available_depth, compute_vwap


class TestConfigIntegration:
    def test_load_new_fields_from_json(self, tmp_path):
        cfg_data = {
            "slippage_edge_ratio": 0.6,
            "depth_fill_ratio": 0.4,
            "vwap_max_levels": 3,
            "trade_metrics": "high,low",
            "same_location_discount": 0.7,
            "same_location_horizon_window": 3,
            "correlation_threshold": 0.25,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg_data))
        cfg = Config.load(str(tmp_path))
        assert cfg.slippage_edge_ratio == 0.6
        assert cfg.depth_fill_ratio == 0.4
        assert cfg.vwap_max_levels == 3
        assert cfg.active_metrics == ["high", "low"]
        assert cfg.same_location_discount == 0.7
        assert cfg.correlation_threshold == 0.25

    def test_defaults_backward_compatible(self):
        cfg = Config()
        assert cfg.active_metrics == ["high"]
        assert cfg.correlation_threshold == 0.3
        assert cfg.slippage_edge_ratio == 0.8


class TestBudgetSizingIntegration:
    def test_exposure_reduces_position(self):
        """Full Kelly pipeline with exposure."""
        s1 = compute_position_size(0.55, 0.40, 50.0, 10.0, kelly_frac=0.25, current_exposure=0.0)
        s2 = compute_position_size(0.55, 0.40, 50.0, 10.0, kelly_frac=0.25, current_exposure=40.0)
        assert s2 < s1
        assert s2 >= 0

    def test_full_exposure_blocks_trading(self):
        s = compute_position_size(0.55, 0.40, 50.0, 10.0, kelly_frac=0.25, current_exposure=50.0)
        assert s == 0.0

    def test_no_exposure_unchanged(self):
        s1 = compute_position_size(0.55, 0.40, 50.0, 10.0, kelly_frac=0.25)
        s2 = compute_position_size(0.55, 0.40, 50.0, 10.0, kelly_frac=0.25, current_exposure=0.0)
        assert s1 == s2


class TestVwapDepthIntegration:
    def test_realistic_orderbook(self):
        """VWAP on a realistic thin orderbook."""
        asks = [
            {"price": "0.35", "size": "50"},
            {"price": "0.38", "size": "30"},
            {"price": "0.42", "size": "100"},
            {"price": "0.45", "size": "200"},
            {"price": "0.50", "size": "500"},
        ]
        depth = compute_available_depth(asks, max_levels=5)
        assert depth > 0

        # VWAP for $5 order — fills mostly from first level
        vwap_small = compute_vwap(asks, 5.0)
        assert 0.35 <= vwap_small <= 0.38

        # VWAP for $30 order — spans multiple levels
        vwap_large = compute_vwap(asks, 30.0)
        assert vwap_large >= vwap_small  # Larger order gets worse average price

    def test_depth_calculation(self):
        asks = [
            {"price": "0.50", "size": "100"},
            {"price": "0.60", "size": "200"},
        ]
        depth = compute_available_depth(asks, max_levels=5)
        expected = 100 * 0.5 + 200 * 0.6
        assert abs(depth - expected) < 0.01


class TestAdaptiveSlippageIntegration:
    def test_high_edge_tolerates_slippage(self):
        from weather.strategy import check_context_safeguards
        config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
        # 25% edge → 12.5% adaptive threshold
        context = {
            "slippage": {"estimates": [{"slippage_pct": 0.10}]},
            "edge": {"user_edge": 0.25},
        }
        ok, _ = check_context_safeguards(context, config)
        assert ok

    def test_low_edge_blocks_slippage(self):
        from weather.strategy import check_context_safeguards
        config = Config(slippage_max_pct=0.15, slippage_edge_ratio=0.5)
        # 5% edge → 2.5% adaptive threshold
        context = {
            "slippage": {"estimates": [{"slippage_pct": 0.10}]},
            "edge": {"user_edge": 0.05},
        }
        ok, _ = check_context_safeguards(context, config)
        assert not ok


class TestCorrelationIntegration:
    def test_cumulative_correlation(self):
        from weather.strategy import _apply_correlation_discount
        config = Config(correlation_threshold=0.3, correlation_discount=0.5)
        size = _apply_correlation_discount(2.0, "Dallas", 10, ["Seattle"], config)
        assert size < 2.0

    def test_no_discount_below_threshold(self):
        from weather.strategy import _apply_correlation_discount
        config = Config(correlation_threshold=0.3, correlation_discount=0.5)
        size = _apply_correlation_discount(2.0, "Atlanta", 1, ["NYC"], config)
        assert size == 2.0
