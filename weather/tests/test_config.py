"""Tests for circuit breaker config parameters."""
import unittest
from weather.config import Config


class TestCircuitBreakerConfig(unittest.TestCase):
    def test_defaults_exist(self):
        c = Config()
        self.assertEqual(c.daily_loss_limit, 10.0)
        self.assertEqual(c.max_positions_per_day, 20)
        self.assertEqual(c.cooldown_hours_after_max_loss, 24.0)
        self.assertEqual(c.max_open_positions, 15)

    def test_overridable(self):
        c = Config(daily_loss_limit=25.0, max_open_positions=30)
        self.assertEqual(c.daily_loss_limit, 25.0)
        self.assertEqual(c.max_open_positions, 30)


def test_new_execution_config_defaults():
    cfg = Config()
    assert cfg.slippage_edge_ratio == 0.8
    assert cfg.depth_fill_ratio == 0.5
    assert cfg.vwap_max_levels == 5
    assert cfg.trade_metrics == "high"
    assert cfg.same_location_discount == 0.5
    assert cfg.same_location_horizon_window == 2
    assert cfg.correlation_threshold == 0.3

def test_active_metrics_single():
    cfg = Config(trade_metrics="high")
    assert cfg.active_metrics == ["high"]

def test_active_metrics_both():
    cfg = Config(trade_metrics="high,low")
    assert cfg.active_metrics == ["high", "low"]

def test_active_metrics_low_only():
    cfg = Config(trade_metrics="low")
    assert cfg.active_metrics == ["low"]


if __name__ == "__main__":
    unittest.main()
