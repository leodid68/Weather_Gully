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


if __name__ == "__main__":
    unittest.main()
