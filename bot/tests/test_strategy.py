"""Tests for bot.strategy — weather pipeline exposure budget."""

import unittest
from unittest.mock import MagicMock, patch

from bot.config import Config
from bot.state import TradingState, TradeRecord
from bot.strategy import _run_weather_pipeline


class TestWeatherPipelineExposureBudget(unittest.TestCase):
    """Weather bridge should receive remaining exposure after bot positions."""

    def _make_config(self, max_total_exposure=50.0):
        config = Config()
        config.max_total_exposure = max_total_exposure
        return config

    def _run(self, state, config, mock_bridge_cls):
        """Helper: run _run_weather_pipeline with all heavy deps mocked."""
        with patch("bot.gamma.GammaClient") as mock_gamma_cls, \
             patch("weather.state.TradingState") as mock_ws, \
             patch("weather.state.state_lock"), \
             patch("weather.strategy.run_weather_strategy"):
            mock_gamma_cls.return_value.__enter__ = lambda s: s
            mock_gamma_cls.return_value.__exit__ = lambda s, *a: None
            mock_ws.load.return_value = mock_ws()
            _run_weather_pipeline(
                client=MagicMock(),
                config=config,
                state=state,
                dry_run=True,
                state_path="/tmp/test_state.json",
            )

    @patch("weather.bridge.CLOBWeatherBridge")
    def test_bridge_gets_remaining_exposure_with_buy(self, mock_bridge_cls):
        state = TradingState()
        state.trades = {
            "pos1": TradeRecord(
                market_id="m1", token_id="t1",
                side="BUY", price=0.5, size=20.0,
            ),
        }
        # BUY exposure = 0.5 * 20 = $10
        config = self._make_config(max_total_exposure=50.0)
        self._run(state, config, mock_bridge_cls)

        # Bridge should receive max_exposure = 50 - 10 = 40
        call_kwargs = mock_bridge_cls.call_args.kwargs
        self.assertAlmostEqual(call_kwargs["max_exposure"], 40.0, delta=0.01)

    @patch("weather.bridge.CLOBWeatherBridge")
    def test_bridge_gets_remaining_exposure_with_sell(self, mock_bridge_cls):
        state = TradingState()
        state.trades = {
            "pos1": TradeRecord(
                market_id="m1", token_id="t1",
                side="SELL", price=0.6, size=50.0,
            ),
        }
        # SELL exposure = (1.0 - 0.6) * 50 = $20
        config = self._make_config(max_total_exposure=50.0)
        self._run(state, config, mock_bridge_cls)

        # Bridge should receive max_exposure = 50 - 20 = 30
        call_kwargs = mock_bridge_cls.call_args.kwargs
        self.assertAlmostEqual(call_kwargs["max_exposure"], 30.0, delta=0.01)

    @patch("weather.bridge.CLOBWeatherBridge")
    def test_bridge_gets_full_exposure_with_no_trades(self, mock_bridge_cls):
        state = TradingState()
        config = self._make_config(max_total_exposure=50.0)
        self._run(state, config, mock_bridge_cls)

        # No trades → full budget
        call_kwargs = mock_bridge_cls.call_args.kwargs
        self.assertAlmostEqual(call_kwargs["max_exposure"], 50.0, delta=0.01)

    @patch("weather.bridge.CLOBWeatherBridge")
    def test_pending_trades_excluded_from_exposure(self, mock_bridge_cls):
        state = TradingState()
        state.trades = {
            "active": TradeRecord(
                market_id="m1", token_id="t1",
                side="BUY", price=0.5, size=20.0,
            ),
            "pending_exit": TradeRecord(
                market_id="m2", token_id="t2",
                side="BUY", price=0.8, size=100.0,
                memo="pending_exit",
            ),
            "pending_fill": TradeRecord(
                market_id="m3", token_id="t3",
                side="BUY", price=0.9, size=100.0,
                memo="pending_fill",
            ),
        }
        # Only active trade counts: 0.5 * 20 = $10
        config = self._make_config(max_total_exposure=50.0)
        self._run(state, config, mock_bridge_cls)

        # Only active trade counted: 50 - 10 = 40
        call_kwargs = mock_bridge_cls.call_args.kwargs
        self.assertAlmostEqual(call_kwargs["max_exposure"], 40.0, delta=0.01)

    @patch("weather.bridge.CLOBWeatherBridge")
    def test_exposure_clamped_to_zero(self, mock_bridge_cls):
        state = TradingState()
        state.trades = {
            "pos1": TradeRecord(
                market_id="m1", token_id="t1",
                side="BUY", price=0.5, size=200.0,
            ),
        }
        # BUY exposure = 0.5 * 200 = $100 > max_total_exposure of 50
        config = self._make_config(max_total_exposure=50.0)
        self._run(state, config, mock_bridge_cls)

        # Clamped to zero
        call_kwargs = mock_bridge_cls.call_args.kwargs
        self.assertAlmostEqual(call_kwargs["max_exposure"], 0.0, delta=0.01)

    @patch("weather.config.Config")
    @patch("weather.bridge.CLOBWeatherBridge")
    def test_weather_config_also_gets_remaining_exposure(
        self, mock_bridge_cls, mock_wc,
    ):
        """WeatherConfig.max_exposure should also receive remaining, not full."""
        state = TradingState()
        state.trades = {
            "pos1": TradeRecord(
                market_id="m1", token_id="t1",
                side="BUY", price=0.4, size=25.0,
            ),
        }
        # BUY exposure = 0.4 * 25 = $10
        config = self._make_config(max_total_exposure=50.0)
        self._run(state, config, mock_bridge_cls)

        # WeatherConfig should receive remaining = 50 - 10 = 40
        wc_kwargs = mock_wc.call_args.kwargs
        self.assertAlmostEqual(wc_kwargs["max_exposure"], 40.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
