"""Tests for METAR-based resolution fallback in _resolve_predictions()."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta
from weather.paper_trade import _resolve_predictions
from weather.state import TradingState, PredictionRecord


def _make_prediction(market_id: str, location: str, forecast_date: str,
                     metric: str = "high", bucket_low: float = 42.0,
                     bucket_high: float = 43.0, prob: float = 0.5,
                     forecast_temp: float = 42.5) -> PredictionRecord:
    return PredictionRecord(
        market_id=market_id,
        event_id="evt1",
        location=location,
        forecast_date=forecast_date,
        metric=metric,
        our_probability=prob,
        forecast_temp=forecast_temp,
        bucket_low=bucket_low,
        bucket_high=bucket_high,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _make_state(predictions: list[PredictionRecord],
                daily_obs: dict | None = None) -> TradingState:
    state = TradingState()
    for pred in predictions:
        state.predictions[pred.market_id] = pred
    if daily_obs:
        state.daily_observations = daily_obs
    return state


class TestMETARResolution:
    @pytest.mark.asyncio
    async def test_resolves_past_prediction_win(self):
        """Prediction with forecast_date 2 days ago + matching obs -> resolved as WIN."""
        past_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        pred = _make_prediction("0xabc", "NYC", past_date, bucket_low=42.0, bucket_high=43.0)
        state = _make_state([pred], {f"NYC|{past_date}": {"obs_high": 42.5}})

        gamma = MagicMock()
        gamma.check_resolution = AsyncMock(return_value=None)

        count = await _resolve_predictions(state, gamma)
        assert count == 1
        assert pred.resolved is True
        assert pred.actual_outcome is True

    @pytest.mark.asyncio
    async def test_resolves_past_prediction_loss(self):
        """Prediction with actual temp outside bucket -> resolved as LOSS."""
        past_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        pred = _make_prediction("0xdef", "NYC", past_date, bucket_low=42.0, bucket_high=43.0)
        state = _make_state([pred], {f"NYC|{past_date}": {"obs_high": 45.0}})

        gamma = MagicMock()
        gamma.check_resolution = AsyncMock(return_value=None)

        count = await _resolve_predictions(state, gamma)
        assert count == 1
        assert pred.resolved is True
        assert pred.actual_outcome is False

    @pytest.mark.asyncio
    async def test_skips_recent_prediction(self):
        """Prediction with forecast_date today -> not resolved (< 24h)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        pred = _make_prediction("0xghi", "NYC", today, bucket_low=42.0, bucket_high=43.0)
        state = _make_state([pred], {f"NYC|{today}": {"obs_high": 42.5}})

        gamma = MagicMock()
        gamma.check_resolution = AsyncMock(return_value=None)

        count = await _resolve_predictions(state, gamma)
        assert count == 0
        assert pred.resolved is False

    @pytest.mark.asyncio
    async def test_skips_no_obs(self):
        """Prediction with no daily observations -> not resolved."""
        past_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        pred = _make_prediction("0xjkl", "NYC", past_date, bucket_low=42.0, bucket_high=43.0)
        state = _make_state([pred])  # no daily_obs

        gamma = MagicMock()
        gamma.check_resolution = AsyncMock(return_value=None)

        count = await _resolve_predictions(state, gamma)
        assert count == 0
        assert pred.resolved is False

    @pytest.mark.asyncio
    async def test_gamma_takes_priority(self):
        """If Gamma resolves first, METAR fallback doesn't double-resolve."""
        past_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        pred = _make_prediction("0xmno", "NYC", past_date, bucket_low=42.0, bucket_high=43.0)
        state = _make_state([pred], {f"NYC|{past_date}": {"obs_high": 42.5}})

        gamma = MagicMock()
        gamma.check_resolution = AsyncMock(return_value={"resolved": True, "outcome": False})

        count = await _resolve_predictions(state, gamma)
        assert count == 1
        assert pred.resolved is True
        assert pred.actual_outcome is False  # Gamma's outcome, not METAR

    @pytest.mark.asyncio
    async def test_open_bottom_bucket(self):
        """Open-bottom bucket (-999, 41): actual=39 -> in bucket -> WIN."""
        past_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        pred = _make_prediction("0xpqr", "NYC", past_date, bucket_low=-999.0, bucket_high=41.0)
        state = _make_state([pred], {f"NYC|{past_date}": {"obs_high": 39.0}})

        gamma = MagicMock()
        gamma.check_resolution = AsyncMock(return_value=None)

        count = await _resolve_predictions(state, gamma)
        assert count == 1
        assert pred.actual_outcome is True

    @pytest.mark.asyncio
    async def test_open_top_bucket(self):
        """Open-top bucket (46, 999): actual=48 -> in bucket -> WIN."""
        past_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        pred = _make_prediction("0xstu", "NYC", past_date, bucket_low=46.0, bucket_high=999.0)
        state = _make_state([pred], {f"NYC|{past_date}": {"obs_high": 48.0}})

        gamma = MagicMock()
        gamma.check_resolution = AsyncMock(return_value=None)

        count = await _resolve_predictions(state, gamma)
        assert count == 1
        assert pred.actual_outcome is True

    @pytest.mark.asyncio
    async def test_low_metric(self):
        """Uses obs_low for metric='low'."""
        past_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        pred = _make_prediction("0xvwx", "NYC", past_date, metric="low",
                                bucket_low=30.0, bucket_high=31.0)
        state = _make_state([pred], {f"NYC|{past_date}": {"obs_low": 30.5}})

        gamma = MagicMock()
        gamma.check_resolution = AsyncMock(return_value=None)

        count = await _resolve_predictions(state, gamma)
        assert count == 1
        assert pred.actual_outcome is True
