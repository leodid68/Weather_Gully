"""Main strategy loop — scores buckets, sizes positions, manages exits.

Uses CLOBWeatherBridge for all market data and order execution (CLOB direct).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .aviation import get_aviation_daily_data
from .config import Config, LOCATIONS, MIN_SHARES_PER_ORDER, MIN_TICK_SIZE
from .ensemble import fetch_ensemble_spread
from .noaa import get_noaa_forecast
from .open_meteo import compute_ensemble_forecast, get_open_meteo_forecast, get_open_meteo_forecast_multi
from .parsing import parse_weather_event, parse_temperature_bucket
from .probability import (
    compute_adaptive_sigma,
    estimate_bucket_probability,
    estimate_bucket_probability_with_obs,
    get_horizon_days,
    get_noaa_probability,
    platt_calibrate,
)
from .bridge import CLOBWeatherBridge
from .feedback import FeedbackState
from .sigma_log import log_sigma_signals
from .sizing import compute_exit_threshold, compute_position_size
from .trade_log import log_trade
from .state import PredictionRecord, TradingState

logger = logging.getLogger(__name__)

# Source tag for filtering positions
_TRADE_SOURCE = "clob:weather"


# --------------------------------------------------------------------------
# Safeguards (ported from monolith, now uses logging)
# --------------------------------------------------------------------------

def check_context_safeguards(
    context: dict | None,
    config: Config,
    use_edge: bool = True,
) -> tuple[bool, list[str]]:
    """Check market context for deal-breakers. Returns ``(should_trade, reasons)``."""
    if not context:
        return True, []

    reasons: list[str] = []
    market = context.get("market", {})
    warnings = context.get("warnings", [])
    discipline = context.get("discipline", {})
    slippage = context.get("slippage", {})
    edge = context.get("edge", {})

    for warning in warnings:
        if "MARKET RESOLVED" in str(warning).upper():
            return False, ["Market already resolved"]

    warning_level = discipline.get("warning_level", "none")
    if warning_level == "severe":
        return False, [f"Severe flip-flop warning: {discipline.get('flip_flop_warning', '')}"]
    elif warning_level == "mild":
        reasons.append("Mild flip-flop warning (proceed with caution)")

    # Time to resolution
    time_str = market.get("time_to_resolution", "")
    if time_str:
        hours = _parse_time_to_hours(time_str)
        if hours is not None and hours < config.time_to_resolution_min_hours:
            return False, [f"Resolves in {hours}h — too soon"]

    # Slippage
    estimates = slippage.get("estimates", []) if slippage else []
    if estimates:
        slippage_pct = estimates[0].get("slippage_pct", 0)
        if slippage_pct > config.slippage_max_pct:
            return False, [f"Slippage too high: {slippage_pct:.1%}"]

    # Edge recommendation
    if use_edge and edge:
        recommendation = edge.get("recommendation")
        user_edge = edge.get("user_edge")
        threshold = edge.get("suggested_threshold", 0)

        if recommendation == "SKIP":
            return False, ["Edge analysis: SKIP"]
        elif recommendation == "HOLD":
            if user_edge is not None and threshold:
                reasons.append(f"Edge {user_edge:.1%} below threshold {threshold:.1%}")
            else:
                reasons.append("Edge analysis recommends HOLD")
        elif recommendation == "TRADE":
            reasons.append(f"Edge {user_edge:.1%} >= threshold {threshold:.1%}")

    return True, reasons


def check_circuit_breaker(state: "TradingState", config: Config) -> tuple[bool, str]:
    """Check global circuit breaker conditions.

    Returns (blocked: bool, reason: str). If blocked, all trading should halt.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Daily loss limit
    daily_pnl = state.get_daily_pnl(today)
    if daily_pnl <= -config.daily_loss_limit:
        return True, f"Daily loss limit hit: ${daily_pnl:.2f} (limit -${config.daily_loss_limit:.2f})"

    # 2. Max positions per day
    positions_today = state.positions_opened_today(today)
    if positions_today >= config.max_positions_per_day:
        return True, f"Max positions per day: {positions_today} (limit {config.max_positions_per_day})"

    # 3. Cooldown after last circuit break
    if state.last_circuit_break:
        try:
            last_break = datetime.fromisoformat(state.last_circuit_break)
            hours_since = (datetime.now(timezone.utc) - last_break).total_seconds() / 3600
            if hours_since < config.cooldown_hours_after_max_loss:
                return True, f"Cooldown active: {hours_since:.1f}h since circuit break (need {config.cooldown_hours_after_max_loss}h)"
        except (ValueError, TypeError):
            pass

    # 4. Max open positions
    if len(state.trades) >= config.max_open_positions:
        return True, f"Max open positions: {len(state.trades)} (limit {config.max_open_positions})"

    return False, ""


def _parse_time_to_hours(time_str: str) -> float | None:
    """Parse '3d 5h' style string to total hours."""
    try:
        hours = 0.0
        if "d" in time_str:
            days = int(time_str.split("d")[0].strip())
            hours += days * 24
        if "h" in time_str:
            h_part = time_str.split("h")[0]
            if "d" in h_part:
                h_part = h_part.split("d")[-1].strip()
            hours += int(h_part)
        return hours
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------
# Bucket scoring (adjacent buckets)
# --------------------------------------------------------------------------

def score_buckets(
    event_markets: list[dict],
    forecast_temp: float,
    forecast_date: str,
    config: Config,
    obs_data: dict | None = None,
    metric: str = "high",
    location: str = "",
    weather_data: dict | None = None,
    sigma_override: float | None = None,
) -> list[dict]:
    """Score all buckets in an event by expected value.

    Args:
        event_markets: Markets in the event.
        forecast_temp: Ensemble forecast temperature.
        forecast_date: Target date.
        config: Strategy configuration.
        obs_data: Optional METAR observation data for the date
            (keys: ``obs_high``, ``obs_low``, ``latest_obs_time``, ``obs_count``).
        metric: ``"high"`` or ``"low"`` — which extreme this event tracks.
        location: Canonical location key (for timezone-aware sigma).
        weather_data: Optional auxiliary weather data (cloud/wind/precip).

    Returns a list of ``{"market": dict, "bucket": tuple, "prob": float,
    "price": float, "ev": float}`` sorted by EV descending.
    """
    scored: list[dict] = []

    for market in event_markets:
        outcome_name = market.get("outcome_name", "")
        bucket = parse_temperature_bucket(outcome_name)
        if not bucket:
            continue

        # Use best_ask for pricing when available (more accurate than mid-price)
        best_ask = market.get("best_ask")
        price = best_ask if best_ask and best_ask > 0 else (market.get("external_price_yes") or 0.5)

        # Skip extreme prices
        if price < MIN_TICK_SIZE or price > (1 - MIN_TICK_SIZE):
            continue

        if obs_data and obs_data.get("obs_count", 0) > 0:
            loc_data = LOCATIONS.get(location, {})
            station_lon = loc_data.get("lon", -74.0)
            station_tz = loc_data.get("tz", "")
            prob = estimate_bucket_probability_with_obs(
                forecast_temp, bucket[0], bucket[1], forecast_date,
                obs_data=obs_data,
                metric=metric,
                apply_seasonal=config.seasonal_adjustments,
                station_lon=station_lon,
                station_tz=station_tz,
                location=location,
                weather_data=weather_data,
                sigma_override=sigma_override,
            )
        else:
            prob = estimate_bucket_probability(
                forecast_temp, bucket[0], bucket[1], forecast_date,
                apply_seasonal=config.seasonal_adjustments,
                location=location,
                weather_data=weather_data,
                metric=metric,
                sigma_override=sigma_override,
            )

        prob_raw = prob
        prob = platt_calibrate(prob)
        # EV adjusted for trading fees (Polymarket ~2%)
        ev = prob * (1.0 - config.trading_fees) - price

        # Skip low-probability buckets (value traps at $0.01-0.03)
        if prob < config.min_probability:
            continue

        scored.append({
            "market": market,
            "bucket": bucket,
            "outcome_name": outcome_name,
            "prob": prob,
            "prob_raw": prob_raw,
            "price": price,
            "ev": ev,
            "side": "yes",
        })

        # ---- NO side: if bucket is overpriced, buying NO is profitable ----
        no_prob = 1.0 - prob    # P(not in this bucket)
        no_price = 1.0 - price  # Price of NO token (complement)
        if no_price >= MIN_TICK_SIZE and no_prob >= config.min_probability:
            ev_no = no_prob * (1.0 - config.trading_fees) - no_price
            if ev_no > 0:
                scored.append({
                    "market": market,
                    "bucket": bucket,
                    "outcome_name": outcome_name,
                    "prob": no_prob,
                    "prob_raw": 1.0 - prob_raw,
                    "price": no_price,
                    "ev": ev_no,
                    "side": "no",
                })

    scored.sort(key=lambda x: x["ev"], reverse=True)
    return scored


# --------------------------------------------------------------------------
# Exit strategy
# --------------------------------------------------------------------------

def check_exit_opportunities(
    client: CLOBWeatherBridge,
    config: Config,
    state: TradingState,
    dry_run: bool = True,
    use_safeguards: bool = True,
) -> tuple[int, int]:
    """Check open positions for exit opportunities. Returns ``(found, executed)``.

    Uses state-tracked trades instead of API positions (bridge.get_positions()
    returns [] since weather positions are tracked in state).
    """
    if not state.trades:
        return 0, 0

    logger.info("Checking %d weather positions for exit...", len(state.trades))
    exits_found = 0
    exits_executed = 0

    for market_id, trade in list(state.trades.items()):
        shares = trade.shares
        if shares < MIN_SHARES_PER_ORDER:
            continue

        # Fetch current price from orderbook
        gm = client._market_cache.get(market_id)
        if not gm or not gm.clob_token_ids:
            continue
        token_id = gm.clob_token_ids[0]
        try:
            book = client.clob.get_orderbook(token_id)
            bids = book.get("bids") or []
            current_price = float(bids[0]["price"]) if bids else 0.0
        except Exception:
            continue

        if current_price <= 0:
            continue

        # Dynamic exit threshold
        cost_basis = trade.cost_basis or config.exit_threshold
        if config.dynamic_exits:
            # Estimate hours to resolution from end_date in cached market
            hours = 168.0
            if gm.end_date:
                try:
                    end_dt = datetime.fromisoformat(gm.end_date.replace("Z", "+00:00"))
                    delta = end_dt - datetime.now(timezone.utc)
                    hours = max(0, delta.total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass
            exit_threshold = compute_exit_threshold(cost_basis, hours)
        else:
            exit_threshold = config.exit_threshold

        outcome_name = trade.outcome_name[:50]

        if current_price >= exit_threshold:
            exits_found += 1
            logger.info(
                "EXIT: %s — price $%.2f >= threshold $%.2f",
                outcome_name, current_price, exit_threshold,
            )

            if use_safeguards:
                context = client.get_market_context(market_id)
                should_trade, reasons = check_context_safeguards(context, config)
                if not should_trade:
                    logger.info("Exit skipped: %s", "; ".join(reasons))
                    continue
                if reasons:
                    logger.info("Exit warnings: %s", "; ".join(reasons))

            if dry_run:
                logger.info("[DRY RUN] Would sell %.1f shares of %s", shares, outcome_name)
            else:
                logger.info("Selling %.1f shares of %s...", shares, outcome_name)
                result = client.execute_sell(market_id, shares)

                if result.get("success"):
                    exits_executed += 1
                    logger.info("Sold %.1f shares @ $%.2f", shares, current_price)
                    state.remove_trade(market_id)
                    for eid, mid in list(state.event_positions.items()):
                        if mid == market_id:
                            state.remove_event_position(eid)
                            break
                else:
                    logger.error("Sell failed: %s", result.get("error", "Unknown"))
        else:
            logger.debug(
                "HOLD: %s — price $%.2f < threshold $%.2f",
                outcome_name, current_price, exit_threshold,
            )

    # Warn about trades with no cached market data (may need manual review)
    unchecked = [mid for mid in state.trades if mid not in client._market_cache]
    if unchecked:
        logger.warning(
            "Exit check: %d trade(s) have no cached market data — may need manual review: %s",
            len(unchecked), ", ".join(mid[:16] for mid in unchecked),
        )

    return exits_found, exits_executed


# --------------------------------------------------------------------------
# Edge inversion exit
# --------------------------------------------------------------------------

def should_exit_on_edge_inversion(
    our_prob: float,
    market_price: float,
    cost_basis: float,
    side: str = "yes",
    min_loss_to_trigger: float = 0.02,
) -> bool:
    """Check if the edge has inverted, suggesting we should exit.

    For YES positions: exit if market values it higher than our model AND we can sell near cost.
    For NO positions: exit if model now favors YES side AND we can sell near cost.
    """
    if side == "yes":
        # We bought YES. Exit if market price > our probability AND we can sell without big loss
        return market_price > our_prob and market_price >= cost_basis - min_loss_to_trigger
    else:
        # We bought NO at (1-market_price). Exit if our model now favors the YES side
        no_market_price = 1.0 - market_price  # current NO price
        return our_prob > (1 - market_price) and no_market_price >= cost_basis - min_loss_to_trigger


# --------------------------------------------------------------------------
# Stop-loss on forecast reversal
# --------------------------------------------------------------------------

def _check_stop_loss_reversals(
    client: CLOBWeatherBridge,
    config: Config,
    state: TradingState,
    noaa_cache: dict[str, dict],
    open_meteo_cache: dict[str, dict],
    dry_run: bool,
    aviation_cache: dict[str, dict] | None = None,
) -> int:
    """Exit positions where the forecast has shifted away from our bucket.

    When aviation observations are available, uses the observed temperature
    instead of the forecast for more reliable stop-loss decisions.

    Returns number of stop-loss exits executed.
    """
    exits = 0
    for market_id, trade in list(state.trades.items()):
        if not trade.forecast_temp or not trade.forecast_date or not trade.location:
            continue

        bucket = parse_temperature_bucket(trade.outcome_name)
        if not bucket:
            continue

        # Get current forecast for this location/date
        noaa_day = noaa_cache.get(trade.location, {}).get(trade.forecast_date, {})
        metric = trade.metric  # Use the metric stored at trade time

        # Prefer observed temperature over forecast when available,
        # but only if the observation is late enough in the day that the
        # extreme is meaningful (avoid premature stop-loss in the morning).
        obs_data = None
        if aviation_cache:
            obs_data = aviation_cache.get(trade.location, {}).get(trade.forecast_date)

        if obs_data:
            obs_key = f"obs_{metric}"
            obs_temp = obs_data.get(obs_key)
            latest_time = obs_data.get("latest_obs_time", "")

            # Only trust obs for stop-loss when the daily extreme is likely
            # to have been reached: after ~14:00 local for highs, ~08:00 for lows.
            loc_data = LOCATIONS.get(trade.location, {})
            tz_name = loc_data.get("tz", "America/New_York")

            obs_usable = False
            if obs_temp is not None and latest_time:
                try:
                    from datetime import datetime as _dt
                    from zoneinfo import ZoneInfo
                    utc_dt = _dt.fromisoformat(latest_time.replace("Z", "+00:00"))
                    local_dt = utc_dt.astimezone(ZoneInfo(tz_name))
                    local_hour = local_dt.hour
                    # For highs: trust after 14:00 local (peak likely passed)
                    # For lows: trust after 08:00 local (morning low passed)
                    if metric == "high" and local_hour >= 14:
                        obs_usable = True
                    elif metric == "low" and local_hour >= 8:
                        obs_usable = True
                except (ValueError, IndexError, KeyError):
                    pass

            if obs_usable:
                current_temp = obs_temp
                logger.debug("Stop-loss using observed %s=%.1f°F for %s",
                             metric, current_temp, trade.location)
            else:
                current_temp = None
        else:
            current_temp = None

        # Fall back to forecast if no observation
        if current_temp is None:
            noaa_temp = noaa_day.get(metric)
            om_data = open_meteo_cache.get(trade.location, {}).get(trade.forecast_date)

            if config.multi_source and (noaa_temp is not None or om_data):
                current_temp, _ = compute_ensemble_forecast(noaa_temp, om_data, metric)
            else:
                current_temp = noaa_temp

        if current_temp is None:
            continue

        # How far has the forecast moved from our original entry forecast?
        shift = abs(current_temp - trade.forecast_temp)

        # Check if current forecast is now outside our bucket AND shifted significantly
        if shift >= config.stop_loss_reversal_threshold and not (bucket[0] <= current_temp <= bucket[1]):
            logger.warning(
                "STOP-LOSS: %s forecast shifted %.1f°F (was %.0f°F, now %.0f°F) — bucket %s no longer viable",
                trade.location, shift, trade.forecast_temp, current_temp, trade.outcome_name,
            )

            # Attempt to sell — use state-based lookup (bridge returns None)
            fresh_trade = state.trades.get(market_id)
            fresh_shares = fresh_trade.shares if fresh_trade else 0.0

            if fresh_shares < MIN_SHARES_PER_ORDER:
                logger.info("Position already closed for %s", market_id)
                state.remove_trade(market_id)
                continue

            if dry_run:
                logger.info("[DRY RUN] Would stop-loss sell %.1f shares of %s", fresh_shares, fresh_trade.outcome_name)
            else:
                logger.info("Stop-loss selling %.1f shares of %s...", fresh_shares, fresh_trade.outcome_name)
                result = client.execute_sell(market_id, fresh_shares)
                if result.get("success"):
                    exits += 1
                    state.remove_trade(market_id)
                    # Clean up correlation guard
                    for eid, mid in list(state.event_positions.items()):
                        if mid == market_id:
                            state.remove_event_position(eid)
                            break
                    logger.info("Stop-loss executed for %s", fresh_trade.outcome_name)
                else:
                    logger.error("Stop-loss sell failed: %s", result.get("error", "Unknown"))

    return exits


# --------------------------------------------------------------------------
# Main strategy
# --------------------------------------------------------------------------

def run_weather_strategy(
    client: CLOBWeatherBridge,
    config: Config,
    state: TradingState,
    dry_run: bool = True,
    positions_only: bool = False,
    show_config: bool = False,
    use_safeguards: bool = True,
    state_path: str | None = None,
) -> None:
    """Run the weather trading strategy."""
    feedback = FeedbackState.load()

    logger.info("Weather Trading Bot (CLOB direct)")
    logger.info("=" * 50)

    if dry_run:
        logger.info("[DRY RUN] No trades will be executed. Use --live to enable trading.")

    logger.info("Config: entry=%.0f%% exit=%.0f%% max=$%.2f kelly=%.0f%% adjacent=%s dynamic_exits=%s",
                config.entry_threshold * 100, config.exit_threshold * 100,
                config.max_position_usd, config.kelly_fraction * 100,
                config.adjacent_buckets, config.dynamic_exits)
    logger.info("Locations: %s", ", ".join(config.active_locations))

    if show_config:
        from dataclasses import fields as dc_fields
        for f in dc_fields(config):
            if f.name == "private_key":
                continue
            logger.info("  %s = %s", f.name, getattr(config, f.name))
        return

    if positions_only:
        logger.info("Current Positions (from state):")
        if not state.trades:
            logger.info("  No open positions")
        else:
            for market_id, trade in state.trades.items():
                logger.info(
                    "  %s — shares: %.1f | cost: $%.2f",
                    trade.outcome_name[:50],
                    trade.shares,
                    trade.cost_basis,
                )
        return

    # Circuit breaker check
    blocked, reason = check_circuit_breaker(state, config)
    if blocked:
        logger.warning("CIRCUIT BREAKER: %s", reason)
        state.last_circuit_break = datetime.now(timezone.utc).isoformat()
        save_path = state_path or config.state_file
        state.save(save_path)
        return

    # Sync bridge exposure from persisted state
    client.sync_exposure_from_state(state.trades)

    # Parallel fetch: portfolio + markets
    logger.info("Fetching portfolio and markets in parallel...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_portfolio = pool.submit(client.get_portfolio)
        fut_markets = pool.submit(client.fetch_weather_markets)
        portfolio = fut_portfolio.result()
        markets = fut_markets.result()

    balance = portfolio.get("balance_usdc", 0) if portfolio else 0
    if portfolio:
        logger.info("Portfolio: balance=$%.2f exposure=$%.2f positions=%d",
                     balance, portfolio.get("total_exposure", 0),
                     portfolio.get("positions_count", 0))

    logger.info("Found %d weather markets", len(markets))

    if not markets:
        logger.info("No weather markets available")
        return

    # Group by event
    events: dict[str, list[dict]] = {}
    for market in markets:
        event_id = market.get("event_id") or market.get("event_name", "unknown")
        events.setdefault(event_id, []).append(market)

    logger.info("Grouped into %d events", len(events))

    noaa_cache: dict[str, dict] = {}  # location → {date: {high, low}}
    open_meteo_cache: dict[str, dict] = {}  # location → {date: {gfs_high, ecmwf_high, ...}}
    aviation_cache: dict[str, dict] = {}  # location → {date: {obs_high, obs_low, ...}}
    trades_executed = 0
    opportunities_found = 0

    # Parallel pre-fetch: NOAA + Open-Meteo + Aviation for all active locations
    active_locs = config.active_locations
    sources = ["NOAA"]
    if config.multi_source:
        sources.append("Open-Meteo")
    if config.aviation_obs:
        sources.append("METAR")
    logger.info("Pre-fetching forecasts for %d locations (%s) in parallel...",
                len(active_locs), " + ".join(sources))

    def _fetch_noaa(loc_name: str) -> tuple[str, dict]:
        return loc_name, get_noaa_forecast(
            loc_name, LOCATIONS,
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
        )

    def _fetch_aviation() -> dict[str, dict]:
        return get_aviation_daily_data(
            active_locs,
            hours=config.aviation_hours,
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
        )

    def _fetch_open_meteo_all() -> dict[str, dict[str, dict]]:
        loc_subset = {k: v for k, v in LOCATIONS.items() if k in active_locs}
        return get_open_meteo_forecast_multi(
            loc_subset,
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
        )

    max_workers = len(active_locs) + 2  # NOAA per location + 1 Open-Meteo + 1 aviation
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Submit all NOAA fetches (still per-location)
        noaa_futures = {pool.submit(_fetch_noaa, loc): loc for loc in active_locs}
        # Submit single Open-Meteo multi-location fetch
        om_future = None
        if config.multi_source:
            om_future = pool.submit(_fetch_open_meteo_all)
        # Submit aviation fetch (single call for all stations)
        aviation_future = None
        if config.aviation_obs:
            aviation_future = pool.submit(_fetch_aviation)

        for fut in as_completed(list(noaa_futures)):
            loc = noaa_futures.get(fut, "?")
            try:
                loc_name, data = fut.result()
                noaa_cache[loc_name] = data
                if not data:
                    logger.warning("NOAA returned empty forecast for %s", loc_name)
            except Exception as exc:
                logger.error("NOAA fetch crashed for %s: %s", loc, exc)

        # Collect Open-Meteo results (single batch)
        if om_future is not None:
            try:
                open_meteo_cache = om_future.result()
            except Exception as exc:
                logger.error("Open-Meteo multi-location fetch crashed: %s", exc)

        # Collect aviation results
        if aviation_future is not None:
            try:
                aviation_cache = aviation_future.result()
                if aviation_cache:
                    for loc, dates in aviation_cache.items():
                        for date_str, obs in dates.items():
                            state.update_daily_obs(loc, date_str, obs)
                else:
                    logger.warning("Aviation API returned no observations")
            except Exception as exc:
                logger.error("Aviation fetch crashed: %s", exc)

    logger.info("Forecasts ready: NOAA=%d, Open-Meteo=%d, METAR=%d",
                len(noaa_cache), len(open_meteo_cache), len(aviation_cache))

    # Health check: at least one weather source must have data
    noaa_ok = sum(1 for v in noaa_cache.values() if v)
    om_ok = sum(1 for v in open_meteo_cache.values() if v)
    if noaa_ok == 0 and om_ok == 0:
        logger.error("HEALTH CHECK FAILED: no weather source returned data — skipping trading")
        save_path = state_path or config.state_file
        state.save(save_path)
        return
    if noaa_ok == 0 or (config.multi_source and om_ok == 0):
        missing = "NOAA" if noaa_ok == 0 else "Open-Meteo"
        logger.warning("HEALTH CHECK: %s returned no data — trading with reduced sources", missing)

    for event_id, event_markets in events.items():
        event_name = event_markets[0].get("event_name", "") if event_markets else ""
        event_info = parse_weather_event(event_name)

        if not event_info:
            continue

        location = event_info["location"]
        date_str = event_info["date"]
        metric = event_info["metric"]

        if location not in config.active_locations:
            continue

        # Check max forecast horizon
        days_ahead = get_horizon_days(date_str)
        if days_ahead > config.max_days_ahead:
            logger.debug("Skipping %s %s — %d days ahead (max %d)",
                         location, date_str, days_ahead, config.max_days_ahead)
            continue

        # Correlation guard: max 1 position per event
        if config.correlation_guard and state.has_event_position(event_id):
            existing_mid = state.event_positions.get(event_id, "")
            logger.info("Correlation guard: already hold position in event %s (market %s) — skip",
                         event_id, existing_mid)
            continue

        logger.info("%s %s (%s temp)", location, date_str, metric)

        forecasts = noaa_cache.get(location, {})
        day_forecast = forecasts.get(date_str, {})
        noaa_temp = day_forecast.get(metric)

        # Aviation observations for this location/date
        obs_data = aviation_cache.get(location, {}).get(date_str)
        aviation_obs_temp = None
        effective_aviation_weight = 0.0

        if obs_data and config.aviation_obs:
            obs_key = f"obs_{metric}"
            aviation_obs_temp = obs_data.get(obs_key)
            if aviation_obs_temp is not None:
                # Weight: full for day-J (today), full for J+1, 0 for J+2+
                if days_ahead == 0:
                    effective_aviation_weight = config.aviation_obs_weight
                elif days_ahead == 1:
                    effective_aviation_weight = config.aviation_obs_weight
                else:
                    effective_aviation_weight = 0.0
                    aviation_obs_temp = None  # Don't use obs for distant dates

                if aviation_obs_temp is not None:
                    logger.info("METAR observed %s: %.1f°F (%d obs, latest %s)",
                                metric, aviation_obs_temp,
                                obs_data.get("obs_count", 0),
                                obs_data.get("latest_obs_time", "?"))

        # Multi-source ensemble forecast (now with optional aviation obs)
        om_data = open_meteo_cache.get(location, {}).get(date_str)
        if config.multi_source and (noaa_temp is not None or om_data or aviation_obs_temp is not None):
            ensemble_temp, model_spread = compute_ensemble_forecast(
                noaa_temp, om_data, metric,
                aviation_obs_temp=aviation_obs_temp,
                aviation_obs_weight=effective_aviation_weight,
                location=location,
            )
            if ensemble_temp is not None:
                obs_str = f", obs={aviation_obs_temp:.0f}°F" if aviation_obs_temp is not None else ""
                logger.info("Ensemble forecast: %.1f°F (NOAA=%s, spread=%.1f°F%s)",
                             ensemble_temp,
                             f"{noaa_temp}°F" if noaa_temp is not None else "N/A",
                             model_spread, obs_str)
                forecast_temp = ensemble_temp
            else:
                forecast_temp = noaa_temp
                if forecast_temp is None:
                    continue
        elif noaa_temp is not None:
            forecast_temp = noaa_temp
        elif om_data:
            # Fallback: NOAA failed but Open-Meteo available
            fallback_temp, _ = compute_ensemble_forecast(None, om_data, metric, location=location)
            if fallback_temp is not None:
                logger.warning("NOAA unavailable for %s %s — falling back to Open-Meteo (%.1f°F)",
                               location, date_str, fallback_temp)
            forecast_temp = fallback_temp
        else:
            forecast_temp = None

        if forecast_temp is None:
            logger.warning("No forecast available for %s %s", location, date_str)
            continue

        # Adaptive sigma: compute from ensemble spread + model spread + feedback EMA
        adaptive_sigma_value = None
        if config.adaptive_sigma:
            loc_data = LOCATIONS.get(location, {})
            ensemble_result = fetch_ensemble_spread(
                loc_data.get("lat", 0), loc_data.get("lon", 0),
                date_str, metric,
            )
            forecast_month = int(date_str.split("-")[1])
            ema_error = feedback.get_abs_error_ema(location, forecast_month)
            adaptive_sigma_value = compute_adaptive_sigma(
                ensemble_result, model_spread if config.multi_source else 0.0,
                ema_error, date_str, location,
            )
            log_sigma_signals(
                location=location,
                date=date_str,
                metric=metric,
                ensemble_stddev=ensemble_result.ensemble_stddev,
                model_spread=model_spread if config.multi_source else 0.0,
                ema_error=ema_error,
                final_sigma=adaptive_sigma_value,
                forecast_temp=forecast_temp,
            )

        # Feedback bias correction (before delta detection so both use corrected temp)
        forecast_month = int(date_str.split("-")[1])
        feedback_bias = feedback.get_bias(location, forecast_month)
        if feedback_bias is not None:
            forecast_temp -= feedback_bias
            logger.info("Feedback bias correction: %+.1f°F → adjusted forecast %.0f°F",
                         -feedback_bias, forecast_temp)

        # Forecast change detection (operates on bias-corrected forecast)
        delta = state.get_forecast_delta(location, date_str, metric, forecast_temp)
        state.store_forecast(location, date_str, metric, forecast_temp)
        if delta is not None:
            abs_delta = abs(delta)
            if abs_delta >= config.forecast_change_threshold:
                logger.info("Forecast shifted %+.1f°F since last run — re-evaluating", delta)
            else:
                logger.debug("Forecast delta: %+.1f°F (below threshold %.1f°F)",
                             delta, config.forecast_change_threshold)

        logger.info("Forecast: %.0f°F", forecast_temp)

        # Dynamic NOAA probability
        noaa_prob = get_noaa_probability(date_str, apply_seasonal=config.seasonal_adjustments)
        logger.info("NOAA probability: %.1f%% (horizon %d days)", noaa_prob * 100, days_ahead)

        # Score all buckets (adjacent scoring) or just the matching one
        # When multi_source is enabled, aviation obs is already baked into
        # forecast_temp via compute_ensemble_forecast(), so we must NOT also
        # pass it to score_buckets() — that would double-weight the observation.
        # Only pass obs_data to scoring when it was NOT used in the ensemble.
        if config.multi_source and aviation_obs_temp is not None:
            scoring_obs = None  # Already incorporated in ensemble forecast
        else:
            scoring_obs = obs_data if (obs_data and days_ahead <= 1 and config.aviation_obs) else None
        if config.adjacent_buckets:
            scored = score_buckets(event_markets, forecast_temp, date_str, config,
                                   obs_data=scoring_obs, metric=metric,
                                   location=location, weather_data=om_data,
                                   sigma_override=adaptive_sigma_value)
            tradeable = [s for s in scored if s["ev"] >= config.min_ev_threshold]
        else:
            # Legacy: single-match
            tradeable = []
            for market in event_markets:
                outcome_name = market.get("outcome_name", "")
                bucket = parse_temperature_bucket(outcome_name)
                if bucket and bucket[0] <= forecast_temp <= bucket[1]:
                    best_ask = market.get("best_ask")
                    price = best_ask if best_ask and best_ask > 0 else (market.get("external_price_yes") or 0.5)
                    if MIN_TICK_SIZE <= price <= (1 - MIN_TICK_SIZE):
                        prob_raw = estimate_bucket_probability(
                            forecast_temp, bucket[0], bucket[1], date_str,
                            apply_seasonal=config.seasonal_adjustments,
                            location=location,
                            weather_data=om_data,
                            metric=metric,
                            sigma_override=adaptive_sigma_value,
                        )
                        prob = platt_calibrate(prob_raw)
                        if prob >= config.min_probability:
                            tradeable.append({
                                "market": market,
                                "bucket": bucket,
                                "outcome_name": outcome_name,
                                "prob": prob,
                                "prob_raw": prob_raw,
                                "price": price,
                                "ev": prob * (1.0 - config.trading_fees) - price,
                                "side": "yes",
                            })
                    break

        if not tradeable:
            logger.info("No tradeable buckets above EV threshold (%.2f)", config.min_ev_threshold)
            continue

        for entry in tradeable:
            market = entry["market"]
            market_id = market.get("id")
            outcome_name = entry["outcome_name"]
            price = entry["price"]
            prob = entry["prob"]
            ev = entry["ev"]

            logger.info(
                "Bucket: %s @ $%.2f — prob=%.1f%% EV=%.3f",
                outcome_name, price, prob * 100, ev,
            )

            if price >= config.entry_threshold:
                logger.info("Price $%.2f above entry threshold $%.2f — skip", price, config.entry_threshold)
                continue

            # Safeguards
            if use_safeguards:
                context = client.get_market_context(market_id, my_probability=prob)
                should_trade, reasons = check_context_safeguards(context, config)
                if not should_trade:
                    logger.info("Safeguard blocked: %s", "; ".join(reasons))
                    continue
                if reasons:
                    logger.info("Warnings: %s", "; ".join(reasons))

            # Kelly position sizing
            position_size = compute_position_size(
                probability=prob,
                price=price,
                balance=balance,
                max_position_usd=config.max_position_usd,
                kelly_frac=config.kelly_fraction,
            )

            if position_size <= 0:
                logger.info("Kelly says no bet (no edge at this price)")
                continue

            # Check minimum shares
            min_cost_for_shares = MIN_SHARES_PER_ORDER * price
            if min_cost_for_shares > position_size:
                logger.warning(
                    "Position size $%.2f too small for %d shares at $%.2f",
                    position_size, MIN_SHARES_PER_ORDER, price,
                )
                continue

            opportunities_found += 1

            # Rate limit
            if trades_executed >= config.max_trades_per_run:
                logger.info("Max trades per run (%d) reached — skipping", config.max_trades_per_run)
                continue

            # Confidence = our probability estimate
            confidence = round(prob, 2)

            side = entry.get("side", "yes")

            if dry_run:
                logger.info(
                    "[DRY RUN] Would buy %s $%.2f (~%.1f shares) of '%s' — confidence=%.0f%%",
                    side.upper(), position_size, position_size / price, outcome_name, confidence * 100,
                )
            else:
                logger.info("Executing trade: %s $%.2f on '%s'...", side.upper(), position_size, outcome_name)
                result = client.execute_trade(
                    market_id, side, position_size,
                    fill_timeout=config.fill_timeout_seconds,
                    fill_poll_interval=config.fill_poll_interval,
                )

                if result.get("success"):
                    trades_executed += 1
                    shares = result.get("shares_bought") or result.get("shares") or 0
                    trade_id = result.get("trade_id")
                    logger.info("Bought %.1f shares @ $%.2f", shares, price)

                    # Circuit breaker: record position opened
                    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    state.record_position_opened(today_str)

                    # Record in state for dynamic exits
                    state.record_trade(
                        market_id=market_id,
                        outcome_name=outcome_name,
                        side=side,
                        cost_basis=price,
                        shares=shares,
                        location=location,
                        forecast_date=date_str,
                        forecast_temp=forecast_temp,
                        metric=metric,
                    )

                    # Correlation guard: record event → market mapping
                    if config.correlation_guard:
                        state.record_event_position(event_id, market_id)

                    # Trade log: record for model improvement
                    bucket = entry["bucket"]
                    try:
                        log_trade(
                            location=location,
                            date=date_str,
                            metric=metric,
                            bucket=tuple(bucket),
                            prob_raw=entry.get("prob_raw", prob),
                            prob_platt=prob,
                            market_price=price,
                            position_usd=position_size,
                            shares=shares,
                            forecast_temp=forecast_temp,
                            sigma=adaptive_sigma_value,
                            horizon=get_horizon_days(date_str),
                        )
                    except Exception:
                        logger.debug("Failed to write trade log", exc_info=True)

                    # Calibration: record prediction
                    state.record_prediction(PredictionRecord(
                        market_id=market_id,
                        event_id=event_id,
                        location=location,
                        forecast_date=date_str,
                        metric=metric,
                        our_probability=prob,
                        forecast_temp=forecast_temp,
                        bucket_low=bucket[0],
                        bucket_high=bucket[1],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))

                    # Update remaining balance
                    balance = max(0.0, balance - position_size)
                else:
                    logger.error("Trade failed: %s", result.get("error", "Unknown"))

    # Stop-loss: check if forecast has reversed away from our held positions
    if config.stop_loss_reversal:
        _check_stop_loss_reversals(client, config, state, noaa_cache, open_meteo_cache, dry_run,
                                    aviation_cache=aviation_cache or None)

    # Check exits
    exits_found, exits_executed = check_exit_opportunities(
        client, config, state, dry_run, use_safeguards,
    )

    # Summary
    logger.info("=" * 50)
    logger.info("Summary: events=%d entries=%d exits=%d trades=%d",
                len(events), opportunities_found, exits_found,
                trades_executed + exits_executed)

    if dry_run:
        logger.info("[DRY RUN MODE — no real trades executed]")

    # Calibration stats
    cal = state.get_calibration_stats()
    if cal["count"] > 0:
        logger.info("Calibration: %d resolved, Brier=%.4f, accuracy=%.1f%%",
                     cal["count"], cal["brier"], cal["accuracy"] * 100)

    # Persist feedback state (EMA corrections)
    try:
        feedback.save()
    except Exception as exc:
        logger.warning("Failed to save feedback state: %s", exc)

    # Persist state
    save_path = state_path or config.state_file
    state.save(save_path)
