"""Main strategy loop — scores buckets, sizes positions, manages exits.

Uses CLOBWeatherBridge for all market data and order execution (CLOB direct).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .config import Config, LOCATIONS, MIN_SHARES_PER_ORDER, MIN_TICK_SIZE
from .noaa import get_noaa_forecast
from .open_meteo import compute_ensemble_forecast, get_open_meteo_forecast
from .parsing import parse_weather_event, parse_temperature_bucket
from .probability import estimate_bucket_probability, get_horizon_days, get_noaa_probability
from .bridge import CLOBWeatherBridge
from .sizing import compute_exit_threshold, compute_position_size
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


def detect_price_trend(history: list, price_drop_threshold: float = 0.10) -> dict:
    """Analyze price history for 24h trend."""
    if not history or len(history) < 2:
        return {"direction": "unknown", "change_24h": 0, "is_opportunity": False}

    recent_price = history[-1].get("price_yes", 0.5)
    lookback = min(96, len(history) - 1)
    old_price = history[-lookback].get("price_yes", recent_price)

    if old_price == 0:
        return {"direction": "unknown", "change_24h": 0, "is_opportunity": False}

    change = (recent_price - old_price) / old_price

    if change < -price_drop_threshold:
        return {"direction": "down", "change_24h": change, "is_opportunity": True}
    elif change > price_drop_threshold:
        return {"direction": "up", "change_24h": change, "is_opportunity": False}
    return {"direction": "flat", "change_24h": change, "is_opportunity": False}


# --------------------------------------------------------------------------
# Bucket scoring (adjacent buckets)
# --------------------------------------------------------------------------

def score_buckets(
    event_markets: list[dict],
    forecast_temp: float,
    forecast_date: str,
    config: Config,
) -> list[dict]:
    """Score all buckets in an event by expected value.

    Returns a list of ``{"market": dict, "bucket": tuple, "prob": float,
    "price": float, "ev": float}`` sorted by EV descending.
    """
    scored: list[dict] = []

    for market in event_markets:
        outcome_name = market.get("outcome_name", "")
        bucket = parse_temperature_bucket(outcome_name)
        if not bucket:
            continue

        price = market.get("external_price_yes") or 0.5

        # Skip extreme prices
        if price < MIN_TICK_SIZE or price > (1 - MIN_TICK_SIZE):
            continue

        prob = estimate_bucket_probability(
            forecast_temp, bucket[0], bucket[1], forecast_date,
            apply_seasonal=config.seasonal_adjustments,
        )

        ev = prob - price  # Expected value above break-even

        scored.append({
            "market": market,
            "bucket": bucket,
            "outcome_name": outcome_name,
            "prob": prob,
            "price": price,
            "ev": ev,
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
    """Check open positions for exit opportunities. Returns ``(found, executed)``."""
    positions = client.get_positions()
    if not positions:
        return 0, 0

    weather_positions = []
    for pos in positions:
        question = pos.get("question", "").lower()
        sources = pos.get("sources", [])
        if _TRADE_SOURCE in sources or any(
            kw in question for kw in ["temperature", "\u00b0f", "highest temp", "lowest temp"]
        ):
            weather_positions.append(pos)

    if not weather_positions:
        return 0, 0

    logger.info("Checking %d weather positions for exit...", len(weather_positions))
    exits_found = 0
    exits_executed = 0

    for pos in weather_positions:
        market_id = pos.get("market_id")
        current_price = pos.get("current_price") or pos.get("price_yes") or 0
        shares = pos.get("shares_yes") or pos.get("shares") or 0
        question = pos.get("question", "Unknown")[:50]

        if shares < MIN_SHARES_PER_ORDER:
            continue

        # Dynamic exit threshold
        cost_basis = state.get_cost_basis(market_id) or config.exit_threshold
        if config.dynamic_exits:
            time_str = pos.get("time_to_resolution", "")
            hours = _parse_time_to_hours(time_str) if time_str else 168.0  # default 7 days
            if hours is None:
                hours = 168.0
            exit_threshold = compute_exit_threshold(cost_basis, hours)
        else:
            exit_threshold = config.exit_threshold

        if current_price >= exit_threshold:
            exits_found += 1
            logger.info(
                "EXIT: %s — price $%.2f >= threshold $%.2f",
                question, current_price, exit_threshold,
            )

            if use_safeguards:
                context = client.get_market_context(market_id)
                should_trade, reasons = check_context_safeguards(context, config)
                if not should_trade:
                    logger.info("Exit skipped: %s", "; ".join(reasons))
                    continue
                if reasons:
                    logger.info("Exit warnings: %s", "; ".join(reasons))

            # Race condition guard: re-read fresh position
            fresh = client.get_position(market_id)
            fresh_shares = 0.0
            if fresh:
                fresh_shares = fresh.get("shares_yes") or fresh.get("shares") or 0
            if fresh_shares < MIN_SHARES_PER_ORDER:
                logger.info("Position already closed for %s, skipping", market_id)
                continue

            if dry_run:
                logger.info("[DRY RUN] Would sell %.1f shares of %s", fresh_shares, question)
            else:
                logger.info("Selling %.1f shares of %s...", fresh_shares, question)
                result = client.execute_sell(market_id, fresh_shares)

                if result.get("success"):
                    exits_executed += 1
                    logger.info("Sold %.1f shares @ $%.2f", fresh_shares, current_price)
                    state.remove_trade(market_id)
                    # Clean up correlation guard
                    for eid, mid in list(state.event_positions.items()):
                        if mid == market_id:
                            state.remove_event_position(eid)
                            break
                else:
                    logger.error("Sell failed: %s", result.get("error", "Unknown"))
        else:
            logger.debug(
                "HOLD: %s — price $%.2f < threshold $%.2f",
                question, current_price, exit_threshold,
            )

    return exits_found, exits_executed


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
) -> int:
    """Exit positions where the forecast has shifted away from our bucket.

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

            # Attempt to sell
            fresh = client.get_position(market_id)
            fresh_shares = 0.0
            if fresh:
                fresh_shares = fresh.get("shares_yes") or fresh.get("shares") or 0

            if fresh_shares < MIN_SHARES_PER_ORDER:
                logger.info("Position already closed for %s", market_id)
                state.remove_trade(market_id)
                continue

            if dry_run:
                logger.info("[DRY RUN] Would stop-loss sell %.1f shares of %s", fresh_shares, trade.outcome_name)
            else:
                logger.info("Stop-loss selling %.1f shares of %s...", fresh_shares, trade.outcome_name)
                result = client.execute_sell(market_id, fresh_shares)
                if result.get("success"):
                    exits += 1
                    state.remove_trade(market_id)
                    # Clean up correlation guard
                    for eid, mid in list(state.event_positions.items()):
                        if mid == market_id:
                            state.remove_event_position(eid)
                            break
                    logger.info("Stop-loss executed for %s", trade.outcome_name)
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
    use_trends: bool = True,
    state_path: str | None = None,
) -> None:
    """Run the weather trading strategy."""
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
            logger.info("  %s = %s", f.name, getattr(config, f.name))
        return

    if positions_only:
        logger.info("Current Positions:")
        positions = client.get_positions()
        if not positions:
            logger.info("  No open positions")
        else:
            for pos in positions:
                logger.info(
                    "  %s — YES: %.1f | NO: %.1f | P&L: $%.2f",
                    pos.get("question", "Unknown")[:50],
                    pos.get("shares_yes", 0),
                    pos.get("shares_no", 0),
                    pos.get("pnl", 0),
                )
        return

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
    trades_executed = 0
    opportunities_found = 0

    # Parallel pre-fetch: NOAA + Open-Meteo for all active locations
    active_locs = config.active_locations
    logger.info("Pre-fetching forecasts for %d locations in parallel...", len(active_locs))

    def _fetch_noaa(loc_name: str) -> tuple[str, dict]:
        return loc_name, get_noaa_forecast(
            loc_name, LOCATIONS,
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
        )

    def _fetch_open_meteo(loc_name: str) -> tuple[str, dict]:
        loc_data = LOCATIONS.get(loc_name)
        if not loc_data:
            return loc_name, {}
        return loc_name, get_open_meteo_forecast(
            loc_data["lat"], loc_data["lon"],
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
        )

    with ThreadPoolExecutor(max_workers=len(active_locs) * 2) as pool:
        # Submit all NOAA fetches
        noaa_futures = {pool.submit(_fetch_noaa, loc): loc for loc in active_locs}
        # Submit all Open-Meteo fetches (if enabled)
        om_futures = {}
        if config.multi_source:
            om_futures = {pool.submit(_fetch_open_meteo, loc): loc for loc in active_locs}

        for fut in as_completed(list(noaa_futures) + list(om_futures)):
            is_noaa = fut in noaa_futures
            loc = noaa_futures.get(fut) or om_futures.get(fut, "?")
            source = "NOAA" if is_noaa else "Open-Meteo"
            try:
                loc_name, data = fut.result()
                if is_noaa:
                    noaa_cache[loc_name] = data
                    if not data:
                        logger.warning("NOAA returned empty forecast for %s", loc_name)
                else:
                    open_meteo_cache[loc_name] = data
                    if not data:
                        logger.warning("Open-Meteo returned empty forecast for %s", loc_name)
            except Exception as exc:
                logger.error("%s fetch crashed for %s: %s", source, loc, exc)

    logger.info("Forecasts ready: NOAA=%d, Open-Meteo=%d", len(noaa_cache), len(open_meteo_cache))

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

        # Multi-source ensemble forecast
        om_data = open_meteo_cache.get(location, {}).get(date_str)
        if config.multi_source and (noaa_temp is not None or om_data):
            ensemble_temp, model_spread = compute_ensemble_forecast(noaa_temp, om_data, metric)
            if ensemble_temp is not None:
                logger.info("Ensemble forecast: %.1f°F (NOAA=%s, spread=%.1f°F)",
                             ensemble_temp,
                             f"{noaa_temp}°F" if noaa_temp is not None else "N/A",
                             model_spread)
                forecast_temp = ensemble_temp
            else:
                forecast_temp = noaa_temp
        elif noaa_temp is not None:
            forecast_temp = noaa_temp
        elif om_data:
            # Fallback: NOAA failed but Open-Meteo available
            fallback_temp, _ = compute_ensemble_forecast(None, om_data, metric)
            if fallback_temp is not None:
                logger.warning("NOAA unavailable for %s %s — falling back to Open-Meteo (%.1f°F)",
                               location, date_str, fallback_temp)
            forecast_temp = fallback_temp
        else:
            forecast_temp = None

        if forecast_temp is None:
            logger.warning("No forecast available for %s %s", location, date_str)
            continue

        # Forecast change detection
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
        if config.adjacent_buckets:
            scored = score_buckets(event_markets, forecast_temp, date_str, config)
            tradeable = [s for s in scored if s["ev"] >= config.min_ev_threshold]
        else:
            # Legacy: single-match
            tradeable = []
            for market in event_markets:
                outcome_name = market.get("outcome_name", "")
                bucket = parse_temperature_bucket(outcome_name)
                if bucket and bucket[0] <= forecast_temp <= bucket[1]:
                    price = market.get("external_price_yes") or 0.5
                    if MIN_TICK_SIZE <= price <= (1 - MIN_TICK_SIZE):
                        prob = estimate_bucket_probability(
                            forecast_temp, bucket[0], bucket[1], date_str,
                            apply_seasonal=config.seasonal_adjustments,
                        )
                        tradeable.append({
                            "market": market,
                            "bucket": bucket,
                            "outcome_name": outcome_name,
                            "prob": prob,
                            "price": price,
                            "ev": prob - price,
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

            # Price trend
            if use_trends:
                history = client.get_price_history(market_id)
                trend = detect_price_trend(history, config.price_drop_threshold)
                if trend["is_opportunity"]:
                    logger.info("Price dropped %.0f%% in 24h — stronger signal", abs(trend["change_24h"]) * 100)

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

            if dry_run:
                logger.info(
                    "[DRY RUN] Would buy $%.2f (~%.1f shares) of '%s' — confidence=%.0f%%",
                    position_size, position_size / price, outcome_name, confidence * 100,
                )
            else:
                logger.info("Executing trade: $%.2f on '%s'...", position_size, outcome_name)
                result = client.execute_trade(market_id, "yes", position_size)

                if result.get("success"):
                    trades_executed += 1
                    shares = result.get("shares_bought") or result.get("shares") or 0
                    trade_id = result.get("trade_id")
                    logger.info("Bought %.1f shares @ $%.2f", shares, price)

                    # Record in state for dynamic exits
                    state.record_trade(
                        market_id=market_id,
                        outcome_name=outcome_name,
                        side="yes",
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

                    # Calibration: record prediction
                    bucket = entry["bucket"]
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
        _check_stop_loss_reversals(client, config, state, noaa_cache, open_meteo_cache, dry_run)

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

    # Persist state
    save_path = state_path or config.state_file
    state.save(save_path)
