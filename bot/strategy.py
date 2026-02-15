"""Strategy runner — full pipeline.

One main function called once per run, no internal loop.
Schedule with cron or OpenClaw for periodic execution.

Pipeline:
    1. Check exits on open positions (stop-loss, take-profit BUY/SELL)
    2. scanner.scan_markets_gamma()  → candidate markets + multi-choice groups
       (falls back to CLOB scan if Gamma unavailable)
    3. scanner.filter_tradeable()    → liquidity filter
    4. signals.scan_for_signals()    → edge detection (4 methods)
    5. For each signal:
       a. sizing.check_risk_limits() → verify limits
       b. sizing.position_size()     → Kelly sizing (BUY + SELL)
       c. client.post_order()        → execute (if --live)
       d. state.record_trade()       → persist
       e. state.record_prediction()  → calibration tracking
    6. Weather pipeline (NOAA forecast → bucket scoring → trading)
    7. Resolve pending predictions (via Gamma)
    8. state.save()
    9. Log calibration stats
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .config import Config
from .scanner import (
    _scan_with_clob_fallback,
    compute_book_metrics,
    filter_tradeable,
    scan_markets_gamma,
)
from .signals import scan_for_signals
from .sizing import check_risk_limits, dynamic_exit_threshold, position_size
from .state import TradingState

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────

def _compute_pnl(pos, current_price: float) -> float:
    """Compute unrealized PnL for a position (BUY or SELL)."""
    if pos.side == "BUY":
        return (current_price - pos.price) * pos.size
    else:
        return (pos.price - current_price) * pos.size


def _compute_hours_to_resolution(end_date_str: str) -> float:
    """Parse ISO end_date → hours remaining until resolution."""
    if not end_date_str:
        return 48.0  # fallback
    try:
        end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = end_dt - now
        hours = delta.total_seconds() / 3600.0
        return max(0.0, hours)
    except (ValueError, TypeError):
        return 48.0


def _find_end_date(markets: list[dict], token_id: str) -> str:
    """Look up end_date_iso for a token from the market list."""
    for m in markets:
        for tok in m.get("tokens", []):
            if tok.get("token_id") == token_id:
                return m.get("end_date_iso", "")
    return ""


def _find_condition_id(markets: list[dict], token_id: str) -> str:
    for m in markets:
        for tok in m.get("tokens", []):
            if tok.get("token_id") == token_id:
                return m.get("condition_id", "")
    return ""


def _build_token_pairs(tradeable: list[dict]) -> dict[str, tuple[str, str]]:
    """Build condition_id → (yes_token_id, no_token_id) map for binary markets."""
    pairs: dict[str, tuple[str, str]] = {}
    for m in tradeable:
        tokens = m.get("tokens", [])
        cid = m.get("condition_id", "")
        if not cid or len(tokens) < 2:
            continue
        # Binary market: first token = YES, second = NO
        yes_tid = tokens[0].get("token_id", "")
        no_tid = tokens[1].get("token_id", "")
        if yes_tid and no_tid:
            pairs[cid] = (yes_tid, no_tid)
    return pairs


# ── Weather pipeline ──────────────────────────────────────────────────

def _run_weather_pipeline(
    client,
    config: Config,
    state: TradingState,
    dry_run: bool,
    state_path: str,
) -> None:
    """Run the full weather strategy pipeline using NOAA forecasts.

    Creates a CLOBWeatherBridge from the existing CLOB client + a GammaClient,
    builds a weather.config.Config from bot.config.Config fields, and delegates
    to weather.strategy.run_weather_strategy().
    """
    from .gamma import GammaClient
    from weather.bridge import CLOBWeatherBridge
    from weather.config import Config as WeatherConfig
    from weather.state import TradingState as WeatherState
    from weather.strategy import run_weather_strategy

    with GammaClient() as gamma:
        bridge = CLOBWeatherBridge(
            clob_client=client,
            gamma_client=gamma,
            max_exposure=config.max_total_exposure,
        )

        weather_config = WeatherConfig(
            locations=config.weather_locations,
            entry_threshold=config.weather_entry_threshold,
            exit_threshold=config.weather_exit_threshold,
            max_days_ahead=config.weather_max_days_ahead,
            seasonal_adjustments=config.weather_seasonal_adjustments,
            multi_source=config.weather_multi_source,
            correlation_guard=config.weather_correlation_guard,
            stop_loss_reversal=config.weather_stop_loss_reversal,
            stop_loss_reversal_threshold=config.weather_stop_loss_reversal_threshold,
            max_exposure=config.max_total_exposure,
            kelly_fraction=config.kelly_fraction,
            max_position_usd=config.max_position_usd,
        )

        # Weather state is separate from bot state
        import os
        base, ext = os.path.splitext(state_path)
        weather_state_path = base + "_weather" + ext
        weather_state = WeatherState.load(weather_state_path)

        run_weather_strategy(
            client=bridge,
            config=weather_config,
            state=weather_state,
            dry_run=dry_run,
            use_safeguards=True,
            use_trends=True,
            state_path=weather_state_path,
        )


# ── Main strategy ──────────────────────────────────────────────────────

def run_strategy(
    client,
    config: Config,
    state: TradingState,
    dry_run: bool = True,
    state_path: str = "state.json",
) -> None:
    """Main entry point — called once per run."""
    trades_this_run = 0

    # ── 1. Check exits on existing positions ─────────────────────────
    positions = state.open_positions()
    logger.info("Open positions: %d", len(positions))

    # Fetch current prices in parallel
    price_map: dict[str, float] = {}
    if positions:
        def _get_price(pos):
            try:
                info = client.get_price(pos.token_id)
                return pos, float(info.get("price", 0))
            except Exception:
                logger.warning("Failed to get price for %s, skipping exit check", pos.token_id[:16])
                return pos, None

        with ThreadPoolExecutor(max_workers=config.parallel_workers) as pool:
            futures = {pool.submit(_get_price, pos): pos for pos in positions}
            for future in as_completed(futures):
                pos, current = future.result()
                if current is None:
                    continue
                price_map[pos.token_id] = current

    for pos in positions:
        current = price_map.get(pos.token_id)
        if current is None:
            continue

        # Skip positions with pending exit orders
        if pos.memo == "pending_exit":
            logger.debug("Skipping %s — pending exit order", pos.token_id[:16])
            continue

        # Stop-loss check
        if pos.side == "BUY":
            loss_pct = (pos.price - current) / pos.price if pos.price > 0 else 0
        else:
            # SELL loss = price moved up against us; clamp denominator to avoid
            # extreme loss_pct when pos.price is near 1.0
            denom = max(0.05, 1.0 - pos.price)
            loss_pct = (current - pos.price) / denom if current > pos.price else 0

        if loss_pct >= config.stop_loss_pct:
            pnl = _compute_pnl(pos, current)
            logger.info(
                "STOP-LOSS: %s %s @ %.4f (entry %.4f, loss %.0f%%, pnl $%.2f)",
                pos.side, pos.token_id[:16], current, pos.price, loss_pct * 100, pnl,
            )
            if not dry_run:
                try:
                    exit_side = "SELL" if pos.side == "BUY" else "BUY"
                    # Aggressive pricing: accept 2% slippage to ensure fill
                    if exit_side == "SELL":
                        aggressive_price = round(max(0.01, current * 0.98), 4)
                    else:
                        aggressive_price = round(min(0.99, current * 1.02), 4)
                    result = client.post_order(
                        pos.token_id, exit_side, aggressive_price, pos.size,
                    )
                    order_id = result.get("orderID", "")
                    # Mark as pending exit — only fully close after fill
                    state.record_trade(
                        market_id=pos.market_id, token_id=pos.token_id,
                        side=pos.side, price=pos.price, size=pos.size,
                        order_id=order_id, memo="pending_exit",
                        end_date=pos.end_date, condition_id=pos.condition_id,
                    )
                    # Best-effort fill check
                    filled = False
                    if order_id:
                        try:
                            filled = client.is_order_filled(order_id)
                        except Exception:
                            pass
                    if filled:
                        state.record_daily_pnl(pnl)
                        state.record_closed_trade(pos, current, pnl)
                        state.remove_trade(pos.market_id)
                    else:
                        logger.warning("Stop-loss order %s pending fill", order_id)
                except Exception as exc:
                    logger.error("Stop-loss order failed: %s", exc)
            continue

        # Dynamic exit threshold
        hours = _compute_hours_to_resolution(pos.end_date)
        threshold = dynamic_exit_threshold(pos.price, hours_to_resolution=hours)

        # Take-profit: BUY exit
        if current >= threshold and pos.side == "BUY":
            pnl = _compute_pnl(pos, current)
            logger.info(
                "EXIT BUY: %s @ %.4f (entry %.4f, pnl $%.2f)",
                pos.token_id[:16], current, pos.price, pnl,
            )
            if not dry_run:
                try:
                    result = client.post_order(
                        pos.token_id, "SELL", current, pos.size,
                    )
                    order_id = result.get("orderID", "")
                    filled = False
                    if order_id:
                        try:
                            filled = client.is_order_filled(order_id)
                        except Exception:
                            pass
                    if filled:
                        state.record_daily_pnl(pnl)
                        state.record_closed_trade(pos, current, pnl)
                        state.remove_trade(pos.market_id)
                    else:
                        state.record_trade(
                            market_id=pos.market_id, token_id=pos.token_id,
                            side=pos.side, price=pos.price, size=pos.size,
                            order_id=order_id, memo="pending_exit",
                            end_date=pos.end_date, condition_id=pos.condition_id,
                        )
                        logger.warning("Exit order %s pending fill", order_id)
                except Exception as exc:
                    logger.error("Exit order failed: %s", exc)

        # Take-profit: SELL exit (price dropped enough to lock in profit)
        # For SELL, we profit when price drops. Exit when profit is sufficient:
        # profit_pct = (entry - current) / entry >= (threshold - entry) / entry
        elif pos.side == "SELL" and current <= 2 * pos.price - threshold:
            pnl = _compute_pnl(pos, current)
            logger.info(
                "EXIT SELL: %s @ %.4f (entry %.4f, pnl $%.2f)",
                pos.token_id[:16], current, pos.price, pnl,
            )
            if not dry_run:
                try:
                    result = client.post_order(
                        pos.token_id, "BUY", current, pos.size,
                    )
                    order_id = result.get("orderID", "")
                    filled = False
                    if order_id:
                        try:
                            filled = client.is_order_filled(order_id)
                        except Exception:
                            pass
                    if filled:
                        state.record_daily_pnl(pnl)
                        state.record_closed_trade(pos, current, pnl)
                        state.remove_trade(pos.market_id)
                    else:
                        state.record_trade(
                            market_id=pos.market_id, token_id=pos.token_id,
                            side=pos.side, price=pos.price, size=pos.size,
                            order_id=order_id, memo="pending_exit",
                            end_date=pos.end_date, condition_id=pos.condition_id,
                        )
                        logger.warning("Exit order %s pending fill", order_id)
                except Exception as exc:
                    logger.error("Exit order failed: %s", exc)

    # ── 2. Scan markets (Gamma preferred, CLOB fallback) ─────────────
    multi_choice_groups = []

    if config.use_gamma:
        try:
            raw_markets, multi_choice_groups = scan_markets_gamma(config)
            logger.info(
                "Gamma: %d markets, %d multi-choice groups",
                len(raw_markets), len(multi_choice_groups),
            )
        except Exception as exc:
            logger.warning("Gamma scan failed (%s), falling back to CLOB", exc)
            raw_markets = _scan_with_clob_fallback(client, config)
    else:
        raw_markets = _scan_with_clob_fallback(client, config)

    # ── 2b. Weather markets (optional) ─────────────────────────────
    # Shared GammaClient for weather scan + prediction resolution
    _shared_gamma = None
    if config.use_gamma:
        from .gamma import GammaClient, group_multi_choice, gamma_to_scanner_format, resolve_pending_predictions
        _shared_gamma = GammaClient()

    try:  # ensure _shared_gamma is always closed

        if config.weather_enabled and _shared_gamma is not None:
            try:
                _, weather_markets = _shared_gamma.fetch_events_with_markets(tag_slug="weather")
                weather_mc_groups = group_multi_choice(weather_markets, gamma_client=_shared_gamma)
                weather_tradeable = gamma_to_scanner_format(weather_markets)
                # Deduplicate: skip markets already in main scan
                existing_cids = {m.get("condition_id") for m in raw_markets}
                weather_tradeable = [
                    m for m in weather_tradeable
                    if m.get("condition_id") not in existing_cids
                ]
                existing_eids = {g.event_id for g in multi_choice_groups}
                weather_mc_groups = [
                    g for g in weather_mc_groups
                    if g.event_id not in existing_eids
                ]
                raw_markets.extend(weather_tradeable)
                multi_choice_groups.extend(weather_mc_groups)
                logger.info("Weather: +%d markets, +%d multi-choice groups (after dedup)",
                            len(weather_tradeable), len(weather_mc_groups))
            except Exception as exc:
                logger.warning("Weather scan failed: %s — continuing without", exc)

        # ── 3. Filter by liquidity ───────────────────────────────────
        tradeable = filter_tradeable(raw_markets, min_liquidity=config.min_liquidity_grade)

        # For CLOB-discovered markets without Gamma grades, compute book metrics
        for m in tradeable:
            if "gamma" not in m:
                for tok in m.get("tokens", []):
                    if "metrics" not in tok:
                        try:
                            book = client.get_orderbook(tok["token_id"])
                            tok["metrics"] = compute_book_metrics(book)
                        except Exception:
                            tok["metrics"] = {"liquidity_grade": "D"}

        # ── 4. Detect signals ────────────────────────────────────────
        token_ids = []
        token_prices: dict[str, float] = {}
        for m in tradeable:
            for tok in m.get("tokens", []):
                token_ids.append(tok["token_id"])
                # Use Gamma price if available (avoids CLOB /price 400 errors)
                if tok.get("price"):
                    token_prices[tok["token_id"]] = float(tok["price"])

        # Build token pairs for binary arbitrage
        token_pairs = _build_token_pairs(tradeable)

        signals = scan_for_signals(
            client, token_ids, config,
            multi_choice_groups=multi_choice_groups,
            token_prices=token_prices,
            token_pairs=token_pairs,
        )
        logger.info("Signals detected: %d", len(signals))

        # ── 5. Execute on signals ────────────────────────────────────
        for sig in signals:
            if trades_this_run >= config.max_trades_per_run:
                logger.info("Max trades per run reached (%d)", config.max_trades_per_run)
                break

            size_usd = position_size(
                probability=sig.fair_value,
                price=sig.market_price,
                bankroll=config.max_total_exposure,
                max_position=config.max_position_usd,
                kelly_frac=config.kelly_fraction,
                side=sig.side,
            )
            if size_usd <= 0:
                continue

            allowed, reason = check_risk_limits(
                state, config, size_usd, current_prices=price_map,
            )
            if not allowed:
                logger.info("Risk limit: %s — skipping %s", reason, sig.token_id[:16])
                continue

            # Find the condition_id and end_date for this token
            condition_id = _find_condition_id(tradeable, sig.token_id)
            end_date = _find_end_date(tradeable, sig.token_id)

            logger.info(
                "TRADE: %s %s @ %.4f, size=$%.2f, edge=%.4f [%s]",
                sig.side, sig.token_id[:16], sig.market_price, size_usd,
                sig.edge, sig.method,
            )

            if not dry_run:
                try:
                    # Re-fetch fresh price to avoid stale data
                    try:
                        fresh_price_info = client.get_price(sig.token_id)
                        fresh_price = float(fresh_price_info.get("price", sig.market_price))
                    except Exception:
                        fresh_price = sig.market_price

                    if fresh_price <= 0:
                        logger.warning("Invalid fresh price for %s — skipping", sig.token_id[:16])
                        continue
                    # BUY: cost = price * shares → shares = usd / price
                    # SELL: max_loss = (1-price) * shares → shares = usd / (1-price)
                    if sig.side == "SELL":
                        shares = size_usd / (1.0 - fresh_price)
                    else:
                        shares = size_usd / fresh_price
                    if shares <= 0:
                        logger.warning("Zero shares for %s — skipping", sig.token_id[:16])
                        continue
                    result = client.post_order(
                        sig.token_id, sig.side, fresh_price, shares,
                    )
                    order_id = result.get("orderID", "")

                    # Verify fill status (best-effort)
                    filled = False
                    if order_id:
                        try:
                            filled = client.is_order_filled(order_id)
                        except Exception:
                            logger.debug("Could not verify fill for order %s", order_id)
                            filled = False

                    if not filled:
                        logger.warning(
                            "Order %s may not be filled — recording as pending", order_id,
                        )

                    state.record_trade(
                        market_id=condition_id or sig.token_id,
                        token_id=sig.token_id,
                        side=sig.side,
                        price=fresh_price,
                        size=shares,
                        order_id=order_id,
                        end_date=end_date,
                        condition_id=condition_id,
                        memo="" if filled else "pending_fill",
                    )
                except Exception as exc:
                    logger.error("Order failed: %s", exc)
                    continue
            else:
                dry_shares = size_usd / sig.market_price if sig.market_price > 0 else 0
                if dry_shares > 0:
                    logger.info(
                        "[DRY RUN] Would %s %.2f shares of %s @ %.4f ($%.2f)",
                        sig.side, dry_shares, sig.token_id[:16], sig.market_price, size_usd,
                    )

            state.record_prediction(
                condition_id or sig.token_id, sig.fair_value, sig.market_price,
            )
            trades_this_run += 1

        logger.info("Trades this run: %d (max %d)", trades_this_run, config.max_trades_per_run)

        # ── 6. Weather pipeline (NOAA forecast → trading) ─────────────
        if config.weather_enabled:
            try:
                _run_weather_pipeline(client, config, state, dry_run, state_path)
            except Exception as exc:
                logger.warning("Weather pipeline failed: %s — continuing", exc)

        # ── 7. Resolve pending predictions (via shared Gamma client) ──
        if _shared_gamma is not None:
            try:
                resolve_pending_predictions(state, _shared_gamma)
            except Exception as exc:
                logger.debug("Prediction resolution skipped: %s", exc)

        # ── 8. Log calibration ───────────────────────────────────────
        cal = state.get_calibration()
        if cal["n"] > 0:
            logger.info(
                "Calibration: Brier=%.4f  Log=%.4f  (n=%d)",
                cal["brier"], cal["log"], cal["n"],
            )

    finally:
        if _shared_gamma is not None:
            _shared_gamma.close()

    # ── 9. Persist state ─────────────────────────────────────────────
    try:
        state.save(state_path)
    except Exception as exc:
        logger.critical("FAILED TO SAVE STATE: %s", exc)
        raise
    logger.info("Done.")


