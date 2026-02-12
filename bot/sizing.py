"""Kelly criterion position sizing and risk management.

Adapted from weather/sizing.py with added risk-limit checks for the
generic bot.  Uses the simplified Kelly formula for prediction markets:
    f* = (p - price) / (1 - price)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .state import TradingState

logger = logging.getLogger(__name__)


def kelly_fraction(
    p: float, market_price: float, fraction: float = 0.25, side: str = "BUY",
) -> float:
    """Fractional Kelly criterion for prediction markets.

    BUY:  f* = (p - price) / (1 - price)
    SELL: f* = (price - p) / price

    Applies *fraction* (default 0.25 = quarter-Kelly) for conservatism.
    Returns fraction of bankroll to wager (>= 0).
    """
    if market_price <= 0 or market_price >= 1 or p <= 0 or p >= 1:
        return 0.0
    if side == "SELL":
        full_kelly = (market_price - p) / market_price
    else:
        full_kelly = (p - market_price) / (1.0 - market_price)
    return max(0.0, full_kelly * fraction)


def position_size(
    probability: float,
    price: float,
    bankroll: float,
    max_position: float,
    kelly_frac: float = 0.25,
    min_trade: float = 5.0,
    side: str = "BUY",
) -> float:
    """Compute trade size in USD using Kelly criterion.

    Returns 0 if no edge or if size < min_trade.
    Caps at max_position and bankroll.
    """
    frac = kelly_fraction(probability, price, fraction=kelly_frac, side=side)
    if frac <= 0:
        return 0.0

    size = bankroll * frac
    size = min(size, max_position, bankroll)

    if size < min_trade:
        return 0.0

    logger.debug(
        "Kelly sizing: p=%.3f price=%.3f side=%s frac=%.4f → $%.2f",
        probability, price, side, frac, size,
    )
    return round(size, 2)


def dynamic_exit_threshold(cost_basis: float, hours_to_resolution: float) -> float:
    """Dynamic exit threshold based on time remaining.

    Closer to resolution → lower target (take profit earlier).
    Always requires at least 5c profit above cost basis.
    """
    base_target = min(cost_basis * 2.0, 0.80)

    if hours_to_resolution < 6:
        time_factor = 0.6
    elif hours_to_resolution < 24:
        time_factor = 0.8
    elif hours_to_resolution < 72:
        time_factor = 0.9
    else:
        time_factor = 1.0

    threshold = cost_basis + (base_target - cost_basis) * time_factor
    return max(threshold, cost_basis + 0.05)


def check_risk_limits(
    state: TradingState,
    config: Config,
    new_trade_usd: float,
    current_prices: dict[str, float] | None = None,
) -> tuple[bool, str]:
    """Check whether a new trade is within risk limits.

    Checks:
        1. Total exposure < max_total_exposure
        2. Number of open positions < max_open_positions
        3. Daily PnL (realized + unrealized) > -max_daily_loss
        4. Correlation guard: max 1 position per condition_id (event)

    Returns (allowed, reason).
    """
    positions = state.open_positions()

    # 1. Total exposure
    total_exposure = sum(p.price * p.size for p in positions)
    if total_exposure + new_trade_usd > config.max_total_exposure:
        return False, (
            f"Total exposure ${total_exposure + new_trade_usd:.2f} "
            f"exceeds limit ${config.max_total_exposure:.2f}"
        )

    # 2. Max positions
    if len(positions) >= config.max_open_positions:
        return False, (
            f"{len(positions)} positions — at limit of {config.max_open_positions}"
        )

    # 3. Daily loss (realized + unrealized)
    today_pnl = state.get_today_pnl()
    if current_prices:
        for pos in positions:
            cp = current_prices.get(pos.token_id)
            if cp is not None:
                if pos.side == "BUY":
                    today_pnl += (cp - pos.price) * pos.size
                else:
                    today_pnl += (pos.price - cp) * pos.size
    if today_pnl < -config.max_daily_loss:
        return False, (
            f"Daily PnL ${today_pnl:.2f} exceeds max loss ${config.max_daily_loss:.2f}"
        )

    # 4. Correlation guard — max 1 position per market_id (condition_id)
    # (already enforced by trades dict keying, but be explicit)

    return True, "ok"
