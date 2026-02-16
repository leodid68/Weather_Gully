"""Position sizing (Kelly criterion) and dynamic exit thresholds."""

import logging

logger = logging.getLogger(__name__)


def kelly_fraction(p: float, b: float, fraction: float = 0.25) -> float:
    """Fractional Kelly criterion.

    Args:
        p: Estimated probability of winning (0–1).
        b: Net odds received on the bet (e.g. ``(1/price) - 1``).
        fraction: Kelly fraction to use (default 0.25 = quarter-Kelly,
                  conservative because our probability estimates are noisy).

    Returns:
        Fraction of bankroll to wager (≥ 0).
    """
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    q = 1.0 - p
    full_kelly = (p * b - q) / b
    return max(0.0, full_kelly * fraction)


def compute_position_size(
    probability: float,
    price: float,
    balance: float,
    max_position_usd: float,
    kelly_frac: float = 0.25,
    min_trade: float = 1.0,
    current_exposure: float = 0.0,
) -> float:
    """Compute trade size in USD using Kelly criterion.

    Args:
        probability: Our estimated probability of the outcome.
        price: Current market price (0–1).
        balance: Available balance in USD.
        max_position_usd: Hard cap per position.
        kelly_frac: Kelly fraction (default quarter-Kelly).
        min_trade: Minimum viable trade size.

    Returns:
        Trade size in USD (0 if no edge).
    """
    if price <= 0 or price >= 1:
        return 0.0

    b = (1.0 / price) - 1.0  # Net odds
    frac = kelly_fraction(probability, b, fraction=kelly_frac)

    if frac <= 0:
        return 0.0

    # Budget: deduct current exposure from available balance
    available = max(0.0, balance - current_exposure)
    if available <= 0:
        return 0.0

    size = available * frac
    size = min(size, max_position_usd)

    # If Kelly says less than the minimum viable trade, don't trade at all
    if size < min_trade:
        return 0.0

    # Don't bet more than we have
    size = min(size, available)

    logger.debug(
        "Kelly sizing: p=%.3f price=%.3f b=%.2f frac=%.4f exposure=$%.2f → $%.2f",
        probability, price, b, frac, current_exposure, size,
    )
    return round(size, 2)


def compute_exit_threshold(
    cost_basis: float,
    hours_to_resolution: float,
) -> float:
    """Dynamic exit threshold based on cost basis and time remaining.

    Closer to resolution → lower target (take profit earlier).

    Args:
        cost_basis: Average price paid per share.
        hours_to_resolution: Hours until market resolves.

    Returns:
        Price threshold above which we should sell.
    """
    base_target = min(cost_basis * 2.0, 0.80)  # 2× cost, capped at 80¢

    if hours_to_resolution < 6:
        time_factor = 0.6
    elif hours_to_resolution < 24:
        time_factor = 0.8
    elif hours_to_resolution < 72:
        time_factor = 0.9
    else:
        time_factor = 1.0

    threshold = cost_basis + (base_target - cost_basis) * time_factor
    # Always require at least 5¢ profit
    return max(threshold, cost_basis + 0.05)
