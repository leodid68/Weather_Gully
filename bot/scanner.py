"""Market scanning, order-book metrics, and liquidity filtering.

Two discovery modes:
- **Gamma** (default): Uses gamma-api.polymarket.com for rich metadata
  (volume, liquidity, bestBid/bestAsk, multi-choice detection).
- **CLOB fallback**: Uses clob.polymarket.com /sampling-markets.

Both feed into the same pipeline via a common dict format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

_GRADE_BPS = {"A": 50, "B": 100, "C": 200}


# ── Gamma-based scanning ──────────────────────────────────────────────


def scan_markets_gamma(
    config: Config,
) -> tuple[list[dict], list]:
    """Scan markets using the Gamma API.

    Returns (markets_in_scanner_format, multi_choice_groups).
    The first list is compatible with filter_tradeable / strategy pipeline.
    The second is a list of MultiChoiceGroup for multi-choice arbitrage.
    """
    from .gamma import GammaClient, gamma_to_scanner_format, group_multi_choice

    with GammaClient() as gamma:
        gamma_markets = gamma.fetch_markets(
            limit=config.scan_limit,
            active=True,
            closed=False,
            min_volume=config.gamma_min_volume,
            min_liquidity=config.gamma_min_liquidity,
        )

        # Convert to scanner dict format (includes liquidity_grade from Gamma spread)
        markets = gamma_to_scanner_format(gamma_markets)

        # Detect multi-choice groups — pass client to complete incomplete groups
        groups = group_multi_choice(gamma_markets, gamma_client=gamma)

    logger.info(
        "Gamma scan: %d markets, %d multi-choice groups",
        len(markets), len(groups),
    )
    return markets, groups


# ── CLOB-based scanning (fallback) ────────────────────────────────────


def scan_markets(client, **filters) -> list[dict]:
    """Fetch active markets from CLOB API and extract token IDs.

    Returns a list of dicts, each with at least:
        condition_id, question, tokens: [{token_id, outcome}], active, ...
    """
    raw = client.get_markets(**filters)
    markets: list[dict] = []

    for m in raw:
        if m.get("closed", False):
            continue
        if not m.get("accepting_orders", m.get("active", True)):
            continue
        if not m.get("enable_order_book", True):
            continue

        tokens = m.get("tokens", [])
        if not tokens:
            continue

        markets.append({
            "condition_id": m.get("condition_id", ""),
            "question": m.get("question", ""),
            "tokens": tokens,
            "active": True,
            "neg_risk": m.get("neg_risk", False),
            "end_date_iso": m.get("end_date_iso", ""),
        })

    logger.info("CLOB scan: %d active markets (from %d raw)", len(markets), len(raw))
    return markets


# ── Order-book metrics ─────────────────────────────────────────────────


def compute_book_metrics(book: dict) -> dict:
    """Compute microstructure metrics from an order-book snapshot.

    Returns dict with: mid_price, spread, spread_bps, depth_bid_5,
    depth_ask_5, imbalance, kyle_lambda, liquidity_grade.
    """
    bids = book.get("bids") or []
    asks = book.get("asks") or []

    empty = {
        "mid_price": 0.0, "spread": 1.0, "spread_bps": 10_000,
        "depth_bid_5": 0.0, "depth_ask_5": 0.0,
        "imbalance": 0.0, "kyle_lambda": 0.0,
        "liquidity_grade": "D",
    }

    if not bids or not asks:
        return empty

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    if mid <= 0:
        return empty

    spread_bps = (spread / mid) * 10_000

    depth_bid = sum(float(b.get("size", 0)) for b in bids[:5])
    depth_ask = sum(float(a.get("size", 0)) for a in asks[:5])
    total_depth = depth_bid + depth_ask
    imbalance = (depth_bid - depth_ask) / total_depth if total_depth > 0 else 0.0
    kyle_lambda = spread / total_depth if total_depth > 0 else 0.0

    # Liquidity grade
    if spread_bps < _GRADE_BPS["A"]:
        grade = "A"
    elif spread_bps < _GRADE_BPS["B"]:
        grade = "B"
    elif spread_bps < _GRADE_BPS["C"]:
        grade = "C"
    else:
        grade = "D"

    return {
        "mid_price": round(mid, 4),
        "spread": round(spread, 4),
        "spread_bps": round(spread_bps, 1),
        "depth_bid_5": round(depth_bid, 2),
        "depth_ask_5": round(depth_ask, 2),
        "imbalance": round(imbalance, 4),
        "kyle_lambda": round(kyle_lambda, 6),
        "liquidity_grade": grade,
    }


# ── Filtering ──────────────────────────────────────────────────────────


def run_scan_pipeline(
    client, config: "Config",
) -> tuple[list[dict], list, list[str], dict[str, float], dict[str, tuple[str, str]]]:
    """Shared scan pipeline used by strategy, --scan, and --signals.

    Returns (tradeable, mc_groups, token_ids, token_prices, token_pairs).
    """
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

    tradeable = filter_tradeable(raw_markets, min_liquidity=config.min_liquidity_grade)

    token_ids: list[str] = []
    token_prices: dict[str, float] = {}
    token_pairs: dict[str, tuple[str, str]] = {}

    for m in tradeable:
        tokens = m.get("tokens", [])
        cid = m.get("condition_id", "")
        for tok in tokens:
            token_ids.append(tok["token_id"])
            if tok.get("price"):
                token_prices[tok["token_id"]] = float(tok["price"])
        # Build token pairs for binary arb
        if cid and len(tokens) >= 2:
            yes_tid = tokens[0].get("token_id", "")
            no_tid = tokens[1].get("token_id", "")
            if yes_tid and no_tid:
                token_pairs[cid] = (yes_tid, no_tid)

    return tradeable, multi_choice_groups, token_ids, token_prices, token_pairs


def _scan_with_clob_fallback(client, config: "Config") -> list[dict]:
    """CLOB-based scan with book metrics computation (used as fallback)."""
    from .scanner import compute_book_metrics, scan_markets
    try:
        raw_markets = scan_markets(client, limit=config.scan_limit)
    except Exception as exc:
        logger.error("CLOB scan failed: %s", exc)
        return []

    for m in raw_markets:
        best_grade = "D"
        for tok in m.get("tokens", []):
            try:
                book = client.get_orderbook(tok["token_id"])
                metrics = compute_book_metrics(book)
                tok["metrics"] = metrics
                grade = metrics["liquidity_grade"]
                if {"A": 0, "B": 1, "C": 2, "D": 3}.get(grade, 3) < {"A": 0, "B": 1, "C": 2, "D": 3}.get(best_grade, 3):
                    best_grade = grade
            except Exception:
                tok["metrics"] = {"liquidity_grade": "D"}
        m["liquidity_grade"] = best_grade
    return raw_markets


def filter_tradeable(
    markets: list[dict], min_liquidity: str = "C",
) -> list[dict]:
    """Keep only markets whose best token meets the minimum liquidity grade.

    Grade ordering: A > B > C > D.  Markets graded below *min_liquidity*
    are excluded.
    """
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    cutoff = grade_order.get(min_liquidity, 2)

    result = []
    for m in markets:
        grade = m.get("liquidity_grade", "D")
        if grade_order.get(grade, 3) <= cutoff:
            result.append(m)

    logger.info(
        "Filtered to %d tradeable markets (min grade %s)", len(result), min_liquidity,
    )
    return result
