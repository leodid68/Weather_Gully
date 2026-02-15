"""Edge detection via four independent methods.

1. **Longshot bias** — empirical mispricing at price extremes (Becker 2024).
2. **Arbitrage** — YES + NO orderbook violations of the $1 invariant.
3. **Microstructure** — order-book imbalance and spread analysis.
4. **Multi-choice arbitrage** — sum of YES prices across grouped markets ≠ $1.

Each method returns a Signal dataclass (or None) when an edge is detected.
The orchestrator ``scan_for_signals`` runs methods 1-3 on each token.
Method 4 operates on multi-choice groups detected by the Gamma API.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polymarket.client import PolymarketClient
    from .config import Config

logger = logging.getLogger(__name__)

# ── Longshot-bias correction table (Becker 2024, 72.1M Kalshi trades) ──
_BIAS_TABLE: list[tuple[float, float, float]] = [
    # (lo, hi, bias)  — negative = market overprices YES
    (0.00, 0.05, -0.008),
    (0.05, 0.15, -0.005),
    (0.15, 0.25, -0.003),
    (0.25, 0.50, -0.001),
    (0.50, 0.75, +0.001),
    (0.75, 0.85, +0.003),
    (0.85, 0.95, +0.005),
    (0.95, 1.00, +0.008),
]


@dataclass
class Signal:
    token_id: str
    side: str              # "BUY" or "SELL"
    fair_value: float      # Our estimated probability
    market_price: float
    edge: float            # fair_value - market_price (or inverse for SELL)
    method: str            # "longshot_bias" | "arbitrage" | "microstructure"
    confidence: float      # 0–1
    meta: dict = field(default_factory=dict)


# ── Method 1: Longshot bias ─────────────────────────────────────────────

def detect_longshot_bias(
    token_id: str, market_price: float, min_edge: float = 0.004,
) -> Signal | None:
    """Detect mispricing from longshot/favourite bias.

    Applies empirical correction from Becker 2024.  Returns a Signal when
    the bias-adjusted fair value diverges from market price by >= *min_edge*.
    Default min_edge=0.004 is below the max bias (0.008) to allow detection.
    """
    bias = 0.0
    for i, (lo, hi, b) in enumerate(_BIAS_TABLE):
        # Half-open intervals [lo, hi) except for the last row which is [lo, hi]
        if i == len(_BIAS_TABLE) - 1:
            if lo <= market_price <= hi:
                bias = b
                break
        else:
            if lo <= market_price < hi:
                bias = b
                break

    if bias == 0.0:
        return None

    fair_value = market_price - bias  # remove the bias
    edge = abs(fair_value - market_price)

    if edge < min_edge:
        return None

    # Negative bias means market overprices this side → SELL
    side = "SELL" if bias < 0 else "BUY"

    return Signal(
        token_id=token_id,
        side=side,
        fair_value=fair_value,
        market_price=market_price,
        edge=edge,
        method="longshot_bias",
        confidence=min(1.0, edge / 0.02),  # higher edge → more confident
        meta={"bias": bias},
    )


# ── Method 2: YES/NO arbitrage ──────────────────────────────────────────

def detect_arbitrage(
    book_yes: dict, book_no: dict, min_edge_bps: int = 20,
) -> Signal | None:
    """Detect arbitrage when YES + NO prices violate the $1 invariant.

    * best_ask_yes + best_ask_no < 1.0 → buy both = risk-free profit
    * best_bid_yes + best_bid_no > 1.0 → sell both = risk-free profit

    Requires >= *min_edge_bps* basis points after fees.
    """
    asks_yes = book_yes.get("asks") or []
    asks_no = book_no.get("asks") or []
    bids_yes = book_yes.get("bids") or []
    bids_no = book_no.get("bids") or []

    # Buy-both arbitrage
    if asks_yes and asks_no:
        best_ask_yes = float(asks_yes[0]["price"])
        best_ask_no = float(asks_no[0]["price"])
        cost = best_ask_yes + best_ask_no
        if cost < 1.0:
            edge = 1.0 - cost
            if edge * 10_000 >= min_edge_bps:
                # Convention: report the YES side token
                token_id = book_yes.get("asset_id", "")
                return Signal(
                    token_id=token_id,
                    side="BUY",
                    fair_value=best_ask_yes + edge / 2,
                    market_price=best_ask_yes,
                    edge=edge,
                    method="arbitrage",
                    confidence=min(1.0, edge / 0.01),
                    meta={
                        "type": "buy_both",
                        "ask_yes": best_ask_yes,
                        "ask_no": best_ask_no,
                        "cost": cost,
                    },
                )

    # Sell-both arbitrage
    if bids_yes and bids_no:
        best_bid_yes = float(bids_yes[0]["price"])
        best_bid_no = float(bids_no[0]["price"])
        revenue = best_bid_yes + best_bid_no
        if revenue > 1.0:
            edge = revenue - 1.0
            if edge * 10_000 >= min_edge_bps:
                token_id = book_yes.get("asset_id", "")
                return Signal(
                    token_id=token_id,
                    side="SELL",
                    fair_value=best_bid_yes - edge / 2,
                    market_price=best_bid_yes,
                    edge=edge,
                    method="arbitrage",
                    confidence=min(1.0, edge / 0.01),
                    meta={
                        "type": "sell_both",
                        "bid_yes": best_bid_yes,
                        "bid_no": best_bid_no,
                        "revenue": revenue,
                    },
                )

    return None


# ── Method 3: Microstructure ────────────────────────────────────────────

def detect_microstructure_edge(
    book: dict, imbalance_threshold: float = 0.30,
) -> Signal | None:
    """Detect edge from order-book imbalance and spread.

    Imbalance I = (depth_bid - depth_ask) / (depth_bid + depth_ask).
    |I| > threshold signals directional pressure.
    """
    bids = book.get("bids") or []
    asks = book.get("asks") or []

    if not bids or not asks:
        return None

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    if spread <= 0:
        return None

    depth_bid = sum(float(b.get("size", 0)) for b in bids[:5])
    depth_ask = sum(float(a.get("size", 0)) for a in asks[:5])
    total_depth = depth_bid + depth_ask

    if total_depth == 0:
        return None

    imbalance = (depth_bid - depth_ask) / total_depth
    kyle_lambda = spread / total_depth if total_depth > 0 else 0.0

    if abs(imbalance) < imbalance_threshold:
        return None

    # Imbalance > 0 → buying pressure → price likely to rise → BUY
    side = "BUY" if imbalance > 0 else "SELL"
    edge = abs(imbalance) * spread  # proxy for expected move

    token_id = book.get("asset_id", "")
    return Signal(
        token_id=token_id,
        side=side,
        fair_value=mid + (spread * imbalance / 2),
        market_price=mid,
        edge=edge,
        method="microstructure",
        confidence=min(1.0, abs(imbalance)),
        meta={
            "imbalance": round(imbalance, 4),
            "spread": round(spread, 4),
            "kyle_lambda": round(kyle_lambda, 6),
            "depth_bid": round(depth_bid, 2),
            "depth_ask": round(depth_ask, 2),
        },
    )


# ── Method 4: Multi-choice arbitrage ──────────────────────────────────

def detect_multi_choice_arbitrage(
    group,  # MultiChoiceGroup from gamma.py
    min_edge_bps: int = 20,
    fee_rate: float = 0.0,
) -> list[Signal]:
    """Detect arbitrage in multi-choice markets (Gamma API groups).

    Multi-choice events on Polymarket are N binary markets whose YES
    prices should sum to $1.00.  Deviations create arbitrage:

    * sum < 1.0 → buy all YES outcomes = guaranteed profit
    * sum > 1.0 → sell all YES (buy NOs) = guaranteed profit

    *fee_rate* is subtracted from edge_per_outcome (e.g. 0.02 for 2%).
    Returns a list of Signals (one per outcome to trade).
    """
    if abs(group.deviation) * 10_000 < min_edge_bps:
        return []

    n = len(group.markets)
    if n < 2:
        return []

    edge_per_outcome = abs(group.deviation) / n
    net_edge = edge_per_outcome - fee_rate
    if net_edge <= 0:
        return []
    edge_per_outcome = net_edge
    signals: list[Signal] = []

    if group.deviation < 0:
        # Sum < 1.0 → buy all YES tokens
        for m in group.markets:
            if not m.clob_token_ids or not m.outcome_prices:
                continue
            yes_token = m.clob_token_ids[0]  # first token = YES
            yes_price = m.outcome_prices[0]
            signals.append(Signal(
                token_id=yes_token,
                side="BUY",
                fair_value=yes_price + edge_per_outcome,
                market_price=yes_price,
                edge=edge_per_outcome,
                method="multi_choice_arb",
                confidence=min(1.0, abs(group.deviation) / 0.02),
                meta={
                    "type": "buy_all_yes",
                    "event_id": group.event_id,
                    "event_title": group.event_title,
                    "yes_sum": group.yes_sum,
                    "deviation": group.deviation,
                    "n_outcomes": n,
                    "group_item": m.group_item_title,
                    "question": m.question,
                },
            ))
    else:
        # Sum > 1.0 → sell all YES (buy NO tokens)
        for m in group.markets:
            if len(m.clob_token_ids) < 2 or not m.outcome_prices:
                continue
            no_token = m.clob_token_ids[1]  # second token = NO
            yes_price = m.outcome_prices[0]
            no_price = m.outcome_prices[1] if len(m.outcome_prices) > 1 else 1.0 - yes_price
            signals.append(Signal(
                token_id=no_token,
                side="BUY",
                fair_value=no_price + edge_per_outcome,
                market_price=no_price,
                edge=edge_per_outcome,
                method="multi_choice_arb",
                confidence=min(1.0, abs(group.deviation) / 0.02),
                meta={
                    "type": "buy_all_no",
                    "event_id": group.event_id,
                    "event_title": group.event_title,
                    "yes_sum": group.yes_sum,
                    "deviation": group.deviation,
                    "n_outcomes": n,
                    "group_item": m.group_item_title,
                    "question": m.question,
                },
            ))

    logger.info(
        "Multi-choice arb: %s — %d outcomes, yes_sum=%.4f, dev=%.4f, %d signals",
        group.event_title[:40], n, group.yes_sum, group.deviation, len(signals),
    )
    return signals


# ── Orchestrator ─────────────────────────────────────────────────────────

def scan_for_signals(
    client: PolymarketClient, token_ids: list[str], config: Config,
    multi_choice_groups: list | None = None,
    token_prices: dict[str, float] | None = None,
    token_pairs: dict[str, tuple[str, str]] | None = None,
) -> list[Signal]:
    """Run all detection methods on each token and return sorted signals.

    Methods 1-3 operate on individual tokens.
    Method 4 (multi-choice arbitrage) operates on groups from Gamma API.

    *token_prices* can supply pre-fetched prices (e.g. from Gamma API)
    to avoid redundant CLOB /price calls.

    *token_pairs* maps condition_id → (yes_token_id, no_token_id) for
    binary arbitrage detection (method 2).
    """
    signals: list[Signal] = []
    prices_cache = dict(token_prices or {})

    # Fetch price + book data in parallel for all tokens
    def _fetch_token_data(tid):
        price = prices_cache.get(tid)
        if price is None:
            try:
                price = float(client.get_price(tid).get("price", 0))
            except Exception:
                logger.debug("Failed to get price for %s, skipping", tid[:16])
                return tid, None, None
        book = None
        if config.microstructure:
            try:
                book = client.get_orderbook(tid)
                book["asset_id"] = tid
            except Exception:
                logger.debug("Failed to get book for %s, skipping", tid[:16])
        return tid, price, book

    with ThreadPoolExecutor(max_workers=config.parallel_workers) as pool:
        results = list(pool.map(_fetch_token_data, token_ids))

    for tid, price, book in results:
        if price is None:
            continue

        # Method 1: Longshot bias (uses separate threshold)
        if config.longshot_bias:
            sig = detect_longshot_bias(tid, price, min_edge=config.longshot_min_edge)
            if sig:
                signals.append(sig)

        # Method 3: Microstructure
        if config.microstructure and book is not None:
            sig = detect_microstructure_edge(
                book, imbalance_threshold=config.imbalance_threshold,
            )
            if sig and sig.edge >= config.min_ev_threshold:
                signals.append(sig)

    # Method 2: YES/NO arbitrage (via token_pairs)
    if config.arbitrage and token_pairs:
        for cid, (yes_tid, no_tid) in token_pairs.items():
            try:
                book_yes = client.get_orderbook(yes_tid)
                book_yes["asset_id"] = yes_tid
                book_no = client.get_orderbook(no_tid)
                book_no["asset_id"] = no_tid
                sig = detect_arbitrage(book_yes, book_no)
                if sig:
                    signals.append(sig)
            except Exception:
                logger.debug("Failed to get arb books for %s, skipping", cid[:16])

    # Method 4: Multi-choice arbitrage (from Gamma groups)
    fee_rate = config.polymarket_fee_bps / 10_000
    if config.multi_choice_arbitrage and multi_choice_groups:
        for group in multi_choice_groups:
            mc_signals = detect_multi_choice_arbitrage(group, fee_rate=fee_rate)
            signals.extend(mc_signals)

    # Deduplicate: keep highest edge*confidence per token_id
    best_by_token: dict[str, Signal] = {}
    for sig in signals:
        key = sig.token_id
        score = sig.edge * sig.confidence
        prev = best_by_token.get(key)
        if prev is None or score > prev.edge * prev.confidence:
            best_by_token[key] = sig
    signals = list(best_by_token.values())

    # Sort by edge * confidence descending
    signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)
    return signals
