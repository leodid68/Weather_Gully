"""Gamma API client — rich market metadata and multi-choice detection.

The Gamma API (gamma-api.polymarket.com) provides richer data than the
CLOB API: volume, liquidity, bestBid/bestAsk, outcomePrices, event
grouping, and multi-choice market detection (negRisk markets).

Multi-choice markets on Polymarket are implemented as N binary markets
grouped under one event, each with negRisk=true.  The sum of all YES
prices across the group should equal ~$1.00.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


@dataclass
class GammaMarket:
    """Single market as returned by the Gamma API."""

    id: str
    question: str
    condition_id: str
    slug: str
    outcomes: list[str]
    outcome_prices: list[float]
    clob_token_ids: list[str]
    volume: float
    volume_24hr: float
    liquidity: float
    best_bid: float
    best_ask: float
    spread: float
    end_date: str
    active: bool
    closed: bool
    neg_risk: bool
    group_item_title: str
    event_id: str
    event_title: str


@dataclass
class MultiChoiceGroup:
    """A group of binary markets forming one multi-choice event."""

    event_id: str
    event_title: str
    markets: list[GammaMarket]
    yes_sum: float  # sum of YES outcome prices — should be ~1.0
    deviation: float  # yes_sum - 1.0 (positive = overpriced, negative = underpriced)


class GammaClient:
    """Read-only client for the Polymarket Gamma API."""

    def __init__(self, base_url: str = GAMMA_BASE_URL, timeout: float = 15):
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def fetch_markets(
        self,
        *,
        limit: int = 100,
        active: bool = True,
        closed: bool = False,
        neg_risk: bool | None = None,
        min_volume: float = 0,
        min_liquidity: float = 0,
    ) -> list[GammaMarket]:
        """Fetch markets from Gamma API with optional filters."""
        params: dict = {
            "limit": str(limit),
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": "volume24hr",
            "ascending": "false",
        }
        if neg_risk is not None:
            params["neg_risk"] = str(neg_risk).lower()

        resp = self._http.get("/markets", params=params)
        resp.raise_for_status()
        raw = resp.json()

        markets: list[GammaMarket] = []
        for m in raw:
            gm = _parse_market(m)
            if gm.volume < min_volume:
                continue
            if gm.liquidity < min_liquidity:
                continue
            markets.append(gm)

        logger.info(
            "Gamma: fetched %d markets (from %d raw, min_vol=%.0f, min_liq=%.0f)",
            len(markets), len(raw), min_volume, min_liquidity,
        )
        return markets

    def fetch_events(
        self,
        *,
        limit: int = 50,
        active: bool = True,
        closed: bool = False,
        tag_slug: str | None = None,
    ) -> list[dict]:
        """Fetch events (groups of related markets).

        Use *tag_slug* to filter by tag (e.g. ``"weather"``).
        Note: the ``tag=`` parameter is broken in the Gamma API — it gets
        ignored and returns default results.  ``tag_slug`` is the correct
        filter parameter.
        """
        params: dict = {
            "limit": str(limit),
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": "volume",
            "ascending": "false",
        }
        if tag_slug:
            params["tag_slug"] = tag_slug
        resp = self._http.get("/events", params=params)
        resp.raise_for_status()
        return resp.json()

    def fetch_weather_events(self, *, limit: int = 100) -> list[dict]:
        """Fetch active weather/temperature events using tag_slug=weather.

        Daily temperature markets have tags ['Weather', 'Recurring', 'Hide From New']
        and are NOT returned by standard search/tag parameters.
        """
        return self.fetch_events(limit=limit, tag_slug="weather")

    def fetch_events_with_markets(
        self,
        *,
        tag_slug: str,
        limit: int = 100,
    ) -> tuple[list[dict], list[GammaMarket]]:
        """Fetch events by tag_slug and parse embedded markets.

        The ``/events`` endpoint embeds full market data in each event,
        which is the only reliable way to get markets since the
        ``event_id`` filter on ``/markets`` is broken.

        Returns (events_raw, all_gamma_markets).
        """
        events = self.fetch_events(limit=limit, tag_slug=tag_slug)
        all_markets: list[GammaMarket] = []
        for ev in events:
            event_id = str(ev.get("id", ""))
            event_title = ev.get("title", "")
            raw_markets = ev.get("markets") or []
            for m in raw_markets:
                gm = _parse_market(m)
                gm.event_id = event_id
                gm.event_title = event_title
                all_markets.append(gm)
        logger.info(
            "Fetched %d markets from %d events (tag_slug=%s)",
            len(all_markets), len(events), tag_slug,
        )
        return events, all_markets

    def fetch_event_markets(self, event_id: str, event_title: str = "") -> list[GammaMarket]:
        """Fetch ALL markets for a specific event (complete multi-choice group).

        The Gamma API's nested ``events`` field is unreliable (returns
        unrelated events), so we override ``event_id`` and ``event_title``
        on the parsed markets with the known values.
        """
        resp = self._http.get("/markets", params={
            "event_id": event_id,
            "closed": "false",
            "limit": "200",
        })
        resp.raise_for_status()
        raw = resp.json()
        markets = [_parse_market(m) for m in raw]
        # Override event_id/title since the API's nested events field is broken
        for gm in markets:
            gm.event_id = event_id
            if event_title:
                gm.event_title = event_title
        return markets

    def check_resolution(self, condition_id: str) -> dict | None:
        """Check if a market is resolved via Gamma API.

        Returns {"resolved": True, "outcome": True/False} if the market
        is closed and has a winning outcome, or None if still open.
        """
        try:
            resp = self._http.get("/markets", params={
                "conditionId": condition_id,
                "limit": "1",
            })
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                return None
            m = raw[0]
            if not m.get("closed", False):
                return None
            # Determine winning outcome from outcome prices
            # A resolved market has one outcome at ~1.0 and others at ~0.0
            outcome_prices_raw = m.get("outcomePrices") or "[]"
            if isinstance(outcome_prices_raw, str):
                import json
                try:
                    prices = [float(p) for p in json.loads(outcome_prices_raw)]
                except (json.JSONDecodeError, ValueError):
                    return None
            elif isinstance(outcome_prices_raw, list):
                prices = [float(p) for p in outcome_prices_raw]
            else:
                return None
            if not prices:
                return None
            # YES wins if first outcome price >= 0.95
            yes_won = prices[0] >= 0.95
            return {"resolved": True, "outcome": yes_won}
        except Exception as exc:
            logger.debug("check_resolution(%s) failed: %s", condition_id[:16], exc)
            return None

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _parse_market(m: dict) -> GammaMarket:
    """Parse a raw Gamma API market dict into a GammaMarket."""
    events = m.get("events") or []
    event = events[0] if events else {}

    outcome_prices_raw = m.get("outcomePrices") or "[]"
    if isinstance(outcome_prices_raw, str):
        import json
        try:
            outcome_prices = [float(p) for p in json.loads(outcome_prices_raw)]
        except (json.JSONDecodeError, ValueError):
            outcome_prices = []
    elif isinstance(outcome_prices_raw, list):
        outcome_prices = [float(p) for p in outcome_prices_raw]
    else:
        outcome_prices = []

    clob_ids_raw = m.get("clobTokenIds") or "[]"
    if isinstance(clob_ids_raw, str):
        import json
        try:
            clob_ids = json.loads(clob_ids_raw)
        except json.JSONDecodeError:
            clob_ids = []
    elif isinstance(clob_ids_raw, list):
        clob_ids = list(clob_ids_raw)
    else:
        clob_ids = []

    outcomes_raw = m.get("outcomes") or "[]"
    if isinstance(outcomes_raw, str):
        import json
        try:
            outcomes = json.loads(outcomes_raw)
        except json.JSONDecodeError:
            outcomes = []
    elif isinstance(outcomes_raw, list):
        outcomes = list(outcomes_raw)
    else:
        outcomes = []

    best_bid = float(m.get("bestBid") or 0)
    best_ask = float(m.get("bestAsk") or 0)

    return GammaMarket(
        id=str(m.get("id", "")),
        question=m.get("question", ""),
        condition_id=m.get("conditionId", ""),
        slug=m.get("slug", ""),
        outcomes=outcomes,
        outcome_prices=outcome_prices,
        clob_token_ids=clob_ids,
        volume=float(m.get("volume") or 0),
        volume_24hr=float(m.get("volume24hr") or 0),
        liquidity=float(m.get("liquidity") or 0),
        best_bid=best_bid,
        best_ask=best_ask,
        spread=round(best_ask - best_bid, 4) if best_ask > best_bid else 0.0,
        end_date=m.get("endDate", ""),
        active=bool(m.get("active", True)),
        closed=bool(m.get("closed", False)),
        neg_risk=bool(m.get("negRisk", False)),
        group_item_title=m.get("groupItemTitle", ""),
        event_id=str(event.get("id", "")),
        event_title=event.get("title", ""),
    )


def group_multi_choice(
    markets: list[GammaMarket],
    gamma_client: GammaClient | None = None,
    min_yes_sum: float = 0.80,
    max_yes_sum: float = 1.20,
) -> list[MultiChoiceGroup]:
    """Group negRisk markets by event and compute YES price sums.

    Multi-choice events on Polymarket consist of N binary markets
    where the sum of all YES prices should ≈ 1.0.  Deviation from
    1.0 indicates a potential arbitrage opportunity.

    Groups with yes_sum outside [min_yes_sum, max_yes_sum] are treated
    as incomplete (missing outcomes in our scan) and skipped.

    If *gamma_client* is provided and a group's yes_sum is close but
    below *min_yes_sum*, attempts to fetch the full event.
    """
    # Only negRisk markets participate in multi-choice groups
    neg_risk = [m for m in markets if m.neg_risk and m.event_id]

    # Group by event_id
    by_event: dict[str, list[GammaMarket]] = {}
    for m in neg_risk:
        by_event.setdefault(m.event_id, []).append(m)

    groups: list[MultiChoiceGroup] = []
    for event_id, event_markets in by_event.items():
        if len(event_markets) < 2:
            continue  # Need at least 2 outcomes for multi-choice

        # YES price is the first outcome price (index 0)
        yes_sum = sum(
            m.outcome_prices[0] for m in event_markets if m.outcome_prices
        )

        # If close to valid range but just below, try completing
        if 0.5 <= yes_sum < min_yes_sum and gamma_client is not None:
            try:
                full_markets = gamma_client.fetch_event_markets(event_id)
                full_neg = [m for m in full_markets if m.neg_risk]
                full_sum = sum(
                    m.outcome_prices[0] for m in full_neg if m.outcome_prices
                )
                # Only use completed group if it yields a valid sum
                if min_yes_sum <= full_sum <= max_yes_sum:
                    event_markets = full_neg
                    yes_sum = full_sum
                    logger.debug(
                        "Completed group %s: %d → %d markets, yes_sum=%.4f",
                        event_id, len(by_event[event_id]),
                        len(event_markets), yes_sum,
                    )
            except Exception as exc:
                logger.debug("Failed to fetch full event %s: %s", event_id, exc)

        # Skip incomplete groups (far from 1.0 → missing outcomes)
        if yes_sum < min_yes_sum or yes_sum > max_yes_sum:
            logger.debug(
                "Skipping incomplete group %s (%s): yes_sum=%.3f, n=%d",
                event_id, event_markets[0].event_title[:30], yes_sum, len(event_markets),
            )
            continue

        deviation = yes_sum - 1.0

        groups.append(MultiChoiceGroup(
            event_id=event_id,
            event_title=event_markets[0].event_title,
            markets=event_markets,
            yes_sum=round(yes_sum, 4),
            deviation=round(deviation, 4),
        ))

    groups.sort(key=lambda g: abs(g.deviation), reverse=True)
    logger.info(
        "Gamma: found %d valid multi-choice groups from %d negRisk markets",
        len(groups), len(neg_risk),
    )
    return groups


def gamma_to_scanner_format(markets: list[GammaMarket]) -> list[dict]:
    """Convert GammaMarkets to the dict format used by scanner/strategy.

    This bridges the Gamma API data into the existing pipeline that
    expects dicts with condition_id, question, tokens, etc.
    """
    result = []
    for gm in markets:
        tokens = []
        for i, clob_id in enumerate(gm.clob_token_ids):
            outcome = gm.outcomes[i] if i < len(gm.outcomes) else f"outcome_{i}"
            price = gm.outcome_prices[i] if i < len(gm.outcome_prices) else 0.0
            tokens.append({
                "token_id": clob_id,
                "outcome": outcome,
                "price": price,
            })

        # Compute a liquidity grade from Gamma's bestBid/bestAsk
        spread_bps = (gm.spread / ((gm.best_bid + gm.best_ask) / 2) * 10_000
                      if gm.best_bid + gm.best_ask > 0 else 10_000)
        if spread_bps < 50:
            grade = "A"
        elif spread_bps < 100:
            grade = "B"
        elif spread_bps < 200:
            grade = "C"
        else:
            grade = "D"

        result.append({
            "condition_id": gm.condition_id,
            "question": gm.question,
            "tokens": tokens,
            "active": gm.active,
            "neg_risk": gm.neg_risk,
            "end_date_iso": gm.end_date,
            "liquidity_grade": grade,
            # Gamma-specific enrichment
            "gamma": {
                "id": gm.id,
                "volume": gm.volume,
                "volume_24hr": gm.volume_24hr,
                "liquidity": gm.liquidity,
                "best_bid": gm.best_bid,
                "best_ask": gm.best_ask,
                "spread": gm.spread,
                "event_id": gm.event_id,
                "event_title": gm.event_title,
                "group_item_title": gm.group_item_title,
            },
        })
    return result


def resolve_pending_predictions(state, gamma_client: GammaClient) -> int:
    """Iterate over unresolved predictions and resolve via Gamma API.

    Returns the number of newly resolved predictions.
    """
    from .state import TradingState

    resolved_count = 0
    for market_id, pred in list(state.predictions.items()):
        if pred.get("resolved"):
            continue
        result = gamma_client.check_resolution(market_id)
        if result and result.get("resolved"):
            state.resolve_prediction(market_id, result["outcome"])
            resolved_count += 1
            logger.info(
                "Resolved prediction %s: outcome=%s",
                market_id[:16], result["outcome"],
            )

    if resolved_count:
        logger.info("Resolved %d pending predictions", resolved_count)
    return resolved_count
