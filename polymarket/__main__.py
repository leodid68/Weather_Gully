"""Polymarket CLOB CLI — run with: python3 -m polymarket <command>"""

import argparse
import asyncio
import json
import logging
import os
import sys

from .ws import PolymarketWS

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _get_public_http():
    """Lightweight httpx client for public (no-auth) endpoints."""
    import httpx
    from .constants import CLOB_BASE_URL
    return httpx.Client(base_url=CLOB_BASE_URL, timeout=15)


def _get_client(args):
    """Full authenticated client — requires private key."""
    from .client import PolymarketClient

    key = args.private_key or os.environ.get("POLY_PRIVATE_KEY")
    if not key:
        print("Error: set POLY_PRIVATE_KEY env var or pass --private-key")
        sys.exit(1)

    creds = None
    creds_file = os.environ.get("POLY_CREDS_FILE")
    # Also check for creds.json in the project directory
    default_creds = os.path.join(os.path.dirname(__file__), "..", "creds.json")
    for path in [creds_file, default_creds]:
        if path and os.path.exists(path):
            with open(path) as f:
                creds = json.load(f)
            logger.info("Loaded API creds from %s", path)
            break

    return PolymarketClient(private_key=key, api_creds=creds)


# -- Commands ----------------------------------------------------------------

def cmd_markets(args):
    """List available markets (public, no auth needed)."""
    with _get_public_http() as http:
        params = f"?limit={args.limit}" if args.limit else ""
        resp = http.get(f"/markets{params}")
        resp.raise_for_status()
        markets = resp.json()
        if isinstance(markets, dict):
            markets = markets.get("data", markets.get("markets", []))
        for m in markets:
            qst = m.get("question", m.get("description", "?"))
            cid = m.get("condition_id", m.get("id", "?"))
            print(f"  {cid[:12]}...  {qst}")
        print(f"\n{len(markets)} market(s)")


def cmd_book(args):
    """Show orderbook for a token (public, no auth needed)."""
    with _get_public_http() as http:
        resp = http.get(f"/book?token_id={args.token_id}")
        resp.raise_for_status()
        book = resp.json()
        print("=== ASKS ===")
        for ask in reversed(book.get("asks", [])):
            print(f"  {float(ask['price']):>8.4f}  |  {ask['size']}")
        print("------------")
        for bid in book.get("bids", []):
            print(f"  {float(bid['price']):>8.4f}  |  {bid['size']}")
        print("=== BIDS ===")


def cmd_price(args):
    """Show current price for a token (public, no auth needed)."""
    with _get_public_http() as http:
        resp = http.get(f"/price?token_id={args.token_id}")
        resp.raise_for_status()
        p = resp.json()
        print(json.dumps(p, indent=2))


def cmd_orders(args):
    """List open orders."""
    with _get_client(args) as client:
        orders = client.get_open_orders()
        if not orders:
            print("No open orders.")
            return
        for o in orders:
            print(f"  {o.get('id', '?')}  {o.get('side', '?')}  "
                  f"price={o.get('price', '?')}  size={o.get('size', '?')}")
        print(f"\n{len(orders)} open order(s)")


def cmd_order(args):
    """Place a limit order."""
    with _get_client(args) as client:
        neg_risk = True if args.neg_risk else None  # None = auto-detect
        result = client.post_order(
            token_id=args.token_id,
            side=args.side.upper(),
            price=args.price,
            size=args.size,
            neg_risk=neg_risk,
        )
        print(json.dumps(result, indent=2))


def cmd_cancel(args):
    """Cancel an order or all orders."""
    with _get_client(args) as client:
        if args.order_id == "all":
            result = client.cancel_all()
        else:
            result = client.cancel_order(args.order_id)
        print(json.dumps(result, indent=2))


def cmd_trades(args):
    """Show trade history."""
    with _get_client(args) as client:
        trades = client.get_trades()
        if not trades:
            print("No trades.")
            return
        for t in trades:
            print(f"  {t.get('id', '?')}  {t.get('side', '?')}  "
                  f"price={t.get('price', '?')}  size={t.get('size', '?')}")
        print(f"\n{len(trades)} trade(s)")


def cmd_ws(args):
    """Subscribe to real-time market data via WebSocket."""
    def on_message(data):
        print(json.dumps(data))

    ws = PolymarketWS(on_message=on_message)

    async def run():
        for tid in args.token_ids:
            await ws.subscribe_market(tid)
            await ws.subscribe_price(tid)
            logger.info("Subscribed to %s", tid)
        await ws.run()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Stopped.")


def cmd_derive(args):
    """Derive API credentials from private key (L1 auth)."""
    from .auth import derive_api_key

    key = args.private_key or os.environ.get("POLY_PRIVATE_KEY")
    if not key:
        print("Error: set POLY_PRIVATE_KEY env var or pass --private-key")
        sys.exit(1)

    creds = derive_api_key(key)
    print(json.dumps(creds, indent=2))

    if args.save:
        with open(args.save, "w") as f:
            json.dump(creds, f, indent=2)
        print(f"\nSaved to {args.save}")


def cmd_approve(args):
    """Approve USDC.e spending for Polymarket exchange contracts."""
    from .approve import approve_exchanges

    key = args.private_key or os.environ.get("POLY_PRIVATE_KEY")
    if not key:
        print("Error: set POLY_PRIVATE_KEY env var or pass --private-key")
        sys.exit(1)

    approve_exchanges(key)


# -- CLI setup ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="python3 -m polymarket",
        description="Polymarket CLOB direct client",
    )
    parser.add_argument("--private-key", help="Ethereum private key (or set POLY_PRIVATE_KEY)")
    sub = parser.add_subparsers(dest="command", required=True)

    # markets
    p = sub.add_parser("markets", help="List markets")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_markets)

    # book
    p = sub.add_parser("book", help="Show orderbook")
    p.add_argument("token_id")
    p.set_defaults(func=cmd_book)

    # price
    p = sub.add_parser("price", help="Show price")
    p.add_argument("token_id")
    p.set_defaults(func=cmd_price)

    # orders
    p = sub.add_parser("orders", help="List open orders")
    p.set_defaults(func=cmd_orders)

    # order
    p = sub.add_parser("order", help="Place a limit order")
    p.add_argument("token_id")
    p.add_argument("side", choices=["buy", "sell", "BUY", "SELL"])
    p.add_argument("price", type=float)
    p.add_argument("size", type=float)
    p.add_argument("--neg-risk", action="store_true")
    p.set_defaults(func=cmd_order)

    # cancel
    p = sub.add_parser("cancel", help="Cancel order(s)")
    p.add_argument("order_id", help="Order ID or 'all'")
    p.set_defaults(func=cmd_cancel)

    # trades
    p = sub.add_parser("trades", help="Show trade history")
    p.set_defaults(func=cmd_trades)

    # ws
    p = sub.add_parser("ws", help="Stream real-time data via WebSocket")
    p.add_argument("token_ids", nargs="+", help="Token ID(s) to subscribe to")
    p.set_defaults(func=cmd_ws)

    # derive
    p = sub.add_parser("derive", help="Derive API creds from private key")
    p.add_argument("--save", metavar="FILE", help="Save creds to JSON file")
    p.set_defaults(func=cmd_derive)

    # approve
    p = sub.add_parser("approve", help="Approve USDC.e for exchange contracts")
    p.set_defaults(func=cmd_approve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
