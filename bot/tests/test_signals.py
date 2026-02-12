"""Tests for bot.signals — edge detection methods."""

import unittest

from bot.signals import (
    Signal,
    detect_arbitrage,
    detect_longshot_bias,
    detect_microstructure_edge,
    detect_multi_choice_arbitrage,
    scan_for_signals,
)
from bot.config import Config


class TestLongshotBias(unittest.TestCase):
    def test_low_price_bias(self):
        sig = detect_longshot_bias("tok1", 0.03, min_edge=0.005)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.method, "longshot_bias")
        self.assertEqual(sig.side, "SELL")  # bias < 0 → market overprices
        self.assertAlmostEqual(sig.edge, 0.008)

    def test_high_price_bias(self):
        sig = detect_longshot_bias("tok2", 0.97, min_edge=0.005)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, "BUY")  # bias > 0
        self.assertAlmostEqual(sig.edge, 0.008)

    def test_mid_price_no_bias(self):
        sig = detect_longshot_bias("tok3", 0.50)
        self.assertIsNone(sig)

    def test_below_threshold(self):
        sig = detect_longshot_bias("tok4", 0.20, min_edge=0.01)
        self.assertIsNone(sig)  # bias=0.003 < 0.01

    # ── Longshot min_edge fix tests ──

    def test_longshot_detects_with_low_min_edge(self):
        """Bug 2: With min_edge=0.005 (config.longshot_min_edge), longshot
        bias is actually detected at extremes where max bias = 0.008."""
        sig = detect_longshot_bias("tok5", 0.02, min_edge=0.005)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, "SELL")

    def test_longshot_blocked_with_default_03(self):
        """With the old min_edge=0.03, max bias 0.008 never triggers."""
        sig = detect_longshot_bias("tok6", 0.02, min_edge=0.03)
        self.assertIsNone(sig)

    def test_longshot_mid_range_low_bias(self):
        """Mid range (0.15-0.25) has bias 0.003, detectable at 0.005 but not 0.03."""
        sig_low = detect_longshot_bias("tok7", 0.20, min_edge=0.002)
        self.assertIsNotNone(sig_low)

        sig_high = detect_longshot_bias("tok8", 0.20, min_edge=0.005)
        self.assertIsNone(sig_high)  # 0.003 < 0.005


class TestArbitrage(unittest.TestCase):
    def _book(self, bid, ask, asset_id="yes"):
        return {
            "bids": [{"price": str(bid), "size": "100"}] if bid else [],
            "asks": [{"price": str(ask), "size": "100"}] if ask else [],
            "asset_id": asset_id,
        }

    def test_buy_both_arb(self):
        book_yes = self._book(bid=0.40, ask=0.45)
        book_no = self._book(bid=0.40, ask=0.50)
        # 0.45 + 0.50 = 0.95 < 1.0 → arb of $0.05
        sig = detect_arbitrage(book_yes, book_no, min_edge_bps=20)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.method, "arbitrage")
        self.assertEqual(sig.side, "BUY")
        self.assertAlmostEqual(sig.edge, 0.05)

    def test_sell_both_arb(self):
        book_yes = self._book(bid=0.55, ask=0.60)
        book_no = self._book(bid=0.50, ask=0.55)
        # 0.55 + 0.50 = 1.05 > 1.0 → arb of $0.05
        sig = detect_arbitrage(book_yes, book_no, min_edge_bps=20)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, "SELL")

    def test_no_arb(self):
        book_yes = self._book(bid=0.50, ask=0.55)
        book_no = self._book(bid=0.40, ask=0.50)
        # 0.55 + 0.50 = 1.05 (buy) — no buy arb
        # 0.50 + 0.40 = 0.90 (sell) — no sell arb
        sig = detect_arbitrage(book_yes, book_no, min_edge_bps=20)
        self.assertIsNone(sig)

    def test_empty_books(self):
        sig = detect_arbitrage({"bids": [], "asks": []}, {"bids": [], "asks": []})
        self.assertIsNone(sig)

    def test_below_min_edge(self):
        book_yes = self._book(bid=0.40, ask=0.50)
        book_no = self._book(bid=0.40, ask=0.499)
        # 0.50 + 0.499 = 0.999 → edge 0.001 = 10 bps < 20 bps
        sig = detect_arbitrage(book_yes, book_no, min_edge_bps=20)
        self.assertIsNone(sig)


class TestArbitrageWiring(unittest.TestCase):
    """Bug 3: Test that arbitrage is actually called from scan_for_signals
    when token_pairs are provided."""

    def test_arb_wired_in_scan(self):
        class FakeClient:
            def get_price(self, tid):
                return {"price": "0.50"}

            def get_orderbook(self, tid):
                if tid == "yes_tok":
                    return {
                        "asks": [{"price": "0.45", "size": "100"}],
                        "bids": [{"price": "0.40", "size": "100"}],
                        "asset_id": tid,
                    }
                elif tid == "no_tok":
                    return {
                        "asks": [{"price": "0.50", "size": "100"}],
                        "bids": [{"price": "0.45", "size": "100"}],
                        "asset_id": tid,
                    }
                return {"bids": [], "asks": []}

        config = Config(
            longshot_bias=False, microstructure=False,
            arbitrage=True, multi_choice_arbitrage=False,
            longshot_min_edge=0.005,
        )
        token_pairs = {"cid1": ("yes_tok", "no_tok")}
        # 0.45 + 0.50 = 0.95 → 5c arb
        signals = scan_for_signals(
            FakeClient(), [], config, token_pairs=token_pairs,
        )
        self.assertGreater(len(signals), 0)
        self.assertEqual(signals[0].method, "arbitrage")

    def test_arb_not_called_without_pairs(self):
        class FakeClient:
            def get_price(self, tid):
                return {"price": "0.50"}
            def get_orderbook(self, tid):
                return {"bids": [], "asks": []}

        config = Config(
            longshot_bias=False, microstructure=False,
            arbitrage=True, multi_choice_arbitrage=False,
            longshot_min_edge=0.005,
        )
        signals = scan_for_signals(FakeClient(), [], config)
        self.assertEqual(len(signals), 0)


class TestMicrostructureEdge(unittest.TestCase):
    def _book(self, bids, asks, asset_id="tok"):
        return {
            "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
            "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
            "asset_id": asset_id,
        }

    def test_buy_imbalance(self):
        # Heavy bids → expect price to rise → BUY
        book = self._book(
            bids=[(0.50, 100), (0.49, 80), (0.48, 60)],
            asks=[(0.52, 10), (0.53, 10)],
        )
        sig = detect_microstructure_edge(book, imbalance_threshold=0.3)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, "BUY")
        self.assertEqual(sig.method, "microstructure")

    def test_sell_imbalance(self):
        # Heavy asks → expect price to fall → SELL
        book = self._book(
            bids=[(0.50, 10)],
            asks=[(0.52, 100), (0.53, 80), (0.54, 60)],
        )
        sig = detect_microstructure_edge(book, imbalance_threshold=0.3)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, "SELL")

    def test_balanced_book(self):
        book = self._book(
            bids=[(0.50, 50)],
            asks=[(0.52, 50)],
        )
        sig = detect_microstructure_edge(book, imbalance_threshold=0.3)
        self.assertIsNone(sig)

    def test_empty_book(self):
        sig = detect_microstructure_edge({"bids": [], "asks": []})
        self.assertIsNone(sig)

    def test_meta_fields(self):
        book = self._book(
            bids=[(0.50, 200)],
            asks=[(0.52, 20)],
        )
        sig = detect_microstructure_edge(book, imbalance_threshold=0.3)
        self.assertIsNotNone(sig)
        self.assertIn("imbalance", sig.meta)
        self.assertIn("spread", sig.meta)
        self.assertIn("kyle_lambda", sig.meta)


class TestSignalDataclass(unittest.TestCase):
    def test_default_meta(self):
        sig = Signal(
            token_id="t", side="BUY", fair_value=0.6,
            market_price=0.5, edge=0.1, method="test", confidence=0.8,
        )
        self.assertEqual(sig.meta, {})
        self.assertEqual(sig.edge, 0.1)


if __name__ == "__main__":
    unittest.main()
