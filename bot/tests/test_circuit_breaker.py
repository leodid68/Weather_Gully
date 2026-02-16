"""Tests for _CircuitBreaker thread safety."""
import threading
import unittest
from polymarket.client import _CircuitBreaker


class TestCircuitBreakerThreadSafety(unittest.TestCase):
    def test_concurrent_failures_open_circuit(self):
        cb = _CircuitBreaker(failure_threshold=10, recovery_timeout=60)
        threads = []
        for _ in range(20):
            t = threading.Thread(target=cb.record_failure)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # After 20 failures (threshold=10), should be OPEN
        self.assertEqual(cb.state, cb.OPEN)
        self.assertFalse(cb.allow_request())

    def test_concurrent_success_resets(self):
        cb = _CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        # Force to OPEN
        for _ in range(5):
            cb.record_failure()
        self.assertEqual(cb.state, cb.OPEN)
        # Simulate half-open probe success from multiple threads
        cb.state = cb.HALF_OPEN
        threads = [threading.Thread(target=cb.record_success) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(cb.state, cb.CLOSED)
        self.assertEqual(cb._failure_count, 0)
