"""Tests for sigma signal logging."""

import json
import os
import tempfile
import unittest

from weather.sigma_log import log_sigma_signals, load_sigma_log


class TestSigmaLog(unittest.TestCase):

    def test_log_and_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)  # Start fresh
        try:
            log_sigma_signals(
                path=path, location="NYC", date="2026-02-15", metric="high",
                ensemble_stddev=2.5, model_spread=1.8, ema_error=2.1,
                final_sigma=3.25, forecast_temp=52.0,
            )
            entries = load_sigma_log(path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["location"], "NYC")
            self.assertAlmostEqual(entries[0]["ensemble_stddev"], 2.5)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_appends_multiple_entries(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            for i in range(3):
                log_sigma_signals(
                    path=path, location="NYC", date=f"2026-02-{15+i}",
                    metric="high", ensemble_stddev=2.0 + i, model_spread=1.0,
                    ema_error=1.5, final_sigma=2.5 + i, forecast_temp=50.0,
                )
            entries = load_sigma_log(path)
            self.assertEqual(len(entries), 3)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_missing_file(self):
        entries = load_sigma_log("/tmp/nonexistent_sigma_log_test.json")
        self.assertEqual(entries, [])

    def test_load_corrupted_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            path = f.name
        try:
            entries = load_sigma_log(path)
            self.assertEqual(entries, [])
        finally:
            os.unlink(path)
