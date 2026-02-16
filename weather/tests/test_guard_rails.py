"""Tests for weather.guard_rails — parameter guard rails for calibration."""

import copy
import unittest

from weather.guard_rails import PARAM_BOUNDS, clamp_calibration


def _make_calibration(**overrides):
    """Build a minimal valid calibration dict.

    Supports dotted keys for nested overrides, e.g.
    ``_make_calibration(**{"platt_scaling.a": 3.0})``.
    """
    cal = {
        "global_sigma": {
            "0": 1.84,
            "1": 2.45,
        },
        "location_sigma": {},
        "seasonal_factors": {
            "1": 1.078,
            "2": 1.024,
        },
        "location_seasonal": {
            "NYC": {"1": 1.012, "2": 0.917},
        },
        "model_weights": {
            "NYC": {
                "gfs_seamless": 0.44,
                "ecmwf_ifs025": 0.36,
                "noaa": 0.20,
            },
        },
        "adaptive_sigma": {
            "underdispersion_factor": 1.3,
            "spread_to_sigma_factor": 0.784,
            "ema_to_sigma_factor": 1.036,
            "samples": 728,
        },
        "platt_scaling": {
            "a": 0.7313,
            "b": 0.318,
        },
        "metadata": {
            "generated": "2026-02-16T00:00:00+00:00",
            "samples": 100,
            "base_sigma_global": 1.84,
        },
    }

    for key, value in overrides.items():
        parts = key.split(".")
        target = cal
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value

    return cal


class TestValidCalibrationUnchanged(unittest.TestCase):

    def test_valid_calibration_unchanged(self):
        cal = _make_calibration()
        original = copy.deepcopy(cal)
        clamped, clamped_list = clamp_calibration(cal)
        self.assertEqual(clamped_list, [])
        self.assertEqual(clamped, original)


class TestBaseSigmaClamping(unittest.TestCase):

    def test_base_sigma_too_low(self):
        cal = _make_calibration(**{"metadata.base_sigma_global": 0.3})
        # Set global_sigma proportionally low
        cal["global_sigma"] = {"0": 0.3, "1": 0.4}
        clamped, clamped_list = clamp_calibration(cal)
        self.assertEqual(clamped["metadata"]["base_sigma_global"], 1.0)
        # global_sigma should be rescaled proportionally (factor = 1.0 / 0.3)
        self.assertAlmostEqual(
            clamped["global_sigma"]["0"], 0.3 * (1.0 / 0.3), places=4
        )
        self.assertAlmostEqual(
            clamped["global_sigma"]["1"], 0.4 * (1.0 / 0.3), places=4
        )
        params_clamped = [e["param"] for e in clamped_list]
        self.assertIn("base_sigma", params_clamped)

    def test_base_sigma_too_high(self):
        cal = _make_calibration(**{"metadata.base_sigma_global": 5.0})
        cal["global_sigma"] = {"0": 5.0, "1": 6.65}
        clamped, clamped_list = clamp_calibration(cal)
        self.assertEqual(clamped["metadata"]["base_sigma_global"], 4.0)
        self.assertAlmostEqual(
            clamped["global_sigma"]["0"], 5.0 * (4.0 / 5.0), places=4
        )
        params_clamped = [e["param"] for e in clamped_list]
        self.assertIn("base_sigma", params_clamped)


class TestPlattClamping(unittest.TestCase):

    def test_platt_a_clamped(self):
        cal = _make_calibration(**{"platt_scaling.a": 3.0})
        clamped, clamped_list = clamp_calibration(cal)
        self.assertEqual(clamped["platt_scaling"]["a"], 2.0)
        entry = [e for e in clamped_list if e["param"] == "platt_a"][0]
        self.assertEqual(entry["original"], 3.0)
        self.assertEqual(entry["clamped"], 2.0)

    def test_platt_b_clamped(self):
        cal = _make_calibration(**{"platt_scaling.b": -2.0})
        clamped, clamped_list = clamp_calibration(cal)
        self.assertEqual(clamped["platt_scaling"]["b"], -1.0)
        entry = [e for e in clamped_list if e["param"] == "platt_b"][0]
        self.assertEqual(entry["original"], -2.0)
        self.assertEqual(entry["clamped"], -1.0)


class TestSeasonalFactorClamping(unittest.TestCase):

    def test_seasonal_factor_clamped(self):
        cal = _make_calibration()
        cal["seasonal_factors"]["1"] = 3.0
        clamped, clamped_list = clamp_calibration(cal)
        self.assertEqual(clamped["seasonal_factors"]["1"], 2.0)
        params_clamped = [e["param"] for e in clamped_list]
        self.assertIn("seasonal_factor", params_clamped)


class TestModelWeightClamping(unittest.TestCase):

    def test_model_weight_clamped(self):
        cal = _make_calibration()
        cal["model_weights"]["NYC"]["gfs_seamless"] = 0.90
        clamped, clamped_list = clamp_calibration(cal)
        self.assertEqual(clamped["model_weights"]["NYC"]["gfs_seamless"], 0.85)
        params_clamped = [e["param"] for e in clamped_list]
        self.assertIn("model_weight", params_clamped)


class TestAdaptiveSigmaClamping(unittest.TestCase):

    def test_spread_to_sigma_clamped(self):
        cal = _make_calibration(**{"adaptive_sigma.spread_to_sigma_factor": 0.1})
        clamped, clamped_list = clamp_calibration(cal)
        self.assertEqual(
            clamped["adaptive_sigma"]["spread_to_sigma_factor"], 0.3
        )
        params_clamped = [e["param"] for e in clamped_list]
        self.assertIn("spread_to_sigma", params_clamped)

    def test_ema_to_sigma_clamped(self):
        cal = _make_calibration(**{"adaptive_sigma.ema_to_sigma_factor": 2.5})
        clamped, clamped_list = clamp_calibration(cal)
        self.assertEqual(
            clamped["adaptive_sigma"]["ema_to_sigma_factor"], 2.0
        )
        params_clamped = [e["param"] for e in clamped_list]
        self.assertIn("ema_to_sigma", params_clamped)


if __name__ == "__main__":
    unittest.main()
