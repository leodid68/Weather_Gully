"""Tests for inter-location forecast error correlation."""

import unittest
from weather.calibrate import _compute_correlation_matrix


class TestCorrelationMatrix(unittest.TestCase):

    def test_perfect_positive_correlation(self):
        errors = [
            {"location": "NYC", "target_date": "2025-01-15", "month": 1, "error": 2.0},
            {"location": "NYC", "target_date": "2025-01-16", "month": 1, "error": -1.0},
            {"location": "NYC", "target_date": "2025-01-17", "month": 1, "error": 3.0},
            {"location": "NYC", "target_date": "2025-01-18", "month": 1, "error": 0.5},
            {"location": "NYC", "target_date": "2025-01-19", "month": 1, "error": -2.0},
            {"location": "Chicago", "target_date": "2025-01-15", "month": 1, "error": 2.0},
            {"location": "Chicago", "target_date": "2025-01-16", "month": 1, "error": -1.0},
            {"location": "Chicago", "target_date": "2025-01-17", "month": 1, "error": 3.0},
            {"location": "Chicago", "target_date": "2025-01-18", "month": 1, "error": 0.5},
            {"location": "Chicago", "target_date": "2025-01-19", "month": 1, "error": -2.0},
        ]
        matrix = _compute_correlation_matrix(["NYC", "Chicago"], errors)
        key = "Chicago|NYC"
        self.assertIn(key, matrix)
        self.assertAlmostEqual(matrix[key]["DJF"], 1.0, places=2)

    def test_negative_correlation(self):
        errors = [
            {"location": "NYC", "target_date": "2025-07-01", "month": 7, "error": 1.0},
            {"location": "NYC", "target_date": "2025-07-02", "month": 7, "error": -1.0},
            {"location": "NYC", "target_date": "2025-07-03", "month": 7, "error": 1.0},
            {"location": "NYC", "target_date": "2025-07-04", "month": 7, "error": -1.0},
            {"location": "NYC", "target_date": "2025-07-05", "month": 7, "error": 1.0},
            {"location": "Miami", "target_date": "2025-07-01", "month": 7, "error": -1.0},
            {"location": "Miami", "target_date": "2025-07-02", "month": 7, "error": 1.0},
            {"location": "Miami", "target_date": "2025-07-03", "month": 7, "error": -1.0},
            {"location": "Miami", "target_date": "2025-07-04", "month": 7, "error": 1.0},
            {"location": "Miami", "target_date": "2025-07-05", "month": 7, "error": -1.0},
        ]
        matrix = _compute_correlation_matrix(["NYC", "Miami"], errors)
        key = "Miami|NYC"
        self.assertIn(key, matrix)
        self.assertAlmostEqual(matrix[key]["JJA"], -1.0, places=2)

    def test_insufficient_data_omitted(self):
        errors = [
            {"location": "NYC", "target_date": "2025-01-15", "month": 1, "error": 2.0},
            {"location": "Chicago", "target_date": "2025-01-15", "month": 1, "error": 1.0},
        ]
        matrix = _compute_correlation_matrix(["NYC", "Chicago"], errors)
        key = "Chicago|NYC"
        self.assertTrue(key not in matrix or "DJF" not in matrix.get(key, {}))

    def test_seasonal_grouping(self):
        errors = []
        for month in [1, 7]:
            for day in range(1, 11):
                date = f"2025-{month:02d}-{day:02d}"
                errors.append({"location": "NYC", "target_date": date, "month": month, "error": float(day)})
                errors.append({"location": "Chicago", "target_date": date, "month": month, "error": float(day) * 0.9})
        matrix = _compute_correlation_matrix(["NYC", "Chicago"], errors)
        key = "Chicago|NYC"
        self.assertIn(key, matrix)
        self.assertIn("DJF", matrix[key])
        self.assertIn("JJA", matrix[key])
