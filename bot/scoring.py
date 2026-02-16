"""Proper scoring rules for prediction calibration tracking.

Implements Brier score, log score, calibration curves, and edge confidence
scoring.  Used to evaluate how well our probability estimates match reality
over time, following Gneiting 2007 strictly proper scoring rules.
"""

import math


def brier_score(predictions: list[float], outcomes: list[int]) -> float:
    """Mean Brier score: BS = (1/N) * sum((p_i - o_i)^2).

    Range [0, 1] — 0 is perfect.  Polymarket median is ~0.058 over 12h.
    """
    if not predictions:
        return float('nan')
    n = len(predictions)
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n


def log_score(predictions: list[float], outcomes: list[int]) -> float:
    """Mean logarithmic score: LS = (1/N) * sum(o*ln(p) + (1-o)*ln(1-p)).

    Penalises overconfidence at extremes more than Brier.
    Clips to [1e-10, 1-1e-10] to avoid log(0).
    """
    if not predictions:
        return float('nan')
    eps = 1e-10
    total = 0.0
    for p, o in zip(predictions, outcomes):
        p = max(eps, min(1.0 - eps, p))
        total += o * math.log(p) + (1 - o) * math.log(1.0 - p)
    return total / len(predictions)


def calibration_curve(
    predictions: list[float], outcomes: list[int], n_bins: int = 10,
) -> dict:
    """Group predictions into bins and compare predicted vs actual frequency.

    Returns dict with keys: bins, predicted, actual, count.
    """
    bins: list[str] = []
    predicted: list[float] = []
    actual: list[float] = []
    counts: list[int] = []

    step = 1.0 / n_bins
    for i in range(n_bins):
        lo = i * step
        hi = lo + step
        label = f"{lo:.2f}-{hi:.2f}"

        bucket_p = []
        bucket_o = []
        for p, o in zip(predictions, outcomes):
            if lo <= p < hi or (i == n_bins - 1 and p == hi):
                bucket_p.append(p)
                bucket_o.append(o)

        bins.append(label)
        counts.append(len(bucket_p))
        predicted.append(sum(bucket_p) / len(bucket_p) if bucket_p else 0.0)
        actual.append(sum(bucket_o) / len(bucket_o) if bucket_o else 0.0)

    return {"bins": bins, "predicted": predicted, "actual": actual, "count": counts}
