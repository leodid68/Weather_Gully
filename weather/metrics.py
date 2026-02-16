"""Forecast quality metrics — Brier score, Sharpe ratio, calibration table."""

import math


def brier_score(predictions: list[tuple[float, bool]]) -> float | None:
    """Brier score: mean((prob - outcome)^2). Lower is better. 0 = perfect."""
    if not predictions:
        return None
    return sum((p - (1.0 if o else 0.0)) ** 2 for p, o in predictions) / len(predictions)


def sharpe_ratio(returns: list[float], annualize_factor: float = 365.0) -> float | None:
    """Annualized Sharpe ratio from a list of per-trade returns."""
    if len(returns) < 2:
        return None
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 1e-10
    return (mean_r / std) * math.sqrt(annualize_factor)


def win_rate(returns: list[float]) -> float | None:
    """Fraction of trades with positive return."""
    if not returns:
        return None
    return sum(1 for r in returns if r > 0) / len(returns)


def average_edge(edges: list[float]) -> float | None:
    """Mean absolute edge |prob - market_price| on trades taken."""
    if not edges:
        return None
    return sum(abs(e) for e in edges) / len(edges)


def calibration_table(
    predictions: list[tuple[float, bool]],
    n_bins: int = 10,
) -> dict[str, dict]:
    """Bin predictions and compute actual frequency per bin.

    Returns dict mapping bin label (e.g. "0.1-0.2") to {"count": N, "actual_freq": float}.
    """
    bins: dict[str, list[bool]] = {}
    step = 1.0 / n_bins
    for prob, outcome in predictions:
        bin_idx = min(int(prob / step), n_bins - 1)
        lo = round(bin_idx * step, 1)
        hi = round((bin_idx + 1) * step, 1)
        key = f"{lo}-{hi}"
        bins.setdefault(key, []).append(outcome)

    table = {}
    for key, outcomes in sorted(bins.items()):
        n = len(outcomes)
        actual = sum(1 for o in outcomes if o) / n if n else 0
        table[key] = {"count": n, "actual_freq": round(actual, 4)}
    return table
