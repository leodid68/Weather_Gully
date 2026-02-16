"""Validate distribution choices against historical forecast errors.

Compares Normal, Student-t(10), Student-t(df_best), and Skew-t(df, gamma)
on Brier score and log-loss using the error cache.

Usage::

    python -m weather.distribution_validation
"""

import json
import logging
import math
import sys
from pathlib import Path

from .calibrate import _fit_student_t_df, _fit_skew_t_params, _test_normality
from .probability import _normal_cdf, _student_t_cdf, _skew_t_cdf

logger = logging.getLogger(__name__)

_ERROR_CACHE_PATH = Path(__file__).parent / "error_history.json"


def _load_errors() -> list[dict]:
    """Load errors from cache."""
    if not _ERROR_CACHE_PATH.exists():
        logger.error("Error cache not found at %s", _ERROR_CACHE_PATH)
        return []
    with open(_ERROR_CACHE_PATH) as f:
        data = json.load(f)
    return data.get("errors", [])


def _bucket_hit(error: float, sigma: float, bucket_low: float, bucket_high: float,
                cdf_fn) -> float:
    """Compute P(actual in bucket) using the given CDF function and sigma.

    Simulates a bucket centered on 0 (the forecast). The error is the actual
    minus forecast, so error=0 means perfect forecast.
    """
    if sigma <= 0:
        sigma = 0.01

    if bucket_low <= -900:
        cdf_low = 0.0
    else:
        cdf_low = cdf_fn((bucket_low - 0.5) / sigma)

    if bucket_high >= 900:
        cdf_high = 1.0
    else:
        cdf_high = cdf_fn((bucket_high + 0.5) / sigma)

    return max(0.001, min(0.999, cdf_high - cdf_low))


def score_distribution(errors: list[float], cdf_fn, sigma: float,
                       bucket_width: float = 5.0) -> dict:
    """Score a distribution using Brier score and log-loss.

    Simulates temperature buckets centered on 0 (the forecast) with
    given width. For each error, checks which bucket the actual fell in
    and computes the probability assigned to that bucket.

    Returns dict with 'brier', 'log_loss', 'n'.
    """
    brier_sum = 0.0
    ll_sum = 0.0
    n = 0

    for error in errors:
        # Which bucket does this error fall in?
        bucket_idx = round(error / bucket_width)
        bucket_low = bucket_idx * bucket_width - bucket_width / 2
        bucket_high = bucket_idx * bucket_width + bucket_width / 2

        prob = _bucket_hit(error, sigma, bucket_low, bucket_high, cdf_fn)

        # Brier: (1 - prob)^2  (the event happened, so outcome = 1)
        brier_sum += (1.0 - prob) ** 2
        # Log-loss: -log(prob)
        ll_sum += -math.log(max(1e-10, prob))
        n += 1

    if n == 0:
        return {"brier": 1.0, "log_loss": 10.0, "n": 0}

    return {
        "brier": round(brier_sum / n, 6),
        "log_loss": round(ll_sum / n, 6),
        "n": n,
    }


def run_validation() -> dict:
    """Run distribution comparison and return results dict."""
    errors_raw = _load_errors()
    if not errors_raw:
        logger.error("No errors to validate against")
        return {}

    error_values = [e["error"] for e in errors_raw if "error" in e]
    if len(error_values) < 50:
        logger.warning("Only %d errors — results may be unreliable", len(error_values))
    if not error_values:
        return {}

    sigma = (sum(e ** 2 for e in error_values) / len(error_values)) ** 0.5

    best_df = _fit_student_t_df(error_values)
    skew_df, skew_gamma = _fit_skew_t_params(error_values)
    normality = _test_normality(error_values)

    distributions = {
        "Normal": _normal_cdf,
        "Student-t(10)": lambda z: _student_t_cdf(z, 10),
        f"Student-t({best_df:.0f})": lambda z, df=best_df: _student_t_cdf(z, df),
        f"Skew-t(df={skew_df:.0f}, g={skew_gamma:.1f})": lambda z, df=skew_df, g=skew_gamma: _skew_t_cdf(z, df, g),
    }

    results = {}
    for name, cdf_fn in distributions.items():
        scores = score_distribution(error_values, cdf_fn, sigma)
        results[name] = scores

    return {
        "n_errors": len(error_values),
        "sigma": round(sigma, 3),
        "normality": normality,
        "fitted_df": best_df,
        "fitted_skew_df": skew_df,
        "fitted_gamma": skew_gamma,
        "scores": results,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = run_validation()
    if not results:
        logger.error("Validation failed — no data")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("DISTRIBUTION VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info("Errors: %d   Sigma: %.3f degF", results["n_errors"], results["sigma"])
    logger.info("Normality: JB=%.1f skew=%.3f kurt=%.3f -> %s",
                results["normality"]["jb_statistic"],
                results["normality"]["skewness"],
                results["normality"]["kurtosis"],
                "normal" if results["normality"]["normal"] else "NON-NORMAL")
    logger.info("Fitted: Student-t df=%.0f  |  Skew-t df=%.0f gamma=%.2f",
                results["fitted_df"], results["fitted_skew_df"], results["fitted_gamma"])
    logger.info("")
    logger.info("%-35s  %10s  %10s  %6s", "Distribution", "Brier", "Log-Loss", "N")
    logger.info("-" * 70)

    best_brier = min(s["brier"] for s in results["scores"].values())

    for name, scores in results["scores"].items():
        marker = " <- best" if scores["brier"] == best_brier else ""
        logger.info("%-35s  %10.6f  %10.6f  %6d%s",
                    name, scores["brier"], scores["log_loss"], scores["n"], marker)

    logger.info("=" * 70)
    best_name = min(results["scores"], key=lambda k: results["scores"][k]["brier"])
    logger.info("RECOMMENDATION: Use %s (lowest Brier score)", best_name)


if __name__ == "__main__":
    main()
