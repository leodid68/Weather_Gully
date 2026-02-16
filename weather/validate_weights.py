"""Cross-validation of ensemble weight methods.

Train on Jan-Sep 2025, test on Oct-Dec 2025.
Compares: (1) old inverse-RMSE, (2) new 2D grid search, (3) equal weights baseline.
"""

import logging
import math
import sys
from collections import defaultdict

from .calibrate import compute_forecast_errors, compute_model_weights
from .config import LOCATIONS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Old method: inverse-RMSE (reproduced from previous code)
# ---------------------------------------------------------------------------

def _inverse_rmse_weights(errors: list[dict], group_by: str = "location",
                          noaa_share: float = 0.20) -> dict[str, dict[str, float]]:
    """Original inverse-RMSE weighting (pre-grid-search)."""
    paired: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for err in errors:
        group_key = str(err.get(group_by, "global"))
        paired[group_key][err["model"]].append((err["forecast"], err["actual"]))

    results: dict[str, dict[str, float]] = {}
    for group_key, model_data in paired.items():
        model_rmse: dict[str, float] = {}
        for model, pairs in model_data.items():
            if not pairs:
                continue
            mse = sum((f - a) ** 2 for f, a in pairs) / len(pairs)
            model_rmse[model] = math.sqrt(mse)

        if not model_rmse:
            continue

        inv_rmse = {m: 1.0 / r for m, r in model_rmse.items() if r > 0}
        total_inv = sum(inv_rmse.values())
        if total_inv == 0:
            continue

        model_map = {"gfs": "gfs_seamless", "ecmwf": "ecmwf_ifs025"}
        weights: dict[str, float] = {}
        for model, inv in inv_rmse.items():
            canonical = model_map.get(model, model)
            weights[canonical] = inv / total_inv

        remaining = 1.0 - noaa_share
        for k in weights:
            weights[k] = round(weights[k] * remaining, 3)
        weights["noaa"] = noaa_share

        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 3) for k, v in weights.items()}

        results[group_key] = weights
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_weights(errors: list[dict], weights_by_loc: dict[str, dict[str, float]],
                      group_by: str = "location") -> dict[str, dict[str, float]]:
    """Evaluate a set of model weights on test errors.

    Returns per-group: {"mae": float, "n": int}.
    """
    # Align by (group, target_date, metric)
    aligned: dict[str, dict[tuple[str, str], dict[str, float]]] = defaultdict(dict)
    for err in errors:
        group_key = str(err.get(group_by, "global"))
        obs_key = (err["target_date"], err["metric"])
        if obs_key not in aligned[group_key]:
            aligned[group_key][obs_key] = {"actual": err["actual"]}
        aligned[group_key][obs_key][err["model"]] = err["forecast"]

    results: dict[str, dict[str, float]] = {}
    for group_key, observations in aligned.items():
        paired = [obs for obs in observations.values()
                  if "gfs" in obs and "ecmwf" in obs]
        if not paired:
            continue

        w = weights_by_loc.get(group_key, {"gfs_seamless": 0.5, "ecmwf_ifs025": 0.3, "noaa": 0.2})
        w_gfs = w.get("gfs_seamless", 0.4)
        w_ecmwf = w.get("ecmwf_ifs025", 0.4)
        w_noaa = w.get("noaa", 0.2)

        total_ae = 0.0
        for obs in paired:
            noaa_proxy = (obs["gfs"] + obs["ecmwf"]) / 2.0
            ens = w_gfs * obs["gfs"] + w_ecmwf * obs["ecmwf"] + w_noaa * noaa_proxy
            total_ae += abs(ens - obs["actual"])

        results[group_key] = {"mae": total_ae / len(paired), "n": len(paired)}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_start, train_end = "2025-01-01", "2025-09-30"
    test_start, test_end = "2025-10-01", "2025-12-31"

    locations = [
        "NYC", "Chicago", "Miami", "Seattle", "Atlanta", "Dallas",
        "London", "Paris", "Seoul", "Toronto",
        "BuenosAires", "SaoPaulo", "Ankara", "Wellington",
    ]

    # Fetch data
    logger.info("=" * 70)
    logger.info("CROSS-VALIDATION: Train Jan-Sep 2025 → Test Oct-Dec 2025")
    logger.info("=" * 70)

    all_train: list[dict] = []
    all_test: list[dict] = []

    for loc_name in locations:
        loc = LOCATIONS.get(loc_name)
        if not loc:
            logger.warning("Unknown location: %s", loc_name)
            continue

        logger.info("\nFetching %s...", loc_name)

        # Train set
        train_errors = compute_forecast_errors(
            loc_name, loc["lat"], loc["lon"], train_start, train_end,
            tz_name=loc.get("tz", "America/New_York"),
        )
        all_train.extend(train_errors)

        # Test set
        test_errors = compute_forecast_errors(
            loc_name, loc["lat"], loc["lon"], test_start, test_end,
            tz_name=loc.get("tz", "America/New_York"),
        )
        all_test.extend(test_errors)

    logger.info("\n%d train errors, %d test errors", len(all_train), len(all_test))

    # --- Method 1: Old inverse-RMSE (trained on train set) ---
    old_weights = _inverse_rmse_weights(all_train)

    # --- Method 2: New 2D grid search (trained on train set) ---
    new_weights = compute_model_weights(all_train, grid_step=0.01)

    # --- Method 3: Equal weights baseline ---
    equal_weights = {loc: {"gfs_seamless": 0.40, "ecmwf_ifs025": 0.40, "noaa": 0.20}
                     for loc in locations}

    # Evaluate all three on test set
    old_eval = _evaluate_weights(all_test, old_weights)
    new_eval = _evaluate_weights(all_test, new_weights)
    equal_eval = _evaluate_weights(all_test, equal_weights)

    # --- Report ---
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS — Out-of-sample MAE on Oct-Dec 2025")
    logger.info("=" * 70)

    logger.info("\n%-10s  %-35s  %-12s  %-12s  %-12s  %-8s",
                "Location", "Weights (GFS/ECMWF/NOAA)", "Equal", "Inv-RMSE", "Grid 2D", "Δ vs old")
    logger.info("-" * 95)

    total_old = total_new = total_equal = 0.0
    total_n = 0

    for loc in locations:
        o = old_eval.get(loc, {"mae": 0, "n": 0})
        n_eval = new_eval.get(loc, {"mae": 0, "n": 0})
        e = equal_eval.get(loc, {"mae": 0, "n": 0})

        ow = old_weights.get(loc, {})
        nw = new_weights.get(loc, {})

        old_str = f"G{ow.get('gfs_seamless', 0):.0%}/E{ow.get('ecmwf_ifs025', 0):.0%}/N{ow.get('noaa', 0):.0%}"
        new_str = f"G{nw.get('gfs_seamless', 0):.0%}/E{nw.get('ecmwf_ifs025', 0):.0%}/N{nw.get('noaa', 0):.0%}"

        delta = n_eval["mae"] - o["mae"]
        delta_str = f"{delta:+.3f}°F"

        logger.info("%-10s  old=%-15s new=%-15s  %7.3f°F    %7.3f°F    %7.3f°F    %s",
                     loc, old_str, new_str,
                     e["mae"], o["mae"], n_eval["mae"], delta_str)

        total_old += o["mae"] * o["n"]
        total_new += n_eval["mae"] * n_eval["n"]
        total_equal += e["mae"] * e["n"]
        total_n += o["n"]

    logger.info("-" * 95)
    if total_n > 0:
        avg_old = total_old / total_n
        avg_new = total_new / total_n
        avg_equal = total_equal / total_n
        delta_avg = avg_new - avg_old
        logger.info("%-10s  %-35s  %7.3f°F    %7.3f°F    %7.3f°F    %+.3f°F",
                     "AVERAGE", "", avg_equal, avg_old, avg_new, delta_avg)

        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info("  Equal weights MAE:      %.3f°F", avg_equal)
        logger.info("  Inverse-RMSE MAE:       %.3f°F", avg_old)
        logger.info("  Grid search 2D MAE:     %.3f°F", avg_new)
        logger.info("  Improvement vs old:     %+.3f°F (%.1f%%)",
                     delta_avg, (delta_avg / avg_old) * 100 if avg_old > 0 else 0)
        logger.info("  Improvement vs equal:   %+.3f°F (%.1f%%)",
                     avg_new - avg_equal,
                     ((avg_new - avg_equal) / avg_equal) * 100 if avg_equal > 0 else 0)


if __name__ == "__main__":
    main()
