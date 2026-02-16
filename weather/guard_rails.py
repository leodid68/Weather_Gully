"""Parameter guard rails for calibration — prevents drift into unreasonable values."""

import copy
import logging

logger = logging.getLogger(__name__)

# (min, max) bounds for each parameter
PARAM_BOUNDS = {
    "base_sigma": (1.0, 4.0),
    "seasonal_factor": (0.5, 2.0),
    "model_weight": (0.0, 0.85),
    "platt_a": (0.3, 2.0),
    "platt_b": (-1.0, 1.0),
    "spread_to_sigma": (0.3, 1.5),
    "ema_to_sigma": (0.5, 2.0),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the range [lo, hi]."""
    return max(lo, min(hi, value))


def clamp_calibration(cal: dict) -> tuple[dict, list[dict]]:
    """Clamp calibration parameters within physical bounds.

    Returns a deep-copied calibration dict with out-of-range values clamped,
    and a list of dicts describing every clamped entry:
    ``{"param": name, "original": value, "clamped": value}``.
    """
    cal = copy.deepcopy(cal)
    clamped_list: list[dict] = []

    def _record(param: str, original: float, clamped: float) -> None:
        logger.warning(
            "Guard rail: %s clamped from %.4f to %.4f", param, original, clamped
        )
        clamped_list.append(
            {"param": param, "original": original, "clamped": clamped}
        )

    # --- base_sigma (metadata.base_sigma_global) + rescale global_sigma -----
    lo, hi = PARAM_BOUNDS["base_sigma"]
    base_sigma = cal.get("metadata", {}).get("base_sigma_global")
    if base_sigma is not None:
        new_base = _clamp(base_sigma, lo, hi)
        if new_base != base_sigma:
            ratio = new_base / base_sigma
            # Rescale global_sigma proportionally
            for key in cal.get("global_sigma", {}):
                cal["global_sigma"][key] = cal["global_sigma"][key] * ratio
            cal["metadata"]["base_sigma_global"] = new_base
            _record("base_sigma", base_sigma, new_base)

    # --- seasonal_factors ---------------------------------------------------
    lo, hi = PARAM_BOUNDS["seasonal_factor"]
    for month, value in cal.get("seasonal_factors", {}).items():
        new_val = _clamp(value, lo, hi)
        if new_val != value:
            cal["seasonal_factors"][month] = new_val
            _record("seasonal_factor", value, new_val)

    # --- location_seasonal --------------------------------------------------
    for loc, months in cal.get("location_seasonal", {}).items():
        for month, value in months.items():
            new_val = _clamp(value, lo, hi)
            if new_val != value:
                cal["location_seasonal"][loc][month] = new_val
                _record("seasonal_factor", value, new_val)

    # --- model_weights (skip noaa which is fixed at 0.20) -------------------
    lo, hi = PARAM_BOUNDS["model_weight"]
    for loc, weights in cal.get("model_weights", {}).items():
        for model in ("gfs_seamless", "ecmwf_ifs025"):
            value = weights.get(model)
            if value is None:
                continue
            new_val = _clamp(value, lo, hi)
            if new_val != value:
                cal["model_weights"][loc][model] = new_val
                _record("model_weight", value, new_val)

    # --- platt_scaling.a ----------------------------------------------------
    lo, hi = PARAM_BOUNDS["platt_a"]
    platt = cal.get("platt_scaling", {})
    if "a" in platt:
        value = platt["a"]
        new_val = _clamp(value, lo, hi)
        if new_val != value:
            cal["platt_scaling"]["a"] = new_val
            _record("platt_a", value, new_val)

    # --- platt_scaling.b ----------------------------------------------------
    lo, hi = PARAM_BOUNDS["platt_b"]
    if "b" in platt:
        value = platt["b"]
        new_val = _clamp(value, lo, hi)
        if new_val != value:
            cal["platt_scaling"]["b"] = new_val
            _record("platt_b", value, new_val)

    # --- adaptive_sigma.spread_to_sigma_factor ------------------------------
    lo, hi = PARAM_BOUNDS["spread_to_sigma"]
    adaptive = cal.get("adaptive_sigma", {})
    if "spread_to_sigma_factor" in adaptive:
        value = adaptive["spread_to_sigma_factor"]
        new_val = _clamp(value, lo, hi)
        if new_val != value:
            cal["adaptive_sigma"]["spread_to_sigma_factor"] = new_val
            _record("spread_to_sigma", value, new_val)

    # --- adaptive_sigma.ema_to_sigma_factor ---------------------------------
    lo, hi = PARAM_BOUNDS["ema_to_sigma"]
    if "ema_to_sigma_factor" in adaptive:
        value = adaptive["ema_to_sigma_factor"]
        new_val = _clamp(value, lo, hi)
        if new_val != value:
            cal["adaptive_sigma"]["ema_to_sigma_factor"] = new_val
            _record("ema_to_sigma", value, new_val)

    return cal, clamped_list
