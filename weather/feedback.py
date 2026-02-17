"""Feedback loop — learn from resolved trades to correct systematic forecast bias.

Tracks per-location+season EMA of (forecast - actual) to detect and correct
persistent bias in the ensemble forecast.

State is stored separately in ``feedback_state.json`` — deleting the file resets.
"""

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 7       # Minimum resolved trades before applying correction
_ALPHA = 0.15          # EMA smoothing factor (slow learning)
_MIN_AR_SAMPLES = 10   # Minimum AR(1) pairs before applying autocorrelation correction

_HALF_LIFE_DAYS = 30.0   # Exponential decay half-life for feedback entries
_DECAY_FLOOR = 0.1       # Below this decay factor, treat as no data

_FEEDBACK_STATE_PATH = Path(__file__).parent / "feedback_state.json"

# Season mapping: month → season name
_SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}


def _season_key(location: str, month: int) -> str:
    """Build a feedback entry key from location and month."""
    season = _SEASON_MAP.get(month, "unknown")
    return f"{location}|{season}"


@dataclass
class FeedbackEntry:
    """EMA-based tracking of forecast bias for a location+season."""
    bias_ema: float = 0.0       # EMA of (forecast - actual) — positive = overpredict
    abs_error_ema: float = 0.0  # EMA of |forecast - actual|
    sample_count: int = 0
    last_updated: str = ""      # ISO timestamp of last update

    # AR(1) online estimator fields
    last_error: float | None = None   # Previous (forecast - actual)
    ar_phi: float = 0.0               # Running AR(1) coefficient
    cov_sum: float = 0.0              # Running sum of e(t)*e(t-1)
    var_sum: float = 0.0              # Running sum of e(t-1)^2
    ar_count: int = 0                 # Number of AR pairs observed

    def update(self, forecast_temp: float, actual_temp: float) -> None:
        """Incorporate one resolved trade's error into the EMA."""
        from datetime import datetime, timezone

        error = forecast_temp - actual_temp
        abs_error = abs(error)
        self.sample_count += 1
        self.last_updated = datetime.now(timezone.utc).isoformat()

        if self.sample_count == 1:
            self.bias_ema = error
            self.abs_error_ema = abs_error
        else:
            self.bias_ema = _ALPHA * error + (1 - _ALPHA) * self.bias_ema
            self.abs_error_ema = _ALPHA * abs_error + (1 - _ALPHA) * self.abs_error_ema

        # Online AR(1) coefficient
        if self.last_error is not None:
            self.cov_sum += error * self.last_error
            self.var_sum += self.last_error * self.last_error
            self.ar_count += 1
            if self.var_sum > 1e-10:
                raw_phi = self.cov_sum / self.var_sum
                self.ar_phi = max(-0.8, min(0.8, raw_phi))
            else:
                self.ar_phi = 0.0
        self.last_error = error

    def decay_factor(self) -> float:
        """Compute time-based decay factor (0.0 to 1.0).

        Returns 0.0 if ``last_updated`` is missing or unparseable.
        """
        if not self.last_updated:
            return 0.0
        try:
            from datetime import datetime, timezone

            last = datetime.fromisoformat(self.last_updated)
            days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
            return 0.5 ** (days / _HALF_LIFE_DAYS)
        except (ValueError, TypeError):
            return 0.0

    @property
    def has_enough_data(self) -> bool:
        return self.sample_count >= _MIN_SAMPLES


@dataclass
class FeedbackState:
    """Collection of feedback entries indexed by ``"location|season"``."""
    entries: dict[str, FeedbackEntry] = field(default_factory=dict)
    model_errors: dict[str, dict[str, float]] = field(default_factory=dict)  # location → {model: ema_error}

    def record_model_error(self, location: str, model: str, error: float) -> None:
        """Track per-model EMA error for dynamic weighting."""
        if location not in self.model_errors:
            self.model_errors[location] = {}
        prev = self.model_errors[location].get(model, abs(error))
        self.model_errors[location][model] = _ALPHA * abs(error) + (1 - _ALPHA) * prev

    def get_model_weights(self, location: str, wu_bonus: float = 0.0) -> dict[str, float] | None:
        """Return normalized weights based on 1/EMA_error^2. Returns None if insufficient data."""
        errors = self.model_errors.get(location, {})
        if len(errors) < 2:
            return None
        weights: dict[str, float] = {}
        for model, ema_err in errors.items():
            if ema_err > 0:
                weights[model] = 1.0 / (ema_err ** 2)
        if not weights:
            return None
        # Apply WU bonus
        if "wu" in weights and wu_bonus > 0:
            weights["wu"] *= (1.0 + wu_bonus)
        # Normalize
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()}

    def record(self, location: str, month: int, forecast_temp: float, actual_temp: float) -> None:
        """Record a resolved trade result."""
        key = _season_key(location, month)
        if key not in self.entries:
            self.entries[key] = FeedbackEntry()
        self.entries[key].update(forecast_temp, actual_temp)
        logger.debug("Feedback recorded for %s: error=%.1f°F (EMA bias=%.2f, n=%d)",
                      key, forecast_temp - actual_temp,
                      self.entries[key].bias_ema, self.entries[key].sample_count)

    def get_bias(self, location: str, month: int,
                 use_autocorrelation: bool = True) -> float | None:
        """Get the bias correction for a location+season.

        Returns the bias EMA (scaled by time-decay) if enough samples exist
        and the data is not too stale, else None.
        When *use_autocorrelation* is True and AR(1) data is sufficient,
        adds a correction term based on serial error correlation.
        Positive bias means forecast tends to overpredict → subtract from forecast.
        """
        key = _season_key(location, month)
        entry = self.entries.get(key)
        if entry and entry.has_enough_data:
            decay = entry.decay_factor()
            if decay < _DECAY_FLOOR:
                return None  # Data too old — treat as no data
            base_bias = entry.bias_ema * decay
            # AR(1) correction
            if (use_autocorrelation
                    and entry.last_error is not None
                    and entry.ar_count >= _MIN_AR_SAMPLES
                    and abs(entry.ar_phi) > 0.1):
                ar_correction = entry.ar_phi * entry.last_error * decay
                correction = base_bias + ar_correction
            else:
                correction = base_bias
            # Cap total correction to prevent extreme forecast shifts
            return max(-5.0, min(5.0, correction))
        return None

    def get_abs_error_ema(self, location: str, month: int) -> float | None:
        """Get the absolute error EMA for a location+season.

        Returns the abs-error EMA (scaled by time-decay) if enough samples
        exist and the data is not too stale, else None.
        Useful for adaptive sigma scaling.
        """
        key = _season_key(location, month)
        entry = self.entries.get(key)
        if entry and entry.has_enough_data:
            decay = entry.decay_factor()
            if decay < _DECAY_FLOOR:
                return None  # Data too old — treat as no data
            return entry.abs_error_ema * decay
        return None

    def save(self, path: str | None = None) -> None:
        """Persist feedback state to JSON (atomic write)."""
        save_path = Path(path) if path else _FEEDBACK_STATE_PATH
        data: dict = {}
        for key, entry in self.entries.items():
            data[key] = asdict(entry)
        # Wrap in envelope so we can store model_errors alongside entries
        envelope = {"entries": data}
        if self.model_errors:
            envelope["model_errors"] = self.model_errors
        fd, tmp = tempfile.mkstemp(dir=str(save_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(envelope, f, indent=2)
            os.replace(tmp, str(save_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.debug("Feedback state saved to %s (%d entries)", save_path, len(data))

    @classmethod
    def load(cls, path: str | None = None) -> "FeedbackState":
        """Load feedback state from JSON. Returns empty state if file missing.

        Handles both legacy (flat dict of entries) and new envelope format
        with ``entries`` + ``model_errors`` keys.
        """
        load_path = Path(path) if path else _FEEDBACK_STATE_PATH
        state = cls()
        if not load_path.exists():
            return state
        try:
            with open(load_path) as f:
                data = json.load(f)
            # Detect format: envelope has "entries" key, legacy is flat
            if "entries" in data and isinstance(data["entries"], dict):
                entries_data = data["entries"]
                state.model_errors = data.get("model_errors", {})
            else:
                entries_data = data  # Legacy flat format
            for key, entry_data in entries_data.items():
                state.entries[key] = FeedbackEntry(**entry_data)
            logger.info("Loaded feedback state: %d entries", len(state.entries))
        except (json.JSONDecodeError, IOError, TypeError) as exc:
            logger.warning("Failed to load feedback state: %s — starting fresh", exc)
        return state
