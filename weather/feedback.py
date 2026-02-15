"""Feedback loop — learn from resolved trades to correct systematic forecast bias.

Tracks per-location+season EMA of (forecast - actual) to detect and correct
persistent bias in the ensemble forecast.

State is stored separately in ``feedback_state.json`` — deleting the file resets.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 7       # Minimum resolved trades before applying correction
_ALPHA = 0.15          # EMA smoothing factor (slow learning)

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

    def update(self, forecast_temp: float, actual_temp: float) -> None:
        """Incorporate one resolved trade's error into the EMA."""
        error = forecast_temp - actual_temp
        abs_error = abs(error)
        self.sample_count += 1

        if self.sample_count == 1:
            self.bias_ema = error
            self.abs_error_ema = abs_error
        else:
            self.bias_ema = _ALPHA * error + (1 - _ALPHA) * self.bias_ema
            self.abs_error_ema = _ALPHA * abs_error + (1 - _ALPHA) * self.abs_error_ema

    @property
    def has_enough_data(self) -> bool:
        return self.sample_count >= _MIN_SAMPLES


@dataclass
class FeedbackState:
    """Collection of feedback entries indexed by ``"location|season"``."""
    entries: dict[str, FeedbackEntry] = field(default_factory=dict)

    def record(self, location: str, month: int, forecast_temp: float, actual_temp: float) -> None:
        """Record a resolved trade result."""
        key = _season_key(location, month)
        if key not in self.entries:
            self.entries[key] = FeedbackEntry()
        self.entries[key].update(forecast_temp, actual_temp)
        logger.debug("Feedback recorded for %s: error=%.1f°F (EMA bias=%.2f, n=%d)",
                      key, forecast_temp - actual_temp,
                      self.entries[key].bias_ema, self.entries[key].sample_count)

    def get_bias(self, location: str, month: int) -> float | None:
        """Get the bias correction for a location+season.

        Returns the bias EMA if enough samples exist, else None.
        Positive bias means forecast tends to overpredict → subtract from forecast.
        """
        key = _season_key(location, month)
        entry = self.entries.get(key)
        if entry and entry.has_enough_data:
            return entry.bias_ema
        return None

    def save(self, path: str | None = None) -> None:
        """Persist feedback state to JSON."""
        save_path = Path(path) if path else _FEEDBACK_STATE_PATH
        data = {}
        for key, entry in self.entries.items():
            data[key] = asdict(entry)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Feedback state saved to %s (%d entries)", save_path, len(data))

    @classmethod
    def load(cls, path: str | None = None) -> "FeedbackState":
        """Load feedback state from JSON. Returns empty state if file missing."""
        load_path = Path(path) if path else _FEEDBACK_STATE_PATH
        state = cls()
        if not load_path.exists():
            return state
        try:
            with open(load_path) as f:
                data = json.load(f)
            for key, entry_data in data.items():
                state.entries[key] = FeedbackEntry(**entry_data)
            logger.info("Loaded feedback state: %d entries", len(state.entries))
        except (json.JSONDecodeError, IOError, TypeError) as exc:
            logger.warning("Failed to load feedback state: %s — starting fresh", exc)
        return state
