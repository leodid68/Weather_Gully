"""Scalar Kalman filter for dynamic sigma estimation per location+horizon.

Learns the true forecast error sigma from resolved trades and blends
with the existing max-of-4-signals adaptive sigma approach.  The blend
weight ramps from 0 to 0.5 as the sample count grows from 5 to 30.

State is persisted in ``kalman_state.json`` — deleting the file resets.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "kalman_state.json"

# sqrt(2/pi) ≈ 0.7979 — E[|N(0,σ)|] = σ * sqrt(2/π)
_SQRT_2_OVER_PI = 0.7979

# Horizon day → bucket name (pool data for faster convergence)
_HORIZON_BUCKETS = {
    0: "short", 1: "short",
    2: "medium", 3: "medium", 4: "medium",
    5: "long", 6: "long", 7: "long",
    8: "extended", 9: "extended", 10: "extended",
}


def horizon_bucket(horizon: int) -> str:
    """Map a horizon (days ahead) to a pooled bucket name.

    Horizons beyond 10 are grouped into ``"extended"``.
    """
    return _HORIZON_BUCKETS.get(horizon, "extended")


@dataclass
class KalmanSigmaEntry:
    """Scalar Kalman filter tracking sigma for one location+horizon bucket."""

    # State
    x: float = 3.0       # sigma estimate (°F)
    P: float = 4.0       # state uncertainty

    # Fixed parameters
    Q: float = 0.05      # process noise
    R: float = 2.0       # measurement noise

    # Tracking
    sample_count: int = 0
    last_updated: str = ""

    def predict(self) -> None:
        """Prediction step — increase uncertainty by process noise."""
        self.P += self.Q

    def update(self, observed_abs_error: float) -> None:
        """Measurement update step.

        Converts ``|forecast - actual|`` to an implied sigma via the
        half-normal relationship: ``E[|N(0, σ)|] = σ * sqrt(2/π)``,
        so ``implied_sigma = |error| / 0.7979``.
        """
        implied_sigma = observed_abs_error / _SQRT_2_OVER_PI

        S = self.P + self.R
        if S < 1e-10:
            S = 1e-10
        K = self.P / S

        self.x = self.x + K * (implied_sigma - self.x)
        self.P = (1.0 - K) * self.P

        # Clamp state to reasonable bounds
        self.x = max(0.5, min(30.0, self.x))
        self.P = max(0.01, min(20.0, self.P))

        self.sample_count += 1
        self.last_updated = datetime.now(timezone.utc).isoformat()

    @property
    def is_warmed_up(self) -> bool:
        """True once we have at least 5 observations."""
        return self.sample_count >= 5

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "P": self.P,
            "Q": self.Q,
            "R": self.R,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KalmanSigmaEntry":
        return cls(
            x=d.get("x", 3.0),
            P=d.get("P", 4.0),
            Q=d.get("Q", 0.05),
            R=d.get("R", 2.0),
            sample_count=d.get("sample_count", 0),
            last_updated=d.get("last_updated", ""),
        )


@dataclass
class KalmanState:
    """Collection of per-location+horizon Kalman filters."""

    entries: dict[str, KalmanSigmaEntry] = field(default_factory=dict)

    @staticmethod
    def _key(location: str, horizon: int) -> str:
        return f"{location}|{horizon_bucket(horizon)}"

    def get_sigma(self, location: str, horizon: int) -> float | None:
        """Return the Kalman sigma estimate if warmed up, else ``None``."""
        key = self._key(location, horizon)
        entry = self.entries.get(key)
        if entry is not None and entry.is_warmed_up:
            return entry.x
        return None

    # -- Pre-warming from calibration data --

    # Average horizon growth factors per bucket (from calibrate._HORIZON_GROWTH)
    _BUCKET_GROWTH = {
        "short": (1.00 + 1.33) / 2,      # horizons 0-1 → 1.165
        "medium": (1.67 + 2.00 + 2.33) / 3,  # horizons 2-4 → 2.0
        "long": (2.67 + 3.33 + 4.00) / 3,    # horizons 5-7 → 3.333
        "extended": (4.67 + 5.33 + 6.00) / 3, # horizons 8-10 → 5.333
    }

    # Calibrated base sigma per location (2025 full year, 6 locations)
    _CALIBRATED_SIGMA = {
        "NYC": 2.15, "Chicago": 1.83, "Miami": 1.94,
        "Seattle": 1.73, "Atlanta": 1.89, "Dallas": 1.99,
    }

    def prewarm(self, overwrite: bool = False) -> int:
        """Seed all location × horizon entries with calibrated sigma priors.

        Only creates entries that don't already exist (unless *overwrite* is
        ``True``).  Returns the number of entries created or overwritten.

        Uses calibrated base sigma per location scaled by horizon bucket growth
        factors, with ``sample_count=10`` and reduced state uncertainty
        (``P=1.0``) to reflect informed priors.
        """
        count = 0
        for location, base_sigma in self._CALIBRATED_SIGMA.items():
            for bucket_name, growth in self._BUCKET_GROWTH.items():
                key = f"{location}|{bucket_name}"
                if not overwrite and key in self.entries:
                    continue
                sigma = base_sigma * growth
                self.entries[key] = KalmanSigmaEntry(
                    x=round(sigma, 2),
                    P=1.0,       # Reduced from default 4.0 (informed prior)
                    Q=0.05,
                    R=2.0,
                    sample_count=10,  # Past warm-up threshold (5)
                    last_updated="prewarm",
                )
                count += 1
        if count:
            logger.info("Kalman pre-warmed: %d entries seeded", count)
        return count

    def get_blend_weight(self, location: str, horizon: int) -> float:
        """Blend weight that ramps 0 → 0.5 as samples go 5 → 30."""
        key = self._key(location, horizon)
        entry = self.entries.get(key)
        if entry is None or entry.sample_count < 5:
            return 0.0
        if entry.sample_count >= 30:
            return 0.5
        # Linear ramp: 0 at 5 samples, 0.5 at 30 samples
        return 0.5 * (entry.sample_count - 5) / 25.0

    def record_error(self, location: str, horizon: int, abs_error: float) -> None:
        """Run predict + update for the matching location+horizon bucket."""
        key = self._key(location, horizon)
        if key not in self.entries:
            self.entries[key] = KalmanSigmaEntry()
        entry = self.entries[key]
        entry.predict()
        entry.update(abs_error)

    def save(self, path: str | None = None) -> None:
        """Persist Kalman state to JSON (atomic write)."""
        save_path = Path(path) if path else _DEFAULT_PATH
        data = {key: entry.to_dict() for key, entry in self.entries.items()}
        fd, tmp = tempfile.mkstemp(dir=str(save_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(save_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.debug("Kalman state saved to %s (%d entries)", save_path, len(data))

    @classmethod
    def load(cls, path: str | None = None) -> "KalmanState":
        """Load Kalman state from JSON.  Returns empty state if file is missing."""
        load_path = Path(path) if path else _DEFAULT_PATH
        state = cls()
        if not load_path.exists():
            return state
        try:
            with open(load_path) as f:
                data = json.load(f)
            for key, entry_data in data.items():
                state.entries[key] = KalmanSigmaEntry.from_dict(entry_data)
            logger.info("Loaded Kalman state: %d entries", len(state.entries))
        except (json.JSONDecodeError, IOError, TypeError) as exc:
            logger.warning("Failed to load Kalman state: %s — starting fresh", exc)
        return state
