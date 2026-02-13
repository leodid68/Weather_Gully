"""Configuration management — dataclass-based, CLOB direct."""

import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported locations (matching Polymarket resolution sources)
LOCATIONS = {
    "NYC": {"lat": 40.7769, "lon": -73.8740, "name": "New York City (LaGuardia)"},
    "Chicago": {"lat": 41.9742, "lon": -87.9073, "name": "Chicago (O'Hare)"},
    "Seattle": {"lat": 47.4502, "lon": -122.3088, "name": "Seattle (Sea-Tac)"},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "name": "Atlanta (Hartsfield)"},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "name": "Dallas (DFW)"},
    "Miami": {"lat": 25.7959, "lon": -80.2870, "name": "Miami (MIA)"},
}

# Polymarket constraints
MIN_SHARES_PER_ORDER = 5.0
MIN_TICK_SIZE = 0.01

# Environment variable mapping
_ENV_MAP = {
    "entry_threshold": "WEATHER_ENTRY_THRESHOLD",
    "exit_threshold": "WEATHER_EXIT_THRESHOLD",
    "max_position_usd": "WEATHER_MAX_POSITION",
    "sizing_pct": "WEATHER_SIZING_PCT",
    "max_trades_per_run": "WEATHER_MAX_TRADES",
    "locations": "WEATHER_LOCATIONS",
    "private_key": "POLY_PRIVATE_KEY",
}


@dataclass
class Config:
    # Auth (CLOB direct)
    private_key: str = ""
    creds_file: str = "creds.json"

    # Entry / exit
    entry_threshold: float = 0.15
    exit_threshold: float = 0.45
    max_position_usd: float = 2.00
    sizing_pct: float = 0.05

    # Kelly
    kelly_fraction: float = 0.25
    min_ev_threshold: float = 0.03

    # Rate limiting
    max_trades_per_run: int = 5

    # Retry
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # Locations
    locations: str = "NYC"

    # Logging
    log_level: str = "INFO"

    # State
    state_file: str = "weather_state.json"

    # Strategy toggles
    max_days_ahead: int = 7
    seasonal_adjustments: bool = True
    adjacent_buckets: bool = True
    dynamic_exits: bool = True

    # Context safeguards
    slippage_max_pct: float = 0.15
    time_to_resolution_min_hours: int = 2
    price_drop_threshold: float = 0.10

    # Multi-source forecasting (Open-Meteo)
    multi_source: bool = True

    # Forecast change detection
    forecast_change_threshold: float = 3.0  # °F change to trigger re-evaluation

    # Correlation guard (max 1 position per event)
    correlation_guard: bool = True

    # Stop-loss on forecast reversal
    stop_loss_reversal: bool = True
    stop_loss_reversal_threshold: float = 5.0  # °F shift away from our bucket

    # Max exposure for portfolio calculation
    max_exposure: float = 50.0

    # Aviation Weather observations (METAR)
    aviation_obs: bool = True
    aviation_obs_weight: float = 0.40
    aviation_hours: int = 24

    @property
    def active_locations(self) -> list[str]:
        """Return canonical location keys matching LOCATIONS dict keys.

        Input is case-insensitive: ``"nyc,chicago"`` → ``["NYC", "Chicago"]``.
        """
        canonical = {k.lower(): k for k in LOCATIONS}
        result = []
        for raw in self.locations.split(","):
            raw = raw.strip()
            if not raw:
                continue
            canon = canonical.get(raw.lower())
            if canon:
                result.append(canon)
            else:
                result.append(raw)  # Pass through unknown locations as-is
        return result

    @classmethod
    def load(cls, config_dir: str) -> "Config":
        """Load config with priority: config.json > env vars > defaults."""
        config_path = Path(config_dir) / "config.json"
        file_cfg: dict = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_cfg = json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Failed to load %s: %s", config_path, exc)

        kwargs: dict = {}
        field_types = {f.name: f.type for f in fields(cls)}

        for f in fields(cls):
            name = f.name
            # Priority 1: config.json
            if name in file_cfg:
                kwargs[name] = _coerce(file_cfg[name], field_types[name])
            # Priority 2: env vars
            elif name in _ENV_MAP:
                env_val = os.environ.get(_ENV_MAP[name])
                if env_val is not None:
                    kwargs[name] = _coerce(env_val, field_types[name])
            # else: default from dataclass

        return cls(**kwargs)

    def save(self, config_dir: str) -> None:
        """Persist current config to config.json."""
        config_path = Path(config_dir) / "config.json"
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.pop("private_key", None)  # Never persist the private key
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Config saved to %s", config_path)

    def update(self, overrides: dict) -> None:
        """Apply key=value overrides (from --set CLI flag)."""
        field_types = {f.name: f.type for f in fields(self)}
        for key, value in overrides.items():
            if key not in field_types:
                logger.warning("Unknown config key: %s", key)
                continue
            setattr(self, key, _coerce(value, field_types[key]))

    def load_api_creds(self, config_dir: str) -> dict | None:
        """Load API creds from creds_file if it exists."""
        for path in [self.creds_file, str(Path(config_dir) / self.creds_file)]:
            if path and os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        return None


def _coerce(value, type_hint: str):
    """Coerce a value to the declared field type."""
    if type_hint == "bool" or type_hint is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")
    if type_hint == "int" or type_hint is int:
        return int(value)
    if type_hint == "float" or type_hint is float:
        return float(value)
    return str(value)
