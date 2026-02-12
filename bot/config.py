"""Configuration management — dataclass with config.json > env > defaults."""

import json
import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_MAP = {
    "private_key": "POLY_PRIVATE_KEY",
    "creds_file": "POLY_CREDS_FILE",
}


@dataclass
class Config:
    # Auth
    private_key: str = ""
    creds_file: str = "creds.json"

    # Entry / exit
    entry_threshold: float = 0.10
    exit_threshold: float = 0.40
    max_position_usd: float = 5.00

    # Kelly sizing
    kelly_fraction: float = 0.25
    min_ev_threshold: float = 0.03

    # Rate limiting
    max_trades_per_run: int = 5

    # Risk management
    max_total_exposure: float = 50.00
    max_open_positions: int = 10
    max_daily_loss: float = 10.00

    # Signals
    longshot_bias: bool = True
    arbitrage: bool = True
    microstructure: bool = True
    imbalance_threshold: float = 0.30

    # Multi-choice
    multi_choice_arbitrage: bool = True
    polymarket_fee_bps: int = 200        # Polymarket fees ~2%

    # Stop-loss
    stop_loss_pct: float = 0.50          # exit if loss > 50%

    # Longshot bias
    longshot_min_edge: float = 0.005     # separate threshold for longshot bias

    # Parallelism
    parallel_workers: int = 10           # ThreadPoolExecutor workers

    # Scanning
    min_liquidity_grade: str = "C"
    scan_limit: int = 100

    # Gamma API
    use_gamma: bool = True
    gamma_min_volume: float = 1000.0
    gamma_min_liquidity: float = 100.0

    # Daemon mode
    run_interval_seconds: int = 60
    retry_max_attempts: int = 5
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 300.0

    # Weather integration
    weather_enabled: bool = True
    weather_locations: str = "NYC"
    weather_entry_threshold: float = 0.15
    weather_exit_threshold: float = 0.45
    weather_max_days_ahead: int = 7
    weather_seasonal_adjustments: bool = True
    weather_multi_source: bool = True
    weather_correlation_guard: bool = True
    weather_stop_loss_reversal: bool = True
    weather_stop_loss_reversal_threshold: float = 5.0

    # State
    state_file: str = "state.json"

    # Logging
    log_level: str = "INFO"

    @classmethod
    def load(cls, config_dir: str) -> "Config":
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
            if name in file_cfg:
                kwargs[name] = _coerce(file_cfg[name], field_types[name])
            elif name in _ENV_MAP:
                env_val = os.environ.get(_ENV_MAP[name])
                if env_val is not None:
                    kwargs[name] = _coerce(env_val, field_types[name])

        return cls(**kwargs)

    def save(self, config_dir: str) -> None:
        config_path = Path(config_dir) / "config.json"
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.pop("private_key", None)  # Never persist the private key
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Config saved to %s", config_path)

    def update(self, overrides: dict) -> None:
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
    if type_hint == "bool" or type_hint is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")
    if type_hint == "int" or type_hint is int:
        return int(value)
    if type_hint == "float" or type_hint is float:
        return float(value)
    return str(value)
