"""Weather Trading Bot — trades Polymarket weather markets using NOAA forecasts via CLOB direct."""

from .config import Config

def load_config(schema=None, skill_file=None, config_filename="config.json"):
    """Load Config from the package directory."""
    from pathlib import Path
    config_dir = str(Path(__file__).parent)
    return Config.load(config_dir)

def get_config_path(skill_file=None, config_filename="config.json"):
    """Return path to config.json."""
    from pathlib import Path
    return Path(__file__).parent / config_filename

def update_config(updates, skill_file=None, config_filename="config.json"):
    """Update and save config."""
    from pathlib import Path
    config_dir = str(Path(__file__).parent)
    cfg = Config.load(config_dir)
    cfg.update(updates)
    cfg.save(config_dir)
    return {f.name: getattr(cfg, f.name) for f in __import__("dataclasses").fields(cfg)}
