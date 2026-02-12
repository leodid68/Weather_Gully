"""Trading bot — generic strategy runner using Polymarket CLOB + Gamma API."""

from .config import Config
from .gamma import GammaClient, GammaMarket, MultiChoiceGroup, group_multi_choice
from .scoring import brier_score, calibration_curve, edge_confidence, log_score
from .signals import Signal, detect_multi_choice_arbitrage, scan_for_signals
from .sizing import check_risk_limits, kelly_fraction, position_size
from .state import TradingState, TradeRecord, state_lock

__all__ = [
    "Config",
    "GammaClient",
    "GammaMarket",
    "MultiChoiceGroup",
    "Signal",
    "TradeRecord",
    "TradingState",
    "brier_score",
    "calibration_curve",
    "check_risk_limits",
    "detect_multi_choice_arbitrage",
    "edge_confidence",
    "group_multi_choice",
    "kelly_fraction",
    "log_score",
    "position_size",
    "scan_for_signals",
    "state_lock",
]
