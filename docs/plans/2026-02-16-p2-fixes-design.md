# P2 Fixes Design — Window Size, Correlation Threshold, Emergency Exit

**Date:** 2026-02-16

## Context

Issues from `p2.md` TODO list. Several items already fixed (P0-1 mean reversion sizing, P1-4 tmp cleanup, P3-10 calibration age, P4-12 parallel exits). Three items remain.

## Fix 1: Mean Reversion Window Size (P1-2)

**File:** `weather/mean_reversion.py`
**Change:** `_WINDOW_SIZE = 20` → `_WINDOW_SIZE = 150`

With cron every 5 min, 20 snapshots = ~1h40 — too short for meaningful mean reversion. 150 snapshots = ~12h of rolling history. Buffer doubles to 300. `_MIN_SNAPSHOTS` stays at 5.

## Fix 2: Correlation Threshold (P1-3)

**File:** `weather/config.py`
**Change:** `correlation_threshold: float = 0.5` → `correlation_threshold: float = 0.7`

At 0.5, nearly all city pairs trigger the discount. At 0.7, only strongly correlated pairs (NYC/Atlanta winter) are affected.

## Fix 3: Emergency Exit on Circuit Breaker (P2-5)

**File:** `weather/strategy.py`
**New function:** `_emergency_exit_losers(client, state, dry_run) -> int`

When circuit breaker triggers, sell only losing positions (current_price < cost_basis). Winning positions kept. Uses parallel orderbook fetch. Respects dry_run mode. Logs `[EMERGENCY EXIT]` for each exit.

Called from `run_weather_strategy()` when `check_circuit_breaker()` returns blocked=True, before the early return.

## Items verified as non-issues

- **P0-1 (mean reversion sizing):** Already applied at `strategy.py:1085-1090`
- **P2-7 (bucket "Other"):** Polymarket weather markets don't have "Other" buckets — open sentinels cover full range
- **P2-6 (AR(1) phi clamp):** Already clamped to [-0.8, 0.8] in `feedback.py:80`
