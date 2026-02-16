# Deep Analysis Pipeline Audit — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Fix 5 bugs found during deep audit of the market analysis pipeline (scoring, probabilities, entry/exit conditions, sizing).

**Tech Stack:** Python 3.14, stdlib only.

---

## Task 1: C1 — entry_threshold bloque les NO valides

**File:** `weather/strategy.py` ~line 1144

**Problem:** `if price >= config.entry_threshold` filters on the token price. For NO entries, `price` is the NO token price (e.g. $0.90 when YES is $0.10). This blocks valid NO opportunities where we short an overpriced unlikely event.

**Fix:** Always compare the YES price against the threshold. For NO entries, `yes_price = 1 - price`.

---

## Task 2: C2 — should_exit_on_edge_inversion inversé pour NO

**File:** `weather/strategy.py` ~line 425-431

**Problem:** The NO condition `our_prob > (1 - market_price)` signals exit when our edge is still positive (inverted logic).

**Fix:** For NO side, edge inversion = `market_price > (1 - our_prob)` — the YES price exceeds our YES probability estimate, meaning our NO bet is now unfavorable.

---

## Task 3: C3 — Platt a < 0 garde

**File:** `weather/probability.py` (platt_calibrate function)

**Problem:** If calibration produces `a < 0`, probability ordering is inverted — higher raw prob → lower calibrated prob. No guard exists.

**Fix:** Clamp `a >= 0.01` with warning log.

---

## Task 4: C4 — Feedback AR(1) sans cap

**File:** `weather/feedback.py` (get_bias function)

**Problem:** `phi=0.8` + `last_error=5°F` + `bias_ema` can push forecast correction to ±6°F+ without bounds.

**Fix:** Cap total correction to ±5.0°F.

---

## Task 5: M5 — Beta non-convergence fallback

**File:** `weather/probability.py` (_regularized_incomplete_beta / _student_t_cdf)

**Problem:** If continued fraction doesn't converge, returns partial value silently. Could produce wrong probabilities.

**Fix:** Track convergence, fall back to normal CDF approximation in _student_t_cdf when beta fails.

---

## Verification

```bash
python3 -m pytest weather/tests/ bot/tests/ -q
```
