# P0 Fixes: Feedback Loop + Value Traps

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** (1) Débloquer le feedback loop via résolution locale METAR quand Gamma ne répond pas, (2) Filtrer les value traps (tail buckets extrêmes) via min_probability + distance filter.

**Tech Stack:** Python 3.14, stdlib, pas de deps externes.

---

## Task 1: Value traps — min_probability + distance filter

**Files:**
- Modify: `weather/config.py` — relever min_probability, ajouter max_bucket_distance_sigma
- Modify: `weather/config.json` — même chose
- Modify: `weather/strategy.py` — ajouter distance filter dans `score_buckets()` (YES et NO)
- Create: `weather/tests/test_value_traps.py`

### What to build

Deux filtres complémentaires dans `score_buckets()` pour éliminer les tail buckets :

1. **min_probability** : 0.15 → 0.25 (config default + config.json)
2. **Distance filter** : skip si `|forecast_temp - bucket_center| > 2.5 * sigma`
   - `bucket_center = (bucket[0] + bucket[1]) / 2.0`
   - Pour les buckets ouverts (-999 ou +999), utiliser l'autre borne comme centre
   - Appliquer au côté YES (ligne 266) ET NO (ligne 283)
   - Utiliser `sigma_override` si fourni, sinon fallback global

### Implementation

**config.py** — après `min_probability` (ligne 66) :
```python
min_probability: float = 0.25  # was 0.15
max_bucket_distance_sigma: float = 2.5  # skip buckets > N*sigma from forecast
```

**config.json** :
```json
"min_probability": 0.25,
"max_bucket_distance_sigma": 2.5
```

**strategy.py** — dans `score_buckets()`, après `if prob < config.min_probability: continue` (ligne 266) :
```python
# Value trap: bucket center too far from forecast
_lo, _hi = bucket
bc = _hi if _lo < -900 else (_lo if _hi > 900 else (_lo + _hi) / 2.0)
if sigma_override and abs(forecast_temp - bc) > config.max_bucket_distance_sigma * sigma_override:
    continue
```

Même chose pour le côté NO (après ligne 283-284).

### Tests

- `test_min_probability_filters_low_prob`: bucket prob=0.20 → filtered (< 0.25)
- `test_distance_filter_blocks_extreme_bucket`: forecast=38.8, bucket=(29,31), sigma=2.15 → distance 7.3 > 5.4 → filtered
- `test_distance_filter_allows_near_bucket`: forecast=43.0, bucket=(42,43), sigma=2.15 → distance 0.5 → allowed
- `test_distance_filter_open_bucket`: bucket=(-999, 41), forecast=38, sigma=2.0 → uses hi=41, distance=3 → allowed
- `test_no_side_also_filtered`: same distance filter applied to NO entries

### Verification
```
python3 -m pytest weather/tests/test_value_traps.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Task 2: METAR-based resolution fallback

**Files:**
- Modify: `weather/paper_trade.py` — ajouter fallback METAR dans `_resolve_predictions()`
- Create: `weather/tests/test_metar_resolution.py`

### What to build

Dans `_resolve_predictions()`, après la passe Gamma, ajouter une deuxième passe pour les prédictions non résolues dont la date est passée de 24h+.

### Implementation

**paper_trade.py** — dans `_resolve_predictions()`, après la boucle Gamma (ligne 66) :

```python
# Fallback: resolve via local METAR observations if Gamma hasn't resolved
now = datetime.now(timezone.utc)
for market_id, pred in list(state.predictions.items()):
    if pred.resolved:
        continue
    # Only resolve if forecast date is 24h+ in the past
    try:
        fd = datetime.strptime(pred.forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        continue
    if (now - fd).total_seconds() < 86400:
        continue
    # Check daily observations
    obs = state.get_daily_obs(pred.location, pred.forecast_date)
    if not obs:
        continue
    actual_key = f"obs_{pred.metric}"
    actual_temp = obs.get(actual_key)
    if actual_temp is None:
        continue
    # Resolve: is actual temp within [bucket_low, bucket_high]?
    in_bucket = pred.bucket_low <= actual_temp <= pred.bucket_high
    pred.resolved = True
    pred.actual_outcome = in_bucket
    resolved_count += 1
    logger.info(
        "Resolved prediction %s via METAR: actual=%.1f°F bucket=[%s,%s] → %s (prob was %.2f)",
        market_id[:16], actual_temp, pred.bucket_low, pred.bucket_high,
        "WIN" if in_bucket else "LOSS", pred.our_probability,
    )
```

Ajouter `from datetime import datetime, timezone` en haut du fichier (déjà importé via pending_state, vérifier).

### Tests

- `test_metar_resolves_past_prediction`: pred avec date J-2, obs existe, actual dans bucket → resolved=True, outcome=True
- `test_metar_resolves_loss`: pred avec date J-2, obs existe, actual hors bucket → resolved=True, outcome=False
- `test_metar_skips_recent_prediction`: pred avec date aujourd'hui → pas résolu (< 24h)
- `test_metar_skips_no_obs`: pred avec date J-2, pas d'obs → pas résolu
- `test_gamma_takes_priority`: si Gamma résout d'abord → pas de double résolution
- `test_open_bucket_resolution`: bucket=(-999, 41), actual=39 → in_bucket=True

### Verification
```
python3 -m pytest weather/tests/test_metar_resolution.py -v
python3 -m pytest weather/tests/ bot/tests/ -q
```

---

## Final Verification

```bash
python3 -m pytest weather/tests/ bot/tests/ -q   # All tests pass
python3 -m weather.paper_trade --set locations=NYC --explain  # Verify value traps filtered + predictions resolved
```
