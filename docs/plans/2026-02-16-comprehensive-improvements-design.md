# Weather Gully — Comprehensive Improvements Design

**Date**: 2026-02-16
**Context**: Transition imminente paper trading → live trading
**Approach**: Safety First — sécuriser avant d'améliorer l'alpha

---

## Phase 1 — Robustesse & Risk Management (Avant live)

### 1.1 Robustesse numérique Student's t CDF

**Fichier**: `weather/probability.py`

- `_regularized_incomplete_beta(x, a, b)`: guards `a <= 0` / `b <= 0` → return 0.0 + warning. Clamp `x` à `[1e-15, 1-1e-15]`. Warning si `max_iter` atteint sans convergence.
- `_student_t_cdf(x, df)`: guard `df <= 0` → return 0.5 + warning. Guard `x = ±inf` → return 0.0/1.0. Guard `x = NaN` → return 0.5 + warning.
- `estimate_bucket_probability`: capper sortie à `min(prob, 1.0)`. Borne haute sigma: `sigma = min(sigma, 50.0)`.

### 1.2 Logging fallback calibration

**Fichier**: `weather/probability.py`

- `_load_calibration()`: si fichier absent → `logger.warning("calibration.json not found, using hardcoded fallbacks")`
- Si clé location absente → `logger.warning("No calibration for %s, using global sigma", location)`
- `estimate_bucket_probability`: logger quel sigma utilisé (calibré vs fallback) au niveau DEBUG

### 1.3 Circuit breaker global

**Fichiers**: `weather/config.py`, `weather/strategy.py`, `weather/state.py`

**Nouveaux paramètres config**:
```
daily_loss_limit: float = 10.0
max_positions_per_day: int = 20
cooldown_hours_after_max_loss: float = 24.0
max_open_positions: int = 15
```

**Logique strategy.py**:
- Avant chaque trade: vérifier P&L du jour via `state.get_daily_pnl()`. Si `< -daily_loss_limit` → skip, log "CIRCUIT BREAKER"
- Vérifier `state.positions_opened_today() >= max_positions_per_day` → skip
- Vérifier cooldown: si dernier circuit break < `cooldown_hours` → skip
- Vérifier `len(state.open_positions) >= max_open_positions` → skip
- Enforcer `max_exposure` dans strategy (pas seulement dans bridge)

**État state.py**:
- `daily_pnl_tracker`: dict `{date_str: float}`
- `last_circuit_break: str | None`

### 1.4 Tests de robustesse numérique

**Fichier**: `weather/tests/test_probability.py`

Nouveaux tests:
- `_student_t_cdf(±1e10, 10)` → ~0.0 / ~1.0 sans crash
- `_student_t_cdf(float('nan'), 10)` → 0.5
- `_student_t_cdf(float('inf'), 10)` → 1.0
- `_student_t_cdf(0, 0.5)` → 0.5
- `_student_t_cdf(0, 1000)` → 0.5
- `_regularized_incomplete_beta(0.5, 0, 1)` → ne crash pas
- `estimate_bucket_probability` avec sigma=0 → résultat valide
- `estimate_bucket_probability` retourne toujours `<= 1.0`

### 1.5 Validation calibration age

**Fichiers**: `weather/probability.py`, `weather/calibrate.py`

- `_load_calibration()`: lire `calibration.json["_metadata"]["generated_at"]`
- Si age > 30 jours → `logger.warning("Calibration is %d days old")`
- Si age > 90 jours → `logger.error("Calibration is %d days old — STALE")`
- Optionnel: `--check-age` flag dans `calibrate.py`

---

## Phase 2 — Alpha Enhancement (Juste après live)

### 2.1 Positions NO (Buy NO + Sell YES)

**Fichiers**: `weather/strategy.py`, `weather/sizing.py`, `weather/state.py`, `weather/bridge.py`

**Buy NO**:
- Condition: `market_price - notre_prob > min_ev_threshold`
- Sizing: `compute_position_size(1 - notre_prob, 1 - market_price, ...)`
- `TradeRecord.side` → `"yes"` ou `"no"`
- Vérifier support `side="no"` dans `CLOBWeatherBridge`

**Sell YES existant**:
- Si on détient YES et `market_price > notre_prob + exit_margin` → vendre
- Enrichir `compute_exit_threshold` avec la probabilité modèle

**Safeguards**: même Kelly, même min_ev, circuit breaker identique, pas de NO + YES sur même event

### 2.2 Externaliser horizon growth

**Fichiers**: `weather/calibrate.py`, `weather/calibration.json`, `weather/probability.py`

- Déplacer `_HORIZON_GROWTH` dans `calibration.json` sous clé `"horizon_growth"`
- `calibrate.py`: écrire les growth factors dans le JSON
- `probability.py`: lire depuis JSON, fallback sur dict hardcodé

### 2.3 Poids exponentiel feedback loop

**Fichier**: `weather/feedback.py`

- Stocker timestamp de chaque update
- Appliquer decay: `decay = 0.5^(days_since_last_update / half_life_days)`
- `half_life_days = 30` (configurable)
- Si `decay < 0.1` (> 100 jours sans données) → reset
- `_MIN_SAMPLES`: 7 → 5

### 2.4 Métriques + CLI report

**Nouveaux fichiers**: `weather/metrics.py`, `weather/report.py`

**Métriques**:
- Brier score rolling 7j
- Calibration table (binned predicted vs actual)
- Sharpe ratio annualisé
- Win rate
- Average edge

**CLI report** (`python3 -m weather.report`):
- Lit `paper_state.json` + `feedback_state.json` + `trade_log.jsonl`
- Output: positions ouvertes, P&L total, P&L aujourd'hui, Brier, Sharpe, calibration drift
- Zéro dépendance externe

---

## Phase 3 — Optimisation (Continu)

### 3.1 Parallélisation des API calls

**Fichiers**: `weather/strategy.py`, `weather/open_meteo.py`

- `concurrent.futures.ThreadPoolExecutor(max_workers=3)`
- Paralléliser NOAA + Open-Meteo + METAR
- Timeout individuel 45s par source
- Dégradation gracieuse si une source timeout

### 3.2 Cache intelligent des forecasts

**Fichiers**: `weather/open_meteo.py`, `weather/noaa.py`

- Cache module-level avec TTL 15 min (in-memory)
- `_fetch_json_cached(url, ttl=900)` wrapper
- Se vide au restart du process

### 3.3 Corrélation inter-location

**Nouveau fichier**: `weather/correlation.py`
**Modifié**: `weather/strategy.py`

- Matrice corrélation statique par saison (6 locations)
- NYC↔Atlanta: 0.5, NYC↔Chicago: 0.4, Miami↔Atlanta: 0.3, Chicago↔Dallas: 0.2, Seattle↔tous: <0.2
- Sizing ajusté: `adjusted_size = base_size * (1 - max_corr * 0.5)`
- Futur: corrélations empiriques depuis données historiques

---

## Items skippés

| Item | Raison |
|------|--------|
| #13 Alertes Telegram | Déjà géré par openclaw |
| #14 Dashboard web | Remplacé par CLI report (Phase 2.4) |
| #15 Mode dry run | Paper trading remplit cette fonction |
