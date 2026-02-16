# OpenClaw

Bot de trading automatique sur les marchés de température [Polymarket](https://polymarket.com).

Analyse les prévisions météo multi-sources (NOAA, ECMWF, GFS, modèles régionaux, observations METAR) pour estimer la probabilité de chaque bucket de température, puis trade quand le prix du marché diverge significativement de la probabilité réelle.

---

## Quickstart

```bash
# Installer
pip install httpx eth-account eth-abi eth-utils websockets certifi

# Dry-run (aucun trade, affiche les opportunités)
python3 -m weather

# Paper trading (prix réels, exécution simulée)
python3 -m weather.paper_trade

# Live trading
export POLY_PRIVATE_KEY=0x...
python3 -m weather --live
```

---

## Villes disponibles

14 villes couvrent **100% des marchés température Polymarket**.

### US (°F)

| Ville | Aéroport | Station METAR |
|-------|----------|---------------|
| NYC | LaGuardia | KLGA |
| Chicago | O'Hare | KORD |
| Seattle | Sea-Tac | KSEA |
| Atlanta | Hartsfield | KATL |
| Dallas | Love Field | KDAL |
| Miami | MIA | KMIA |

### International (°C)

| Ville | Aéroport | Station METAR | Modèle local |
|-------|----------|---------------|-------------|
| London | City Airport | EGLC | ICON (DWD, 7km) |
| Paris | CDG | LFPG | AROME (Météo-France, 1.3km) |
| Seoul | Incheon | RKSI | KMA (5km) |
| Toronto | Pearson | CYYZ | GEM (ECCC, 2.5km) |
| Buenos Aires | Ezeiza | SAEZ | -- |
| Sao Paulo | Guarulhos | SBGR | -- |
| Ankara | Esenboga | LTAC | ICON (DWD, 7km) |
| Wellington | Intl | NZWN | -- |

Sélection via `--set locations=NYC,London,Paris` (case-insensitive).

---

## Paramètres principaux

Configurable via `--set key=value`, `config.json`, ou variables d'env.

### Trading

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `entry_threshold` | 0.15 | Edge minimum (probabilité - prix) pour entrer |
| `exit_threshold` | 0.45 | Edge en dessous duquel sortir |
| `min_ev_threshold` | 0.03 | EV minimum en $ |
| `kelly_fraction` | 0.25 | Fraction Kelly (0.25 = quarter-Kelly) |
| `max_position_usd` | 2.00 | Taille max par position ($) |
| `max_exposure` | 50.00 | Exposition totale max ($) |
| `max_trades_per_run` | 5 | Trades max par exécution |

### Risk Management

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `daily_loss_limit` | 10.00 | Coupe-circuit : perte quotidienne max ($) |
| `max_open_positions` | 15 | Positions ouvertes simultanées max |
| `max_positions_per_day` | 20 | Nouvelles positions max par jour |
| `stop_loss_reversal_threshold` | 5.0 | Shift forecast (°F) déclenchant le stop-loss |
| `slippage_max_pct` | 0.15 | Slippage max autorisé (15%) |
| `time_to_resolution_min_hours` | 2 | Heures min avant résolution pour trader |

### Stratégie

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `locations` | NYC | Villes à trader (séparées par virgule) |
| `max_days_ahead` | 7 | Horizon max (jours) |
| `multi_source` | true | Ensemble multi-modèles (ECMWF + GFS) |
| `aviation_obs` | true | Observations METAR temps réel |
| `adaptive_sigma` | true | Sigma adaptatif (spread + EMA) |
| `kalman_sigma` | true | Filtre de Kalman pour sigma dynamique |
| `mean_reversion` | true | Sizing Z-score (mean-reversion) |
| `correlation_guard` | true | Max 1 position par événement |
| `stop_loss_reversal` | true | Stop-loss sur retournement forecast |
| `ar_autocorrelation` | true | Correction AR(1) du biais forecast |

---

## Variables d'environnement

| Variable | Description |
|----------|-------------|
| `POLY_PRIVATE_KEY` | Clé privée Ethereum (hex, 0x-prefixed) |
| `WEATHER_LOCATIONS` | Villes (ex: `NYC,London,Paris`) |
| `WEATHER_ENTRY_THRESHOLD` | Edge minimum |
| `WEATHER_MAX_POSITION` | Taille max par position |
| `WEATHER_MAX_TRADES` | Trades max par run |

---

## Modes d'exécution

```bash
# Dry-run (défaut) — affiche les opportunités
python3 -m weather

# Explain — dry-run détaillé avec chaîne de décision complète
python3 -m weather --explain

# Paper trading — prix réels, exécution simulée, suivi P&L
python3 -m weather.paper_trade

# Live — exécution réelle sur Polymarket
python3 -m weather --live

# Live toutes villes
python3 -m weather --live --set locations=NYC,Chicago,Seattle,Atlanta,Dallas,Miami,London,Paris,Seoul,Toronto,BuenosAires,SaoPaulo,Ankara,Wellington

# Verbose (logs DEBUG)
python3 -m weather --verbose

# JSON logs (pour agrégation)
python3 -m weather --json-log

# Voir positions ouvertes
python3 -m weather --positions

# Voir config
python3 -m weather --config
```

---

## Calibration

```bash
# Calibration complète (recommandé avant premier trading)
python3 -m weather.calibrate \
  --locations NYC,Chicago,Seattle,Atlanta,Dallas,Miami \
  --start-date 2025-01-01 --end-date 2026-01-01

# Recalibration incrémentale (rapide, fenêtre glissante 90j)
python3 -m weather.recalibrate \
  --locations NYC,Chicago,Seattle,Atlanta,Dallas,Miami
```

Produit `weather/calibration.json` avec sigma par horizon/ville, poids des modèles, facteurs saisonniers, et paramètres Platt.

---

## Sources de données

| Source | Couverture | Poids | Usage |
|--------|-----------|-------|-------|
| **ECMWF IFS** | Mondial | 50% | Prévision principale |
| **GFS** | Mondial | 30% | Second modèle global |
| **NOAA** | US uniquement | 20% | Prévision officielle US |
| **Modèles régionaux** | International | 20% | AROME, ICON, KMA, GEM |
| **METAR** | Toutes villes | Dynamique | Observations temps réel (aéroport) |

Toutes les APIs sont gratuites, sans clé requise. Seul Polymarket nécessite une clé privée Ethereum pour le trading live.

---

## Profils de risque

### Conservateur (recommandé pour débuter)

```bash
python3 -m weather --live \
  --set locations=NYC \
  --set kelly_fraction=0.25 \
  --set max_position_usd=2.00 \
  --set max_exposure=20.00
```

### Modéré (après 2 semaines de paper trading profitable)

```bash
python3 -m weather --live \
  --set locations=NYC,Chicago,London,Paris \
  --set kelly_fraction=0.25 \
  --set max_position_usd=5.00 \
  --set max_exposure=50.00
```

### Agressif (Brier < 0.20 confirmé)

```bash
python3 -m weather --live \
  --set locations=NYC,Chicago,Seattle,Atlanta,Dallas,Miami,London,Paris,Seoul,Toronto \
  --set kelly_fraction=0.35 \
  --set max_position_usd=10.00 \
  --set max_exposure=100.00
```

---

## Tests

```bash
python3 -m pytest weather/tests/ bot/tests/ -q   # 801 tests
```

---

## Architecture

```
weather/
  strategy.py       — Boucle principale (scoring, sizing, entries/exits)
  open_meteo.py     — Prévisions ECMWF + GFS + modèles régionaux
  noaa.py           — Prévisions NOAA (US)
  aviation.py       — Observations METAR temps réel
  probability.py    — Probabilité Student's t + Platt scaling
  bridge.py         — Adaptateur CLOB Polymarket
  config.py         — Configuration (14 villes, paramètres)
  calibrate.py      — Calibration empirique sigma/poids
  feedback.py       — Correction biais EMA + AR(1)
  kalman.py         — Filtre de Kalman sigma dynamique
  mean_reversion.py — Z-score sizing
```
