# Audit des parametres — Dynamiques vs Statiques

> Audit du 17 fevrier 2026 — a utiliser comme base pour la prochaine session de calibration.

## Parametres calibres (depuis donnees historiques)

| Parametre | Source | Fichier | Notes |
|-----------|--------|---------|-------|
| Sigma (ecart-type erreur) | RMSE par location/horizon | `calibration.json` > `location_sigma`, `global_sigma` | 6 villes US, 11 horizons (0-10j) |
| Poids modeles (GFS/ECMWF/NOAA) | Grid search MAE | `calibration.json` > `model_weights` | GFS domine sur la plupart des villes |
| Facteurs saisonniers | sigma_mensuelle / sigma_moyenne | `calibration.json` > `seasonal_factors`, `location_seasonal` | Mois 1,2,11,12 uniquement |
| Matrice de correlation | Pearson par paire/saison | `calibration.json` > `correlation_matrix` | 15 paires, 2 saisons (DJF, SON) |
| Skew-t (df=50, gamma=0.8) | MLE sur 2136 erreurs | `calibration.json` > `student_t_df`, `skew_t_gamma` | Valide en OOS (ratio 0.906) |
| Adaptive sigma facteurs | Regression sur erreurs | `calibration.json` > `adaptive_sigma` | underdispersion=1.3, spread_to_sigma=0.616, ema_to_sigma=1.065 |

## Parametres dynamiques (s'adaptent au runtime per-trade)

| Parametre | Mecanisme | Fichier |
|-----------|-----------|---------|
| Adaptive sigma | MAX(ensemble spread, model spread, EMA erreurs, floor calibre) | `probability.py` > `compute_adaptive_sigma()` |
| Kalman filter sigma | Filtre scalaire par location+horizon, update a chaque trade resolu | `kalman.py` > `KalmanSigmaFilter` |
| Feedback bias AR(1) | EMA du biais + correction autoregressive, decroissance temporelle (demi-vie 30j) | `feedback.py` > `FeedbackState.get_bias()` |
| Mean-reversion Z-score | Rolling Z-score sur fenetre 150 snapshots de prix | `mean_reversion.py` > `PriceHistory.z_score()` |
| Slippage adaptatif | `min(max_pct, edge * ratio)` — trades a fort EV tolerent plus de spread | `strategy.py` > `check_context_safeguards()` |
| Exit dynamique | Seuil se resserre selon heures restantes avant resolution | `sizing.py` > `compute_exit_threshold()` |
| Budget dynamique | Kelly applique sur `balance - exposure` au lieu de `balance` | `sizing.py` > `compute_position_size()` |

## Parametres statiques — candidats a la calibration

### Priorite haute

**1. Kelly fraction = 0.25**
- Fichier: `config.py` L64
- Actuellement: quarter-Kelly par convention (conservateur)
- Calibration possible: backtest sur l'historique 2025, optimiser la fraction qui maximise le ratio Sharpe ou le log-wealth
- Complexite: faible — une boucle de backtest avec differentes fractions (0.10 a 0.50)
- Impact attendu: potentiellement +20-40% de rendement si la fraction optimale est plus haute

**2. Horizon growth factors = valeurs NWP litterature**
- Fichier: `calibrate.py` L48-60, `calibration.json` > `horizon_growth`
- Actuellement: {0: 1.0, 1: 1.33, 2: 1.67, ..., 10: 6.0} — modele lineaire NWP
- Calibration possible: on a maintenant assez de donnees (2136 erreurs) pour calculer le RMSE reel par horizon et en deduire les facteurs empiriques
- Complexite: moyenne — modifier `calibrate.py` pour calculer `sigma_reel(h) / sigma_reel(0)` par horizon
- Impact attendu: meilleure precision sur les horizons lointains (J+5 a J+10)

**3. Mean-reversion seuils = -1/+1, pente 0.25**
- Fichier: `mean_reversion.py` L121-133
- Actuellement: Z < -1 boost (jusqu'a 1.5x), Z > +1 reduit (jusqu'a 0.5x), pente 0.25
- Calibration possible: backtest avec differents seuils/pentes, mesurer l'amelioration du P&L
- Complexite: moyenne — grille 2D (seuil, pente) sur les donnees de backtest
- Impact attendu: sizing plus precis quand les prix divergent de la moyenne

### Priorite moyenne

**4. Correlation threshold = 0.3, discount = 0.5**
- Fichier: `config.py` L132-133
- Calibration possible: optimiser sur le backtest (minimiser la variance du P&L tout en maximisant le rendement)
- Impact: mieux calibrer le risque de portefeuille

**5. Same-location discount = 0.5, window = 2j**
- Fichier: `config.py` L153-154
- Calibration possible: mesurer la correlation empirique entre trades same-city a J+0 vs J+2
- Impact: eviter la sur-exposition sur une ville

**6. Entry threshold = 0.15**
- Fichier: `config.py` L58
- Calibration possible: backtest — quel seuil de prix YES maximise le P&L?
- Impact: filtrer mieux les marches trop chers

### Pas prioritaire

**7. Platt scaling (desactive)**
- `a=1.0, b=0.0` — desactive car casse la coherence des buckets mutuellement exclusifs
- Alternative future: calibration isotonique (non-parametrique) qui preserve la monotonie

**8. max_position_usd = $2.00**
- Contrainte de risque, pas un parametre de modele
- Devrait etre proportionnel au bankroll plutot que fixe

## Plan pour la prochaine session

1. Calibrer Kelly fraction par backtest (grille 0.10 a 0.50 par pas de 0.05)
2. Calculer les horizon growth factors empiriques depuis les erreurs
3. Optimiser les seuils mean-reversion par backtest
4. Eventuellement: rendre max_position_usd proportionnel au balance (ex: 4% du bankroll)
