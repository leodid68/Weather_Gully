📋 **TODO LIST pour Claude Code**



✅ **POINTS FORTS (Rien à changer)**



1. **Architecture modulaire** - Séparation claire strategy/probability/sizing/bridge
2. **Student's t-distribution** - df=10 pour heavy tails, implémentation custom sans scipy
3. **Platt scaling** - Calibration a=0.6205, b=0.6942 sur 2112+ samples
4. **Kelly criterion** - Fraction 0.25, sizing dynamique par bankroll
5. **Safeguards complets** - Slippage, time-to-resolution, correlation guard
6. **Paper trading identique au live** - Même code, bridge différent
7. **Tests unitaires** - 15+ fichiers de test couvrant les composants critiques



🔴 **CRITIQUE - À FIXER IMMÉDIATEMENT**



**1. Robustesse numérique Student's t CDF**



\# Fichier: weather/probability.py (lignes ~125-170)

def _regularized_incomplete_beta(x, a, b, max_iter=200):

  \# PROBLÈME: Pas de gestion d'erreur sur math.lgamma, math.log

  \# RISQUE: Overflow/NaN sur x proche de 0 ou 1



**Action:** Ajouter try/except et clamping des valeurs d'entrée



**2. Logging fallback calibration**



\# Fichier: weather/probability.py (lignes ~35-55)

def _load_calibration():

  \# PROBLÈME: Si calibration.json absent ou corrompu → silence

  \# On utilise les tables hardcodées sans le savoir



**Action:** Logger WARNING explicite quand fallback utilisé



**3. Pas de position "NO" (short)**



\# Fichier: weather/strategy.py (lignes ~350-400)

side: str = "yes" # TOUJOURS yes, jamais no

\# PROBLÈME: On ne short jamais les buckets sur-évalués

\# EXEMPLE: Si market donne 80% et modèle 30% → on ne trade pas



**Action:** Implémenter la logique inverse pour les buckets improbables



**4. Pas de circuit breaker global**



\# Fichier: weather/config.py

\# MANQUE: daily_loss_limit, max_positions_per_day, cooldown_after_loss_streak



**Action:** Ajouter:



daily_loss_limit: float = 10.0 # Stop trading after $10 loss

max_positions_per_day: int = 20

cooldown_hours_after_max_loss: float = 24.0





🟡 **AMÉLIORATION - À IMPLÉMENTER**



**5. Poids exponentiel feedback loop**



\# Fichier: weather/feedback.py

\# ACTUEL: 90-day rolling window uniforme

\# AMÉLIORATION: half_life_days=30 pour plus de poids sur erreurs récentes



**6. Externaliser horizon growth**



\# Fichier: weather/calibrate.py (lignes ~45-60)

_HORIZON_GROWTH = {0: 1.00, 1: 1.33, ...} # Hardcodé

\# AMÉLIORATION: Déplacer dans calibration.json pour tuning sans redéploy



**7. Corrélation inter-location**



\# Fichier: weather/strategy.py

\# MANQUE: Détection de clusters (NYC+Chicago même front froid)

\# AMÉLIORATION: Matrice de corrélation location×location par saison



**8. Tests de robustesse numérique**



\# Fichier: weather/tests/test_probability.py

\# À AJOUTER: Tests sur _student_t_cdf avec:

\# - x = ±1e10 (overflow)

\# - x = nan, inf

\# - df = 0.5, 1000 (edge cases)



**9. Validation calibration en temps réel**



\# Fichier: weather/calibrate.py

\# À AJOUTER: Vérifier que calibration.json n'est pas plus vieux que X jours

\# Si oui → WARNING + auto-recalibration trigger



**10. Métriques de qualité des prévisions**



\# Fichier: weather/strategy.py ou nouveau fichier metrics.py

\# À AJOUTER:

\# - Brier score rolling (7 jours)

\# - Calibration plot (predicted vs actual)

\# - Sharpe ratio des trades





📊 **OPTIMISATION - NICE TO HAVE**



**11. Parallélisation des API calls**



\# Fichier: weather/strategy.py (fonction run_weather_strategy)

\# ACTUEL: Sequential fetching

\# OPTIMISATION: asyncio.gather pour NOAA + Open-Meteo + METAR



**12. Cache intelligent des forecasts**



\# Fichier: weather/open_meteo.py, weather/noaa.py

\# ACTUEL: Re-fetch à chaque run (toutes les 5 min)

\# OPTIMISATION: TTL cache de 15-30 min pour économiser les rate limits



**13. Alertes Discord/Telegram**



\# Fichier: weather/paper_trade.py (fonction main)

\# À AJOUTER: Notification sur:

\# - Trade exécuté (avec détails)

\# - Résolution (gain/perte)

\# - Erreur API répétée

\# - Calibration outdated



**14. Dashboard temps réel**



\# NOUVEAU FICHIER: weather/dashboard.py

\# Simple Flask/FastAPI pour voir:

\# - Positions ouvertes

\# - P&L temps réel

\# - Calibration drift



**15. Mode "dry run" avec logging détaillé**



\# Fichier: weather/paper_trade.py



