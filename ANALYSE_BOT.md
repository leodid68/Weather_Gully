# Weather Gully — Analyse complète : Forces, Faiblesses & Opportunités d'Arbitrage

> Audit réalisé le 17 février 2026 sur l'ensemble du codebase (weather/, bot/, polymarket/)
> État : paper trading actif, 9 positions ouvertes, 63 runs complétés, feedback_state.json vide

---

## 1. Points forts

### 1.1 Architecture & qualité de code

Le projet suit un pattern Adapter + Strategy + Bridge bien exécuté. Les 3 packages (`polymarket/`, `bot/`, `weather/`) sont indépendants et composables. Le `CLOBWeatherBridge` découple la stratégie de l'exécution, ce qui permet au paper trading de partager exactement le même code que le live — seul le bridge change. C'est la bonne approche : aucun mock, aucune divergence possible entre test et production.

Le client CLOB est implémenté sans SDK (EIP-712 signing, HMAC L2 auth, WebSocket reconnect). Les ordres utilisent `Decimal` (pas de float rounding), les fichiers d'état sont écrits atomiquement (`tempfile` + `os.replace`) avec verrouillage `fcntl.flock`. Les HMAC headers sont rebuilds dans la retry loop pour garantir un timestamp frais. 846 tests, 5 rounds d'audit, 0 issues critiques restantes.

### 1.2 Pipeline de données météo

Triple source pour les villes US — NOAA (api.weather.gov), Open-Meteo (GFS + ECMWF), et METAR (observations aéroport en temps réel). L'ensemble forecast pondère les modèles par performance historique (poids dans `calibration.json`). Quand les modèles divergent (NOAA vs Open-Meteo > 3°F), le sigma est automatiquement boosté (ex: Chicago 3.72 → 5.59), ce qui est un comportement correct : plus d'incertitude = distribution plus plate = moins de trades agressifs.

Le fetch des données est parallélisé par `ThreadPoolExecutor`, les orderbooks de sortie sont pre-fetched en `asyncio.gather`.

### 1.3 Modèle probabiliste

Student's t-distribution (df=10) pour les heavy tails — meilleur qu'une gaussienne pour les surprises météo. Le sigma est calibré empiriquement par location et horizon sur 2112+ échantillons historiques. Exemples de valeurs calibrées : NYC horizon 0 = 2.38°F, Seattle = 1.04°F, global = 1.64°F. Les facteurs saisonniers ajustent le sigma par mois.

Le Kalman filter (`kalman.py`) apprend dynamiquement le sigma par location+horizon à partir des trades résolus — c'est un avantage compétitif réel si alimenté correctement en données.

### 1.4 Gestion des risques

Couverture complète :

- **Circuit breaker** : daily loss limit, max positions/jour, cooldown après max loss, max open positions
- **Slippage adaptatif** : seuil proportionnel à l'edge (trades à haute EV tolèrent plus de slippage)
- **Correlation guard** : empêche l'empilement sur villes corrélées (NYC/Chicago même front froid)
- **Flip-flop detection** : bloque les trades si la probabilité oscille trop entre les runs
- **Kelly fractionnel** : quarter-Kelly (fraction=0.25) conservateur car les estimations de probabilité sont bruitées
- **Guard rails calibration** : `guard_rails.py` clampe tous les paramètres dans des bornes physiques (sigma 1-4°F, seasonal 0.5-2.0x, etc.)
- **Same-location penalty** : réduit le sizing si le bot a déjà une position sur la même ville/horizon
- **Min probability filter** : skip les buckets à trop faible probabilité

### 1.5 Logging & traçabilité

Structured trade log (`trade_log.json`) enregistrant prob_raw, prob_platt, price, sigma, horizon par trade — indispensable pour le re-fitting Platt et le tuning des seuils. Price snapshots enregistrés à chaque run (48643 snapshots au dernier comptage). Paper state complet avec forecasts historiques par ville/date.

---

## 2. Points faibles

### 2.1 CRITIQUE — Feedback loop non alimenté

Le fichier `feedback.py` existe et implémente un système EMA + AR(1) d'autocorrélation des erreurs de forecast. Le `feedback_state.json` est **vide** (`{"entries": {}}`). Le bot ne s'auto-corrige pas encore. C'est le problème numéro un : chaque trade perdu ne modifie aucun paramètre pour les trades suivants.

**Impact** : Le warm bias hivernal documenté (6/7 forecasts au-dessus du réel dans l'audit) continue de s'appliquer sans correction. Dallas winter bias mesuré à +2 à +9°F.

**Action** : Vérifier que `on_trade_resolved()` est bien appelé dans le pipeline de résolution. Le code est prêt dans `feedback.py`, il faut s'assurer que `paper_bridge.py` / `paper_trade.py` l'invoque quand un marché se résout.

### 2.2 CRITIQUE — Value traps systématiques

Positions actuelles en paper trading :

| Location | Bucket | Cost basis | Shares | Problème |
|----------|--------|-----------|--------|----------|
| NYC | 41°F or below | $0.04 | 50 | Forecast 43.7°F — 2.7°F au-dessus |
| Seattle | 39°F or below | $0.011 | 181.8 | Forecast 39.2°F — juste au-dessus, bucket extrême |
| NYC | 31°F or below | $0.01 | 75 | Forecast 38.8°F — 7.8°F au-dessus |
| Dallas | 70-71°F | $0.024 | 62 | Raisonnable (forecast 71.6°F) |
| NYC | 46°F or higher | $0.019 | 41 | Forecast 43°F — 3°F en dessous |

Le pattern est clair : le bot achète massivement des tail buckets à $0.01-$0.04 avec des probabilités estimées faibles mais un "edge" calculé. Le marché price ces buckets bas parce qu'ils sont effectivement improbables. La position "NYC 31°F or below" à $0.01 avec forecast 38.8°F est un pur value trap.

**Action** : Relever `min_probability` de la valeur actuelle à 0.25 minimum. Ajouter un filtre : si `|forecast - bucket_center| > 2 * sigma`, skip.

### 2.3 IMPORTANT — Platt scaling incohérent

Le trade log montre :

- NYC: `prob_raw=0.3391 → prob_platt=0.5696` (scaling actif, boost +68%)
- Atlanta: `prob_raw=0.183 → prob_platt=0.183` (scaling inactif)
- Miami: `prob_raw=0.2843 → prob_platt=0.2843` (scaling inactif)

L'application inégale du Platt scaling est problématique. Quand actif, il gonfle agressivement les probabilités. Pour NYC à $0.058, passer de 34% à 57% crée un edge apparent de 51 cents — complètement irréaliste pour un bucket weather. Le Platt scaling avec a=0.6205 sur des buckets mutuellement exclusifs brise la contrainte sum-to-1 (le README le mentionne comme un bug corrigé, mais les traces montrent qu'il est encore actif dans certains cas).

**Action** : Soit désactiver Platt complètement (il est déjà noté comme problématique dans le README), soit le re-calibrer avec la contrainte sum-to-1 appliquée post-scaling via normalisation.

### 2.4 IMPORTANT — Mean reversion sizing non branché

Le `PriceTracker` dans `mean_reversion.py` calcule un `sizing_multiplier` (0.5x à 1.5x) basé sur le Z-score rolling des prix de marché. Ce multiplier **n'est jamais appliqué** au sizing final dans `strategy.py`. Le p2.md le signale comme P0 critique.

**Action** : Dans la fonction d'exécution de `strategy.py`, multiplier `base_size` par `price_tracker.sizing_multiplier(...)` avant de placer l'ordre.

### 2.5 MOYEN — Villes internationales dégradées

Pour Seoul, London, Ankara, Wellington, São Paulo, Buenos Aires, Toronto, Paris : seul Open-Meteo est disponible (pas de NOAA, METAR limité). Le modèle perd la diversification des sources qui fait sa force sur les villes US. Le sigma calibré pour ces villes est potentiellement moins fiable.

Les logs montrent systématiquement `NOAA=N/A` pour ces villes et des spreads parfois élevés (Buenos Aires spread=9.3°F). Le bot devrait être plus conservateur sur les marchés internationaux ou augmenter le sigma minimum.

**Action** : Ajouter un facteur multiplicatif au sigma pour les villes sans NOAA (ex: `sigma *= 1.3`). Ou exiger un edge minimum plus élevé pour les villes internationales.

### 2.6 MOYEN — Positions NO sous-exploitées

Le code `score_buckets()` génère bien des scores pour le côté NO (lignes 277-292 de `strategy.py`), mais l'audit initial (`improve.md`) notait `side: str = "yes"  # TOUJOURS yes, jamais no`. Les traces récentes montrent une position NO sur Chicago ("54°F or higher", side=no), ce qui suggère que c'est partiellement corrigé. Cependant, la logique NO utilise `no_prob = 1.0 - prob` sans tenir compte du bucket "Other" qui existe souvent dans les events Polymarket.

**Action** : Détecter la présence d'un bucket "Other" dans l'event et ajuster `no_prob` en conséquence (ex: `no_prob *= 0.9` si "Other" présent).

### 2.7 MINEUR — Pas de daily P&L tracking visible

Les paper trades n'ont pas de `outcome` ni `pnl` renseignés dans `trade_log.json` (tous à `null`). Impossible de calculer un win rate, un Sharpe ratio, ou de valider la profitabilité du modèle sans résolution des trades.

**Action** : S'assurer que le pipeline de résolution remplit les champs `outcome` et `pnl` dans le trade log quand les marchés se résolvent.

---

## 3. Analyse des opportunités d'arbitrage

### 3.1 Arbitrage YES/NO (Méthode 2) — Faible potentiel

**Code** : `bot/signals.py`, fonction `detect_arbitrage()`

Le code vérifie que `best_ask_YES + best_ask_NO < 1.00` (buy-both) ou `best_bid_YES + best_bid_NO > 1.00` (sell-both). Seuil minimum : 20 bps après fees.

**Réalité** : Sur les weather markets Polymarket, les spreads YES/NO sont généralement efficients. Le maker/taker model et les market makers automatiques maintiennent YES+NO ≈ $1.00. Ce type d'arb se déclenche très rarement. Potentiel estimé : occasionnel, quelques cents par trade, pas assez fréquent pour en faire une stratégie.

### 3.2 Arbitrage multi-choice (Méthode 4) — Fort potentiel inexploité

**Code** : `bot/signals.py`, fonction `detect_multi_choice_arbitrage()`

Les weather markets sont des `negRisk` groups : un événement "NYC high temp Feb 17" contient 5-8 buckets mutuellement exclusifs (≤41°F, 42-43°F, 44-45°F, ..., ≥46°F). La somme des prix YES de tous les buckets devrait = $1.00 exactement.

**Opportunité** : Si la somme > $1.00, acheter tous les NO = profit garanti (un seul bucket peut gagner). Si la somme < $1.00, acheter tous les YES = profit garanti. Le code est **entièrement implémenté** avec déduction des fees (`fee_rate = config.polymarket_fee_bps / 10_000`).

**Problème** : Les logs ne montrent aucun déclenchement de cette méthode. Le scan multi-choice passe par `bot/strategy.py` → `scan_for_signals()` avec les `multi_choice_groups` détectés par Gamma. Mais la weather strategy dans `weather/strategy.py` ne passe PAS par ce pipeline — elle a son propre scoring de buckets. Les deux pipelines coexistent sans se parler.

**Taille de l'opportunité** : 14 villes × 1 event/jour × ~6 buckets = ~84 marchés par jour. En période de volatilité météo (modèles qui divergent, fronts froids), les buckets extrêmes sont souvent sous-pricés individuellement, ce qui pousse la somme au-dessus ou en dessous de $1.00. Des déviations de 2-5% existent régulièrement.

**Calcul exemple** :
- Somme YES d'un event = $1.03 (déviation +3%)
- 6 buckets → acheter les 6 NO coûte ~$5.97
- Payoff garanti = $6.00 (un NO gagne forcément)
- Profit brut = $0.03 par lot, moins fees (2% × 6 ordres)
- Profit net ≈ $0.03 - $0.12 = **négatif après fees sur petit écart**
- Seuil rentable après fees : déviation > ~5% sur un event à 6 buckets

**Action** :

1. Brancher le scan multi-choice directement dans le pipeline weather (pas seulement dans bot/strategy.py)
2. Calculer la somme YES à chaque run pour chaque event weather
3. Logger la déviation même quand elle est sous le seuil (pour identifier les patterns)
4. Ajuster le seuil `min_edge_bps` en tenant compte du nombre de buckets et du fee_rate réel

### 3.3 Arbitrage cross-temporel — Potentiel moyen, non implémenté

**Concept** : Si NYC demain = "44-45°F" à $0.20 et NYC après-demain = "44-45°F" à $0.15, avec un forecast stable entre les deux dates, il y a un edge directionnel. Le marché à J+2 devrait converger vers le marché à J+1 à mesure que l'horizon se réduit.

Ce n'est pas de l'arbitrage pur (pas risk-free) mais c'est un signal exploitable. Le bot a déjà les prix snapshottés (`price_snapshots.json`, 48643 entrées) et les forecasts par date — toute l'infrastructure est là.

**Action** : Créer un signal dans `signals.py` qui compare les prix d'un même bucket à des horizons différents. Si l'écart dépasse le decay attendu de sigma (la calibration fournit le ratio horizon 1 vs horizon 2), c'est un signal.

### 3.4 Arbitrage par convergence de modèles — Potentiel moyen

**Concept** : Quand NOAA et Open-Meteo divergent fortement (log: "Model disagreement: NOAA=79°F vs Open-Meteo=75°F"), le sigma est boosté et le bot s'abstient souvent. Mais si un des modèles a un track record nettement meilleur pour cette ville/saison (information dans `calibration.json` via `model_weights`), on peut parier dans la direction du modèle le plus fiable.

Le bot fait déjà une pondération par modèle dans l'ensemble forecast, mais il pourrait aller plus loin : si le modèle dominant (poids > 0.6) pointe vers un bucket et que le marché est pricé sur le consensus, c'est un edge.

**Action** : Ajouter un signal "model_conviction" : si le modèle le mieux calibré historiquement pour cette ville/saison disagree avec le marché, et que le poids du modèle > 0.6, c'est un entry signal avec sizing réduit (demi-Kelly).

---

## 4. Recommandations prioritaires

### P0 — Immédiat (cette semaine)

1. **Activer le feedback loop** : vérifier que `on_trade_resolved()` est appelé dans `paper_trade.py` quand les marchés se résolvent. Le code est prêt, il faut juste le brancher.
2. **Relever min_probability à 0.25** : stopper les value traps à $0.01. Ajouter un filtre distance forecast-bucket.
3. **Désactiver ou normaliser Platt scaling** : l'application incohérente crée de faux edges. Soit off partout, soit avec normalisation sum-to-1 post-scaling.

### P1 — Court terme (1-2 semaines)

4. **Brancher mean reversion sizing** : le multiplier est calculé mais pas appliqué.
5. **Activer le scan multi-choice** dans le pipeline weather pour détecter les arb structurels sur les events temperature.
6. **Logger la déviation sum-YES** à chaque run pour chaque event, même sans trader.
7. **Augmenter sigma pour villes internationales** sans NOAA (facteur 1.3x).

### P2 — Moyen terme (2-4 semaines)

8. **Implémenter le signal cross-temporel** (arb de convergence entre horizons).
9. **Ajouter le signal model_conviction** pour exploiter les divergences NOAA/Open-Meteo.
10. **Résoudre les trades dans le trade log** (remplir outcome/pnl) pour calculer win rate et Sharpe.
11. **Ajuster no_prob** pour les events avec bucket "Other".
12. **Dashboard** temps réel (Streamlit ou Flask) pour suivre positions, P&L, calibration drift.

---

## 5. Résumé

| Aspect | Note | Commentaire |
|--------|------|-------------|
| Architecture | A | Bridge pattern, séparation des concerns, tests solides |
| Pipeline météo | A | Triple source, ensemble pondéré, METAR real-time |
| Modèle probabiliste | B+ | Student's t + Kalman bien conçus, mais Platt problématique |
| Risk management | A- | Circuit breaker + correlation guard + Kelly, mais mean reversion non branché |
| Feedback & apprentissage | D | Code prêt mais non alimenté — le bot ne s'auto-corrige pas |
| Sélection de trades | C | Value traps sur tail buckets, pas assez sélectif |
| Arbitrage | C+ | 4 méthodes implémentées, 1 seule active, multi-choice inexploité |
| Profitabilité estimée | ? | Impossible à évaluer — aucun trade résolu avec P&L dans les logs |

**Verdict** : Le bot est techniquement excellent (architecture, robustesse, risk management) mais opérationnellement sous-optimal. Les deux problèmes majeurs sont le feedback loop vide et la sélection de trades trop agressive sur les extrêmes. L'opportunité d'arbitrage multi-choice est la piste la plus prometteuse à court terme — le code est prêt, il suffit de l'activer dans le bon pipeline.
