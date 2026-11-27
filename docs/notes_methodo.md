# Notes méthodologiques : résultats à exploiter dans le rapport

Ces résultats sont reproductibles avec `scripts/feature_ablation.py` et `scripts/run_pipeline.py`. Données BTC : Yahoo, 2017-01-01 → 2026-11-18.

## 1. Validation sur marché simulé (vérité terrain connue, K = 3)

| Features d'observation | K choisi par BIC | K choisi par validation hors-échantillon | Précision de Viterbi (K = 3) |
|---|---|---|---|
| log_ret | 2, 2, 2 | 2, 2, 4 | 69 % |
| log_ret + vol_20 | 6, 6, 6 | 6, 6, 6 | 56 % |
| log_ret + park_vol_20 | 6, 6, 6 | 6, 6, 6 | **73 %** |
| log_ret + log_range | 6, 5, 6 | 6, 5, 6 | 42 % |
| log_ret + log_range + volume_z | 5, 5, 5 | 6, 5, 5 | 42 % |

(3 trajectoires simulées par ligne ; chaque liste donne le K retenu sur chacune.)

**Constat 1 : sur données synthétiques gaussiennes, les algorithmes sont exacts.** Les tests unitaires le confirment : énumération exhaustive, retour des vrais paramètres, et K = 3 retrouvé par le BIC.

**Constat 2 : avec des features réalistes, tous les critères surestiment K.**
- **Features glissantes autocorrélées** (`vol_20`) : elles violent l'hypothèse d'indépendance conditionnelle x_t ⊥ x_{t-1} | s_t.
- **Émissions non gaussiennes** (`log_range` est asymétrique) : le HMM gaussien empile des états pour approximer la densité, comme le ferait un mélange.
- **La validation hors-échantillon n'y échappe pas** : les états en trop améliorent *réellement* la densité prédictive. Ce n'est donc pas du sur-apprentissage au sens strict, c'est une **mauvaise spécification** du modèle.

**Décision :** K = 3 est fixé pour l'interprétabilité. Les critères sont conservés comme diagnostic (`model_selection.png`).

**Perspective :** des émissions Student-t ou des mélanges de gaussiennes (GMM-HMM), ou encore un HMM semi-markovien pour les durées.

## 2. Interprétation économique sur BTC (K = 3, ex-post)

| Régime | Rendement annualisé | Volatilité annualisée | Fréquence | Durée moyenne |
|---|---|---|---|---|
| calme | 31 % | 39 % | 42 % | 48 j |
| normal | 56 % | 60 % | 35 % | 25 j |
| turbulent | 41 % | 106 % | 23 % | 37 j |

**Constat 3 : le HMM identifie des régimes de volatilité, pas de direction.** Les trois régimes ont un rendement moyen positif.
- Le régime **turbulent** couvre la bulle et le krach de 2017-2018, le krach Covid de mars 2020 et le sommet de 2021.
- Le régime **calme** domine de 2023 à 2026.

C'est cohérent avec deux faits stylisés : la volatilité forme des grappes (*volatility clustering*) et reste prévisible, alors que le signe des rendements journaliers ne l'est quasiment pas. → Le nommage « bull/bear/panic » proposé dans le sujet est à **nuancer** dans le rapport.

## 3. Backtest hors-échantillon (walk-forward, réentraînement tous les 30 j, fenêtre de 730 j, frais de 10 bps)

| BTC 2019-2026 | Stratégie directionnelle | Stratégie moyenne-variance | Buy & Hold |
|---|---|---|---|
| CAGR | 24 % | 18 % | 45 % |
| Volatilité | 52 % | 38 % | 61 % |
| Sharpe | 0,68 | 0,62 | **0,92** |
| Drawdown maximum | −77 % | **−62 %** | −77 % |

**Constat 4 : aucune des deux stratégies HMM ne bat le Buy & Hold en Sharpe sur BTC.** La stratégie moyenne-variance réduit en revanche la volatilité et le drawdown maximum.

C'est un résultat **honnête** :
- le backtest est strictement causal (testé) ;
- les frais sont inclus ;
- les paramètres n'ont pas été optimisés sur la période de test.

Une surperformance obtenue en ajustant les seuils sur cette même période serait du *data snooping*. Il faudra le signaler au jury si d'autres variantes sont essayées.

**Pistes d'amélioration (à tester sans fuite d'information) :**
- réentraînement plus fréquent ;
- seuil de probabilité avec hystérésis, pour réduire le turnover ;
- multi-actifs (ETH) ;
- features de marché supplémentaires (*funding rate*, open interest) ;
- extension LSTM-HMM (transitions dépendant du contexte).
