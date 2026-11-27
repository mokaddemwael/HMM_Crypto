# HMM_Crypto
Modélisation des régimes de marché des cryptomonnaies par HMM

Projet Math 5INFINI : modélisation et prédiction des régimes de marché (haussier, baissier, panique) par HMM, puis application au trading algorithmique.

> Le script `collecte_donnees.py` et le notebook `notebooks/HMM_Crypto_CRISP_DM.ipynb` (branche `main`) sont conservés tels quels. La branche `melek` ajoute un package Python testé (`src/hmm_crypto/`), décrit ci-dessous.

## Ce qui est implémenté

| Module | Rôle |
|---|---|
| `src/hmm_crypto/hmm.py` | **HMM gaussien codé à la main en NumPy** : Forward-Backward normalisé (Rabiner), Viterbi en log-space, Baum-Welch (EM) avec plusieurs initialisations k-means++, AIC/BIC, prévision du régime à t+1, distribution stationnaire, durées moyennes des régimes |
| `src/hmm_crypto/features.py` | Rendements log, volatilité réalisée, volatilité de Parkinson, amplitude journalière, RSI (Wilder), MACD, z-score du volume. **Toutes les features sont causales** |
| `src/hmm_crypto/data.py` | Chargeur CSV/Excel tolérant (formats Binance, Yahoo, Kaggle, noms de colonnes en français), nettoyage, téléchargement Yahoo avec cache |
| `src/hmm_crypto/simulate.py` | Marché synthétique à 3 régimes connus (vérité terrain), pour valider la méthode |
| `src/hmm_crypto/selection.py` | Choix de K par AIC, BIC ou **vraisemblance hors-échantillon**, statistiques et noms économiques des régimes |
| `src/hmm_crypto/strategy.py` | **Walk-forward** : réentraînement périodique sur le passé seulement, probabilités filtrées, P(sₜ₊₁ \| x₁:ₜ) = αₜ·A |
| `src/hmm_crypto/backtest.py` | Backtest vectorisé avec frais de transaction, Sharpe, Sortino, drawdown maximum, Calmar ; comparaison au Buy & Hold |
| `src/hmm_crypto/plotting.py` | Figures pour le rapport |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev]"
```

Dans VS Code, sélectionnez l'interpréteur `.venv` (Ctrl+Shift+P, puis « Python: Select Interpreter »).

## Utilisation

```bash
pytest                                                    # 21 tests (algorithmes, causalité, backtest)
python scripts/run_pipeline.py --source sim               # marché simulé (vérité connue)
python scripts/run_pipeline.py --source yahoo --ticker BTC-USD
python scripts/run_pipeline.py --source csv --csv data/raw/ma_base.csv --name base_encadrant
python scripts/feature_ablation.py                        # quelles features permettent de retrouver les vrais régimes ?
```

Options utiles : `--features log_ret,park_vol_20`, `--n-states 3`, `--select cv|bic|aic`, `--train-window 730`, `--refit-every 30`, `--cost-bps 10`, `--allow-short`.

Les sorties sont écrites dans `results/<nom>/` : figures PNG, tableaux CSV et `summary.md`.

## Garanties méthodologiques (vérifiées par les tests)

1. **Exactitude des algorithmes** : la vraisemblance (forward), les postérieurs (forward-backward) et le chemin de Viterbi sont comparés à l'**énumération exhaustive** des Kᵀ chemins possibles.
2. **Convergence d'EM** : la log-vraisemblance ne décroît jamais d'une itération à l'autre.
3. **Identifiabilité** : Baum-Welch retrouve les paramètres d'un HMM simulé, à une permutation des états près.
4. **Absence de fuite d'information future** (*look-ahead*) : les features, le filtrage et le walk-forward sont testés en **modifiant le futur** et en vérifiant que le passé ne change pas.
5. **Décalage temporel du backtest** : une position décidée à la clôture du jour t ne touche que le rendement de t+1.

## Points de vigilance (à discuter dans le rapport)

- Le **lissage** P(sₜ \| x₁:T) et **Viterbi** utilisent les données futures. Ils servent à l'**interprétation ex-post**, jamais à générer des signaux de trading. Seul le **filtrage** P(sₜ \| x₁:ₜ) est causal.
- **Le BIC surestime K** dès que le modèle est mal spécifié : features glissantes autocorrélées, émissions non gaussiennes (queues épaisses des cryptos). Voir `scripts/feature_ablation.py`.
- Les états sont **identifiés à une permutation près** et peuvent donc changer d'indice à chaque réentraînement. Ils sont renommés par leurs statistiques (rendement moyen, volatilité) à chaque fois.
