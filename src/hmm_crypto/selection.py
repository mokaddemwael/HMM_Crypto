"""Model selection (number of hidden states) and economic labelling of regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm import GaussianHMM


class Standardizer:
    """z-score scaler. Fitted on the training window ONLY (no look-ahead)."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def select_n_states(
    X: np.ndarray,
    n_states_range=range(2, 7),
    covariance_type: str = "full",
    n_init: int = 5,
    random_state: int = 0,
    val_fraction: float = 0.3,
) -> pd.DataFrame:
    """Fit one HMM per K and report log-likelihood, #params, AIC, BIC and a
    hold-out validation score.

    BIC = -2 ln L + p ln T   (consistent *if the model is well specified*)
    AIC = -2 ln L + 2 p      (tends to over-select states)
    val_ll_per_obs           mean log-likelihood on the last `val_fraction`
                             of the sample for a model fitted on the first
                             part (temporal hold-out, no shuffling).

    Under misspecification (heavy tails, autocorrelated features) BIC and AIC
    keep adding states to approximate the non-Gaussian density; the hold-out
    criterion is more robust because extra states that only fit noise do not
    generalise. See scripts/feature_ablation.py.
    """
    T = len(X)
    split = int(T * (1 - val_fraction))
    rows = []
    for k in n_states_range:
        m = GaussianHMM(k, covariance_type=covariance_type, n_init=n_init,
                        random_state=random_state).fit(X)
        row = {
            "n_states": k,
            "log_likelihood": m.score(X),
            "n_params": m.n_parameters(),
            "aic": m.aic(X),
            "bic": m.bic(X),
            "converged": m.history_.converged,
        }
        if val_fraction > 0:
            mv = GaussianHMM(k, covariance_type=covariance_type, n_init=n_init,
                             random_state=random_state).fit(X[:split])
            # condition on the training part, score only the validation block
            row["val_ll_per_obs"] = (mv.score(X) - mv.score(X[:split])) / (T - split)
        rows.append(row)
    return pd.DataFrame(rows).set_index("n_states")


def choose_k(sel: pd.DataFrame, criterion: str = "bic") -> int:
    if criterion in ("bic", "aic"):
        return int(sel[criterion].idxmin())
    if criterion == "cv":
        return int(sel["val_ll_per_obs"].idxmax())
    raise ValueError(criterion)


def regime_statistics(
    model: GaussianHMM, X: np.ndarray, returns: np.ndarray, periods_per_year: int = 365
) -> pd.DataFrame:
    """Posterior-weighted statistics of the *raw* log-returns in each state."""
    gamma = model.predict_proba(X)
    w = gamma / gamma.sum(axis=0, keepdims=True)
    mu = w.T @ returns
    var = np.einsum("tk,tk->k", w, (returns[:, None] - mu[None]) ** 2)
    sd = np.sqrt(var)
    return pd.DataFrame({
        "mean_daily_ret": mu,
        "vol_daily": sd,
        "ann_return": mu * periods_per_year,
        "ann_vol": sd * np.sqrt(periods_per_year),
        "frequency": gamma.mean(axis=0),
        "expected_duration_days": model.expected_durations(),
    })


VOL_TIERS = {
    1: ["unique"],
    2: ["calme", "turbulent"],
    3: ["calme", "normal", "turbulent"],
    4: ["calme", "normal", "agite", "turbulent"],
    5: ["tres_calme", "calme", "normal", "agite", "turbulent"],
    6: ["tres_calme", "calme", "normal", "agite", "turbulent", "crise"],
    7: ["tres_calme", "calme", "normal", "agite", "turbulent", "crise", "crise_extreme"],
}


def label_regimes(stats: pd.DataFrame) -> list[str]:
    """Name anonymous states by their volatility rank.

    Why volatility and not "bull/bear"? On real BTC data every state found by
    the HMM has a *positive* mean return: what the model separates robustly
    is the volatility level (a well-known stylised fact: volatility clusters
    and is predictable, the sign of daily returns is not). Ranking by
    volatility also gives labels that are stable across walk-forward refits
    (states are only identified up to a permutation). The mean return of each
    regime is reported next to its name in `regime_statistics`.
    """
    K = len(stats)
    names = VOL_TIERS.get(K, [f"vol_{i + 1}" for i in range(K)])
    labels = [""] * K
    for rank, i in enumerate(stats["vol_daily"].to_numpy().argsort()):
        labels[i] = names[rank]
    return labels
