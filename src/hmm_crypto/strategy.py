"""Walk-forward regime forecasting and signal generation.

Causality contract (the #1 source of fake performance in student projects):
  * the scaler and the HMM are fitted on data strictly before t0;
  * from t0 onward only the *filtered* probability P(s_t | x_{1:t}) is used
    (forward pass), never the smoothed one nor Viterbi on the full sample;
  * the decision taken at the close of day t is applied to the return of t+1
    (the shift is done in `backtest.py`).
`tests/test_strategy.py` checks this by altering future data and verifying
that past signals do not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hmm import GaussianHMM
from .selection import Standardizer, label_regimes, regime_statistics


@dataclass
class WalkForwardConfig:
    feature_cols: tuple[str, ...] = ("log_ret", "vol_20")
    return_col: str = "log_ret"
    n_states: int = 3
    covariance_type: str = "full"
    train_window: int = 730           # days of history used to fit (rolling)
    expanding: bool = False           # True -> use all history since the start
    refit_every: int = 30             # re-estimate the model every N days
    n_init: int = 3
    random_state: int = 0


def walk_forward_forecast(features: pd.DataFrame, cfg: WalkForwardConfig) -> pd.DataFrame:
    """Out-of-sample one-step-ahead regime forecasts.

    Returns, for every out-of-sample date t, columns:
      p_next_<label>   P(s_{t+1} = label | x_{1:t})
      exp_ret_next     sum_j P(s_{t+1}=j | x_{1:t}) * E[r | s=j]   (expected log-return)
      regime_now       most probable current regime (filtered)
    """
    cols = list(cfg.feature_cols)
    data = features[cols + ([cfg.return_col] if cfg.return_col not in cols else [])].dropna()
    X_all = data[cols].to_numpy()
    r_all = data[cfg.return_col].to_numpy()
    T = len(data)
    if T <= cfg.train_window:
        raise ValueError(f"Need more than train_window={cfg.train_window} rows, got {T}")

    out = []
    for t0 in range(cfg.train_window, T, cfg.refit_every):
        t_end = min(t0 + cfg.refit_every, T)
        start = 0 if cfg.expanding else t0 - cfg.train_window

        scaler = Standardizer().fit(X_all[start:t0])
        X_train = scaler.transform(X_all[start:t0])
        model = GaussianHMM(
            cfg.n_states, covariance_type=cfg.covariance_type,
            n_init=cfg.n_init, random_state=cfg.random_state,
        ).fit(X_train)

        stats = regime_statistics(model, X_train, r_all[start:t0])
        labels = label_regimes(stats)
        mu = stats["mean_daily_ret"].to_numpy()
        var = stats["vol_daily"].to_numpy() ** 2

        # Forward filter over [start, t_end): alpha_t depends only on x_{start..t}
        filt = model.filter_proba(scaler.transform(X_all[start:t_end]))[t0 - start:]
        p_next = filt @ model.transmat_

        block = pd.DataFrame(index=data.index[t0:t_end])
        for j, lab in enumerate(labels):
            block[f"p_next_{lab}"] = block.get(f"p_next_{lab}", 0.0) + p_next[:, j]
        block["exp_ret_next"] = p_next @ mu
        # Variance of the predictive mixture: E[sigma^2 + mu^2] - E[mu]^2
        block["exp_var_next"] = p_next @ (var + mu**2) - (p_next @ mu) ** 2
        block["regime_now"] = [labels[j] for j in filt.argmax(axis=1)]
        block["refit_date"] = data.index[t0]
        out.append(block)

    res = pd.concat(out).fillna(0.0)
    return res


STRATEGIES = ("expected_return", "mean_variance")


def signal_from_forecast(
    forecast: pd.DataFrame,
    method: str = "expected_return",
    threshold: float = 0.0,
    allow_short: bool = False,
    risk_aversion: float = 2.0,
) -> pd.Series:
    """Position decided at the close of t (to be applied to r_{t+1}).

    expected_return  (directional)
        +1 if E[r_{t+1}] > threshold, -1 if < -threshold (when shorting is
        allowed), 0 otherwise.
    mean_variance    (volatility timing, Merton / Kelly fraction)
        w_t = E[r_{t+1}] / (gamma * Var[r_{t+1}]), clipped to [0, 1]
        (or [-1, 1] with shorting). Exposure shrinks when the HMM forecasts
        a turbulent regime -- cf. Moreira & Muir (2017), "Volatility-Managed
        Portfolios", Journal of Finance.
    """
    e = forecast["exp_ret_next"]
    if method == "expected_return":
        pos = pd.Series(0.0, index=forecast.index)
        pos[e > threshold] = 1.0
        if allow_short:
            pos[e < -threshold] = -1.0
    elif method == "mean_variance":
        w = e / (risk_aversion * forecast["exp_var_next"])
        pos = w.clip(-1.0 if allow_short else 0.0, 1.0)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {STRATEGIES}")
    return pos.rename("position")
