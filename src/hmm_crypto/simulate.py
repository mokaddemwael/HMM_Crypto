"""Synthetic regime-switching crypto market.

Why simulate?  On real data the "true" regimes are unknown, so we cannot
measure whether the HMM recovers them. On simulated data the ground truth is
known: we can check parameter recovery, decoding accuracy and whether the
trading strategy captures the edge that is *by construction* present.

Default regimes (daily, crypto-like orders of magnitude):
    0 bull   : mu = +0.30 %/day, sigma = 2.5 %
    1 bear   : mu = -0.25 %/day, sigma = 3.5 %
    2 panic  : mu = -1.00 %/day, sigma = 7.0 %
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RegimeSpec:
    names: list[str] = field(default_factory=lambda: ["bull", "bear", "panic"])
    mu: np.ndarray = field(default_factory=lambda: np.array([0.003, -0.0025, -0.010]))
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.025, 0.035, 0.070]))
    transmat: np.ndarray = field(default_factory=lambda: np.array([
        [0.985, 0.012, 0.003],
        [0.020, 0.970, 0.010],
        [0.050, 0.100, 0.850],
    ]))
    volume_level: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.2, 2.5]))


def simulate_market(
    n_days: int = 2000,
    spec: RegimeSpec | None = None,
    start_price: float = 10_000.0,
    start_date: str = "2018-01-01",
    random_state: int | None = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (ohlcv, true_regimes)."""
    spec = spec or RegimeSpec()
    rng = np.random.default_rng(random_state)
    K = len(spec.mu)

    states = np.empty(n_days, dtype=int)
    states[0] = 0
    for t in range(1, n_days):
        states[t] = rng.choice(K, p=spec.transmat[states[t - 1]])

    r = rng.normal(spec.mu[states], spec.sigma[states])
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.r_[start_price, close[:-1]]
    # Intraday range scales with regime volatility
    span = np.abs(rng.normal(0, spec.sigma[states] * 0.6))
    high = np.maximum(open_, close) * np.exp(span)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, spec.sigma[states] * 0.6)))
    volume = 1e6 * spec.volume_level[states] * rng.lognormal(0, 0.3, n_days)

    idx = pd.date_range(start_date, periods=n_days, freq="D", name="date")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    return df, pd.Series(states, index=idx, name="true_regime")
