import numpy as np
import pandas as pd
import pytest

from hmm_crypto.features import build_features, macd, rsi
from hmm_crypto.simulate import simulate_market


@pytest.fixture
def market():
    df, _ = simulate_market(400, random_state=0)
    return df


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up).dropna().eq(100).all()
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi(down).dropna().eq(0).all()
    noisy = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, .02, 300))))
    r = rsi(noisy).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_macd_definition(market):
    c = market["close"]
    m = macd(c)
    line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    np.testing.assert_allclose(m["macd"].dropna(), line.loc[m["macd"].dropna().index])


def test_features_are_causal(market):
    """Changing the future must not change any feature in the past."""
    f1 = build_features(market)
    altered = market.copy()
    altered.iloc[300:, :4] *= 3.0
    altered.iloc[300:, 4] *= 10.0
    f2 = build_features(altered)
    pd.testing.assert_frame_equal(f1.iloc[:300], f2.iloc[:300])


def test_feature_warmup_then_complete(market):
    f = build_features(market).dropna()
    assert len(f) > 350
    assert np.isfinite(f.to_numpy()).all()
