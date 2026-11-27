import numpy as np
import pandas as pd

from hmm_crypto.features import build_features
from hmm_crypto.simulate import simulate_market
from hmm_crypto.strategy import WalkForwardConfig, signal_from_forecast, walk_forward_forecast

CFG = WalkForwardConfig(n_states=3, train_window=300, refit_every=60, n_init=1)


def test_walk_forward_has_no_look_ahead():
    df, _ = simulate_market(700, random_state=5)
    cut = 550
    f1 = walk_forward_forecast(build_features(df), CFG)

    altered = df.copy()
    altered.iloc[cut:, :4] *= np.exp(np.linspace(0, 2, len(df) - cut))[:, None]
    f2 = walk_forward_forecast(build_features(altered), CFG)

    past = f1.index < df.index[cut]
    a, b = f1[past], f2.loc[f1.index[past]]
    # Later refits may introduce a new label (e.g. "bull_2") -> extra column,
    # which must be identically 0 on past dates.
    common = a.columns.intersection(b.columns)
    pd.testing.assert_frame_equal(a[common], b[common])
    for extra in a.columns.symmetric_difference(b.columns):
        src = a if extra in a.columns else b
        assert (src[extra] == 0).all()


def test_forecast_probabilities_sum_to_one():
    df, _ = simulate_market(600, random_state=6)
    f = walk_forward_forecast(build_features(df), CFG)
    p = f.filter(like="p_next_").sum(axis=1)
    np.testing.assert_allclose(p, 1.0)
    pos = signal_from_forecast(f, allow_short=True)
    assert set(pos.unique()) <= {-1.0, 0.0, 1.0}
