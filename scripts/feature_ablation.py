"""Which observation features let the HMM recover the TRUE regimes?

On a simulated market with known ground truth (K = 3), for each feature set:
  * the K selected by BIC,
  * the Viterbi decoding accuracy with K = 3 (best label permutation),
averaged over several simulated paths. Results -> results/feature_ablation.csv
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from hmm_crypto.features import build_features
from hmm_crypto.hmm import GaussianHMM
from hmm_crypto.selection import Standardizer, select_n_states
from hmm_crypto.simulate import simulate_market

FEATURE_SETS = [
    ["log_ret"],
    ["log_ret", "vol_20"],
    ["log_ret", "park_vol_20"],
    ["log_ret", "log_range"],
    ["log_ret", "log_range", "volume_z"],
]


def accuracy(z, t, K):
    C = np.zeros((K, t.max() + 1))
    np.add.at(C, (z, t), 1)
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(t)


rows = []
for seed in range(3):
    df, truth = simulate_market(2500, random_state=seed)
    f = build_features(df).dropna()
    t = truth.loc[f.index].to_numpy()
    for cols in FEATURE_SETS:
        X = Standardizer().fit_transform(f[cols].to_numpy())
        sel = select_n_states(X, range(2, 7), n_init=3, random_state=seed)
        m = GaussianHMM(3, n_init=5, random_state=seed).fit(X)
        rows.append({"features": " + ".join(cols), "seed": seed,
                     "bic_K": int(sel["bic"].idxmin()),
                     "cv_K": int(sel["val_ll_per_obs"].idxmax()),
                     "accuracy_K3": accuracy(m.predict(X), t, 3)})
        print(rows[-1])

res = pd.DataFrame(rows).groupby("features", sort=False).agg(
    bic_K=("bic_K", list), cv_K=("cv_K", list), accuracy_K3=("accuracy_K3", "mean"))
print("\n", res.round(3).to_string())
res.to_csv("results/feature_ablation.csv")
