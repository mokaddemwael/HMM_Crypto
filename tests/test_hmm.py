"""Correctness tests for the from-scratch HMM.

Strategy: on tiny problems, compare against BRUTE-FORCE enumeration of all
K^T state paths (the definition itself); on larger simulated problems, check
that Baum-Welch recovers the true parameters.
"""

import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal

from hmm_crypto.hmm import GaussianHMM


def small_model():
    m = GaussianHMM(n_states=3)
    m.set_params(
        startprob=[0.5, 0.3, 0.2],
        transmat=[[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
        means=[[0.0, 0.0], [1.0, -1.0], [-2.0, 1.5]],
        covars=[np.eye(2) * 0.5, [[1.0, 0.3], [0.3, 0.8]], np.eye(2) * 2.0],
    )
    return m


def brute_force(m, X):
    """Joint probability of every path -> likelihood, posteriors, MAP path."""
    K, T = m.n_states, len(X)
    dens = np.array([[multivariate_normal(m.means_[k], m.covars_[k]).pdf(x) for k in range(K)]
                     for x in X])
    joint = {}
    for path in itertools.product(range(K), repeat=T):
        p = m.startprob_[path[0]] * dens[0, path[0]]
        for t in range(1, T):
            p *= m.transmat_[path[t - 1], path[t]] * dens[t, path[t]]
        joint[path] = p
    Z = sum(joint.values())
    gamma = np.zeros((T, K))
    for path, p in joint.items():
        gamma[np.arange(T), path] += p / Z
    best = max(joint, key=joint.get)
    return np.log(Z), gamma, np.array(best), np.log(joint[best])


@pytest.fixture
def tiny():
    m = small_model()
    X, _ = m.sample(6, random_state=1)
    return m, X


def test_forward_likelihood_matches_brute_force(tiny):
    m, X = tiny
    ll_bf, *_ = brute_force(m, X)
    assert m.score(X) == pytest.approx(ll_bf, rel=1e-10)


def test_smoothed_posteriors_match_brute_force(tiny):
    m, X = tiny
    _, gamma_bf, _, _ = brute_force(m, X)
    np.testing.assert_allclose(m.predict_proba(X), gamma_bf, atol=1e-10)


def test_viterbi_matches_brute_force(tiny):
    m, X = tiny
    *_, path_bf, lp_bf = brute_force(m, X)
    path, lp = m.decode(X)
    np.testing.assert_array_equal(path, path_bf)
    assert lp == pytest.approx(lp_bf, rel=1e-10)


def test_filtering_is_causal_and_normalised(tiny):
    m, X = tiny
    f = m.filter_proba(X)
    np.testing.assert_allclose(f.sum(axis=1), 1.0)
    X2 = X.copy()
    X2[4:] += 10.0  # change the future
    np.testing.assert_allclose(m.filter_proba(X2)[:4], f[:4])
    # last filtered = last smoothed
    np.testing.assert_allclose(f[-1], m.predict_proba(X)[-1], atol=1e-12)


def test_next_state_forecast(tiny):
    m, X = tiny
    p = m.predict_next_proba(X)
    np.testing.assert_allclose(p, m.filter_proba(X) @ m.transmat_)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_long_sequence_no_underflow():
    m = small_model()
    X, _ = m.sample(5000, random_state=0)
    ll = m.score(X)
    assert np.isfinite(ll) and ll < 0


def test_em_log_likelihood_is_monotone():
    X, _ = small_model().sample(800, random_state=2)
    m = GaussianHMM(3, n_init=1, n_iter=60, tol=-np.inf, min_covar=1e-9,
                    transition_prior=0.0, random_state=0).fit(X)
    lls = np.array(m.history_.log_likelihoods)
    assert np.all(np.diff(lls) >= -1e-6 * np.abs(lls[:-1]))


def test_baum_welch_recovers_parameters():
    true = small_model()
    X, z = true.sample(4000, random_state=3)
    m = GaussianHMM(3, n_init=5, random_state=0).fit(X)
    # states are identified up to a permutation -> match by means
    cost = ((true.means_[:, None, :] - m.means_[None]) ** 2).sum(-1)
    _, perm = linear_sum_assignment(cost)
    np.testing.assert_allclose(m.means_[perm], true.means_, atol=0.15)
    np.testing.assert_allclose(m.transmat_[np.ix_(perm, perm)], true.transmat_, atol=0.06)
    inv = np.argsort(perm)
    acc = np.mean(inv[m.predict(X)] == z)
    assert acc > 0.85


def test_bic_selects_true_number_of_states():
    X, _ = small_model().sample(2500, random_state=4)
    bics = {k: GaussianHMM(k, n_init=3, random_state=0).fit(X).bic(X) for k in (2, 3, 4)}
    assert min(bics, key=bics.get) == 3


def test_stationary_distribution_and_durations():
    m = small_model()
    pi = m.stationary_distribution()
    np.testing.assert_allclose(pi @ m.transmat_, pi, atol=1e-12)
    np.testing.assert_allclose(m.expected_durations(), [5.0, 5.0, 2.5])


def test_matches_hmmlearn_if_available(tiny):
    hmmlearn = pytest.importorskip("hmmlearn.hmm")
    m, X = tiny
    ref = hmmlearn.GaussianHMM(3, covariance_type="full", init_params="", params="")
    ref.startprob_, ref.transmat_ = m.startprob_, m.transmat_
    ref.means_, ref.covars_ = m.means_, m.covars_
    assert m.score(X) == pytest.approx(ref.score(X), rel=1e-8)
