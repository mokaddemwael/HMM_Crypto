"""Gaussian Hidden Markov Model implemented from scratch (NumPy only).

Notation (Rabiner, 1989)
------------------------
K        number of hidden states
T        length of the observation sequence
d        dimension of each observation x_t in R^d
pi       initial distribution,             pi_i   = P(s_1 = i)
A        transition matrix,                A_ij   = P(s_{t+1} = j | s_t = i)
b_j(x)   emission density,                 b_j(x) = N(x; mu_j, Sigma_j)

Naive alpha/beta recursions underflow for long series (T ~ thousands of daily
candles). Forward-Backward therefore uses Rabiner's scaling (normalised alpha,
with emissions shifted by their row max); Viterbi works in log-space.

Algorithms
----------
forward           log alpha_t(j) = log b_j(x_t) + LSE_i [log alpha_{t-1}(i) + log A_ij]
backward          log beta_t(i)  = LSE_j [log A_ij + log b_j(x_{t+1}) + log beta_{t+1}(j)]
posteriors        gamma_t(i) = P(s_t=i | x_{1:T}),  xi_t(i,j) = P(s_t=i, s_{t+1}=j | x_{1:T})
Viterbi           argmax_{s_{1:T}} P(s_{1:T} | x_{1:T})  (dynamic programming)
Baum-Welch (EM)   maximises the log-likelihood; guaranteed non-decreasing at each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

CovarianceType = Literal["full", "diag"]

_LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class FitHistory:
    """Trace of an EM run (useful for the report: convergence plots)."""

    log_likelihoods: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


class GaussianHMM:
    """Hidden Markov Model with multivariate Gaussian emissions.

    Parameters
    ----------
    n_states : int
        Number of hidden regimes K.
    covariance_type : {"full", "diag"}
        Structure of each emission covariance matrix.
    n_iter : int
        Maximum number of Baum-Welch iterations per restart.
    tol : float
        Convergence threshold on the log-likelihood improvement.
    n_init : int
        Number of random restarts; the solution with the best log-likelihood
        is kept (EM only finds a local optimum).
    min_covar : float
        Ridge added to covariance diagonals (prevents singular matrices when a
        state collapses onto few points).
    transition_prior : float
        Dirichlet pseudo-count added to transition counts (MAP smoothing);
        avoids zero probabilities in A.
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: CovarianceType = "full",
        n_iter: int = 200,
        tol: float = 1e-4,
        n_init: int = 5,
        min_covar: float = 1e-4,
        transition_prior: float = 1e-2,
        random_state: int | None = None,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if covariance_type not in ("full", "diag"):
            raise ValueError("covariance_type must be 'full' or 'diag'")
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.min_covar = min_covar
        self.transition_prior = transition_prior
        self.random_state = random_state

        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None  # always stored as (K, d, d)
        self.history_: FitHistory | None = None
        self.n_features_: int | None = None

    # ------------------------------------------------------------------ #
    # Parameter management
    # ------------------------------------------------------------------ #
    def set_params(
        self,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
    ) -> "GaussianHMM":
        """Set parameters manually (used for simulation and unit tests)."""
        startprob = np.asarray(startprob, dtype=float)
        transmat = np.asarray(transmat, dtype=float)
        means = np.atleast_2d(np.asarray(means, dtype=float))
        covars = np.asarray(covars, dtype=float)
        K, d = means.shape
        if covars.ndim == 2:  # diagonal given as (K, d)
            covars = np.stack([np.diag(c) for c in covars])
        if startprob.shape != (K,) or transmat.shape != (K, K) or covars.shape != (K, d, d):
            raise ValueError("Inconsistent parameter shapes")
        if not np.allclose(startprob.sum(), 1) or not np.allclose(transmat.sum(axis=1), 1):
            raise ValueError("startprob and rows of transmat must sum to 1")
        self.n_states, self.n_features_ = K, d
        self.startprob_, self.transmat_ = startprob, transmat
        self.means_, self.covars_ = means, covars
        return self

    def _check_fitted(self) -> None:
        if self.means_ is None:
            raise RuntimeError("Model is not fitted. Call fit() or set_params() first.")

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or inf; clean the features first.")
        return X

    # ------------------------------------------------------------------ #
    # Emissions
    # ------------------------------------------------------------------ #
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """log b_j(x_t) for all t, j  ->  array (T, K).

        Uses the Cholesky factor L of Sigma_j:
            log N(x; mu, Sigma) = -1/2 [d log 2pi + 2 sum log diag(L) + ||L^{-1}(x-mu)||^2]
        """
        T, d = X.shape
        out = np.empty((T, self.n_states))
        for j in range(self.n_states):
            L = np.linalg.cholesky(self.covars_[j])
            diff = X - self.means_[j]
            z = np.linalg.solve(L, diff.T)  # (d, T)
            maha = np.sum(z**2, axis=0)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            out[:, j] = -0.5 * (d * _LOG_2PI + log_det + maha)
        return out

    # ------------------------------------------------------------------ #
    # Forward / Backward
    # ------------------------------------------------------------------ #
    # Implementation note: the recursions below use Rabiner's *scaling*
    # technique, which is mathematically equivalent to the log-space version
    # but ~5x faster in NumPy. Each emission row is shifted by its max
    # (b~_t = exp(log b_t - c_t) <= 1) and alpha_t is renormalised by
    # s_t = sum_j alpha~_t(j), so nothing under/overflows and
    #     log P(x_{1:T}) = sum_t (log s_t + c_t).
    def _forward_scaled(self, log_b: np.ndarray):
        T, K = log_b.shape
        c = log_b.max(axis=1)
        b = np.exp(log_b - c[:, None])
        A = self.transmat_
        alpha = np.empty((T, K))
        s = np.empty(T)
        a = self.startprob_ * b[0]
        s[0] = a.sum()
        alpha[0] = a / s[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * b[t]
            s[t] = a.sum()
            alpha[t] = a / s[t]
        ll = float(np.sum(np.log(s)) + np.sum(c))
        return alpha, s, b, ll

    def _backward_scaled(self, b: np.ndarray, s: np.ndarray) -> np.ndarray:
        T, K = b.shape
        A = self.transmat_
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (b[t + 1] * beta[t + 1])) / s[t + 1]
        return beta

    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (normalised filtered probabilities alpha_t, log-likelihood)."""
        alpha, _, _, ll = self._forward_scaled(log_b)
        return alpha, ll

    def _e_step(self, X: np.ndarray):
        log_b = self._log_emission(X)
        alpha, s, b, ll = self._forward_scaled(log_b)
        beta = self._backward_scaled(b, s)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        # sum_t xi_t(i,j) = sum_t alpha_t(i) A_ij b_j(x_{t+1}) beta_{t+1}(j) / s_{t+1}
        w = (b[1:] * beta[1:]) / s[1:, None]  # (T-1, K)
        xi_sum = self.transmat_ * (alpha[:-1].T @ w)
        return gamma, xi_sum, ll

    # ------------------------------------------------------------------ #
    # Baum-Welch
    # ------------------------------------------------------------------ #
    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi_sum: np.ndarray) -> None:
        K, d = self.n_states, X.shape[1]
        self.startprob_ = np.clip(gamma[0], 1e-12, None)
        self.startprob_ /= self.startprob_.sum()

        counts = xi_sum + self.transition_prior
        self.transmat_ = counts / counts.sum(axis=1, keepdims=True)

        Nk = gamma.sum(axis=0) + 1e-12
        self.means_ = (gamma.T @ X) / Nk[:, None]
        covars = np.empty((K, d, d))
        for j in range(K):
            diff = X - self.means_[j]
            C = (gamma[:, j, None] * diff).T @ diff / Nk[j]
            if self.covariance_type == "diag":
                C = np.diag(np.diag(C))
            covars[j] = C + self.min_covar * np.eye(d)
        self.covars_ = covars

    def _init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        """K-means++-style initialisation of the emissions, sticky transitions."""
        T, d = X.shape
        K = self.n_states
        # k-means++ seeding followed by a few Lloyd iterations
        centers = [X[rng.integers(T)]]
        for _ in range(1, K):
            d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(T, 1 / T)
            centers.append(X[rng.choice(T, p=probs)])
        centers = np.array(centers)
        for _ in range(10):
            labels = np.argmin(((X[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
            for j in range(K):
                if np.any(labels == j):
                    centers[j] = X[labels == j].mean(axis=0)
        global_cov = np.atleast_2d(np.cov(X.T)) + self.min_covar * np.eye(d)
        covars = np.empty((K, d, d))
        for j in range(K):
            pts = X[labels == j]
            C = np.atleast_2d(np.cov(pts.T)) if len(pts) > d + 1 else global_cov.copy()
            if self.covariance_type == "diag":
                C = np.diag(np.diag(C))
            covars[j] = C + self.min_covar * np.eye(d)
        self.means_, self.covars_ = centers, covars
        self.startprob_ = np.full(K, 1.0 / K)
        # Financial regimes are persistent: start with a diagonal-heavy A.
        A = np.full((K, K), 0.1 / max(K - 1, 1)) if K > 1 else np.ones((1, 1))
        if K > 1:
            np.fill_diagonal(A, 0.9)
        self.transmat_ = A

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Estimate (pi, A, mu, Sigma) by Baum-Welch with multiple restarts."""
        X = self._as_2d(X)
        T, d = X.shape
        if T < 2 * self.n_states:
            raise ValueError("Not enough observations for this number of states.")
        self.n_features_ = d
        rng = np.random.default_rng(self.random_state)

        best_ll, best = -np.inf, None
        for _ in range(self.n_init):
            self._init_params(X, rng)
            hist = FitHistory()
            prev = -np.inf
            for it in range(self.n_iter):
                gamma, xi_sum, ll = self._e_step(X)
                hist.log_likelihoods.append(ll)
                if ll - prev < self.tol and it > 0:
                    hist.converged = True
                    break
                prev = ll
                self._m_step(X, gamma, xi_sum)
            hist.n_iter = len(hist.log_likelihoods)
            final_ll = hist.log_likelihoods[-1]
            if final_ll > best_ll:
                best_ll = final_ll
                best = (
                    self.startprob_.copy(),
                    self.transmat_.copy(),
                    self.means_.copy(),
                    self.covars_.copy(),
                    hist,
                )
        self.startprob_, self.transmat_, self.means_, self.covars_, self.history_ = best
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def score(self, X: np.ndarray) -> float:
        """Log-likelihood log P(x_{1:T} | theta)."""
        self._check_fitted()
        X = self._as_2d(X)
        _, ll = self._forward(self._log_emission(X))
        return ll

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed posteriors gamma_t(i) = P(s_t = i | x_{1:T}).

        WARNING: uses future observations -> only for ex-post analysis, never
        for trading signals. Use `filter_proba` for causal inference.
        """
        self._check_fitted()
        gamma, _, _ = self._e_step(self._as_2d(X))
        return gamma

    def filter_proba(self, X: np.ndarray) -> np.ndarray:
        """Filtered probabilities P(s_t = i | x_{1:t})  (causal, no look-ahead)."""
        self._check_fitted()
        alpha, _ = self._forward(self._log_emission(self._as_2d(X)))
        return alpha

    def predict_next_proba(self, X: np.ndarray) -> np.ndarray:
        """One-step-ahead regime forecast P(s_{t+1} = j | x_{1:t}) = sum_i alpha_t(i) A_ij."""
        return self.filter_proba(X) @ self.transmat_

    def decode(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Viterbi: most likely state path and its log-probability."""
        self._check_fitted()
        X = self._as_2d(X)
        log_b = self._log_emission(X)
        T, K = log_b.shape
        log_A = np.log(self.transmat_)
        delta = np.empty((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = np.log(self.startprob_) + log_b[0]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + log_A  # (i, j)
            psi[t] = np.argmax(cand, axis=0)
            delta[t] = cand[psi[t], np.arange(K)] + log_b[t]
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, float(delta[-1, path[-1]])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decode(X)[0]

    # ------------------------------------------------------------------ #
    # Simulation & model selection
    # ------------------------------------------------------------------ #
    def sample(self, n: int, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Draw (X, states) from the generative model."""
        self._check_fitted()
        rng = np.random.default_rng(random_state)
        K, d = self.means_.shape
        states = np.empty(n, dtype=int)
        X = np.empty((n, d))
        states[0] = rng.choice(K, p=self.startprob_)
        for t in range(n):
            if t > 0:
                states[t] = rng.choice(K, p=self.transmat_[states[t - 1]])
            X[t] = rng.multivariate_normal(self.means_[states[t]], self.covars_[states[t]])
        return X, states

    def n_parameters(self) -> int:
        """Number of free parameters p (used by AIC/BIC)."""
        K, d = self.n_states, self.n_features_
        cov = K * d * (d + 1) // 2 if self.covariance_type == "full" else K * d
        return (K - 1) + K * (K - 1) + K * d + cov

    def aic(self, X: np.ndarray) -> float:
        return -2.0 * self.score(X) + 2.0 * self.n_parameters()

    def bic(self, X: np.ndarray) -> float:
        T = self._as_2d(X).shape[0]
        return -2.0 * self.score(X) + self.n_parameters() * np.log(T)

    def expected_durations(self) -> np.ndarray:
        """Mean sojourn time in each state: E[D_i] = 1 / (1 - A_ii) (geometric law)."""
        self._check_fitted()
        return 1.0 / np.clip(1.0 - np.diag(self.transmat_), 1e-12, None)

    def stationary_distribution(self) -> np.ndarray:
        """Left eigenvector of A for eigenvalue 1 (long-run regime frequencies)."""
        self._check_fitted()
        w, v = np.linalg.eig(self.transmat_.T)
        vec = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        return vec / vec.sum()
