from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.cluster import KMeans


SEMANTIC_STATES = ["Stable", "Movement", "LossShort", "LossLong", "Unstable"]
FEATURE_COLUMNS = ["m_t", "r_t", "c_t", "alpha_t", "u_t", "p_t"]


@dataclass
class DiagnosticFeatureTransform:
    """Robust transform for heterogeneous STRATUS diagnostic variables."""

    median_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        # Heavy-tailed run lengths, velocity, acceleration, and jitter.
        x[:, 1:5] = np.log1p(np.maximum(x[:, 1:5], 0.0))
        x[~np.isfinite(x)] = np.nan
        return x

    def fit(self, frames: Iterable[pd.DataFrame]) -> "DiagnosticFeatureTransform":
        x = np.vstack([self._raw(frame) for frame in frames])
        self.median_ = np.nanmedian(x, axis=0)
        q25 = np.nanpercentile(x, 25, axis=0)
        q75 = np.nanpercentile(x, 75, axis=0)
        scale = q75 - q25
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        self.scale_ = scale
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("Feature transform must be fitted first.")
        x = self._raw(df)
        missing = ~np.isfinite(x)
        if missing.any():
            x[missing] = np.take(self.median_, np.where(missing)[1])
        return (x - self.median_) / self.scale_


class DiagonalGaussianHMM:
    """Small diagonal-Gaussian HMM trained by Baum--Welch.

    This implementation avoids an external hmmlearn dependency and keeps the
    learning procedure visible and reproducible inside the repository.
    """

    def __init__(
        self,
        n_states: int = 5,
        n_iter: int = 20,
        tol: float = 1e-3,
        covariance_floor: float = 1e-3,
        random_state: int = 42,
        self_transition_init: float = 0.92,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_floor = covariance_floor
        self.random_state = random_state
        self.self_transition_init = self_transition_init
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.log_likelihood_history_: list[float] = []

    def _initialize(self, sequences: list[np.ndarray]) -> None:
        x = np.vstack(sequences)
        km = KMeans(
            n_clusters=self.n_states,
            n_init=30,
            random_state=self.random_state,
        ).fit(x)
        self.means_ = km.cluster_centers_.copy()

        global_var = np.var(x, axis=0) + self.covariance_floor
        self.covars_ = np.tile(global_var, (self.n_states, 1))
        for k in range(self.n_states):
            cluster = x[km.labels_ == k]
            if len(cluster) > 2:
                self.covars_[k] = np.var(cluster, axis=0) + self.covariance_floor

        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        off = (1.0 - self.self_transition_init) / (self.n_states - 1)
        self.transmat_ = np.full((self.n_states, self.n_states), off)
        np.fill_diagonal(self.transmat_, self.self_transition_init)

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - self.means_[None, :, :]
        log_det = np.sum(np.log(2.0 * np.pi * self.covars_), axis=1)
        quad = np.sum((diff ** 2) / self.covars_[None, :, :], axis=2)
        return -0.5 * (quad + log_det[None, :])

    def _forward_backward(self, x: np.ndarray):
        log_b = self._log_emission(x)
        log_start = np.log(np.clip(self.startprob_, 1e-300, None))
        log_trans = np.log(np.clip(self.transmat_, 1e-300, None))
        t_len = len(x)

        alpha = np.empty((t_len, self.n_states))
        alpha[0] = log_start + log_b[0]
        for t in range(1, t_len):
            alpha[t] = log_b[t] + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)

        beta = np.zeros((t_len, self.n_states))
        for t in range(t_len - 2, -1, -1):
            beta[t] = logsumexp(
                log_trans + log_b[t + 1][None, :] + beta[t + 1][None, :],
                axis=1,
            )

        ll = float(logsumexp(alpha[-1]))
        log_gamma = alpha + beta - ll
        gamma = np.exp(log_gamma)

        xi_sum = np.zeros((self.n_states, self.n_states))
        for t in range(t_len - 1):
            log_xi = (
                alpha[t][:, None]
                + log_trans
                + log_b[t + 1][None, :]
                + beta[t + 1][None, :]
                - ll
            )
            xi_sum += np.exp(log_xi)
        return ll, gamma, xi_sum

    def fit(self, sequences: list[np.ndarray]) -> "DiagonalGaussianHMM":
        sequences = [np.asarray(seq, dtype=float) for seq in sequences if len(seq) > 1]
        if not sequences:
            raise ValueError("No non-empty sequences supplied.")
        self._initialize(sequences)
        previous = -np.inf

        for _ in range(self.n_iter):
            start_acc = np.zeros(self.n_states)
            trans_acc = np.zeros((self.n_states, self.n_states))
            gamma_acc = np.zeros(self.n_states)
            mean_acc = np.zeros_like(self.means_)
            second_acc = np.zeros_like(self.means_)
            total_ll = 0.0

            for x in sequences:
                ll, gamma, xi_sum = self._forward_backward(x)
                total_ll += ll
                start_acc += gamma[0]
                trans_acc += xi_sum
                weights = gamma.sum(axis=0)
                gamma_acc += weights
                mean_acc += gamma.T @ x
                second_acc += gamma.T @ (x ** 2)

            self.startprob_ = start_acc / np.clip(start_acc.sum(), 1e-12, None)
            row_sums = trans_acc.sum(axis=1, keepdims=True)
            self.transmat_ = trans_acc / np.clip(row_sums, 1e-12, None)

            self.means_ = mean_acc / np.clip(gamma_acc[:, None], 1e-12, None)
            second = second_acc / np.clip(gamma_acc[:, None], 1e-12, None)
            self.covars_ = np.maximum(
                second - self.means_ ** 2,
                self.covariance_floor,
            )

            self.log_likelihood_history_.append(total_ll)
            if np.isfinite(previous) and abs(total_ll - previous) < self.tol:
                break
            previous = total_ll
        return self

    def predict_components(self, x: np.ndarray) -> np.ndarray:
        """Viterbi decoding in component space."""
        x = np.asarray(x, dtype=float)
        log_b = self._log_emission(x)
        log_start = np.log(np.clip(self.startprob_, 1e-300, None))
        log_trans = np.log(np.clip(self.transmat_, 1e-300, None))

        n = len(x)
        delta = np.empty((n, self.n_states))
        back = np.zeros((n, self.n_states), dtype=int)
        delta[0] = log_start + log_b[0]

        for t in range(1, n):
            candidates = delta[t - 1][:, None] + log_trans
            back[t] = np.argmax(candidates, axis=0)
            delta[t] = log_b[t] + np.max(candidates, axis=0)

        path = np.zeros(n, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(n - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        return path


@dataclass
class BaumWelchSTRATUS:
    """Baum--Welch variant with training-only semantic component mapping."""

    transform: DiagnosticFeatureTransform
    hmm: DiagonalGaussianHMM
    component_to_state: dict[int, str]

    @classmethod
    def fit(
        cls,
        diagnostic_frames: list[pd.DataFrame],
        true_state_sequences: list[pd.Series],
        n_iter: int = 20,
        random_state: int = 42,
    ) -> "BaumWelchSTRATUS":
        transform = DiagnosticFeatureTransform().fit(diagnostic_frames)
        sequences = [transform.transform(frame) for frame in diagnostic_frames]
        hmm = DiagonalGaussianHMM(
            n_states=len(SEMANTIC_STATES),
            n_iter=n_iter,
            random_state=random_state,
        ).fit(sequences)

        # Hidden-state labels are permutation invariant. We use training labels
        # only to assign each learned component one operational state name.
        counts = np.zeros((len(SEMANTIC_STATES), len(SEMANTIC_STATES)), dtype=int)
        state_index = {state: i for i, state in enumerate(SEMANTIC_STATES)}
        for x, y in zip(sequences, true_state_sequences):
            comp = hmm.predict_components(x)
            y_arr = y.astype(str).to_numpy()
            for k in range(len(SEMANTIC_STATES)):
                mask = comp == k
                if not mask.any():
                    continue
                for state, j in state_index.items():
                    counts[k, j] += int(np.sum(y_arr[mask] == state))

        # One-to-one assignment maximizing training overlap.
        rows, cols = linear_sum_assignment(-counts)
        mapping = {int(r): SEMANTIC_STATES[int(c)] for r, c in zip(rows, cols)}
        for k in range(len(SEMANTIC_STATES)):
            mapping.setdefault(k, SEMANTIC_STATES[k])

        return cls(transform=transform, hmm=hmm, component_to_state=mapping)

    def predict(self, diagnostic_frame: pd.DataFrame) -> pd.Series:
        x = self.transform.transform(diagnostic_frame)
        components = self.hmm.predict_components(x)
        states = [self.component_to_state[int(k)] for k in components]
        return pd.Series(states, index=diagnostic_frame.index, dtype="object")
