"""Selective hybrid posterior/HMM decoder used by STRATUS-H.

Missing states are determined from complete run duration. Learned temporal
coupling is applied only inside contiguous finite blocks, where a discriminative
observation model distinguishes Stable, Movement, UnstableImpulse, and
UnstableBurst. Setting transition_strength=0 yields the matched Hybrid-P model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SHORT_GAP_SECONDS = 0.50
EXPANDED_STATES = (
    "Stable", "Movement", "LossShort", "LossLong", "UnstableImpulse", "UnstableBurst"
)
FINITE_GLOBAL_IDS = np.array([0, 1, 4, 5], dtype=int)
GLOBAL_TO_FINITE = {0: 0, 1: 1, 4: 2, 5: 3}


def finite_blocks(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    i = 0
    while i < len(mask):
        while i < len(mask) and not mask[i]:
            i += 1
        if i >= len(mask):
            return
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        yield i, j
        i = j


def viterbi(scores: np.ndarray, transition: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    n, k = scores.shape
    if n == 0:
        return np.empty(0, dtype=int)
    dp = np.full((n, k), -np.inf)
    back = np.zeros((n, k), dtype=int)
    dp[0] = scores[0] + (0.0 if start is None else start)
    for t in range(1, n):
        candidates = dp[t - 1][:, None] + transition
        back[t] = np.argmax(candidates, axis=0)
        dp[t] = scores[t] + np.max(candidates, axis=0)
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def _rank_feature(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(float)
    finite = np.isfinite(values)
    result = np.full(len(values), 0.5, dtype=float)
    if finite.any():
        result[finite] = pd.Series(values[finite]).rank(method="average", pct=True).to_numpy()
    return result


def finite_features(diagnostics: pd.DataFrame, fs_hz: float) -> np.ndarray:
    required = {"c_t", "alpha_t", "u_t", "p_t"}
    missing = required.difference(diagnostics.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")
    raw = [diagnostics[c].to_numpy(float) for c in ("c_t", "alpha_t", "u_t")]
    logged = [np.log1p(np.maximum(a, 0)) for a in raw]
    for values in logged:
        values[~np.isfinite(values)] = 0.0
    c_rank, a_rank, u_rank = (_rank_feature(diagnostics[c]) for c in ("c_t", "alpha_t", "u_t"))
    short_window = max(3, int(round(0.05 * fs_hz)))
    long_window = max(5, int(round(0.15 * fs_hz)))
    a_short = pd.Series(a_rank).rolling(short_window, center=True, min_periods=1).mean().to_numpy()
    u_short = pd.Series(u_rank).rolling(short_window, center=True, min_periods=1).mean().to_numpy()
    a_long = pd.Series(a_rank).rolling(long_window, center=True, min_periods=1).max().to_numpy()
    u_long = pd.Series(u_rank).rolling(long_window, center=True, min_periods=1).max().to_numpy()
    return np.column_stack([
        logged[0], logged[1], logged[2], c_rank, a_rank, u_rank,
        a_short, u_short, a_long, u_long, a_rank-a_short, u_rank-u_short,
        diagnostics["p_t"].to_numpy(float),
    ])


def expanded_labels(diagnostics: pd.DataFrame) -> np.ndarray:
    regime = diagnostics["true_regime"].astype(str).to_numpy()
    subtype = diagnostics["unstable_subtype"].astype(str).to_numpy()
    labels = np.zeros(len(diagnostics), dtype=int)
    labels[regime == "Movement"] = 1
    labels[regime == "LossShort"] = 2
    labels[regime == "LossLong"] = 3
    labels[(regime == "Unstable") & (subtype == "impulse")] = 4
    labels[(regime == "Unstable") & (subtype != "impulse")] = 5
    return labels


@dataclass
class TrainingSequence:
    diagnostics: pd.DataFrame
    fs_hz: float


@dataclass
class SelectiveHybridHMM:
    classifier: object
    transition_log: np.ndarray
    start_log: np.ndarray
    c_value: float

    def emission_log_probabilities(self, diagnostics: pd.DataFrame, fs_hz: float) -> np.ndarray:
        probabilities = self.classifier.predict_proba(finite_features(diagnostics, fs_hz))
        emissions = np.full((len(diagnostics), 4), math.log(1e-12), dtype=float)
        for column, global_class in enumerate(self.classifier.classes_):
            global_class = int(global_class)
            if global_class in GLOBAL_TO_FINITE:
                emissions[:, GLOBAL_TO_FINITE[global_class]] = np.log(
                    np.clip(probabilities[:, column], 1e-12, 1.0)
                )
        return emissions

    def predict(self, diagnostics: pd.DataFrame, fs_hz: float, transition_strength: float = 1.0) -> np.ndarray:
        missing = diagnostics["m_t"].to_numpy(bool)
        short_limit = max(1, int(round(SHORT_GAP_SECONDS * fs_hz)))
        result = np.zeros(len(diagnostics), dtype=int)
        result[missing & (diagnostics["r_t"].to_numpy(int) <= short_limit)] = 2
        result[missing & (diagnostics["r_t"].to_numpy(int) > short_limit)] = 3
        finite = ~missing
        emissions = self.emission_log_probabilities(diagnostics, fs_hz)
        for start, end in finite_blocks(finite):
            block = emissions[start:end]
            local = np.argmax(block, axis=1) if transition_strength <= 0 else viterbi(
                block,
                transition_strength * self.transition_log,
                transition_strength * self.start_log,
            )
            result[start:end] = FINITE_GLOBAL_IDS[local]
        return result

    def state_names(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray([EXPANDED_STATES[int(i)] for i in indices], dtype=object)


def fit_selective_hmm(
    sequences: list[TrainingSequence],
    c_value: float = 1.0,
    transition_pseudocount: float = 0.5,
    max_samples_per_class: int = 75000,
    random_state: int = 20260728,
) -> SelectiveHybridHMM:
    if not sequences:
        raise ValueError("At least one training sequence is required.")
    features, labels = [], []
    transitions = np.full((4, 4), transition_pseudocount, dtype=float)
    starts = np.full(4, transition_pseudocount, dtype=float)
    for sequence in sequences:
        diagnostics = sequence.diagnostics
        y_global = expanded_labels(diagnostics)
        finite = ~diagnostics["m_t"].to_numpy(bool)
        features.append(finite_features(diagnostics, sequence.fs_hz)[finite])
        labels.append(y_global[finite])
        for start, end in finite_blocks(finite):
            local = np.asarray([GLOBAL_TO_FINITE[int(v)] for v in y_global[start:end]], dtype=int)
            if len(local):
                starts[local[0]] += 1
                for left, right in zip(local[:-1], local[1:]):
                    transitions[left, right] += 1
    x = np.vstack(features)
    y = np.concatenate(labels)
    required = {0, 1, 4, 5}
    if required.difference(set(map(int, np.unique(y)))):
        raise ValueError("Training sequences must contain all four finite operational classes.")
    rng = np.random.default_rng(random_state)
    selected = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if len(idx) > max_samples_per_class:
            idx = rng.choice(idx, max_samples_per_class, replace=False)
        selected.append(idx)
    idx = np.concatenate(selected)
    rng.shuffle(idx)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1500, class_weight="balanced", C=float(c_value),
            solver="lbfgs", random_state=random_state,
        ),
    )
    classifier.fit(x[idx], y[idx])
    transition = transitions / transitions.sum(axis=1, keepdims=True)
    start = starts / starts.sum()
    return SelectiveHybridHMM(
        classifier=classifier,
        transition_log=np.log(np.clip(transition, 1e-12, 1.0)),
        start_log=np.log(np.clip(start, 1e-12, 1.0)),
        c_value=float(c_value),
    )
