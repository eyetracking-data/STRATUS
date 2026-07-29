from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Reproducible configuration
# -----------------------------------------------------------------------------
RANDOM_SEED = 20260728
SHORT_GAP_SECONDS = 0.50
SEVERITIES = ("low", "medium", "high")
CORRUPTIONS = ("short_gap", "long_gap", "jitter", "spike", "mixed")
EVAL_SEEDS = tuple(range(10))
TRAIN_SEEDS = (0, 1, 2, 3, 4)
C_GRID = (0.1, 1.0, 10.0)
LAMBDA_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)

BASE_STATES = ("Stable", "Movement", "LossShort", "LossLong", "Unstable")
EXPANDED_STATES = (
    "Stable",
    "Movement",
    "LossShort",
    "LossLong",
    "UnstableImpulse",
    "UnstableBurst",
)
FINITE_GLOBAL_IDS = np.array([0, 1, 4, 5], dtype=int)
GLOBAL_TO_FINITE = {0: 0, 1: 1, 4: 2, 5: 3}

DEFAULT_METADATA = {
    "ETDD70": {"sampling_rate_hz": 250.0, "width": 1600.0, "height": 1000.0},
    "Autism": {"sampling_rate_hz": 59.88382537870461, "width": 1400.0, "height": 1200.0},
}


# -----------------------------------------------------------------------------
# Validation and shared helpers
# -----------------------------------------------------------------------------
def normalize_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def validate_reference(df: pd.DataFrame) -> None:
    required = {
        "dataset",
        "source_file",
        "participant_id",
        "segment_id",
        "time_s",
        "x",
        "y",
        "case_segment",
        "case_index",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Reference CSV is missing columns: {missing}")
    if df.empty:
        raise ValueError("Reference CSV is empty.")
    if not set(df["dataset"].astype(str).unique()).issubset(DEFAULT_METADATA):
        raise ValueError(f"Unexpected datasets: {sorted(df['dataset'].astype(str).unique())}")
    bad = ~np.isfinite(pd.to_numeric(df["x"], errors="coerce")) | ~np.isfinite(
        pd.to_numeric(df["y"], errors="coerce")
    )
    if bad.any():
        raise ValueError("Clean reference contains non-finite x/y samples.")


def estimate_sampling_rate_hz(df: pd.DataFrame, fallback: float) -> float:
    t = np.sort(pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float))
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt)) if len(dt) else float(fallback)


def _full_missing_run_lengths(missing: np.ndarray) -> np.ndarray:
    missing = np.asarray(missing, dtype=bool)
    lengths = np.zeros(len(missing), dtype=int)
    i = 0
    while i < len(missing):
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < len(missing) and missing[j]:
            j += 1
        lengths[i:j] = j - i
        i = j
    return lengths


def _finite_blocks(mask: np.ndarray) -> Iterable[tuple[int, int]]:
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


def safe_quantile(values: Sequence[float], q: float, default: float = np.inf) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else float(default)


# -----------------------------------------------------------------------------
# Exact v7-style controlled corruption, plus an auxiliary unstable subtype label
# -----------------------------------------------------------------------------
def _mark_natural_movement(out: pd.DataFrame, quantile: float = 0.90) -> pd.Series:
    dx = out["x_clean"].diff()
    dy = out["y_clean"].diff()
    displacement = np.sqrt(dx**2 + dy**2)
    threshold = np.nanquantile(displacement.to_numpy(dtype=float), quantile)
    return displacement >= threshold


def inject_corruption(df: pd.DataFrame, corruption: str, severity: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)
    out["x_clean"] = out["x"]
    out["y_clean"] = out["y"]
    out["true_regime"] = "Stable"
    out["unstable_subtype"] = ""

    movement_mask = _mark_natural_movement(out, quantile=0.90)
    out.loc[movement_mask, "true_regime"] = "Movement"

    sev = {"low": 1.0, "medium": 1.7, "high": 2.5}[severity]

    def interval(start_frac: float, length_frac: float) -> np.ndarray:
        start = int(n * start_frac)
        length = max(2, int(n * length_frac))
        return np.arange(start, min(n, start + length))

    if corruption in {"short_gap", "mixed"}:
        idx = interval(0.25, 0.025 * sev)
        out.loc[idx, ["x", "y"]] = np.nan
        out.loc[idx, "true_regime"] = "LossShort"

    if corruption in {"long_gap", "mixed"}:
        idx = interval(0.55, 0.08 * sev)
        out.loc[idx, ["x", "y"]] = np.nan
        out.loc[idx, "true_regime"] = "LossLong"

    finite_available = out["x"].notna() & out["y"].notna()
    if corruption in {"jitter", "mixed"}:
        idx = interval(0.78, 0.08)
        idx = idx[finite_available.iloc[idx].to_numpy()]
        noise = rng.normal(0, 8.0 * sev, size=(len(idx), 2))
        out.loc[idx, ["x", "y"]] = out.loc[idx, ["x", "y"]].to_numpy() + noise
        out.loc[idx, "true_regime"] = "Unstable"
        out.loc[idx, "unstable_subtype"] = "burst"

    finite_available = out["x"].notna() & out["y"].notna()
    if corruption in {"spike", "mixed"}:
        k = max(2, int(0.015 * n * sev))
        candidates = np.where(finite_available.to_numpy())[0]
        if len(candidates):
            k = min(k, len(candidates))
            idx = rng.choice(candidates, size=k, replace=False)
            out.loc[idx, "x"] = out["x_clean"].median() + rng.normal(0, 350 * sev, size=k)
            out.loc[idx, "y"] = out["y_clean"].median() + rng.normal(0, 250 * sev, size=k)
            out.loc[idx, "true_regime"] = "Unstable"
            out.loc[idx, "unstable_subtype"] = "impulse"

    out["x_degraded"] = out["x"]
    out["y_degraded"] = out["y"]
    out["corruption"] = corruption
    out["severity"] = severity
    out["seed"] = int(seed)
    return out


# -----------------------------------------------------------------------------
# Diagnostics and original deterministic baselines
# -----------------------------------------------------------------------------
def compute_diagnostics(df: pd.DataFrame, width: float, height: float) -> pd.DataFrame:
    out = df.copy()
    x = out["x"].to_numpy(dtype=float)
    y = out["y"].to_numpy(dtype=float)
    t = out["time_s"].to_numpy(dtype=float)

    finite_xy = np.isfinite(x) & np.isfinite(y)
    plausibility = np.zeros(len(out), dtype=bool)
    plausibility[finite_xy] = (
        (x[finite_xy] < 0)
        | (x[finite_xy] > width)
        | (y[finite_xy] < 0)
        | (y[finite_xy] > height)
    )
    missing = ~finite_xy
    invalid_dynamics = missing | plausibility
    run_length = _full_missing_run_lengths(missing)

    dt = np.diff(t, prepend=np.nan)
    dt[~np.isfinite(dt) | (dt <= 0)] = np.nan
    distance = np.sqrt(np.diff(x, prepend=np.nan) ** 2 + np.diff(y, prepend=np.nan) ** 2)
    velocity = distance / dt
    velocity[invalid_dynamics] = np.nan
    velocity[np.r_[True, invalid_dynamics[:-1]]] = np.nan
    acceleration = np.abs(np.diff(velocity, prepend=np.nan)) / dt
    acceleration[invalid_dynamics] = np.nan

    coordinates = pd.DataFrame({"x": x, "y": y})
    med_x = coordinates["x"].rolling(7, center=True, min_periods=3).median()
    med_y = coordinates["y"].rolling(7, center=True, min_periods=3).median()
    jitter = np.sqrt((coordinates["x"] - med_x) ** 2 + (coordinates["y"] - med_y) ** 2)
    jitter[invalid_dynamics] = np.nan

    out["m_t"] = missing.astype(int)
    out["r_t"] = run_length
    out["c_t"] = velocity
    out["alpha_t"] = acceleration
    out["u_t"] = jitter
    out["p_t"] = plausibility.astype(int)
    return out


def original_pointwise_scores(diag: pd.DataFrame, fs_hz: float) -> np.ndarray:
    n = len(diag)
    scores = np.zeros((n, 5), dtype=float)
    missing = diag["m_t"].to_numpy(bool)
    finite = ~missing
    c = diag["c_t"].to_numpy(float)
    u = diag["u_t"].to_numpy(float)
    alpha = diag["alpha_t"].to_numpy(float)

    velocity_threshold = safe_quantile(c[finite], 0.90)
    jitter_threshold = safe_quantile(u[finite], 0.995)
    acceleration_threshold = safe_quantile(alpha[finite], 0.995)
    short_samples = max(1, int(round(SHORT_GAP_SECONDS * fs_hz)))
    short = missing & (diag["r_t"].to_numpy(int) <= short_samples)
    long = missing & ~short

    scores[finite, 0] = 2.0
    scores[finite & (c >= velocity_threshold), 1] = 5.0
    scores[short, 2] = 8.0
    scores[long, 3] = 10.0
    unstable = finite & (
        diag["p_t"].to_numpy(bool)
        | (u >= jitter_threshold)
        | (alpha >= acceleration_threshold)
    )
    scores[unstable, 4] = 7.0
    scores[unstable, 0] = 0.0
    scores[np.ix_(np.where(missing)[0], [0, 1, 4])] = -8.0
    scores[short, 3] = -4.0
    scores[long, 2] = -8.0
    scores[finite, 2] = -8.0
    scores[finite, 3] = -8.0
    return scores


def viterbi(scores: np.ndarray, transition_scores: np.ndarray, start_scores: np.ndarray | None = None) -> np.ndarray:
    n, k = scores.shape
    if n == 0:
        return np.array([], dtype=int)
    if start_scores is None:
        start_scores = np.zeros(k, dtype=float)
    dp = np.empty((n, k), dtype=float)
    back = np.zeros((n, k), dtype=int)
    dp[0] = scores[0] + start_scores
    for t in range(1, n):
        candidates = dp[t - 1][:, None] + transition_scores
        back[t] = np.argmax(candidates, axis=0)
        dp[t] = scores[t] + np.max(candidates, axis=0)
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def predict_pointwise_d(diag: pd.DataFrame, fs_hz: float) -> np.ndarray:
    return np.argmax(original_pointwise_scores(diag, fs_hz), axis=1)


def predict_stratus_d(diag: pd.DataFrame, fs_hz: float) -> np.ndarray:
    transition = np.full((5, 5), -0.75, dtype=float)
    np.fill_diagonal(transition, 1.25)
    return viterbi(original_pointwise_scores(diag, fs_hz), transition)


# -----------------------------------------------------------------------------
# Hybrid observation model and selective constrained HMM
# -----------------------------------------------------------------------------
def _rank_feature(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(float)
    finite = np.isfinite(values)
    result = np.full(len(values), 0.5, dtype=float)
    if finite.any():
        result[finite] = pd.Series(values[finite]).rank(method="average", pct=True).to_numpy()
    return result


def finite_features(diag: pd.DataFrame, fs_hz: float) -> np.ndarray:
    c_raw = diag["c_t"].to_numpy(float)
    a_raw = diag["alpha_t"].to_numpy(float)
    u_raw = diag["u_t"].to_numpy(float)
    c_log = np.log1p(np.maximum(c_raw, 0))
    a_log = np.log1p(np.maximum(a_raw, 0))
    u_log = np.log1p(np.maximum(u_raw, 0))
    for arr in (c_log, a_log, u_log):
        arr[~np.isfinite(arr)] = 0.0

    c_rank = _rank_feature(diag["c_t"])
    a_rank = _rank_feature(diag["alpha_t"])
    u_rank = _rank_feature(diag["u_t"])
    p = diag["p_t"].to_numpy(float)

    short_window = max(3, int(round(0.05 * fs_hz)))
    long_window = max(5, int(round(0.15 * fs_hz)))
    a_short_mean = pd.Series(a_rank).rolling(short_window, center=True, min_periods=1).mean().to_numpy()
    u_short_mean = pd.Series(u_rank).rolling(short_window, center=True, min_periods=1).mean().to_numpy()
    a_long_max = pd.Series(a_rank).rolling(long_window, center=True, min_periods=1).max().to_numpy()
    u_long_max = pd.Series(u_rank).rolling(long_window, center=True, min_periods=1).max().to_numpy()
    isolated_a = a_rank - a_short_mean
    isolated_u = u_rank - u_short_mean

    return np.column_stack(
        [
            c_log,
            a_log,
            u_log,
            c_rank,
            a_rank,
            u_rank,
            a_short_mean,
            u_short_mean,
            a_long_max,
            u_long_max,
            isolated_a,
            isolated_u,
            p,
        ]
    )


def expanded_labels(diag: pd.DataFrame) -> np.ndarray:
    regime = diag["true_regime"].astype(str).to_numpy()
    subtype = diag["unstable_subtype"].astype(str).to_numpy()
    labels = np.zeros(len(diag), dtype=int)
    labels[regime == "Movement"] = 1
    labels[regime == "LossShort"] = 2
    labels[regime == "LossLong"] = 3
    labels[(regime == "Unstable") & (subtype == "impulse")] = 4
    labels[(regime == "Unstable") & (subtype != "impulse")] = 5
    return labels


@dataclass
class PreparedCase:
    dataset: str
    participant_id: str
    case_segment: str
    corruption: str
    severity: str
    seed: int
    fs_hz: float
    width: float
    height: float
    diag: pd.DataFrame


@dataclass
class HybridModel:
    classifier: object
    finite_transition_log: np.ndarray
    finite_start_log: np.ndarray
    c_value: float

    def emission_log_probabilities(self, diag: pd.DataFrame, fs_hz: float) -> np.ndarray:
        probabilities = self.classifier.predict_proba(finite_features(diag, fs_hz))
        emissions = np.full((len(diag), 4), math.log(1e-12), dtype=float)
        for column, global_class in enumerate(self.classifier.classes_):
            global_class = int(global_class)
            if global_class not in GLOBAL_TO_FINITE:
                continue
            emissions[:, GLOBAL_TO_FINITE[global_class]] = np.log(
                np.clip(probabilities[:, column], 1e-12, 1.0)
            )
        return emissions

    def predict(self, diag: pd.DataFrame, fs_hz: float, transition_strength: float) -> np.ndarray:
        missing = diag["m_t"].to_numpy(bool)
        short_samples = max(1, int(round(SHORT_GAP_SECONDS * fs_hz)))
        short = missing & (diag["r_t"].to_numpy(int) <= short_samples)
        long = missing & ~short
        result = np.zeros(len(diag), dtype=int)
        result[short] = 2
        result[long] = 3

        finite = ~missing
        emissions = self.emission_log_probabilities(diag, fs_hz)
        for start, end in _finite_blocks(finite):
            block = emissions[start:end]
            if transition_strength <= 0:
                local_path = np.argmax(block, axis=1)
            else:
                local_path = viterbi(
                    block,
                    transition_strength * self.finite_transition_log,
                    transition_strength * self.finite_start_log,
                )
            result[start:end] = FINITE_GLOBAL_IDS[local_path]
        return result


def balanced_sample(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    max_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[np.ndarray] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(np.asarray(idx, dtype=int))
    all_idx = np.concatenate(selected)
    rng.shuffle(all_idx)
    return x[all_idx], y[all_idx]


def fit_hybrid(
    cases: list[PreparedCase],
    c_value: float,
    max_samples_per_class: int,
    seed: int,
) -> HybridModel:
    if not cases:
        raise ValueError("No training cases supplied.")
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    transition_counts = np.full((4, 4), 0.5, dtype=float)
    start_counts = np.full(4, 0.5, dtype=float)

    for case in cases:
        diag = case.diag
        y_global = expanded_labels(diag)
        finite = ~diag["m_t"].to_numpy(bool)
        x = finite_features(diag, case.fs_hz)
        features.append(x[finite])
        labels.append(y_global[finite])

        for start, end in _finite_blocks(finite):
            local = np.array([GLOBAL_TO_FINITE[int(v)] for v in y_global[start:end]], dtype=int)
            if len(local):
                start_counts[local[0]] += 1
                for left, right in zip(local[:-1], local[1:]):
                    transition_counts[left, right] += 1

    x_all = np.vstack(features)
    y_all = np.concatenate(labels)
    required = {0, 1, 4, 5}
    observed = set(int(v) for v in np.unique(y_all))
    if required.difference(observed):
        raise ValueError(f"Training data lacks finite classes: {sorted(required.difference(observed))}")

    rng = np.random.default_rng(seed)
    x_fit, y_fit = balanced_sample(x_all, y_all, rng, max_samples_per_class)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            C=float(c_value),
            solver="lbfgs",
            random_state=seed,
        ),
    )
    classifier.fit(x_fit, y_fit)

    transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    start = start_counts / start_counts.sum()
    return HybridModel(
        classifier=classifier,
        finite_transition_log=np.log(np.clip(transition, 1e-12, 1.0)),
        finite_start_log=np.log(np.clip(start, 1e-12, 1.0)),
        c_value=float(c_value),
    )


# -----------------------------------------------------------------------------
# Shared state-to-action policy and v7 metrics
# -----------------------------------------------------------------------------
def apply_policy(diag: pd.DataFrame, prediction: np.ndarray, expanded: bool) -> pd.DataFrame:
    output = diag.copy()
    names = EXPANDED_STATES if expanded else BASE_STATES
    predicted = np.array([names[int(index)] for index in prediction], dtype=object)
    mapped = np.where(np.char.startswith(predicted.astype(str), "Unstable"), "Unstable", predicted)
    output["pred_regime"] = mapped

    x_out = output["x"].copy()
    y_out = output["y"].copy()
    short = mapped == "LossShort"
    full_x = x_out.interpolate(method="linear", limit_direction="both")
    full_y = y_out.interpolate(method="linear", limit_direction="both")
    x_out.loc[short] = full_x.loc[short]
    y_out.loc[short] = full_y.loc[short]

    long = mapped == "LossLong"
    x_out.loc[long] = np.nan
    y_out.loc[long] = np.nan

    unstable = mapped == "Unstable"
    median = output[["x", "y"]].rolling(window=7, center=True, min_periods=1).median()
    x_out.loc[unstable] = median.loc[unstable, "x"]
    y_out.loc[unstable] = median.loc[unstable, "y"]
    output["x"] = x_out
    output["y"] = y_out
    return output


def evaluate_output(df: pd.DataFrame, width: float, height: float) -> dict[str, float]:
    x = df["x_clean"].to_numpy(dtype=float)
    y = df["y_clean"].to_numpy(dtype=float)
    xo = df["x"].to_numpy(dtype=float)
    yo = df["y"].to_numpy(dtype=float)
    finite_ref = np.isfinite(x) & np.isfinite(y)
    finite_out = np.isfinite(xo) & np.isfinite(yo)
    valid = finite_ref & finite_out
    distance = np.sqrt((xo[valid] - x[valid]) ** 2 + (yo[valid] - y[valid]) ** 2)
    mae = float(np.mean(distance)) if len(distance) else np.nan
    rmse = float(np.sqrt(np.mean(distance**2))) if len(distance) else np.nan
    retention = float(np.mean(finite_out))

    regime = df["true_regime"].astype(str).to_numpy()
    long = regime == "LossLong"
    short = regime == "LossShort"
    movement = regime == "Movement"
    stable = regime == "Stable"
    unstable = regime == "Unstable"
    long_hallucination = float(np.mean(finite_out[long])) if long.any() else np.nan
    short_recovery = float(np.mean(finite_out[short])) if short.any() else np.nan

    clean_step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    out_step = np.sqrt(np.diff(xo) ** 2 + np.diff(yo) ** 2)
    movement_steps = movement[1:] & finite_ref[1:] & finite_ref[:-1] & finite_out[1:] & finite_out[:-1]
    if movement_steps.any():
        clean_dynamics = float(np.mean(clean_step[movement_steps]))
        output_dynamics = float(np.mean(out_step[movement_steps]))
        movement_preservation = (
            float(np.clip(1.0 - abs(output_dynamics / clean_dynamics - 1.0), 0.0, 1.0))
            if np.isfinite(clean_dynamics) and clean_dynamics > 0 and np.isfinite(output_dynamics)
            else np.nan
        )
    else:
        movement_preservation = np.nan

    stable_reference = stable & finite_ref
    unchanged = finite_out & np.isclose(xo, x, atol=1e-9, rtol=0) & np.isclose(yo, y, atol=1e-9, rtol=0)
    stable_preservation = float(np.mean(unchanged[stable_reference])) if stable_reference.any() else np.nan

    unstable_repair_gain = np.nan
    unstable_repair_score = np.nan
    if unstable.any():
        xd = df["x_degraded"].to_numpy(dtype=float)
        yd = df["y_degraded"].to_numpy(dtype=float)
        finite_degraded = np.isfinite(xd) & np.isfinite(yd)
        base_mask = unstable & finite_ref & finite_degraded
        if base_mask.any():
            base_error = np.sqrt((xd[base_mask] - x[base_mask]) ** 2 + (yd[base_mask] - y[base_mask]) ** 2)
            base_mae = float(np.mean(base_error))
            if np.isfinite(base_mae) and base_mae > 1e-12:
                output_error = np.sqrt((xo[base_mask] - x[base_mask]) ** 2 + (yo[base_mask] - y[base_mask]) ** 2)
                missing_output = ~finite_out[base_mask]
                output_error[missing_output] = base_error[missing_output]
                unstable_repair_gain = float(1.0 - np.mean(output_error) / base_mae)
                unstable_repair_score = float(np.clip(unstable_repair_gain, 0.0, 1.0))

    plausibility_violation = np.zeros(len(df), dtype=bool)
    plausibility_violation[finite_out] = (
        (xo[finite_out] < 0)
        | (xo[finite_out] > width)
        | (yo[finite_out] < 0)
        | (yo[finite_out] > height)
    )
    plausibility_score = 1.0 - float(np.mean(plausibility_violation[finite_out])) if finite_out.any() else np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "retention": retention,
        "long_hallucination": long_hallucination,
        "short_recovery": short_recovery,
        "movement_preservation": movement_preservation,
        "stable_preservation": stable_preservation,
        "unstable_repair_gain": unstable_repair_gain,
        "unstable_repair_score": unstable_repair_score,
        "plausibility_score": plausibility_score,
    }


Q_COMPONENTS = (
    "long_safety",
    "short_recovery",
    "movement_preservation",
    "stable_preservation",
    "unstable_repair_score",
)


def add_local_action_score(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["long_safety"] = 1.0 - result["long_hallucination"]
    result["Q_local"] = result[list(Q_COMPONENTS)].mean(axis=1, skipna=True)
    return result


def participant_aggregate(sequence_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mae",
        "rmse",
        "retention",
        "long_hallucination",
        "short_recovery",
        "movement_preservation",
        "stable_preservation",
        "unstable_repair_gain",
        "unstable_repair_score",
        "plausibility_score",
    ]
    grouped = (
        sequence_results.groupby(["dataset", "participant_id", "method"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    return add_local_action_score(grouped)


def dataset_and_macro_aggregate(participant_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "mae",
        "rmse",
        "retention",
        "long_hallucination",
        "short_recovery",
        "movement_preservation",
        "stable_preservation",
        "unstable_repair_gain",
        "unstable_repair_score",
        "plausibility_score",
        "long_safety",
        "Q_local",
    ]
    dataset = participant_results.groupby(["dataset", "method"], as_index=False)[metrics].mean(numeric_only=True)
    macro = dataset.groupby("method", as_index=False)[metrics].mean(numeric_only=True)
    return dataset, macro


# -----------------------------------------------------------------------------
# Case preparation, participant-safe tuning, and paired bootstrap
# -----------------------------------------------------------------------------
def build_clean_cases(reference: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cases: dict[str, pd.DataFrame] = {}
    for case_segment, group in reference.groupby("case_segment", sort=True):
        clean = group.sort_values("time_s").reset_index(drop=True).copy()
        clean["participant_id"] = clean["participant_id"].map(normalize_id)
        cases[str(case_segment)] = clean
    return cases


def prepare_case(clean: pd.DataFrame, corruption: str, severity: str, seed: int) -> PreparedCase:
    dataset = str(clean["dataset"].iloc[0])
    metadata = DEFAULT_METADATA[dataset]
    fs_hz = estimate_sampling_rate_hz(clean, metadata["sampling_rate_hz"])
    degraded = inject_corruption(clean, corruption, severity, seed)
    diag = compute_diagnostics(degraded, metadata["width"], metadata["height"])
    return PreparedCase(
        dataset=dataset,
        participant_id=normalize_id(clean["participant_id"].iloc[0]),
        case_segment=str(clean["case_segment"].iloc[0]),
        corruption=corruption,
        severity=severity,
        seed=int(seed),
        fs_hz=fs_hz,
        width=metadata["width"],
        height=metadata["height"],
        diag=diag,
    )


def generate_prepared_cases(
    clean_cases: dict[str, pd.DataFrame],
    participants: set[tuple[str, str]],
    seeds: Sequence[int],
    severities: Sequence[str],
    corruptions: Sequence[str],
    cap_per_dataset: int | None,
    selection_seed: int,
) -> list[PreparedCase]:
    specs: list[tuple[str, str, str, int]] = []
    for case_segment, clean in clean_cases.items():
        key = (str(clean["dataset"].iloc[0]), normalize_id(clean["participant_id"].iloc[0]))
        if key not in participants:
            continue
        for seed in seeds:
            for severity in severities:
                for corruption in corruptions:
                    specs.append((case_segment, corruption, severity, int(seed)))

    if cap_per_dataset is not None:
        rng = np.random.default_rng(selection_seed)
        selected_specs: list[tuple[str, str, str, int]] = []
        by_dataset: dict[str, list[tuple[str, str, str, int]]] = {}
        for spec in specs:
            dataset = str(clean_cases[spec[0]]["dataset"].iloc[0])
            by_dataset.setdefault(dataset, []).append(spec)
        for dataset, dataset_specs in sorted(by_dataset.items()):
            if len(dataset_specs) > cap_per_dataset:
                idx = rng.choice(len(dataset_specs), size=cap_per_dataset, replace=False)
                dataset_specs = [dataset_specs[int(i)] for i in sorted(idx)]
            selected_specs.extend(dataset_specs)
        specs = selected_specs

    return [prepare_case(clean_cases[segment], corruption, severity, seed) for segment, corruption, severity, seed in specs]


def split_training_and_development(
    outer_train: set[tuple[str, str]],
    dev_fraction: float,
    seed: int,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    inner_train: set[tuple[str, str]] = set()
    development: set[tuple[str, str]] = set()
    for dataset in sorted({dataset for dataset, _ in outer_train}):
        ids = sorted(participant for ds, participant in outer_train if ds == dataset)
        permutation = list(np.asarray(ids, dtype=object)[rng.permutation(len(ids))])
        n_dev = max(1, int(round(dev_fraction * len(ids))))
        n_dev = min(n_dev, len(ids) - 1)
        development.update((dataset, normalize_id(value)) for value in permutation[:n_dev])
        inner_train.update((dataset, normalize_id(value)) for value in permutation[n_dev:])
    return inner_train, development


def evaluate_hybrid_cases(
    cases: list[PreparedCase],
    model: HybridModel,
    transition_strength: float,
    method_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases:
        prediction = model.predict(case.diag, case.fs_hz, transition_strength)
        metrics = evaluate_output(apply_policy(case.diag, prediction, expanded=True), case.width, case.height)
        rows.append(
            {
                "dataset": case.dataset,
                "participant_id": case.participant_id,
                "case_segment": case.case_segment,
                "corruption": case.corruption,
                "severity": case.severity,
                "seed": case.seed,
                "method": method_name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def tune_observation_model(
    train_cases: list[PreparedCase],
    development_cases: list[PreparedCase],
    max_samples_per_class: int,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for c_value in C_GRID:
        model = fit_hybrid(train_cases, c_value, max_samples_per_class, RANDOM_SEED)
        sequence = evaluate_hybrid_cases(development_cases, model, 0.0, "Hybrid-P")
        participant = participant_aggregate(sequence)
        _, macro = dataset_and_macro_aggregate(participant)
        row = macro.iloc[0].to_dict()
        rows.append({"C": c_value, "Q_local": row["Q_local"], "mae": row["mae"], "rmse": row["rmse"]})
    tuning = pd.DataFrame(rows).sort_values(["Q_local", "mae"], ascending=[False, True]).reset_index(drop=True)
    return float(tuning.iloc[0]["C"]), tuning


def tune_transition_strength(
    model: HybridModel,
    development_cases: list[PreparedCase],
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for strength in LAMBDA_GRID:
        sequence = evaluate_hybrid_cases(development_cases, model, strength, "STRATUS-H")
        participant = participant_aggregate(sequence)
        _, macro = dataset_and_macro_aggregate(participant)
        row = macro.iloc[0].to_dict()
        rows.append(
            {
                "transition_strength": strength,
                "Q_local": row["Q_local"],
                "mae": row["mae"],
                "rmse": row["rmse"],
            }
        )
    tuning = pd.DataFrame(rows).sort_values(["Q_local", "mae"], ascending=[False, True]).reset_index(drop=True)
    return float(tuning.iloc[0]["transition_strength"]), tuning


def evaluate_all_methods(test_cases: list[PreparedCase], model: HybridModel, strength: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, case in enumerate(test_cases, start=1):
        methods = {
            "Pointwise-D": (predict_pointwise_d(case.diag, case.fs_hz), False),
            "STRATUS-D": (predict_stratus_d(case.diag, case.fs_hz), False),
            "Hybrid-P": (model.predict(case.diag, case.fs_hz, 0.0), True),
            "STRATUS-H": (model.predict(case.diag, case.fs_hz, strength), True),
        }
        for method, (prediction, expanded) in methods.items():
            output = apply_policy(case.diag, prediction, expanded)
            metrics = evaluate_output(output, case.width, case.height)
            rows.append(
                {
                    "dataset": case.dataset,
                    "participant_id": case.participant_id,
                    "case_segment": case.case_segment,
                    "corruption": case.corruption,
                    "severity": case.severity,
                    "seed": case.seed,
                    "method": method,
                    **metrics,
                }
            )
        if index % 100 == 0 or index == len(test_cases):
            print(f"  evaluated {index}/{len(test_cases)} corrupted test sequences", flush=True)
    return pd.DataFrame(rows)


def paired_bootstrap(
    participant_results: pd.DataFrame,
    method_a: str,
    method_b: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ("Q_local", "mae", "rmse")
    pivoted: dict[str, pd.DataFrame] = {}
    for dataset in sorted(participant_results["dataset"].unique()):
        data = participant_results[participant_results["dataset"] == dataset]
        wide = data.pivot(index="participant_id", columns="method", values=list(metrics))
        if method_a not in wide.columns.get_level_values(1) or method_b not in wide.columns.get_level_values(1):
            raise ValueError(f"Missing methods in {dataset} bootstrap data.")
        pivoted[dataset] = wide

    observed: dict[str, float] = {}
    for metric in metrics:
        differences = []
        for dataset, wide in pivoted.items():
            differences.append(float((wide[(metric, method_a)] - wide[(metric, method_b)]).mean()))
        observed[metric] = float(np.mean(differences))

    draws = {metric: np.empty(replicates, dtype=float) for metric in metrics}
    for replicate in range(replicates):
        per_dataset = {metric: [] for metric in metrics}
        for _, wide in pivoted.items():
            indices = rng.integers(0, len(wide), size=len(wide))
            sampled = wide.iloc[indices]
            for metric in metrics:
                per_dataset[metric].append(
                    float((sampled[(metric, method_a)] - sampled[(metric, method_b)]).mean())
                )
        for metric in metrics:
            draws[metric][replicate] = float(np.mean(per_dataset[metric]))

    rows = []
    for metric in metrics:
        lower, upper = np.quantile(draws[metric], [0.025, 0.975])
        rows.append(
            {
                "contrast": f"{method_a} minus {method_b}",
                "metric": metric,
                "difference": observed[metric],
                "ci_low": float(lower),
                "ci_high": float(upper),
                "favorable_direction": "positive" if metric == "Q_local" else "negative",
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Self-test data
# -----------------------------------------------------------------------------
def synthetic_clean_trace(dataset: str, participant_id: str, fs_hz: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(round(8.05 * fs_hz))
    t = np.arange(n) / fs_hz
    x = np.empty(n)
    y = np.empty(n)
    cursor = 0
    current = np.array([600.0, 400.0])
    while cursor < n:
        fix_length = min(n - cursor, max(4, int(rng.uniform(0.4, 0.9) * fs_hz)))
        noise = rng.normal(0, 1.2, size=(fix_length, 2)).cumsum(axis=0) * 0.05
        values = current + noise
        x[cursor : cursor + fix_length] = values[:, 0]
        y[cursor : cursor + fix_length] = values[:, 1]
        cursor += fix_length
        if cursor >= n:
            break
        target = np.array([rng.uniform(200, 1200), rng.uniform(150, 850)])
        move_length = min(n - cursor, max(3, int(0.04 * fs_hz)))
        fraction = np.linspace(0, 1, move_length + 2)[1:-1]
        values = current + fraction[:, None] * (target - current)
        x[cursor : cursor + move_length] = values[:, 0]
        y[cursor : cursor + move_length] = values[:, 1]
        cursor += move_length
        current = target
    return pd.DataFrame(
        {
            "dataset": dataset,
            "source_file": f"synthetic_{participant_id}.csv",
            "participant_id": participant_id,
            "segment_id": "synthetic",
            "time_s": t,
            "x": x,
            "y": y,
            "case_segment": f"{dataset}__{participant_id}__seg0",
            "case_index": 0,
        }
    )


def create_self_test_files(folder: Path) -> tuple[Path, Path]:
    frames = []
    split_rows = []
    for dataset, fs, count in (("ETDD70", 250, 6), ("Autism", 60, 8)):
        for index in range(count):
            participant = str(index + 1)
            frames.append(synthetic_clean_trace(dataset, participant, fs, RANDOM_SEED + index + fs))
            split_rows.append(
                {
                    "dataset": dataset,
                    "participant_id": participant,
                    "split": "test" if index >= count - 2 else "train",
                }
            )
    reference_path = folder / "self_test_reference.csv"
    split_path = folder / "self_test_split.csv"
    pd.concat(frames, ignore_index=True).to_csv(reference_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    return reference_path, split_path


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Participant-safe comparison of Pointwise-D, STRATUS-D, Hybrid-P, and selective STRATUS-H."
    )
    parser.add_argument("--reference", type=Path, help="Path to v7 clean_reference_segments.csv")
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "tables" / "participant_split.csv",
        help="Participant split CSV.",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "hybrid" / "primary")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--train-cap-per-dataset", type=int, default=120)
    parser.add_argument("--max-samples-per-class", type=int, default=75000)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--quick", action="store_true", help="Use 2 seeds and one severity for a fast smoke run.")
    parser.add_argument("--self-test", action="store_true", help="Generate synthetic reference data and validate the full runner.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        reference_path, split_path = create_self_test_files(args.output)
        args.reference = reference_path
        args.split = split_path
        args.quick = True
        args.train_cap_per_dataset = 40
        args.max_samples_per_class = 5000
        args.bootstrap = min(args.bootstrap, 200)
    if args.reference is None:
        raise SystemExit("--reference is required unless --self-test is used.")
    if not args.reference.exists():
        raise SystemExit(f"Reference file not found: {args.reference}")
    if not args.split.exists():
        raise SystemExit(f"Split file not found: {args.split}")

    reference = pd.read_csv(args.reference, low_memory=False)
    reference["participant_id"] = reference["participant_id"].map(normalize_id)
    validate_reference(reference)
    split = pd.read_csv(args.split, dtype=str)
    split["participant_id"] = split["participant_id"].map(normalize_id)
    split["dataset"] = split["dataset"].astype(str)
    split["split"] = split["split"].astype(str).str.lower()

    clean_cases = build_clean_cases(reference)
    available = {
        (str(frame["dataset"].iloc[0]), normalize_id(frame["participant_id"].iloc[0]))
        for frame in clean_cases.values()
    }
    outer_train = {
        (row.dataset, row.participant_id)
        for row in split.itertuples(index=False)
        if row.split == "train" and (row.dataset, row.participant_id) in available
    }
    outer_test = {
        (row.dataset, row.participant_id)
        for row in split.itertuples(index=False)
        if row.split == "test" and (row.dataset, row.participant_id) in available
    }
    if not outer_train or not outer_test:
        raise ValueError("The split does not overlap the reference participants as expected.")

    inner_train, development = split_training_and_development(
        outer_train, dev_fraction=args.dev_fraction, seed=RANDOM_SEED
    )
    eval_seeds = (0, 1) if args.quick else EVAL_SEEDS
    train_seeds = (0, 1) if args.quick else TRAIN_SEEDS
    severities = ("medium",) if args.quick else SEVERITIES

    print(
        f"Reference: {len(reference):,} rows, {len(clean_cases)} segments, "
        f"{len(available)} participants. Outer train={len(outer_train)}, test={len(outer_test)}.",
        flush=True,
    )
    print(f"Inner train={len(inner_train)}, development={len(development)}.", flush=True)

    print("Preparing inner-training corruption cases...", flush=True)
    training_cases = generate_prepared_cases(
        clean_cases,
        inner_train,
        train_seeds,
        severities,
        CORRUPTIONS,
        cap_per_dataset=args.train_cap_per_dataset,
        selection_seed=RANDOM_SEED,
    )
    print("Preparing development cases...", flush=True)
    development_cases = generate_prepared_cases(
        clean_cases,
        development,
        train_seeds,
        severities,
        CORRUPTIONS,
        cap_per_dataset=None,
        selection_seed=RANDOM_SEED + 1,
    )

    print("Tuning the observation model on held-out development participants...", flush=True)
    best_c, c_tuning = tune_observation_model(
        training_cases, development_cases, args.max_samples_per_class
    )
    tuning_model = fit_hybrid(
        training_cases,
        best_c,
        args.max_samples_per_class,
        RANDOM_SEED,
    )
    print("Tuning only the temporal strength on the same development participants...", flush=True)
    best_strength, lambda_tuning = tune_transition_strength(tuning_model, development_cases)

    print(
        f"Selected C={best_c:g}; transition strength={best_strength:g}. "
        "Now refitting on all outer-training participants.",
        flush=True,
    )
    full_training_cases = generate_prepared_cases(
        clean_cases,
        outer_train,
        train_seeds,
        severities,
        CORRUPTIONS,
        cap_per_dataset=args.train_cap_per_dataset,
        selection_seed=RANDOM_SEED + 2,
    )
    final_model = fit_hybrid(
        full_training_cases,
        best_c,
        args.max_samples_per_class,
        RANDOM_SEED + 3,
    )

    print("Preparing held-out test cases...", flush=True)
    test_cases = generate_prepared_cases(
        clean_cases,
        outer_test,
        eval_seeds,
        severities,
        CORRUPTIONS,
        cap_per_dataset=None,
        selection_seed=RANDOM_SEED + 4,
    )
    print(f"Evaluating {len(test_cases)} corrupted held-out sequences × 4 methods...", flush=True)
    sequence_results = evaluate_all_methods(test_cases, final_model, best_strength)
    sequence_results = add_local_action_score(sequence_results)
    participant_results = participant_aggregate(sequence_results)
    dataset_results, macro_results = dataset_and_macro_aggregate(participant_results)

    bootstrap_frames = [
        paired_bootstrap(
            participant_results,
            "STRATUS-H",
            "Hybrid-P",
            args.bootstrap,
            RANDOM_SEED + 10,
        ),
        paired_bootstrap(
            participant_results,
            "STRATUS-H",
            "Pointwise-D",
            args.bootstrap,
            RANDOM_SEED + 20,
        ),
    ]
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)

    corruption_results = (
        sequence_results.groupby(["dataset", "corruption", "method"], as_index=False)[
            [
                "mae",
                "rmse",
                "long_hallucination",
                "short_recovery",
                "movement_preservation",
                "stable_preservation",
                "unstable_repair_score",
                "Q_local",
            ]
        ]
        .mean(numeric_only=True)
    )

    c_tuning.to_csv(args.output / "observation_C_tuning.csv", index=False)
    lambda_tuning.to_csv(args.output / "transition_strength_tuning.csv", index=False)
    sequence_results.to_csv(args.output / "sequence_results.csv", index=False)
    participant_results.to_csv(args.output / "participant_results.csv", index=False)
    dataset_results.to_csv(args.output / "dataset_results.csv", index=False)
    macro_results.to_csv(args.output / "macro_results.csv", index=False)
    corruption_results.to_csv(args.output / "corruption_results.csv", index=False)
    bootstrap.to_csv(args.output / "paired_bootstrap_contrasts.csv", index=False)

    summary = {
        "reference_file": str(args.reference.resolve()),
        "split_file": str(args.split.resolve()),
        "quick_mode": bool(args.quick),
        "n_reference_segments": len(clean_cases),
        "n_outer_train_participants": len(outer_train),
        "n_outer_test_participants": len(outer_test),
        "n_test_sequences": len(test_cases),
        "best_C": best_c,
        "best_transition_strength": best_strength,
        "bootstrap_replicates": args.bootstrap,
        "runtime_seconds": time.time() - started,
        "primary_contrast": "STRATUS-H versus Hybrid-P",
        "interpretation_rule": (
            "The HMM contributes independent value only when the paired STRATUS-H minus "
            "Hybrid-P Q_local interval is above zero without worsening safety components."
        ),
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDataset-macro held-out results:")
    display_columns = [
        "method",
        "mae",
        "rmse",
        "long_hallucination",
        "short_recovery",
        "movement_preservation",
        "stable_preservation",
        "unstable_repair_score",
        "Q_local",
    ]
    print(macro_results[display_columns].sort_values("Q_local", ascending=False).to_string(index=False))
    print("\nPaired participant bootstrap contrasts:")
    print(bootstrap.to_string(index=False))
    print(f"\nOutputs written to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
