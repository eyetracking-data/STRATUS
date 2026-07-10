from __future__ import annotations

import numpy as np
import pandas as pd


STATES = ["Stable", "Movement", "LossShort", "LossLong", "Unstable"]


def _safe_quantile(s: pd.Series, q: float, default: float) -> float:
    values = s.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return default
    return float(np.nanquantile(values, q))


def _initial_state_scores(
    df: pd.DataFrame,
    fs_hz: float,
    short_gap_seconds: float = 0.50,
    velocity_quantile: float = 0.90,
    jitter_quantile: float = 0.995,
    alpha_quantile: float = 0.995,
) -> pd.DataFrame:
    """Create interpretable emission-like scores for the five STRATUS states.

    Higher score means more compatible with the state. Scores are heuristic
    diagnostic emissions, then Viterbi adds temporal persistence.
    """
    n = len(df)
    scores = pd.DataFrame(0.0, index=df.index, columns=STATES)

    short_gap_samples = max(1, int(round(short_gap_seconds * fs_hz)))
    missing = df["m_t"].astype(bool)
    finite = ~missing

    c = df["c_t"].astype(float)
    u = df["u_t"].astype(float)
    alpha = df["alpha_t"].astype(float)

    vel_thr = _safe_quantile(c[finite], velocity_quantile, default=np.inf)
    # Extreme thresholds for unstable: avoid consuming genuine movement.
    jit_thr = _safe_quantile(u[finite], jitter_quantile, default=np.inf)
    alpha_thr = _safe_quantile(alpha[finite], alpha_quantile, default=np.inf)

    loss_short = missing & (df["r_t"] <= short_gap_samples)
    loss_long = missing & (df["r_t"] > short_gap_samples)

    # Base scores.
    scores.loc[finite, "Stable"] = 2.0
    scores.loc[finite & (c >= vel_thr), "Movement"] = 5.0
    scores.loc[loss_short, "LossShort"] = 8.0
    scores.loc[loss_long, "LossLong"] = 10.0

    # Conservative unstable detection: implausible finite samples or extreme local instability.
    unstable = finite & ((df["p_t"].astype(bool)) | (u >= jit_thr) | (alpha >= alpha_thr))
    scores.loc[unstable, "Unstable"] = 7.0
    scores.loc[unstable, "Stable"] = 0.0

    # Missing samples should not be movement/stable/unstable.
    scores.loc[missing, ["Stable", "Movement", "Unstable"]] = -8.0
    scores.loc[loss_short, ["LossLong"]] = -4.0
    scores.loc[loss_long, ["LossShort"]] = -8.0

    # Finite samples should not be loss states.
    scores.loc[finite, ["LossShort", "LossLong"]] = -8.0

    return scores


def viterbi_decode(scores: pd.DataFrame, stay_bonus: float = 1.25, switch_penalty: float = -0.75) -> pd.Series:
    """Viterbi decoder over diagnostic emission-like scores.

    This implements the temporally persistent STRATUS decoder without external
    HMM dependencies. It is equivalent to a log-domain HMM with interpretable
    diagnostic emissions and a transition matrix favoring self-transitions.
    """
    states = list(scores.columns)
    score_arr = scores.to_numpy(dtype=float)
    n, k = score_arr.shape

    trans = np.full((k, k), switch_penalty, dtype=float)
    np.fill_diagonal(trans, stay_bonus)

    dp = np.zeros((n, k), dtype=float)
    back = np.zeros((n, k), dtype=int)
    dp[0] = score_arr[0]

    for t in range(1, n):
        prev = dp[t - 1][:, None] + trans
        back[t] = np.argmax(prev, axis=0)
        dp[t] = score_arr[t] + np.max(prev, axis=0)

    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]

    return pd.Series([states[i] for i in path], index=scores.index, dtype="object")


def predict_regimes_stratus(
    df: pd.DataFrame,
    fs_hz: float,
    short_gap_seconds: float = 0.50,
    velocity_quantile: float = 0.90,
    stay_bonus: float = 1.25,
) -> pd.Series:
    """Predict STRATUS regimes with diagnostic emissions + Viterbi persistence."""
    scores = _initial_state_scores(
        df,
        fs_hz=fs_hz,
        short_gap_seconds=short_gap_seconds,
        velocity_quantile=velocity_quantile,
    )
    return viterbi_decode(scores, stay_bonus=stay_bonus)


# Backward-compatible name used by earlier notebooks.
def predict_regimes_simple(
    df: pd.DataFrame,
    fs_hz: float,
    short_gap_seconds: float = 0.50,
    velocity_quantile: float = 0.90,
    jitter_quantile: float = 0.995,
) -> pd.Series:
    return predict_regimes_stratus(
        df,
        fs_hz=fs_hz,
        short_gap_seconds=short_gap_seconds,
        velocity_quantile=velocity_quantile,
    )


def _interpolate_only_short_runs(values: pd.Series, short_mask: pd.Series) -> pd.Series:
    """Interpolate only samples marked as short recoverable loss.

    Long-loss samples stay missing even if they lie between finite values.
    """
    full_interp = values.interpolate(method="linear", limit_direction="both")
    out = values.copy()
    out.loc[short_mask] = full_interp.loc[short_mask]
    return out


def apply_state_to_action(
    df: pd.DataFrame,
    regimes: pd.Series,
    stable_smoothing: bool = False,
) -> pd.DataFrame:
    """Apply explicit eye-tracking state-to-action policy.

    Stable: preserve by default. This is the v3 tuned behavior because
    unnecessary smoothing distorted already valid reference segments and
    reduced movement preservation.
    Movement: preserve raw dynamics
    LossShort: interpolate only these short-run samples
    LossLong: keep missing
    Unstable: robust local median correction
    """
    out = df.copy()
    out["pred_regime"] = regimes.values

    out["x_out"] = out["x"]
    out["y_out"] = out["y"]

    # LossShort: interpolate cautiously only on samples decoded as LossShort.
    short_mask = out["pred_regime"].eq("LossShort")
    out["x_out"] = _interpolate_only_short_runs(out["x_out"], short_mask)
    out["y_out"] = _interpolate_only_short_runs(out["y_out"], short_mask)

    # LossLong: keep missing for the complete long-loss run.
    long_mask = out["pred_regime"].eq("LossLong")
    out.loc[long_mask, ["x_out", "y_out"]] = np.nan

    # Stable: preserve by default. Optional mild smoothing can be enabled
    # for applications where stable noise reduction is more important than
    # exact reconstruction of clean reference segments.
    stable_mask = out["pred_regime"].eq("Stable")
    if stable_smoothing:
        smooth = out[["x", "y"]].rolling(window=5, center=True, min_periods=1).mean()
        out.loc[stable_mask, "x_out"] = smooth.loc[stable_mask, "x"]
        out.loc[stable_mask, "y_out"] = smooth.loc[stable_mask, "y"]

    # Unstable: robust median correction.
    med = out[["x", "y"]].rolling(window=7, center=True, min_periods=1).median()
    unstable_mask = out["pred_regime"].eq("Unstable")
    out.loc[unstable_mask, "x_out"] = med.loc[unstable_mask, "x"]
    out.loc[unstable_mask, "y_out"] = med.loc[unstable_mask, "y"]

    # Movement: preserve raw degraded signal by leaving x_out/y_out unchanged.

    result = out.copy()
    result["x"] = result["x_out"]
    result["y"] = result["y_out"]
    return result.drop(columns=["x_out", "y_out"])



def apply_oracle_policy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the same action policy using injected ground-truth regimes.

    This is an upper-bound analysis of the action policy, not a deployable
    preprocessing method.
    """
    if "true_regime" not in df.columns:
        raise ValueError("Oracle policy requires a true_regime column.")
    return apply_state_to_action(
        df,
        regimes=df["true_regime"].astype(str),
        stable_smoothing=False,
    )
