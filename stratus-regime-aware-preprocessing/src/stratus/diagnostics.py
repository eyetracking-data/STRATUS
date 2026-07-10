from __future__ import annotations

import numpy as np
import pandas as pd


def _full_missing_run_lengths(missing: np.ndarray) -> np.ndarray:
    """Assign the total contiguous missing-run length to every sample in the run.

    Example: [F,T,T,T,F,T] -> [0,3,3,3,0,1].
    This is important because the beginning of a long gap must not be
    misclassified as a short recoverable loss.
    """
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


def compute_diagnostics(df: pd.DataFrame, screen_width: float, screen_height: float) -> pd.DataFrame:
    """Compute STRATUS diagnostic components for a canonical x/y stream.

    Output diagnostics:
    - m_t: missing/invalid indicator
    - r_t: total length of the current missing run, assigned to the full run
    - c_t: first-order local change / velocity
    - alpha_t: second-order local change / acceleration
    - u_t: local instability / jitter
    - p_t: plausibility violation indicator
    """
    out = df.copy()
    x = out["x"].to_numpy(dtype=float)
    y = out["y"].to_numpy(dtype=float)
    t = out["time_s"].to_numpy(dtype=float)

    finite_xy = np.isfinite(x) & np.isfinite(y)
    plaus = np.zeros(len(out), dtype=bool)
    plaus[finite_xy] = (
        (x[finite_xy] < 0) | (x[finite_xy] > screen_width) |
        (y[finite_xy] < 0) | (y[finite_xy] > screen_height)
    )

    # Missingness and plausibility are distinct diagnostics. A finite
    # out-of-domain value is not converted into a loss state; it remains
    # available to the Unstable state through p_t.
    missing = ~finite_xy
    invalid_for_dynamics = missing | plaus
    full_run = _full_missing_run_lengths(missing)

    dt = np.diff(t, prepend=np.nan)
    dt[~np.isfinite(dt) | (dt <= 0)] = np.nan

    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    dist = np.sqrt(dx**2 + dy**2)
    velocity = dist / dt

    # Avoid turning transitions around missing values into artificial velocities.
    velocity[invalid_for_dynamics] = np.nan
    velocity[np.r_[True, invalid_for_dynamics[:-1]]] = np.nan

    dv = np.diff(velocity, prepend=np.nan)
    acceleration = np.abs(dv) / dt
    acceleration[invalid_for_dynamics] = np.nan

    # Rolling local instability around the coordinate median.
    coords = pd.DataFrame({"x": x, "y": y})
    med_x = coords["x"].rolling(window=7, center=True, min_periods=3).median()
    med_y = coords["y"].rolling(window=7, center=True, min_periods=3).median()
    jitter = np.sqrt((coords["x"] - med_x)**2 + (coords["y"] - med_y)**2)
    jitter[invalid_for_dynamics] = np.nan

    out["m_t"] = missing.astype(int)
    out["r_t"] = full_run
    out["c_t"] = velocity
    out["alpha_t"] = acceleration
    out["u_t"] = jitter
    out["p_t"] = plaus.astype(int)
    return out
