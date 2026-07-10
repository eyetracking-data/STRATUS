from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _point_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def evaluate_output(df: pd.DataFrame, screen_width: float, screen_height: float) -> dict:
    """Evaluate output against the clean reference and injected regimes.

    The returned metrics deliberately separate numerical reconstruction from
    local action quality. ``unstable_repair_gain`` may be negative when a
    method makes an injected unstable region worse. Its clipped companion is
    used only as one bounded component of the composite local-action score.
    """
    x = df["x_clean"].to_numpy(dtype=float)
    y = df["y_clean"].to_numpy(dtype=float)
    xo = df["x"].to_numpy(dtype=float)
    yo = df["y"].to_numpy(dtype=float)

    finite_ref = np.isfinite(x) & np.isfinite(y)
    finite_out = np.isfinite(xo) & np.isfinite(yo)
    valid = finite_ref & finite_out

    dist = _point_distance(xo[valid], yo[valid], x[valid], y[valid])
    mae = float(np.mean(dist)) if len(dist) else np.nan
    rmse = float(np.sqrt(np.mean(dist ** 2))) if len(dist) else np.nan
    retention = float(np.mean(finite_out))

    regime = df["true_regime"].astype(str)
    long_mask = regime.eq("LossLong").to_numpy()
    short_mask = regime.eq("LossShort").to_numpy()
    move_mask = regime.eq("Movement").to_numpy()
    stable_mask = regime.eq("Stable").to_numpy()
    unstable_mask = regime.eq("Unstable").to_numpy()

    long_halluc = float(np.mean(finite_out[long_mask])) if long_mask.any() else np.nan
    short_recovery = float(np.mean(finite_out[short_mask])) if short_mask.any() else np.nan

    # Preserve genuine dynamics at transitions labelled Movement. The mask is
    # applied to consecutive steps instead of joining non-adjacent movement
    # samples into artificial jumps.
    clean_step = _point_distance(np.diff(x), np.diff(y), 0.0, 0.0)
    out_step = _point_distance(np.diff(xo), np.diff(yo), 0.0, 0.0)
    move_step_mask = move_mask[1:] & finite_ref[1:] & finite_ref[:-1] & finite_out[1:] & finite_out[:-1]
    if move_step_mask.any():
        clean_dyn = float(np.mean(clean_step[move_step_mask]))
        out_dyn = float(np.mean(out_step[move_step_mask]))
        if np.isfinite(clean_dyn) and clean_dyn > 0 and np.isfinite(out_dyn):
            movement_preservation = float(np.clip(1.0 - abs(out_dyn / clean_dyn - 1.0), 0.0, 1.0))
        else:
            movement_preservation = np.nan
    else:
        movement_preservation = np.nan

    # Stable means preserve. This metric is therefore the fraction of stable
    # samples left numerically unchanged (within floating-point tolerance).
    stable_reference = stable_mask & finite_ref
    if stable_reference.any():
        unchanged = finite_out & np.isclose(xo, x, rtol=0.0, atol=1e-9) & np.isclose(yo, y, rtol=0.0, atol=1e-9)
        stable_preservation = float(np.mean(unchanged[stable_reference]))
    else:
        stable_preservation = np.nan

    # Improvement in injected Unstable regions relative to the corrupted input.
    # A raw stream scores 0, perfect restoration scores 1, and harmful changes
    # yield a negative gain. Missing outputs receive no improvement credit.
    unstable_repair_gain = np.nan
    unstable_repair_score = np.nan
    if unstable_mask.any() and {"x_degraded", "y_degraded"}.issubset(df.columns):
        xd = df["x_degraded"].to_numpy(dtype=float)
        yd = df["y_degraded"].to_numpy(dtype=float)
        finite_deg = np.isfinite(xd) & np.isfinite(yd)
        base_mask = unstable_mask & finite_ref & finite_deg
        if base_mask.any():
            base_error = _point_distance(xd[base_mask], yd[base_mask], x[base_mask], y[base_mask])
            base_mae = float(np.mean(base_error))
            if np.isfinite(base_mae) and base_mae > 1e-12:
                output_error = _point_distance(xo[base_mask], yo[base_mask], x[base_mask], y[base_mask])
                output_error[~(finite_out[base_mask])] = base_error[~(finite_out[base_mask])]
                out_mae = float(np.mean(output_error))
                unstable_repair_gain = float(1.0 - out_mae / base_mae)
                unstable_repair_score = float(np.clip(unstable_repair_gain, 0.0, 1.0))

    plaus_violation = np.zeros(len(df), dtype=bool)
    plaus_violation[finite_out] = (
        (xo[finite_out] < 0) | (xo[finite_out] > screen_width) |
        (yo[finite_out] < 0) | (yo[finite_out] > screen_height)
    )
    plaus_score = 1.0 - float(np.mean(plaus_violation[finite_out])) if finite_out.any() else np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "retention": retention,
        "long_hallucination": long_halluc,
        "short_recovery": short_recovery,
        "movement_preservation": movement_preservation,
        "stable_preservation": stable_preservation,
        "unstable_repair_gain": unstable_repair_gain,
        "unstable_repair_score": unstable_repair_score,
        "plausibility_score": plaus_score,
    }


def regime_f1(df: pd.DataFrame) -> pd.DataFrame:
    """Class-conditional F1, leaving absent reference classes undefined."""
    if "pred_regime" not in df.columns:
        return pd.DataFrame()
    labels = ["Stable", "Movement", "LossShort", "LossLong", "Unstable"]
    y_true = df["true_regime"].astype(str)
    y_pred = df["pred_regime"].astype(str)
    rows = []
    for label in labels:
        truth = y_true.eq(label)
        support = int(truth.sum())
        score = np.nan if support == 0 else float(
            f1_score(truth, y_pred.eq(label), zero_division=0)
        )
        rows.append({"regime": label, "f1": score, "support": support})
    return pd.DataFrame(rows)
