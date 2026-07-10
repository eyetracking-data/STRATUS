from __future__ import annotations

import numpy as np
import pandas as pd


REGIMES = ["Stable", "Movement", "LossShort", "LossLong", "Unstable"]


def _mark_natural_movement(out: pd.DataFrame, quantile: float = 0.90) -> pd.Series:
    dx = out["x_clean"].diff()
    dy = out["y_clean"].diff()
    v = np.sqrt(dx**2 + dy**2)
    thr = np.nanquantile(v.to_numpy(dtype=float), quantile)
    return v >= thr


def inject_corruption(df: pd.DataFrame, corruption: str, severity: str, seed: int) -> pd.DataFrame:
    """Inject controlled corruption into a clean x/y reference segment.

    Corruption types are kept non-overlapping where possible. In particular,
    spikes and jitter are not injected into missing intervals, because that
    would split a long loss into artificial finite islands and make the
    operator metric ambiguous.
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)
    out["x_clean"] = out["x"]
    out["y_clean"] = out["y"]
    out["true_regime"] = "Stable"

    movement_mask = _mark_natural_movement(out, quantile=0.90)
    out.loc[movement_mask, "true_regime"] = "Movement"

    sev = {"low": 1.0, "medium": 1.7, "high": 2.5}[severity]

    def interval(start_frac, length_frac):
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
        idx = interval(0.78, 0.08)  # separated from long gap even for high severity
        idx = idx[finite_available.iloc[idx].to_numpy()]
        noise = rng.normal(0, 8.0 * sev, size=(len(idx), 2))
        out.loc[idx, ["x", "y"]] = out.loc[idx, ["x", "y"]].to_numpy() + noise
        out.loc[idx, "true_regime"] = "Unstable"

    finite_available = out["x"].notna() & out["y"].notna()

    if corruption in {"spike", "mixed"}:
        k = max(2, int(0.015 * n * sev))
        candidates = np.where(finite_available.to_numpy())[0]
        if len(candidates) > 0:
            k = min(k, len(candidates))
            idx = rng.choice(candidates, size=k, replace=False)
            out.loc[idx, "x"] = out["x_clean"].median() + rng.normal(0, 350 * sev, size=k)
            out.loc[idx, "y"] = out["y_clean"].median() + rng.normal(0, 250 * sev, size=k)
            out.loc[idx, "true_regime"] = "Unstable"

    # Preserve the actually observed corrupted stream for repair metrics.
    # Processing methods may overwrite x/y, but x_degraded/y_degraded remain
    # unchanged and let us measure improvement specifically in Unstable regions.
    out["x_degraded"] = out["x"]
    out["y_degraded"] = out["y"]
    out["corruption"] = corruption
    out["severity"] = severity
    out["seed"] = seed
    return out
