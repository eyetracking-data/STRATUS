from __future__ import annotations

import numpy as np
import pandas as pd


def _interp_xy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["x", "y"]:
        out[c] = out[c].interpolate(method="linear", limit_direction="both")
    return out


def _full_run_lengths(mask: np.ndarray) -> np.ndarray:
    """Assign the complete contiguous run length to every True position."""
    mask = np.asarray(mask, dtype=bool)
    lengths = np.zeros(len(mask), dtype=int)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        lengths[i:j] = j - i
        i = j
    return lengths


def raw_degraded(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return df.copy()


def global_interpolate(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return _interp_xy(df)


def short_gap_only(df: pd.DataFrame, max_gap_samples: int = 25, **kwargs) -> pd.DataFrame:
    """Interpolate complete missing runs only when their total length is short.

    Pandas' ``limit=`` parameter is deliberately not used: with
    ``limit_direction='both'`` it fills the ends of a long gap and therefore
    does not implement a genuine short-gap-only baseline.
    """
    if max_gap_samples < 1:
        raise ValueError("max_gap_samples must be at least one")

    out = df.copy()
    missing = out[["x", "y"]].isna().any(axis=1).to_numpy()
    run_lengths = _full_run_lengths(missing)
    short_mask = pd.Series(
        missing & (run_lengths <= int(max_gap_samples)),
        index=out.index,
    )

    for c in ["x", "y"]:
        fully_interpolated = out[c].interpolate(
            method="linear",
            limit_direction="both",
        )
        out.loc[short_mask, c] = fully_interpolated.loc[short_mask]
    return out


def global_smooth(df: pd.DataFrame, window: int = 7, **kwargs) -> pd.DataFrame:
    out = df.copy()
    for c in ["x", "y"]:
        out[c] = out[c].rolling(window=window, center=True, min_periods=1).mean()
    return out


def interp_smooth(df: pd.DataFrame, window: int = 7, **kwargs) -> pd.DataFrame:
    return global_smooth(global_interpolate(df), window=window)


def robust_clip(df: pd.DataFrame, screen_width: float, screen_height: float, **kwargs) -> pd.DataFrame:
    out = df.copy()
    out["x"] = out["x"].clip(0, screen_width)
    out["y"] = out["y"].clip(0, screen_height)
    return out
