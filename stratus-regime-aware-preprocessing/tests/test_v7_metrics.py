import numpy as np
import pandas as pd

from stratus.baselines import short_gap_only
from stratus.metrics import evaluate_output


def test_short_gap_only_never_partly_fills_long_run():
    df = pd.DataFrame({
        "x": [0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0],
        "y": [0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0],
    })
    out = short_gap_only(df, max_gap_samples=2)
    assert out.loc[1:2, ["x", "y"]].notna().all().all()
    assert out.loc[4:7, ["x", "y"]].isna().all().all()


def test_unstable_repair_gain_raw_zero_and_clean_one():
    base = pd.DataFrame({
        "x_clean": [0.0, 1.0, 2.0], "y_clean": [0.0, 1.0, 2.0],
        "x_degraded": [0.0, 11.0, 2.0], "y_degraded": [0.0, 11.0, 2.0],
        "true_regime": ["Stable", "Unstable", "Stable"],
    })
    raw = base.assign(x=base.x_degraded, y=base.y_degraded)
    repaired = base.assign(x=base.x_clean, y=base.y_clean)
    m_raw = evaluate_output(raw, 100, 100)
    m_repaired = evaluate_output(repaired, 100, 100)
    assert abs(m_raw["unstable_repair_gain"]) < 1e-12
    assert abs(m_repaired["unstable_repair_gain"] - 1.0) < 1e-12
    assert m_repaired["stable_preservation"] == 1.0
