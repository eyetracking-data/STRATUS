from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "results" / "hybrid" / "primary"
SHIFTED = ROOT / "results" / "hybrid" / "shifted"
EXPECTED_REFERENCE_SHA256 = "c4e4b81c948e28c80e0ba50c84e118e54081d5257cb3c5b7a2a9d1e85e467525"


def close(actual: float, expected: float, tolerance: float = 5e-6) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"Expected {expected}, obtained {actual}")


def verify_reference(path: Path | None) -> None:
    if path is None:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_REFERENCE_SHA256:
        raise AssertionError(f"Reference SHA-256 mismatch: {digest}")


def main(reference: Path | None = None) -> int:
    macro = pd.read_csv(PRIMARY / "macro_results.csv").set_index("method")
    close(macro.loc["STRATUS-H", "mae"], 3.597716)
    close(macro.loc["STRATUS-H", "rmse"], 14.615126)
    close(macro.loc["STRATUS-H", "Q_local"], 0.919967)
    close(macro.loc["Hybrid-P", "Q_local"], 0.914260)
    close(macro.loc["Pointwise-D", "Q_local"], 0.881643)
    close(macro.loc["STRATUS-D", "Q_local"], 0.840825)
    for method in ["STRATUS-H", "Hybrid-P", "Pointwise-D", "STRATUS-D"]:
        close(macro.loc[method, "long_hallucination"], 0.0)
        close(macro.loc[method, "short_recovery"], 1.0)

    primary_ci = pd.read_csv(PRIMARY / "paired_bootstrap_contrasts.csv")
    q = primary_ci[(primary_ci["contrast"] == "STRATUS-H minus Hybrid-P") & (primary_ci["metric"] == "Q_local")].iloc[0]
    if not (q["difference"] > 0 and q["ci_low"] > 0):
        raise AssertionError("Primary STRATUS-H minus Hybrid-P Q_local interval is not above zero.")

    participants = pd.read_csv(PRIMARY / "participant_results.csv")
    pivot = participants.pivot_table(index=["dataset", "participant_id"], columns="method", values="Q_local")
    if len(pivot) != 14 or not (pivot["STRATUS-H"] > pivot["Hybrid-P"]).all():
        raise AssertionError("Expected STRATUS-H to exceed Hybrid-P for all 14 test identifiers.")

    shifted = pd.read_csv(SHIFTED / "shifted_macro_results.csv").set_index("method")
    close(shifted.loc["STRATUS-H", "Q_local"], 0.906536)
    shifted_ci = pd.read_csv(SHIFTED / "shifted_bootstrap_contrasts.csv")
    sq = shifted_ci[(shifted_ci["contrast"] == "STRATUS-H minus Hybrid-P") & (shifted_ci["metric"] == "Q_local")].iloc[0]
    if not (sq["difference"] > 0 and sq["ci_low"] > 0):
        raise AssertionError("Shifted STRATUS-H minus Hybrid-P Q_local interval is not above zero.")

    if len(pd.read_csv(PRIMARY / "sequence_results.csv")) != 8400:
        raise AssertionError("Primary sequence result row count changed.")
    if len(pd.read_csv(SHIFTED / "shifted_sequence_results.csv")) != 1680:
        raise AssertionError("Shifted sequence result row count changed.")

    verify_reference(reference)
    print("PASS: committed STRATUS-H results, contrasts, invariants, and row counts are consistent.")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.reference))
