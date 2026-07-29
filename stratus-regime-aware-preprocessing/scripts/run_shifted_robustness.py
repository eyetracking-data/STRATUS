from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_hybrid_experiment.py"
spec = importlib.util.spec_from_file_location("stratus_hybrid_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
sys.modules["stratus_hybrid_runner"] = runner
assert spec.loader is not None
spec.loader.exec_module(runner)


def inject_shifted(df: pd.DataFrame, corruption: str, severity: str, seed: int) -> pd.DataFrame:
    """Inject mechanisms deliberately shifted from the primary generator."""
    rng = np.random.default_rng(900000 + seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)
    out["x_clean"] = out["x"]
    out["y_clean"] = out["y"]
    out["true_regime"] = "Stable"
    out["unstable_subtype"] = ""
    movement = runner._mark_natural_movement(out, quantile=0.85)
    out.loc[movement, "true_regime"] = "Movement"
    dataset = str(out["dataset"].iloc[0])
    fs_hz = runner.estimate_sampling_rate_hz(
        out, runner.DEFAULT_METADATA[dataset]["sampling_rate_hz"]
    )
    severity_factor = {"low": 1.0, "medium": 1.7, "high": 2.5}[severity]
    occupied = np.zeros(n, dtype=bool)

    def sample_interval(duration_low: float, duration_high: float) -> np.ndarray:
        length = max(2, int(round(rng.uniform(duration_low, duration_high) * fs_hz)))
        min_start = int(0.08 * n)
        max_start = max(min_start, int(0.92 * n) - length)
        for _ in range(100):
            start = int(rng.integers(min_start, max_start + 1)) if max_start >= min_start else 0
            idx = np.arange(start, min(n, start + length))
            if not occupied[idx].any():
                occupied[idx] = True
                return idx
        start = max(0, min(n - length, min_start))
        idx = np.arange(start, min(n, start + length))
        occupied[idx] = True
        return idx

    if corruption in {"short_gap", "mixed"}:
        idx = sample_interval(0.10 * severity_factor, min(0.49, 0.22 * severity_factor))
        out.loc[idx, ["x", "y"]] = np.nan
        out.loc[idx, "true_regime"] = "LossShort"
    if corruption in {"long_gap", "mixed"}:
        idx = sample_interval(max(0.51, 0.58 * severity_factor), min(1.8, 0.85 * severity_factor))
        out.loc[idx, ["x", "y"]] = np.nan
        out.loc[idx, "true_regime"] = "LossLong"

    finite = out["x"].notna() & out["y"].notna()
    if corruption in {"jitter", "mixed"}:
        idx = sample_interval(0.12, min(0.65, 0.25 * severity_factor))
        idx = idx[finite.iloc[idx].to_numpy()]
        if len(idx):
            laplace = rng.laplace(0, 4.5 * severity_factor, size=(len(idx), 2))
            phase = np.linspace(0, rng.uniform(1.0, 2.5) * np.pi, len(idx))
            drift = np.column_stack([np.sin(phase), np.cos(phase)]) * (5.0 * severity_factor)
            out.loc[idx, ["x", "y"]] = out.loc[idx, ["x", "y"]].to_numpy() + laplace + drift
            out.loc[idx, "true_regime"] = "Unstable"
            out.loc[idx, "unstable_subtype"] = "burst"

    finite = out["x"].notna() & out["y"].notna()
    if corruption in {"spike", "mixed"}:
        candidates = np.where(finite.to_numpy())[0]
        if len(candidates):
            event_count = max(1, int(round(0.006 * n * severity_factor)))
            pool = candidates[:-2] if len(candidates) > 2 else candidates
            starts = rng.choice(pool, size=min(event_count, max(1, len(pool))), replace=False)
            indices: list[int] = []
            for start in starts:
                width = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
                indices.extend(
                    j for j in range(int(start), min(n, int(start) + width)) if finite.iloc[j]
                )
            idx = np.asarray(sorted(set(indices)), dtype=int)
            if len(idx):
                out.loc[idx, "x"] = out.loc[idx, "x"].to_numpy() + rng.standard_t(3, len(idx)) * 220 * severity_factor
                out.loc[idx, "y"] = out.loc[idx, "y"].to_numpy() + rng.standard_t(3, len(idx)) * 160 * severity_factor
                out.loc[idx, "true_regime"] = "Unstable"
                out.loc[idx, "unstable_subtype"] = "impulse"

    out["x_degraded"] = out["x"]
    out["y_degraded"] = out["y"]
    out["corruption"] = corruption
    out["severity"] = severity
    out["seed"] = seed
    return out


def prepare_shifted(clean: pd.DataFrame, corruption: str, severity: str, seed: int):
    dataset = str(clean["dataset"].iloc[0])
    metadata = runner.DEFAULT_METADATA[dataset]
    fs_hz = runner.estimate_sampling_rate_hz(clean, metadata["sampling_rate_hz"])
    degraded = inject_shifted(clean, corruption, severity, seed)
    diagnostics = runner.compute_diagnostics(degraded, metadata["width"], metadata["height"])
    return runner.PreparedCase(
        dataset=dataset,
        participant_id=runner.normalize_id(clean["participant_id"].iloc[0]),
        case_segment=str(clean["case_segment"].iloc[0]),
        corruption=corruption,
        severity=severity,
        seed=seed,
        fs_hz=fs_hz,
        width=metadata["width"],
        height=metadata["height"],
        diag=diagnostics,
    )


def parse_args() -> argparse.Namespace:
    root = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(description="Auxiliary shifted-generator robustness check.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=root / "results" / "tables" / "participant_split.csv")
    parser.add_argument("--output", type=Path, default=root / "results" / "hybrid" / "shifted")
    parser.add_argument("--seeds", type=int, default=2, help="Number of shifted evaluation seeds; paper uses 2.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference, low_memory=False)
    reference["participant_id"] = reference["participant_id"].map(runner.normalize_id)
    runner.validate_reference(reference)
    split = pd.read_csv(args.split, dtype=str)
    split["participant_id"] = split["participant_id"].map(runner.normalize_id)
    split["split"] = split["split"].str.lower()
    clean = runner.build_clean_cases(reference)
    available = {
        (str(frame["dataset"].iloc[0]), runner.normalize_id(frame["participant_id"].iloc[0]))
        for frame in clean.values()
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
    training = runner.generate_prepared_cases(
        clean,
        outer_train,
        runner.TRAIN_SEEDS,
        runner.SEVERITIES,
        runner.CORRUPTIONS,
        120,
        runner.RANDOM_SEED + 2,
    )
    model = runner.fit_hybrid(training, c_value=1.0, max_samples_per_class=75000, seed=runner.RANDOM_SEED + 3)
    shifted_cases = []
    for frame in clean.values():
        key = (str(frame["dataset"].iloc[0]), runner.normalize_id(frame["participant_id"].iloc[0]))
        if key not in outer_test:
            continue
        for seed in range(args.seeds):
            for severity in runner.SEVERITIES:
                for corruption in runner.CORRUPTIONS:
                    shifted_cases.append(prepare_shifted(frame, corruption, severity, seed))

    sequence = runner.add_local_action_score(runner.evaluate_all_methods(shifted_cases, model, 1.0))
    participant = runner.participant_aggregate(sequence)
    dataset, macro = runner.dataset_and_macro_aggregate(participant)
    bootstrap = pd.concat(
        [
            runner.paired_bootstrap(participant, "STRATUS-H", "Hybrid-P", args.bootstrap, runner.RANDOM_SEED + 110),
            runner.paired_bootstrap(participant, "STRATUS-H", "Pointwise-D", args.bootstrap, runner.RANDOM_SEED + 120),
        ],
        ignore_index=True,
    )
    sequence.to_csv(args.output / "shifted_sequence_results.csv", index=False)
    participant.to_csv(args.output / "shifted_participant_results.csv", index=False)
    dataset.to_csv(args.output / "shifted_dataset_results.csv", index=False)
    macro.to_csv(args.output / "shifted_macro_results.csv", index=False)
    bootstrap.to_csv(args.output / "shifted_bootstrap_contrasts.csv", index=False)
    print(macro[["method", "mae", "rmse", "Q_local"]].sort_values("Q_local", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
