"""Recreate the qualitative STRATUS-H versus Hybrid-P paper figure.

The clean reference CSV is intentionally not committed because it contains
coordinate-level windows from third-party datasets. Point --reference to the
v7 clean_reference_segments.csv produced by the extraction notebook.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).with_name("run_hybrid_experiment.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("stratus_hybrid_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def q_local(metrics: dict[str, float]) -> float:
    values = [
        1.0 - metrics["long_hallucination"],
        metrics["short_recovery"],
        metrics["movement_preservation"],
        metrics["stable_preservation"],
        metrics["unstable_repair_score"],
    ]
    return float(np.nanmean(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "participant_split.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper" / "figures" / "figure2_qualitative_hybrid.pdf",
    )
    parser.add_argument(
        "--examples-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "hybrid" / "qualitative" / "figure2_examples.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = load_runner()

    reference = pd.read_csv(args.reference, low_memory=False)
    reference["participant_id"] = reference["participant_id"].map(runner.normalize_id)
    runner.validate_reference(reference)
    split = pd.read_csv(args.split, dtype=str)
    split["participant_id"] = split["participant_id"].map(runner.normalize_id)
    split["dataset"] = split["dataset"].astype(str)
    split["split"] = split["split"].astype(str).str.lower()

    clean_cases = runner.build_clean_cases(reference)
    available = {
        (str(frame["dataset"].iloc[0]), runner.normalize_id(frame["participant_id"].iloc[0]))
        for frame in clean_cases.values()
    }
    outer_train = {
        (row.dataset, row.participant_id)
        for row in split.itertuples(index=False)
        if row.split == "train" and (row.dataset, row.participant_id) in available
    }

    training_cases = runner.generate_prepared_cases(
        clean_cases,
        outer_train,
        runner.TRAIN_SEEDS,
        runner.SEVERITIES,
        runner.CORRUPTIONS,
        cap_per_dataset=120,
        selection_seed=runner.RANDOM_SEED + 2,
    )
    model = runner.fit_hybrid(
        training_cases,
        c_value=1.0,
        max_samples_per_class=75000,
        seed=runner.RANDOM_SEED + 3,
    )

    choices = [
        {
            "dataset": "ETDD70",
            "participant": "1003",
            "segment": "ETDD70__1003__seg0",
            "corruption": "mixed",
            "severity": "medium",
            "seed": 5,
            "panel": "(a) ETDD70: mixed corruption",
        },
        {
            "dataset": "Autism",
            "participant": "37",
            "segment": "Autism__37__seg18",
            "corruption": "mixed",
            "severity": "medium",
            "seed": 2,
            "panel": "(b) Autism: ambiguous unstable region",
        },
    ]
    missing_segments = [choice["segment"] for choice in choices if choice["segment"] not in clean_cases]
    if missing_segments:
        raise ValueError(f"Reference file does not contain paper examples: {missing_segments}")

    state_order = ["Stable", "Movement", "LossShort", "LossLong", "Unstable"]
    state_to_int = {state: index for index, state in enumerate(state_order)}
    state_colors = ["#d9d9d9", "#4c78a8", "#f2cf5b", "#e07b39", "#b54a65"]
    cmap = ListedColormap(state_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(state_order) + 0.5), cmap.N)

    figure = plt.figure(figsize=(13.2, 6.1), constrained_layout=False)
    grid = figure.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[7.6, 0.58, 0.58, 0.58],
        hspace=0.11,
        wspace=0.16,
        left=0.065,
        right=0.985,
        top=0.86,
        bottom=0.12,
    )
    line_handles = None
    summaries: list[dict[str, object]] = []

    for column, choice in enumerate(choices):
        case = runner.prepare_case(
            clean_cases[choice["segment"]],
            choice["corruption"],
            choice["severity"],
            choice["seed"],
        )
        pointwise_indices = model.predict(case.diag, case.fs_hz, 0.0)
        hmm_indices = model.predict(case.diag, case.fs_hz, 1.0)
        pointwise_output = runner.apply_policy(case.diag, pointwise_indices, expanded=True)
        hmm_output = runner.apply_policy(case.diag, hmm_indices, expanded=True)
        pointwise_metrics = runner.evaluate_output(pointwise_output, case.width, case.height)
        hmm_metrics = runner.evaluate_output(hmm_output, case.width, case.height)
        pointwise_q = q_local(pointwise_metrics)
        hmm_q = q_local(hmm_metrics)
        summaries.append(
            {
                **choice,
                "hybrid_p_Q_local": pointwise_q,
                "stratus_h_Q_local": hmm_q,
                "hybrid_p_mae": pointwise_metrics["mae"],
                "stratus_h_mae": hmm_metrics["mae"],
                "state_disagreement_fraction": float(np.mean(pointwise_indices != hmm_indices)),
            }
        )

        time = case.diag["time_s"].to_numpy(float)
        time = time - time[0]
        axis = figure.add_subplot(grid[0, column])
        degraded_line, = axis.plot(
            time, case.diag["x_degraded"], color="#9a9a9a", linewidth=1.0,
            linestyle="--", alpha=0.8, label="Degraded input"
        )
        pointwise_line, = axis.plot(
            time, pointwise_output["x"], color="#2878b5", linewidth=1.35,
            alpha=0.9, label="Hybrid-P output"
        )
        hmm_line, = axis.plot(
            time, hmm_output["x"], color="#c66a16", linewidth=1.55,
            alpha=0.95, label="STRATUS-H output"
        )
        clean_line, = axis.plot(
            time, case.diag["x_clean"], color="#222222", linewidth=1.05,
            alpha=0.9, label="Clean reference"
        )
        if line_handles is None:
            line_handles = [clean_line, degraded_line, pointwise_line, hmm_line]
        axis.set_title(str(choice["panel"]), fontsize=10.5, pad=5, fontweight="semibold")
        axis.text(
            0.01,
            0.98,
            rf"$Q_{{local}}$: {pointwise_q:.3f} $\rightarrow$ {hmm_q:.3f}    "
            rf"MAE: {pointwise_metrics['mae']:.2f} $\rightarrow$ {hmm_metrics['mae']:.2f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8.1,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
        )
        axis.set_ylabel("Gaze x [px]", fontsize=9)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.7)
        axis.tick_params(labelsize=8)
        axis.set_xlim(time[0], time[-1])
        axis.set_xticklabels([])

        true_names = case.diag["true_regime"].astype(str).to_numpy()
        pointwise_names = np.asarray(
            [runner.EXPANDED_STATES[int(index)] for index in pointwise_indices], dtype=object
        )
        hmm_names = np.asarray(
            [runner.EXPANDED_STATES[int(index)] for index in hmm_indices], dtype=object
        )
        pointwise_names = np.where(
            np.char.startswith(pointwise_names.astype(str), "Unstable"), "Unstable", pointwise_names
        )
        hmm_names = np.where(
            np.char.startswith(hmm_names.astype(str), "Unstable"), "Unstable", hmm_names
        )

        for row, (label, names) in enumerate(
            [("True", true_names), ("Hybrid-P", pointwise_names), ("STRATUS-H", hmm_names)],
            start=1,
        ):
            strip = figure.add_subplot(grid[row, column], sharex=axis)
            values = np.asarray([state_to_int.get(str(value), 0) for value in names], dtype=float)[None, :]
            strip.imshow(
                values,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
                extent=[time[0], time[-1], 0, 1],
                origin="lower",
            )
            strip.set_yticks([0.5])
            strip.set_yticklabels([label], fontsize=7.7)
            strip.tick_params(axis="y", length=0, pad=3)
            for spine in strip.spines.values():
                spine.set_linewidth(0.45)
                spine.set_edgecolor("#888888")
            if row < 3:
                strip.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                strip.set_xlabel("Time [s]", fontsize=9)
                strip.tick_params(axis="x", labelsize=8)

    figure.legend(
        handles=line_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9.2,
        bbox_to_anchor=(0.5, 0.985),
        handlelength=3.0,
        columnspacing=1.8,
    )
    state_handles = [
        Patch(facecolor=color, edgecolor="#777777", linewidth=0.4, label=state)
        for state, color in zip(state_order, state_colors)
    ]
    figure.legend(
        handles=state_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.012),
        handlelength=1.2,
        columnspacing=1.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.examples_csv.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(summaries).to_csv(args.examples_csv, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.examples_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
