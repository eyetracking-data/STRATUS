#!/usr/bin/env python3
"""Reproduce the STRATUS-D no-persistence (pointwise) ablation.

This add-on intentionally leaves the existing STRATUS implementation and
notebook unchanged.  It uses the same diagnostic potentials and action policy
for both variants:

* Pointwise-D: independent row-wise argmax of the diagnostic potentials.
* STRATUS-D: Viterbi decoding of the same potentials with the published
  transition terms.

Only the temporal decoder differs.  Data loading, reference extraction,
participant splitting, corruptions, evaluation seeds, metrics, and aggregation
match the main case-study notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stratus.corruptions import inject_corruption
from stratus.diagnostics import compute_diagnostics
from stratus.loaders import (
    estimate_sampling_rate_hz,
    estimate_segment_sampling_rate_hz,
    extract_valid_segments,
    infer_coordinate_bounds,
    load_dataset,
)
from stratus.metrics import evaluate_output, regime_f1
from stratus.plotting import save_grouped_bar, save_tradeoff_scatter
from stratus.regimes import (
    _initial_state_scores,
    apply_state_to_action,
    viterbi_decode,
)


METHODS = ["Pointwise-D", "STRATUS-D"]
EVAL_SEEDS = list(range(10))
SEVERITIES = ["low", "medium", "high"]
CORRUPTIONS = ["short_gap", "long_gap", "jitter", "spike", "mixed"]

N_FILES_PER_DATASET = 10
SEGMENT_SECONDS = 8.0
MAX_SEGMENTS_PER_PARTICIPANT = 1
MIN_VALID_FRACTION = 0.98
TEST_FRACTION = 0.30
SPLIT_SEED = 42

COMPONENT_COLUMNS = [
    "long_gap_conservatism",
    "short_recovery",
    "movement_preservation",
    "stable_preservation",
    "unstable_repair_score",
]
METRIC_COLUMNS = [
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


def aggregate_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    table = frame.groupby(group_columns, as_index=False).agg(
        **{column: (column, "mean") for column in METRIC_COLUMNS}
    )
    table["long_gap_conservatism"] = 1.0 - table["long_hallucination"]
    table["local_action_score"] = table[COMPONENT_COLUMNS].mean(
        axis=1, skipna=False
    )
    return table


def hierarchical_bootstrap_ci(
    participant_table: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Match the participant-within-dataset bootstrap used by the main study."""
    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, float | int | str]] = []
    methods = participant_table["method"].unique()
    datasets = participant_table["dataset"].unique()
    value_columns = METRIC_COLUMNS + ["long_gap_conservatism"]

    for method in methods:
        method_data = participant_table[participant_table["method"] == method]
        draws = {
            column: [] for column in value_columns + ["local_action_score"]
        }

        for _ in range(n_boot):
            dataset_means = []
            for dataset in datasets:
                part = method_data[method_data["dataset"] == dataset]
                sampled_positions = rng.integers(0, len(part), size=len(part))
                dataset_means.append(part.iloc[sampled_positions][value_columns].mean())

            macro = pd.DataFrame(dataset_means).mean()
            for column in value_columns:
                draws[column].append(float(macro[column]))
            draws["local_action_score"].append(
                float(macro[COMPONENT_COLUMNS].mean(skipna=False))
            )

        row: dict[str, float | int | str] = {
            "method": method,
            "bootstrap_replicates": n_boot,
        }
        for column, values in draws.items():
            row[f"{column}_ci_low"] = float(np.nanquantile(values, 0.025))
            row[f"{column}_ci_high"] = float(np.nanquantile(values, 0.975))
        bootstrap_rows.append(row)

    return pd.DataFrame(bootstrap_rows)


def paired_difference_ci(
    participant_table: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Bootstrap Pointwise-D minus STRATUS-D with paired participant draws."""
    metrics = METRIC_COLUMNS + [
        "long_gap_conservatism",
        "local_action_score",
    ]
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in metrics}

    dataset_wide = {}
    for dataset, group in participant_table.groupby("dataset", sort=False):
        dataset_wide[dataset] = group.pivot(
            index="participant_id",
            columns="method",
            values=metrics,
        )

    for _ in range(n_boot):
        dataset_differences = []
        for wide in dataset_wide.values():
            sampled_positions = rng.integers(0, len(wide), size=len(wide))
            sampled = wide.iloc[sampled_positions]
            dataset_differences.append(
                pd.Series(
                    {
                        metric: (
                            sampled[(metric, "Pointwise-D")].mean()
                            - sampled[(metric, "STRATUS-D")].mean()
                        )
                        for metric in metrics
                    }
                )
            )
        macro_difference = pd.DataFrame(dataset_differences).mean()
        for metric in metrics:
            draws[metric].append(float(macro_difference[metric]))

    pointwise = (
        participant_table[participant_table["method"] == "Pointwise-D"]
        .groupby("dataset")[metrics]
        .mean()
        .mean()
    )
    persistent = (
        participant_table[participant_table["method"] == "STRATUS-D"]
        .groupby("dataset")[metrics]
        .mean()
        .mean()
    )
    point_difference = pointwise - persistent

    return pd.DataFrame(
        [
            {
                "contrast": "Pointwise-D minus STRATUS-D",
                "metric": metric,
                "difference": float(point_difference[metric]),
                "ci_low": float(np.nanquantile(values, 0.025)),
                "ci_high": float(np.nanquantile(values, 0.975)),
                "bootstrap_replicates": n_boot,
            }
            for metric, values in draws.items()
        ]
    )


def prepare_reference_data(
    etdd70_dir: Path,
    autism_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[float, float]],
]:
    """Load the canonical inputs and recreate the published participant split."""
    raw_by_dataset = {
        "ETDD70": load_dataset(
            etdd70_dir, dataset="dyslexia", n_files=N_FILES_PER_DATASET
        ),
        "Autism": load_dataset(
            autism_dir, dataset="autism", n_files=N_FILES_PER_DATASET
        ),
    }
    bounds_by_dataset = {
        name: infer_coordinate_bounds(frame)
        for name, frame in raw_by_dataset.items()
    }

    metadata_rows = []
    segments = []
    extraction_rows = []
    for name, frame in raw_by_dataset.items():
        width, height = bounds_by_dataset[name]
        metadata_rows.append(
            {
                "dataset": name,
                "rows": len(frame),
                "files": frame["source_file"].nunique(),
                "participants": frame["participant_id"].nunique(),
                "sampling_rate_hz": estimate_sampling_rate_hz(frame),
                "coordinate_width": width,
                "coordinate_height": height,
            }
        )
        selected = extract_valid_segments(
            frame,
            segment_seconds=SEGMENT_SECONDS,
            max_segments_per_participant=MAX_SEGMENTS_PER_PARTICIPANT,
            min_valid_fraction=MIN_VALID_FRACTION,
            selection_seed=SPLIT_SEED,
        )
        segments.append(selected)
        extraction_rows.append(
            {
                "dataset": name,
                "loaded_files": frame["source_file"].nunique(),
                "available_participants": frame["participant_id"].nunique(),
                "usable_participants": selected["participant_id"].nunique(),
                "reference_segments": selected["case_segment"].nunique(),
                "median_segment_seconds": selected.groupby("case_segment")[
                    "time_s"
                ]
                .agg(lambda series: series.max() - series.min())
                .median(),
            }
        )

    clean_segments = pd.concat(segments, ignore_index=True)
    test_parts = []
    split_rows = []
    for dataset, group in clean_segments.groupby("dataset", sort=False):
        participants = np.array(sorted(group["participant_id"].astype(str).unique()))
        local_rng = np.random.default_rng(SPLIT_SEED)
        local_rng.shuffle(participants)
        n_test = max(1, int(round(len(participants) * TEST_FRACTION)))
        if len(participants) > 1:
            n_test = min(n_test, len(participants) - 1)
        test_ids = set(participants[:n_test])
        train_ids = set(participants[n_test:])
        test_parts.append(
            group[group["participant_id"].astype(str).isin(test_ids)].copy()
        )
        split_rows.extend(
            {
                "dataset": dataset,
                "participant_id": participant,
                "split": split,
            }
            for split, ids in (("train", train_ids), ("test", test_ids))
            for participant in sorted(ids)
        )

    test_segments = pd.concat(test_parts, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows)
    extraction = pd.DataFrame(extraction_rows)
    split_table = pd.DataFrame(split_rows)
    return (
        test_segments,
        metadata,
        extraction,
        split_table,
        bounds_by_dataset,
    )


def evaluate_ablation(
    test_segments: pd.DataFrame,
    bounds_by_dataset: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the two decoders while sharing diagnostics for every case."""
    rows = []
    f1_rows = []

    for case_id, clean in test_segments.groupby("case_segment", sort=False):
        clean = clean.sort_values("time_s").reset_index(drop=True)
        dataset = clean["dataset"].iloc[0]
        width, height = bounds_by_dataset[dataset]

        for seed in EVAL_SEEDS:
            for severity in SEVERITIES:
                for corruption in CORRUPTIONS:
                    degraded = inject_corruption(
                        clean,
                        corruption=corruption,
                        severity=severity,
                        seed=seed,
                    )
                    fs_hz = estimate_segment_sampling_rate_hz(degraded)
                    if not np.isfinite(fs_hz):
                        raise ValueError(
                            f"Could not estimate sampling rate for {case_id}"
                        )

                    diagnostics = compute_diagnostics(
                        degraded,
                        screen_width=width,
                        screen_height=height,
                    )
                    potentials = _initial_state_scores(
                        diagnostics,
                        fs_hz=fs_hz,
                        short_gap_seconds=0.50,
                        velocity_quantile=0.90,
                    )
                    paths = {
                        "Pointwise-D": potentials.idxmax(axis=1).astype("object"),
                        "STRATUS-D": viterbi_decode(
                            potentials,
                            stay_bonus=1.25,
                            switch_penalty=-0.75,
                        ),
                    }

                    for method, regimes in paths.items():
                        output = apply_state_to_action(
                            diagnostics.copy(),
                            regimes,
                            stable_smoothing=False,
                        )
                        metrics = evaluate_output(
                            output,
                            screen_width=width,
                            screen_height=height,
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "case_segment": case_id,
                                "participant_id": clean["participant_id"].iloc[0],
                                "seed": seed,
                                "severity": severity,
                                "corruption": corruption,
                                "method": method,
                                **metrics,
                            }
                        )

                        for _, item in regime_f1(output).iterrows():
                            f1_rows.append(
                                {
                                    "dataset": dataset,
                                    "case_segment": case_id,
                                    "participant_id": clean[
                                        "participant_id"
                                    ].iloc[0],
                                    "seed": seed,
                                    "severity": severity,
                                    "corruption": corruption,
                                    "method": method,
                                    "regime": item["regime"],
                                    "f1": item["f1"],
                                    "support": item["support"],
                                }
                            )

    return pd.DataFrame(rows), pd.DataFrame(f1_rows)


def compare_with_committed_stratus_d(
    macro_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Report numerical agreement with the committed v7 STRATUS-D tables."""
    comparison_rows = []
    table_specs = [
        (
            "macro",
            macro_summary,
            REPO_ROOT / "results" / "tables" / "macro_results.csv",
            [],
        ),
        (
            "dataset",
            dataset_summary,
            REPO_ROOT / "results" / "tables" / "dataset_results.csv",
            ["dataset"],
        ),
    ]

    for scope, rerun, committed_path, key_columns in table_specs:
        if not committed_path.exists():
            continue
        committed = pd.read_csv(committed_path)
        rerun_d = rerun[rerun["method"] == "STRATUS-D"].copy()
        committed_d = committed[committed["method"] == "STRATUS-D"].copy()
        merged = rerun_d.merge(
            committed_d,
            on=key_columns + ["method"],
            suffixes=("_rerun", "_committed"),
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            key = (
                "all"
                if not key_columns
                else "|".join(str(row[column]) for column in key_columns)
            )
            for metric in METRIC_COLUMNS + [
                "long_gap_conservatism",
                "local_action_score",
            ]:
                rerun_value = float(row[f"{metric}_rerun"])
                committed_value = float(row[f"{metric}_committed"])
                comparison_rows.append(
                    {
                        "scope": scope,
                        "key": key,
                        "metric": metric,
                        "rerun_value": rerun_value,
                        "committed_value": committed_value,
                        "absolute_difference": abs(rerun_value - committed_value),
                    }
                )

    return pd.DataFrame(comparison_rows)


def run_experiment(
    etdd70_dir: Path,
    autism_dir: Path,
    output_dir: Path,
    bootstrap_replicates: int = 2000,
) -> dict[str, pd.DataFrame]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    intermediate_dir = output_dir / "intermediate"
    for folder in (tables_dir, figures_dir, intermediate_dir):
        folder.mkdir(parents=True, exist_ok=True)

    (
        test_segments,
        metadata,
        extraction,
        split_table,
        bounds_by_dataset,
    ) = prepare_reference_data(etdd70_dir, autism_dir)
    results, f1_results = evaluate_ablation(test_segments, bounds_by_dataset)

    dataset_summary = aggregate_metrics(results, ["dataset", "method"]).sort_values(
        ["dataset", "local_action_score"], ascending=[True, False]
    )
    macro_summary = (
        dataset_summary.groupby("method", as_index=False)
        .agg(
            **{
                column: (column, "mean")
                for column in METRIC_COLUMNS + ["long_gap_conservatism"]
            }
        )
    )
    macro_summary["local_action_score"] = macro_summary[COMPONENT_COLUMNS].mean(
        axis=1, skipna=False
    )
    macro_summary = macro_summary.sort_values(
        "local_action_score", ascending=False
    )

    participant_summary = aggregate_metrics(
        results, ["dataset", "participant_id", "method"]
    )
    bootstrap_ci = hierarchical_bootstrap_ci(
        participant_summary,
        n_boot=bootstrap_replicates,
        seed=SPLIT_SEED,
    )
    macro_with_ci = macro_summary.merge(bootstrap_ci, on="method", how="left")
    paired_differences = paired_difference_ci(
        participant_summary,
        n_boot=bootstrap_replicates,
        seed=SPLIT_SEED,
    )
    severity_summary = aggregate_metrics(results, ["severity", "method"])

    regime_f1_summary = (
        f1_results.groupby(["method", "regime"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            evaluated_cases=("f1", "count"),
            total_support=("support", "sum"),
        )
        .sort_values(["method", "mean_f1"], ascending=[True, False])
    )
    verification = compare_with_committed_stratus_d(
        macro_summary, dataset_summary
    )

    metadata.to_csv(tables_dir / "dataset_metadata.csv", index=False)
    extraction.to_csv(tables_dir / "reference_extraction_summary.csv", index=False)
    dataset_summary.to_csv(
        tables_dir / "pointwise_dataset_results.csv", index=False
    )
    macro_summary.to_csv(tables_dir / "pointwise_macro_results.csv", index=False)
    macro_with_ci.to_csv(
        tables_dir / "pointwise_macro_results_with_ci.csv", index=False
    )
    paired_differences.to_csv(
        tables_dir / "pointwise_paired_differences.csv", index=False
    )
    severity_summary.to_csv(
        tables_dir / "pointwise_severity_results.csv", index=False
    )
    regime_f1_summary.to_csv(
        tables_dir / "pointwise_regime_f1_summary.csv", index=False
    )
    verification.to_csv(
        tables_dir / "stratus_d_reproduction_check.csv", index=False
    )

    results.to_csv(intermediate_dir / "ablation_case_results.csv", index=False)
    f1_results.to_csv(
        intermediate_dir / "ablation_regime_f1_results.csv", index=False
    )
    participant_summary.to_csv(
        intermediate_dir / "ablation_participant_results.csv", index=False
    )
    split_table.to_csv(intermediate_dir / "participant_split.csv", index=False)

    save_tradeoff_scatter(
        macro_summary,
        x="mae",
        y="local_action_score",
        label="method",
        title="Effect of temporal persistence",
        path=figures_dir / "pointwise_persistence_tradeoff.pdf",
        xlabel="Dataset-macro MAE",
        ylabel="Local action score",
    )
    save_grouped_bar(
        regime_f1_summary,
        index="regime",
        columns="method",
        values="mean_f1",
        title="Regime inference with and without persistence",
        path=figures_dir / "pointwise_regime_f1.pdf",
        ylabel="Mean F1",
    )

    manifest = {
        "comparison": {
            "Pointwise-D": (
                "row-wise argmax of STRATUS-D diagnostic potentials"
            ),
            "STRATUS-D": (
                "Viterbi path over the same potentials; stay_bonus=1.25, "
                "switch_penalty=-0.75"
            ),
        },
        "shared_action_policy": True,
        "n_files_per_dataset": N_FILES_PER_DATASET,
        "segment_seconds": SEGMENT_SECONDS,
        "max_segments_per_participant": MAX_SEGMENTS_PER_PARTICIPANT,
        "minimum_valid_fraction": MIN_VALID_FRACTION,
        "test_fraction": TEST_FRACTION,
        "split_seed": SPLIT_SEED,
        "evaluation_seeds": EVAL_SEEDS,
        "severities": SEVERITIES,
        "corruptions": CORRUPTIONS,
        "bootstrap_replicates": bootstrap_replicates,
        "loaded_files": {
            "ETDD70": sorted(
                path.name for path in Path(etdd70_dir).glob("*.csv")
            )[:N_FILES_PER_DATASET],
            "Autism": sorted(
                path.name for path in Path(autism_dir).glob("*.csv")
            )[:N_FILES_PER_DATASET],
        },
    }
    with (tables_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    if not verification.empty:
        maximum_difference = verification["absolute_difference"].max()
        if maximum_difference > 1e-12:
            raise AssertionError(
                "The STRATUS-D rerun differs from the committed v7 table: "
                f"max absolute difference={maximum_difference:.3e}"
            )

    return {
        "metadata": metadata,
        "extraction": extraction,
        "dataset_summary": dataset_summary,
        "macro_summary": macro_summary,
        "macro_with_ci": macro_with_ci,
        "paired_differences": paired_differences,
        "severity_summary": severity_summary,
        "regime_f1_summary": regime_f1_summary,
        "verification": verification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--etdd70-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "STRATUS_ETDD70_DIR", REPO_ROOT / "data" / "raw" / "etdd70"
            )
        ),
    )
    parser.add_argument(
        "--autism-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "STRATUS_AUTISM_DIR", REPO_ROOT / "data" / "raw" / "autism"
            )
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = run_experiment(
        etdd70_dir=args.etdd70_dir,
        autism_dir=args.autism_dir,
        output_dir=args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print("\nDataset metadata")
    print(summaries["metadata"].to_string(index=False))
    print("\nPointwise ablation (dataset macro)")
    print(
        summaries["macro_with_ci"][
            [
                "method",
                "mae",
                "mae_ci_low",
                "mae_ci_high",
                "local_action_score",
                "local_action_score_ci_low",
                "local_action_score_ci_high",
            ]
        ].to_string(index=False)
    )
    print(
        "\nOutputs:",
        Path(args.output_dir).resolve(),
    )


if __name__ == "__main__":
    main()
