# STRATUS

**Selective Temporal Coupling for Auditable Time-Series Preprocessing**

STRATUS provides an auditable diagnostic–state–action framework for eye-tracking time-series cleaning. It makes a **local decision for each sample**: retain plausible behavior, interpolate a sufficiently short gap, keep a long unsupported gap missing, or repair an unstable finite value. A selective hybrid hidden Markov model (STRATUS-H) adds temporal coupling only among finite states. The matched Hybrid-P control uses the **same features, labels, hard gap constraints, and action policy**, but performs pointwise classification without temporal decoding.

This repository contains the implementation, controlled quality experiments, generator-shift check, **ICPE 2027 runtime benchmark**, verification scripts, and recorded outputs. Raw third-party eye-tracking data are **not** redistributed.

## At a glance

| Experiment | Main question | Where to find it |
| --- | --- | --- |
| Primary held-out quality study | Does selective temporal coupling improve cleaning quality? | [`results/hybrid/primary/`](results/hybrid/primary/) |
| Shifted-generator stress test | Do quality differences persist when the error generator changes? | [`results/hybrid/shifted/`](results/hybrid/shifted/) |
| Score-weight sensitivity | Are rankings dependent on the local-action score weights? | [`results/hybrid/weight_sensitivity/`](results/hybrid/weight_sensitivity/) |
| **Runtime benchmark (RQ4)** | What processing-time overhead does Viterbi coupling add? | **[`runtime_benchmark/`](runtime_benchmark/)** |
| Historical diagnostic and contextual baselines | How do earlier or nonadaptive methods compare? | [`results/tables/`](results/tables/) and [`pointwise-baseline/`](pointwise-baseline/) |

For architecture details, see [`docs/HYBRID_MODEL.md`](docs/HYBRID_MODEL.md). The [workflow illustration](paper/figures/hmm.png) and [method-family illustration](paper/figures/methods2.png) provide visual overviews.

## Methods and operational policy

| Method | Decision mechanism |
| --- | --- |
| **Pointwise-D** | Hand-designed diagnostic potentials, independent assignment |
| **STRATUS-D** | Same diagnostic potentials with fixed Viterbi persistence |
| **STRATUS-BW** | Generative diagonal-Gaussian Baum–Welch HMM with training-only state alignment |
| **Hybrid-P** | Supervised discriminative finite-state model; hard missing-run rules; no transitions |
| **STRATUS-H** | The identical Hybrid-P model **plus learned transitions and Viterbi decoding** within finite blocks |

The public operational states are `Stable`, `Movement`, `LossShort`, `LossLong`, and `Unstable`. `Stable` and `Movement` are preserved; gaps no longer than **0.5 s** are interpolated; longer gaps remain unavailable; unstable finite positions are repaired by a centered seven-sample rolling median. STRATUS-H distinguishes `UnstableImpulse` and `UnstableBurst` internally, both mapped to the public `Unstable` action. Missing-run decisions are deterministic and cannot be overridden by Viterbi decoding.

## Headline quality results

The primary evaluation uses **47 reference windows** from ETDD70 and the Eye-Tracking Autism dataset, a participant-separated **33/14** training/test split, and **2,100** held-out corruption sequences. Results below are equal-weight dataset macro-averages. Lower MAE/RMSE is better; higher study-specific local-action score is better.

| Method | MAE ↓ | RMSE ↓ | Local-action score ↑ |
| --- | ---: | ---: | ---: |
| **STRATUS-H** | 3.598 | **14.615** | **0.920** |
| **Hybrid-P** | 3.825 | 14.821 | 0.914 |
| **Pointwise-D** | **3.566** | 22.703 | 0.882 |
| **STRATUS-D** | 4.849 | 32.456 | 0.841 |
| **STRATUS-BW** | 5.542 | 39.121 | 0.642 |

The isolated **STRATUS-H minus Hybrid-P** quality differences are:

- Local-action score: **+0.0057**, 95% participant-bootstrap interval **[0.0043, 0.0071]**.
- MAE: **−0.227**, interval **[−0.320, −0.148]**.
- RMSE: **−0.206**, interval **[−0.348, −0.063]**.

These temporal gains are **small but consistent** for the fixed participant split. The larger performance gap relative to Pointwise-D is **not** an isolated HMM effect: the two model families also differ in features, supervision, and state constraints. Pointwise-D retains the best dataset-macro MAE. A separate **420-case** generator-shift sensitivity check reduces the broader differences; see [`docs/RESULTS_HYBRID.md`](docs/RESULTS_HYBRID.md) for interpretation and [`docs/METRICS.md`](docs/METRICS.md) for score definitions.

## Runtime benchmark (ICPE 2027)

The runtime extension measures **Hybrid-P versus STRATUS-H on identical inputs**, isolating the processing overhead of adding learned temporal Viterbi coupling to the matched pointwise model. It includes **4,200 individual timed executions (2,100 paired comparisons)** over 14 held-out participants, five corruption types, 2-, 4-, and 8-second windows, and ten repetitions per case/method. Timings include diagnostics, feature computation, prediction/decoding, and policy execution; they exclude loading, synthetic corruption generation, metric computation, training, and tuning.

**Recorded median processing time (milliseconds):**

| Dataset | Window | Hybrid-P | STRATUS-H | H / P |
| --- | ---: | ---: | ---: | ---: |
| ETDD70 | 2 s | 14.95 | 23.81 | 1.59× |
| ETDD70 | 4 s | 17.87 | 34.79 | 1.95× |
| ETDD70 | 8 s | 22.08 | 58.54 | **2.65×** |
| Autism | 2 s | 13.89 | 15.97 | 1.15× |
| Autism | 4 s | 14.17 | 18.14 | 1.28× |
| Autism | 8 s | 14.86 | 22.30 | **1.50×** |

For the 8-second windows, the **median within-input paired differences** are **+36.20 ms** (ETDD70) and **+7.21 ms** (Autism). These are **not** obtained by simply subtracting marginal medians. Shared-model fitting took **3.278 s median** over five runs; training-case preparation was measured separately. Hardware and software details, all measured data, and the exact aggregation protocol are committed at [`runtime_benchmark/README.md`](runtime_benchmark/README.md).

**Scope:** one Windows 11 machine, two eye-tracking datasets, short **offline** sequences. The study does not establish streaming latency, performance on long recordings, or independent sampling-rate effects. The modest quality improvement must be evaluated against the extra processing cost.

### Verify the published runtime data (no external dataset required)

From this project directory:

```bash
python runtime_benchmark/verify_runtime_results.py
```

This checks all 4,200 observations, pair completeness, the reported aggregates, training timings, and recorded hashes. To regenerate the Markdown timing summary from the recorded CSV files:

```bash
python runtime_benchmark/summarize_runtime.py
```

The latter **rewrites** `runtime_benchmark/results/runtime/RUNTIME_RESULTS.md`; it does not rerun the timings. To make fresh timing measurements, an exact local reference CSV and the published source version are required:

```bash
python runtime_benchmark/run_runtime_benchmark.py \
  --reference /path/to/clean_reference_segments.csv --smoke
python runtime_benchmark/run_runtime_benchmark.py \
  --reference /path/to/clean_reference_segments.csv --repeats 10
```

New timing runs are written to separate output directories instead of overwriting the published logs. The reproduction runner validates the reference-file and source-code SHA-256 hashes; see the [benchmark instructions](runtime_benchmark/README.md) for historical environment versions and options.

## Installation

Clone this repository and change into the actual Python project directory:

```bash
git clone https://github.com/eyetracking-data/STRATUS.git
cd STRATUS/stratus-regime-aware-preprocessing
python -m venv .venv
```

Activate the environment on your operating system, then install and test:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev,notebook]'
python -m pytest
```

Supported Python versions are given in [`pyproject.toml`](pyproject.toml) (`>=3.11,<3.14`). The published runtime machine used **Python 3.13.16** and its exact library versions are recorded in [`runtime_benchmark/results/runtime/environment.json`](runtime_benchmark/results/runtime/environment.json). Current compatible package versions may produce different absolute timings.

## Verify or reproduce quality experiments

**Verify the committed quality results** without retraining or raw datasets:

```bash
python scripts/verify_hybrid_results.py
```

A synthetic implementation self-test also needs no private data:

```bash
python scripts/run_hybrid_experiment.py --self-test --output /tmp/stratus-self-test
```

To **rerun** the matched quality experiment, first obtain the original third-party datasets and reconstruct the exact reference file according to [`data/README.md`](data/README.md) and [`docs/REFERENCE_DATA.md`](docs/REFERENCE_DATA.md). Then supply its local path:

```bash
python scripts/run_hybrid_experiment.py \
  --reference /path/to/clean_reference_segments.csv \
  --output results/reproduced/primary
```

For the shifted-generator sensitivity check:

```bash
python scripts/run_shifted_robustness.py \
  --reference /path/to/clean_reference_segments.csv --seeds 2
```

The quality experiments and runtime benchmark do **not** share the same measurement objective: quality figures come from the held-out corruption experiments; processing times come from a separate fixed-configuration timing study. More information is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository guide

```text
stratus-regime-aware-preprocessing/
├── src/stratus/                 Core implementation and operators
├── scripts/                     Quality experiments and result verification
├── runtime_benchmark/           RQ4 timing code, raw logs, summary and validation
├── results/hybrid/              Primary, shifted, sensitivity and qualitative results
├── results/tables/              Data splits and historical/context-baseline tables
├── results/figures/             Exploratory and comparison plots
├── pointwise-baseline/          Earlier diagnostic/persistence ablation
├── tests/                       Automated tests
├── docs/                        Methods, metrics and reproduction guidance
├── data/                        Instructions and empty raw-data placeholders
├── notebooks/                   Interactive and executed case-study versions
└── paper/                       Historical manuscript mirror (see below)
```

Older result tables, diagnostic ablations, notebooks, and exported figures are retained intentionally for traceability. They should not be confused with the **current primary hybrid** outputs or the **separate runtime benchmark**. No program or results file needs to be removed to make the repository usable.

> **Manuscript version warning:** The tracked `paper/` directory still contains an **EDBT 2027** manuscript/PDF snapshot. The **ICPE 2027** revision with the added runtime research question is maintained separately until that manuscript mirror is updated. Do not cite the old PDF as the current version.

## Data, validity, and citation

ETDD70 and the Eye-Tracking Autism dataset belong to their original providers. **Neither raw recordings nor extracted coordinate-level reference segments are publicly redistributed here.** The exact extraction SHA-256 and split metadata are documented in [`docs/REFERENCE_DATA.md`](docs/REFERENCE_DATA.md) and [`results/tables/`](results/tables/); see [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md) for license boundaries.

The experimental operational labels are synthetic; the `Movement` state is a displacement proxy rather than a clinician-annotated oculomotor category. The main study uses a single outer participant split; the shift experiment is a synthetic sensitivity analysis. Processing is offline because complete gap duration and centered features can use future observations.

The implementation is licensed under [MIT](LICENSE); source-dataset terms apply separately. Citation metadata are in [`CITATION.cff`](CITATION.cff). This repository is public and author-identifying; a double-anonymous conference review requires a separate anonymized artifact access route.
