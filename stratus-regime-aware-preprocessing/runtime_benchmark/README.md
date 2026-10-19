# STRATUS runtime benchmark (ICPE 2027)

This **additive** artifact contains the measured end-to-end processing-time contrast between **Hybrid-P** (matched pointwise control) and **STRATUS-H** (the same model with temporal Viterbi coupling). The benchmark reuses the unchanged `scripts/run_hybrid_experiment.py` in the parent STRATUS project and does not refit the quality comparisons.

## Folder location

`eyetracking-data/STRATUS/stratus-regime-aware-preprocessing/runtime_benchmark/`

All the files in this folder are new additions, not replacements for the original STRATUS repository or its paper.

## Published measurements

- 14 held-out participants: ETDD70 (3), Autism (11).
- 2-, 4-, and 8-second excerpts, five corruption types at medium severity, ten repetitions per case and method, in alternating order.
- 4,200 timed method executions = 2,100 paired comparisons.
- 240 training cases; five timed shared-model fits. Training case preparation measured separately.
- Wall-clock processing includes diagnostics, feature calculation, prediction and policy application; excludes file I/O, corruption generation, quality evaluation, hyperparameter search and model training.

| Dataset | Duration | Hybrid-P median (ms) | STRATUS-H median (ms) |
|---|---:|---:|---:|
| ETDD70 | 2 s | 14.9473 | 23.8140 |
| ETDD70 | 4 s | 17.8735 | 34.7900 |
| ETDD70 | 8 s | 22.0792 | 58.5388 |
| Autism | 2 s | 13.8914 | 15.9718 |
| Autism | 4 s | 14.1735 | 18.1382 |
| Autism | 8 s | 14.8571 | 22.3005 |

For eight-second windows the **median paired differences** (STRATUS-H minus Hybrid-P, matched by excerpt/corruption/repeat) are +36.20 ms on ETDD70 and +7.21 ms on Autism. The medians in each method column are computed separately: subtracting those two marginal medians does not generally equal the median of the paired differences. Shared-model training: median 3.278 s over five fits; training-case preparation 4.378 s.

**Interpretation:** temporal coupling introduces measurable processing overhead in this implementation. This is an offline single-machine benchmark of short windows, not a streaming latency guarantee or a scaling study. The quality scores are documented elsewhere in the STRATUS repository and were not recomputed by this timing run.

## Recompute the published statistics, without private data

From the project directory `stratus-regime-aware-preprocessing`:

```bash
python runtime_benchmark/verify_runtime_results.py
python runtime_benchmark/summarize_runtime.py
```

`verify_runtime_results.py` checks all individual timing records, pair completeness, 12 aggregate rows, participant split/count, the eight-second paired differences and shared training measurements. The summary script writes `results/runtime/RUNTIME_RESULTS.md` **inside this runtime_benchmark directory** and does not modify the main STRATUS experiments.

## Repeat timing measurements locally (third-party input required)

Create an environment as described in the parent `README.md` (the published historical environment is recorded in `results/runtime/environment.json`):

```bash
python -m pip install -e .
python runtime_benchmark/run_runtime_benchmark.py --reference /path/to/clean_reference_segments.csv --smoke
python runtime_benchmark/run_runtime_benchmark.py --reference /path/to/clean_reference_segments.csv --repeats 10
```

Windows users should replace `/path/to/...` with the actual path to the local CSV. The original benchmark used Python 3.13.16, NumPy 2.5.4, pandas 3.0.6 and scikit-learn 1.9.1 on Windows 11. Install `threadpoolctl` if it is absent; scikit-learn normally installs it. Set up the reference file using the main project's dataset instructions (`data/README.md` and `THIRD_PARTY_DATA.md`).

The runner **requires** the reference input to match SHA-256 `c4e4b81c948e28c80e0ba50c84e118e54081d5257cb3c5b7a2a9d1e85e467525` and the repository runner to match SHA-256 `414958e7966fdf206f97cb3a9222c354703d0c28a3ac4e3dd258692a2dc50f0c`. These safeguards prevent inadvertently labeling a different run as an exact replication of the published protocol. To test a later code version, retain a versioned copy of the published setup and document changed hashes.

New measurements are written into **separate directories** (`results/runtime_smoke/` or `results/runtime_reproduced/`), leaving the original `results/runtime/` files intact. To choose your own output directory, pass `--output <path>`. Run `summarize_runtime.py --results <path>` on the resulting full measurements.

## Files

- `run_runtime_benchmark.py`: reproduced measurement design, adapted to import the existing repository's original source rather than a copied nested ZIP.
- `summarize_runtime.py`: recalculates the paired eight-second summary and environment description.
- `verify_runtime_results.py`: independent standard-library validation (no new timing run).
- `results/runtime/timings.csv`: all 4,200 timed executions.
- `results/runtime/summary.csv`: twelve method/dataset/duration aggregate rows.
- `results/runtime/training.csv`: five shared training repetitions.
- `results/runtime/environment.json`: recorded hardware/software and measurement protocol. Absolute, user-specific local library paths have been replaced by DLL basenames; scientific timings and all recorded numerical settings are unchanged.
- `data/README.md`: third-party data policy and precise reference-file hash.

## Reproducibility and data sharing boundaries

- The runner's original source is part of the parent repository and was verified byte-for-byte against the source used for the benchmark at package construction time.
- Historical timings are measured, not synthetic; verification checks aggregates but cannot reproduce the actual wall-clock conditions on a different machine.
- Third-party raw data and derived coordinate-level reference windows are **not** in this public upload. Dataset redistribution permission must be reviewed independently before publishing those records.
- Note that publishing the repository as-is may reveal the authors' identity; if a particular venue requires double-anonymous artifact access, use an appropriately anonymized submission mechanism during review.
