# STRATUS-D pointwise (no-persistence) ablation

This additive experiment isolates the effect of temporal persistence in
STRATUS-D. It requires no replacement of existing repository files.

Both compared variants use the same six diagnostics, diagnostic potentials,
thresholds, and state-to-action policy. `Pointwise-D` selects the highest
diagnostic potential independently at every sample. `STRATUS-D` applies the
existing Viterbi decoder to those same potentials (`stay_bonus=1.25`,
`switch_penalty=-0.75`). Consequently, the only experimental factor is
temporal persistence.

## Data

Use the same public merged-by-subject CSV inputs as the main case study:

- `data/raw/etdd70/`: ETDD70 `Subject_*_combined_raw.csv` files
- `data/raw/autism/`: Autism eye-tracking export CSV files

The script selects the first ten lexicographically sorted CSV files per
dataset, exactly like the main notebook. Alternative locations can be supplied
with `STRATUS_ETDD70_DIR` and `STRATUS_AUTISM_DIR` or with command-line flags.
Raw data are not included in this folder.

## Run

From `stratus-regime-aware-preprocessing/`:

```bash
python pointwise-baseline/run_pointwise_ablation.py
```

Or with explicit data paths:

```bash
python pointwise-baseline/run_pointwise_ablation.py \
  --etdd70-dir /path/to/etdd70 \
  --autism-dir /path/to/autism
```

The run uses the main study's participant split (`seed=42`), ten evaluation
seeds, three severities, five corruption types, and 2,000 hierarchical
bootstrap replicates. It writes publication-level aggregate tables and PDF
figures to `pointwise-baseline/results/`. Participant- and case-level
intermediate files are generated locally but excluded by `.gitignore`.
The table `pointwise_paired_differences.csv` reports participant-paired,
dataset-macro bootstrap intervals for `Pointwise-D` minus `STRATUS-D`.

`stratus_d_reproduction_check.csv` compares the STRATUS-D rerun with the
committed diagnostic-baseline result tables and the script fails if any reported metric differs
by more than `1e-12`.

## Add to GitHub

Upload this complete `pointwise-baseline/` folder. No existing notebook,
source module, result, or paper file has to be replaced.
