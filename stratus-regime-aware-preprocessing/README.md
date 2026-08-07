# STRATUS

**Auditable local operator selection with selective temporal coupling**

STRATUS treats time-series preprocessing as a sequence of explicit local decisions rather than one uniform global pipeline. The artifact separates the contribution of the operational representation from the narrower contribution of temporal coupling and provides a matched pointwise control for that comparison.

## Model variants

- **Pointwise-D:** interpretable diagnostic potentials, independently maximized.
- **STRATUS-D:** the same diagnostic potentials plus fixed Viterbi persistence.
- **STRATUS-BW:** diagonal-Gaussian Baum--Welch HMM with training-only semantic alignment.
- **Hybrid-P:** hard gap-duration constraints and a supervised discriminative finite-state observation model, without transitions.
- **STRATUS-H:** the identical Hybrid-P model plus a learned transition matrix and Viterbi decoding within finite blocks.
- **Oracle policy:** shared state-to-action policy applied to injected operational labels; analysis upper bound only.

The public operational states are `Stable`, `Movement`, `LossShort`, `LossLong`, and `Unstable`. STRATUS-H internally divides `Unstable` into impulse and burst substates; both map to the same rolling-median action. Complete missing-run duration directly selects `LossShort` or `LossLong`, so the learned finite-state decoder cannot bridge gaps or convert finite samples into loss states.

## Headline primary results

Equal-weight dataset macro-average over held-out ETDD70 and Autism identifiers:

| Method | MAE ↓ | RMSE ↓ | Local action score ↑ |
|---|---:|---:|---:|
| STRATUS-H | 3.598 | **14.615** | **0.920** |
| Hybrid-P | 3.825 | 14.821 | 0.914 |
| Pointwise-D | **3.566** | 22.703 | 0.882 |
| STRATUS-D | 4.849 | 32.456 | 0.841 |
| STRATUS-BW | 5.542 | 39.121 | 0.642 |

The matched temporal contrast is **STRATUS-H minus Hybrid-P**:

- local action score: `+0.0057`, participant-bootstrap 95% interval `[0.0043, 0.0071]`;
- MAE: `-0.227`, interval `[-0.320, -0.148]`;
- RMSE: `-0.206`, interval `[-0.348, -0.063]`.

The temporal effect is small but consistent. The larger gain over Pointwise-D must not be attributed to the HMM alone because state topology, supervision, and observation features also change. Pointwise-D retains a slightly lower primary MAE.

## Repository layout

```text
src/stratus/                         reusable implementation
scripts/run_hybrid_experiment.py     primary participant-separated experiment
scripts/run_shifted_robustness.py    shifted-generator sensitivity check
scripts/make_qualitative_figure.py   recreate the held-out qualitative comparison
scripts/verify_hybrid_results.py     deterministic verification of committed results
results/hybrid/primary/              primary sequence/participant/dataset/macro outputs
results/hybrid/shifted/              shifted-generator outputs
results/hybrid/weight_sensitivity/   score-weight checks
results/hybrid/qualitative/          exact qualitative-example specification
results/tables/                      participant split and extraction metadata
notebooks/                           case-study workflow including retained context baselines
paper/                               manuscript mirror for convenience; not required to run the artifact
```

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[dev,notebook]'
pytest
```

## Data

Raw third-party datasets and coordinate-level reference windows are not redistributed. Obtain ETDD70 and the Eye-Tracking Autism dataset from their original sources and follow `data/README.md`. The fixed reference extraction used for the reported experiments has SHA-256:

```text
c4e4b81c948e28c80e0ba50c84e118e54081d5257cb3c5b7a2a9d1e85e467525
```

The participant split and extraction summary required to check the reported setup are committed under `results/tables/`. Additional details are in `docs/REFERENCE_DATA.md` and `THIRD_PARTY_DATA.md`.

## Reproduce the decisive matched comparison

```bash
python scripts/run_hybrid_experiment.py \
  --reference /path/to/clean_reference_segments.csv \
  --output results/hybrid/primary
```

This runner evaluates Pointwise-D, STRATUS-D, Hybrid-P, and STRATUS-H and writes sequence-, participant-, dataset-, and macro-level outputs together with the selected hyperparameters and paired bootstrap contrasts.

A technical smoke test does not require external data:

```bash
python scripts/run_hybrid_experiment.py --self-test --output /tmp/stratus-self-test
```

The auxiliary generator-shift check uses two seeds in the manuscript:

```bash
python scripts/run_shifted_robustness.py \
  --reference /path/to/clean_reference_segments.csv \
  --seeds 2
```

Recreate the qualitative held-out comparison used in the manuscript:

```bash
python scripts/make_qualitative_figure.py \
  --reference /path/to/clean_reference_segments.csv
```

The script retains the historical artifact filenames `figure2_qualitative_hybrid.pdf` and `figure2_examples.csv`; in the current manuscript this visualization is Figure 4. The examples are illustrative only and are not used for model selection or statistical inference.

## Verify committed results without refitting

```bash
python scripts/verify_hybrid_results.py
```

Add `--reference /path/to/clean_reference_segments.csv` to verify the local reference-file hash as well. The verifier checks headline values, row counts, hard-gap safety invariants, and the paired primary and shifted contrasts.

The repository also retains the STRATUS-BW implementation and the policy/global context baselines used in the manuscript's main comparison table. Their committed results can be inspected without rerunning the full third-party-data workflow.

## Interpretation boundary

The operational labels are injected for controlled evaluation. `Movement` is a displacement-quantile proxy rather than an expert physiological label. The fixed participant split supports exact paired comparisons but does not measure uncertainty across repeated outer splits. The shifted-generator analysis is sensitivity evidence, not an independent natural-error benchmark.

## Manuscript mirror

`paper/` mirrors the current EDBT 2027 manuscript for convenience. It is **not** required to install, run, or verify the experimental artifact.
