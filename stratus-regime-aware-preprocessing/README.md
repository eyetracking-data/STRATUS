# STRATUS

**Auditable local operator selection with selective temporal coupling**

STRATUS treats time-series preprocessing as a sequence of explicit local
decisions rather than one global pipeline. The revised artifact separates a
better finite-state representation from the independent contribution of a
Markov transition layer.

## Model variants

- **Pointwise-D:** original interpretable diagnostic potentials, independently maximized.
- **STRATUS-D:** the same diagnostic potentials plus fixed Viterbi persistence.
- **STRATUS-BW:** diagonal-Gaussian Baum-Welch HMM with training-only semantic alignment.
- **Hybrid-P:** hard gap-duration constraints and a supervised discriminative finite-state observation model, without transitions.
- **STRATUS-H:** the identical Hybrid-P model plus a learned transition matrix and Viterbi decoding within finite blocks.
- **Oracle policy:** shared state-to-action policy applied to injected operational labels; analysis upper bound only.

Operational states are `Stable`, `Movement`, `LossShort`, `LossLong`, and
`Unstable`. STRATUS-H internally divides Unstable into impulse and burst
substates; both map to the same rolling-median action. Complete missing-run
duration directly selects LossShort or LossLong, so the HMM cannot bridge gaps
or convert finite samples into loss states.

## Headline primary results

Equal-weight dataset macro-average over held-out ETDD70 and Autism identifiers:

| Method | MAE ↓ | RMSE ↓ | Local action score ↑ |
|---|---:|---:|---:|
| STRATUS-H | 3.598 | **14.615** | **0.920** |
| Hybrid-P | 3.825 | 14.821 | 0.914 |
| Pointwise-D | **3.566** | 22.703 | 0.882 |
| STRATUS-D | 4.849 | 32.456 | 0.841 |
| STRATUS-BW | 5.542 | 39.121 | 0.642 |

The fair HMM-only contrast is **STRATUS-H minus Hybrid-P**:

- local action score: `+0.0057`, participant-bootstrap 95% interval `[0.0043, 0.0071]`;
- MAE: `-0.227`, interval `[-0.320, -0.148]`;
- RMSE: `-0.206`, interval `[-0.348, -0.063]`.

The temporal effect is small but consistent. The larger gain over Pointwise-D
must not be attributed to the HMM alone because state topology, supervision,
and observation features also change. Pointwise-D retains a slightly lower
primary MAE.

## Repository layout

```text
src/stratus/                         reusable implementation
scripts/run_hybrid_experiment.py     primary participant-safe experiment
scripts/run_shifted_robustness.py    auxiliary shifted-generator check
scripts/make_qualitative_figure.py  recreate the held-out time-series Figure 2
scripts/verify_hybrid_results.py     deterministic result verification
results/hybrid/primary/              committed primary outputs
results/hybrid/shifted/              committed sensitivity outputs
results/hybrid/weight_sensitivity/   score-weight checks
results/hybrid/qualitative/          exact Figure 2 example specification
results/tables/                      participant split and extraction metadata
paper/                               final LaTeX source, figures, and PDF
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

Raw third-party datasets and coordinate-level reference windows are not
redistributed. Obtain ETDD70 and the Eye-Tracking Autism dataset from their
original repositories and follow `data/README.md` from the existing artifact.
The v7 reference extraction must reproduce this SHA-256 value:

```text
c4e4b81c948e28c80e0ba50c84e118e54081d5257cb3c5b7a2a9d1e85e467525
```

## Reproduce the primary experiment

```bash
python scripts/run_hybrid_experiment.py \
  --reference /path/to/clean_reference_segments.csv \
  --output results/hybrid/primary
```

A quick technical smoke test does not require external data:

```bash
python scripts/run_hybrid_experiment.py --self-test --output /tmp/stratus-self-test
```

The auxiliary generator-shift check uses two seeds in the paper:

```bash
python scripts/run_shifted_robustness.py \
  --reference /path/to/clean_reference_segments.csv \
  --seeds 2
```

Recreate the qualitative time-series comparison used as Figure 2:

```bash
python scripts/make_qualitative_figure.py \
  --reference /path/to/clean_reference_segments.csv
```

The figure uses two held-out mixed-corruption sequences and reports the exact
example metrics in `results/hybrid/qualitative/figure2_examples.csv`. The
examples are illustrative only; model selection and quantitative claims use the
complete held-out evaluation.

Verify the committed outputs without refitting:

```bash
python scripts/verify_hybrid_results.py
```

Add `--reference /path/to/clean_reference_segments.csv` to verify the local
reference-file hash too.

## Interpretation boundary

The operational labels are injected for controlled evaluation. Movement is a
displacement-quantile proxy rather than an expert physiological label. The
fixed participant split supports exact paired comparisons, but not uncertainty
across repeated outer splits. The shifted-generator result is sensitivity
evidence, not an independent natural-error benchmark.

## Paper

The revised manuscript and compiled PDF are in `paper/`:

**STRATUS: Selective Temporal Coupling for Auditable Time-Series Preprocessing**

The compiled paper preview is `paper/STRATUS_EDBT_2027_FINAL.pdf`.


## Final manuscript figure set

The committed paper contains three figures: the architecture, the qualitative
held-out time-series comparison, and the matched-versus-shifted generator
comparison. The former aggregate scatter/bootstrap panel was removed because
its values are reported more clearly in the main table and paired-bootstrap
text.
