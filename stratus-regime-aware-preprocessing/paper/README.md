# STRATUS-H - EDBT 2027 final paper package

This directory contains the final manuscript
**"STRATUS: Selective Temporal Coupling for Auditable Time-Series Preprocessing"**.
It uses the official EDBT 2027 `acmart` template and A4 geometry.

## Compile

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Equivalent manual compilation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The checked submission preview is included as:

```text
STRATUS_EDBT_2027_FINAL.pdf
```

## Main revision

The paper replaces the claim that generic persistence should improve STRATUS
with a matched selective-temporal experiment:

- `Pointwise-D`: original diagnostic pointwise model;
- `STRATUS-D`: original diagnostic model with fixed Viterbi persistence;
- `Hybrid-P`: constrained discriminative finite-state model without transitions;
- `STRATUS-H`: the identical hybrid model with learned transitions and Viterbi;
- `STRATUS-BW`: the original diagonal-Gaussian Baum-Welch model.

The fair HMM-only contrast is `STRATUS-H - Hybrid-P`. The primary result is a
small but participant-consistent improvement in local-action quality, MAE, and
RMSE. The manuscript explicitly states that the larger gain over Pointwise-D
comes mainly from the constrained state topology and discriminative observation
model, not from temporal decoding alone.

## Final figure set

- Figure 1: architecture and selective temporal coupling.
- Figure 2: held-out qualitative time-series examples.
- Figure 3: matched-versus-shifted generator comparison.

The former two-panel aggregate scatter/bootstrap figure was removed. Its values
remain in Table 2 and in the paired-bootstrap result paragraph, avoiding a
redundant and weakly informative visualization.

Figure 2 is pre-rendered as `figures/figure2_qualitative_hybrid.pdf`.
Reproduction code is in `../scripts/make_qualitative_figure.py`; it requires the
local v7 reference CSV because coordinate-level windows are not redistributed.

## Data and reproducibility

Third-party raw data and coordinate-level reference windows are not included.
The repository contains deterministic extraction and experiment code, the exact
participant split, extraction metadata, the SHA-256 hash of the local v7
reference file, primary and shifted-generator outputs, tests, and verification
scripts.

Expected local reference-file SHA-256:

```text
c4e4b81c948e28c80e0ba50c84e118e54081d5257cb3c5b7a2a9d1e85e467525
```
