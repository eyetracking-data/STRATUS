# STRATUS — EDBT 2027 manuscript mirror

This directory mirrors the current manuscript **“STRATUS: Selective Temporal Coupling for Auditable Time-Series Preprocessing”**. It is included for convenience only; the experimental artifact can be installed, executed, and verified without the paper files.

## Compile

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

or use the equivalent `pdflatex`/BibTeX sequence.

## Current figure set

- Figure 1: integrated STRATUS workflow;
- Figure 2: controlled STRATUS method family;
- Figure 3: primary matched versus shifted-generator local-action score;
- Figure 4: qualitative held-out Hybrid-P versus STRATUS-H examples.

The qualitative reproduction script retains the historical filename `figure2_qualitative_hybrid.pdf` for artifact stability even though the visualization is Figure 4 in the current manuscript.

## Data boundary

Third-party raw data and coordinate-level reference windows are not included. The repository contains the fixed participant split, extraction summary, reference-file SHA-256 record, committed experimental outputs, tests, and verification scripts. See the repository-level README and `docs/REPRODUCIBILITY.md` for the execution path.
