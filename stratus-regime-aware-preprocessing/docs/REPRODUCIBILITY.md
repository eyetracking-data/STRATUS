# Reproducibility notes

## Reported experimental design

- 47 participant-specific reference windows: 10 ETDD70 and 37 Autism;
- participant-separated outer split: 33 train/development identifiers and 14 held-out identifiers;
- inner development split: 8 identifiers, leaving 25 for inner fitting;
- primary test: 10 corruption seeds, 3 severities, and 5 corruption settings per held-out identifier (2,100 degraded sequences);
- shifted-generator check: 2 seeds, 3 severities, 5 settings, and the same 14 held-out identifiers (420 sequences);
- short/long gap boundary: 0.5 seconds;
- hybrid observation regularization selected on development data: `C = 1`;
- temporal transition strength selected on development data: `lambda = 1`;
- participant-bootstrap replicates: 2,000.

The exact participant split is in `results/tables/participant_split.csv`. The extraction summary is in `results/tables/reference_extraction_summary.csv`.

## Primary manuscript outputs

For the current hybrid evaluation, use:

- `results/hybrid/primary/macro_results.csv` for the dataset-macro headline values;
- `results/hybrid/primary/paired_bootstrap_contrasts.csv` for the matched contrasts;
- `results/hybrid/primary/participant_results.csv` for participant-level aggregation;
- `results/hybrid/primary/sequence_results.csv` for sequence-level results;
- `results/hybrid/primary/observation_C_tuning.csv` and `transition_strength_tuning.csv` for development selection;
- `results/hybrid/primary/run_summary.json` for the fixed run summary.

The older tables under `results/tables/` are retained because they support the original diagnostic/STRATUS-BW and context-baseline analyses. They should not be mistaken for the current four-model hybrid runner output.

## Verification without refitting

The committed results can be checked directly:

```bash
python scripts/verify_hybrid_results.py
```

This verifies the reported headline values, expected row counts, hard-gap safety behavior, and the primary and shifted STRATUS-H-versus-Hybrid-P contrasts. The repository test suite can be run with:

```bash
pytest
```

A local copy of the reference file is only needed to verify its SHA-256 hash or to rerun the full experiments.

## Reference data

The coordinate-level reference file is intentionally not committed. Its expected structure and SHA-256 are documented in `docs/REFERENCE_DATA.md`. Raw dataset acquisition and directory layout are documented in `data/README.md` and `THIRD_PARTY_DATA.md`.
