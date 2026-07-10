# STRATUS

**Regime-aware local operator selection for adaptive time-series preprocessing**

STRATUS treats time-series preprocessing as a sequence of local operator decisions rather than one global pipeline. The repository contains the implementation, the participant-safe and metric-safe eye-tracking evaluation, final aggregate results, and the LaTeX manuscript.

## Model variants

- **STRATUS-D:** interpretable diagnostic potentials plus Viterbi decoding.
- **STRATUS-BW:** diagonal-Gaussian HMM trained with Baum--Welch, followed by training-only semantic component alignment and Viterbi decoding.
- **Oracle policy:** the shared action policy applied to known injected regimes; this is an analysis upper bound, not a deployable method.

Operational states are `Stable`, `Movement`, `LossShort`, `LossLong`, and `Unstable`. Their actions are preserve, preserve, interpolate, keep missing, and rolling-median correction.

## Final v7 headline results

The primary paper table is the equal-weight macro-average over ETDD70 and the Autism dataset. Values below are the committed final outputs; rerunning is not required merely to inspect the artifact.

| Method | MAE ↓ | Long-gap hallucination ↓ | Short recovery ↑ | Local action score ↑ |
|---|---:|---:|---:|---:|
| Oracle policy | 1.526 | 0.000 | 1.000 | 0.956 |
| STRATUS-D | 4.849 | 0.000 | 1.000 | 0.841 |
| Short-gap only | 10.408 | 0.000 | 1.000 | 0.761 |
| STRATUS-BW | 5.542 | 0.000 | 0.000 | 0.642 |

The local action score is paper-specific and is always reported together with its five components. See `docs/METRICS.md` and `results/tables/macro_results_with_ci.csv`.

## Repository layout

```text
src/stratus/        reusable implementation
notebooks/          clean reproducibility notebook
notebooks/executed/ sanitized executed v7 notebook
results/            committed aggregate tables and final figures
paper/              LaTeX manuscript, bibliography, figures, preview PDF
data/               dataset acquisition and folder instructions
scripts/            validation and execution helpers
tests/              regression tests for critical v7 fixes
```

## Quick start

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[notebook,dev]"
pytest
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[notebook,dev]'
pytest
```

## Data

Raw datasets are **not redistributed**. Obtain ETDD70 and the Eye-Tracking Autism dataset from their original repositories and place them as described in [`data/README.md`](data/README.md).

Default locations:

```text
data/raw/etdd70/
data/raw/autism/
```

Alternative locations can be provided with environment variables:

```text
STRATUS_ETDD70_DIR
STRATUS_AUTISM_DIR
```

## Reproduce the full experiment

The complete run is computationally expensive. Final outputs are already committed. To regenerate them:

```bash
python scripts/validate_data.py
python scripts/run_experiment.py
```

The run overwrites files in `results/` and `paper/figures/`. Use the clean notebook in `notebooks/`; the committed executed notebook is retained only as a transparent record of the final v7 run.

## Tests

```bash
pytest
```

The tests cover two publication-critical properties:

1. the short-gap baseline never partially fills a long missing run;
2. unstable-repair gain is zero for unchanged degraded data and one for perfect repair.

## Paper

The manuscript source is in `paper/main.tex`. The architecture diagram is defined directly in TikZ. The lower state legend uses a color family distinct from the workflow stages to avoid semantic ambiguity.

## Citation

See [`CITATION.cff`](CITATION.cff). Update the citation metadata and add a DOI after publication or archival release.

## License

Code is released under the MIT License. The manuscript, source datasets, and third-party materials remain subject to their own terms; see [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md).
