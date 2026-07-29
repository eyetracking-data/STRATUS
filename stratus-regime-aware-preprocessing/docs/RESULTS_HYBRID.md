# STRATUS-H result record

## Primary held-out test

- 47 reference windows total;
- 33 outer-training/development identifiers;
- 14 held-out identifiers;
- 10 corruption seeds;
- 3 severities;
- 5 corruption settings;
- 2,100 degraded sequences and 8,400 method-sequence rows.

| Method | MAE | RMSE | Q_local |
|---|---:|---:|---:|
| STRATUS-H | 3.597716 | 14.615126 | 0.919967 |
| Hybrid-P | 3.824765 | 14.820809 | 0.914260 |
| Pointwise-D | 3.566230 | 22.703464 | 0.881643 |
| STRATUS-D | 4.849231 | 32.456003 | 0.840825 |

### Isolated temporal contrast

| Metric | STRATUS-H minus Hybrid-P | 95% participant-bootstrap interval |
|---|---:|---:|
| Q_local | +0.005707 | [0.004270, 0.007139] |
| MAE | -0.227048 | [-0.320299, -0.148167] |
| RMSE | -0.205682 | [-0.347815, -0.063293] |

All 14 held-out identifiers have higher participant-level Q_local under
STRATUS-H than Hybrid-P.

## Auxiliary shifted-generator check

The check changes locations, gap durations, Movement quantile, jitter
mechanism, and spike structure. It uses two seeds and 420 degraded sequences.

| Method | MAE | RMSE | Q_local |
|---|---:|---:|---:|
| STRATUS-H | 3.258123 | 14.641408 | 0.906536 |
| Hybrid-P | 3.473623 | 14.809625 | 0.903081 |
| Pointwise-D | 2.493234 | 16.822263 | 0.901581 |
| STRATUS-D | 2.857654 | 20.880645 | 0.869952 |

The STRATUS-H-minus-Hybrid-P Q_local difference remains positive at +0.003455
with interval [0.001532, 0.005430]. The contrast with Pointwise-D is no longer
decisive in Q_local, and Pointwise-D has lower MAE.

## Interpretation

The evidence supports a small independent temporal effect inside the hybrid
model. It does not support attributing the entire STRATUS-H-versus-Pointwise-D
gain to an HMM. The state topology, hard constraints, and observation model are
the larger architectural change.


## Qualitative held-out examples

Paper Figure 2 visualizes two mixed-corruption test sequences:

- ETDD70 participant 1003, `ETDD70__1003__seg0`, medium severity, seed 5;
- Autism participant 37, `Autism__37__seg18`, medium severity, seed 2.

The exact per-example values are committed in
`results/hybrid/qualitative/figure2_examples.csv`. Hybrid-P and STRATUS-H use
the same local model and policy; the state strips therefore isolate the effect
of temporal decoding. The examples were chosen after evaluation to make that
mechanism visible and are not used for inference.
