# Reproducibility notes

## Fixed experimental design

- at most 10 source CSV files per dataset;
- one valid 8-second reference segment per participant;
- minimum valid fraction: 0.98;
- participant-level split, test fraction 0.30, split seed 42;
- corruption seeds 0--9;
- severity levels: low, medium, high;
- corruption types: short gap, long gap, jitter, spike, mixed;
- short/long gap boundary: 0.5 seconds;
- Baum--Welch training seeds 0--4, at most 120 sequences per dataset, at most 50 iterations;
- K-means initialization with 30 restarts.

## Final outputs

Use `results/tables/macro_results_with_ci.csv` as the primary result table. The pooled table is a sensitivity analysis because the held-out Autism set is larger than the ETDD70 set.

## No additional run required

The committed tables, plots, and executed notebook correspond to the final v7 run used for the manuscript. Re-execution is optional and intended only for independent reproduction.
