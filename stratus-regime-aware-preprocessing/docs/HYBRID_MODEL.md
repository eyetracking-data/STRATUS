# STRATUS-H model definition

## Why the model is hybrid

STRATUS-H is not a classical fully generative Gaussian HMM. Its local terms are
posterior potentials from a balanced multinomial logistic model. The HMM layer
contributes a learned start distribution, a learned transition matrix, and
Viterbi path decoding.

## State topology

External operational states:

- `Stable` -> preserve;
- `Movement` -> preserve;
- `LossShort` -> linear interpolation;
- `LossLong` -> retain missingness;
- `Unstable` -> centered seven-sample rolling median.

Internal finite states:

- `Stable`;
- `Movement`;
- `UnstableImpulse`;
- `UnstableBurst`.

The two unstable substates share the external `Unstable` action. Complete
missing-run duration deterministically selects LossShort or LossLong using the
0.5-second policy boundary. Therefore loss states are outside the learned
finite-state decoder.

## Features

The observation model uses thirteen finite-sample features derived from:

- log velocity, acceleration, and local instability;
- within-sequence ranks of those diagnostics;
- centered short-window means;
- centered longer-window maxima;
- isolated-versus-neighborhood contrasts;
- plausibility violation.

No future raw target values or clean coordinates are supplied to the model.
Centered diagnostic windows make the current implementation offline rather
than strictly causal.

## Matched ablation

`Hybrid-P` and `STRATUS-H` share:

- training participants;
- features;
- logistic classifier;
- hard gap constraints;
- state topology;
- state-to-action policy.

`Hybrid-P` sets transition strength to zero and independently maximizes the
finite-state posterior. `STRATUS-H` adds only the learned Markov transition
terms and Viterbi decoding. Their difference is therefore the isolated temporal
contribution.
