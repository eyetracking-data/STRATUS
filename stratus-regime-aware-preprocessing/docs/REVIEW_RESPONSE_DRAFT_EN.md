# Draft response to the methodological review

We thank the reviewer for identifying that the earlier manuscript did not
isolate the contribution of temporal inference and that its strongest
pointwise baseline weakened the HMM claim. The revision changes both the model
and the interpretation.

## 1. Pointwise-D was stronger than the temporal variants

**Revision.** We retain Pointwise-D and STRATUS-D and introduce a matched pair:
Hybrid-P and STRATUS-H. They share the same hard gap constraints, finite-state
features, classifier, topology, and action policy. STRATUS-H differs only by a
learned transition matrix and Viterbi decoding. This makes STRATUS-H minus
Hybrid-P the isolated temporal contrast.

**Result.** The primary participant-paired difference is +0.0057 in local-action
score with 95% interval [0.0043, 0.0071], accompanied by lower MAE and RMSE. We
therefore describe the temporal effect as small but consistent. We explicitly
state that the larger improvement over Pointwise-D is mainly architectural and
must not be attributed to the HMM alone.

## 2. Coupling between injected ground truth and model definitions

**Revision.** We now call these labels injected operational reference labels,
not physiological ground truth. LossShort/LossLong are treated as a policy and
implementation check. Movement is explicitly described as a displacement-
quantile proxy. An auxiliary generator-shift check changes positions, gap
durations around the policy boundary, Movement quantile, jitter distribution,
and spike structure.

**Remaining limitation.** The shifted check is still synthetic, uses the same
reference participants, and has two seeds. It does not replace naturally
annotated artifacts.

## 3. Participant split and uncertainty

**Revision.** Hyperparameter selection uses an inner participant development
split and final evaluation uses the unchanged 14-identifier outer test set.
All primary contrasts use paired participant bootstrap intervals, and
participant-level results are committed.

**Remaining limitation.** The revision does not claim uncertainty across
repeated outer splits. The fixed split is retained for an exact matched
comparison with the earlier artifact; repeated split or leave-one-participant-
out evaluation remains future work.

## 4. HMM robustness and alternative sequence models

**Revision.** The weak diagonal-Gaussian Baum-Welch result is retained rather
than hidden. The new model uses a constrained hybrid posterior/HMM formulation
with hard missingness states and learned transitions only among ambiguous
finite states. The paper clearly distinguishes this from a fully generative
Gaussian HMM and reports development selection of the observation regularizer
and transition strength.

**Remaining limitation.** Full covariance, mixture emissions, calibrated
posterior-to-likelihood conversion, multi-initialization uncertainty, and
hidden semi-Markov durations are not claimed as completed experiments; they
are listed as future work.

## 5. Stronger data-cleaning baselines and third domain

**Revision.** Related work now discusses BoostClean, DiffPrep, and PClean and
explains why they are not direct drop-in position-specific temporal baselines.
The experimental comparison retains temporal, gap, global transformation, and
oracle baselines.

**Remaining limitation.** We did not add a third naturally annotated dataset or
implement the full modern data-cleaning systems. The paper narrows its claim to
the controlled eye-tracking case study and states this limitation explicitly.
