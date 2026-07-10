# Evaluation metrics

STRATUS separates numerical reconstruction quality from local action quality.

## Reconstruction

- **MAE / RMSE:** computed where a clean reference and a finite output are available.
- **Retention:** fraction of positions with finite output coordinates.

## Local action components

- **Long-gap hallucination:** fraction of true long-loss positions that were filled. Lower is better.
- **Short-gap recovery:** fraction of true short-loss positions reconstructed. Higher is better.
- **Movement preservation:** preservation of injected/identified movement regions. Higher is better.
- **Stable preservation:** exact preservation of stable samples. Higher is better.
- **Unstable repair gain:** error reduction in injected unstable regions relative to the degraded input. The unclipped value reveals harmful processing; a clipped version is used only in the composite.

## Local action score

The paper-specific summary is

\[
\frac{1}{5}\left((1-H_{long}) + R_{short} + P_{movement} + P_{stable} + R_{unstable}\right),
\]

where the unstable-repair component is clipped to `[0,1]` for aggregation. This score is not presented as a universal data-quality metric. Every component remains available in the tables.

## Aggregation and uncertainty

The headline result is an equal-weight macro-average over ETDD70 and Autism. Participant-level hierarchical bootstrap intervals are reported to avoid treating corruption repetitions as independent participants.
