# Binary Node Deviation Rule

Binary labels are used as a stage-1 sanity-check task before multiclass subtype classification.

## Classes
- `within_reference`: node is close enough to the normal reference sequence.
- `deviating`: node is sufficiently far from the normal reference sequence.

## Rule Used In Current Smoke Test
- Compute `average_deviation_score` per node from trial-vs-reference comparison.
- Set threshold from train split only: quantile=0.75, threshold=0.6934.
- Assign `deviating` when `average_deviation_score >= threshold`.
- Also assign `deviating` when magnitude status is `too_weak` or `too_high`, or smoothness status is `not_smooth`.
- Otherwise assign `within_reference`.

This intentionally avoids mapping every `onset_shift` to binary deviation, because the current onset rule is known to dominate multiclass labels.