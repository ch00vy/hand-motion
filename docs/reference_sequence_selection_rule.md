# Reference Sequence Selection Rule

The normal reference sequence should be built from trials that pass two levels
of screening.

## Level 1: Dataset Eligibility

A trial is eligible only when:

- it represents one continuous `open_to_fist` sequence, or is explicitly marked
  as proxy data during tooling validation;
- qpos dimensionality matches the reference pool;
- frame count passes the configured minimum;
- progress normalization succeeds;
- biomechanical `ctrl_ref` generation succeeds.

Proxy data is allowed only for engineering smoke tests and provisional audits.
It is not sufficient for final GCN training.

## Level 2: Biomechanical Reference Quality

Eligible trials are ranked using a composite quality score over:

- `rmse_forward_error`
- `mean_joint_error`
- `ctrl_delta_mean`
- `upper_saturation_fraction`

Each metric is min-max normalized within the candidate pool. Lower composite
score is better. The default provisional pool selection keeps the best trials
and holds out the rest for trial-vs-reference testing.

## Exclusion Guidance

Exclude or hold out trials when they show:

- high forward RMSE relative to the candidate pool;
- high mean joint error;
- large control jumps;
- high actuator saturation;
- unstable graph sequence after progress normalization;
- metadata indicating proxy or non-continuous source when final reference data
  is required.

## Current Status

The current reference-strengthening run still uses proxy fist recordings because
no formal continuous open-to-fist source dataset was found. The selection rule is
valid, but the resulting reference is provisional until real continuous
open-to-fist trials are available.
