# Open-to-Fist Continuous Sequence Data Spec

This document defines the minimum dataset contract for the offline biomechanical
pipeline:

recorded qpos -> qvel/qacc -> inverse dynamics -> ctrl_ref -> graph sequence ->
reference sequence -> node deviation analysis.

## Scope

One trial is one continuous motion sequence from an open hand toward a closed
fist. Open-only and fist-only recordings can be used as temporary proxies for
tooling tests, but they are not valid reference-quality open-to-fist trials.

## Required Trial Metadata

Each trial should provide:

- `subject_id`: stable subject identifier, anonymized if needed.
- `trial_id`: unique trial identifier within the dataset.
- `motion_type`: expected value `open_to_fist`.
- `exercise_name`: descriptive label, preferably `open_to_fist`.
- `source_file`: original source file path.
- `num_frames`: number of raw frames before resampling.
- `frame_rate` or `dt`: recording cadence if known.
- `is_proxy`: `false` for real continuous open-to-fist data.

Recommended optional metadata:

- `session_id`
- `hand_side`
- `device_name`
- `recorded_at`
- `operator_notes`

## Required Framewise Data

Each frame should contain:

- `frame_index` or timestamp.
- `qpos_0 ... qpos_N`: MyoSuite/MuJoCo qpos vector in a consistent dimension.
- `progress_percent`: normalized motion progress from 0 to 100.

Optional framewise data:

- `qvel_0 ... qvel_N`
- `qacc_0 ... qacc_N`
- `ctrl_ref_0 ... ctrl_ref_M`
- phase label: `start_open`, `transition`, `near_fist`, `hold_fist`

## Progress Normalization

All analytical sequence datasets must be progress-normalized, not compared on
absolute time. The baseline rule is linear resampling:

- first usable frame = 0 percent
- last usable frame = 100 percent
- sequence resampled to fixed `sequence_length`

If explicit segment annotations are available, they should be preserved, but the
baseline pipeline must still work with linear progress only.

## Phase Labels

If no manual segmentation exists, the baseline phase labels are:

- `start_open`: progress 0 to 10 percent
- `transition`: progress greater than 10 and less than 85 percent
- `near_fist`: progress 85 to less than 95 percent
- `hold_fist`: progress 95 to 100 percent

These labels are descriptive helpers; they are not classifier labels.

## Quality Gates

A trial is eligible for reference building only if it passes these basic checks:

- `num_frames >= min_frames`
- qpos dimension is consistent across the dataset
- qpos values are finite after cleaning
- start-to-end qpos distance is non-trivial for a transition trial
- no large missing blocks in the frame sequence
- biomechanical ctrl_ref generation succeeds
- forward validation error is within the selected reference-quality threshold

Proxy data must be marked explicitly and should not be treated as final
reference data for GCN training.

## Current Repository Status

The current repository contains open-only and fist-only records that can exercise
the tooling, but they are not a formal continuous open-to-fist dataset. Any
reference built from those records remains provisional until true continuous
trials are recorded and accepted.
