# Transition Sequence Usability Rule

This rule determines whether legacy `open` and `fist` recordings can be
reinterpreted as continuous transition sequences.

## Intended Relabeling

- Legacy `fist` trial -> `open_to_fist`
- Legacy `open` trial -> `fist_to_open`

The original labels describe the target gesture, not necessarily a static pose.

## Data-Driven Posture Prototypes

Because old records do not include explicit segment metadata, posture references
are estimated from the dataset itself:

- `open` prototype = median of first-window frames from legacy `fist` trials and
  last-window frames from legacy `open` trials.
- `fist` prototype = median of last-window frames from legacy `fist` trials and
  first-window frames from legacy `open` trials.

This makes the audit dependent on actual qpos trajectories rather than folder
names alone.

## Per-Trial Metrics

Each trial is scored using:

- `num_frames`: raw frame count.
- `total_qpos_change`: norm between mean start qpos and mean end qpos.
- `path_efficiency`: start-to-end displacement divided by qpos path length.
- `start_stability_score`: mean frame-to-frame qpos change in the first window.
- `end_stability_score`: mean frame-to-frame qpos change in the final window.
- `qpos_noise_score`: median second-difference norm normalized by total change.
- `has_end_hold`: final-window motion is small enough to count as a hold/plateau.
- posture distances from start/end windows to inferred `open` and `fist`
  prototypes.

## Default Usability Criteria

A legacy trial is usable as a transition sequence when all are true:

- `num_frames >= 30`
- `total_qpos_change >= 2.5`
- `path_efficiency >= 0.55`
- `start_stability_score <= 0.04`
- `end_stability_score <= 0.04`
- `qpos_noise_score <= 0.08`
- start posture is closer to the expected start prototype than the opposite
  prototype, with adequate margin.
- end posture is closer to the expected end prototype than the opposite
  prototype, with adequate margin.
- `has_end_hold == true`

The thresholds are intentionally simple and auditable. They should be tightened
after formal continuous open-to-fist and fist-to-open recordings are available.

## Interpretation

Passing this audit means the old trial can be formalized as a transition
sequence for baseline offline analysis. It does not automatically mean the trial
is high-quality biomechanical ground truth; ctrl_ref validation remains a
separate requirement.
