# Open-to-Fist Reference Selection Rule

This rule is used after legacy transition trials have been formalized as
`open_to_fist` sequences.

## Candidate Source

Only trials with:

- `motion_type == open_to_fist`
- `usable_as_transition == 1`
- valid qpos sequence
- successful biomechanical ctrl_ref generation

are eligible for the open-to-fist reference pool.

## Ranking Metrics

Candidate trials are ranked by a composite quality score computed from:

- `rmse_forward_error`
- `mean_joint_error`
- `ctrl_delta_mean`
- `upper_saturation_fraction`

Each metric is min-max normalized within the candidate pool. Lower is better.

## Pool Decision

The default rebuild keeps the best trials as the reference pool and holds out
the remaining trials for trial-vs-reference evaluation. A trial should be held
out or excluded when it has high forward error, high joint error, unstable ctrl,
or unusually high saturation relative to the pool.

This selection only validates reference quality for analysis. It does not train
or tune a GCN model.
