# Neo4j Graph Schema

Neo4j is an auxiliary graph database and visual exploration layer. It is not the training engine and does not replace the offline biomechanical pipeline.

## Time-Series Representation

One `open_to_fist` trial is represented as a sequence of graph snapshots:

`(:Trial)-[:HAS_STEP]->(:SequenceStep)-[:HAS_MUSCLE_STATE]->(:MuscleState)-[:STATE_OF]->(:Muscle)`

This is intentionally not a single static graph. `SequenceStep` stores normalized progress/time. `MuscleState` stores per-muscle features for that specific step.

## Node Labels

### `Trial`
Properties:
- `trial_id`
- `motion_type`
- `source_file`
- `subject_id`
- `cohort`
- `is_reference`
- `is_holdout`
- `num_frames`
- `num_steps`
- `num_windows`
- `edge_mode`

### `SequenceStep`
Properties:
- `step_id`
- `trial_id`
- `step_index`
- `progress`
- `progress_start`
- `progress_end`
- `time_normalized`
- `reference_sequence_id`

### `Muscle`
Static muscle/actuator identity.

Properties:
- `muscle_name`
- `node_index`

### `MuscleState`
Per-step muscle state.

Properties:
- `state_id`
- `trial_id`
- `step_index`
- `progress`
- `muscle_name`
- `node_index`
- `mean_ctrl`
- `std_ctrl`
- `max_ctrl`
- `delta_ctrl`
- `auc_ctrl`
- `deviation_score`
- `binary_label`
- `predicted_label`
- `prediction_confidence`

### `ReferenceSequence`
Normal reference sequence for a motion type.

Properties:
- `reference_sequence_id`
- `motion_type`
- `source_dir`
- `num_reference_trials`
- `num_steps`
- `notes`

### `ReferenceStep`
Reference step at normalized progress.

Properties:
- `reference_step_id`
- `reference_sequence_id`
- `step_index`
- `progress`
- `progress_start`
- `progress_end`
- `time_normalized`

### `ReferenceMuscleState`
Reference statistics for a muscle at a reference step.

Properties:
- `reference_state_id`
- `reference_sequence_id`
- `step_index`
- `muscle_name`
- `mean_ctrl_mean`
- `mean_ctrl_std`
- `max_ctrl_mean`
- `max_ctrl_std`
- `delta_ctrl_mean`
- `auc_ctrl_mean`

### `ClassificationResult`
Binary node classification output.

Properties:
- `result_id`
- `trial_id`
- `step_index`
- `muscle_name`
- `baseline_label`
- `predicted_label`
- `confidence`
- `source`

## Relationship Types

### Temporal/data relationships
- `(Trial)-[:HAS_STEP]->(SequenceStep)`
- `(SequenceStep)-[:HAS_MUSCLE_STATE]->(MuscleState)`
- `(MuscleState)-[:STATE_OF]->(Muscle)`
- `(MuscleState)-[:HAS_RESULT]->(ClassificationResult)`
- `(SequenceStep)-[:COMPARED_TO]->(ReferenceSequence)`
- `(ReferenceSequence)-[:HAS_REFERENCE_STEP]->(ReferenceStep)`
- `(ReferenceStep)-[:HAS_REFERENCE_MUSCLE_STATE]->(ReferenceMuscleState)`
- `(ReferenceMuscleState)-[:REFERENCE_STATE_OF]->(Muscle)`

### Static anatomical relationships
- `(Muscle)-[:SYNERGIST_WITH]->(Muscle)`
- `(Muscle)-[:ANTAGONIST_WITH]->(Muscle)`
- `(Muscle)-[:FUNCTIONAL_GROUP_WITH]->(Muscle)`

Properties:
- `edge_type`
- `prior_weight`

### Dynamic per-step relationships
- `(MuscleState)-[:DYNAMIC_RELATION]->(MuscleState)`

Properties:
- `trial_id`
- `step_index`
- `edge_idx`
- `edge_type`
- `weight`
- `active`
- `prior_weight`

## Why `MuscleState` Is Used

`Muscle` is static. Its activation and deviation change over time. Storing step-specific values directly on `Muscle` would overwrite the time series. `MuscleState` keeps temporal values explicit and queryable.

