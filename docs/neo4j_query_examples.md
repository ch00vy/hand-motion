# Neo4j Query Examples

These queries assume the `open_to_fist` export bundle has been imported.

## 1. Muscles Most Often Predicted Deviating

```cypher
MATCH (ms:MuscleState)
WHERE ms.predicted_label = 'deviating'
RETURN ms.muscle_name AS muscle, count(*) AS deviating_count
ORDER BY deviating_count DESC
LIMIT 20;
```

## 2. Progress Regions With Most Deviations

```cypher
MATCH (ms:MuscleState)
WHERE ms.predicted_label = 'deviating'
WITH floor(ms.progress / 10) * 10 AS progress_bin, count(*) AS deviating_count
RETURN progress_bin, progress_bin + 10 AS progress_bin_end, deviating_count
ORDER BY deviating_count DESC;
```

## 3. Trials With Most Deviating Muscles

```cypher
MATCH (t:Trial)-[:HAS_STEP]->(:SequenceStep)-[:HAS_MUSCLE_STATE]->(ms:MuscleState)
WHERE ms.predicted_label = 'deviating'
RETURN t.trial_id AS trial_id, t.is_reference AS is_reference, t.is_holdout AS is_holdout, count(ms) AS deviating_states
ORDER BY deviating_states DESC;
```

## 4. Muscles Deviating During Mid-Transition

```cypher
MATCH (ms:MuscleState)
WHERE ms.progress >= 35 AND ms.progress <= 65
  AND ms.predicted_label = 'deviating'
RETURN ms.muscle_name AS muscle, count(*) AS deviating_mid_transition
ORDER BY deviating_mid_transition DESC
LIMIT 20;
```

## 5. Within vs Deviating Nodes For One Trial

```cypher
MATCH (t:Trial {trial_id: 'open_to_fist__fist_t12'})-[:HAS_STEP]->(s:SequenceStep)-[:HAS_MUSCLE_STATE]->(ms:MuscleState)
RETURN s.step_index AS step_index,
       round(s.progress, 2) AS progress,
       ms.muscle_name AS muscle,
       ms.predicted_label AS predicted_label,
       ms.binary_label AS baseline_label,
       round(ms.deviation_score, 3) AS deviation_score
ORDER BY s.step_index, ms.muscle_name;
```

## 6. Dynamic Relations Around Deviating Muscle States

```cypher
MATCH (a:MuscleState)-[r:DYNAMIC_RELATION]->(b:MuscleState)
WHERE a.predicted_label = 'deviating'
  AND a.progress >= 35 AND a.progress <= 65
RETURN a.muscle_name AS source,
       b.muscle_name AS target,
       r.edge_type AS edge_type,
       count(*) AS relation_count,
       avg(abs(r.weight)) AS mean_abs_weight
ORDER BY relation_count DESC, mean_abs_weight DESC
LIMIT 30;
```

## 7. Holdout Deviations Only

```cypher
MATCH (t:Trial)-[:HAS_STEP]->(:SequenceStep)-[:HAS_MUSCLE_STATE]->(ms:MuscleState)
WHERE t.is_holdout = true AND ms.predicted_label = 'deviating'
RETURN t.trial_id AS trial_id, ms.muscle_name AS muscle, count(*) AS deviating_states
ORDER BY trial_id, deviating_states DESC;
```

## 8. Compare Trial Muscle State To Reference Muscle State By Step

```cypher
MATCH (t:Trial {trial_id: 'open_to_fist__fist_t12'})-[:HAS_STEP]->(s:SequenceStep)-[:HAS_MUSCLE_STATE]->(ms:MuscleState)
MATCH (:ReferenceSequence {reference_sequence_id: s.reference_sequence_id})-[:HAS_REFERENCE_STEP]->(rs:ReferenceStep)
MATCH (rs)-[:HAS_REFERENCE_MUSCLE_STATE]->(rms:ReferenceMuscleState)
WHERE rs.step_index = s.step_index AND rms.muscle_name = ms.muscle_name
RETURN s.step_index AS step_index,
       ms.muscle_name AS muscle,
       ms.mean_ctrl AS trial_mean_ctrl,
       rms.mean_ctrl_mean AS reference_mean_ctrl,
       ms.deviation_score AS deviation_score
ORDER BY deviation_score DESC
LIMIT 50;
```

