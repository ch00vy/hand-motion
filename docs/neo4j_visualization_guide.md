# Neo4j Visualization Guide

Neo4j is used here for inspection, visual analytics, and audit. It is not part of realtime control or GCN training.

## Import

Generate the bundle:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\export_open_to_fist_to_neo4j.py --input-dir artifacts\open_to_fist_graph_sequence_rebuild\dataset --motion-type open_to_fist --include-reference --include-holdout --include-deviation --dry-run
```

Copy the generated CSV files from `artifacts/neo4j_export_bundle` into Neo4j's configured `import` directory, then run:

```cypher
:source import_open_to_fist_bundle.cypher
```

In `cypher-shell`, run:

```powershell
cypher-shell -u neo4j -p <password> -f import_open_to_fist_bundle.cypher
```

## Visualize One Trial

```cypher
MATCH p=(t:Trial {trial_id: 'open_to_fist__fist_t12'})-[:HAS_STEP]->(s:SequenceStep)-[:HAS_MUSCLE_STATE]->(ms:MuscleState)-[:STATE_OF]->(m:Muscle)
WHERE s.step_index IN [0, 15, 30, 45, 55]
RETURN p;
```

Use a limited set of steps first. A full trial contains many `MuscleState` and `DYNAMIC_RELATION` relationships.

## Visualize One Step

```cypher
MATCH p=(s:SequenceStep {step_id: 'open_to_fist__fist_t12::step::30'})-[:HAS_MUSCLE_STATE]->(ms:MuscleState)-[:STATE_OF]->(m:Muscle)
RETURN p;
```

To include dynamic edges:

```cypher
MATCH p=(a:MuscleState)-[:DYNAMIC_RELATION]->(b:MuscleState)
WHERE a.trial_id = 'open_to_fist__fist_t12' AND a.step_index = 30
RETURN p
LIMIT 100;
```

## Highlight Deviating Nodes

In Neo4j Browser or Bloom, style `MuscleState.predicted_label = 'deviating'` with a strong color. Query:

```cypher
MATCH (s:SequenceStep)-[:HAS_MUSCLE_STATE]->(ms:MuscleState)
WHERE ms.trial_id = 'open_to_fist__fist_t12'
  AND ms.predicted_label = 'deviating'
RETURN s, ms
ORDER BY s.step_index;
```

## Distinguish Reference vs Holdout

Use `Trial.is_reference` and `Trial.is_holdout`.

```cypher
MATCH (t:Trial)
RETURN t.trial_id, t.is_reference, t.is_holdout, t.motion_type
ORDER BY t.is_holdout DESC, t.trial_id;
```

In Bloom, create styling rules:
- `Trial.is_reference = true`: blue
- `Trial.is_holdout = true`: orange
- `MuscleState.predicted_label = 'deviating'`: red
- `MuscleState.predicted_label = 'within_reference'`: green/gray

## Find Problematic Transition Region

```cypher
MATCH (ms:MuscleState)
WHERE ms.predicted_label = 'deviating'
WITH floor(ms.progress / 10) * 10 AS bin_start, count(*) AS c
RETURN bin_start, bin_start + 10 AS bin_end, c
ORDER BY c DESC;
```

Use the returned bin to visualize a narrower step range.

## Practical Limits

The export includes per-step `MuscleState` nodes and optional per-step dynamic relations. Start with a single trial and step range in Browser/Bloom; avoid returning the entire database graph at once.

