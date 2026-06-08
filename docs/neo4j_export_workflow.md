# Neo4j Export Workflow

## Dry-Run Bundle Generation

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\export_open_to_fist_to_neo4j.py --input-dir artifacts\open_to_fist_graph_sequence_rebuild\dataset --reference-root-dir artifacts\open_to_fist_reference_sequence_rebuild --compare-dir artifacts\open_to_fist_trial_vs_reference_reaudit --binary-predictions-dir artifacts\gcn_binary_smoke --output-dir artifacts\neo4j_export_bundle --motion-type open_to_fist --include-reference --include-holdout --include-deviation --dry-run
```

This creates:
- `trials.csv`
- `sequence_steps.csv`
- `muscles.csv`
- `muscle_states.csv`
- `static_relations.csv`
- `dynamic_relations.csv`
- `classification_results.csv`
- `reference_sequences.csv`
- `reference_steps.csv`
- `reference_muscle_states.csv`
- `reference_edges.csv`
- `import_open_to_fist_bundle.cypher`
- `export_summary.json`

## Optional Neo4j Constraint Smoke

If the `neo4j` Python driver is installed and a Neo4j server is running:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\export_open_to_fist_to_neo4j.py --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password <password> --input-dir artifacts\open_to_fist_graph_sequence_rebuild\dataset --motion-type open_to_fist --include-reference --include-holdout --include-deviation
```

The script installs constraints if it can connect. The CSV import is still done through the generated Cypher bundle because Neo4j `LOAD CSV` reads from Neo4j's import directory.

