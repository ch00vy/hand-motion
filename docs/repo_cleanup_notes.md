# Repository Cleanup Notes

Date: 2026-06-03

## Goal

Prepare the project for a clean GitHub repository by keeping source code and final workflow documentation while removing old generated development artifacts.

## Kept Locally

These folders are still useful for local execution but are ignored by git:

- `venv/`: local Python environment
- `Ultraleap/`: local Ultraleap SDK install
- `leapc-python-bindings/`: local LeapC Python bindings checkout
- `records/`: private/local motion recordings
- `artifacts/`: generated model, reference, ctrl_ref, graph, and training outputs
- `proposal_figures/`: generated report/proposal figures

## Kept Generated Artifacts

Only the current/latest local artifacts were kept:

- `artifacts/final_binary_gcn_run`
- `artifacts/final_binary_gcn_run_gui_plus_previous`
- `artifacts/open_to_fist_primary_source`
- `artifacts/open_to_fist_biomech_reference_latest_short_horizon`
- `artifacts/open_to_fist_graph_sequence_latest_short_horizon`
- `artifacts/open_to_fist_reference_sequence_latest_short_horizon`
- `artifacts/retrain_gui_plus_previous`

These remain local and are ignored by git.

## Removed Generated Artifacts

Removed old smoke/debug/export artifacts, including:

- old ctrl_ref smoke folders
- old mismatch feedback smoke folders
- old graph/reference rebuild audit folders
- old Neo4j export bundle
- old Leap image test output
- old legacy transition audit outputs
- old live predictor/runtime experiment outputs
- old generated markdown reports under `reports/`
- Python `__pycache__` folders

## Removed Legacy Data Folders

Removed root-level old data folders that were not referenced by the active workflow:

- `dataset old/`
- `batch/`

## Added Repo Files

- `.gitignore`
- `README.md`
- `requirements.txt`
- `docs/repo_cleanup_notes.md`

## Active Workflow Reminder

The current main workflow is:

```text
GUI record/load qpos
-> offline biomech ctrl_ref
-> graph sequence
-> reference sequence
-> binary GCN node deviation classifier
```

Neo4j is optional for graph database/query/visual analytics and is not required for the GUI or model training workflow.
