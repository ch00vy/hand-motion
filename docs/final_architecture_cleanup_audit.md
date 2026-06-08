# Final Architecture Cleanup Audit

Tanggal audit: 2026-05-12

Arsitektur final yang berlaku:

- Realtime: `Leap / joint rotations / qpos -> t_des -> ctrl mapping -> MyoSuite/MuJoCo`.
- Offline biomech: `recorded qpos -> qvel/qacc -> inverse dynamics -> tau_target -> OSQP/bounded muscle solve -> ctrl_ref -> graph sequence -> reference sequence -> node deviation -> GCN`.

Realtime tidak memakai inverse dynamics, OSQP, atau fast predictor sebagai jalur utama. Offline biomech adalah satu-satunya jalur untuk reference, node deviation, dan node classification. Neo4j tidak menjadi core runtime/training; posisinya hanya optional storage/query/visual analytics.

## Canonical Commands

Realtime final:

```powershell
.\venv\Scripts\python.exe scripts\realtime\run_live_realtime_ctrl.py --main-sim-mode estimated_ctrl_step --ctrl-mode control
```

Offline biomech ctrl reference:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\build_biomech_ctrl_reference.py --input-path records\open_to_fist_normal --output-path artifacts\offline_biomech_ctrl_reference --reference-mode dynamic_with_static_prior --gesture-select open_to_fist
```

Open-to-fist graph sequence dataset:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\build_graph_sequence_dataset.py --normal-input artifacts\offline_biomech_ctrl_reference\accepted_trials --output-dir artifacts\open_to_fist_graph_sequence_dataset --sequence-length 50 --window-size 5 --window-stride 1 --progress-mode linear --edge-mode dynamic_short_window --edge-threshold 0.02 --include-edge-sign
```

Normal reference sequence:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\build_reference_sequence.py --dataset-dir artifacts\open_to_fist_graph_sequence_dataset --output-dir artifacts\open_to_fist_reference_sequence --cohort normal
```

Trial vs reference:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\compare_trial_to_reference.py --trial-file artifacts\open_to_fist_graph_sequence_dataset\trials\TRIAL_ID.json --reference-dir artifacts\open_to_fist_reference_sequence --output-dir artifacts\trial_vs_reference
```

Optional GCN node classification:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\train_gcn_node_classifier.py --dataset-dir artifacts\open_to_fist_graph_sequence_dataset --reference-dir artifacts\open_to_fist_reference_sequence --output-dir artifacts\gcn_node_classifier --epochs 60
```

## KEEP_MAIN_REALTIME

- `scripts/realtime/run_live_realtime_ctrl.py`: canonical live entrypoint.
- `scripts/realtime/README.md`: realtime command and constraints.
- `app/realtime/main_realtime.py`: canonical Python realtime entrypoint.
- `app/main.py`: thin compatibility shim that re-exports the realtime backend for old imports.
- `app/realtime/monolithic_runtime_backend.py`: transitional backend moved out of `app/main.py`.
- `app/realtime_runtime_support.py`: runtime env/render helpers used by `app/main.py`.
- `app/runtime_recording.py`: recording support for live captures.
- `app/runtime_logger.py`: runtime CSV logging.
- `app/realtime/*`: modular realtime namespace/stubs for the ongoing split from `app/main.py`.

Notes:

- `run_live_realtime_ctrl.py` defaults to `--input-mode leap_live`, `--ctrl-runtime-mode legacy_solver`, and `--main-sim-mode estimated_ctrl_step`.
- `app/main.py` is now short. The remaining large runtime implementation is isolated behind `app/realtime/monolithic_runtime_backend.py`.

## KEEP_MAIN_OFFLINE_BIOMECH

- `scripts/offline_biomech/build_biomech_ctrl_reference.py`: canonical qpos-to-ctrl_ref entrypoint.
- `scripts/offline_biomech/biomech_reference_backend.py`: clean-name facade over the legacy trajectory inverse-dynamics backend.
- `scripts/offline_biomech/build_graph_sequence_dataset.py`: canonical open-to-fist sequence graph dataset entrypoint.
- `scripts/offline_biomech/build_reference_sequence.py`: canonical normal reference sequence builder.
- `scripts/offline_biomech/compare_trial_to_reference.py`: canonical trial-vs-reference deviation analyzer.
- `scripts/offline_biomech/run_node_deviation_pipeline.py`: legacy-compatible node deviation/classification wrapper.
- `scripts/offline_biomech/train_gcn_node_classifier.py`: offline GCN node classifier trainer.
- `scripts/offline_biomech/README.md`: offline command map.
- `app/offline_biomech/offline_biomech_reference.py`: clean offline inverse-dynamics + OSQP/PGD ctrl_ref primitives.
- `app/offline_biomech/build_reference_sequence.py`: biomech trial to graph/reference helpers.
- `app/offline_biomech/compare_trial_to_reference.py`: node deviation labels and report helpers.
- `app/graph_sequence_baseline.py`: progress-normalized graph sequence baseline.
- `app/graph_sequence/*`: graph sequence namespace wrapper.
- `app/gcn_node_classifier.py`: offline GCN baseline.
- `scripts/build_open_to_fist_graph_sequence_dataset.py`: compatibility backend for graph sequence dataset.
- `scripts/build_reference_graph_sequence.py`: compatibility backend for reference sequence.
- `scripts/compare_trial_to_reference_sequence.py`: compatibility backend for trial comparison.
- `offline_node_classifier.py`, `reference_profile.py`, `deviation_analysis.py`, `node_features.py`, `phase_segmentation.py`: existing offline analysis helpers.

Notes:

- `app/graph_sequence_baseline.py` accepts `ctrl_ref_`, legacy `ctrl_teacher_`, `ctrl_target_`, `ctrl_analysis_`, and `ctrl_` columns so offline biomech reference CSVs can feed the graph sequence builder directly.
- The `teacher` column prefix is treated as backward-compatible storage for biomechanical `ctrl_ref`, not as a realtime teacher/predictor workflow.

## KEEP_OPTIONAL_REFERENCE

- `scripts/build_teacher_from_trajectory_inverse.py`: compatibility backend used by the canonical biomech wrapper. It is not a user-facing main command.
- `scripts/build_teacher_inverse_dynamics.py`: compatibility core for inverse dynamics and bounded muscle solve helpers.
- `scripts/build_teacher_from_qpos.py`: older offline reference/helper still imported by the trajectory backend.
- `scripts/audit_joint_actuator_sparsity.py`: optional actuator/muscle audit.
- `offline_u_classifier.py`: older offline classifier helper; not the final GCN workflow.
- `app/realtime_graph.py`: optional realtime visualization/reference.
- `app/archive_legacy/*`: archived predictor/static optimization references.
- `graph_pipeline/*`: reference documentation for graph concepts.

## ARCHIVE_OR_REMOVE

Archived experiment scripts:

- `scripts/archive_experiments/analyze_live_runtime_log.py`
- `scripts/archive_experiments/build_realtime_ctrl_teacher_dataset.py`
- `scripts/archive_experiments/compare_inverse_vs_previous_solver.py`
- `scripts/archive_experiments/compare_teacher_modes.py`
- `scripts/archive_experiments/compare_teacher_vs_current_solver.py`
- `scripts/archive_experiments/evaluate_realtime_ctrl_pipeline.py`
- `scripts/archive_experiments/replay_teacher_controls.py`
- `scripts/archive_experiments/sweep_short_horizon_teacher.py`
- `scripts/archive_experiments/train_fast_ctrl_predictor.py`
- `scripts/archive_experiments/validate_teacher_with_mj_forward.py`
- `scripts/archive_experiments/view_teacher_musculoskeletal_replay.py`

Archived snapshots:

- `archive/legacy_snapshots/aa.py`
- `archive/legacy_snapshots/bb.py`
- `archive/legacy_snapshots/main1.py`

Reason:

- These files represent fast-predictor runtime experiments, teacher-vs-runtime comparisons, old replay/visual validation tools, or code snapshots. They are useful for historical reference only and should not be shown as the main workflow.

## Changes Applied

- Moved legacy teacher replay, mj_forward validation, visual replay, and live predictor log analyzer into `scripts/archive_experiments/`.
- Added `scripts/README.md`, `scripts/realtime/README.md`, and `scripts/offline_biomech/README.md`.
- Moved the large realtime implementation from `app/main.py` to `app/realtime/monolithic_runtime_backend.py`.
- Added `app/realtime/README.md` and `docs/final_system_architecture.md`.
- Added `scripts/offline_biomech/biomech_reference_backend.py` as the clean-name facade for active offline commands.
- Updated `scripts/offline_biomech/build_biomech_ctrl_reference.py` to use clean default output/report paths.
- Added clean aliases `--reference-mode` and `--reference-prior-mode` while keeping old hidden `--teacher-*` arguments for backward compatibility.
- Added clean output aliases: `biomech_reference_trial_summary.csv`, `biomech_reference_dataset_summary.json`, and `biomech_reference_dataset_combined.csv` when source files exist.
- Updated `scripts/build_teacher_from_trajectory_inverse.py` messaging to describe it as a compatibility backend for offline biomech reference.
- Updated graph sequence input loading so `ctrl_ref_` and legacy `ctrl_teacher_` columns are valid sources.

## Checked Files

- `app/main.py`
- `app/realtime/monolithic_runtime_backend.py`
- `app/realtime_runtime_support.py`
- `app/runtime_logger.py`
- `app/runtime_recording.py`
- `app/realtime/*`
- `app/offline_biomech/*`
- `app/graph_sequence_baseline.py`
- `app/graph_sequence/*`
- `app/gcn_node_classifier.py`
- `scripts/realtime/run_live_realtime_ctrl.py`
- `scripts/offline_biomech/*`
- `scripts/build_teacher_from_trajectory_inverse.py`
- `scripts/build_teacher_inverse_dynamics.py`
- `scripts/build_teacher_from_qpos.py`
- `scripts/build_open_to_fist_graph_sequence_dataset.py`
- `scripts/build_reference_graph_sequence.py`
- `scripts/compare_trial_to_reference_sequence.py`
- `scripts/archive_experiments/*`
- `offline_node_classifier.py`
- `offline_u_classifier.py`
- `reference_profile.py`
- `deviation_analysis.py`
- `node_features.py`
- `phase_segmentation.py`

## Remaining Decisions

- The entrypoint is now thin, but `app/realtime/monolithic_runtime_backend.py` is still large. The next cleanup should extract real implementations into `realtime_controller.py`, `realtime_io.py`, and `realtime_loop.py` instead of re-exporting.
- Root compatibility backends named `build_teacher_*` still exist because the canonical offline wrapper depends on them. They should be renamed internally in a later focused pass after compatibility tests.
- Per-trial output filenames from the compatibility backend still include `teacher_*`. Summary aliases now have clean names, but a later migration can rename per-trial CSVs after downstream consumers are checked.
- Neo4j integration was not changed because it is not currently needed for the core offline pipeline.
