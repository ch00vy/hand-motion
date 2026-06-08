# Final System Architecture

## Realtime

Runtime flow:

`Leap Motion -> hand pose / joint rotations -> qpos -> t_des -> ctrl mapping -> MyoSuite/MuJoCo -> realtime hand motion`

Main entrypoint:

```powershell
.\venv\Scripts\python.exe scripts\realtime\run_live_realtime_ctrl.py --main-sim-mode estimated_ctrl_step --ctrl-mode control
```

Implementation map:

- `scripts/realtime/run_live_realtime_ctrl.py`: command wrapper.
- `app/realtime/main_realtime.py`: Python entrypoint.
- `app/main.py`: thin compatibility shim.
- `app/realtime/monolithic_runtime_backend.py`: transitional backend moved out of `app/main.py`.
- `app/realtime/realtime_controller.py`, `app/realtime/realtime_io.py`, `app/realtime/realtime_loop.py`: target module boundaries for the ongoing extraction.

Realtime constraints:

- No inverse dynamics in the live loop.
- No OSQP in the live loop.
- No fast predictor as the main path.
- Realtime graph is optional visualization, not the analytical core.

## Offline Biomech

Analysis flow:

`recorded qpos -> qvel/qacc -> inverse dynamics -> tau_target/qfrc_inverse -> OSQP/bounded muscle solve -> ctrl_ref -> openfist graph sequence -> normal reference sequence -> trial comparison -> GCN node classification`

Main commands:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\build_biomech_ctrl_reference.py --input-path records\open_to_fist_normal --output-path artifacts\offline_biomech_ctrl_reference --reference-mode dynamic_with_static_prior --gesture-select open_to_fist
```

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\build_graph_sequence_dataset.py --normal-input artifacts\offline_biomech_ctrl_reference\accepted_trials --output-dir artifacts\open_to_fist_graph_sequence_dataset --sequence-length 50 --window-size 5 --window-stride 1 --progress-mode linear --edge-mode dynamic_short_window --edge-threshold 0.02 --include-edge-sign
```

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\build_reference_sequence.py --dataset-dir artifacts\open_to_fist_graph_sequence_dataset --output-dir artifacts\open_to_fist_reference_sequence --cohort normal
```

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\compare_trial_to_reference.py --trial-file artifacts\open_to_fist_graph_sequence_dataset\trials\TRIAL_ID.json --reference-dir artifacts\open_to_fist_reference_sequence --output-dir artifacts\trial_vs_reference
```

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\train_gcn_node_classifier.py --dataset-dir artifacts\open_to_fist_graph_sequence_dataset --reference-dir artifacts\open_to_fist_reference_sequence --output-dir artifacts\gcn_node_classifier --epochs 60
```

Implementation map:

- `scripts/offline_biomech/build_biomech_ctrl_reference.py`: canonical qpos to ctrl_ref command.
- `scripts/offline_biomech/biomech_reference_backend.py`: clean-name facade over legacy inverse-dynamics backend.
- `app/offline_biomech/offline_biomech_reference.py`: clean inverse-dynamics + OSQP/PGD primitives.
- `app/graph_sequence_baseline.py`: graph sequence/reference/deviation baseline.
- `app/gcn_node_classifier.py`: offline GCN node classifier baseline.

Node semantics:

- Nodes are muscles/actuators.
- Node labels are deviation classes: `within_reference`, `onset_shift`, `too_weak`, `too_high`, `not_smooth`.
- GCN stays in the offline analytical layer.

## Compatibility Notes

Some root `scripts/build_teacher_*` files remain as compatibility backends.
They are not user-facing main workflows. The canonical command wrappers use
biomech/reference naming and create clean summary aliases where possible.
