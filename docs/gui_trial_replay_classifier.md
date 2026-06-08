# Trial Replay + Binary GCN Classifier GUI

## Status

This GUI implements the trial-based workflow:

```text
Record -> Save/Load -> Replay Recorded Kinematics for qpos verification
Record -> Save/Load -> Run Analysis -> Replay Estimated Control for ctrl_ref validation
Run Analysis -> Show node results
```

It does not change the training pipeline, realtime control path, Neo4j export, or multiclass labeling.

## GUI Backend

The current implementation uses standard-library `tkinter` so it can run without adding GUI dependencies. It is styled as a dark scientific dashboard and acts as the control + analysis hub only.

## Final Layout

- Left column: trial control, external viewer controls/status, trial summary, top deviating nodes.
- Center column: classifier plots.
  - Bar plot: deviation score per node.
  - Heatmap: node deviation over progress-normalized movement.
- Right column: node result table.
- Bottom: status bar with runtime/model/replay status.

The GUI intentionally does not embed Leap or MuJoCo viewers.

Color semantics:

- Blue: primary actions.
- Amber: analysis / processing.
- Green: `within_reference`.
- Red: `deviating` / stop / error.

## Run Command

```powershell
.\venv\Scripts\python.exe -m scripts.gui.run_trial_replay_and_classifier_gui
```

Smoke-test imports and paths without opening the GUI:

```powershell
.\venv\Scripts\python.exe -m scripts.gui.run_trial_replay_and_classifier_gui --smoke-test
```

Standalone Leap viewer:

```powershell
.\venv\Scripts\python.exe scripts\dev\view_leap_camera_overlay.py --scale 1.6
```

The standalone viewer defaults to skeleton-only. This is the preferred preview mode because it uses tracking coordinates directly and does not depend on image projection/rectification.

If IR image debugging is needed:

```powershell
.\venv\Scripts\python.exe scripts\dev\view_leap_camera_overlay.py --show-image --scale 1.6
```

## Main Files

- `scripts/gui/run_trial_replay_and_classifier_gui.py`: GUI launcher.
- `app/gui/main_window.py`: desktop window and button workflow.
- `app/gui/state.py`: explicit GUI state machine.
- `app/gui/plot_controller.py`: node deviation bar plot and node-vs-progress heatmap.
- `app/gui/leap_preview_controller.py`: external Leap viewer process controller.
- `app/gui/leap_camera_controller.py`: Leap tracking/image adapter used by the standalone viewer.
- `app/gui/recorder_controller.py`: qpos recording buffer and Leap qpos provider.
- `app/gui/trial_loader.py`: `qpos.csv` / `meta.json` save-load helpers.
- `app/gui/mujoco_replay_controller.py`: external MyoSuite/MuJoCo replay controller.
- `app/gui/inference_controller.py`: offline preprocessing and final binary GCN inference.

## Trial Storage Format

New recordings are saved as:

```text
records/
  <subject_id>/
    <timestamp>_<motion_type>/
      qpos.csv
      meta.json
      result.json
      result.csv
      analysis/
        final_binary_gcn/
```

`qpos.csv` includes:

- `frame_index`
- `t`
- `progress_percent`
- `subject_id`
- `trial_id`
- `exercise_name`
- `motion_type`
- `qpos_0 ... qpos_N`

`meta.json` includes:

- `trial_id`
- `subject_id`
- `motion_type`
- `num_frames`
- `qpos_dim`
- `dt`
- `created_at`
- `source_mode`

## Workflow

1. Open the GUI.
2. Set `subject_id`.
3. Keep `motion_type=open_to_fist` unless testing another transition.
4. Optionally click `Open Leap Viewer` to verify tracking.
5. For repeated recording, set `Auto-stop after` to a fixed duration such as `4.0` seconds.
6. Click `Start Recording` or `Record Next Trial` to begin saving qpos frames. This does not open MuJoCo replay.
7. Click `Stop Recording`, or let auto-stop save the trial automatically.
8. The saved trial is added to the `Recording Session` list. You can immediately click `Record Next Trial` again without restarting the GUI.
9. Use `Load Selected Session Trial` to reload any trial recorded during the current GUI session.
10. Confirm the active `qpos.csv` path appears in the GUI.
11. Optionally click `Open Replay Viewer` or `Replay Recorded Kinematics` to inspect the saved qpos trial in the external MuJoCo viewer.
12. Click `Run Analysis`.
13. After analysis completes, `Control available` becomes `YES`.
14. Optionally click `Replay Estimated Control` to run forward simulation from ctrl_ref.
15. Read the GUI output:
    - trial summary
    - top deviating nodes
    - node result table
    - node deviation bar plot
    - node-vs-progress heatmap

## Repeated Data Collection

The GUI is intended to stay open while collecting multiple trials.

- `Record Next Trial` starts a fresh buffer for the next trial.
- `Auto-stop after` saves consistent-duration trials without needing manual timing.
- `Run analysis after each saved trial` is optional; keep it off if you want fast batch recording first, then analyze selected trials later.
- The session list only tracks trials recorded since the GUI was opened; the files remain saved under `records/<subject_id>/<timestamp>_<motion_type>/`.
- Keyboard shortcuts:
  - `Ctrl+R`: start recording / record next trial
  - `Ctrl+S`: stop recording
  - `Ctrl+L`: load an existing trial

Recommended collection flow for repeated `open_to_fist` data:

1. Set `subject_id`.
2. Set `motion_type=open_to_fist`.
3. Enable `Auto-stop after` and choose a duration, for example `4.0`.
4. Put the hand in the open start pose.
5. Press `Ctrl+R` or click `Record Next Trial`.
6. Perform open-to-fist once.
7. Wait for auto-save.
8. Return to the open pose and repeat from step 5.

## Classifier Used

Default model:

```text
artifacts/final_binary_gcn_run/checkpoints/best_final_binary_gcn.pt
```

Default reference:

```text
artifacts/open_to_fist_reference_sequence_latest_short_horizon/reference
```

The GUI analysis path runs:

1. qpos trial -> offline biomech ctrl reference
2. ctrl_ref -> graph sequence
3. graph sequence + reference sequence -> node features/deviation features
4. final binary vanilla GCN inference

By default, GUI analysis now sends all recorded frames to the offline biomech ctrl_ref builder. The later graph sequence stage is still resampled to the configured progress-normalized sequence length, so full-frame ctrl_ref generation and fixed-length GCN input are separate steps.

## Replay Modes

The GUI separates two replay modes because they answer different research questions.

### Kinematic Replay

Button: `Replay Recorded Kinematics`

Purpose:

- verify the recorded trial
- inspect whether qpos mapping is plausible
- replay recorded movement exactly as kinematic data

Behavior:

- applies each recorded `qpos` frame directly to the MyoSuite/MuJoCo hand model
- calls `mj_forward/mj_render`
- does not validate whether estimated muscle controls reproduce the movement

### Control Replay

Button: `Replay Estimated Control`

Purpose:

- validate estimated muscle activity / ctrl_ref
- test whether ctrl_ref can move the musculoskeletal model toward the recorded target movement

Behavior:

- requires analysis output
- loads `analysis/final_binary_gcn/ctrl_ref/accepted_trials/*.csv`
- resets the model to the recorded initial qpos
- runs forward simulation with ctrl_ref
- reports qpos RMSE against recorded qpos in the status text when possible

## Replay Control Semantics

- `Open Replay Viewer`: opens the external MuJoCo viewer with a neutral model only.
- `Replay Recorded Kinematics`: opens the viewer if needed, then renders the active trial qpos sequence.
- `Replay Estimated Control`: opens the viewer if needed, then drives forward simulation from ctrl_ref.
- `Play`, `Pause`, `Reset Active Replay`: control whichever replay mode is active.
- Recording and replay are independent. `Start Recording` and `Stop Recording` never auto-open MuJoCo.

## Current Limitations

- Leap viewer is external, not embedded in Tkinter.
- MuJoCo viewer is external, not embedded in Tkinter.
- Kinematic replay is driven by direct qpos application (`qpos -> mj_forward/mj_render`).
- Control replay is driven by ctrl_ref forward simulation and is the relevant replay mode for validating estimated muscle activity.
- Control replay depends on successful analysis output. If ctrl_ref is missing, the GUI warns the user to run analysis first.
- `Run Analysis` invokes offline biomechanical preprocessing, so it can take time for real trials.
- The classifier is binary only: `within_reference` vs `deviating`.
- Multiclass classification is intentionally not enabled in this GUI.
