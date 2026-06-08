# GUI Final Architecture

## Final Position

The desktop system is finalized as a trial-based research tool:

```text
Record -> Save/Load -> Kinematic Replay for qpos verification
Record -> Save/Load -> Analyze -> Control Replay for ctrl_ref validation
Analyze -> Show Result
```

The stable component split is:

1. Tkinter GUI: control and analysis hub.
2. Leap viewer: optional external tracking monitor.
3. MuJoCo viewer: external MyoSuite/MuJoCo live preview and trial replay monitor.
4. Analysis backend: offline biomechanical preprocessing and final binary vanilla GCN inference.
5. Neo4j: optional graph database/export target, not required for GUI operation.

The GUI does not embed Leap or MuJoCo. MuJoCo live preview is opened automatically at startup when available, while trial replay remains a manual action.

## Main GUI

File: `app/gui/main_window.py`

Responsibilities:

- subject and motion metadata
- `Start Recording` / `Stop Recording`
- `Load Existing Trial`
- `Run Analysis`
- `Open Replay Viewer`
- `Replay Recorded Kinematics`
- `Replay Estimated Control`
- optional `Close Replay Viewer`
- trial path, trial id, frame count, status
- external Leap viewer process control/status
- external MuJoCo replay process control/status
- current replay mode: `NONE`, `KINEMATIC`, or `CONTROL`
- control availability: `YES` only after ctrl_ref analysis output exists
- trial summary
- top deviating node list
- binary node result table
- bar plot of node deviation scores
- heatmap of node deviation over normalized progress
- state machine and button gating

The GUI intentionally focuses on trial control and analytical output.

## Leap Viewer

Files:

- `scripts/dev/view_leap_camera_overlay.py`
- `app/gui/leap_preview_controller.py`
- `app/gui/leap_camera_controller.py`

Responsibilities:

- external skeleton-only tracking monitor by default
- loaded from `C:\Users\amar\hand\leapc-python-bindings` when available
- used to verify tracking before or during recording
- not embedded in Tkinter
- not required for analysis of an already recorded trial

The IR image overlay path remains a debug path only:

```powershell
.\venv\Scripts\python.exe scripts\dev\view_leap_camera_overlay.py --show-image --scale 1.6
```

## MuJoCo Viewer

File: `app/gui/mujoco_replay_controller.py`

Responsibilities:

- external MyoSuite/MuJoCo window
- opened automatically for live preview when the GUI starts, if MyoSuite is available
- live preview uses mapped Leap qpos and displays the realtime ctrl estimate for monitoring
- `Replay Recorded Kinematics`: qpos-driven replay for trial verification
- `Replay Estimated Control`: ctrl_ref-driven forward simulation for validating estimated muscle activity

MuJoCo is not embedded in Tkinter. Recording is separate from replay. The automatic MuJoCo window is a live preview/monitor, not a replay of a saved trial.

## Analysis Backend

Files:

- `app/gui/inference_controller.py`
- `app/gui/trial_loader.py`
- `app/gui/recorder_controller.py`
- `app/gui/plot_controller.py`
- `app/gui/state.py`

Responsibilities:

- save `qpos.csv` and `meta.json`
- load existing trial folders
- run offline biomech preprocessing
- build graph sequence for the trial
- compare trial to reference sequence
- run final binary vanilla GCN
- write `result.json` and `result.csv`
- display node labels:
  - `within_reference`
  - `deviating`

## Neo4j

Neo4j remains optional. It is not a GUI dependency.

Intended role:

- store graph sequence / reference / deviation results
- query node and edge deviation
- visual analytics outside the main Tkinter workflow

If Neo4j is unavailable, the GUI still records, replays, analyzes, and saves results locally.

## State Machine

### `IDLE`

No current trial is active.

Allowed actions:

- `Start Recording`
- `Load Existing Trial`
- `Open Replay Viewer`
- `Open Leap Viewer`

MuJoCo live preview opens automatically if possible. Saved-trial replay is still not loaded unless explicitly requested.

`Open Replay Viewer` remains available in every state. It opens/focuses the external MuJoCo window, but it does not load recording buffers or start analysis.

### `RECORDING`

`Start Recording` only starts saving qpos frames to the recorder buffer.

Disallowed:

- loading a trial
- running analysis
- replaying another trial

`Stop Recording`:

- writes `qpos.csv`
- writes `meta.json`
- returns and displays the saved path
- sets the saved trial as the current trial
- moves GUI state to `TRIAL_READY`

It does not automatically open MuJoCo replay.

### `TRIAL_READY`

A current trial exists from recording or loading.

Allowed actions:

- `Run Analysis`
- `Open Replay Viewer`
- `Replay Recorded Kinematics`
- `Play` / `Pause` / `Reset Replay`

If `Replay Recorded Kinematics` is clicked while the viewer is closed, the viewer is opened first and the recorded qpos sequence is rendered.

`Replay Estimated Control` is disabled until analysis output with ctrl_ref is available.

### `ANALYZING`

The offline analysis pipeline is running.

Buttons that could mutate the active trial or replay state are disabled.

### `ANALYSIS_DONE`

The result table, summary, bar plot, and heatmap are available.

The current trial can still be replayed from the external MuJoCo viewer controls.

### `KINEMATIC_REPLAY`

The external MuJoCo viewer is showing the recorded qpos sequence.

Meaning:

- verifies the recorded trial kinematics
- does not validate whether ctrl_ref can reproduce the movement

### `CONTROL_REPLAY`

The external MuJoCo viewer is running forward simulation from ctrl_ref.

Meaning:

- validates the estimated muscle/control sequence
- starts from the recorded initial qpos
- compares simulated qpos against recorded qpos in the status text when possible

## Main Command

```powershell
.\venv\Scripts\python.exe -m scripts.gui.run_trial_replay_and_classifier_gui
```

Smoke test:

```powershell
.\venv\Scripts\python.exe -m scripts.gui.run_trial_replay_and_classifier_gui --smoke-test
```
