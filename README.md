# Hand Motion Biomechanical Analysis Tool

Trial-based hand motion analysis system for:

- recording `open_to_fist` hand motion trials,
- replaying recorded kinematics in MuJoCo/MyoSuite,
- estimating biomechanical muscle control references offline,
- building graph sequences over muscle/actuator nodes,
- running binary GCN node deviation classification.

The current stable scope is **trial-based analysis**, not live diagnosis.

## Final System Flow

### Realtime / GUI

```text
Leap / hand tracking -> qpos -> record qpos.csv + meta.json
```

The Tkinter GUI is the control and analysis hub. MuJoCo replay and Leap preview are treated as external/viewer components.

### Offline Biomechanics

```text
recorded qpos
-> qvel/qacc
-> inverse dynamics
-> OSQP muscle redundancy solve
-> ctrl_ref
-> open_to_fist graph sequence
-> normal reference sequence
-> binary GCN node deviation classifier
```

### Output Labels

The active classifier is binary:

- `within_reference`
- `deviating`

Multiclass labels are still experimental and are not the main workflow.

## Setup

Create a local environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Ultraleap SDK, local records, generated artifacts, and model checkpoints are intentionally ignored by git.

## Run GUI

```powershell
.\venv\Scripts\python.exe -m scripts.gui.run_trial_replay_and_classifier_gui
```

If using a newly trained model/reference, set:

```powershell
$env:HAND_GCN_MODEL_PATH=(Resolve-Path artifacts\final_binary_gcn_run_gui_plus_previous\checkpoints\best_final_binary_gcn.pt).Path
$env:HAND_REFERENCE_DIR=(Resolve-Path artifacts\retrain_gui_plus_previous\reference_sequence\reference).Path
$env:HAND_CANONICAL_DATASET_DIR=(Resolve-Path artifacts\retrain_gui_plus_previous\graph_sequence_dataset).Path
```

## Prepare Combined Normal Dataset

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage prepare_combined --date-prefix 20260603_ --clean-prepare
```

## Build Control Reference

Full run:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage ctrl_ref_combined --date-prefix 20260603_ --clean-ctrl-ref
```

Quick debug subset:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage ctrl_ref_combined --date-prefix 20260603_ --max-combined-trials 5 --clean-ctrl-ref
```

## Train Final Binary GCN

After `ctrl_ref_combined` finishes:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage all_after_combined_ctrl_ref
```

Expected final checkpoint:

```text
artifacts/final_binary_gcn_run_gui_plus_previous/checkpoints/best_final_binary_gcn.pt
```

Training figures are generated automatically after the training stage:

```text
artifacts/final_binary_gcn_run_gui_plus_previous/final_figures/final_binary_accuracy.png
artifacts/final_binary_gcn_run_gui_plus_previous/final_figures/final_binary_f1.png
artifacts/final_binary_gcn_run_gui_plus_previous/final_figures/final_binary_loss.png
artifacts/final_binary_gcn_run_gui_plus_previous/final_figures/final_binary_confusion_matrix.png
artifacts/final_binary_gcn_run_gui_plus_previous/final_figures/final_binary_confusion_matrix_normalized.png
```

To regenerate figures from an existing completed run without retraining:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage figures_only
```

## Add Abnormal Validation Records

User-declared abnormal records must be kept out of the normal reference pool. Stage them separately:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage prepare_abnormal --abnormal-date-prefix 20260604_ --clean-prepare
```

Build their biomechanical `ctrl_ref`:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage ctrl_ref_abnormal --abnormal-date-prefix 20260604_ --clean-ctrl-ref
```

After the normal model/reference already exists, build abnormal graph validation and evaluate the final binary GCN:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage all_abnormal_after_ctrl_ref --abnormal-date-prefix 20260604_ --clean-graph
```

One-shot full retrain plus abnormal validation:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage all_combined_with_abnormal_validation --date-prefix 20260603_ --abnormal-date-prefix 20260604_ --clean-prepare --clean-ctrl-ref --clean-graph --clean-reference
```

Abnormal validation output:

```text
artifacts/retrain_gui_plus_previous/abnormal_validation_eval/
```

## GitHub Notes

Generated/local folders are ignored:

- `venv/`
- `artifacts/`
- `reports/`
- `records/`
- `Ultraleap/`
- `leapc-python-bindings/`
- `proposal_figures/`

Keep source code, documentation, and reproducible commands in git. Keep private datasets and generated models outside git, or publish checkpoints separately as releases if needed.
