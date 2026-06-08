# Dynamic Temporal GCN Node Classifier

## Status

Dynamic Temporal GCN is now the active node-deviation classifier path for trial analysis.

The task remains binary node classification:

- `within_reference`
- `deviating`

The node is still a muscle / actuator node, not a joint node.

## Input

The model consumes one `open_to_fist` graph sequence per trial:

- `node_feature_seq[t]`: per-muscle node features at each progress-normalized step
- `edge_weight_seq[t]`: dynamic muscle relation edge weights at each step
- `edges`: anatomical / functional muscle relations

This is different from the previous vanilla GCN, which used summary node features and one static averaged adjacency matrix.

## Active GUI Inference Path

The GUI analysis controller now defaults to:

```text
recorded qpos
-> biomech ctrl_ref
-> graph sequence
-> dynamic Temporal GCN
-> per-muscle binary node result
```

Default checkpoint:

```text
artifacts/temporal_gcn_prototype_dynamic/checkpoints/best_temporal_gcn.pt
```

Environment override:

```powershell
$env:HAND_TEMPORAL_GCN_MODEL_PATH="C:\path\to\best_temporal_gcn.pt"
```

## Training Command

Use the explicit migration-ready dynamic command:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\train_dynamic_temporal_gcn.py --epochs 80 --patience 12 --force-clean
```

Default output:

```text
artifacts/dynamic_temporal_gcn_node_classifier/
```

The wrapper automatically uses:

- `--model-variant dynamic`
- `--add-progress-feature`
- `--temporal-pooling attention`
- `--class-weighting`
- figure generation under `figures/`

The older prototype command still exists for comparison experiments:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\train_temporal_gcn_prototype.py --model-variant dynamic
```

## Interpretation

The model output answers:

```text
Which muscle nodes look normal vs problematic across the open_to_fist sequence?
```

It does not classify the gesture itself.

Use the output as analytical support alongside:

- baseline deviation score
- reference sequence comparison
- control replay quality
- biomechanical plausibility

Do not treat the model as a final clinical diagnostic result.

## Abnormal Validation Records

Abnormal records are staged separately from normal reference data. They are used
for validation/evaluation and should not enter the normal reference pool.

Current abnormal staging supports multiple date prefixes:

```powershell
.\venv\Scripts\python.exe scripts\offline_biomech\run_retrain_with_gui_records.py --stage prepare_abnormal --clean-prepare
```

The current default includes:

- `20260604_`
- `20260606_`
