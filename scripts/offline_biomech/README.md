# Offline Biomech Workflow

Final offline scope:

`recorded qpos -> qvel/qacc -> inverse dynamics -> tau_target -> OSQP/bounded muscle solve -> ctrl_ref -> graph sequence -> reference sequence -> node deviation -> GCN`

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

The compatibility backend still writes some legacy filenames/columns containing
`teacher`, but these outputs are treated as biomechanical `ctrl_ref`.
