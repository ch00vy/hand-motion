# Archived Experiments

These scripts are kept for reference only. They are not part of the final
main architecture.

Archived here:
- realtime predictor training / evaluation
- live fast-predictor runtime-log analysis
- teacher-vs-runtime comparison experiments
- teacher replay / visualization / mj_forward validators kept as historical
  validation tools
- short-horizon teacher sweeps that are not the canonical offline path

Use the canonical entrypoints instead:
- `scripts/realtime/run_live_realtime_ctrl.py`
- `scripts/offline_biomech/build_biomech_ctrl_reference.py`
- `scripts/offline_biomech/build_graph_sequence_dataset.py`
- `scripts/offline_biomech/build_reference_sequence.py`
- `scripts/offline_biomech/compare_trial_to_reference.py`
- `scripts/offline_biomech/run_node_deviation_pipeline.py`
