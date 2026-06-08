"""
app/offline_biomech/__init__.py
================================
Package offline biomekanik — jalur utama node classification.

Pipeline:
  1. offline_biomech_reference  : qpos → ctrl_ref (inverse dynamics + OSQP)
  2. build_reference_sequence   : ctrl_ref trials → reference graph sequence
  3. compare_trial_to_reference : trial vs reference → NodeDeviationResult

Contoh penggunaan cepat:
    from app.offline_biomech import (
        load_qpos_trial_csv,
        build_biomech_reference_ctrl,
        build_reference_pipeline,
        compare_trial_to_reference,
        generate_deviation_report,
        LABEL_WITHIN_REFERENCE, LABEL_ONSET_SHIFT, LABEL_TOO_WEAK,
        LABEL_TOO_HIGH, LABEL_NOT_SMOOTH,
    )
"""

from app.offline_biomech.offline_biomech_reference import (  # noqa: F401
    load_qpos_trial_csv,
    compute_qvel_from_qpos,
    compute_qacc_from_qvel,
    run_inverse_dynamics,
    solve_muscle_ctrl_from_qpos,
    build_biomech_reference_ctrl,
    process_recording_dir,
)

from app.offline_biomech.build_reference_sequence import (  # noqa: F401
    biomech_trial_to_graph_input,
    build_open_to_fist_sequence,
    build_reference_from_trials,
    build_reference_pipeline,
)

from app.offline_biomech.compare_trial_to_reference import (  # noqa: F401
    NodeDeviationResult,
    compare_trial_to_reference,
    generate_deviation_report,
    build_node_label_sequence,
    LABEL_WITHIN_REFERENCE,
    LABEL_ONSET_SHIFT,
    LABEL_TOO_WEAK,
    LABEL_TOO_HIGH,
    LABEL_NOT_SMOOTH,
    LABEL_TO_INT,
)

__all__ = [
    # offline_biomech_reference
    "load_qpos_trial_csv",
    "compute_qvel_from_qpos",
    "compute_qacc_from_qvel",
    "run_inverse_dynamics",
    "solve_muscle_ctrl_from_qpos",
    "build_biomech_reference_ctrl",
    "process_recording_dir",
    # build_reference_sequence
    "biomech_trial_to_graph_input",
    "build_open_to_fist_sequence",
    "build_reference_from_trials",
    "build_reference_pipeline",
    # compare_trial_to_reference
    "NodeDeviationResult",
    "compare_trial_to_reference",
    "generate_deviation_report",
    "build_node_label_sequence",
    "LABEL_WITHIN_REFERENCE",
    "LABEL_ONSET_SHIFT",
    "LABEL_TOO_WEAK",
    "LABEL_TOO_HIGH",
    "LABEL_NOT_SMOOTH",
    "LABEL_TO_INT",
]
