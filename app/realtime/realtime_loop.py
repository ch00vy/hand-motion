"""
realtime/realtime_loop.py
==========================
STUB — Fase 1c: Main loop, recorded-qpos validation, diagnostic runner.

TODO: Pindahkan blok-blok berikut dari app/main.py ke file ini:

  - Baris ~100–260   : class RollingRuntimeProfiler
  - Baris ~260–270   : dataclass RuntimeDiagnosticOptions
  - Baris ~270–310   : draw_runtime_profiler_overlay(), draw_recording_overlay()
  - Baris ~310–595   : Helper loop:
                         build_diagnostic_ctrl_state(), create_probe_env(),
                         create_myosuite_env(), create_controller_context(),
                         CtrlEstimationConfig, estimate_ctrl_signals(),
                         prepare_target_qpos(), apply_qpos_to_view_env(),
                         update_main_env_ctrl_view(), step_ctrl_env(),
                         reset_ctrl_env_state(), reset_controller_context_state(),
                         handle_runtime_key(), build_live_runtime_log_row(),
                         print_ctrl_summary()
  - Baris ~5600–6500 : Helper validation:
                         _load_recorded_qpos_trial_runtime(),
                         _estimate_recorded_qpos_kinematics_runtime(),
                         _set_full_state_runtime(), _forward_validate_sequence_runtime(),
                         _compute_recorded_validation_windows_runtime(),
                         _write_recorded_validation_summary_runtime(),
                         _build_substeps_from_dt_runtime()
  - Baris ~6500–7150 : def main(...)  ← loop utama Leap realtime
  - Baris ~7152–7572 : def run_recorded_qpos_runtime_validation(...)
  - Baris ~7575–7882 : _run_single_diagnostic_trial()
  - Baris ~7885–8019 : run_bottleneck_diagnostic()
  - Baris ~8022–8134 : _cli_main() + parse_runtime_args()

Cara menggunakan stub ini sebelum split selesai:
    from app.realtime.monolithic_runtime_backend import main  # noqa
"""
from __future__ import annotations

try:
    from app.realtime.monolithic_runtime_backend import (  # type: ignore[attr-defined]
        main,
        run_recorded_qpos_runtime_validation,
        run_bottleneck_diagnostic,
        RollingRuntimeProfiler,
        RuntimeDiagnosticOptions,
        draw_runtime_profiler_overlay,
        draw_recording_overlay,
        estimate_ctrl_signals,
        create_controller_context,
        prepare_target_qpos,
        build_diagnostic_ctrl_state,
        handle_runtime_key,
        build_live_runtime_log_row,
        print_ctrl_summary,
        step_ctrl_env,
        apply_qpos_to_view_env,
        update_main_env_ctrl_view,
        reset_ctrl_env_state,
        reset_controller_context_state,
        CtrlEstimationConfig,
        _merge_diagnostic_options,
        _synthetic_q_target,
    )
except ImportError as e:
    raise ImportError(
        f"[realtime_loop] Gagal import dari backend runtime: {e}\n"
        "Pastikan app/realtime/monolithic_runtime_backend.py tersedia sebelum split selesai."
    ) from e
