"""
realtime/realtime_io.py
========================
STUB — Fase 1b: Leap IO, hand→qpos mapper, Canvas, tracking state.

TODO: Pindahkan blok-blok berikut dari app/main.py ke file ini:

  - Baris ~3383–3434 : Runtime Safety constants + helpers:
                         AUTO_RESET_ON_TRACK_LOSS, TRACK_LOSS_RESET_SECONDS,
                         update_listener_hud(), clear_listener_outputs(),
                         reset_runtime_tracking(), get_tracking_loss_reset_reason(),
                         get_invalid_qpos_reset_reason()
  - Baris ~3438–3520 : Tracking Geometry:
                         vector_to_np(), quaternion_to_yaw(), unit(), direction(),
                         vector_angle_rad(), vector_angle_deg(),
                         extract_palm_basis_from_hand(), blend_coupled_finger_flexion(),
                         leap_hand_to_joint_angles()
  - Baris ~3524–3557 : dataclass RealtimeTrackingState
  - Baris ~3562–4186 : Constants + class LeapToMyoSuiteMapper:
                         QPOS_SIZE, QPOS_SMOOTH_ALPHA, JOINT_TUNING,
                         TRACKING_TUNING_PRESETS, DEFAULT_TRACKING_TUNING_PRESET,
                         MapperDebugSnapshot, LeapToMyoSuiteMapper
  - Baris ~4190–4325 : class LeapDataListener + setup_leap_motion_device()
  - Baris ~4330–4460 : class Canvas (OpenCV visualizer Leap)
  - Baris ~3666–3693 : TRACKING_TUNING_PRESETS, DEFAULT_TRACKING_TUNING_PRESET
  - Baris ~5xxx–5xxx : apply_tracking_tuning_preset(), print_hand_mapping_debug()

Cara menggunakan stub ini sebelum split selesai:
    from app.realtime.monolithic_runtime_backend import LeapToMyoSuiteMapper  # noqa
"""
from __future__ import annotations

try:
    from app.realtime.monolithic_runtime_backend import (  # type: ignore[attr-defined]
        LeapToMyoSuiteMapper,
        LeapDataListener,
        Canvas,
        MapperDebugSnapshot,
        RealtimeTrackingState,
        setup_leap_motion_device,
        apply_tracking_tuning_preset,
        reset_runtime_tracking,
        update_listener_hud,
        clear_listener_outputs,
        get_tracking_loss_reset_reason,
        get_invalid_qpos_reset_reason,
        vector_to_np,
        quaternion_to_yaw,
        unit,
        direction,
        vector_angle_rad,
        vector_angle_deg,
        extract_palm_basis_from_hand,
        blend_coupled_finger_flexion,
        leap_hand_to_joint_angles,
        JOINT_TUNING,
        TRACKING_TUNING_PRESETS,
        DEFAULT_TRACKING_TUNING_PRESET,
        QPOS_SIZE,
        QPOS_SMOOTH_ALPHA,
        FINGER_BASE_IDX,
        AUTO_RESET_ON_TRACK_LOSS,
        TRACK_LOSS_RESET_SECONDS,
        TRACK_REACQUIRE_RESET_SECONDS,
        MIN_RESET_INTERVAL_SECONDS,
        VISUALIZE_LEAP,
    )
except ImportError as e:
    raise ImportError(
        f"[realtime_io] Gagal import dari backend runtime: {e}\n"
        "Pastikan app/realtime/monolithic_runtime_backend.py tersedia sebelum split selesai."
    ) from e
