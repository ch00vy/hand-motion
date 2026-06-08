"""
realtime/realtime_controller.py
================================
STUB — Fase 1a: RealtimeQPController + semua konstanta dan helper QP.

TODO: Pindahkan blok-blok berikut dari app/main.py ke file ini:

Blok yang perlu dipindah (referensi baris di main.py v1):
  - Baris ~737–828   : Konstanta controller (KP_GAIN, KD_GAIN, CTRL_BASELINE, dll.)
  - Baris ~828–1070  : Helper matematika QP:
                         _get_raw(), _get_raw_model_data(), _dense_moment(),
                         _gain_bias(), _clip_to_limits(), _torque_weights(),
                         _compute_hold_torque(), _joint_family(), _muscle_family(),
                         _normalize_nonnegative(), _sanitize_ctrl(), _find_actuated_dofs(),
                         build_hold_prior(), _extract_hold_flexor_diagnostics(),
                         _normalize_hold_exercise_name(), FINGER_BASE_IDX
  - Baris ~1072–1779 : class RealtimeQPController
  - Baris ~1780–2098 : class DynamicEdgeScorer + DynamicEdgeScoringConfig
  - Baris ~2099–2360 : Hand layout constants:
                         HAND_LAYOUT_REFERENCE_WIDTH, MUSCLE_VISUAL_ZONES,
                         NORMALIZED_NODE_LAYOUT, HAND_LAYOUT_COORDS,
                         build_hand_layout_positions(), scaled_zone_label_positions()
  - Baris ~2365–2808 : MuscleModelContext, MuscleGraphVisualizationConfig,
                         build_muscle_graph_spec(), build_muscle_nodes(), build_muscle_edges(),
                         extract_muscle_signals()
  - Baris ~2880–3379 : class RealtimeMuscleGraph
  - Baris ~4500–5600 : MuscleGraphCanvas + draw_muscle_graph_canvas() (rendering OpenCV)

Cara menggunakan stub ini sebelum split selesai:
    from app.realtime.monolithic_runtime_backend import RealtimeQPController  # noqa

Setelah split selesai, ganti import di seluruh codebase:
    from app.realtime.realtime_controller import (
        RealtimeQPController,
        DynamicEdgeScorer,
        DynamicEdgeScoringConfig,
        RealtimeMuscleGraph,
        MuscleModelContext,
        MuscleGraphVisualizationConfig,
        build_muscle_graph_spec,
        build_hand_layout_positions,
        ...
    )
"""
from __future__ import annotations

# Temporary re-export dari backend runtime selama split belum selesai
try:
    from app.realtime.monolithic_runtime_backend import (  # type: ignore[attr-defined]
        RealtimeQPController,
        DynamicEdgeScoringConfig,
        DynamicEdgeScorer,
        MuscleModelContext,
        MuscleGraphVisualizationConfig,
        RealtimeMuscleGraph,
        build_muscle_graph_spec,
        build_muscle_nodes,
        build_muscle_edges,
        extract_muscle_signals,
        create_muscle_model_context,
        build_hand_layout_positions,
        scaled_zone_label_positions,
        zone_for_muscle,
        resolve_dominant_muscle_subset,
        apply_visualization_preset,
        apply_visualization_mode,
        HAND_LAYOUT_REFERENCE_WIDTH,
        HAND_LAYOUT_REFERENCE_HEIGHT,
        MUSCLE_VISUAL_ZONES,
        ZONE_DISPLAY_NAMES,
        FUNCTIONAL_GROUPS,
        FALLBACK_MUSCLE_NAMES,
        GRAPH_EXCLUDED_NODE_NAMES,
        NODE_FEATURE_NAMES,
        EDGE_TYPES,
    )
except ImportError as e:
    raise ImportError(
        f"[realtime_controller] Gagal import dari backend runtime: {e}\n"
        "Pastikan app/realtime/monolithic_runtime_backend.py tersedia sebelum split selesai."
    ) from e
