"""Node-first temporal feature extraction for muscle activation sequences.

These features summarize how each node behaves over time relative to a target
gesture context. They support simulation-based deviation analysis and are not
clinical measurements.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from phase_segmentation import extract_hold_segment, segment_motion_phases


NODE_TEMPORAL_FEATURE_NAMES = (
    "mean_activation_full",
    "std_activation_full",
    "peak_activation",
    "time_to_peak",
    "mean_transition",
    "mean_hold",
    "activation_drop_after_peak",
    "auc_activation",
)


def _safe_auc(values: np.ndarray, timestamps: np.ndarray | None) -> float:
    if values.size <= 0:
        return 0.0
    if timestamps is None or timestamps.shape[0] != values.shape[0]:
        return float(np.trapz(values))
    return float(np.trapz(values, timestamps))


def build_node_temporal_features(ctrl_analysis_seq, *, timestamps: Sequence[float] | None = None, phase_seq: Sequence[str] | None = None):
    values = np.asarray(ctrl_analysis_seq, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.empty((0, len(NODE_TEMPORAL_FEATURE_NAMES)), dtype=float), list(NODE_TEMPORAL_FEATURE_NAMES)

    num_frames, num_nodes = values.shape
    ts = None
    if timestamps is not None:
        ts = np.asarray(list(timestamps), dtype=float).reshape(-1)
        if ts.shape[0] != num_frames:
            ts = None
    phases = list(phase_seq) if phase_seq is not None else segment_motion_phases(values, timestamps=timestamps, strategy="fixed_ratio")
    if len(phases) != num_frames:
        phases = segment_motion_phases(values, timestamps=timestamps, strategy="fixed_ratio")

    transition_mask = np.asarray([phase in {"transition", "near_target"} for phase in phases], dtype=bool)
    hold_mask = np.asarray([phase == "hold" for phase in phases], dtype=bool)
    if not np.any(transition_mask):
        transition_mask[:] = True
    if not np.any(hold_mask):
        hold_mask[max(0, int(round(0.7 * num_frames))) :] = True

    feature_rows: list[list[float]] = []
    for node_idx in range(num_nodes):
        node_values = values[:, node_idx]
        peak_idx = int(np.argmax(node_values)) if node_values.size else 0
        peak_value = float(node_values[peak_idx]) if node_values.size else 0.0
        if ts is not None and ts.size:
            time_to_peak = float(max(0.0, ts[peak_idx] - ts[0]))
        else:
            time_to_peak = float(peak_idx)
        hold_values = node_values[hold_mask]
        transition_values = node_values[transition_mask]
        drop_after_peak = float(peak_value - np.mean(hold_values)) if hold_values.size else 0.0
        feature_rows.append(
            [
                float(np.mean(node_values)),
                float(np.std(node_values)),
                peak_value,
                time_to_peak,
                float(np.mean(transition_values)) if transition_values.size else 0.0,
                float(np.mean(hold_values)) if hold_values.size else 0.0,
                drop_after_peak,
                _safe_auc(node_values, ts),
            ]
        )
    return np.asarray(feature_rows, dtype=float), list(NODE_TEMPORAL_FEATURE_NAMES)
