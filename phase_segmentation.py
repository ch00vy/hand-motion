"""Reusable phase segmentation helpers for node-first sequence analysis.

These utilities tag motion phases for simulation-based gesture analysis.
They do not assign clinical meaning. The output is intended to support
reference-relative node feature extraction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


VALID_PHASE_SEGMENTATION_STRATEGIES = ("fixed_ratio", "velocity_based")
PHASE_LABELS = ("start", "transition", "near_target", "hold")


def extract_hold_segment(sequence, hold_ratio: float = 0.3):
    values = np.asarray(sequence)
    if values.ndim == 0 or values.shape[0] == 0:
        return values[:0]
    ratio = float(np.clip(hold_ratio, 0.0, 1.0))
    start_idx = max(0, int(np.floor((1.0 - ratio) * int(values.shape[0]))))
    return values[start_idx:]


def segment_motion_phases(
    sequence,
    *,
    timestamps: Sequence[float] | None = None,
    strategy: str = "fixed_ratio",
    velocity_threshold_quantile: float = 0.35,
):
    values = np.asarray(sequence, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    num_frames = int(values.shape[0]) if values.ndim >= 2 else 0
    if num_frames <= 0:
        return []

    phase_seq = np.full(num_frames, "transition", dtype=object)
    strategy_name = str(strategy or "fixed_ratio").strip().lower()
    if strategy_name not in VALID_PHASE_SEGMENTATION_STRATEGIES:
        strategy_name = "fixed_ratio"

    if num_frames == 1:
        phase_seq[0] = "hold"
        return phase_seq.tolist()

    if strategy_name == "velocity_based":
        deltas = np.diff(values, axis=0)
        dt = None
        if timestamps is not None:
            ts = np.asarray(list(timestamps), dtype=float).reshape(-1)
            if ts.shape[0] == num_frames:
                dt = np.diff(ts)
                dt = np.where(np.abs(dt) > 1e-6, dt, 1.0)
        if dt is None:
            dt = np.ones(max(1, num_frames - 1), dtype=float)
        speed = np.linalg.norm(deltas, axis=1) / np.maximum(dt, 1e-6)
        padded_speed = np.concatenate(([speed[0] if speed.size else 0.0], speed))
        speed_threshold = float(np.quantile(padded_speed, np.clip(velocity_threshold_quantile, 0.05, 0.95)))
        near_threshold = float(np.quantile(padded_speed, 0.55))
        phase_seq[: max(1, int(round(0.15 * num_frames)))] = "start"
        phase_seq[padded_speed <= near_threshold] = "near_target"
        phase_seq[padded_speed <= speed_threshold] = "hold"
        phase_seq[0] = "start"
        return phase_seq.tolist()

    start_end = max(1, int(round(0.15 * num_frames)))
    near_start = max(start_end, int(round(0.65 * num_frames)))
    hold_start = max(near_start, int(round(0.80 * num_frames)))
    phase_seq[:start_end] = "start"
    phase_seq[start_end:near_start] = "transition"
    phase_seq[near_start:hold_start] = "near_target"
    phase_seq[hold_start:] = "hold"
    return phase_seq.tolist()
