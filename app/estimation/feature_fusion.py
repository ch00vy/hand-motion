from __future__ import annotations

"""Feature fusion for node-level weakness indication.

The fused features combine control indicator, inverse-dynamics effort, and qpos
motion descriptors. They are inputs to graph/node inference, not standalone
diagnoses.
"""

from dataclasses import dataclass

import numpy as np

from app.estimation.effort_normalizer import finite_vector, normalize_abs_indicator


WEAKNESS_FEATURE_NAMES = [
    "pd_ctrl_indicator",
    "id_effort_indicator",
    "rom_score",
    "velocity_score",
    "completion_score",
    "delay_score",
]


@dataclass
class WeaknessFeatureBundle:
    x: np.ndarray
    feature_names: list[str]


def fuse_weakness_node_features(
    *,
    pd_ctrl_indicator: np.ndarray,
    id_effort_indicator: np.ndarray,
    rom_score: np.ndarray | float,
    velocity_score: np.ndarray | float,
    completion_score: np.ndarray | float,
    delay_score: np.ndarray | float,
    num_nodes: int | None = None,
) -> WeaknessFeatureBundle:
    width = int(num_nodes or max(np.asarray(pd_ctrl_indicator).size, np.asarray(id_effort_indicator).size, 1))
    columns = [
        _as_node_indicator(pd_ctrl_indicator, width),
        _as_node_indicator(id_effort_indicator, width),
        _as_node_indicator(rom_score, width),
        _as_node_indicator(velocity_score, width),
        _as_node_indicator(completion_score, width),
        _as_node_indicator(delay_score, width),
    ]
    x = np.stack(columns, axis=1).astype(np.float32)
    return WeaknessFeatureBundle(x=x, feature_names=list(WEAKNESS_FEATURE_NAMES))


def qpos_motion_scores(
    qpos_sequence: np.ndarray,
    *,
    reference_range: np.ndarray | None = None,
    target_qpos: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute ROM, velocity, completion, and delay indicators from qpos."""

    qpos = np.asarray(qpos_sequence, dtype=float)
    if qpos.ndim == 1:
        qpos = qpos.reshape(1, -1)
    if qpos.ndim != 2 or qpos.shape[1] == 0:
        empty = np.zeros((0,), dtype=float)
        return {"rom_score": empty, "velocity_score": empty, "completion_score": empty, "delay_score": empty}

    rom = np.ptp(qpos, axis=0)
    if reference_range is not None:
        ref = np.maximum(np.abs(finite_vector(reference_range, width=qpos.shape[1])), 1e-6)
        rom_score = np.clip(rom / ref, 0.0, 1.0)
    else:
        rom_score = normalize_abs_indicator(rom)

    velocity = np.diff(qpos, axis=0) if qpos.shape[0] > 1 else np.zeros_like(qpos)
    velocity_score = normalize_abs_indicator(np.mean(np.abs(velocity), axis=0))

    if target_qpos is not None:
        start_error = np.abs(finite_vector(target_qpos, width=qpos.shape[1]) - qpos[0])
        end_error = np.abs(finite_vector(target_qpos, width=qpos.shape[1]) - qpos[-1])
        completion_score = np.clip(1.0 - (end_error / np.maximum(start_error, 1e-6)), 0.0, 1.0)
    else:
        completion_score = rom_score.copy()

    speed = np.linalg.norm(velocity, axis=1) if velocity.size else np.zeros((0,), dtype=float)
    if speed.size and float(np.max(speed)) > 1e-9:
        onset_idx = int(np.argmax(speed > (0.15 * float(np.max(speed)))))
        delay = np.full((qpos.shape[1],), onset_idx / max(1, speed.size), dtype=float)
    else:
        delay = np.ones((qpos.shape[1],), dtype=float)

    return {
        "rom_score": np.clip(rom_score, 0.0, 1.0),
        "velocity_score": np.clip(velocity_score, 0.0, 1.0),
        "completion_score": np.clip(completion_score, 0.0, 1.0),
        "delay_score": np.clip(delay, 0.0, 1.0),
    }


def _as_node_indicator(values: np.ndarray | float, width: int) -> np.ndarray:
    if np.isscalar(values):
        return np.full((width,), float(values), dtype=float)
    return np.clip(finite_vector(values, width=width), 0.0, 1.0)
