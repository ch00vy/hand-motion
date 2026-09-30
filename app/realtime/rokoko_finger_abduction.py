from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


ABDUCTION_QPOS_INDICES = (8, 12, 16, 20)
FINGER_NAMES = ("index", "middle", "ring", "little")


@dataclass(frozen=True)
class FingerAbductionConfig:
    qpos_indices: tuple[int, int, int, int] = ABDUCTION_QPOS_INDICES
    signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    limits: tuple[tuple[float, float], ...] = (
        (-0.7, 0.7),
        (-0.7, 0.7),
        (-0.7, 0.7),
        (-0.7, 0.7),
    )


class RokokoFingerAbductionMapper:
    """Map Rokoko proximal finger directions to MCP abduction qpos values."""

    def __init__(self, config: FingerAbductionConfig | None = None) -> None:
        self.config = config or FingerAbductionConfig()
        self._reference_angles: np.ndarray | None = None

    def reset(self) -> None:
        self._reference_angles = None

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            raise ValueError("finger direction is too short")
        return vector / norm

    @classmethod
    def _hand_basis(cls, points: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        wrist = points["wrist"]
        finger_vectors = [points[f"{name}_proximal"] - wrist for name in FINGER_NAMES]
        palm_axis = cls._unit(np.mean(finger_vectors, axis=0))
        palm_normal = cls._unit(np.cross(finger_vectors[0], finger_vectors[-1]))
        return palm_axis, palm_normal

    @classmethod
    def _angles(cls, points: Mapping[str, np.ndarray]) -> np.ndarray:
        wrist = points["wrist"]
        palm_axis, palm_normal = cls._hand_basis(points)
        angles = []
        for name in FINGER_NAMES:
            direction = cls._unit(points[f"{name}_proximal"] - wrist)
            sine = float(np.dot(palm_normal, np.cross(palm_axis, direction)))
            cosine = float(np.dot(palm_axis, direction))
            angles.append(float(np.arctan2(sine, cosine)))
        return np.asarray(angles, dtype=float)

    def map_points(self, points: Mapping[str, np.ndarray], *, calibrate: bool = False) -> np.ndarray:
        current = self._angles(points)
        if self._reference_angles is None or calibrate:
            self._reference_angles = current.copy()
        mapped = (current - self._reference_angles) * np.asarray(self.config.signs, dtype=float)
        for index, (lower, upper) in enumerate(self.config.limits):
            mapped[index] = float(np.clip(mapped[index], lower, upper))
        return mapped

    def map_into_qpos(self, qpos: np.ndarray, points: Mapping[str, np.ndarray], *, calibrate: bool = False) -> np.ndarray:
        result = np.asarray(qpos, dtype=float).reshape(-1).copy()
        if result.size < max(self.config.qpos_indices) + 1:
            raise ValueError("qpos vector is too short for finger abduction mapping")
        mapped = self.map_points(points, calibrate=calibrate)
        for qpos_index, value in zip(self.config.qpos_indices, mapped):
            result[qpos_index] = value
        return result
