from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


WRIST_QPOS_INDICES = (0, 1, 2)


@dataclass(frozen=True)
class WristMappingConfig:
    """Initial wrist convention for the MyoSuite qpos layout."""

    euler_order: str = "xyz"
    qpos_indices: tuple[int, int, int] = WRIST_QPOS_INDICES
    signs: tuple[float, float, float] = (1.0, -1.0, 1.0)
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-1.5708, 1.5708),
        (-1.5708, 1.5708),
        (-1.5708, 1.5708),
    )


class RokokoWristMapper:
    """Convert Rokoko right-hand quaternions into three wrist qpos values."""

    def __init__(self, config: WristMappingConfig | None = None) -> None:
        self.config = config or WristMappingConfig()
        self._reference: np.ndarray | None = None

    @staticmethod
    def quaternion_from_node(node: Mapping[str, object]) -> np.ndarray:
        values = np.asarray(
            [node["x"], node["y"], node["z"], node["w"]],
            dtype=float,
        )
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("rightHand rotation must contain finite x, y, z, w values")
        norm = float(np.linalg.norm(values))
        if norm <= 1e-9:
            raise ValueError("rightHand rotation quaternion has zero length")
        return values / norm

    def reset(self) -> None:
        self._reference = None

    def map_rotation(self, quaternion_xyzw: Sequence[float] | np.ndarray, *, calibrate: bool = False) -> np.ndarray:
        current = np.asarray(quaternion_xyzw, dtype=float).reshape(-1)
        if current.shape != (4,) or not np.all(np.isfinite(current)):
            raise ValueError("wrist quaternion must have four finite values")
        current = current / max(float(np.linalg.norm(current)), 1e-12)
        if self._reference is None or calibrate:
            self._reference = current.copy()
        relative = Rotation.from_quat(self._reference).inv() * Rotation.from_quat(current)
        angles = np.asarray(relative.as_euler(self.config.euler_order, degrees=False), dtype=float)
        mapped = angles * np.asarray(self.config.signs, dtype=float)
        for index, (lower, upper) in enumerate(self.config.limits):
            mapped[index] = float(np.clip(mapped[index], lower, upper))
        return mapped

    def map_into_qpos(self, qpos: np.ndarray, quaternion_xyzw: Sequence[float] | np.ndarray, *, calibrate: bool = False) -> np.ndarray:
        result = np.asarray(qpos, dtype=float).reshape(-1).copy()
        if result.size < max(self.config.qpos_indices) + 1:
            raise ValueError("qpos vector is too short for wrist mapping")
        mapped = self.map_rotation(quaternion_xyzw, calibrate=calibrate)
        for qpos_index, value in zip(self.config.qpos_indices, mapped):
            result[qpos_index] = value
        return result
