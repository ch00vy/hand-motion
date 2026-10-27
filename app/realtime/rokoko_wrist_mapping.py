from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


WRIST_QPOS_INDICES = (0, 1, 2)


@dataclass(frozen=True)
class WristMappingConfig:
    """Rokoko-to-MyoSuite wrist convention and neutral orientation."""

    qpos_indices: tuple[int, int, int] = WRIST_QPOS_INDICES
    # Live Rokoko palm rotation runs opposite to MyoHand pro_sup.
    # Apply this to the calibrated delta so the neutral pose stays unchanged.
    pronation_sign: float = -1.0
    # Sideways deviation retains the model convention. Bending toward/away
    # from the palm needs the live-stream correction, independently of twist.
    deviation_sign: float = 1.0
    flexion_sign: float = -1.0
    # MyoHand zero-pose hinge axes expressed in its anatomical palm basis
    # (radial, longitudinal, cross(radial, longitudinal)). These include the
    # oblique model axes rather than assuming three orthogonal Euler axes.
    axes: tuple[tuple[float, float, float], ...] = (
        (-0.03650157, -0.99885089, 0.03105690),
        (0.30824213, 0.07906504, -0.94801662),
        (0.95115903, -0.00798288, 0.30859809),
    )
    neutral_qpos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-1.5708, 1.5708),
        (-0.174533, 0.436332),
        (-0.785398, 0.785398),
    )


class RokokoWristMapper:
    """Convert Rokoko right-hand quaternions into three wrist qpos values."""

    def __init__(self, config: WristMappingConfig | None = None) -> None:
        self.config = config or WristMappingConfig()
        self._reference: np.ndarray | None = None
        self._basis = np.eye(3)
        self._last_angles = np.zeros(3)

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
        self._basis = np.eye(3)
        self._last_angles = np.zeros(3)

    def calibrate(self, quaternion_xyzw: Sequence[float], basis: np.ndarray) -> None:
        """Anchor world rotation to the palm axes measured in the neutral pose."""
        current = self._validated_quaternion(quaternion_xyzw)
        basis = np.asarray(basis, dtype=float)
        if basis.shape != (3, 3) or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6) or np.linalg.det(basis) < 0:
            raise ValueError("wrist basis must be a right-handed orthonormal matrix")
        self._reference = current.copy()
        self._basis = basis.copy()
        self._last_angles = np.zeros(3)

    @staticmethod
    def _validated_quaternion(quaternion_xyzw: Sequence[float]) -> np.ndarray:
        current = np.asarray(quaternion_xyzw, dtype=float).reshape(-1)
        if current.shape != (4,) or not np.all(np.isfinite(current)) or np.linalg.norm(current) < 1e-9:
            raise ValueError("wrist quaternion must have four finite values and nonzero length")
        return current / np.linalg.norm(current)

    def map_rotation(self, quaternion_xyzw: Sequence[float] | np.ndarray, *, calibrate: bool = False) -> np.ndarray:
        current = self._validated_quaternion(quaternion_xyzw)
        if self._reference is None or calibrate:
            self._reference = current.copy()
            self._last_angles = np.zeros(3)
        world_delta = Rotation.from_quat(current) * Rotation.from_quat(self._reference).inv()
        relative = Rotation.from_matrix(self._basis.T @ world_delta.as_matrix() @ self._basis)
        axes = np.asarray(self.config.axes, dtype=float)
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        neutral = np.asarray(self.config.neutral_qpos, dtype=float)
        limits = np.asarray(self.config.limits)
        signs = np.asarray([self.config.pronation_sign, self.config.deviation_sign, self.config.flexion_sign])
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("wrist direction signs must be -1 or 1")
        # Deviation limits are asymmetric. Transform the fit bounds as well as
        # the output so reversing direction retains the full model range.
        angle_bounds = (limits - neutral[:, None]) * signs[:, None]
        lower_bounds = np.min(angle_bounds, axis=1)
        upper_bounds = np.max(angle_bounds, axis=1)
        if relative.magnitude() < 1e-10:
            self._last_angles = np.zeros(3)
            return neutral.copy()

        def residual(angles: np.ndarray) -> np.ndarray:
            predicted = Rotation.identity()
            for axis, angle in zip(axes, angles):
                predicted = predicted * Rotation.from_rotvec(axis * angle)
            return (predicted.inv() * relative).as_rotvec()

        fit = least_squares(residual, np.clip(self._last_angles, lower_bounds, upper_bounds),
                            bounds=(lower_bounds, upper_bounds), max_nfev=20)
        self._last_angles = fit.x.copy()
        mapped = neutral + signs * fit.x
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
