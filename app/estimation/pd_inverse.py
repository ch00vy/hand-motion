from __future__ import annotations

"""PD inverse estimates in qpos/qvel space.

In the final demo pipeline, qpos drives the model pose. The PD inverse result is
only a realtime control indicator for GUI sliders and graph display.
"""

from dataclasses import dataclass
import time

import numpy as np

from app.estimation.effort_normalizer import finite_vector, normalize_abs_indicator


@dataclass
class PDInverseConfig:
    kp: float = 1.35
    kd: float = 0.055
    qvel_clip: float = 20.0
    tau_clip: float = 2.5
    indicator_scale: float | None = None


@dataclass
class PDInverseResult:
    tau_pd: np.ndarray
    pd_ctrl_indicator: np.ndarray
    qpos_error: np.ndarray
    qvel_error: np.ndarray
    method: str = "pd_inverse_indicator"

    @property
    def summary(self) -> dict[str, float | str | int]:
        return {
            "method": self.method,
            "active": int(np.sum(self.pd_ctrl_indicator > 0.01)),
            "q_error_norm": float(np.linalg.norm(self.qpos_error)),
            "qvel_error_norm": float(np.linalg.norm(self.qvel_error)),
            "tau_pd_abs_mean": float(np.mean(np.abs(self.tau_pd))) if self.tau_pd.size else 0.0,
            "indicator_max": float(np.max(self.pd_ctrl_indicator)) if self.pd_ctrl_indicator.size else 0.0,
            "indicator_mean": float(np.mean(self.pd_ctrl_indicator)) if self.pd_ctrl_indicator.size else 0.0,
        }


class PDInverseEstimator:
    """Estimate q-space PD effort from current pose and target pose."""

    def __init__(self, config: PDInverseConfig | None = None) -> None:
        self.config = config or PDInverseConfig()
        self._prev_target: np.ndarray | None = None
        self._prev_time: float | None = None

    def reset(self) -> None:
        self._prev_target = None
        self._prev_time = None

    def estimate(
        self,
        *,
        qpos_target: np.ndarray,
        qpos_current: np.ndarray,
        qvel_current: np.ndarray | None = None,
        qvel_target: np.ndarray | None = None,
        dt_override: float | None = None,
        now: float | None = None,
    ) -> PDInverseResult:
        now_value = float(time.perf_counter() if now is None else now)
        target_raw = finite_vector(qpos_target)
        current_raw = finite_vector(qpos_current)
        width = min(target_raw.size, current_raw.size)
        if width <= 0:
            empty = np.zeros((0,), dtype=float)
            return PDInverseResult(tau_pd=empty, pd_ctrl_indicator=empty, qpos_error=empty, qvel_error=empty)

        target = target_raw[:width]
        current = current_raw[:width]
        current_vel = finite_vector(qvel_current if qvel_current is not None else [], width=width)
        target_vel = self._target_qvel(target, qvel_target, dt_override, now_value)

        cfg = self.config
        qpos_error = target - current
        qvel_error = target_vel - current_vel
        tau_pd = float(cfg.kp) * qpos_error + float(cfg.kd) * qvel_error
        if float(cfg.tau_clip) > 0.0:
            tau_pd = np.clip(tau_pd, -float(cfg.tau_clip), float(cfg.tau_clip))
        tau_pd = np.where(np.isfinite(tau_pd), tau_pd, 0.0)
        indicator = normalize_abs_indicator(tau_pd, scale=cfg.indicator_scale, clip=cfg.tau_clip)

        self._prev_target = target.copy()
        self._prev_time = now_value
        return PDInverseResult(
            tau_pd=tau_pd.astype(float),
            pd_ctrl_indicator=indicator.astype(float),
            qpos_error=qpos_error.astype(float),
            qvel_error=qvel_error.astype(float),
        )

    def _target_qvel(
        self,
        target: np.ndarray,
        qvel_target: np.ndarray | None,
        dt_override: float | None,
        now: float,
    ) -> np.ndarray:
        cfg = self.config
        if qvel_target is not None:
            provided = finite_vector(qvel_target, width=len(target))
            return np.clip(provided, -float(cfg.qvel_clip), float(cfg.qvel_clip))

        if self._prev_target is None or self._prev_time is None or len(self._prev_target) != len(target):
            return np.zeros_like(target)
        dt = float(dt_override or 0.0)
        if dt <= 0.0:
            dt = now - float(self._prev_time)
        dt = max(1e-4, dt)
        qvel = (target - self._prev_target) / dt
        return np.clip(np.where(np.isfinite(qvel), qvel, 0.0), -float(cfg.qvel_clip), float(cfg.qvel_clip))
