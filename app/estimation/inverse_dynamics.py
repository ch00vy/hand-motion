from __future__ import annotations

"""Inverse-dynamics effort indicators.

This module uses MuJoCo inverse dynamics to estimate q-space effort from
qpos/qvel/qacc. The result is an effort indicator for analysis features, not the
primary command used to move the hand model in the demo.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.estimation.effort_normalizer import finite_vector, normalize_abs_indicator


@dataclass
class InverseDynamicsConfig:
    disable_constraints: bool = True
    tau_clip: float = 0.0
    indicator_scale: float | None = None


@dataclass
class InverseDynamicsResult:
    qfrc_inverse: np.ndarray
    id_effort_indicator: np.ndarray
    method: str = "inverse_dynamics_effort_indicator"

    @property
    def summary(self) -> dict[str, float | str | int]:
        return {
            "method": self.method,
            "active": int(np.sum(self.id_effort_indicator > 0.01)),
            "qfrc_inverse_abs_mean": float(np.mean(np.abs(self.qfrc_inverse))) if self.qfrc_inverse.size else 0.0,
            "effort_max": float(np.max(self.id_effort_indicator)) if self.id_effort_indicator.size else 0.0,
            "effort_mean": float(np.mean(self.id_effort_indicator)) if self.id_effort_indicator.size else 0.0,
        }


def compute_inverse_dynamics_effort(
    *,
    model: Any,
    data: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    config: InverseDynamicsConfig | None = None,
) -> InverseDynamicsResult:
    cfg = config or InverseDynamicsConfig()
    try:
        import mujoco
    except Exception as exc:
        raise RuntimeError("MuJoCo is required for inverse-dynamics effort estimation.") from exc

    raw_model = getattr(model, "ptr", model)
    raw_data = getattr(data, "ptr", data)
    nq = int(getattr(raw_model, "nq", 0))
    nv = int(getattr(raw_model, "nv", 0))
    if nq <= 0 or nv <= 0:
        empty = np.zeros((0,), dtype=float)
        return InverseDynamicsResult(qfrc_inverse=empty, id_effort_indicator=empty)

    raw_data.qpos[:nq] = finite_vector(qpos, width=nq)
    raw_data.qvel[:nv] = finite_vector(qvel, width=nv)
    raw_data.qacc[:nv] = finite_vector(qacc, width=nv)
    if hasattr(raw_data, "ctrl") and int(getattr(raw_model, "nu", 0)) > 0:
        raw_data.ctrl[:] = 0.0
    if hasattr(raw_data, "act") and int(getattr(raw_model, "na", 0)) > 0:
        raw_data.act[:] = 0.0

    old_flags = int(raw_model.opt.disableflags)
    try:
        if bool(cfg.disable_constraints):
            raw_model.opt.disableflags = old_flags | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        mujoco.mj_forward(raw_model, raw_data)
        mujoco.mj_inverse(raw_model, raw_data)
        tau = np.asarray(raw_data.qfrc_inverse[:nv], dtype=float).copy()
    finally:
        raw_model.opt.disableflags = old_flags

    if float(cfg.tau_clip) > 0.0:
        tau = np.clip(tau, -float(cfg.tau_clip), float(cfg.tau_clip))
    tau = np.where(np.isfinite(tau), tau, 0.0)
    effort = normalize_abs_indicator(tau, scale=cfg.indicator_scale, clip=cfg.tau_clip)
    return InverseDynamicsResult(qfrc_inverse=tau.astype(float), id_effort_indicator=effort.astype(float))
