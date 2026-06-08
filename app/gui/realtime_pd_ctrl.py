from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Iterable

import numpy as np


@dataclass
class RealtimePDDynamicCtrlConfig:
    """Lightweight realtime ctrl display estimator.

    This is intentionally pragmatic: it estimates muscle ctrl values from the
    current model state and target qpos using PD drive. It is for realtime
    preview/display only, not a replacement for offline inverse dynamics + OSQP.
    """

    kp: float = 1.35
    kd: float = 0.055
    gain: float = 0.95
    deadband: float = 0.012
    ema_alpha: float = 0.42
    max_step: float = 0.16
    qvel_clip: float = 20.0
    drive_clip: float = 2.5


class RealtimePDDynamicCtrlEstimator:
    """Map qpos/qvel PD drive to actuator ctrl for realtime visualization.

    The estimator uses actuator names where available to route flexion/extension
    drive to likely finger muscle groups. It keeps the realtime loop stable and
    fast, while the research-grade ctrl_ref remains the offline biomech solver.
    """

    def __init__(
        self,
        *,
        actuator_names: Iterable[str],
        config: RealtimePDDynamicCtrlConfig | None = None,
    ) -> None:
        self.actuator_names = [str(name or "") for name in actuator_names]
        self.config = config or RealtimePDDynamicCtrlConfig()
        self._prev_target: np.ndarray | None = None
        self._prev_time: float | None = None
        self._prev_ctrl: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_target = None
        self._prev_time = None
        self._prev_ctrl = None

    def estimate(
        self,
        *,
        q_target: np.ndarray,
        qpos_current: np.ndarray,
        qvel_current: np.ndarray | None = None,
        qvel_target: np.ndarray | None = None,
        dt_override: float | None = None,
        now: float | None = None,
    ) -> tuple[np.ndarray, dict[str, float | str | int]]:
        now = float(time.perf_counter() if now is None else now)
        q_target_arr = np.asarray(q_target, dtype=float).reshape(-1)
        q_current_arr = np.asarray(qpos_current, dtype=float).reshape(-1)
        n = min(len(q_target_arr), len(q_current_arr))
        if n <= 0:
            return np.zeros((len(self.actuator_names),), dtype=float), {"method": "pd_dynamic", "active": 0}

        target = np.where(np.isfinite(q_target_arr[:n]), q_target_arr[:n], 0.0)
        current = np.where(np.isfinite(q_current_arr[:n]), q_current_arr[:n], 0.0)
        qvel_current_arr = self._fit_width(qvel_current, n)
        qvel_target_arr = self._target_qvel(target, qvel_target, dt_override, now)

        cfg = self.config
        q_error = target - current
        qvel_error = qvel_target_arr - qvel_current_arr
        drive = (float(cfg.kp) * q_error) + (float(cfg.kd) * qvel_error)
        drive = np.clip(np.where(np.isfinite(drive), drive, 0.0), -float(cfg.drive_clip), float(cfg.drive_clip))

        group_drive = self._finger_group_drive(drive)
        ctrl_raw = self._drive_to_ctrl(group_drive)
        ctrl = self._postprocess_ctrl(ctrl_raw)

        self._prev_target = target.copy()
        self._prev_time = now
        self._prev_ctrl = ctrl.copy()

        info = {
            "method": "pd_dynamic",
            "active": int(np.sum(ctrl > 0.01)),
            "q_error_norm": float(np.linalg.norm(q_error)),
            "qvel_error_norm": float(np.linalg.norm(qvel_error)),
            "drive_abs_mean": float(np.mean(np.abs(drive))) if drive.size else 0.0,
            "ctrl_max": float(np.max(ctrl)) if ctrl.size else 0.0,
            "ctrl_mean": float(np.mean(ctrl)) if ctrl.size else 0.0,
        }
        return ctrl, info

    def _target_qvel(
        self,
        target: np.ndarray,
        qvel_target: np.ndarray | None,
        dt_override: float | None,
        now: float,
    ) -> np.ndarray:
        cfg = self.config
        provided = self._fit_width(qvel_target, len(target), fill=None)
        if provided is not None:
            return np.clip(provided, -float(cfg.qvel_clip), float(cfg.qvel_clip))

        if self._prev_target is None or self._prev_time is None or len(self._prev_target) != len(target):
            return np.zeros_like(target)
        dt = float(dt_override or 0.0)
        if dt <= 0.0:
            dt = now - float(self._prev_time)
        dt = max(1e-4, dt)
        qvel = (target - self._prev_target) / dt
        qvel = np.where(np.isfinite(qvel), qvel, 0.0)
        return np.clip(qvel, -float(cfg.qvel_clip), float(cfg.qvel_clip))

    @staticmethod
    def _fit_width(values: np.ndarray | None, width: int, *, fill: float | None = 0.0) -> np.ndarray | None:
        if values is None:
            if fill is None:
                return None
            return np.full((int(width),), float(fill), dtype=float)
        arr = np.asarray(values, dtype=float).reshape(-1)
        out = np.full((int(width),), 0.0, dtype=float)
        out[: min(len(out), len(arr))] = arr[: min(len(out), len(arr))]
        return np.where(np.isfinite(out), out, 0.0)

    def _finger_group_drive(self, drive: np.ndarray) -> dict[str, float]:
        groups = self._qpos_groups(len(drive))
        values: dict[str, float] = {}
        for name, idx in groups.items():
            local = drive[idx] if len(idx) else np.empty((0,), dtype=float)
            values[f"{name}_flex"] = self._positive_mean(local)
            values[f"{name}_ext"] = self._negative_mean(local)

        values["global_flex"] = self._positive_mean(drive)
        values["global_ext"] = self._negative_mean(drive)
        return values

    @staticmethod
    def _positive_mean(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        pos = np.maximum(arr, 0.0)
        return float(np.mean(pos))

    @staticmethod
    def _negative_mean(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        neg = np.maximum(-arr, 0.0)
        return float(np.mean(neg))

    @staticmethod
    def _qpos_groups(width: int) -> dict[str, np.ndarray]:
        width = max(0, int(width))
        if width >= 23:
            ranges = {
                "wrist": range(0, 3),
                "thumb": range(3, 7),
                "index": range(7, 11),
                "middle": range(11, 15),
                "ring": range(15, 19),
                "little": range(19, 23),
            }
        else:
            splits = np.array_split(np.arange(width), 6)
            ranges = {
                "wrist": splits[0],
                "thumb": splits[1],
                "index": splits[2],
                "middle": splits[3],
                "ring": splits[4],
                "little": splits[5],
            }
        return {key: np.asarray(list(value), dtype=int) for key, value in ranges.items()}

    def _drive_to_ctrl(self, group_drive: dict[str, float]) -> np.ndarray:
        ctrl = np.zeros((len(self.actuator_names),), dtype=float)
        for idx, name in enumerate(self.actuator_names):
            name_upper = name.upper()
            finger = self._muscle_finger_key(name_upper)
            is_flexor = self._is_flexor(name_upper)
            is_extensor = self._is_extensor(name_upper)
            if is_flexor:
                value = self._group_value(group_drive, finger, "flex")
            elif is_extensor:
                value = self._group_value(group_drive, finger, "ext")
            else:
                value = max(float(group_drive.get("global_flex", 0.0)), float(group_drive.get("global_ext", 0.0))) * 0.35

            if self._is_intrinsic(name_upper):
                value *= 0.68
            if self._is_wrist_muscle(name_upper):
                value = max(
                    float(group_drive.get("wrist_flex", 0.0)),
                    float(group_drive.get("wrist_ext", 0.0)),
                    value * 0.55,
                )
            ctrl[idx] = float(value)
        return np.clip(float(self.config.gain) * ctrl, 0.0, 1.0)

    @staticmethod
    def _group_value(group_drive: dict[str, float], finger: str | None, direction: str) -> float:
        if finger is not None:
            return float(group_drive.get(f"{finger}_{direction}", 0.0))
        return float(group_drive.get(f"global_{direction}", 0.0))

    @staticmethod
    def _muscle_finger_key(name_upper: str) -> str | None:
        if any(token in name_upper for token in ("TH", "FPL", "EPL", "EPB", "APL", "OPP", "APB")):
            return "thumb"
        match = re.search(r"([2-5])", name_upper)
        if match is None:
            return None
        return {
            "2": "index",
            "3": "middle",
            "4": "ring",
            "5": "little",
        }.get(match.group(1))

    @staticmethod
    def _is_flexor(name_upper: str) -> bool:
        return any(token in name_upper for token in ("FDS", "FDP", "FPL", "FCR", "FCU", "PL", "OPP", "APB"))

    @staticmethod
    def _is_extensor(name_upper: str) -> bool:
        return any(token in name_upper for token in ("EDC", "EDM", "EIP", "EPL", "EPB", "ECR", "ECU", "APL"))

    @staticmethod
    def _is_intrinsic(name_upper: str) -> bool:
        return any(token in name_upper for token in ("LUM", "LU", "RI", "UI", "INT", "IO"))

    @staticmethod
    def _is_wrist_muscle(name_upper: str) -> bool:
        return any(token in name_upper for token in ("FCR", "FCU", "ECR", "ECU", "PL", "PT", "PQ"))

    def _postprocess_ctrl(self, ctrl_raw: np.ndarray) -> np.ndarray:
        ctrl = np.asarray(ctrl_raw, dtype=float).reshape(-1)
        ctrl = np.clip(np.where(np.isfinite(ctrl), ctrl, 0.0), 0.0, 1.0)
        deadband = max(0.0, float(self.config.deadband))
        if deadband > 0.0:
            ctrl[ctrl < deadband] = 0.0
        if self._prev_ctrl is not None and len(self._prev_ctrl) == len(ctrl):
            alpha = float(np.clip(self.config.ema_alpha, 0.0, 1.0))
            smoothed = alpha * ctrl + (1.0 - alpha) * self._prev_ctrl
            max_step = max(0.0, float(self.config.max_step))
            if max_step > 0.0:
                delta = np.clip(smoothed - self._prev_ctrl, -max_step, max_step)
                smoothed = self._prev_ctrl + delta
            ctrl = smoothed
        return np.clip(ctrl, 0.0, 1.0)
