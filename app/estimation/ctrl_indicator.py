from __future__ import annotations

"""Realtime slider/control indicators derived from PD inverse estimates."""

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from app.estimation.effort_normalizer import finite_vector
from app.estimation.pd_inverse import PDInverseConfig, PDInverseEstimator


@dataclass
class RealtimePDDynamicCtrlConfig(PDInverseConfig):
    """Configuration for GUI slider indicators.

    The resulting vector is a display indicator. It is not written back as the
    primary actuator command for moving the model.
    """

    gain: float = 0.95
    deadband: float = 0.012
    ema_alpha: float = 0.42
    max_step: float = 0.16
    drive_clip: float | None = None
    hold_gain: float = 0.28
    hold_deadband: float = 0.035
    finger_extensor_gain: float = 1.35
    finger_extensor_cocontraction: float = 0.16
    wrist_cocontraction: float = 0.08

    def __post_init__(self) -> None:
        if self.drive_clip is not None:
            self.tau_clip = float(self.drive_clip)


class RealtimePDDynamicCtrlEstimator:
    """Map qpos/qvel PD inverse estimates to muscle slider indicators."""

    def __init__(
        self,
        *,
        actuator_names: Iterable[str],
        config: RealtimePDDynamicCtrlConfig | None = None,
    ) -> None:
        self.actuator_names = [str(name or "") for name in actuator_names]
        self.config = config or RealtimePDDynamicCtrlConfig()
        self.pd_inverse = PDInverseEstimator(self.config)
        self._prev_ctrl: np.ndarray | None = None

    def reset(self) -> None:
        self.pd_inverse.reset()
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
        result = self.pd_inverse.estimate(
            qpos_target=q_target,
            qpos_current=qpos_current,
            qvel_current=qvel_current,
            qvel_target=qvel_target,
            dt_override=dt_override,
            now=now,
        )
        if result.tau_pd.size == 0:
            return np.zeros((len(self.actuator_names),), dtype=float), {"method": "pd_inverse_indicator", "active": 0}

        movement_drive = self._finger_group_drive(result.tau_pd)
        hold_drive = self._posture_hold_drive(q_target)
        group_drive = self._combine_movement_and_hold(movement_drive, hold_drive)
        ctrl_raw = self._drive_to_ctrl(group_drive)
        ctrl = self._postprocess_ctrl(ctrl_raw)
        self._prev_ctrl = ctrl.copy()

        info = dict(result.summary)
        info.update(
            {
                "method": "pd_inverse_plus_isometric_hold_indicator",
                "ctrl_max": float(np.max(ctrl)) if ctrl.size else 0.0,
                "ctrl_mean": float(np.mean(ctrl)) if ctrl.size else 0.0,
                "active": int(np.sum(ctrl > 0.01)),
                "hold_drive_max": float(max(hold_drive.values())) if hold_drive else 0.0,
                "movement_drive_max": float(max(movement_drive.values())) if movement_drive else 0.0,
            }
        )
        return ctrl, info

    def _finger_group_drive(self, drive: np.ndarray) -> dict[str, float]:
        groups = self._qpos_groups(len(drive))
        values: dict[str, float] = {}
        for name, idx in groups.items():
            local = drive[idx] if len(idx) else np.empty((0,), dtype=float)
            values[f"{name}_flex"] = self._positive_mean(local)
            values[f"{name}_ext"] = self._negative_mean(local)
        if len(drive) >= 3:
            values.update(self._wrist_direction_drives(drive[:3]))
        values["global_flex"] = self._positive_mean(drive)
        values["global_ext"] = self._negative_mean(drive)
        return values

    @staticmethod
    def _wrist_direction_drives(wrist_drive: np.ndarray) -> dict[str, float]:
        wrist = finite_vector(wrist_drive, width=3)
        pro_sup = float(wrist[0])
        deviation = float(wrist[1])
        flexion = float(wrist[2])
        return {
            "pro_sup_pos": max(0.0, pro_sup),
            "pro_sup_neg": max(0.0, -pro_sup),
            "wrist_radial": max(0.0, deviation),
            "wrist_ulnar": max(0.0, -deviation),
            "wrist_flex": max(0.0, flexion),
            "wrist_ext": max(0.0, -flexion),
        }

    def _posture_hold_drive(self, q_target: np.ndarray) -> dict[str, float]:
        q = finite_vector(q_target)
        values: dict[str, float] = {
            "wrist_flex": 0.0,
            "wrist_ext": 0.0,
            "thumb_flex": 0.0,
            "thumb_ext": 0.0,
            "index_flex": 0.0,
            "index_ext": 0.0,
            "middle_flex": 0.0,
            "middle_ext": 0.0,
            "ring_flex": 0.0,
            "ring_ext": 0.0,
            "little_flex": 0.0,
            "little_ext": 0.0,
            "pro_sup_pos": 0.0,
            "pro_sup_neg": 0.0,
            "wrist_radial": 0.0,
            "wrist_ulnar": 0.0,
            "global_flex": 0.0,
            "global_ext": 0.0,
        }
        if q.size == 0:
            return values
        gain = max(0.0, float(self.config.hold_gain))
        deadband = max(0.0, float(self.config.hold_deadband))
        if gain <= 0.0:
            return values

        def _level(raw: float, scale: float = 1.5708) -> float:
            value = max(0.0, abs(float(raw)) - deadband) / max(1e-6, float(scale))
            return float(np.clip(value * gain, 0.0, 1.0))

        if q.size >= 3:
            values["pro_sup_pos"] = _level(max(0.0, float(q[0])), 1.5708)
            values["pro_sup_neg"] = _level(min(0.0, float(q[0])), 1.5708)
            values["wrist_radial"] = _level(max(0.0, float(q[1])), 0.4363)
            values["wrist_ulnar"] = _level(min(0.0, float(q[1])), 0.1745)
            values["wrist_flex"] = _level(max(0.0, float(q[2])), 0.7854)
            values["wrist_ext"] = _level(min(0.0, float(q[2])), 0.7854)
        if q.size >= 7:
            thumb_flex_terms = [
                _level(max(0.0, float(q[4])), 0.78),
                _level(min(0.0, float(q[5])), 0.7854),
                _level(min(0.0, float(q[6])), 1.309),
            ]
            thumb_abd = _level(q[3], 0.78)
            values["thumb_flex"] = float(np.clip(max(thumb_flex_terms + [0.45 * thumb_abd]), 0.0, 1.0))
            values["thumb_ext"] = float(np.clip(0.30 * thumb_abd, 0.0, 1.0))

        for name, start in (("index", 7), ("middle", 11), ("ring", 15), ("little", 19)):
            if q.size <= start:
                continue
            idx = [item for item in (start, start + 2, start + 3) if item < q.size]
            if idx:
                flex = float(np.mean([_level(max(0.0, float(q[item]))) for item in idx]))
                values[f"{name}_flex"] = flex
            abd_idx = start + 1
            if abd_idx < q.size:
                values[f"{name}_ext"] = 0.25 * _level(q[abd_idx], 0.2618)

        flex_keys = [key for key in values if key.endswith("_flex") and key != "global_flex"]
        ext_keys = [key for key in values if key.endswith("_ext") and key != "global_ext"]
        values["global_flex"] = float(np.mean([values[key] for key in flex_keys])) if flex_keys else 0.0
        values["global_ext"] = float(np.mean([values[key] for key in ext_keys])) if ext_keys else 0.0
        return values

    @staticmethod
    def _combine_movement_and_hold(movement_drive: dict[str, float], hold_drive: dict[str, float]) -> dict[str, float]:
        keys = set(movement_drive) | set(hold_drive)
        return {
            key: float(max(float(movement_drive.get(key, 0.0)), float(hold_drive.get(key, 0.0))))
            for key in keys
        }

    @staticmethod
    def _positive_mean(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.mean(np.maximum(arr, 0.0)))

    @staticmethod
    def _negative_mean(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.mean(np.maximum(-arr, 0.0)))

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
            wrist_value = self._wrist_muscle_value(name_upper, group_drive)
            if wrist_value is not None:
                value = wrist_value
            elif self._is_flexor(name_upper):
                value = self._group_value(group_drive, finger, "flex")
            elif self._is_extensor(name_upper):
                ext = self._group_value(group_drive, finger, "ext")
                flex = self._group_value(group_drive, finger, "flex")
                value = float(self.config.finger_extensor_gain) * ext + float(self.config.finger_extensor_cocontraction) * flex
            else:
                value = max(float(group_drive.get("global_flex", 0.0)), float(group_drive.get("global_ext", 0.0))) * 0.35

            if self._is_intrinsic(name_upper):
                value *= 0.68
            ctrl[idx] = float(value)
        return np.clip(float(self.config.gain) * ctrl, 0.0, 1.0)

    def _wrist_muscle_value(self, name_upper: str, group_drive: dict[str, float]) -> float | None:
        flex = float(group_drive.get("wrist_flex", 0.0))
        ext = float(group_drive.get("wrist_ext", 0.0))
        radial = float(group_drive.get("wrist_radial", 0.0))
        ulnar = float(group_drive.get("wrist_ulnar", 0.0))
        pro_pos = float(group_drive.get("pro_sup_pos", 0.0))
        pro_neg = float(group_drive.get("pro_sup_neg", 0.0))
        co = max(0.0, float(self.config.wrist_cocontraction))

        if name_upper == "ECRL":
            return max(0.85 * ext + 0.40 * radial, co * flex)
        if name_upper == "ECRB":
            return max(0.75 * ext + 0.30 * radial, co * flex)
        if name_upper == "ECU":
            return max(0.80 * ext + 0.45 * ulnar, co * flex)
        if name_upper == "FCR":
            return max(0.85 * flex + 0.40 * radial, co * ext)
        if name_upper == "FCU":
            return max(0.85 * flex + 0.45 * ulnar, co * ext)
        if name_upper == "PL":
            return 0.60 * flex + 0.12 * float(group_drive.get("global_flex", 0.0))
        if name_upper == "PT":
            return max(0.85 * pro_neg, 0.20 * flex)
        if name_upper == "PQ":
            return max(0.65 * pro_neg, 0.12 * flex)
        return None

    @staticmethod
    def _group_value(group_drive: dict[str, float], finger: str | None, direction: str) -> float:
        if finger is not None:
            return float(group_drive.get(f"{finger}_{direction}", 0.0))
        return float(group_drive.get(f"global_{direction}", 0.0))

    @staticmethod
    def _muscle_finger_key(name_upper: str) -> str | None:
        if name_upper == "EIP":
            return "index"
        if name_upper == "EDM":
            return "little"
        if any(token in name_upper for token in ("TH", "FPL", "EPL", "EPB", "APL", "OPP", "APB")):
            return "thumb"
        match = re.search(r"([2-5])", name_upper)
        if match is None:
            return None
        return {"2": "index", "3": "middle", "4": "ring", "5": "little"}.get(match.group(1))

    @staticmethod
    def _is_flexor(name_upper: str) -> bool:
        if any(token in name_upper for token in ("FDS", "FDP", "FPL", "FCR", "FCU", "OPP", "APB")):
            return True
        return name_upper in {"PL", "PALMARIS", "PALMARIS_LONGUS"}

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


def map_qspace_signal_to_actuator_indicators(
    signal: np.ndarray | list[float],
    *,
    actuator_names: Iterable[str],
    gain: float = 1.0,
) -> np.ndarray:
    """Project joint/q-space indicator magnitudes to muscle/node indicators."""

    estimator = RealtimePDDynamicCtrlEstimator(
        actuator_names=actuator_names,
        config=RealtimePDDynamicCtrlConfig(gain=float(gain), deadband=0.0, ema_alpha=1.0, max_step=0.0, tau_clip=0.0),
    )
    group_drive = estimator._finger_group_drive(finite_vector(signal))
    return estimator._drive_to_ctrl(group_drive)
