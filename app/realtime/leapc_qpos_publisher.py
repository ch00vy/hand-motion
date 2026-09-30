from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


QPOS_SIZE = 23
WRIST_PRO_SUP_IDX = 0
WRIST_YAW_IDX = WRIST_PRO_SUP_IDX
WRIST_DEV_IDX = 1
WRIST_FLEX_IDX = 2
THUMB_CMC_ABD_IDX = 3
THUMB_CMC_FLEX_IDX = 4
THUMB_MCP_IDX = 5
THUMB_IP_IDX = 6
FINGER_BASE_IDX = {"index": 7, "middle": 11, "ring": 15, "little": 19}
FINGER_CONFIGS = (
    ("index", 1, 1.00, 1.10, 1.10, 0.16, 0.20),
    ("middle", 2, 0.95, 1.00, 1.00, 0.14, 0.18),
    ("ring", 3, 0.95, 1.00, 1.00, 0.14, 0.18),
    ("little", 4, 0.98, 1.06, 1.08, 0.18, 0.22),
)
FLEXION_JOINTS = (
    THUMB_MCP_IDX,
    THUMB_IP_IDX,
    FINGER_BASE_IDX["index"],
    FINGER_BASE_IDX["index"] + 2,
    FINGER_BASE_IDX["index"] + 3,
    FINGER_BASE_IDX["middle"],
    FINGER_BASE_IDX["middle"] + 2,
    FINGER_BASE_IDX["middle"] + 3,
    FINGER_BASE_IDX["ring"],
    FINGER_BASE_IDX["ring"] + 2,
    FINGER_BASE_IDX["ring"] + 3,
    FINGER_BASE_IDX["little"],
    FINGER_BASE_IDX["little"] + 2,
    FINGER_BASE_IDX["little"] + 3,
)
QPOS_LOW = np.array(
    [
        -1.5708,
        -0.174533,
        -0.785398,
        -0.5,
        -0.78,
        -0.7854,
        -1.309,
        0.0,
        -0.2618,
        0.0,
        0.0,
        0.0,
        -0.2618,
        0.0,
        0.0,
        0.0,
        -0.2618,
        0.0,
        0.0,
        0.0,
        -0.2618,
        0.0,
        0.0,
    ],
    dtype=float,
)
QPOS_HIGH = np.array(
    [
        1.5708,
        0.436332,
        0.785398,
        0.78,
        0.7,
        0.6981,
        0.4363,
        1.5708,
        0.2618,
        1.5708,
        1.5708,
        1.5708,
        0.2618,
        1.5708,
        1.5708,
        1.5708,
        0.2618,
        1.5708,
        1.5708,
        1.5708,
        0.2618,
        1.5708,
        1.5708,
    ],
    dtype=float,
)
DESKTOP_UP_AXIS = np.array([0.0, 0.0, -1.0], dtype=float)
CLOSED_FIST_QPOS = np.zeros((QPOS_SIZE,), dtype=float)
CLOSED_FIST_QPOS[THUMB_CMC_ABD_IDX] = 0.42
CLOSED_FIST_QPOS[THUMB_CMC_FLEX_IDX] = 0.42
CLOSED_FIST_QPOS[THUMB_MCP_IDX] = -0.68
CLOSED_FIST_QPOS[THUMB_IP_IDX] = -1.02
for _base in FINGER_BASE_IDX.values():
    CLOSED_FIST_QPOS[_base + 0] = 1.32
    CLOSED_FIST_QPOS[_base + 1] = 0.0
    CLOSED_FIST_QPOS[_base + 2] = 1.45
    CLOSED_FIST_QPOS[_base + 3] = 1.22
CLOSED_FIST_QPOS = np.clip(CLOSED_FIST_QPOS, QPOS_LOW, QPOS_HIGH)


DEFAULT_ULTRALEAP_SDK_DIR = Path(r"C:\Program Files\Ultraleap\LeapSDK")


def _sdk_dir_looks_valid(path: Path) -> bool:
    return bool((path / "leapc_cffi").exists())


def _normalize_sdk_dir(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.name.lower() == "leapc_cffi":
        return path.parent
    if path.name.lower() == "x64" and path.parent.name.lower() == "lib":
        return path.parent.parent
    if _sdk_dir_looks_valid(path):
        return path
    nested = path / "LeapSDK"
    if _sdk_dir_looks_valid(nested):
        return nested
    return path


def _candidate_sdk_dirs() -> list[Path]:
    candidates: list[Path] = []
    for name in ("ULTRALEAP_SDK_DIR", "LEAPC_SDK_DIR"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        [
            DEFAULT_ULTRALEAP_SDK_DIR,
            Path(r"C:\Program Files\Ultraleap"),
            Path(r"C:\Program Files (x86)\Ultraleap"),
        ]
    )
    for root in (Path(r"C:\Program Files\Ultraleap"), Path(r"C:\Program Files (x86)\Ultraleap")):
        try:
            candidates.extend(sorted(root.glob("*LeapSDK*")))
            candidates.extend(sorted(root.glob("**/LeapSDK")))
        except Exception:
            pass
    return candidates


def resolve_ultraleap_sdk_dir() -> Path:
    seen: set[str] = set()
    fallback: Optional[Path] = None
    for candidate in _candidate_sdk_dirs():
        normalized = _normalize_sdk_dir(candidate)
        key = str(normalized).lower()
        if key in seen:
            continue
        seen.add(key)
        if fallback is None:
            fallback = normalized
        if _sdk_dir_looks_valid(normalized):
            return normalized
    return fallback or DEFAULT_ULTRALEAP_SDK_DIR


def _sdk_dir_from_env_or_default() -> Path:
    return resolve_ultraleap_sdk_dir()


def _float_from_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _load_leapc_cffi(sdk_dir: Path) -> tuple[Any, Any]:
    sdk_dir = _normalize_sdk_dir(Path(sdk_dir)).resolve()
    if not _sdk_dir_looks_valid(sdk_dir):
        raise FileNotFoundError(
            "LeapC Python binding 'leapc_cffi' tidak ditemukan. "
            f"SDK yang dicoba: {sdk_dir}. "
            "Set ULTRALEAP_SDK_DIR ke folder LeapSDK yang berisi subfolder leapc_cffi."
        )
    for dll_dir in (sdk_dir / "leapc_cffi", sdk_dir / "lib" / "x64"):
        if dll_dir.exists() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(dll_dir))
    if str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    from leapc_cffi import ffi, libleapc

    return ffi, libleapc


def _vec(v: Any, *, desktop_mode_transform: bool = True) -> np.ndarray:
    if desktop_mode_transform:
        return np.array([float(v.x), float(v.z), -float(v.y)], dtype=float)
    return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.zeros_like(arr)
    return arr / norm


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    au = _unit(a)
    bu = _unit(b)
    dot = float(np.clip(np.dot(au, bu), -1.0, 1.0))
    return float(math.acos(dot))


def _proj_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    nu = _unit(n)
    return np.asarray(v, dtype=float) - float(np.dot(v, nu)) * nu


def _signed_angle_on_plane(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> float:
    au = _unit(a)
    bu = _unit(b)
    nu = _unit(n)
    x = float(np.clip(np.dot(au, bu), -1.0, 1.0))
    y = float(np.dot(nu, np.cross(au, bu)))
    return float(math.atan2(y, x))


def _blend_coupled_flexion(direct_value: float, coupled_value: float, mcp_value: float, *, weight: float = 0.25) -> float:
    if float(mcp_value) <= 0.08:
        return float(direct_value)
    w = float(np.clip(weight, 0.0, 1.0))
    return float((1.0 - w) * float(direct_value) + w * float(coupled_value))


def _bone_vec(bone: Any, *, desktop_mode_transform: bool) -> np.ndarray:
    return _vec(bone.next_joint, desktop_mode_transform=desktop_mode_transform) - _vec(
        bone.prev_joint,
        desktop_mode_transform=desktop_mode_transform,
    )


def _quaternion_to_yaw(q: Any) -> float:
    qw, qx, qy, qz = float(q.w), float(q.x), float(q.y), float(q.z)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


@dataclass
class QposMappingState:
    prev_qpos: Optional[np.ndarray] = None
    prev_yaw: Optional[float] = None
    yaw_baseline: Optional[float] = None
    open_flex_baseline: Optional[np.ndarray] = None
    open_calibration_samples: int = 0
    mcp_abd_base: dict[int, float] = field(default_factory=dict)


class LeapCHandToQposMapper:
    """Lightweight LeapC skeleton to 23-dim MyoSuite qpos mapper."""

    def __init__(
        self,
        *,
        desktop_mode_transform: bool = True,
        desktop_yaw_offset: float = -2.040,
        auto_calibrate_open_hand: bool = True,
        pro_sup_sign: Optional[float] = None,
        pro_sup_offset: Optional[float] = None,
        grip_boost: Optional[float] = None,
        thumb_grip_boost: Optional[float] = None,
        use_legacy_mapping: Optional[bool] = None,
    ):
        self.desktop_mode_transform = bool(desktop_mode_transform)
        self.desktop_yaw_offset = float(desktop_yaw_offset)
        self.auto_calibrate_open_hand = bool(auto_calibrate_open_hand)
        self.use_legacy_mapping = _bool_from_env("HAND_USE_LEGACY_MAPPING", True) if use_legacy_mapping is None else bool(use_legacy_mapping)
        self.pro_sup_sign = float(_float_from_env("HAND_PRO_SUP_SIGN", 1.0) if pro_sup_sign is None else pro_sup_sign)
        self.pro_sup_offset = float(_float_from_env("HAND_PRO_SUP_OFFSET", 0.0) if pro_sup_offset is None else pro_sup_offset)
        self.grip_boost = float(np.clip(_float_from_env("HAND_GRIP_BOOST", 0.70) if grip_boost is None else grip_boost, 0.0, 1.0))
        self.thumb_grip_boost = float(
            np.clip(_float_from_env("HAND_THUMB_GRIP_BOOST", 0.20) if thumb_grip_boost is None else thumb_grip_boost, 0.0, 1.0)
        )
        self.state = QposMappingState()
        self.last_debug: dict[str, Any] = {}
        self._legacy_mapper: Any = None
        if self.use_legacy_mapping:
            self._legacy_mapper = self._build_legacy_mapper()

    def _build_legacy_mapper(self) -> Any:
        try:
            from app.realtime.realtime_io import LeapToMyoSuiteMapper

            return LeapToMyoSuiteMapper(
                desktop_mode_transform=self.desktop_mode_transform,
                desktop_yaw_offset=self.desktop_yaw_offset,
            )
        except Exception as exc:
            self.last_debug = {
                "mapping": "fallback_direct",
                "legacy_mapping_error": f"{type(exc).__name__}: {exc}",
            }
            return None

    def reset_tracking_state(self) -> None:
        self.state = QposMappingState()
        if self._legacy_mapper is not None:
            try:
                self._legacy_mapper.reset_tracking_state()
            except Exception:
                pass
        self.last_debug = {"stage": "reset"}

    def map_hand(self, hand: Any) -> np.ndarray:
        if self._legacy_mapper is not None:
            try:
                qpos = self._legacy_mapper.map_hand_to_qpos(hand, getattr(hand, "arm", None))
                qpos_arr = np.asarray(qpos, dtype=float).reshape(-1)
                fixed = np.zeros((QPOS_SIZE,), dtype=float)
                fixed[: min(QPOS_SIZE, qpos_arr.size)] = qpos_arr[:QPOS_SIZE]
                fixed = np.clip(np.where(np.isfinite(fixed), fixed, 0.0), QPOS_LOW, QPOS_HIGH)
                self.last_debug = {
                    "mapping": "legacy_pre_cleanup",
                    "raw_qpos": fixed.copy(),
                    "calibrated_qpos": fixed.copy(),
                    "open_calibration_samples": 0,
                }
                return fixed
            except Exception as exc:
                self.last_debug = {
                    "mapping": "fallback_direct",
                    "legacy_mapping_error": f"{type(exc).__name__}: {exc}",
                }

        qpos = np.zeros((QPOS_SIZE,), dtype=float)
        palm_direction, palm_normal, palm_right = self._palm_basis(hand)
        arm = hand.arm
        arm_dir = _unit(_bone_vec(arm, desktop_mode_transform=self.desktop_mode_transform))

        qpos[WRIST_PRO_SUP_IDX] = self._wrist_pro_supination(arm_dir, palm_normal)
        qpos[WRIST_DEV_IDX] = self._wrist_deviation(hand, arm_dir, palm_normal, palm_direction)
        qpos[WRIST_FLEX_IDX] = self._wrist_flexion(arm_dir, palm_right, palm_direction)
        self._map_thumb(hand, palm_right, palm_direction, qpos)
        self._map_fingers(hand, palm_right, qpos)
        raw_qpos = qpos.copy()
        self._apply_open_hand_calibration(hand, qpos)
        grip_score = self._apply_grip_boost(hand, qpos)
        self.last_debug = {
            "grab_strength": float(getattr(hand, "grab_strength", 0.0)),
            "grip_score": float(grip_score),
            "raw_qpos": raw_qpos.copy(),
            "calibrated_qpos": qpos.copy(),
            "open_calibration_samples": int(self.state.open_calibration_samples),
        }
        return self._smooth_and_clamp(qpos)

    def _palm_basis(self, hand: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        palm_direction = _unit(_vec(hand.palm.direction, desktop_mode_transform=self.desktop_mode_transform))
        palm_normal = _unit(_vec(hand.palm.normal, desktop_mode_transform=self.desktop_mode_transform))
        palm_normal = _unit(palm_normal - float(np.dot(palm_normal, palm_direction)) * palm_direction)
        palm_right = _unit(np.cross(palm_normal, palm_direction))
        if np.linalg.norm(palm_right) <= 1e-6:
            palm_right = np.array([1.0, 0.0, 0.0], dtype=float)
        return palm_direction, palm_normal, palm_right

    def _wrist_yaw(self, arm: Any) -> float:
        yaw = _quaternion_to_yaw(arm.rotation)
        if self.desktop_mode_transform:
            yaw -= self.desktop_yaw_offset
        if self.state.yaw_baseline is None:
            self.state.yaw_baseline = float(yaw)
        yaw = float(yaw - self.state.yaw_baseline)
        if self.state.prev_yaw is not None:
            while yaw - self.state.prev_yaw > math.pi:
                yaw -= 2.0 * math.pi
            while yaw - self.state.prev_yaw < -math.pi:
                yaw += 2.0 * math.pi
        self.state.prev_yaw = float(yaw)
        return float(np.clip(yaw, -1.2, 1.2))

    def _wrist_pro_supination(self, arm_dir: np.ndarray, palm_normal: np.ndarray) -> float:
        arm_axis = _unit(arm_dir)
        if np.linalg.norm(arm_axis) <= 1e-6:
            return 0.0
        up_axis = DESKTOP_UP_AXIS if self.desktop_mode_transform else np.array([0.0, 1.0, 0.0], dtype=float)
        side_ref = _unit(np.cross(up_axis, arm_axis))
        if np.linalg.norm(side_ref) <= 1e-6:
            side_ref = _unit(np.cross(np.array([1.0, 0.0, 0.0], dtype=float), arm_axis))
        palm_proj = _proj_to_plane(palm_normal, arm_axis)
        if np.linalg.norm(side_ref) <= 1e-6 or np.linalg.norm(palm_proj) <= 1e-6:
            return 0.0
        angle = _signed_angle_on_plane(side_ref, palm_proj, arm_axis)
        value = self.pro_sup_sign * angle + self.pro_sup_offset
        return float(np.clip(value, QPOS_LOW[WRIST_PRO_SUP_IDX], QPOS_HIGH[WRIST_PRO_SUP_IDX]))

    def _wrist_deviation(self, hand: Any, arm_dir: np.ndarray, palm_normal: np.ndarray, palm_direction: np.ndarray) -> float:
        arm_proj = _proj_to_plane(arm_dir, palm_normal)
        palm_proj = _proj_to_plane(palm_direction, palm_normal)
        if np.linalg.norm(arm_proj) > 1e-6 and np.linalg.norm(palm_proj) > 1e-6:
            angle = _signed_angle_on_plane(arm_proj, palm_proj, palm_normal)
        else:
            angle = math.atan2(float(palm_normal[0]), float(palm_normal[1]))
        is_left = int(getattr(hand, "type", 1)) == 0
        handedness_sign = 1.0 if is_left else -1.0
        return float(np.clip(handedness_sign * angle * 0.35, -0.174533, 0.436332))

    def _wrist_flexion(self, arm_dir: np.ndarray, palm_right: np.ndarray, palm_direction: np.ndarray) -> float:
        arm_proj = _proj_to_plane(arm_dir, palm_right)
        palm_proj = _proj_to_plane(palm_direction, palm_right)
        if np.linalg.norm(arm_proj) > 1e-6 and np.linalg.norm(palm_proj) > 1e-6:
            angle = _signed_angle_on_plane(arm_proj, palm_proj, palm_right)
        else:
            angle = math.asin(float(np.clip(_unit(palm_direction)[1], -1.0, 1.0)))
        return float(np.clip(-1.2 * angle, -0.785398, 0.785398))

    def _map_thumb(self, hand: Any, palm_right: np.ndarray, palm_direction: np.ndarray, qpos: np.ndarray) -> None:
        thumb = hand.digits[0]
        prox = _bone_vec(thumb.proximal, desktop_mode_transform=self.desktop_mode_transform)
        inter = _bone_vec(thumb.intermediate, desktop_mode_transform=self.desktop_mode_transform)
        distal = _bone_vec(thumb.distal, desktop_mode_transform=self.desktop_mode_transform)
        prox_u = _unit(prox if np.linalg.norm(prox) > 1e-6 else distal)
        radial = float(np.clip(np.dot(prox_u, _unit(palm_right)), -1.0, 1.0))
        qpos[THUMB_CMC_ABD_IDX] = float(np.clip(-0.75 * radial, -0.75, 0.75))
        qpos[THUMB_CMC_FLEX_IDX] = float(np.clip(0.55 * _signed_angle_on_plane(palm_direction, prox_u, palm_right), -0.60, 0.75))
        qpos[THUMB_MCP_IDX] = float(np.clip(-0.85 * _angle(prox, inter), -0.80, 0.05))
        qpos[THUMB_IP_IDX] = float(np.clip(-0.95 * _angle(inter, distal), -1.10, 0.05))

    def _map_fingers(self, hand: Any, palm_right: np.ndarray, qpos: np.ndarray) -> None:
        for name, digit_index, mcp_gain, pip_gain, dip_gain, pip_from_mcp_gain, dip_from_pip_gain in FINGER_CONFIGS:
            digit = hand.digits[int(digit_index)]
            base = int(FINGER_BASE_IDX[name])
            meta = _bone_vec(digit.metacarpal, desktop_mode_transform=self.desktop_mode_transform)
            prox = _bone_vec(digit.proximal, desktop_mode_transform=self.desktop_mode_transform)
            inter = _bone_vec(digit.intermediate, desktop_mode_transform=self.desktop_mode_transform)
            distal = _bone_vec(digit.distal, desktop_mode_transform=self.desktop_mode_transform)
            mcp = float(np.clip(_angle(meta, prox) * mcp_gain, 0.0, 1.5708))
            pip_direct = float(_angle(prox, inter) * pip_gain)
            pip_coupled = float(pip_from_mcp_gain * mcp)
            pip = float(np.clip(_blend_coupled_flexion(pip_direct, pip_coupled, mcp, weight=0.22), 0.0, 1.5708))
            dip_direct = float(_angle(inter, distal) * dip_gain)
            dip_coupled = float(dip_from_pip_gain * pip + 0.04 * mcp)
            dip = float(np.clip(_blend_coupled_flexion(dip_direct, dip_coupled, mcp, weight=0.30), 0.0, 1.5708))
            radial = float(np.clip(np.dot(_unit(prox), _unit(palm_right)), -1.0, 1.0))
            if digit_index not in self.state.mcp_abd_base or mcp < 0.45:
                prev = float(self.state.mcp_abd_base.get(digit_index, radial))
                self.state.mcp_abd_base[digit_index] = 0.92 * prev + 0.08 * radial
            abd = float(np.clip(-(radial - self.state.mcp_abd_base[digit_index]) * 0.9, -0.22, 0.22))
            curl_level = float(np.clip((0.45 * mcp + 0.35 * pip + 0.20 * dip) / 1.1, 0.0, 1.0))
            qpos[base + 0] = mcp
            qpos[base + 1] = abd * max(0.0, 1.0 - 0.75 * curl_level)
            qpos[base + 2] = pip
            qpos[base + 3] = dip

    def _apply_open_hand_calibration(self, hand: Any, qpos: np.ndarray) -> None:
        if not self.auto_calibrate_open_hand:
            return
        grab_strength = float(getattr(hand, "grab_strength", 0.0))
        extended_count = 0
        try:
            extended_count = sum(int(bool(hand.digits[idx].is_extended)) for idx in range(1, 5))
        except Exception:
            extended_count = 0
        flex_values = np.asarray([abs(float(qpos[idx])) for idx in FLEXION_JOINTS if 0 <= idx < qpos.size], dtype=float)
        mean_flex = float(np.mean(flex_values)) if flex_values.size else 0.0
        open_like = bool(grab_strength <= 0.22 and (extended_count >= 3 or mean_flex <= 0.45))
        if open_like:
            if self.state.open_flex_baseline is None:
                self.state.open_flex_baseline = qpos.copy()
            else:
                alpha = 0.12 if self.state.open_calibration_samples < 30 else 0.02
                for idx in FLEXION_JOINTS:
                    self.state.open_flex_baseline[idx] = (
                        (1.0 - alpha) * float(self.state.open_flex_baseline[idx]) + alpha * float(qpos[idx])
                    )
            self.state.open_calibration_samples += 1
        baseline = self.state.open_flex_baseline
        if baseline is None:
            return
        for idx in FLEXION_JOINTS:
            corrected = float(qpos[idx]) - float(baseline[idx])
            if abs(corrected) < 0.025:
                corrected = 0.0
            qpos[idx] = corrected

    def _apply_grip_boost(self, hand: Any, qpos: np.ndarray) -> float:
        grab_strength = float(np.clip(getattr(hand, "grab_strength", 0.0), 0.0, 1.0))
        try:
            extended_count = sum(int(bool(hand.digits[idx].is_extended)) for idx in range(1, 5))
        except Exception:
            extended_count = 4
        closed_by_extension = float(np.clip((4.0 - float(extended_count)) / 4.0, 0.0, 1.0))
        flex_values = np.asarray([abs(float(qpos[idx])) for idx in FLEXION_JOINTS if 0 <= idx < qpos.size], dtype=float)
        mean_flex = float(np.mean(flex_values)) if flex_values.size else 0.0
        closed_by_flex = float(np.clip((mean_flex - 0.28) / 0.72, 0.0, 1.0))
        grip_score = max(
            float(np.clip((grab_strength - 0.20) / 0.80, 0.0, 1.0)),
            0.70 * closed_by_extension if extended_count <= 2 else 0.0,
            0.55 * closed_by_flex,
        )
        if grip_score <= 1e-6 or (self.grip_boost <= 1e-6 and self.thumb_grip_boost <= 1e-6):
            return 0.0
        finger_blend = float(np.clip(grip_score * self.grip_boost, 0.0, 1.0))
        thumb_blend = float(np.clip(grip_score * self.thumb_grip_boost, 0.0, 1.0))
        for idx in (THUMB_MCP_IDX, THUMB_IP_IDX):
            qpos[idx] = (1.0 - thumb_blend) * float(qpos[idx]) + thumb_blend * float(CLOSED_FIST_QPOS[idx])
        for idx in (
            FINGER_BASE_IDX["index"],
            FINGER_BASE_IDX["index"] + 2,
            FINGER_BASE_IDX["index"] + 3,
            FINGER_BASE_IDX["middle"],
            FINGER_BASE_IDX["middle"] + 2,
            FINGER_BASE_IDX["middle"] + 3,
            FINGER_BASE_IDX["ring"],
            FINGER_BASE_IDX["ring"] + 2,
            FINGER_BASE_IDX["ring"] + 3,
            FINGER_BASE_IDX["little"],
            FINGER_BASE_IDX["little"] + 2,
            FINGER_BASE_IDX["little"] + 3,
        ):
            qpos[idx] = (1.0 - finger_blend) * float(qpos[idx]) + finger_blend * float(CLOSED_FIST_QPOS[idx])
        return float(grip_score)

    def _smooth_and_clamp(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.where(np.isfinite(qpos), qpos, 0.0)
        qpos = np.clip(qpos, QPOS_LOW, QPOS_HIGH)
        prev = self.state.prev_qpos
        if prev is None or prev.shape != qpos.shape:
            self.state.prev_qpos = qpos.copy()
            return qpos
        max_step = np.full_like(qpos, 0.25)
        max_step[WRIST_YAW_IDX] = 0.35
        max_step[WRIST_DEV_IDX] = 0.12
        max_step[WRIST_FLEX_IDX] = 0.30
        for joint_idx in FLEXION_JOINTS:
            max_step[joint_idx] = 0.35
        clamped = prev + np.clip(qpos - prev, -max_step, max_step)
        alpha = np.full_like(qpos, 0.35)
        for base in FINGER_BASE_IDX.values():
            alpha[base + 0] = 0.60
            alpha[base + 2] = 0.60
            alpha[base + 3] = 0.60
        alpha[THUMB_MCP_IDX] = 0.55
        alpha[THUMB_IP_IDX] = 0.55
        out = prev * (1.0 - alpha) + clamped * alpha
        out = np.clip(out, QPOS_LOW, QPOS_HIGH)
        self.state.prev_qpos = out.copy()
        return out


def _configure_tracking_policy(lib: Any, connection: Any, *, tracking_mode: str) -> None:
    mode_value = {
        "desktop": lib.eLeapTrackingMode_Desktop,
        "hmd": lib.eLeapTrackingMode_HMD,
        "screentop": lib.eLeapTrackingMode_ScreenTop,
    }.get(str(tracking_mode).strip().lower(), lib.eLeapTrackingMode_Desktop)
    mode_result = lib.LeapSetTrackingMode(connection, mode_value)
    if mode_result != lib.eLeapRS_Success:
        raise RuntimeError(f"LeapSetTrackingMode failed: {int(mode_result)}")
    try:
        set_flags = int(lib.eLeapPolicyFlag_BackgroundFrames)
        clear_flags = 0
        mode_name = str(tracking_mode).strip().lower()
        if mode_name == "screentop":
            set_flags |= int(lib.eLeapPolicyFlag_OptimizeScreenTop)
            clear_flags |= int(lib.eLeapPolicyFlag_OptimizeHMD)
        elif mode_name == "hmd":
            set_flags |= int(lib.eLeapPolicyFlag_OptimizeHMD)
            clear_flags |= int(lib.eLeapPolicyFlag_OptimizeScreenTop)
        else:
            clear_flags |= int(lib.eLeapPolicyFlag_OptimizeHMD) | int(lib.eLeapPolicyFlag_OptimizeScreenTop)
        policy_result = lib.LeapSetPolicyFlags(connection, set_flags, clear_flags)
        if policy_result != lib.eLeapRS_Success:
            raise RuntimeError(f"LeapSetPolicyFlags failed: {int(policy_result)}")
    except Exception:
        # Some runtime/SDK combinations still stream tracking without policy flags.
        pass


def _open_connection(ffi: Any, lib: Any, *, tracking_mode: str) -> Any:
    conn_ptr = ffi.new("LEAP_CONNECTION *")
    result = lib.LeapCreateConnection(ffi.NULL, conn_ptr)
    if result != lib.eLeapRS_Success:
        raise RuntimeError(f"LeapCreateConnection failed: {int(result)}")
    connection = conn_ptr[0]
    result = lib.LeapOpenConnection(connection)
    if result != lib.eLeapRS_Success:
        lib.LeapDestroyConnection(connection)
        raise RuntimeError(f"LeapOpenConnection failed: {int(result)}")
    try:
        _configure_tracking_policy(lib, connection, tracking_mode=tracking_mode)
    except Exception as exc:
        lib.LeapCloseConnection(connection)
        lib.LeapDestroyConnection(connection)
        raise exc
    return connection


def run_publisher(args: argparse.Namespace) -> int:
    sdk_dir = Path(args.sdk_dir).resolve()
    ffi, lib = _load_leapc_cffi(sdk_dir)
    connection = _open_connection(ffi, lib, tracking_mode=str(args.tracking_mode))
    mapper = LeapCHandToQposMapper(desktop_mode_transform=not bool(args.no_desktop_transform))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (str(args.host), int(args.port))
    message = ffi.new("LEAP_CONNECTION_MESSAGE *")
    message.size = ffi.sizeof("LEAP_CONNECTION_MESSAGE")
    seq = 0
    last_print = 0.0
    print(f"[LeapC] publishing qpos UDP to {target[0]}:{target[1]} from SDK {sdk_dir}")
    try:
        while True:
            result = lib.LeapPollConnection(connection, int(args.poll_timeout_ms), message)
            if result == lib.eLeapRS_Timeout:
                continue
            if result != lib.eLeapRS_Success:
                print(f"[LeapC][WARN] LeapPollConnection returned {int(result)}")
                time.sleep(0.05)
                continue
            if message.type != lib.eLeapEventType_Tracking:
                continue
            event = message.tracking_event
            if event == ffi.NULL or int(event.nHands) <= 0:
                continue
            hand = event.pHands[0]
            qpos = mapper.map_hand(hand)
            packet = {
                "type": "hand_qpos",
                "version": 1,
                "seq": int(seq),
                "timestamp": time.time(),
                "qpos": [float(x) for x in qpos.tolist()],
            }
            sock.sendto(json.dumps(packet, separators=(",", ":")).encode("utf-8"), target)
            seq += 1
            now = time.perf_counter()
            if now - last_print >= float(args.print_every_seconds):
                last_print = now
                print(f"[LeapC] seq={seq} hands={int(event.nHands)} qpos0={qpos[0]:.3f} qpos_norm={float(np.linalg.norm(qpos)):.3f}")
    except KeyboardInterrupt:
        print("[LeapC] stopped by user")
    finally:
        sock.close()
        lib.LeapCloseConnection(connection)
        lib.LeapDestroyConnection(connection)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish LeapC CFFI skeleton frames as 23-dim qpos over UDP.")
    parser.add_argument("--sdk-dir", default=str(_sdk_dir_from_env_or_default()), help="Ultraleap LeapSDK directory.")
    parser.add_argument("--host", default="127.0.0.1", help="UDP target host used by the GUI receiver.")
    parser.add_argument("--port", type=int, default=15000, help="UDP target port used by the GUI receiver.")
    parser.add_argument("--tracking-mode", default="desktop", choices=("desktop", "hmd", "screentop"), help="Leap tracking mode.")
    parser.add_argument("--poll-timeout-ms", type=int, default=100, help="LeapPollConnection timeout in milliseconds.")
    parser.add_argument("--print-every-seconds", type=float, default=1.0, help="Console status print interval.")
    parser.add_argument("--no-desktop-transform", action="store_true", help="Disable desktop coordinate transform.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    return run_publisher(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
