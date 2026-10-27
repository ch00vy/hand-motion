"""Shared right-hand JSON v3 mapping for the GUI and UDP publisher.

Positions and rightHand rotation must use the same world coordinate system.
The calibrated pose maps to zero qpos in MyoHand. Wrist orientation is fitted
to the model hinge axes. Finger geometry is an approximation rather than a
full inverse-kinematics fit, particularly for thumb opposition.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from app.realtime.rokoko_geometry import hand_basis, unit
from app.realtime.rokoko_wrist_mapping import RokokoWristMapper
from app.realtime.rokoko_finger_abduction import RokokoFingerAbductionMapper

FINGER_LAYOUT = (("index", 7), ("middle", 11), ("ring", 15), ("little", 19))
QPOS_LIMITS = np.asarray([
    (-1.5708, 1.5708), (-.174533, .436332), (-.785398, .785398),
    (-.5, .78), (-.78, .7), (-.785398, .698132), (-1.309, .436332),
    *([(0., 1.5708), (-.261799, .261799), (0., 1.5708), (0., 1.5708)] * 4),
])
# Explicit fist calibration maps the user's measured closed pose to the full
# flexion range. Direct mapping keeps the measured angles until calibrated.
FLEXION_TARGETS = {
    4: float(QPOS_LIMITS[4, 1]),
    5: float(QPOS_LIMITS[5, 0]),
    6: float(QPOS_LIMITS[6, 0]),
    **{base + offset: float(QPOS_LIMITS[base + offset, 1])
       for _, base in FINGER_LAYOUT for offset in (0, 2, 3)},
}


def hand_points(packet: dict[str, Any]) -> dict[str, np.ndarray]:
    body = packet["scene"]["actors"][0]["body"]
    def position(name: str) -> np.ndarray:
        value = body[name]["position"]
        result = np.asarray([value[axis] for axis in "xyz"], dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Non-finite position: {name}")
        return result
    points = {"wrist": position("rightHand"),
              "wrist_rotation": RokokoWristMapper.quaternion_from_node(body["rightHand"]["rotation"])}
    for name in ("thumb", "index", "middle", "ring", "little"):
        for segment in ("Proximal", "Medial", "Distal", "Tip"):
            points[f"{name}_{segment.lower()}"] = position(f"right{name.title()}{segment}")
    # Validate before a malformed packet can replace the calibration reference.
    hand_basis(points)
    for name in ("thumb", "index", "middle", "ring", "little"):
        finger_flexion(points, name)
    return points


def finger_flexion(points: dict[str, np.ndarray], name: str) -> np.ndarray:
    basis = hand_basis(points)
    segments = [unit(points[f"{name}_{end}"] - points[f"{name}_{start}"])
                for start, end in (("proximal", "medial"), ("medial", "distal"), ("distal", "tip"))]
    if name == "thumb":
        metacarpal = unit(points["thumb_proximal"] - points["wrist"])
        mcp = np.arccos(np.clip(np.dot(metacarpal, segments[0]), -1., 1.))
    else:
        # Elevation out of the palm plane, independent of in-plane finger spread.
        local = basis.T @ segments[0]
        # Keep the forward/backward sign: an MCP bend past 90 degrees must
        # not fold back toward zero as it would with an unsigned projection.
        forward = np.copysign(np.hypot(local[0], local[1]), local[1])
        mcp = np.arctan2(local[2], forward)
    return np.asarray([mcp, *[np.arccos(np.clip(np.dot(a, b), -1., 1.))
                              for a, b in zip(segments, segments[1:])]])


def thumb_abduction(points: dict[str, np.ndarray]) -> float:
    direction = hand_basis(points).T @ unit(points["thumb_proximal"] - points["wrist"])
    return float(np.arctan2(direction[2], np.hypot(direction[0], direction[1])))


def raw_flexion_qpos(points: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> np.ndarray:
    """Signed changes before model limits; preserve these for calibration."""
    result = np.zeros(23)
    for name, base in FINGER_LAYOUT:
        result[[base, base + 2, base + 3]] = finger_flexion(points, name) - finger_flexion(reference, name)
    # MyoHand thumb MP/IP flex in the negative direction, including without fist scaling.
    result[4:7] = (finger_flexion(points, "thumb") - finger_flexion(reference, "thumb")) * [1., -1., -1.]
    result[3] = thumb_abduction(points) - thumb_abduction(reference)
    return result


def flexion_qpos(points: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> np.ndarray:
    return np.clip(raw_flexion_qpos(points, reference), QPOS_LIMITS[:, 0], QPOS_LIMITS[:, 1])


class RokokoHandMapper:
    def __init__(self) -> None:
        self.wrist = RokokoWristMapper()
        self.abduction = RokokoFingerAbductionMapper()
        self.reset()

    def reset(self) -> None:
        self.reference: dict[str, np.ndarray] | None = None
        self.latest_points: dict[str, np.ndarray] | None = None
        self.closed_flexion: np.ndarray | None = None
        self.open_calibrated = False
        self.wrist.reset()
        self.abduction.reset()

    def calibrate_open(self, points: dict[str, np.ndarray], *, explicit: bool = True) -> np.ndarray:
        basis = hand_basis(points)
        self.wrist.calibrate(points["wrist_rotation"], basis)
        self.abduction.map_points(points, calibrate=True)
        self.reference = {key: value.copy() for key, value in points.items()}
        self.latest_points = self.reference
        self.closed_flexion = None
        self.open_calibrated = explicit
        return self.map_points(points)

    def map_points(self, points: dict[str, np.ndarray]) -> np.ndarray:
        if self.reference is None:
            return self.calibrate_open(points, explicit=False)
        raw = raw_flexion_qpos(points, self.reference)
        result = np.clip(raw, QPOS_LIMITS[:, 0], QPOS_LIMITS[:, 1])
        result = self.wrist.map_into_qpos(result, points["wrist_rotation"])
        result = self.abduction.map_into_qpos(result, points)
        if self.closed_flexion is not None:
            for index, target in FLEXION_TARGETS.items():
                baseline = float(self.closed_flexion[index])
                # An unmoved joint keeps its direct-angle mapping.
                if abs(baseline) >= .05:
                    # The closed-pose sign defines bending direction for this
                    # stream. Never discard negative MCP motion before scaling.
                    result[index] = np.clip(raw[index] / baseline, 0., 1.) * target
        if not np.all(np.isfinite(result)):
            raise ValueError("Mapping produced non-finite qpos")
        self.latest_points = {key: value.copy() for key, value in points.items()}
        return np.clip(result, QPOS_LIMITS[:, 0], QPOS_LIMITS[:, 1])

    def calibrate_closed(self) -> tuple[bool, str]:
        if not self.open_calibrated or self.latest_points is None or self.reference is None:
            return False, "Kalibrasikan pose netral terlebih dahulu."
        closed = raw_flexion_qpos(self.latest_points, self.reference)
        calibrated_count = sum(abs(closed[i]) >= .05 for i in FLEXION_TARGETS)
        if not calibrated_count:
            return False, "Perubahan jari terlalu kecil. Bentuk kepalan setelah kalibrasi netral."
        self.closed_flexion = closed
        message = f"Kepalan maksimal tersimpan: {calibrated_count}/{len(FLEXION_TARGETS)} sendi terkalibrasi."
        missing_mcp = [name for name, base in FINGER_LAYOUT if abs(closed[base]) < .05]
        if missing_mcp:
            names = {"index": "telunjuk", "middle": "tengah", "ring": "manis", "little": "kelingking"}
            message += " Pangkal belum terbaca: " + ", ".join(names[name] for name in missing_mcp) + "."
        if calibrated_count < len(FLEXION_TARGETS):
            message += " Sendi yang perubahannya terlalu kecil tetap memakai sudut langsung."
        return True, message
