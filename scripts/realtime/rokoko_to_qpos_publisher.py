"""Receive Rokoko JSON v3 hand bones and publish the project's qpos packets.

Rokoko Custom streaming:
    127.0.0.1:14043, JSON v3

Project UDP qpos consumer:
    127.0.0.1:15000

The first received pose is used as the open-hand reference. Finger flexion is
computed from consecutive bone positions; wrist and abduction coordinates are
left at zero until a pose-specific calibration is added.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from typing import Any

import numpy as np


FINGER_LAYOUT = (
    ("rightIndex", 7),
    ("rightMiddle", 11),
    ("rightRing", 15),
    ("rightLittle", 19),
)
QPOS_SIZE = 23


def position(node: dict[str, Any]) -> np.ndarray:
    value = node["position"]
    return np.array([value["x"], value["y"], value["z"]], dtype=float)


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(math.acos(np.clip(cosine, -1.0, 1.0)))


def hand_points(packet: dict[str, Any]) -> dict[str, np.ndarray]:
    body = packet["scene"]["actors"][0]["body"]
    points = {"wrist": position(body["rightHand"])}
    for prefix, rokoko_name in (
        ("thumb", "rightThumb"),
        ("index", "rightIndex"),
        ("middle", "rightMiddle"),
        ("ring", "rightRing"),
        ("little", "rightLittle"),
    ):
        for segment in ("Proximal", "Medial", "Distal", "Tip"):
            key = f"{prefix}_{segment.lower()}"
            points[key] = position(body[f"{rokoko_name}{segment}"])
    return points


def finger_flexion(points: dict[str, np.ndarray], prefix: str) -> tuple[float, float, float]:
    joints = (
        angle_between(points[f"{prefix}_proximal"] - points["wrist"], points[f"{prefix}_medial"] - points[f"{prefix}_proximal"]),
        angle_between(points[f"{prefix}_medial"] - points[f"{prefix}_proximal"], points[f"{prefix}_distal"] - points[f"{prefix}_medial"]),
        angle_between(points[f"{prefix}_distal"] - points[f"{prefix}_medial"], points[f"{prefix}_tip"] - points[f"{prefix}_distal"]),
    )
    return tuple(float(np.clip(value, 0.0, math.pi / 2.0)) for value in joints)


def to_qpos(points: dict[str, np.ndarray], reference: dict[str, np.ndarray] | None) -> np.ndarray:
    qpos = np.zeros(QPOS_SIZE, dtype=float)
    for prefix, base in FINGER_LAYOUT:
        current = finger_flexion(points, prefix.removeprefix("right").lower())
        if reference is not None:
            baseline = finger_flexion(reference, prefix.removeprefix("right").lower())
            current = tuple(max(0.0, value - base_value) for value, base_value in zip(current, baseline))
        qpos[base] = current[0]
        qpos[base + 2] = current[1]
        qpos[base + 3] = current[2]

    thumb = finger_flexion(points, "thumb")
    thumb_base = finger_flexion(reference, "thumb") if reference is not None else (0.0, 0.0, 0.0)
    qpos[4] = max(0.0, thumb[0] - thumb_base[0])
    qpos[5] = max(0.0, thumb[1] - thumb_base[1])
    qpos[6] = max(0.0, thumb[2] - thumb_base[2])
    return qpos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rokoko-host", default="127.0.0.1")
    parser.add_argument("--rokoko-port", type=int, default=14043)
    parser.add_argument("--qpos-host", default="127.0.0.1")
    parser.add_argument("--qpos-port", type=int, default=15000)
    args = parser.parse_args()

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind((args.rokoko_host, args.rokoko_port))
    publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reference = None
    sequence = 0
    received = 0
    last_error = None
    print(f"Rokoko input: {args.rokoko_host}:{args.rokoko_port}")
    print(f"qpos output: {args.qpos_host}:{args.qpos_port}")
    try:
        while True:
            payload, _sender = receiver.recvfrom(65535)
            received += 1
            try:
                packet = json.loads(payload.decode("utf-8"))
                points = hand_points(packet)
                if reference is None:
                    reference = points
                    print("Open-hand reference captured.")
                qpos = to_qpos(points, reference)
                output = {
                    "type": "hand_qpos",
                    "version": 1,
                    "seq": sequence,
                    "timestamp": time.time(),
                    "qpos": qpos.tolist(),
                }
                publisher.sendto(
                    json.dumps(output, separators=(",", ":")).encode("utf-8"),
                    (args.qpos_host, args.qpos_port),
                )
                sequence += 1
                if sequence == 1 or sequence % 30 == 0:
                    print(f"received={received} seq={sequence} qpos_norm={np.linalg.norm(qpos):.3f}", flush=True)
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if str(exc) != last_error:
                    print(f"packet={received} ignored: {type(exc).__name__}: {exc}", flush=True)
                    last_error = str(exc)
                continue
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        receiver.close()
        publisher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())