"""Receive Rokoko JSON v3 hand bones and publish the project's qpos packets.

Rokoko Custom streaming:
    127.0.0.1:14043, JSON v3

Project UDP qpos consumer:
    127.0.0.1:15000

The first received pose is used as the open-hand reference. Finger flexion is
mapped with the same neutral, anatomical wrist axes and joint limits as the GUI.
Restart this publisher in the neutral pose to recalibrate. GUI calibration
buttons require a direct JSON connection, rather than these qpos packets.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
import sys

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.realtime.rokoko_mapping import RokokoHandMapper, hand_points


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
    mapper = RokokoHandMapper()
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
                first_frame = mapper.reference is None
                qpos = mapper.map_points(points)
                if first_frame:
                    print("Neutral reference captured (shared GUI mapper).")
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
