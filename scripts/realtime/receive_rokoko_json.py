"""Print Rokoko Studio Custom JSON v3 packets received over UDP.

Configure Rokoko Studio Custom streaming with:
    Forward IP: 127.0.0.1
    Port: 14043
    Data format: JSON v3

This probe intentionally does not map bones to MyoSuite qpos yet. It captures
one packet so the mapping can be implemented against the actual enabled
Rokoko streams (body, hands, face, or props).
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14043)
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("artifacts/rokoko_json_v3_sample.json"),
        help="Path for the first decoded packet.",
    )
    args = parser.parse_args()

    address = (str(args.host), int(args.port))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(address)
        print(f"Listening for Rokoko JSON v3 UDP on {address[0]}:{address[1]}")
        print("Move in Rokoko Studio. Press Ctrl+C to stop.")
        packet_number = 0
        while True:
            payload, sender = sock.recvfrom(65535)
            packet_number += 1
            text = payload.decode("utf-8", errors="replace")
            print(f"packet={packet_number} bytes={len(payload)} sender={sender}")
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                print(text[:1000])
                continue

            if packet_number == 1:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                args.save.write_text(json.dumps(decoded, indent=2), encoding="utf-8")
                print(f"Saved first decoded packet to {args.save}")
            if isinstance(decoded, dict):
                print("top-level keys:", ", ".join(sorted(decoded)))
            else:
                print("JSON root:", type(decoded).__name__)


if __name__ == "__main__":
    raise SystemExit(main())