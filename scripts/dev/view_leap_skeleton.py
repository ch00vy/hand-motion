from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gui.recorder_controller import LiveLeapQposProvider  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Leap skeleton tracking viewer.")
    parser.add_argument("--window-name", default="Ultraleap Tracking Preview", help="Compatibility option; Canvas owns the actual window name.")
    parser.add_argument("--scale", type=float, default=1.6, help="Compatibility option; ignored by the skeleton Canvas.")
    return parser.parse_args()


def main() -> int:
    _args = parse_args()
    provider = LiveLeapQposProvider(show_leap_view=True)
    if not provider.start():
        print(f"[LEAP][FAIL] {provider.message}", flush=True)
        return 2
    print("[LEAP] skeleton viewer active. Close from GUI or press Ctrl+C in this process.", flush=True)
    try:
        while True:
            _ = provider.latest_qpos()
            provider.show_preview()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        return 0
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
