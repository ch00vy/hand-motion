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
    parser.add_argument("--window-name", default="Ultraleap Tracking Preview", help="OpenCV window title.")
    parser.add_argument("--scale", type=float, default=1.6, help="Initial display window scale. The window remains user-resizable.")
    parser.add_argument("--window-x", type=int, default=None)
    parser.add_argument("--window-y", type=int, default=None)
    parser.add_argument("--window-width", type=int, default=600)
    parser.add_argument("--window-height", type=int, default=450)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provider = LiveLeapQposProvider(
        show_leap_view=True,
        window_name=str(args.window_name),
        window_scale=float(args.scale),
    )
    if not provider.start():
        print(f"[LEAP][FAIL] {provider.message}", flush=True)
        return 2
    try:
        import cv2

        cv2.namedWindow(str(args.window_name), cv2.WINDOW_NORMAL)
        cv2.resizeWindow(str(args.window_name), int(args.window_width), int(args.window_height))
        if args.window_x is not None and args.window_y is not None:
            cv2.moveWindow(str(args.window_name), int(args.window_x), int(args.window_y))
        cv2.waitKey(1)
    except Exception as exc:
        print(f"[LEAP][WARN] Could not position window: {type(exc).__name__}: {exc}", flush=True)
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
