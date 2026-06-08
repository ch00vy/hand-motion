from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDK = PROJECT_ROOT / "Ultraleap" / "LeapSDK"


def _configure_sdk_path() -> None:
    if "LEAPSDK_INSTALL_LOCATION" not in os.environ and DEFAULT_SDK.exists():
        os.environ["LEAPSDK_INSTALL_LOCATION"] = str(DEFAULT_SDK)


class ImageProbeListener:
    def __init__(self, *, max_images: int = 5, save_dir: Path | None = None):
        from leap.event_listener import Listener

        class _Listener(Listener):
            def __init__(self, owner: "ImageProbeListener"):
                self.owner = owner

            def on_connection_event(self, event: Any) -> None:
                print("[LEAP] connection opened")

            def on_device_event(self, event: Any) -> None:
                print("[LEAP] device event received")

            def on_policy_event(self, event: Any) -> None:
                try:
                    print(f"[LEAP] policy flags: {event.current_policy_flags}")
                except Exception:
                    print("[LEAP] policy event received")

            def on_tracking_event(self, event: Any) -> None:
                self.owner.tracking_count += 1

            def on_image_event(self, event: Any) -> None:
                self.owner.handle_image_event(event)

            def on_image_request_error_event(self, event: Any) -> None:
                print(f"[LEAP][WARN] image request error: {event}")

            def on_error(self, error: Exception) -> None:
                print(f"[LEAP][ERROR] {type(error).__name__}: {error}")

        self.listener = _Listener(self)
        self.max_images = max(1, int(max_images))
        self.save_dir = save_dir
        self.image_count = 0
        self.tracking_count = 0

    @property
    def done(self) -> bool:
        return self.image_count >= self.max_images

    def handle_image_event(self, event: Any) -> None:
        self.image_count += 1
        images = list(getattr(event, "image", []) or [])
        print(f"[LEAP] image event {self.image_count}: stereo_images={len(images)}")
        for idx, image in enumerate(images):
            self._print_image_info(idx, image)
            if self.save_dir is not None:
                self._save_image(idx, image)

    def _print_image_info(self, idx: int, image: Any) -> None:
        data = getattr(image, "c_data", None)
        props = getattr(data, "properties", None)
        if props is None:
            print(f"  image[{idx}] properties unavailable")
            return
        print(
            "  "
            f"image[{idx}] width={int(props.width)} height={int(props.height)} "
            f"bpp={int(props.bpp)} format={int(props.format)} type={int(props.type)}"
        )

    def _save_image(self, idx: int, image: Any) -> None:
        from leapc_cffi import ffi

        data = getattr(image, "c_data", None)
        props = getattr(data, "properties", None)
        if data is None or props is None:
            return
        width = int(props.width)
        height = int(props.height)
        bpp = int(props.bpp)
        if width <= 0 or height <= 0 or bpp <= 0:
            return
        start = int(getattr(data, "offset", 0))
        byte_count = width * height * bpp
        raw = bytes(ffi.buffer(data.data + start, byte_count))
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, bpp)
        if bpp == 1:
            out = arr[:, :, 0]
        else:
            out = arr[:, :, 0]
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"leap_image_{self.image_count:03d}_{idx}.npy"
        np.save(path, out)
        print(f"  saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Ultraleap ImageEvent stream.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Maximum test duration.")
    parser.add_argument("--max-images", type=int, default=5, help="Stop after this many image events.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory to save raw left/right images as .npy.")
    return parser.parse_args()


def main() -> int:
    _configure_sdk_path()
    args = parse_args()

    from leap import Connection
    from leap.enums import PolicyFlag

    probe = ImageProbeListener(max_images=int(args.max_images), save_dir=args.save_dir)
    connection = Connection(listeners=[probe.listener])
    print(f"[LEAP] using LEAPSDK_INSTALL_LOCATION={os.environ.get('LEAPSDK_INSTALL_LOCATION', '')}")
    try:
        connection.connect(auto_poll=True)
        flags = connection.set_policy_flags([PolicyFlag.Images])
        print(f"[LEAP] requested image policy, current flags={flags}")
        deadline = time.time() + max(0.5, float(args.seconds))
        while time.time() < deadline and not probe.done:
            time.sleep(0.05)
    finally:
        try:
            connection.disconnect()
        except Exception:
            pass

    print(f"[LEAP] tracking_events={probe.tracking_count} image_events={probe.image_count}")
    if probe.image_count <= 0:
        print("[LEAP][FAIL] No ImageEvent received. Check Ultraleap service/device image permissions.")
        return 1
    print("[LEAP][OK] ImageEvent stream is available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
