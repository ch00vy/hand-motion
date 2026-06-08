from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gui.leap_camera_controller import LeapCameraController  # noqa: E402


PROJECTION_MODES = (
    "inverse_z",
    "inverse_neg_z",
    "inverse_y",
    "inverse_neg_y",
    "inverse_x",
    "inverse_neg_x",
    "forward_z",
    "forward_neg_z",
    "forward_y",
    "forward_neg_y",
    "forward_x",
    "forward_neg_x",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Ultraleap IR camera + skeleton overlay viewer.")
    parser.add_argument("--camera", choices=("left", "right"), default="right", help="IR camera image to display.")
    parser.add_argument("--scale", type=float, default=1.6, help="Display scale multiplier.")
    parser.add_argument("--window-name", default="Ultraleap IR + Skeleton", help="OpenCV window title.")
    parser.add_argument("--show-image", action="store_true", help="Show IR image behind the skeleton. Default is skeleton-only.")
    parser.add_argument("--projection-mode", choices=PROJECTION_MODES, default="inverse_neg_y", help="Camera matrix/ray projection variant to audit.")
    parser.add_argument("--flip-x", action="store_true", help="Flip projected skeleton horizontally after calibration projection.")
    parser.add_argument("--flip-y", action="store_true", default=False, help="Flip projected skeleton vertically after calibration projection.")
    parser.add_argument("--no-flip-y", action="store_false", dest="flip_y", help="Disable vertical flip.")
    parser.add_argument("--calibration-offset-x", type=float, default=-26.0, help="Optional debug X correction in source-image pixels.")
    parser.add_argument("--calibration-offset-y", type=float, default=0.0, help="Optional debug Y correction in source-image pixels.")
    parser.add_argument("--calibration-scale", type=float, default=1.0, help="Optional debug scale correction around image center.")
    parser.add_argument("--calibration-rotation-deg", type=float, default=-3.0, help="Fixed post-projection rotation correction in degrees.")
    parser.add_argument("--image-mode", choices=("raw", "rectified", "rectified_transpose"), default="rectified_transpose", help="IR image display mode.")
    parser.add_argument("--calibration-step", type=float, default=2.0, help="Keyboard offset step in source-image pixels.")
    parser.add_argument("--rotation-step", type=float, default=1.0, help="Keyboard rotation step in degrees.")
    parser.add_argument("--scale-step", type=float, default=0.01, help="Keyboard scale step.")
    return parser.parse_args()


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo:
        return arr
    return np.clip((arr.astype(float) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def fallback_project(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    out = np.full((len(pts), 2), np.nan, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        return out
    valid = np.all(np.isfinite(pts), axis=1)
    if not np.any(valid):
        return out
    x_raw = pts[:, 0]
    y_raw = pts[:, 2]
    min_x, max_x = np.percentile(x_raw[valid], [5, 95])
    min_y, max_y = np.percentile(y_raw[valid], [5, 95])
    if abs(max_x - min_x) < 1e-6 or abs(max_y - min_y) < 1e-6:
        return out
    pad = 0.14
    span_x = max_x - min_x
    span_y = max_y - min_y
    min_x -= span_x * pad
    max_x += span_x * pad
    min_y -= span_y * pad
    max_y += span_y * pad
    out[:, 0] = ((pts[:, 0] - min_x) / max(1e-6, max_x - min_x)) * width
    out[:, 1] = ((pts[:, 2] - min_y) / max(1e-6, max_y - min_y)) * height
    return out


def draw_skeleton(
    image_bgr: np.ndarray,
    skeleton: np.ndarray | None,
    edges: list[tuple[int, int]],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    source_width: int,
    source_height: int,
) -> None:
    if skeleton is None:
        return
    pts = np.asarray(skeleton, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        return
    height, width = image_bgr.shape[:2]
    if pts.shape[1] >= 3:
        pts2 = fallback_project(pts, source_width, source_height)
    else:
        pts2 = pts[:, :2].copy()
    pts2[:, 0] *= scale_x
    pts2[:, 1] *= scale_y
    valid = np.all(np.isfinite(pts2), axis=1)

    for i0, i1 in edges:
        if i0 >= len(pts2) or i1 >= len(pts2) or not (valid[i0] and valid[i1]):
            continue
        p0 = tuple(np.round(pts2[i0]).astype(int).tolist())
        p1 = tuple(np.round(pts2[i1]).astype(int).tolist())
        cv2.line(image_bgr, p0, p1, (80, 255, 120), 2, cv2.LINE_AA)
    for idx in range(len(pts2)):
        if not valid[idx]:
            continue
        p = tuple(np.round(pts2[idx]).astype(int).tolist())
        cv2.circle(image_bgr, p, 3, (245, 245, 245), -1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    projection_index = PROJECTION_MODES.index(str(args.projection_mode))
    controller = LeapCameraController(
        projection_mode=PROJECTION_MODES[projection_index],
        flip_x=bool(args.flip_x),
        flip_y=bool(args.flip_y),
        calibration_offset_x=float(args.calibration_offset_x),
        calibration_offset_y=float(args.calibration_offset_y),
        calibration_scale=float(args.calibration_scale),
        calibration_rotation_deg=float(args.calibration_rotation_deg),
        image_mode=str(args.image_mode),
        use_images=bool(args.show_image),
    )
    if not controller.start():
        print(f"[LEAP][FAIL] {controller.message}")
        return 1

    window_name = str(args.window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    last_status = 0.0
    try:
        while True:
            if bool(args.show_image):
                image, skeleton, edges = controller.latest_view(camera=str(args.camera))
            else:
                image = None
                skeleton, edges = controller.latest_skeleton()

            if image is None:
                scale = max(0.2, float(args.scale))
                frame = np.zeros((int(240 * scale), int(640 * scale), 3), dtype=np.uint8)
                frame[:, :] = (8, 10, 14)
                draw_skeleton(
                    frame,
                    skeleton,
                    edges,
                    scale_x=scale,
                    scale_y=scale,
                    source_width=640,
                    source_height=240,
                )
                cv2.putText(
                    frame,
                    f"LEAP skeleton-only | track={controller.tracking_count}",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (40, 230, 255),
                    2,
                    cv2.LINE_AA,
                )
                if skeleton is None:
                    cv2.putText(frame, controller.message, (24, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2, cv2.LINE_AA)
            else:
                gray = contrast_stretch(image)
                scale = max(0.2, float(args.scale))
                source_h, source_w = gray.shape[:2]
                disp_w = max(1, int(gray.shape[1] * scale))
                disp_h = max(1, int(gray.shape[0] * scale))
                gray = cv2.resize(gray, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                draw_skeleton(
                    frame,
                    skeleton,
                    edges,
                    scale_x=scale,
                    scale_y=scale,
                    source_width=source_w,
                    source_height=source_h,
                )
                cv2.putText(
                    frame,
                    f"{str(args.camera).upper()} {controller.image_mode} IR + skeleton | img={controller.image_count} track={controller.tracking_count} "
                    f"projection={controller.projection_mode} flip=({int(controller.flip_x)},{int(controller.flip_y)})",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (40, 230, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"calib x={controller.calibration_offset_x:.1f} y={controller.calibration_offset_y:.1f} "
                    f"scale={controller.calibration_scale:.3f} rot={controller.calibration_rotation_deg:.1f} | "
                    "official projection + rectified image | optional debug: WASD move QE rotate +/- scale R reset",
                    (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (255, 220, 90),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, frame)
            key = cv2.waitKeyEx(1)
            key_char = key & 0xFF
            if key_char in (27, ord("z"), ord("Z")):
                break
            if key_char == ord("p"):
                projection_index = (projection_index + 1) % len(PROJECTION_MODES)
                controller.projection_mode = PROJECTION_MODES[projection_index]
                print(f"[LEAP] projection mode -> {controller.projection_mode}")
            elif key_char == ord("x"):
                controller.flip_x = not controller.flip_x
                print(f"[LEAP] flip_x -> {controller.flip_x}")
            elif key_char == ord("y"):
                controller.flip_y = not controller.flip_y
                print(f"[LEAP] flip_y -> {controller.flip_y}")
            elif key_char == ord("a"):
                controller.calibration_offset_x -= float(args.calibration_step)
            elif key_char == ord("d"):
                controller.calibration_offset_x += float(args.calibration_step)
            elif key_char == ord("w"):
                controller.calibration_offset_y -= float(args.calibration_step)
            elif key_char == ord("s"):
                controller.calibration_offset_y += float(args.calibration_step)
            elif key_char == ord("q"):
                controller.calibration_rotation_deg -= float(args.rotation_step)
            elif key_char == ord("e"):
                controller.calibration_rotation_deg += float(args.rotation_step)
            elif key_char in (ord("+"), ord("=")):
                controller.calibration_scale += float(args.scale_step)
            elif key_char in (ord("-"), ord("_")):
                controller.calibration_scale = max(0.05, controller.calibration_scale - float(args.scale_step))
            elif key_char == ord("r"):
                controller.calibration_offset_x = float(args.calibration_offset_x)
                controller.calibration_offset_y = float(args.calibration_offset_y)
                controller.calibration_scale = float(args.calibration_scale)
                controller.calibration_rotation_deg = float(args.calibration_rotation_deg)
                print("[LEAP] calibration reset")
            elif key_char == ord("m"):
                modes = ("raw", "rectified", "rectified_transpose")
                next_index = (modes.index(controller.image_mode) + 1) % len(modes) if controller.image_mode in modes else 0
                controller.image_mode = modes[next_index]
                print(f"[LEAP] image_mode -> {controller.image_mode}")
            if key_char in (
                ord("a"),
                ord("d"),
                ord("w"),
                ord("s"),
                ord("q"),
                ord("e"),
                ord("+"),
                ord("="),
                ord("-"),
                ord("_"),
            ):
                print(
                    "[LEAP] calibration "
                    f"x={controller.calibration_offset_x:.1f} "
                    f"y={controller.calibration_offset_y:.1f} "
                    f"scale={controller.calibration_scale:.3f} "
                    f"rot={controller.calibration_rotation_deg:.1f}"
                )
            now = time.time()
            if now - last_status > 2.0:
                print(f"[LEAP] {controller.message} | images={controller.image_count} tracking={controller.tracking_count}")
                last_status = now
    finally:
        controller.close()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
