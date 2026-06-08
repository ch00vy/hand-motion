from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDK = PROJECT_ROOT / "Ultraleap" / "LeapSDK"
LEAP_BINDINGS_ROOT = PROJECT_ROOT / "leapc-python-bindings"
LEAP_API_SRC = LEAP_BINDINGS_ROOT / "leapc-python-api" / "src"
LEAP_CFFI_SRC = LEAP_BINDINGS_ROOT / "leapc-cffi" / "src"


def _configure_sdk_path() -> None:
    if "LEAPSDK_INSTALL_LOCATION" not in os.environ and DEFAULT_SDK.exists():
        os.environ["LEAPSDK_INSTALL_LOCATION"] = str(DEFAULT_SDK)
    # Prefer the local LeapC Python bindings checked into this workspace.
    # This makes the GUI independent from whichever `leap` package happens to
    # be installed globally in the active virtual environment.
    import sys

    for path in (LEAP_CFFI_SRC, LEAP_API_SRC):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


class LeapCameraController:
    """Small Ultraleap IR image-stream adapter for the GUI.

    This is intentionally separate from the qpos provider. The GUI can use it
    only as a visual diagnostic panel while recording and analysis keep using
    the existing tracking/qpos path.
    """

    def __init__(
        self,
        *,
        projection_mode: str = "inverse_neg_y",
        flip_x: bool = False,
        flip_y: bool = False,
        calibration_offset_x: float = -26.0,
        calibration_offset_y: float = 0.0,
        calibration_scale: float = 1.0,
        calibration_rotation_deg: float = -3.0,
        image_mode: str = "rectified_transpose",
        use_images: bool = False,
    ):
        self.connection = None
        self.listener = None
        self.projection_mode = str(projection_mode)
        self.flip_x = bool(flip_x)
        self.flip_y = bool(flip_y)
        self.calibration_offset_x = float(calibration_offset_x)
        self.calibration_offset_y = float(calibration_offset_y)
        self.calibration_scale = float(calibration_scale)
        self.calibration_rotation_deg = float(calibration_rotation_deg)
        self.image_mode = str(image_mode)
        self.use_images = bool(use_images)
        self.lock = threading.Lock()
        self.left_image: Optional[np.ndarray] = None
        self.right_image: Optional[np.ndarray] = None
        self.skeleton_points: Optional[np.ndarray] = None
        self.skeleton_pixels_left: Optional[np.ndarray] = None
        self.skeleton_pixels_right: Optional[np.ndarray] = None
        self.skeleton_edges: list[tuple[int, int]] = []
        self.tracking_cache: OrderedDict[int, tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], list[tuple[int, int]]]] = OrderedDict()
        self._distortion_maps: dict[tuple[int, int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.latest_image_frame_id: Optional[int] = None
        self.image_count = 0
        self.tracking_count = 0
        self.available = False
        self.message = "Leap camera not started."

    def start(self) -> bool:
        _configure_sdk_path()
        try:
            from leap import Connection
            from leap.event_listener import Listener
            if self.use_images:
                from leap.enums import PolicyFlag
            else:
                PolicyFlag = None
        except Exception as exc:
            self.message = f"Leap camera import failed: {type(exc).__name__}: {exc}"
            return False

        owner = self

        class _Listener(Listener):
            def on_connection_event(self, event):
                owner.message = "Leap camera connection opened."

            def on_device_event(self, event):
                owner.message = "Leap camera device detected."

            def on_tracking_event(self, event):
                owner.tracking_count += 1
                owner._handle_tracking_event(event)

            def on_policy_event(self, event):
                owner.message = "Leap image policy active." if owner.use_images else "Leap tracking active."

            def on_image_event(self, event):
                if owner.use_images:
                    owner._handle_image_event(event)

            def on_error(self, error: Exception):
                owner.message = f"Leap camera error: {type(error).__name__}: {error}"

        try:
            self.listener = _Listener()
            self.connection = Connection(listeners=[self.listener])
            self.connection.connect(auto_poll=True)
            if self.use_images and PolicyFlag is not None:
                self.connection.set_policy_flags([PolicyFlag.Images])
            self.available = True
            self.message = "Leap tracking stream starting." if not self.use_images else "Leap camera stream starting."
            return True
        except Exception as exc:
            self.message = f"Leap camera start failed: {type(exc).__name__}: {exc}"
            self.close()
            return False

    def _handle_image_event(self, event) -> None:
        try:
            images = list(getattr(event, "image", []) or [])
            if len(images) < 2:
                return
            left = self._image_to_display_array(images[0], camera_index=0)
            right = self._image_to_display_array(images[1], camera_index=1)
            if left is None or right is None:
                return
            image_info = getattr(getattr(event, "c_data", None), "info", None)
            frame_id = int(getattr(image_info, "frame_id", -1))
            cached = self._find_cached_skeleton(frame_id)
            with self.lock:
                self.left_image = left
                self.right_image = right
                self.latest_image_frame_id = frame_id
                if cached is not None:
                    self.skeleton_points, self.skeleton_pixels_left, self.skeleton_pixels_right, self.skeleton_edges = cached
                self.image_count += 1
                self.message = f"Leap IR camera live | frames={self.image_count}"
        except Exception as exc:
            self.message = f"Leap image decode failed: {type(exc).__name__}: {exc}"

    def _handle_tracking_event(self, event) -> None:
        try:
            hands = list(getattr(event, "hands", []) or [])
            if not hands:
                with self.lock:
                    self.skeleton_points = None
                    self.skeleton_pixels_left = None
                    self.skeleton_pixels_right = None
                    self.skeleton_edges = []
                return
            frame_id = int(getattr(event.info, "frame_id", -1))
            points, edges = self._hand_to_skeleton(hands[0])
            left_pixels = self._project_points_to_camera(points, camera_index=0)
            right_pixels = self._project_points_to_camera(points, camera_index=1)
            with self.lock:
                if frame_id >= 0:
                    self.tracking_cache[frame_id] = (points, left_pixels, right_pixels, edges)
                    while len(self.tracking_cache) > 90:
                        self.tracking_cache.popitem(last=False)
                self.skeleton_points = points
                self.skeleton_pixels_left = left_pixels
                self.skeleton_pixels_right = right_pixels
                self.skeleton_edges = edges
                if not self.use_images:
                    self.message = f"Leap tracking live | frames={self.tracking_count}"
        except Exception:
            pass

    def _find_cached_skeleton(
        self,
        frame_id: int,
    ) -> Optional[tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], list[tuple[int, int]]]]:
        if not self.tracking_cache:
            return None
        if int(frame_id) in self.tracking_cache:
            return self.tracking_cache[int(frame_id)]
        nearest_id = min(self.tracking_cache.keys(), key=lambda item: abs(int(item) - int(frame_id)))
        if abs(int(nearest_id) - int(frame_id)) <= 3:
            return self.tracking_cache[int(nearest_id)]
        return None

    def _hand_to_skeleton(self, hand) -> tuple[np.ndarray, list[tuple[int, int]]]:
        points: list[list[float]] = []
        edges: list[tuple[int, int]] = []

        def add_vec(vec) -> int:
            points.append([float(vec.x), float(vec.y), float(vec.z)])
            return len(points) - 1

        palm_idx = add_vec(hand.palm.position)
        try:
            arm0 = add_vec(hand.arm.prev_joint)
            arm1 = add_vec(hand.arm.next_joint)
            edges.append((arm0, arm1))
            edges.append((arm1, palm_idx))
        except Exception:
            pass
        for digit in list(hand.digits):
            previous = palm_idx
            for bone in list(digit.bones):
                p0 = add_vec(bone.prev_joint)
                p1 = add_vec(bone.next_joint)
                edges.append((previous, p0))
                edges.append((p0, p1))
                previous = p1
        return np.asarray(points, dtype=float), edges

    def _project_points_to_camera(self, points: np.ndarray, *, camera_index: int) -> Optional[np.ndarray]:
        if self.connection is None or points.size == 0:
            return None
        try:
            from leapc_cffi import ffi, libleapc
        except Exception:
            return None
        try:
            connection_ptr = self.connection.get_connection_ptr()
            extrinsic_buf = ffi.new("float[]", 16)
            scale_offset_buf = ffi.new("float[]", 16)
            libleapc.LeapExtrinsicCameraMatrixByIndex(connection_ptr, int(camera_index), extrinsic_buf)
            libleapc.LeapScaleOffsetMatrixByIndex(connection_ptr, int(camera_index), scale_offset_buf)
            extrinsic = np.asarray([float(extrinsic_buf[i]) for i in range(16)], dtype=float).reshape((4, 4), order="F")
            scale_offset = np.asarray([float(scale_offset_buf[i]) for i in range(16)], dtype=float).reshape((4, 4), order="F")
            mode = str(self.projection_mode).strip().lower()
            if mode.startswith("forward"):
                transform = extrinsic
            else:
                transform = np.linalg.inv(extrinsic)
        except Exception:
            return None

        pixels = np.full((int(points.shape[0]), 2), np.nan, dtype=float)
        for idx, point in enumerate(np.asarray(points, dtype=float)):
            if not np.all(np.isfinite(point)):
                continue
            try:
                camera_point = transform @ np.asarray([point[0], point[1], point[2], 1.0], dtype=float)
                ray_xy = self._camera_point_to_ray(camera_point, mode)
                if ray_xy is None:
                    continue
                image_point = scale_offset @ np.asarray([float(ray_xy[0]), 0.0, float(ray_xy[1]), 1.0], dtype=float)
                pixels[idx, 0] = float(image_point[0])
                pixels[idx, 1] = float(image_point[2])
            except Exception:
                continue
        return pixels

    def _camera_point_to_ray(self, camera_point: np.ndarray, mode: str) -> Optional[tuple[float, float]]:
        axis = "z"
        sign = 1.0
        if "_neg_" in mode:
            sign = -1.0
        if mode.endswith("_y") or "_y" in mode:
            axis = "y"
        elif mode.endswith("_x") or "_x" in mode:
            axis = "x"
        denom_index = {"x": 0, "y": 1, "z": 2}[axis]
        denom = float(camera_point[denom_index]) * sign
        if abs(denom) < 1e-6:
            return None
        if axis == "z":
            return float(camera_point[0] / denom), float(camera_point[1] / denom)
        if axis == "y":
            return float(camera_point[0] / denom), float(camera_point[2] / denom)
        return float(camera_point[1] / denom), float(camera_point[2] / denom)

    def _image_to_display_array(self, image, *, camera_index: int) -> Optional[np.ndarray]:
        try:
            from leapc_cffi import ffi
        except Exception:
            return None
        data = getattr(image, "c_data", None)
        props = getattr(data, "properties", None)
        if data is None or props is None:
            return None
        width = int(props.width)
        height = int(props.height)
        bpp = int(props.bpp)
        if width <= 0 or height <= 0 or bpp <= 0:
            return None
        offset = int(getattr(data, "offset", 0))
        byte_count = width * height * bpp
        raw = bytes(ffi.buffer(data.data + offset, byte_count))
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, bpp)
        if bpp == 1:
            raw_image = arr[:, :, 0].copy()
        else:
            raw_image = arr[:, :, 0].copy()
        if str(self.image_mode).strip().lower() == "raw":
            return raw_image
        rectified = self._rectify_image(raw_image, image=image, camera_index=int(camera_index), width=width, height=height)
        return rectified if rectified is not None else raw_image

    def _rectify_image(self, raw_image: np.ndarray, *, image, camera_index: int, width: int, height: int) -> Optional[np.ndarray]:
        try:
            from leapc_cffi import ffi
        except Exception:
            return None
        data = getattr(image, "c_data", None)
        distortion_matrix = getattr(data, "distortion_matrix", None)
        if distortion_matrix in (None, ffi.NULL):
            return None
        matrix_version = int(getattr(data, "matrix_version", 0))
        mode = str(self.image_mode).strip().lower()
        cache_key = (int(camera_index), matrix_version, int(width), int(height), 1 if mode.endswith("transpose") else 0)
        maps = self._distortion_maps.get(cache_key)
        if maps is None:
            grid_x = np.empty((64, 64), dtype=np.float32)
            grid_y = np.empty((64, 64), dtype=np.float32)
            try:
                for row in range(64):
                    for col in range(64):
                        point = distortion_matrix.matrix[col][row] if mode.endswith("transpose") else distortion_matrix.matrix[row][col]
                        grid_x[row, col] = float(point.x)
                        grid_y[row, col] = float(point.y)
            except Exception:
                return None
            if float(np.nanmax(grid_x)) <= 4.0 and float(np.nanmax(grid_y)) <= 4.0:
                grid_x *= float(width)
                grid_y *= float(height)
            map_x = cv2.resize(grid_x, (int(width), int(height)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            map_y = cv2.resize(grid_y, (int(width), int(height)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            maps = (map_x, map_y)
            self._distortion_maps[cache_key] = maps
        map_x, map_y = maps
        return cv2.remap(
            np.asarray(raw_image, dtype=np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def latest_stereo(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self.lock:
            if self.left_image is None or self.right_image is None:
                return None
            return self.left_image.copy(), self.right_image.copy()

    def latest_view(self, *, camera: str = "left") -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[tuple[int, int]]]:
        with self.lock:
            use_right = str(camera).lower() == "right"
            image = self.right_image if use_right else self.left_image
            if image is None:
                image_out = None
            else:
                image_out = image.copy()
            projected = self.skeleton_pixels_right if use_right else self.skeleton_pixels_left
            if projected is not None and np.any(np.isfinite(projected)):
                skeleton_out = projected.copy()
                if image_out is not None:
                    height, width = image_out.shape[:2]
                    if self.flip_x:
                        skeleton_out[:, 0] = float(width) - skeleton_out[:, 0]
                    if self.flip_y:
                        skeleton_out[:, 1] = float(height) - skeleton_out[:, 1]
                    center = np.asarray([float(width) * 0.5, float(height) * 0.5], dtype=float)
                    centered = (skeleton_out - center) * self.calibration_scale
                    angle = np.deg2rad(float(self.calibration_rotation_deg))
                    cos_a = float(np.cos(angle))
                    sin_a = float(np.sin(angle))
                    rot = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
                    skeleton_out = (centered @ rot.T) + center
                    skeleton_out[:, 0] += self.calibration_offset_x
                    skeleton_out[:, 1] += self.calibration_offset_y
            elif self.skeleton_points is None:
                skeleton_out = None
            else:
                skeleton_out = self.skeleton_points.copy()
            return image_out, skeleton_out, list(self.skeleton_edges)

    def latest_skeleton(self) -> tuple[Optional[np.ndarray], list[tuple[int, int]]]:
        with self.lock:
            points = None if self.skeleton_points is None else self.skeleton_points.copy()
            return points, list(self.skeleton_edges)

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.disconnect()
            except Exception:
                pass
        self.connection = None
        self.listener = None
        self.available = False
