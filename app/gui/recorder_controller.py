from __future__ import annotations

import math
import json
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app.gui.trial_loader import DEFAULT_RECORDS_ROOT, TrialData, save_trial
from app.realtime.rokoko_mapping import RokokoHandMapper, hand_points as _rokoko_hand_points


QposProvider = Callable[[], Optional[np.ndarray]]


@dataclass
class RecordingStatus:
    active: bool
    frame_count: int
    source_mode: str
    elapsed_seconds: float


@dataclass
class QposProviderStatus:
    available: bool
    has_qpos: bool
    qpos_dim: int
    samples_seen: int
    latest_age_seconds: float
    unchanged_seconds: float
    delta_norm: float
    stuck: bool
    message: str


class TrialRecorder:
    """Small qpos recorder with a provider hook for Leap/realtime integration.

    If no provider is connected, it records a deterministic synthetic
    open-to-fist-like qpos trajectory so the GUI workflow can still be smoke
    tested end-to-end.
    """

    def __init__(self, *, qpos_provider: Optional[QposProvider] = None, qpos_dim: int = 23, dt: float = 1.0 / 60.0):
        self.qpos_provider = qpos_provider
        self.qpos_dim = max(1, int(qpos_dim))
        self.dt = float(dt)
        self._frames: list[np.ndarray] = []
        self._last_qpos: Optional[np.ndarray] = None
        self._start_time = 0.0
        self._active = False
        self._source_mode = "idle"

    def set_qpos_provider(self, provider: Optional[QposProvider]) -> None:
        self.qpos_provider = provider

    def start(self) -> None:
        self._frames = []
        self._last_qpos = None
        self._start_time = time.perf_counter()
        self._active = True
        self._source_mode = "provider" if self.qpos_provider is not None else "synthetic_fallback"

    def append_current_frame(self) -> None:
        if not self._active:
            return
        qpos = None
        if self.qpos_provider is not None:
            try:
                qpos = self.qpos_provider()
            except Exception:
                qpos = None
        if qpos is None:
            qpos = self._synthetic_open_to_fist(len(self._frames), self.qpos_dim)
        arr = np.asarray(qpos, dtype=float).reshape(-1)
        if arr.size < self.qpos_dim:
            padded = np.zeros((self.qpos_dim,), dtype=float)
            padded[: arr.size] = arr
            arr = padded
        self._last_qpos = arr[: self.qpos_dim].copy()
        self._frames.append(self._last_qpos.copy())

    def latest_qpos(self) -> Optional[np.ndarray]:
        if self._last_qpos is None:
            return None
        return np.asarray(self._last_qpos, dtype=float).copy()

    def stop_and_save(
        self,
        *,
        subject_id: str,
        motion_type: str,
        trial_label: str = "normal",
        records_root: Path = DEFAULT_RECORDS_ROOT,
    ) -> TrialData:
        if not self._active:
            raise RuntimeError("Recording is not active.")
        self._active = False
        if not self._frames:
            raise RuntimeError("No qpos frames captured.")
        return save_trial(
            qpos=np.stack(self._frames, axis=0),
            subject_id=subject_id,
            motion_type=motion_type,
            trial_label=trial_label,
            dt=self.dt,
            records_root=records_root,
            source_mode=self._source_mode,
        )

    def status(self) -> RecordingStatus:
        elapsed = (time.perf_counter() - self._start_time) if self._active else 0.0
        return RecordingStatus(
            active=bool(self._active),
            frame_count=int(len(self._frames)),
            source_mode=str(self._source_mode),
            elapsed_seconds=float(elapsed),
        )

    @staticmethod
    def _synthetic_open_to_fist(frame_index: int, width: int) -> np.ndarray:
        progress = min(1.0, max(0.0, float(frame_index) / 90.0))
        smooth = 0.5 - 0.5 * math.cos(math.pi * progress)
        qpos = np.zeros((int(width),), dtype=float)
        for idx in range(int(width)):
            gain = 0.10 + 0.015 * float(idx % 5)
            phase = 0.02 * math.sin(0.1 * frame_index + idx)
            qpos[idx] = (gain * smooth) + phase
        return qpos


class LiveLeapQposProvider:
    """Thin adapter around the existing Leap mapper/listener.

    It is intentionally optional: if the Leap device or SDK connection is not
    available, callers can keep using the recorder's synthetic fallback.
    """

    def __init__(
        self,
        *,
        show_leap_view: bool = True,
        window_name: str = "Ultraleap Tracking Preview",
        window_scale: float = 1.0,
    ):
        self.connection = None
        self.listener = None
        self._connection_context = None
        self.canvas = None
        self.show_leap_view = bool(show_leap_view)
        self.window_name = str(window_name)
        self.window_scale = max(0.2, float(window_scale))
        self._preview_window_ready = False
        self.available = False
        self.message = "not started"
        self._last_qpos: Optional[np.ndarray] = None
        self._last_qpos_sample_time = 0.0
        self._last_qpos_change_time = 0.0
        self._last_qpos_delta_norm = 0.0
        self._samples_seen = 0
        self._stuck_after_seconds = 1.25
        self._change_epsilon = 1e-5

    def start(self) -> bool:
        try:
            import leap
            from app.realtime.realtime_io import Canvas, LeapDataListener, LeapToMyoSuiteMapper, setup_leap_motion_device
        except Exception as exc:
            self.message = f"Leap import unavailable: {type(exc).__name__}: {exc}"
            return False
        try:
            tracking_mode = leap.TrackingMode.Desktop
            mapper = LeapToMyoSuiteMapper(desktop_mode_transform=True)
            self.canvas = Canvas() if self.show_leap_view else None
            if self.canvas is not None:
                self.canvas.name = self.window_name
            self.listener = LeapDataListener(mapper=mapper, canvas=self.canvas)
            self.connection = leap.Connection()
            self.connection.add_listener(self.listener)
            self._connection_context = self.connection.open()
            self._connection_context.__enter__()
            self.connection.set_tracking_mode(tracking_mode)
            mapper.desktop_mode_transform = True
            self.available = bool(setup_leap_motion_device(self.connection))
            self.message = "Leap qpos provider active." if self.available else "No Leap Motion device detected."
            return bool(self.available)
        except Exception as exc:
            self.message = f"Leap provider failed: {type(exc).__name__}: {exc}"
            self.close()
            return False

    def __call__(self) -> Optional[np.ndarray]:
        if self.listener is None:
            return None
        qpos = getattr(self.listener, "latest_qpos", None)
        if qpos is None:
            return None
        arr = np.asarray(qpos, dtype=float).reshape(-1)
        self._register_qpos_sample(arr)
        return arr

    def latest_qpos(self) -> Optional[np.ndarray]:
        return self()

    def _register_qpos_sample(self, qpos: np.ndarray) -> None:
        now = time.perf_counter()
        arr = np.asarray(qpos, dtype=float).reshape(-1)
        self._samples_seen += 1
        self._last_qpos_sample_time = now
        if self._last_qpos is None or self._last_qpos.shape != arr.shape:
            self._last_qpos_delta_norm = 0.0
            self._last_qpos_change_time = now
        else:
            delta = float(np.linalg.norm(arr - self._last_qpos))
            self._last_qpos_delta_norm = delta
            if delta > self._change_epsilon:
                self._last_qpos_change_time = now
        self._last_qpos = arr.copy()

    def status(self) -> QposProviderStatus:
        now = time.perf_counter()
        has_qpos = self._last_qpos is not None
        latest_age = (now - self._last_qpos_sample_time) if self._last_qpos_sample_time > 0.0 else float("inf")
        unchanged = (now - self._last_qpos_change_time) if self._last_qpos_change_time > 0.0 else float("inf")
        stuck = bool(self.available and has_qpos and unchanged >= self._stuck_after_seconds)
        if not self.available:
            message = self.message
        elif not has_qpos:
            message = "Leap qpos provider active, waiting for mapped qpos."
        elif stuck:
            message = f"QPOS STUCK: no mapped qpos change for {unchanged:.1f}s."
        else:
            message = f"QPOS live: dim={self._last_qpos.size} delta={self._last_qpos_delta_norm:.5f}."
        return QposProviderStatus(
            available=bool(self.available),
            has_qpos=bool(has_qpos),
            qpos_dim=int(self._last_qpos.size if has_qpos else 0),
            samples_seen=int(self._samples_seen),
            latest_age_seconds=float(latest_age),
            unchanged_seconds=float(unchanged),
            delta_norm=float(self._last_qpos_delta_norm),
            stuck=stuck,
            message=message,
        )

    def show_preview(self) -> None:
        if self.canvas is None:
            return
        try:
            import cv2

            if not self._preview_window_ready:
                flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
                    flags |= cv2.WINDOW_GUI_EXPANDED
                cv2.namedWindow(self.canvas.name, flags)
                height, width = self.canvas.output_image.shape[:2]
                cv2.resizeWindow(
                    self.canvas.name,
                    max(320, int(width * self.window_scale)),
                    max(240, int(height * self.window_scale)),
                )
                self._preview_window_ready = True
            cv2.imshow(self.canvas.name, self.canvas.output_image)
            cv2.waitKey(1)
        except Exception:
            pass

    def close(self) -> None:
        if self.connection is not None and self.listener is not None:
            try:
                self.connection.remove_listener(self.listener)
            except Exception:
                pass
        if self._connection_context is not None:
            try:
                self._connection_context.__exit__(None, None, None)
            except Exception:
                pass
        self.connection = None
        self.listener = None
        self._connection_context = None
        if self.canvas is not None:
            try:
                import cv2

                cv2.destroyWindow(self.canvas.name)
            except Exception:
                pass
        self.canvas = None
        self._preview_window_ready = False
        self.available = False
        self._last_qpos = None
        self._last_qpos_sample_time = 0.0
        self._last_qpos_change_time = 0.0
        self._last_qpos_delta_norm = 0.0
        self._samples_seen = 0


class LeapCQposProvider:
    """Direct LeapC CFFI qpos provider running inside the GUI process."""

    def __init__(
        self,
        *,
        sdk_dir: str = r"C:\Program Files\Ultraleap\LeapSDK",
        qpos_dim: int = 23,
        tracking_mode: str = "auto",
        poll_timeout_ms: int = 8,
    ):
        self.sdk_dir = str(sdk_dir)
        self.qpos_dim = max(1, int(qpos_dim))
        self.tracking_mode = str(tracking_mode).strip().lower() or "auto"
        self._tracking_modes_to_try = (
            ("desktop", "screentop", "hmd") if self.tracking_mode == "auto" else (self.tracking_mode,)
        )
        self._active_tracking_mode = self._tracking_modes_to_try[0]
        self._last_mode_switch_time = 0.0
        self.poll_timeout_ms = max(1, int(poll_timeout_ms))
        self.available = False
        self.message = "not started"
        self._ffi = None
        self._lib = None
        self._connection = None
        self._message = None
        self._mapper = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_qpos: Optional[np.ndarray] = None
        self._last_qpos_sample_time = 0.0
        self._last_qpos_change_time = 0.0
        self._last_qpos_delta_norm = 0.0
        self._samples_seen = 0
        self._last_error = ""
        self._hand_missing_since = 0.0
        self._last_reset_time = 0.0
        self._last_reset_reason = ""
        self._stuck_after_seconds = 1.25
        self._stale_after_seconds = 0.75
        self._tracking_loss_reset_seconds = 0.35
        self._min_reset_interval_seconds = 0.75
        self._change_epsilon = 1e-5

    def start(self) -> bool:
        self.close()
        try:
            from app.realtime.leapc_qpos_publisher import LeapCHandToQposMapper, _load_leapc_cffi, _open_connection

            self._ffi, self._lib = _load_leapc_cffi(Path(self.sdk_dir))
            self._active_tracking_mode = self._tracking_modes_to_try[0]
            self._connection = _open_connection(self._ffi, self._lib, tracking_mode=self._active_tracking_mode)
            self._message = self._ffi.new("LEAP_CONNECTION_MESSAGE *")
            self._message.size = self._ffi.sizeof("LEAP_CONNECTION_MESSAGE")
            self._mapper = LeapCHandToQposMapper(desktop_mode_transform=True)
            self._stop_event.clear()
            self.available = True
            self._last_mode_switch_time = time.perf_counter()
            mode_text = "auto" if self.tracking_mode == "auto" else self._active_tracking_mode
            self.message = f"Direct LeapC qpos provider active. tracking={mode_text}"
            self._thread = threading.Thread(target=self._poll_loop, name="LeapC-QPOS", daemon=True)
            self._thread.start()
            return True
        except Exception as exc:
            self.message = f"Direct LeapC provider failed: {type(exc).__name__}: {exc}"
            self.close()
            return False

    def __call__(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last_qpos is None:
                return None
            return self._last_qpos.copy()

    def latest_qpos(self) -> Optional[np.ndarray]:
        return self()

    def _poll_loop(self) -> None:
        assert self._ffi is not None
        assert self._lib is not None
        assert self._connection is not None
        assert self._message is not None
        assert self._mapper is not None
        lib = self._lib
        ffi = self._ffi
        while not self._stop_event.is_set():
            try:
                result = lib.LeapPollConnection(self._connection, int(self.poll_timeout_ms), self._message)
                if result == lib.eLeapRS_Timeout:
                    self._maybe_cycle_tracking_mode()
                    continue
                if result != lib.eLeapRS_Success:
                    self._last_error = f"LeapPollConnection returned {int(result)}"
                    self._maybe_cycle_tracking_mode()
                    time.sleep(0.01)
                    continue
                if self._message.type != lib.eLeapEventType_Tracking:
                    self._maybe_cycle_tracking_mode()
                    continue
                event = self._message.tracking_event
                if event == ffi.NULL or int(event.nHands) <= 0:
                    self._handle_no_hands()
                    continue
                self._hand_missing_since = 0.0
                qpos = self._mapper.map_hand(event.pHands[0])
                arr = np.asarray(qpos, dtype=float).reshape(-1)
                if arr.size <= 0 or not np.all(np.isfinite(arr)):
                    continue
                if arr.size != self.qpos_dim:
                    fixed = np.zeros((self.qpos_dim,), dtype=float)
                    fixed[: min(arr.size, self.qpos_dim)] = arr[: self.qpos_dim]
                    arr = fixed
                self._register_qpos_sample(arr)
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.02)

    def _maybe_cycle_tracking_mode(self) -> None:
        if len(self._tracking_modes_to_try) <= 1 or self._samples_seen > 0:
            return
        now = time.perf_counter()
        if (now - self._last_mode_switch_time) < 1.5:
            return
        if self._lib is None or self._connection is None:
            return
        try:
            from app.realtime.leapc_qpos_publisher import _configure_tracking_policy

            current_index = self._tracking_modes_to_try.index(self._active_tracking_mode)
            next_mode = self._tracking_modes_to_try[(current_index + 1) % len(self._tracking_modes_to_try)]
            _configure_tracking_policy(self._lib, self._connection, tracking_mode=next_mode)
            self._active_tracking_mode = next_mode
            self._last_mode_switch_time = now
            self._last_reset_reason = f"tracking_mode_{next_mode}"
            if self._mapper is not None:
                self._mapper.reset_tracking_state()
        except Exception as exc:
            self._last_error = f"tracking mode switch failed: {type(exc).__name__}: {exc}"

    def _handle_no_hands(self) -> None:
        now = time.perf_counter()
        if self._hand_missing_since <= 0.0:
            self._hand_missing_since = now
            return
        if self._mapper is None or self._samples_seen <= 0:
            return
        if (now - self._hand_missing_since) < self._tracking_loss_reset_seconds:
            return
        if (now - self._last_reset_time) < self._min_reset_interval_seconds:
            return
        try:
            self._mapper.reset_tracking_state()
            self._last_reset_time = now
            self._last_reset_reason = "tracking_lost"
        except Exception as exc:
            self._last_error = f"mapping reset failed: {type(exc).__name__}: {exc}"

    def _register_qpos_sample(self, qpos: np.ndarray) -> None:
        now = time.perf_counter()
        arr = np.asarray(qpos, dtype=float).reshape(-1)
        with self._lock:
            self._samples_seen += 1
            self._last_qpos_sample_time = now
            if self._last_qpos is None or self._last_qpos.shape != arr.shape:
                self._last_qpos_delta_norm = 0.0
                self._last_qpos_change_time = now
            else:
                delta = float(np.linalg.norm(arr - self._last_qpos))
                self._last_qpos_delta_norm = delta
                if delta > self._change_epsilon:
                    self._last_qpos_change_time = now
            self._last_qpos = arr.copy()

    def status(self) -> QposProviderStatus:
        now = time.perf_counter()
        with self._lock:
            last_qpos = None if self._last_qpos is None else self._last_qpos.copy()
            samples_seen = int(self._samples_seen)
            sample_time = float(self._last_qpos_sample_time)
            change_time = float(self._last_qpos_change_time)
            delta_norm = float(self._last_qpos_delta_norm)
        has_qpos = last_qpos is not None
        latest_age = (now - sample_time) if sample_time > 0.0 else float("inf")
        unchanged = (now - change_time) if change_time > 0.0 else float("inf")
        stale = bool(self.available and has_qpos and latest_age >= self._stale_after_seconds)
        stuck = bool(self.available and has_qpos and unchanged >= self._stuck_after_seconds)
        if not self.available:
            message = self.message
        elif not has_qpos:
            detail = f" Last error: {self._last_error}" if self._last_error else ""
            message = (
                f"Direct LeapC provider active, waiting for hand tracking qpos "
                f"(tracking={self._active_tracking_mode}).{detail}"
            )
        elif stale:
            message = f"Direct LeapC qpos stale: no frame for {latest_age:.2f}s."
        elif stuck:
            message = f"QPOS STUCK: no mapped qpos change for {unchanged:.1f}s."
        else:
            calib_samples = 0
            grip_score = 0.0
            mapping_mode = "direct"
            try:
                calib_samples = int(getattr(self._mapper.state, "open_calibration_samples", 0))
                mapper_debug = getattr(self._mapper, "last_debug", {}) or {}
                if isinstance(mapper_debug, dict):
                    grip_score = float(mapper_debug.get("grip_score", 0.0))
                    mapping_mode = str(mapper_debug.get("mapping", mapping_mode))
            except Exception:
                calib_samples = 0
            q_min = float(np.min(last_qpos)) if last_qpos.size else 0.0
            q_max = float(np.max(last_qpos)) if last_qpos.size else 0.0
            q0 = float(last_qpos[0]) if last_qpos.size else 0.0
            message = (
                f"Direct LeapC qpos live: dim={last_qpos.size} tracking={self._active_tracking_mode} "
                f"mode={mapping_mode} q0={q0:.3f} delta={delta_norm:.5f} range=[{q_min:.3f},{q_max:.3f}]"
            )
            if mapping_mode != "legacy_pre_cleanup":
                message += f" grip={grip_score:.2f} calib={calib_samples}"
            message += "."
            if self._last_reset_reason:
                message += f" reset={self._last_reset_reason}."
        return QposProviderStatus(
            available=bool(self.available),
            has_qpos=bool(has_qpos),
            qpos_dim=int(last_qpos.size if has_qpos else 0),
            samples_seen=samples_seen,
            latest_age_seconds=float(latest_age),
            unchanged_seconds=float(unchanged),
            delta_norm=delta_norm,
            stuck=bool(stuck or stale),
            message=message,
        )

    def show_preview(self) -> None:
        return None

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass
        self._thread = None
        if self._lib is not None and self._connection is not None:
            try:
                self._lib.LeapCloseConnection(self._connection)
            except Exception:
                pass
            try:
                self._lib.LeapDestroyConnection(self._connection)
            except Exception:
                pass
        self._connection = None
        self._message = None
        self._mapper = None
        self._hand_missing_since = 0.0
        self._last_reset_time = 0.0
        self._last_reset_reason = ""
        self.available = False
        with self._lock:
            self._last_qpos = None
            self._last_qpos_sample_time = 0.0
            self._last_qpos_change_time = 0.0
            self._last_qpos_delta_norm = 0.0
            self._samples_seen = 0


class UdpQposProvider:
    """Receive 23-value hand pose frames from an external UDP publisher."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 15000,
        qpos_dim: int = 23,
        stale_after_seconds: float = 1.0,
    ):
        self.host = str(host)
        self.port = int(port)
        self.qpos_dim = max(1, int(qpos_dim))
        self.stale_after_seconds = max(0.1, float(stale_after_seconds))
        self.socket: Optional[socket.socket] = None
        self.available = False
        self.message = "not started"
        self._last_qpos: Optional[np.ndarray] = None
        self._last_qpos_sample_time = 0.0
        self._last_qpos_change_time = 0.0
        self._last_qpos_delta_norm = 0.0
        self._samples_seen = 0
        self._last_sequence = -1
        self._rokoko_mapper = RokokoHandMapper()
        self._input_kind = ""
        self._change_epsilon = 1e-5

    def start(self) -> bool:
        self.close()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.setblocking(False)
            self.socket = sock
            self.available = True
            self.message = f"UDP qpos receiver active on {self.host}:{self.port}."
            return True
        except Exception as exc:
            self.message = f"UDP qpos receiver failed: {type(exc).__name__}: {exc}"
            self.close()
            return False

    def __call__(self) -> Optional[np.ndarray]:
        self._drain_socket()
        if self._last_qpos is None:
            return None
        return self._last_qpos.copy()

    def latest_qpos(self) -> Optional[np.ndarray]:
        return self()

    def _drain_socket(self) -> None:
        if self.socket is None:
            return
        while True:
            try:
                payload, _addr = self.socket.recvfrom(65535)
            except BlockingIOError:
                return
            except OSError:
                return
            try:
                packet = json.loads(payload.decode("utf-8"))
                packet_type = str(packet.get("type", "")).strip().lower()
                if packet_type in {"hand_qpos", "smart_glove", "glove_qpos"}:
                    values = packet.get("qpos")
                    if values is None:
                        values = packet.get("joint_angles", packet.get("angles"))
                    qpos = np.asarray(values, dtype=float).reshape(-1)
                    input_kind = "qpos"
                elif isinstance(packet.get("scene"), dict):
                    points = _rokoko_hand_points(packet)
                    qpos = self._rokoko_mapper.map_points(points)
                    input_kind = "rokoko"
                else:
                    continue
                if qpos.size <= 0 or not np.all(np.isfinite(qpos)):
                    continue
                if packet_type in {"hand_qpos", "smart_glove", "glove_qpos"} and str(packet.get("units", "radians")).strip().lower() in {"degree", "degrees", "deg"}:
                    qpos = np.deg2rad(qpos)
                arr = np.zeros((self.qpos_dim,), dtype=float)
                width = min(int(qpos.size), self.qpos_dim)
                arr[:width] = qpos[:width]
                self._last_sequence = int(packet.get("seq", self._last_sequence))
                self._register_qpos_sample(arr)
                self._input_kind = input_kind
            except Exception:
                continue

    def calibrate_rokoko_open(self) -> tuple[bool, str]:
        self._drain_socket()
        if self._input_kind != "rokoko":
            return False, "Kalibrasi GUI memerlukan JSON Rokoko langsung (port 14043)."
        points = self._rokoko_mapper.latest_points
        if points is None or time.perf_counter() - self._last_qpos_sample_time >= self.stale_after_seconds:
            return False, "Pose Rokoko tidak tersedia atau stream terhenti."
        neutral = self._rokoko_mapper.calibrate_open(points)
        self._register_qpos_sample(neutral)
        return True, "Pose netral tersimpan. Calibrate Fist opsional untuk penskalaan rentang."

    def calibrate_rokoko_closed(self) -> tuple[bool, str]:
        self._drain_socket()
        if self._input_kind != "rokoko" or time.perf_counter() - self._last_qpos_sample_time >= self.stale_after_seconds:
            return False, "Kalibrasi memerlukan stream JSON Rokoko langsung yang aktif."
        success, message = self._rokoko_mapper.calibrate_closed()
        if success:
            self._register_qpos_sample(self._rokoko_mapper.map_points(self._rokoko_mapper.latest_points))
        return success, message

    def _register_qpos_sample(self, qpos: np.ndarray) -> None:
        now = time.perf_counter()
        arr = np.asarray(qpos, dtype=float).reshape(-1)
        self._samples_seen += 1
        self._last_qpos_sample_time = now
        if self._last_qpos is None or self._last_qpos.shape != arr.shape:
            self._last_qpos_delta_norm = 0.0
            self._last_qpos_change_time = now
        else:
            delta = float(np.linalg.norm(arr - self._last_qpos))
            self._last_qpos_delta_norm = delta
            if delta > self._change_epsilon:
                self._last_qpos_change_time = now
        self._last_qpos = arr.copy()

    def status(self) -> QposProviderStatus:
        self._drain_socket()
        now = time.perf_counter()
        has_qpos = self._last_qpos is not None
        latest_age = (now - self._last_qpos_sample_time) if self._last_qpos_sample_time > 0.0 else float("inf")
        unchanged = (now - self._last_qpos_change_time) if self._last_qpos_change_time > 0.0 else float("inf")
        stale = bool(self.available and has_qpos and latest_age >= self.stale_after_seconds)
        # A held pose is valid live input; only missing packets are stale.
        if not self.available:
            message = self.message
        elif not has_qpos:
            message = f"UDP receiver active on {self.host}:{self.port}, waiting for Rokoko JSON v3 or qpos packets."
        elif stale:
            message = f"UDP qpos stale: no packet for {latest_age:.1f}s."
        else:
            message = f"UDP qpos live: dim={self._last_qpos.size} seq={self._last_sequence} delta={self._last_qpos_delta_norm:.5f}."
        return QposProviderStatus(
            available=bool(self.available),
            has_qpos=bool(has_qpos),
            qpos_dim=int(self._last_qpos.size if has_qpos else 0),
            samples_seen=int(self._samples_seen),
            latest_age_seconds=float(latest_age),
            unchanged_seconds=float(unchanged),
            delta_norm=float(self._last_qpos_delta_norm),
            stuck=stale,
            message=message,
        )

    def show_preview(self) -> None:
        return None

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
        self.socket = None
        self.available = False
        self._last_qpos = None
        self._rokoko_mapper.reset()
        self._input_kind = ""
        self._last_qpos_sample_time = 0.0
        self._last_qpos_change_time = 0.0
        self._last_qpos_delta_norm = 0.0
        self._samples_seen = 0
