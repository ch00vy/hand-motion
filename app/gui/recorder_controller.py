from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app.gui.trial_loader import DEFAULT_RECORDS_ROOT, TrialData, save_trial


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

    def __init__(self, *, show_leap_view: bool = True):
        self.connection = None
        self.listener = None
        self._connection_context = None
        self.canvas = None
        self.show_leap_view = bool(show_leap_view)
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
        self.available = False
        self._last_qpos = None
        self._last_qpos_sample_time = 0.0
        self._last_qpos_change_time = 0.0
        self._last_qpos_delta_norm = 0.0
        self._samples_seen = 0
