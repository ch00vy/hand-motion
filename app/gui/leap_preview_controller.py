from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LeapPreviewStatus:
    running: bool
    message: str
    log_path: Path


class LeapExternalPreviewController:
    """Manage the standalone Ultraleap tracking monitor process."""

    def __init__(self, *, show_image: bool = False, scale: float = 1.6):
        self.show_image = bool(show_image)
        self.scale = float(scale)
        self.process: Optional[subprocess.Popen] = None
        self.log_path = PROJECT_ROOT / "artifacts" / "gui_logs" / "leap_viewer.log"
        self._log_handle = None
        self.message = "Leap viewer not started."

    def start(self) -> bool:
        if self.is_running():
            self.message = "External Leap viewer already running."
            return True
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        script = PROJECT_ROOT / "scripts" / "dev" / (
            "view_leap_camera_overlay.py" if self.show_image else "view_leap_skeleton.py"
        )
        if not script.exists():
            self.message = f"Leap viewer script not found: {script}"
            return False
        args = [
            sys.executable,
            str(script),
            "--scale",
            str(self.scale),
            "--window-name",
            "Ultraleap Tracking Preview",
        ]
        if self.show_image:
            args.append("--show-image")
        try:
            self._log_handle = self.log_path.open("a", encoding="utf-8")
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            self.process = subprocess.Popen(
                args,
                cwd=str(PROJECT_ROOT),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            self.message = "External Leap viewer launched."
            return True
        except Exception as exc:
            self._close_log_handle()
            self.process = None
            self.message = f"Leap viewer launch failed: {type(exc).__name__}: {exc}"
            return False

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass
        self.process = None
        self._close_log_handle()
        self.message = "External Leap viewer stopped."

    def restart(self) -> bool:
        self.stop()
        return self.start()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def status(self) -> LeapPreviewStatus:
        if self.process is not None and self.process.poll() is not None:
            self.message = f"External Leap viewer exited with code {self.process.returncode}."
            self.process = None
            self._close_log_handle()
        return LeapPreviewStatus(running=self.is_running(), message=self.message, log_path=self.log_path)

    def _close_log_handle(self) -> None:
        if self._log_handle is None:
            return
        try:
            self._log_handle.close()
        except Exception:
            pass
        self._log_handle = None

    def close(self) -> None:
        self.stop()
