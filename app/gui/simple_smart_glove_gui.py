from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from app.gui.mujoco_replay_controller import MujocoReplayController
from app.gui.recorder_controller import UdpQposProvider
from app.gui.style import COLORS, FONTS, button, card, configure_dark_theme


class SimpleSmartGloveApp(tk.Tk):
    """Small connection panel for driving the external MuJoCo viewer live."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Smart Glove + MuJoCo Live Preview")
        self.geometry("1100x700")
        self.minsize(860, 560)
        configure_dark_theme(self)

        self.host_var = tk.StringVar(value="127.0.0.1")
        self.port_var = tk.StringVar(value="14043")
        self.status_var = tk.StringVar(value="Disconnected")
        self.viewer_var = tk.StringVar(value="MuJoCo viewer closed")
        self.packet_var = tk.StringVar(value="Packets: 0")
        self.qpos_var = tk.StringVar(value="QPOS: waiting for glove frames")
        self.calibration_var = tk.StringVar(value="Calibration: capture neutral open hand. Fist scaling is optional.")
        self._provider: Optional[UdpQposProvider] = None
        self._packets_seen = 0
        self._last_qpos_sample_count = 0
        self._controller = MujocoReplayController(use_ctrl_estimate_preview=True)

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._tick)

    def _build_layout(self) -> None:
        shell = tk.Frame(self, bg=COLORS["bg"], padx=14, pady=14)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(1, weight=1)

        tk.Label(shell, text="Smart Glove Realtime Control", bg=COLORS["bg"], fg=COLORS["text"], font=FONTS["title"]).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )

        left = card(shell, padx=14, pady=14)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.configure(width=300)
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)

        right = card(shell, padx=14, pady=14)
        right.grid(row=1, column=1, sticky="nsew")

        self._section_title(left, "Connection", "UDP smart glove input")
        self._field(left, "Host", self.host_var, 0)
        self._field(left, "Port", self.port_var, 1)

        actions = tk.Frame(left, bg=COLORS["panel"])
        actions.pack(fill=tk.X, pady=(12, 8))
        actions.columnconfigure((0, 1), weight=1)
        button(actions, "Connect", self.connect, variant="primary").grid(row=0, column=0, sticky="ew", padx=(0, 4))
        button(actions, "Disconnect", self.disconnect, variant="secondary").grid(row=0, column=1, sticky="ew", padx=(4, 0))

        calibration = tk.Frame(left, bg=COLORS["panel"])
        calibration.pack(fill=tk.X, pady=(4, 8))
        calibration.columnconfigure((0, 1), weight=1)
        button(calibration, "Calibrate Neutral", self.calibrate_open, variant="secondary").grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        button(calibration, "Calibrate Fist", self.calibrate_closed, variant="secondary").grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        tk.Label(
            left,
            textvariable=self.calibration_var,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONTS["small"],
            wraplength=260,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 8))

        self._status_row(left, "Status", self.status_var, 0)
        self._status_row(left, "Packets", self.packet_var, 1)
        self._status_row(left, "QPOS", self.qpos_var, 2)

        self._section_title(right, "MuJoCo Model", "The standard MuJoCo viewer opens in a separate window")
        tk.Label(
            right,
            textvariable=self.viewer_var,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONTS["body"],
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 8))
        preview = tk.Frame(right, bg=COLORS["viewer"], highlightbackground=COLORS["border"], highlightthickness=1)
        preview.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            preview,
            text="MuJoCo live model\n\nUse Connect to open the model viewer.\nIncoming glove frames will move the hand in realtime.",
            bg=COLORS["viewer"],
            fg=COLORS["muted"],
            font=FONTS["body"],
            justify=tk.CENTER,
        ).pack(fill=tk.BOTH, expand=True)

    def _section_title(self, parent: tk.Widget, title: str, subtitle: str) -> None:
        tk.Label(parent, text=title, bg=COLORS["panel"], fg=COLORS["text"], font=FONTS["section"]).pack(anchor="w")
        tk.Label(parent, text=subtitle, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).pack(anchor="w", pady=(2, 14))

    def _field(self, parent: tk.Widget, label: str, variable: tk.StringVar, row: int) -> None:
        wrapper = tk.Frame(parent, bg=COLORS["panel"])
        wrapper.pack(fill=tk.X, pady=4)
        tk.Label(wrapper, text=label, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).pack(anchor="w")
        ttk.Entry(wrapper, textvariable=variable).pack(fill=tk.X, pady=(4, 0))

    def _status_row(self, parent: tk.Widget, label: str, variable: tk.StringVar, row: int) -> None:
        row_frame = tk.Frame(parent, bg=COLORS["panel"])
        row_frame.pack(fill=tk.X, pady=3)
        tk.Label(row_frame, text=label, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).pack(side=tk.LEFT)
        tk.Label(row_frame, textvariable=variable, bg=COLORS["panel"], fg=COLORS["text"], font=FONTS["small"], anchor="e").pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def connect(self) -> None:
        host = str(self.host_var.get()).strip() or "127.0.0.1"
        try:
            port = int(str(self.port_var.get()).strip())
            if not 1 <= port <= 65535:
                raise ValueError("port must be between 1 and 65535")
        except Exception as exc:
            self.status_var.set(f"Invalid port: {exc}")
            return

        self.disconnect(update_status=False)
        provider = UdpQposProvider(host=host, port=port, qpos_dim=23)
        if not provider.start():
            self.status_var.set(provider.message)
            return
        self._provider = provider
        self.host_var.set(host)
        self.port_var.set(str(port))
        if not self._controller.open_or_focus_viewer():
            self.status_var.set(f"Connected, viewer failed: {self._controller.message}")
            provider.close()
            self._provider = None
            return
        self.status_var.set(f"Connected ({host}:{port})")
        self.viewer_var.set("MuJoCo viewer open | waiting for glove frames")

    def calibrate_open(self) -> None:
        if self._provider is None:
            self.calibration_var.set("Connect to Rokoko before calibrating.")
            return
        success, message = self._provider.calibrate_rokoko_open()
        self.calibration_var.set(message)
        if success:
            self._controller.render_qpos(self._provider.latest_qpos(), message="SMART_GLOVE NEUTRAL")

    def calibrate_closed(self) -> None:
        if self._provider is None:
            self.calibration_var.set("Connect to Rokoko before calibrating.")
            return
        success, message = self._provider.calibrate_rokoko_closed()
        self.calibration_var.set(message)
        if success:
            self._controller.render_qpos(self._provider.latest_qpos(), message="SMART_GLOVE FULL FIST")

    def disconnect(self, *, update_status: bool = True) -> None:
        if self._provider is not None:
            self._provider.close()
            self._provider = None
        if update_status:
            self.status_var.set("Disconnected")
            self.viewer_var.set("MuJoCo viewer remains open")
            self.qpos_var.set("QPOS: waiting for glove frames")

    def _tick(self) -> None:
        if self._provider is not None:
            qpos = self._provider.latest_qpos()
            provider_status = self._provider.status()
            self.packet_var.set(f"Packets: {provider_status.samples_seen}")
            if qpos is not None and not provider_status.stuck:
                self._controller.render_qpos(qpos, message="SMART_GLOVE LIVE")
                self.qpos_var.set(f"QPOS: live ({len(qpos)} joints)")
                self.viewer_var.set("MuJoCo viewer open | smart glove live")
            elif provider_status.stuck:
                self.qpos_var.set("QPOS: stream stale")
        self.after(33, self._tick)

    def _on_close(self) -> None:
        self.disconnect(update_status=False)
        self._controller.close()
        self.destroy()


def main() -> int:
    app = SimpleSmartGloveApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
