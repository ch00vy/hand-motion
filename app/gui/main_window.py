from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from app.gui.inference_controller import BinaryGCNInferenceController
from app.gui.leap_preview_controller import LeapExternalPreviewController
from app.gui.mujoco_replay_controller import MujocoReplayController
from app.gui.plot_controller import NodeDeviationPlot
from app.gui.recorder_controller import LiveLeapQposProvider, TrialRecorder
from app.gui.replay_mode import ReplayMode
from app.gui.state import GuiState
from app.gui.style import COLORS, FONTS, button, card, configure_dark_theme
from app.gui.trial_loader import TrialData, load_trial


class TrialReplayClassifierApp(tk.Tk):
    def __init__(self, *, show_external_leap_skeleton: bool = True):
        super().__init__()
        self.title("Biomech Trial Analysis - Binary GCN")
        self.geometry("1440x920")
        self.minsize(1180, 760)
        self.resizable(True, True)
        configure_dark_theme(self)
        self.show_external_leap_skeleton = bool(show_external_leap_skeleton)

        self.recorder = TrialRecorder()
        self.leap_provider: Optional[LiveLeapQposProvider] = None
        self.leap_viewer = LeapExternalPreviewController(show_image=False)
        self.replay = MujocoReplayController()
        self.inference = BinaryGCNInferenceController()
        self.current_trial: Optional[TrialData] = None
        self.current_analysis_result: Optional[dict] = None
        self.analysis_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.analysis_running = False
        self._slider_updating = False
        self._session_trials: list[TrialData] = []
        self._auto_stop_after_id: Optional[str] = None
        self._plot_resize_after_id: Optional[str] = None
        self._record_target_seconds = 0.0
        self.state = GuiState.IDLE

        self.subject_var = tk.StringVar(value="subject_gui")
        self.motion_var = tk.StringVar(value="open_to_fist")
        self.auto_stop_enabled_var = tk.BooleanVar(value=True)
        self.record_duration_var = tk.StringVar(value="4.0")
        self.auto_analyze_after_record_var = tk.BooleanVar(value=False)
        self.session_count_var = tk.StringVar(value="0")
        self.last_saved_var = tk.StringVar(value="-")
        self.path_var = tk.StringVar(value="-")
        self.trial_id_var = tk.StringVar(value="-")
        self.frame_count_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Ready.")
        self.replay_status_var = tk.StringVar(value="Replay viewer closed.")
        self.summary_var = tk.StringVar(value="No classification result.")
        self.frame_info_var = tk.StringVar(value="Frame 0 / 0")
        self.total_nodes_var = tk.StringVar(value="0")
        self.within_count_var = tk.StringVar(value="0")
        self.deviating_count_var = tk.StringVar(value="0")
        self.top_nodes_var = tk.StringVar(value="-")
        self.highest_region_var = tk.StringVar(value="-")
        self.analysis_state_var = tk.StringVar(value="NOT_RUN")
        self.replay_mode_var = tk.StringVar(value=ReplayMode.NONE.value)
        self.control_available_var = tk.StringVar(value="NO")
        self.model_var = tk.StringVar(value=Path(str(self.inference.config.model_path)).name)
        self.score_var = tk.StringVar(value="-")
        self.preview_ready_var = tk.StringVar(value="GUI idle. MuJoCo live preview starts automatically.")
        self.leap_viewer_status_var = tk.StringVar(value="Leap viewer: starting")
        self.qpos_status_var = tk.StringVar(value="QPOS provider: starting")
        self.state_var = tk.StringVar(value=self.state.value)

        self._build_layout()
        self.plot_controller = NodeDeviationPlot(self.bar_plot_canvas, self.heatmap_canvas)
        self.plot_controller.draw_empty()
        self._set_status("Ready", "normal")
        self._set_state(GuiState.IDLE)
        self.after(250, self._initialize_live_viewers)
        self.after(33, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind_all("<Control-r>", self._shortcut_start_recording)
        self.bind_all("<Control-s>", self._shortcut_stop_recording)
        self.bind_all("<Control-l>", self._shortcut_load_existing_trial)

    def _build_layout(self) -> None:
        shell = tk.Frame(self, bg=COLORS["bg"], padx=14, pady=12)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        self._build_header(shell)

        body = tk.Frame(shell, bg=COLORS["bg"])
        body.grid(row=1, column=0, sticky="nsew", pady=(12, 10))
        body.columnconfigure(0, weight=3, minsize=300)
        body.columnconfigure(1, weight=5, minsize=430)
        body.columnconfigure(2, weight=4, minsize=330)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=COLORS["bg"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)

        center = tk.Frame(body, bg=COLORS["bg"])
        center.grid(row=0, column=1, sticky="nsew", padx=(0, 12))
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)

        right = tk.Frame(body, bg=COLORS["bg"])
        right.grid(row=0, column=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_trial_control(left)
        self._build_recording_session_panel(left)
        self._build_external_viewer_panel(left)
        self._build_summary_panel(left)
        self._build_top_nodes_panel(left)
        self._build_plot_panel(center)
        self._build_result_table(right)
        self._build_status_bar(shell)

    def _build_header(self, parent: tk.Widget) -> None:
        header = tk.Frame(parent, bg=COLORS["bg"])
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Biomechanical Trial Replay + Binary GCN",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="open_to_fist analytical workflow",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.status_chip = tk.Label(
            header,
            text="READY",
            bg=COLORS["success_soft"],
            fg=COLORS["success"],
            padx=14,
            pady=6,
            font=FONTS["section"],
        )
        self.status_chip.grid(row=0, column=1, rowspan=2, sticky="e")

    def _section_title(self, parent: tk.Widget, title: str, subtitle: str = "") -> None:
        tk.Label(parent, text=title, bg=COLORS["panel"], fg=COLORS["text"], font=FONTS["section"]).pack(anchor="w")
        if subtitle:
            tk.Label(parent, text=subtitle, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).pack(anchor="w", pady=(2, 8))
        else:
            tk.Frame(parent, bg=COLORS["panel"], height=8).pack(fill=tk.X)

    def _build_trial_control(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=14, pady=14)
        panel.pack(fill=tk.X, pady=(0, 12))
        panel.columnconfigure(1, weight=1)
        self._section_title(panel, "Trial Control Panel", "Record, load, replay, and analyze one transition trial")

        form = tk.Frame(panel, bg=COLORS["panel"])
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)
        self._form_label(form, "Subject ID", 0)
        ttk.Entry(form, textvariable=self.subject_var).grid(row=0, column=1, sticky="ew", pady=4)
        self._form_label(form, "Motion Type", 1)
        motion = ttk.Combobox(form, textvariable=self.motion_var, values=("open_to_fist", "fist_to_open"), state="readonly")
        motion.grid(row=1, column=1, sticky="ew", pady=4)

        actions = tk.Frame(panel, bg=COLORS["panel"])
        actions.pack(fill=tk.X, pady=(10, 8))
        actions.columnconfigure((0, 1), weight=1)
        self.start_button = button(actions, "Start Recording", self.start_recording, variant="primary")
        self.stop_button = button(actions, "Stop Recording", self.stop_recording, variant="danger")
        self.load_button = button(actions, "Load Existing Trial", self.load_existing_trial, variant="secondary")
        self.run_button = button(actions, "Run Analysis", self.run_analysis, variant="warning")
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=4)
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=4)
        self.load_button.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=4)
        self.run_button.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=4)
        self.stop_button.configure(state=tk.DISABLED)

        info = tk.Frame(panel, bg=COLORS["panel"])
        info.pack(fill=tk.X, pady=(4, 0))
        info.columnconfigure(1, weight=1)
        self._info_row(info, "trial_id", self.trial_id_var, 0)
        self._info_row(info, "frames", self.frame_count_var, 1)
        self._info_row(info, "qpos path", self.path_var, 2)

    def _build_recording_session_panel(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=14, pady=14)
        panel.pack(fill=tk.X, pady=(0, 12))
        panel.columnconfigure(0, weight=1)
        self._section_title(panel, "Recording Session", "Record many trials without restarting the GUI")

        controls = tk.Frame(panel, bg=COLORS["panel"])
        controls.pack(fill=tk.X)
        controls.columnconfigure(1, weight=1)
        tk.Checkbutton(
            controls,
            text="Auto-stop after",
            variable=self.auto_stop_enabled_var,
            bg=COLORS["panel"],
            activebackground=COLORS["panel"],
            fg=COLORS["text"],
            activeforeground=COLORS["text"],
            selectcolor=COLORS["input"],
            font=FONTS["small"],
        ).grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(controls, textvariable=self.record_duration_var, width=8).grid(row=0, column=1, sticky="w", padx=(8, 4), pady=3)
        tk.Label(controls, text="seconds", bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).grid(row=0, column=2, sticky="w", pady=3)
        tk.Checkbutton(
            controls,
            text="Run analysis after each saved trial",
            variable=self.auto_analyze_after_record_var,
            bg=COLORS["panel"],
            activebackground=COLORS["panel"],
            fg=COLORS["text"],
            activeforeground=COLORS["text"],
            selectcolor=COLORS["input"],
            font=FONTS["small"],
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 6))

        actions = tk.Frame(panel, bg=COLORS["panel"])
        actions.pack(fill=tk.X, pady=(2, 8))
        actions.columnconfigure((0, 1), weight=1)
        self.record_next_button = button(actions, "Record Next Trial", self.start_recording, variant="success")
        self.load_session_button = button(actions, "Load Selected Session Trial", self.load_selected_session_trial, variant="secondary")
        self.record_next_button.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=3)
        self.load_session_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=3)

        status = tk.Frame(panel, bg=COLORS["panel"])
        status.pack(fill=tk.X)
        status.columnconfigure(1, weight=1)
        self._info_row(status, "saved this session", self.session_count_var, 0)
        self._info_row(status, "last saved", self.last_saved_var, 1)

        list_wrap = tk.Frame(panel, bg=COLORS["panel"])
        list_wrap.pack(fill=tk.X, pady=(8, 0))
        list_wrap.columnconfigure(0, weight=1)
        self.session_trial_list = tk.Listbox(
            list_wrap,
            height=4,
            bg=COLORS["input"],
            fg=COLORS["text"],
            selectbackground=COLORS["accent"],
            selectforeground=COLORS["text"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            relief=tk.FLAT,
            font=FONTS["mono"],
        )
        session_scroll = ttk.Scrollbar(list_wrap, orient=tk.VERTICAL, command=self.session_trial_list.yview)
        self.session_trial_list.configure(yscrollcommand=session_scroll.set)
        self.session_trial_list.grid(row=0, column=0, sticky="ew")
        session_scroll.grid(row=0, column=1, sticky="ns")
        self.session_trial_list.bind("<Double-Button-1>", lambda _event: self.load_selected_session_trial())

    def _build_external_viewer_panel(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=14, pady=14)
        panel.pack(fill=tk.X, pady=(0, 12))
        self._section_title(panel, "External Viewers", "Leap and MuJoCo stay outside Tkinter; GUI only controls/statuses them")

        tk.Label(panel, textvariable=self.leap_viewer_status_var, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).pack(anchor="w", pady=(0, 6))
        viewer_actions = tk.Frame(panel, bg=COLORS["panel"])
        viewer_actions.pack(fill=tk.X, pady=(0, 8))
        self.open_leap_button = button(viewer_actions, "Open Leap Viewer", self.open_leap_viewer, variant="secondary")
        self.close_leap_button = button(viewer_actions, "Stop Leap Viewer", self.close_leap_viewer, variant="secondary")
        self.reset_qpos_button = button(viewer_actions, "Reset QPOS Provider", self.reset_qpos_provider, variant="secondary")
        self.open_leap_button.pack(side=tk.LEFT, padx=(0, 8), pady=3)
        self.close_leap_button.pack(side=tk.LEFT, padx=(0, 8), pady=3)
        self.reset_qpos_button.pack(side=tk.LEFT, pady=3)

        tk.Label(panel, textvariable=self.qpos_status_var, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"], wraplength=300, justify=tk.LEFT).pack(
            anchor="w", pady=(0, 8)
        )
        replay_actions = tk.Frame(panel, bg=COLORS["panel"])
        replay_actions.pack(fill=tk.X, pady=(0, 8))
        replay_actions.columnconfigure((0, 1), weight=1)
        self.open_replay_button = button(replay_actions, "Open MuJoCo Viewer", self.open_replay_viewer, variant="secondary")
        self.close_replay_button = button(replay_actions, "Close Replay Viewer", self.close_replay_viewer, variant="secondary")
        self.replay_kinematic_button = button(replay_actions, "Replay Recorded Kinematics", self.replay_recorded_kinematics, variant="primary")
        self.replay_control_button = button(replay_actions, "Replay Estimated Control", self.replay_estimated_control, variant="warning")
        self.play_button = button(replay_actions, "Play Active Replay", self.play_replay, variant="primary")
        self.pause_button = button(replay_actions, "Pause", self.pause_replay, variant="secondary")
        self.reset_button = button(replay_actions, "Reset Active Replay", self.reset_replay, variant="secondary")
        self.open_replay_button.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=3)
        self.close_replay_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=3)
        self.replay_kinematic_button.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=3)
        self.replay_control_button.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=3)
        self.play_button.grid(row=2, column=0, sticky="ew", padx=(0, 6), pady=3)
        self.pause_button.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=3)
        self.reset_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=3)

        self.progress_scale = ttk.Scale(panel, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_slider)
        self.progress_scale.pack(fill=tk.X, pady=(2, 4))
        tk.Label(panel, textvariable=self.frame_info_var, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["mono"]).pack(anchor="e")

    def _form_label(self, parent: tk.Widget, text: str, row: int) -> None:
        tk.Label(parent, text=text, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).grid(row=row, column=0, sticky="w", padx=(0, 10))

    def _info_row(self, parent: tk.Widget, label: str, var: tk.StringVar, row: int) -> None:
        tk.Label(parent, text=label, bg=COLORS["panel"], fg=COLORS["muted_dark"], font=FONTS["small"]).grid(row=row, column=0, sticky="nw", pady=2)
        tk.Label(parent, textvariable=var, bg=COLORS["panel"], fg=COLORS["text"], font=FONTS["mono"], wraplength=330, justify=tk.LEFT).grid(
            row=row, column=1, sticky="ew", padx=(10, 0), pady=2
        )

    def _build_summary_panel(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=14, pady=14)
        panel.pack(fill=tk.X, pady=(0, 12))
        self._section_title(panel, "Trial Summary", "Final binary vanilla GCN: within_reference vs deviating")

        metrics = tk.Frame(panel, bg=COLORS["panel"])
        metrics.pack(fill=tk.X)
        metrics.columnconfigure((0, 1), weight=1)
        self._metric_card(metrics, "total nodes", self.total_nodes_var, COLORS["cyan"], 0, 0)
        self._metric_card(metrics, "deviating", self.deviating_count_var, COLORS["danger"], 0, 1)
        self._metric_card(metrics, "within ref", self.within_count_var, COLORS["success"], 1, 0)
        self._metric_card(metrics, "analysis", self.analysis_state_var, COLORS["warning"], 1, 1)

        details = tk.Frame(panel, bg=COLORS["panel"])
        details.pack(fill=tk.X, pady=(10, 0))
        details.columnconfigure(1, weight=1)
        self._info_row(details, "model", self.model_var, 0)
        self._info_row(details, "score", self.score_var, 1)
        self._info_row(details, "highest region", self.highest_region_var, 2)
        self._info_row(details, "replay mode", self.replay_mode_var, 3)
        self._info_row(details, "control available", self.control_available_var, 4)
        self._info_row(details, "preview", self.preview_ready_var, 5)
        self._info_row(details, "state", self.state_var, 6)

    def _build_top_nodes_panel(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=14, pady=14)
        panel.pack(fill=tk.BOTH, expand=True)
        self._section_title(panel, "Top Deviating Nodes", "Top 5 nodes ranked by final GCN/deviation output")
        tk.Label(
            panel,
            textvariable=self.top_nodes_var,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=FONTS["mono"],
            wraplength=300,
            justify=tk.LEFT,
        ).pack(anchor="nw", fill=tk.BOTH, expand=True)

    def _build_plot_panel(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=14, pady=14)
        panel.pack(fill=tk.BOTH, expand=True)
        self._section_title(panel, "Classifier Plots", "Bar plot and node-vs-progress deviation heatmap")
        self.bar_plot_canvas = tk.Canvas(panel, height=280, bg=COLORS["viewer"], highlightthickness=1, highlightbackground=COLORS["border"])
        self.bar_plot_canvas.pack(fill=tk.BOTH, expand=False, pady=(0, 12))
        self.heatmap_canvas = tk.Canvas(panel, height=450, bg=COLORS["viewer"], highlightthickness=1, highlightbackground=COLORS["border"])
        self.heatmap_canvas.pack(fill=tk.BOTH, expand=True)
        self.plot_canvas = self.bar_plot_canvas
        self.bar_plot_canvas.bind("<Configure>", self._schedule_plot_redraw)
        self.heatmap_canvas.bind("<Configure>", self._schedule_plot_redraw)

    def _metric_card(self, parent: tk.Widget, label: str, var: tk.StringVar, accent: str, row: int, col: int) -> None:
        box = tk.Frame(parent, bg=COLORS["panel_alt"], highlightbackground=COLORS["border"], highlightthickness=1, padx=12, pady=10)
        box.grid(row=row, column=col, sticky="ew", padx=(0, 8) if col == 0 else (8, 0))
        tk.Label(box, text=label.upper(), bg=COLORS["panel_alt"], fg=COLORS["muted"], font=FONTS["small"]).pack(anchor="w")
        tk.Label(box, textvariable=var, bg=COLORS["panel_alt"], fg=accent, font=FONTS["metric"]).pack(anchor="w")

    def _build_result_table(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=14, pady=14)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.rowconfigure(2, weight=1)
        panel.columnconfigure(0, weight=1)
        self._section_title(panel, "Node Result Table", "Red rows are predicted deviating; green rows are within reference")

        tk.Label(panel, textvariable=self.summary_var, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"], wraplength=350, justify=tk.LEFT).pack(
            anchor="w", fill=tk.X, pady=(0, 8)
        )
        table_wrap = tk.Frame(panel, bg=COLORS["panel"])
        table_wrap.pack(fill=tk.BOTH, expand=True)
        table_wrap.rowconfigure(0, weight=1)
        table_wrap.columnconfigure(0, weight=1)

        columns = ("node_name", "predicted_label", "prob_deviating", "confidence", "deviation_score")
        self.table = ttk.Treeview(table_wrap, columns=columns, show="headings", height=14)
        headings = {
            "node_name": "Muscle / Node",
            "predicted_label": "Label",
            "prob_deviating": "P(deviating)",
            "confidence": "Confidence",
            "deviation_score": "Deviation",
        }
        widths = {"node_name": 160, "predicted_label": 130, "prob_deviating": 95, "confidence": 90, "deviation_score": 90}
        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=widths[col], anchor="w")
        self.table.tag_configure("deviating", background=COLORS["danger_soft"], foreground="#FCA5A5")
        self.table.tag_configure("within_reference", background=COLORS["success_soft"], foreground="#BBF7D0")
        scroll = ttk.Scrollbar(table_wrap, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def _build_status_bar(self, parent: tk.Widget) -> None:
        bar = tk.Frame(parent, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1, padx=10, pady=7)
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)
        tk.Label(bar, text="STATUS", bg=COLORS["panel"], fg=COLORS["muted_dark"], font=FONTS["small"]).grid(row=0, column=0, sticky="w")
        tk.Label(bar, textvariable=self.status_var, bg=COLORS["panel"], fg=COLORS["text"], font=FONTS["body"], anchor="w").grid(
            row=0, column=1, sticky="ew", padx=(10, 10)
        )
        tk.Label(bar, textvariable=self.replay_status_var, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).grid(row=0, column=2, sticky="e")

    def _set_status(self, text: str, level: str = "normal") -> None:
        self.status_var.set(text)
        palette = {
            "normal": ("READY", COLORS["success_soft"], COLORS["success"]),
            "recording": ("RECORDING", COLORS["danger_soft"], COLORS["danger"]),
            "analyzing": ("ANALYZING", COLORS["warning_soft"], COLORS["warning"]),
            "done": ("DONE", COLORS["success_soft"], COLORS["success"]),
            "error": ("ERROR", COLORS["danger_soft"], COLORS["danger"]),
        }
        label, bg, fg = palette.get(level, palette["normal"])
        self.status_chip.configure(text=label, bg=bg, fg=fg)

    def _schedule_plot_redraw(self, _event: object | None = None) -> None:
        if not hasattr(self, "plot_controller"):
            return
        if self._plot_resize_after_id is not None:
            try:
                self.after_cancel(self._plot_resize_after_id)
            except Exception:
                pass
        self._plot_resize_after_id = self.after(120, self._redraw_plots_for_current_size)

    def _redraw_plots_for_current_size(self) -> None:
        self._plot_resize_after_id = None
        if not hasattr(self, "plot_controller"):
            return
        if self.current_analysis_result is not None:
            self.plot_controller.draw_result(self.current_analysis_result)
        else:
            self.plot_controller.draw_empty()

    def _set_state(self, state: GuiState) -> None:
        self.state = state
        self.state_var.set(state.value)
        if state == GuiState.ANALYZING:
            self.analysis_state_var.set("RUNNING")
        elif self.current_analysis_result is not None or (self.current_trial is not None and self._control_reference_available()):
            self.analysis_state_var.set("DONE")
        else:
            self.analysis_state_var.set("NOT_RUN")
        has_trial = self.current_trial is not None
        is_recording = state == GuiState.RECORDING
        is_analyzing = state == GuiState.ANALYZING
        can_run_analysis = has_trial and state in (
            GuiState.TRIAL_READY,
            GuiState.ANALYSIS_DONE,
            GuiState.KINEMATIC_REPLAY,
            GuiState.CONTROL_REPLAY,
        )
        can_replay = has_trial and not is_recording
        control_available = has_trial and self._control_reference_available()

        if hasattr(self, "start_button"):
            self.start_button.configure(state=tk.DISABLED if is_recording or is_analyzing else tk.NORMAL)
            self.stop_button.configure(state=tk.NORMAL if is_recording else tk.DISABLED)
            self.load_button.configure(state=tk.DISABLED if is_recording or is_analyzing else tk.NORMAL)
            self.run_button.configure(state=tk.NORMAL if can_run_analysis and not is_analyzing else tk.DISABLED)
            self.record_next_button.configure(state=tk.DISABLED if is_recording or is_analyzing else tk.NORMAL)
            self.load_session_button.configure(state=tk.DISABLED if is_recording or is_analyzing or not self._session_trials else tk.NORMAL)
            self.open_replay_button.configure(state=tk.NORMAL)
            self.replay_kinematic_button.configure(state=tk.NORMAL if can_replay and not is_analyzing else tk.DISABLED)
            self.replay_control_button.configure(state=tk.NORMAL if can_replay and control_available and not is_analyzing else tk.DISABLED)
            self.close_replay_button.configure(state=tk.NORMAL)
            self.play_button.configure(state=tk.NORMAL if can_replay and not is_analyzing else tk.DISABLED)
            self.pause_button.configure(state=tk.NORMAL if can_replay and not is_analyzing else tk.DISABLED)
            self.reset_button.configure(state=tk.NORMAL if can_replay and not is_analyzing else tk.DISABLED)

    def open_leap_viewer(self) -> None:
        if self.leap_viewer.restart():
            self.leap_viewer_status_var.set(self.leap_viewer.status().message)
            self._set_status("External Leap viewer running.", "done")
        else:
            self.leap_viewer_status_var.set(self.leap_viewer.status().message)
            self._set_status(self.leap_viewer.status().message, "error")

    def close_leap_viewer(self) -> None:
        self.leap_viewer.close()
        self.leap_viewer_status_var.set(self.leap_viewer.status().message)

    def reset_qpos_provider(self) -> None:
        if self.state == GuiState.RECORDING:
            self._set_status("Stop recording before resetting the qpos provider.", "error")
            return
        if self.leap_provider is not None:
            self.leap_provider.close()
        provider = LiveLeapQposProvider(show_leap_view=False)
        if provider.start():
            self.leap_provider = provider
            self.recorder.set_qpos_provider(provider)
            self.qpos_status_var.set(provider.status().message)
            self._set_status("Leap qpos provider reset.", "done")
        else:
            self.leap_provider = None
            self.recorder.set_qpos_provider(None)
            self.qpos_status_var.set(provider.message)
            self._set_status(f"Qpos provider reset failed: {provider.message}", "error")

    def _control_reference_available(self) -> bool:
        if self.current_trial is None:
            return False
        try:
            return bool(self.inference.control_reference_available(self.current_trial))
        except Exception:
            return False

    def _refresh_control_available_status(self) -> None:
        self.control_available_var.set("YES" if self._control_reference_available() else "NO")

    def open_replay_viewer(self) -> None:
        if self.replay.open_or_focus_viewer():
            self._set_status("MuJoCo viewer opened. Live preview remains active until a replay mode is selected.", "done")
        else:
            self._set_status(self.replay.status().message, "error")
        self._refresh_replay_status()

    def replay_recorded_kinematics(self) -> None:
        if self.current_trial is None:
            messagebox.showwarning("No Trial", "Record or load a trial before kinematic replay.")
            self._set_status("No current trial for kinematic replay.", "error")
            return
        if self.replay.replay_kinematics(self.current_trial):
            self._set_state(GuiState.KINEMATIC_REPLAY)
            self._set_status("KINEMATIC_REPLAY: recorded qpos loaded into external MuJoCo viewer.", "done")
        else:
            self._set_status(self.replay.status().message, "error")
        self._refresh_replay_status()

    def replay_estimated_control(self) -> None:
        if self.current_trial is None:
            messagebox.showwarning("No Trial", "Record or load a trial before control replay.")
            self._set_status("No current trial for control replay.", "error")
            return
        try:
            control_ref = self.inference.load_control_reference(self.current_trial)
        except Exception as exc:
            messagebox.showwarning("Control Replay Requires Analysis", "Control replay requires analysis output. Please run analysis first.")
            self._set_status(str(exc), "error")
            self._refresh_control_available_status()
            return
        if self.replay.replay_control(
            trial=self.current_trial,
            ctrl_sequence=control_ref["ctrl_sequence"],
            dt=float(control_ref["dt"]),
            source_path=str(control_ref["source_path"]),
        ):
            self._set_state(GuiState.CONTROL_REPLAY)
            quality = dict(control_ref.get("quality") or {})
            rmse = quality.get("rmse_forward_error")
            joint_error = quality.get("mean_joint_error")
            if rmse is not None:
                self._set_status(
                    f"CONTROL_REPLAY: ctrl_ref forward simulation active | rmse={float(rmse):.3f} mean_joint={float(joint_error or 0.0):.3f}",
                    "done",
                )
            else:
                self._set_status("CONTROL_REPLAY: ctrl_ref is driving forward simulation.", "done")
        else:
            self._set_status(self.replay.status().message, "error")
        self._refresh_replay_status()
        self._refresh_control_available_status()

    def close_replay_viewer(self) -> None:
        self.replay.close()
        self._refresh_replay_status()
        if self.current_trial is not None and self.state in (GuiState.KINEMATIC_REPLAY, GuiState.CONTROL_REPLAY):
            self._set_state(GuiState.ANALYSIS_DONE if self._control_reference_available() else GuiState.TRIAL_READY)
        self._set_status("MuJoCo replay viewer closed.", "done")

    def _initialize_live_viewers(self) -> None:
        messages = []
        messages.append("GUI idle. Leap viewer and MuJoCo live preview are monitoring-only; trial replay remains manual.")
        if self.show_external_leap_skeleton:
            if self.leap_viewer.start():
                messages.append("Leap viewer ready")
            else:
                messages.append(f"Leap viewer unavailable: {self.leap_viewer.status().message}")
        self.leap_viewer_status_var.set(self.leap_viewer.status().message)
        if self.leap_provider is None:
            provider = LiveLeapQposProvider(show_leap_view=False)
            if provider.start():
                self.leap_provider = provider
                self.recorder.set_qpos_provider(provider)
                messages.append("Leap qpos provider ready")
                self.qpos_status_var.set(provider.status().message)
            else:
                messages.append(f"Leap qpos provider unavailable: {provider.message}")
                self.qpos_status_var.set(provider.message)
        if self.replay.open_or_focus_viewer():
            messages.append("MuJoCo live preview ready")
        else:
            messages.append(f"MuJoCo preview unavailable: {self.replay.status().message}")
        self.preview_ready_var.set(" | ".join(messages))
        self._set_status(" | ".join(messages), "normal")
        self._refresh_replay_status()
        self._set_state(GuiState.IDLE)
        self.after(1200, lambda: self._retry_mujoco_live_preview(1))

    def _retry_mujoco_live_preview(self, attempt: int) -> None:
        if self.replay.replay_mode != ReplayMode.NONE:
            return
        if self.replay.open_or_focus_viewer():
            self.preview_ready_var.set("MuJoCo live preview active | trial replay is manual")
            self._refresh_replay_status()
            return
        if int(attempt) < 3:
            self.after(1200, lambda: self._retry_mujoco_live_preview(int(attempt) + 1))

    def _live_preview_allowed(self) -> bool:
        return self.replay.replay_mode == ReplayMode.NONE and not bool(self.replay.playing)

    def _render_live_preview(self, qpos: Optional[object], *, message: str) -> None:
        if qpos is None or not self._live_preview_allowed():
            return
        self.replay.render_qpos(qpos, message=message)

    def _refresh_qpos_provider_status(self) -> None:
        if self.leap_provider is None:
            self.qpos_status_var.set("QPOS provider: unavailable, synthetic fallback used while recording.")
            return
        status = self.leap_provider.status()
        if status.stuck:
            self.qpos_status_var.set(f"{status.message} External Leap viewer can still be live. Use Reset QPOS Provider.")
        else:
            self.qpos_status_var.set(
                f"{status.message} samples={status.samples_seen} age={status.latest_age_seconds:.2f}s unchanged={status.unchanged_seconds:.2f}s"
            )

    def start_recording(self) -> None:
        if self.state == GuiState.RECORDING:
            return
        self._cancel_auto_stop_timer()
        provider_message = ""
        if self.leap_provider is None:
            provider = LiveLeapQposProvider(show_leap_view=False)
            if provider.start():
                self.leap_provider = provider
                self.recorder.set_qpos_provider(provider)
                provider_message = "Leap qpos provider active."
            else:
                provider_message = f"{provider.message} Synthetic fallback active."
        self.recorder.start()
        self._record_target_seconds = self._parse_record_duration_seconds()
        if bool(self.auto_stop_enabled_var.get()) and self._record_target_seconds > 0.0:
            self._auto_stop_after_id = self.after(int(round(self._record_target_seconds * 1000.0)), self._auto_stop_recording)
            provider_message = f"{provider_message} Auto-stop={self._record_target_seconds:.1f}s."
        self._set_state(GuiState.RECORDING)
        self._set_status(f"Recording qpos frames. {provider_message}", "recording")

    def stop_recording(self, *, auto_triggered: bool = False) -> None:
        self._cancel_auto_stop_timer()
        try:
            trial = self.recorder.stop_and_save(subject_id=self.subject_var.get(), motion_type=self.motion_var.get())
        except Exception as exc:
            self._set_state(GuiState.IDLE)
            self._set_status("Stop recording failed.", "error")
            messagebox.showerror("Stop Recording Failed", str(exc))
            return
        self._set_trial(trial)
        self._register_session_trial(trial)
        self._set_state(GuiState.TRIAL_READY)
        prefix = "Auto-saved recording" if auto_triggered else "Saved recording"
        self._set_status(f"{prefix}: {trial.qpos_csv}", "done")
        if bool(self.auto_analyze_after_record_var.get()):
            self.after(250, self.run_analysis)

    def _auto_stop_recording(self) -> None:
        self._auto_stop_after_id = None
        if self.state == GuiState.RECORDING:
            self.stop_recording(auto_triggered=True)

    def _cancel_auto_stop_timer(self) -> None:
        if self._auto_stop_after_id is None:
            return
        try:
            self.after_cancel(self._auto_stop_after_id)
        except Exception:
            pass
        self._auto_stop_after_id = None

    def _parse_record_duration_seconds(self) -> float:
        if not bool(self.auto_stop_enabled_var.get()):
            return 0.0
        try:
            value = float(str(self.record_duration_var.get()).strip())
        except Exception:
            self.record_duration_var.set("4.0")
            return 4.0
        if value <= 0.0:
            return 0.0
        value = max(0.5, min(60.0, value))
        self.record_duration_var.set(f"{value:.1f}")
        return float(value)

    def _register_session_trial(self, trial: TrialData) -> None:
        self._session_trials.append(trial)
        index = len(self._session_trials)
        self.session_count_var.set(str(index))
        self.last_saved_var.set(str(trial.qpos_csv))
        label = f"{index:02d} | {trial.trial_id} | {trial.num_frames} frames"
        self.session_trial_list.insert(tk.END, label)
        self.session_trial_list.selection_clear(0, tk.END)
        self.session_trial_list.selection_set(tk.END)
        self.session_trial_list.see(tk.END)
        self._set_state(self.state)

    def load_selected_session_trial(self) -> None:
        if not self._session_trials:
            self._set_status("No session trials have been recorded yet.", "error")
            return
        selection = self.session_trial_list.curselection()
        index = int(selection[0]) if selection else len(self._session_trials) - 1
        try:
            trial = load_trial(self._session_trials[index].qpos_csv)
        except Exception as exc:
            self._set_status("Failed to load selected session trial.", "error")
            messagebox.showerror("Load Session Trial Failed", str(exc))
            return
        self._set_trial(trial)
        self._set_state(GuiState.TRIAL_READY)
        self._set_status(f"Loaded session trial: {trial.qpos_csv}", "done")

    def load_existing_trial(self) -> None:
        selected = filedialog.askopenfilename(
            title="Load qpos.csv",
            filetypes=(("QPOS CSV", "qpos.csv"), ("CSV files", "*.csv"), ("All files", "*.*")),
            initialdir=str(Path("records").resolve()),
        )
        if not selected:
            return
        try:
            trial = load_trial(selected)
        except Exception as exc:
            self._set_status("Load failed.", "error")
            messagebox.showerror("Load Failed", str(exc))
            return
        self._set_trial(trial)
        self._set_state(GuiState.TRIAL_READY)
        self._set_status(f"Loaded trial: {trial.qpos_csv}", "done")

    def _set_trial(self, trial: TrialData) -> None:
        self.current_trial = trial
        self.current_analysis_result = None
        self.path_var.set(str(trial.qpos_csv))
        self.trial_id_var.set(trial.trial_id)
        self.frame_count_var.set(str(trial.num_frames))
        self.summary_var.set("No classification result for this trial yet.")
        self.total_nodes_var.set("0")
        self.within_count_var.set("0")
        self.deviating_count_var.set("0")
        self.top_nodes_var.set("-")
        self.highest_region_var.set("-")
        self.score_var.set("-")
        if hasattr(self, "plot_controller"):
            self.plot_controller.draw_empty("Run analysis to show node deviation plot.")
        self._clear_table()
        self.replay.set_trial(trial)
        self.progress_scale.configure(from_=0, to=max(0, trial.num_frames - 1))
        self._refresh_control_available_status()
        self._refresh_replay_status()

    def play_replay(self) -> None:
        self.replay.play()
        self._refresh_replay_status()
        status = self.replay.status()
        if status.replay_mode == ReplayMode.CONTROL.value:
            self._set_state(GuiState.CONTROL_REPLAY)
        elif status.replay_mode == ReplayMode.KINEMATIC.value:
            self._set_state(GuiState.KINEMATIC_REPLAY)

    def pause_replay(self) -> None:
        self.replay.pause()
        self._refresh_replay_status()

    def reset_replay(self) -> None:
        self.replay.reset()
        self._refresh_replay_status()

    def _on_slider(self, value: str) -> None:
        if self._slider_updating or self.current_trial is None:
            return
        try:
            frame_index = int(float(value))
        except Exception:
            return
        self.replay.set_frame(frame_index)
        self._refresh_replay_status()

    def run_analysis(self) -> None:
        if self.current_trial is None:
            messagebox.showwarning("No Trial", "Load or record a trial first.")
            return
        if self.analysis_running:
            return
        self.analysis_running = True
        self._set_state(GuiState.ANALYZING)
        self._set_status("Running offline ctrl_ref -> graph sequence -> binary GCN inference. This can take time.", "analyzing")
        trial = self.current_trial
        worker = threading.Thread(target=self._analysis_worker, args=(trial,), daemon=True)
        worker.start()

    def _analysis_worker(self, trial: TrialData) -> None:
        try:
            result = self.inference.run_trial_analysis(trial)
            self.analysis_queue.put(("result", result))
        except Exception as exc:
            self.analysis_queue.put(("error", exc))

    def _handle_analysis_queue(self) -> None:
        try:
            kind, payload = self.analysis_queue.get_nowait()
        except queue.Empty:
            return
        self.analysis_running = False
        if kind == "error":
            self._set_state(GuiState.TRIAL_READY if self.current_trial is not None else GuiState.IDLE)
            self._set_status("Analysis failed.", "error")
            messagebox.showerror("Analysis Failed", str(payload))
            return
        result = payload
        if isinstance(result, dict):
            self._show_result(result)

    def _show_result(self, result: dict) -> None:
        self.current_analysis_result = result
        self._clear_table()
        summary = dict(result.get("summary") or {})
        top_nodes = list(summary.get("top_deviating_nodes") or [])
        num_deviating = int(summary.get("num_deviating", 0))
        num_nodes = int(summary.get("num_nodes", 0))
        num_within = int(summary.get("num_within_reference", max(0, num_nodes - num_deviating)))
        highest_region = dict(summary.get("highest_deviation_region") or {})
        if highest_region:
            region_text = (
                f"{float(highest_region.get('start_progress', 0.0)):.1f}% -> "
                f"{float(highest_region.get('end_progress', 0.0)):.1f}% "
                f"(score={float(highest_region.get('score', 0.0)):.3f})"
            )
        else:
            region_text = "-"
        self.total_nodes_var.set(str(num_nodes))
        self.deviating_count_var.set(f"{num_deviating}/{num_nodes}")
        self.within_count_var.set(f"{num_within}/{num_nodes}")
        self.highest_region_var.set(region_text)
        self.top_nodes_var.set("\n".join(f"{idx}. {name}" for idx, name in enumerate(top_nodes[:5], start=1)) or "-")
        self.score_var.set(
            f"node={float(summary.get('overall_node_deviation_score', 0.0)):.3f} | edge={float(summary.get('overall_edge_deviation_score', 0.0)):.3f}"
        )
        self.summary_var.set(
            f"Trial {result.get('trial_id')} | deviating nodes: {num_deviating}/{num_nodes} | top: {', '.join(top_nodes[:5]) or '-'}"
        )
        for row in list(result.get("nodes") or []):
            label = str(row.get("predicted_label") or "")
            self.table.insert(
                "",
                tk.END,
                values=(
                    row.get("node_name", ""),
                    label,
                    f"{float(row.get('probability_deviating', 0.0)):.3f}",
                    f"{float(row.get('confidence', 0.0)):.3f}",
                    f"{float(row.get('deviation_score', 0.0)):.3f}",
                ),
                tags=("deviating" if label == "deviating" else "within_reference",),
            )
        if hasattr(self, "plot_controller"):
            self.plot_controller.draw_result(result)
        self._refresh_control_available_status()
        self._set_state(GuiState.ANALYSIS_DONE)
        self._set_status("Analysis complete. Result saved beside qpos.csv.", "done")

    def _clear_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

    def _refresh_replay_status(self) -> None:
        status = self.replay.status()
        current = status.frame_index + 1 if status.num_frames else 0
        self.frame_info_var.set(f"Frame {current} / {status.num_frames}")
        self.replay_mode_var.set(str(status.replay_mode))
        self.replay_status_var.set(
            f"{status.message} | mode={status.replay_mode} viewer={int(status.viewer_open)} playing={int(status.playing)}"
        )
        self.progress_scale.configure(from_=0, to=max(0, int(status.num_frames) - 1))
        if not self._slider_updating:
            self._slider_updating = True
            try:
                self.progress_scale.set(status.frame_index)
            finally:
                self._slider_updating = False

    def _tick(self) -> None:
        rec_status = self.recorder.status()
        if rec_status.active:
            self.recorder.append_current_frame()
            rec_status = self.recorder.status()
            self._refresh_qpos_provider_status()
            self._render_live_preview(self.recorder.latest_qpos(), message="LIVE_PREVIEW recording qpos + ctrl estimate")
            self.frame_count_var.set(str(rec_status.frame_count))
            target_text = f" / {self._record_target_seconds:.1f}s" if self._record_target_seconds > 0.0 else ""
            self._set_status(
                f"Recording... frames={rec_status.frame_count} elapsed={rec_status.elapsed_seconds:.1f}s{target_text} source={rec_status.source_mode}",
                "recording",
            )
        elif self.leap_provider is not None:
            latest_qpos = self.leap_provider.latest_qpos()
            if latest_qpos is not None:
                provider_status = self.leap_provider.status()
                if not provider_status.stuck:
                    self._render_live_preview(latest_qpos, message="LIVE_PREVIEW Leap qpos + ctrl estimate")
                    self.preview_ready_var.set("Leap qpos provider live | MuJoCo live preview active | trial replay is manual")
                else:
                    self.preview_ready_var.set("Leap qpos mapper is stuck. Reset QPOS Provider.")
            self._refresh_qpos_provider_status()
        if self.replay.playing:
            self.replay.step(1)
        self._refresh_replay_status()
        self.leap_viewer_status_var.set(self.leap_viewer.status().message)
        self._handle_analysis_queue()
        self.after(33, self._tick)

    def _on_close(self) -> None:
        self._cancel_auto_stop_timer()
        self.replay.close()
        self.leap_viewer.close()
        if self.leap_provider is not None:
            self.leap_provider.close()
        self.destroy()

    def _shortcut_start_recording(self, _event: object) -> str:
        if self.state != GuiState.RECORDING:
            self.start_recording()
        return "break"

    def _shortcut_stop_recording(self, _event: object) -> str:
        if self.state == GuiState.RECORDING:
            self.stop_recording()
        return "break"

    def _shortcut_load_existing_trial(self, _event: object) -> str:
        if self.state != GuiState.RECORDING:
            self.load_existing_trial()
        return "break"


def smoke_test() -> dict[str, str]:
    return {
        "gui_backend": "tkinter_dark_dashboard",
        "leap_viewer": "external_opencv_skeleton_viewer_from_local_leapc_python_bindings",
        "mujoco_viewer": "external_mujoco_kinematic_and_control_replay_on_request",
        "model_path": str(BinaryGCNInferenceController().config.model_path),
        "reference_dir": str(BinaryGCNInferenceController().config.reference_dir),
    }


def main(*, show_external_leap_skeleton: bool = True) -> None:
    app = TrialReplayClassifierApp(show_external_leap_skeleton=show_external_leap_skeleton)
    app.mainloop()
