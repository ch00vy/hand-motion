from __future__ import annotations

import json
import queue
import threading
import time
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from app.display_labels import (
    SIMULATION_INTERPRETATION_NOTE,
    TRIAL_LABEL_DISPLAY_VALUES,
    display_label,
    display_status,
    display_trial_label,
    internal_trial_label,
)
from app.gui.inference_controller import BinaryGCNInferenceController
from app.gui.leap_preview_controller import LeapExternalPreviewController
from app.gui.muscle_graph_controller import MuscleGraphController
from app.gui.mujoco_replay_controller import MujocoReplayController
from app.gui.plot_controller import NodeDeviationPlot
from app.gui.recorder_controller import LeapCQposProvider, LiveLeapQposProvider, TrialRecorder, UdpQposProvider
from app.gui.replay_mode import ReplayMode
from app.gui.state import GuiState
from app.gui.style import COLORS, FONTS, button, card, configure_button_variant, configure_dark_theme
from app.gui.trial_loader import TrialData, load_trial
from app.gui.window_layout import MAIN_GUI
from app.realtime.leapc_qpos_publisher import resolve_ultraleap_sdk_dir


class TrialReplayClassifierApp(tk.Tk):
    def __init__(self, *, show_external_leap_skeleton: bool = True):
        super().__init__()
        self.title("Muscle Activity Pattern Analysis - Dynamic Temporal GCN")
        self.geometry(f"{MAIN_GUI.width}x{MAIN_GUI.height}+{MAIN_GUI.x}+{MAIN_GUI.y}")
        self.minsize(980, 560)
        self.resizable(True, True)
        configure_dark_theme(self)
        self.show_external_leap_skeleton = bool(show_external_leap_skeleton)
        self.qpos_source_mode = str(os.environ.get("HAND_QPOS_SOURCE", "leapc")).strip().lower()
        compact_env = str(os.environ.get("HAND_GUI_COMPACT", "auto")).strip().lower()
        screen_height = int(self.winfo_screenheight() or MAIN_GUI.height)
        self.compact_layout = compact_env in {"1", "true", "yes", "on"} or (
            compact_env == "auto" and screen_height <= 950
        )
        if self.compact_layout:
            self.geometry(f"{MAIN_GUI.width}x860+{MAIN_GUI.x}+{MAIN_GUI.y}")

        self.recorder = TrialRecorder()
        self.leap_provider: Optional[object] = None
        self.leap_viewer = LeapExternalPreviewController(show_image=False)
        self.replay = MujocoReplayController()
        self.muscle_graph = MuscleGraphController()
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
        self.trial_label_var = tk.StringVar(value=display_trial_label("normal"))
        self.auto_stop_enabled_var = tk.BooleanVar(value=True)
        self.record_duration_var = tk.StringVar(value="4.0")
        self.auto_analyze_after_record_var = tk.BooleanVar(value=False)
        self.advanced_controls_visible_var = tk.BooleanVar(value=False)
        self.session_count_var = tk.StringVar(value="0")
        self.last_saved_var = tk.StringVar(value="-")
        self.path_var = tk.StringVar(value="-")
        self.trial_id_var = tk.StringVar(value="-")
        self.active_trial_label_var = tk.StringVar(value="-")
        self.frame_count_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Ready.")
        self.replay_status_var = tk.StringVar(value="Replay viewer closed.")
        self.summary_var = tk.StringVar(value="No activity-pattern result.")
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
        self.preview_ready_var = tk.StringVar(value="GUI idle. External viewers open on demand.")
        self.leap_viewer_status_var = tk.StringVar(value="Leap viewer: closed")
        self.qpos_status_var = tk.StringVar(value="QPOS provider: starting")
        self.glove_host_var = tk.StringVar(value=str(os.environ.get("HAND_QPOS_UDP_HOST", "127.0.0.1")))
        self.glove_port_var = tk.StringVar(value=str(os.environ.get("HAND_QPOS_UDP_PORT", "14043")))
        self.glove_connection_status_var = tk.StringVar(value="Disconnected")
        self.muscle_graph_status_var = tk.StringVar(value="Muscle graph: closed")
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
        outer_pad_x = 10 if self.compact_layout else 14
        outer_pad_y = 8 if self.compact_layout else 12
        shell = tk.Frame(self, bg=COLORS["bg"], padx=outer_pad_x, pady=outer_pad_y)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        self._build_header(shell)

        body = tk.Frame(shell, bg=COLORS["bg"])
        body.grid(row=1, column=0, sticky="nsew", pady=(6 if self.compact_layout else 10, 6 if self.compact_layout else 8))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        # ==========================================================
        # TOP CONTROL AREA: 3 equal-height cards
        # ==========================================================
        top_controls = tk.Frame(body, bg=COLORS["bg"], height=285 if self.compact_layout else 340)
        top_controls.grid(row=0, column=0, sticky="ew", pady=(0, 6 if self.compact_layout else 10))
        top_controls.grid_propagate(False)

        top_controls.rowconfigure(0, weight=1)
        top_controls.columnconfigure(0, weight=1, uniform="top_cards")
        top_controls.columnconfigure(1, weight=1, uniform="top_cards")
        top_controls.columnconfigure(2, weight=1, uniform="top_cards")

        trial_col = tk.Frame(top_controls, bg=COLORS["bg"])
        session_col = tk.Frame(top_controls, bg=COLORS["bg"])
        viewer_col = tk.Frame(top_controls, bg=COLORS["bg"])

        trial_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        session_col.grid(row=0, column=1, sticky="nsew", padx=(4, 4))
        viewer_col.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        trial_col.columnconfigure(0, weight=1)
        session_col.columnconfigure(0, weight=1)
        viewer_col.columnconfigure(0, weight=1)

        self._build_trial_control(trial_col)
        self._build_recording_session_panel(session_col)
        self._build_external_viewer_panel(viewer_col)

        # ==========================================================
        # MAIN ANALYSIS AREA: plot left, table right.
        # ==========================================================
        main = tk.Frame(body, bg=COLORS["bg"])
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=6, minsize=620)
        main.columnconfigure(1, weight=5, minsize=500)
        main.rowconfigure(0, weight=1)

        center = tk.Frame(main, bg=COLORS["bg"])
        center.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)

        right = tk.Frame(main, bg=COLORS["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_plot_panel(center)
        self._build_summary_panel(right)
        self._build_result_table(right)
        self._build_status_bar(shell)

    def _make_scrollable_column(self, parent: tk.Widget, *, min_width: int) -> tuple[tk.Frame, tk.Frame]:
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, bg=COLORS["bg"], highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg=COLORS["bg"])
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        def _update_scroll_region(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_inner_width(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=max(int(min_width), int(event.width)))

        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        inner.bind("<Configure>", _update_scroll_region)
        canvas.bind("<Configure>", _sync_inner_width)
        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        return outer, inner

    def _build_header(self, parent: tk.Widget) -> None:
        header = tk.Frame(parent, bg=COLORS["bg"])
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Hand Weakness Pattern Demo",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="Leap qpos replay, realtime control indicators, and Dynamic Temporal GCN weakness indication",
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
            tk.Label(parent, text=subtitle, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).pack(
                anchor="w", pady=(1 if self.compact_layout else 2, 5 if self.compact_layout else 8)
            )
        else:
            tk.Frame(parent, bg=COLORS["panel"], height=5 if self.compact_layout else 8).pack(fill=tk.X)

    def _build_trial_control(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=10 if self.compact_layout else 12, pady=8 if self.compact_layout else 10)
        panel.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        panel.columnconfigure(1, weight=1)
        self._section_title(
            panel,
            "Demo Workflow",
            "Record one hand transition, preview motion, then run weakness-pattern analysis",
        )

        form = tk.Frame(panel, bg=COLORS["panel"])
        form.pack(fill=tk.X, pady=(0, 6))
        form.columnconfigure(1, weight=1)
        self._form_label(form, "Subject", 0)
        ttk.Entry(form, textvariable=self.subject_var).grid(row=0, column=1, sticky="ew", pady=2)
        self._form_label(form, "Motion", 1)
        motion = ttk.Combobox(form, textvariable=self.motion_var, values=("open_to_fist", "fist_to_open"), state="readonly")
        motion.grid(row=1, column=1, sticky="ew", pady=2)
        self._form_label(form, "Reference Label", 2)
        trial_label = ttk.Combobox(form, textvariable=self.trial_label_var, values=TRIAL_LABEL_DISPLAY_VALUES, state="readonly")
        trial_label.grid(row=2, column=1, sticky="ew", pady=2)

        actions = tk.Frame(panel, bg=COLORS["panel"])
        actions.pack(fill=tk.X, pady=(4 if self.compact_layout else 6, 4 if self.compact_layout else 6))
        actions.columnconfigure((0, 1), weight=1)
        self.start_button = button(actions, "Record", self.start_recording, variant="primary")
        self.stop_button = button(actions, "Stop", self.stop_recording, variant="danger")
        self.load_button = button(actions, "Load Trial", self.load_existing_trial, variant="secondary")
        self.run_button = button(actions, "Analyze", self.run_analysis, variant="secondary")
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=2)
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=2)
        self.load_button.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=2)
        self.run_button.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=2)
        self.stop_button.configure(state=tk.DISABLED)

        info = tk.Frame(panel, bg=COLORS["panel"])
        info.pack(fill=tk.X, pady=(2, 0))
        info.columnconfigure(1, weight=1)
        self._info_row(info, "trial_id", self.trial_id_var, 0)
        self._info_row(info, "category", self.active_trial_label_var, 1)
        self._info_row(info, "frames", self.frame_count_var, 2)

    def _build_recording_session_panel(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=10 if self.compact_layout else 12, pady=8 if self.compact_layout else 10)
        panel.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        panel.columnconfigure(0, weight=1)
        self._section_title(
            panel,
            "Recording Session",
            "Batch recording for repeated normal/abnormal trials",
        )

        controls = tk.Frame(panel, bg=COLORS["panel"])
        controls.pack(fill=tk.X, pady=(0, 4))
        controls.columnconfigure(1, weight=0)
        controls.columnconfigure(3, weight=1)
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
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(controls, textvariable=self.record_duration_var, width=7).grid(row=0, column=1, sticky="w", padx=(6, 4), pady=2)
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
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 4))

        actions = tk.Frame(panel, bg=COLORS["panel"])
        actions.pack(fill=tk.X, pady=(1 if self.compact_layout else 2, 4 if self.compact_layout else 6))
        actions.columnconfigure((0, 1), weight=1)
        self.record_next_button = button(actions, "Record Next", self.start_recording, variant="primary")
        self.load_session_button = button(actions, "Load Selected", self.load_selected_session_trial, variant="secondary")
        self.record_next_button.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=2)
        self.load_session_button.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=2)

        status = tk.Frame(panel, bg=COLORS["panel"])
        status.pack(fill=tk.X, pady=(0, 4))
        status.columnconfigure(1, weight=1)
        self._info_row(status, "saved", self.session_count_var, 0)
        self._info_row(status, "last saved", self.last_saved_var, 1)

        list_wrap = tk.Frame(panel, bg=COLORS["panel"])
        list_wrap.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        list_wrap.columnconfigure(0, weight=1)
        list_wrap.rowconfigure(0, weight=1)
        self.session_trial_list = tk.Listbox(
            list_wrap,
            height=3,
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
        self.session_trial_list.grid(row=0, column=0, sticky="nsew")
        session_scroll.grid(row=0, column=1, sticky="ns")
        self.session_trial_list.bind("<Double-Button-1>", lambda _event: self.load_selected_session_trial())

    def _build_external_viewer_panel(self, parent: tk.Widget) -> None:
        panel_shell = card(parent, padx=10 if self.compact_layout else 12, pady=8 if self.compact_layout else 10)
        panel_shell.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        panel_shell.rowconfigure(0, weight=1)
        panel_shell.columnconfigure(0, weight=1)

        preview_canvas = tk.Canvas(panel_shell, bg=COLORS["panel"], highlightthickness=0, bd=0)
        preview_scrollbar = ttk.Scrollbar(panel_shell, orient=tk.VERTICAL, command=preview_canvas.yview)
        panel = tk.Frame(preview_canvas, bg=COLORS["panel"])
        preview_window = preview_canvas.create_window((0, 0), window=panel, anchor="nw")
        preview_canvas.configure(yscrollcommand=preview_scrollbar.set)
        preview_canvas.grid(row=0, column=0, sticky="nsew")
        preview_scrollbar.grid(row=0, column=1, sticky="ns")

        def _update_preview_scroll_region(_event: tk.Event) -> None:
            preview_canvas.configure(scrollregion=preview_canvas.bbox("all"))

        def _sync_preview_width(event: tk.Event) -> None:
            preview_canvas.itemconfigure(preview_window, width=max(1, int(event.width)))

        def _scroll_preview(event: tk.Event) -> None:
            preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        panel.bind("<Configure>", _update_preview_scroll_region)
        preview_canvas.bind("<Configure>", _sync_preview_width)
        preview_canvas.bind("<Enter>", lambda _event: preview_canvas.bind_all("<MouseWheel>", _scroll_preview))
        preview_canvas.bind("<Leave>", lambda _event: preview_canvas.unbind_all("<MouseWheel>"))

        self._section_title(
            panel,
            "Live Preview",
            "Open the hand model and muscle activity graph for the demo",
        )

        connection = tk.LabelFrame(
            panel,
            text=" Smart glove connection ",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            bd=1,
            relief=tk.GROOVE,
            padx=8,
            pady=5,
            font=FONTS["small"],
        )
        connection.pack(fill=tk.X, pady=(0, 6))
        connection.columnconfigure(1, weight=1)
        self._form_label(connection, "Host", 0)
        ttk.Entry(connection, textvariable=self.glove_host_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(6, 0), pady=2)
        self._form_label(connection, "Port", 1)
        ttk.Entry(connection, textvariable=self.glove_port_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(6, 0), pady=2)
        connection_actions = tk.Frame(connection, bg=COLORS["panel"])
        connection_actions.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(5, 2))
        connection_actions.columnconfigure((0, 1), weight=1)
        self.glove_connect_button = button(connection_actions, "Connect", self.connect_smart_glove, variant="primary")
        self.glove_disconnect_button = button(connection_actions, "Disconnect", self.disconnect_smart_glove, variant="secondary")
        self.glove_connect_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.glove_disconnect_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        tk.Label(
            connection,
            text="Status",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONTS["small"],
        ).grid(row=3, column=0, sticky="w", pady=(2, 0))
        tk.Label(
            connection,
            textvariable=self.glove_connection_status_var,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=FONTS["small"],
            anchor="w",
        ).grid(row=3, column=1, columnspan=2, sticky="ew", padx=(6, 0), pady=(2, 0))

        status_row = tk.Frame(panel, bg=COLORS["panel"])
        status_row.pack(fill=tk.X, pady=(0, 4))
        status_row.columnconfigure(0, weight=1)

        tk.Label(
            status_row,
            textvariable=self.leap_viewer_status_var,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONTS["small"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        grid = tk.Frame(panel, bg=COLORS["panel"])
        grid.pack(fill=tk.X, pady=(0, 4 if self.compact_layout else 6))
        grid.columnconfigure((0, 1, 2), weight=1)

        self.open_leap_button = None
        self.open_replay_button = None
        self.close_leap_button = button(grid, "Leap Preview", self.open_leap_viewer, variant="secondary")
        self.reset_qpos_button = button(grid, "Reset QPOS", self.reset_qpos_provider, variant="secondary")
        self.open_replay_button = button(grid, "Open Hand Model", self.open_replay_viewer, variant="secondary")
        self.close_replay_button = button(grid, "Close Model", self.close_replay_viewer, variant="secondary")
        self.replay_kinematic_button = button(grid, "Load QPOS Replay", self.replay_recorded_kinematics, variant="secondary")
        self.replay_control_button = button(grid, "Load CTRL Replay", self.replay_estimated_control, variant="secondary")
        self.play_button = button(grid, "Play", self.play_replay, variant="secondary")
        self.pause_button = button(grid, "Pause", self.pause_replay, variant="secondary")
        self.reset_button = button(grid, "Reset Replay", self.reset_replay, variant="secondary")
        self.open_muscle_graph_button = button(grid, "Muscle Graph", self.open_muscle_graph, variant="secondary")
        self.close_muscle_graph_button = button(grid, "Close Graph", self.close_muscle_graph, variant="secondary")
        self.generate_current_embedding_button = button(
            grid,
            "Generate Embedding",
            self.generate_current_trial_embedding_plot,
            variant="secondary",
        )
        self.open_current_embedding_plot_button = button(
            grid,
            "Open Embedding",
            self.open_current_trial_embedding_plot,
            variant="secondary",
        )
        self.open_relation_scatter_plot_button = button(
            grid,
            "Relation Scatter",
            self.open_relation_scatter_plot,
            variant="secondary",
        )
        self.open_top_relation_table_button = button(
            grid,
            "Relation Table",
            self.open_top_relation_table,
            variant="secondary",
        )
        self.open_top_relation_bar_button = button(
            grid,
            "Relation Bar",
            self.open_top_relation_bar_chart,
            variant="secondary",
        )
        viewer_buttons = (
            self.close_leap_button,
            self.reset_qpos_button,
            self.open_replay_button,
            self.close_replay_button,
            self.replay_kinematic_button,
            self.replay_control_button,
            self.play_button,
            self.pause_button,
            self.reset_button,
            self.open_muscle_graph_button,
            self.close_muscle_graph_button,
            self.generate_current_embedding_button,
            self.open_current_embedding_plot_button,
            self.open_relation_scatter_plot_button,
            self.open_top_relation_table_button,
            self.open_top_relation_bar_button,
        )
        for viewer_button in viewer_buttons:
            viewer_button.configure(pady=3 if self.compact_layout else 4, padx=8)
        if self.compact_layout:
            self.open_top_relation_table_button.configure(padx=6)
            self.open_top_relation_bar_button.configure(padx=6)

        self.close_leap_button.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.reset_qpos_button.grid(row=0, column=1, sticky="ew", padx=(4, 4), pady=2)
        self.open_replay_button.grid(row=0, column=2, sticky="ew", padx=(4, 0), pady=2)
        self.open_muscle_graph_button.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.replay_kinematic_button.grid(row=1, column=1, sticky="ew", padx=(4, 4), pady=2)
        self.replay_control_button.grid(row=1, column=2, sticky="ew", padx=(4, 0), pady=2)
        self.play_button.grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.pause_button.grid(row=2, column=1, sticky="ew", padx=(4, 4), pady=2)
        self.reset_button.grid(row=2, column=2, sticky="ew", padx=(4, 0), pady=2)
        slider_row = tk.Frame(grid, bg=COLORS["panel"])
        slider_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 2))
        slider_row.columnconfigure(0, weight=1)

        self.progress_scale = ttk.Scale(slider_row, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_slider)
        self.progress_scale.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        tk.Label(
            slider_row,
            textvariable=self.frame_info_var,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONTS["mono"],
            width=14,
            anchor="e",
        ).grid(row=0, column=1, sticky="e")
        if self.compact_layout:
            self.open_top_relation_table_button.grid(row=4, column=0, columnspan=2, sticky="ew", padx=(0, 4), pady=(5, 2))
            self.open_top_relation_bar_button.grid(row=4, column=2, sticky="ew", padx=(4, 0), pady=(5, 2))
        else:
            tk.Label(
                grid,
                text="Advanced replay/export tools",
                bg=COLORS["panel"],
                fg=COLORS["muted_dark"],
                font=FONTS["small"],
                anchor="w",
            ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 2))
            self.generate_current_embedding_button.grid(row=5, column=0, sticky="ew", padx=(0, 4), pady=2)
            self.open_current_embedding_plot_button.grid(row=5, column=1, sticky="ew", padx=(4, 4), pady=2)
            self.open_relation_scatter_plot_button.grid(row=5, column=2, sticky="ew", padx=(4, 0), pady=2)
            self.open_top_relation_table_button.grid(row=6, column=0, columnspan=3, sticky="ew", padx=0, pady=2)
            self.open_top_relation_bar_button.grid(row=7, column=0, columnspan=3, sticky="ew", padx=0, pady=2)

    def _form_label(self, parent: tk.Widget, text: str, row: int) -> None:
        tk.Label(parent, text=text, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"]).grid(row=row, column=0, sticky="w", padx=(0, 10))

    def _info_row(self, parent: tk.Widget, label: str, var: tk.StringVar, row: int) -> None:
        tk.Label(parent, text=label, bg=COLORS["panel"], fg=COLORS["muted_dark"], font=FONTS["small"]).grid(row=row, column=0, sticky="nw", pady=2)
        tk.Label(parent, textvariable=var, bg=COLORS["panel"], fg=COLORS["text"], font=FONTS["mono"], wraplength=330, justify=tk.LEFT).grid(
            row=row, column=1, sticky="ew", padx=(10, 0), pady=2
        )

    def _build_summary_panel(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=10 if self.compact_layout else 14, pady=8 if self.compact_layout else 14)
        panel.pack(fill=tk.X, pady=(0, 6 if self.compact_layout else 12))
        self._section_title(panel, "Weakness Indication Summary", "Ringkasan hasil trial dibandingkan dengan acuan normal")

        metrics = tk.Frame(panel, bg=COLORS["panel"])
        metrics.pack(fill=tk.X)
        metrics.columnconfigure((0, 1), weight=1)
        self._metric_card(metrics, "otot dianalisis", self.total_nodes_var, COLORS["cyan"], 0, 0)
        self._metric_card(metrics, "indikasi berbeda", self.deviating_count_var, COLORS["danger"], 0, 1)
        self._metric_card(metrics, "sesuai acuan", self.within_count_var, COLORS["success"], 1, 0)
        self._metric_card(metrics, "analysis", self.analysis_state_var, COLORS["warning"], 1, 1)

        details = tk.Frame(panel, bg=COLORS["panel"])
        details.pack(fill=tk.X, pady=(6 if self.compact_layout else 10, 0))
        details.columnconfigure(1, weight=1)
        self._info_row(details, "model", self.model_var, 0)
        self._info_row(details, "skor pola", self.score_var, 1)
        self._info_row(details, "fase tertinggi", self.highest_region_var, 2)
        self._info_row(details, "state", self.state_var, 3)

    def _build_top_nodes_panel(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=14, pady=14)
        panel.pack(fill=tk.BOTH, expand=True)
        self._section_title(panel, "Top Muscle Activity Pattern Differences", "Top 5 otot berdasarkan skor perbedaan pola")
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
        panel = card(parent, padx=10 if self.compact_layout else 14, pady=8 if self.compact_layout else 14)
        panel.pack(fill=tk.BOTH, expand=True)
        self._section_title(panel, "Activity Pattern Plot", "Grafik otot dan heatmap progres gerakan")
        self.bar_plot_canvas = tk.Canvas(
            panel,
            height=210 if self.compact_layout else 280,
            bg=COLORS["viewer"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.bar_plot_canvas.pack(fill=tk.BOTH, expand=False, pady=(0, 6 if self.compact_layout else 12))
        self.heatmap_canvas = tk.Canvas(
            panel,
            height=260 if self.compact_layout else 450,
            bg=COLORS["viewer"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.heatmap_canvas.pack(fill=tk.BOTH, expand=True)
        self.plot_canvas = self.bar_plot_canvas
        self.bar_plot_canvas.bind("<Configure>", self._schedule_plot_redraw)
        self.heatmap_canvas.bind("<Configure>", self._schedule_plot_redraw)

    def _metric_card(self, parent: tk.Widget, label: str, var: tk.StringVar, accent: str, row: int, col: int) -> None:
        box = tk.Frame(
            parent,
            bg=COLORS["panel_alt"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            padx=10 if self.compact_layout else 12,
            pady=6 if self.compact_layout else 10,
        )
        box.grid(row=row, column=col, sticky="ew", padx=(0, 8) if col == 0 else (8, 0))
        tk.Label(box, text=label.upper(), bg=COLORS["panel_alt"], fg=COLORS["muted"], font=FONTS["small"]).pack(anchor="w")
        tk.Label(box, textvariable=var, bg=COLORS["panel_alt"], fg=accent, font=FONTS["metric"]).pack(anchor="w")

    def _build_result_table(self, parent: tk.Widget) -> None:
        panel = card(parent, padx=10 if self.compact_layout else 14, pady=8 if self.compact_layout else 14)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.rowconfigure(2, weight=1)
        panel.columnconfigure(0, weight=1)
        self._section_title(panel, "Muscle-Level Result", "Status akhir tiap otot berdasarkan pola temporal dan relasi otot")

        tk.Label(
            panel,
            text=SIMULATION_INTERPRETATION_NOTE,
            bg=COLORS["panel"],
            fg=COLORS["warning"],
            font=FONTS["small"],
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X, pady=(0, 4))
        tk.Label(panel, textvariable=self.summary_var, bg=COLORS["panel"], fg=COLORS["muted"], font=FONTS["small"], wraplength=520, justify=tk.LEFT).pack(
            anchor="w", fill=tk.X, pady=(0, 4)
        )
        table_wrap = tk.Frame(panel, bg=COLORS["panel"])
        table_wrap.pack(fill=tk.BOTH, expand=True)
        table_wrap.rowconfigure(0, weight=1)
        table_wrap.columnconfigure(0, weight=1)

        columns = ("node_name", "final_status", "predicted_label", "prob_deviating", "relational_risk", "deviation_score")
        self.table = ttk.Treeview(table_wrap, columns=columns, show="headings", height=7 if self.compact_layout else 14)
        headings = {
            "node_name": "Muscle / Node",
            "final_status": "Status Tampilan",
            "predicted_label": "Label Model",
            "prob_deviating": "Prob. Pola Berbeda",
            "relational_risk": "Skor Relasi",
            "deviation_score": "Skor Pola",
        }
        widths = {
            "node_name": 130,
            "final_status": 170,
            "predicted_label": 110,
            "prob_deviating": 90,
            "relational_risk": 80,
            "deviation_score": 80,
        }
        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=widths[col], anchor="w")
        self.table.tag_configure("deviating", background=COLORS["danger_tint"], foreground=COLORS["danger"])
        self.table.tag_configure("affected", background=COLORS["warning_tint"], foreground=COLORS["warning"])
        self.table.tag_configure("within_reference", background=COLORS["success_tint"], foreground=COLORS["success"])
        scroll = ttk.Scrollbar(table_wrap, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def _build_status_bar(self, parent: tk.Widget) -> None:
        bar = tk.Frame(
            parent,
            bg=COLORS["panel"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            padx=10,
            pady=4 if self.compact_layout else 7,
        )
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
            "recording": ("RECORDING", COLORS["accent_soft"], COLORS["text"]),
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
        replay_mode_selected = self.replay.replay_mode in (ReplayMode.KINEMATIC, ReplayMode.CONTROL)
        control_available = has_trial and self._control_reference_available()

        if hasattr(self, "start_button"):
            if can_run_analysis and not is_analyzing and not is_recording:
                configure_button_variant(self.start_button, "secondary")
                configure_button_variant(self.run_button, "primary")
            else:
                configure_button_variant(self.start_button, "primary")
                configure_button_variant(self.run_button, "secondary")
            self.start_button.configure(state=tk.DISABLED if is_recording or is_analyzing else tk.NORMAL)
            self.stop_button.configure(state=tk.NORMAL if is_recording else tk.DISABLED)
            self.load_button.configure(state=tk.DISABLED if is_recording or is_analyzing else tk.NORMAL)
            self.run_button.configure(state=tk.NORMAL if can_run_analysis and not is_analyzing else tk.DISABLED)
            self.record_next_button.configure(state=tk.DISABLED if is_recording or is_analyzing else tk.NORMAL)
            self.load_session_button.configure(state=tk.DISABLED if is_recording or is_analyzing or not self._session_trials else tk.NORMAL)
            self.open_replay_button.configure(state=tk.NORMAL if not is_analyzing else tk.DISABLED)
            self.close_replay_button.configure(state=tk.NORMAL if self.replay.status().viewer_open and not is_analyzing else tk.DISABLED)
            self.replay_kinematic_button.configure(state=tk.NORMAL if can_replay and not is_analyzing else tk.DISABLED)
            self.replay_control_button.configure(state=tk.NORMAL if can_replay and control_available and not is_analyzing else tk.DISABLED)
            self.play_button.configure(state=tk.NORMAL if can_replay and replay_mode_selected and not is_analyzing else tk.DISABLED)
            self.pause_button.configure(state=tk.NORMAL if can_replay and replay_mode_selected and not is_analyzing else tk.DISABLED)
            self.reset_button.configure(state=tk.NORMAL if can_replay and replay_mode_selected and not is_analyzing else tk.DISABLED)
            self.open_muscle_graph_button.configure(state=tk.NORMAL)
            self.close_muscle_graph_button.configure(state=tk.NORMAL if self.muscle_graph.status().open else tk.DISABLED)
            self.generate_current_embedding_button.configure(
                state=tk.NORMAL if self.current_analysis_result is not None and not is_analyzing else tk.DISABLED
            )
            self.open_current_embedding_plot_button.configure(state=tk.NORMAL if not is_analyzing else tk.DISABLED)
            relation_state = tk.NORMAL if self.current_analysis_result is not None and not is_analyzing else tk.DISABLED
            self.open_relation_scatter_plot_button.configure(state=relation_state)
            self.open_top_relation_table_button.configure(state=relation_state)
            self.open_top_relation_bar_button.configure(state=relation_state)

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

    def open_muscle_graph(self) -> None:
        if self.muscle_graph.open_or_focus():
            if self.current_analysis_result is not None:
                self.muscle_graph.set_analysis_result(self.current_analysis_result)
            self._refresh_muscle_graph_status()
            self._set_status("Muscle graph viewer opened. It follows live ctrl during recording and replay ctrl during replay.", "done")
        else:
            self._refresh_muscle_graph_status()
            self._set_status(self.muscle_graph.status().message, "error")

    def close_muscle_graph(self) -> None:
        self.muscle_graph.close()
        self._refresh_muscle_graph_status()
        self._set_status("Muscle graph viewer closed.", "done")

    def _current_trial_embedding_png(self) -> Path:
        return (
            Path("artifacts")
            / "dynamic_temporal_gcn_node_classifier"
            / "current_trial_embedding"
            / "current_trial_node_embedding_pca.png"
        )

    def _warn_current_trial_embedding_requires_analysis(self) -> None:
        message = "Jalankan analisis pada trial terlebih dahulu sebelum membuka embedding current trial."
        self._set_status(message, "error")
        messagebox.showwarning("Current Trial Embedding", message)

    def generate_current_trial_embedding_plot(self) -> None:
        if self.current_trial is None or self.current_analysis_result is None:
            self._warn_current_trial_embedding_requires_analysis()
            return
        artifact_paths = dict(self.current_analysis_result.get("artifact_paths") or {})
        graph_json = artifact_paths.get("graph_json")
        if not graph_json:
            self._set_status("Current analysis result has no graph_json artifact path.", "error")
            messagebox.showerror("Generate Current Trial Embedding Failed", "Current analysis result has no graph_json artifact path.")
            return
        try:
            from scripts.export_node_embedding_visualization import export_current_trial_embedding_visualization

            summary = export_current_trial_embedding_visualization(
                graph_json=Path(str(graph_json)),
                analysis_result=self.current_analysis_result,
                checkpoint_path=Path(str(self.current_analysis_result.get("model_path") or self.inference.config.model_path)),
            )
            self._set_status(f"Generated current-trial embedding: {summary.get('png')}", "done")
        except Exception as exc:
            self._set_status("Failed to generate current-trial embedding plot.", "error")
            messagebox.showerror("Generate Current Trial Embedding Failed", str(exc))

    def open_current_trial_embedding_plot(self) -> None:
        if self.current_trial is None or self.current_analysis_result is None:
            self._warn_current_trial_embedding_requires_analysis()
            return
        figure_path = self._current_trial_embedding_png()
        if not figure_path.exists():
            messagebox.showwarning(
                "Current Trial Embedding Plot Not Found",
                "Current trial embedding figure is not available yet.\n\n"
                "Run Analysis first, then click Generate Trial Embedding Plot.",
            )
            return
        try:
            os.startfile(str(figure_path.resolve()))
            self._set_status(f"Opened current-trial node embedding plot: {figure_path}", "done")
        except Exception as exc:
            self._set_status("Failed to open current-trial node embedding plot.", "error")
            messagebox.showerror("Open Current Trial Embedding Plot Failed", str(exc))

    def _export_relation_score_outputs(self) -> Optional[dict]:
        if self.current_trial is None or self.current_analysis_result is None:
            message = "Jalankan analisis pada trial terlebih dahulu sebelum membuat visualisasi skor relasi antarotot."
            self._set_status(message, "error")
            messagebox.showwarning("Relation Score Plot", message)
            return None
        try:
            from scripts.export_relation_score_visualization import export_relation_score_visualization

            return export_relation_score_visualization(analysis_result=self.current_analysis_result)
        except Exception as exc:
            self._set_status("Failed to export relation score outputs.", "error")
            messagebox.showerror("Export Relation Score Outputs Failed", str(exc))
            return None

    def _open_relation_output(self, key: str, *, label: str) -> None:
        summary = self._export_relation_score_outputs()
        if not summary:
            return
        try:
            path = Path(str(summary[key]))
            os.startfile(str(path.resolve()))
            self._set_status(f"Opened {label}: {path}", "done")
        except Exception as exc:
            self._set_status(f"Failed to open {label}.", "error")
            messagebox.showerror(f"Open {label} Failed", str(exc))

    def open_relation_scatter_plot(self) -> None:
        self._open_relation_output("png", label="relation scatter plot")

    def open_top_relation_table(self) -> None:
        self._open_relation_output("top_relations_csv", label="top relation table")

    def open_top_relation_bar_chart(self) -> None:
        self._open_relation_output("top_relation_bar_png", label="top relation bar chart")

    def _refresh_muscle_graph_status(self) -> None:
        status = self.muscle_graph.status()
        self.muscle_graph_status_var.set(
            f"{status.message} | nodes={status.nodes_displayed} edges={status.edges_displayed}"
        )

    def reset_qpos_provider(self) -> None:
        if self.state == GuiState.RECORDING:
            self._set_status("Stop recording before resetting the qpos provider.", "error")
            return
        old_provider = self.leap_provider
        provider = self._create_qpos_provider()
        try:
            started = bool(provider.start())
        except Exception as exc:
            started = False
            try:
                provider.message = f"{type(exc).__name__}: {exc}"
            except Exception:
                pass
        if started:
            if old_provider is not None:
                old_provider.close()
            self.leap_provider = provider
            self.recorder.set_qpos_provider(provider)
            self.qpos_status_var.set(provider.status().message)
            self._set_status(f"{self._qpos_provider_name()} qpos provider reset.", "done")
        else:
            failed_status = provider.status().message
            self.leap_provider = old_provider
            self.recorder.set_qpos_provider(old_provider)
            self.qpos_status_var.set(failed_status)
            self._set_status(f"Qpos provider reset failed: {failed_status}", "error")

    def connect_smart_glove(self) -> None:
        if self.state == GuiState.RECORDING:
            self._set_status("Stop recording before changing the smart glove connection.", "error")
            return
        host = str(self.glove_host_var.get()).strip() or "127.0.0.1"
        try:
            port = int(str(self.glove_port_var.get()).strip())
            if not 1 <= port <= 65535:
                raise ValueError("port must be between 1 and 65535")
        except Exception as exc:
            self.glove_connection_status_var.set("Invalid port")
            self._set_status(f"Smart glove connection failed: {exc}", "error")
            return

        provider = UdpQposProvider(host=host, port=port, qpos_dim=23)
        if not provider.start():
            self.glove_connection_status_var.set("Connection failed")
            self.qpos_status_var.set(provider.message)
            self._set_status(provider.message, "error")
            return

        old_provider = self.leap_provider
        if old_provider is not None:
            old_provider.close()
        self.qpos_source_mode = "smart_glove"
        self.glove_host_var.set(host)
        self.glove_port_var.set(str(port))
        self.leap_provider = provider
        self.recorder.set_qpos_provider(provider)
        self.glove_connection_status_var.set(f"Connected ({host}:{port})")
        self.qpos_status_var.set(provider.status().message)
        viewer_ready = bool(self.replay.open_or_focus_viewer())
        if viewer_ready:
            self.preview_ready_var.set("Smart glove connected | MuJoCo live preview active | waiting for glove frames")
            self._set_status(f"Smart glove connected on {host}:{port}; live hand model ready.", "done")
        else:
            self.preview_ready_var.set(f"Smart glove connected | MuJoCo viewer unavailable: {self.replay.status().message}")
            self._set_status(f"Smart glove connected, but hand model could not open: {self.replay.status().message}", "error")

    def disconnect_smart_glove(self) -> None:
        if self.state == GuiState.RECORDING:
            self._set_status("Stop recording before disconnecting the smart glove.", "error")
            return
        if isinstance(self.leap_provider, UdpQposProvider):
            self.leap_provider.close()
            self.leap_provider = None
            self.recorder.set_qpos_provider(None)
        self.glove_connection_status_var.set("Disconnected")
        self.qpos_status_var.set("Smart glove disconnected.")
        self._set_status("Smart glove disconnected.", "done")

    def _create_qpos_provider(self) -> object:
        if self.qpos_source_mode in {"leapc", "direct", "leapc-direct", "leapc_direct"}:
            sdk_dir = str(resolve_ultraleap_sdk_dir())
            tracking_mode = str(os.environ.get("HAND_LEAP_TRACKING_MODE", "auto")).strip().lower() or "auto"
            return LeapCQposProvider(sdk_dir=sdk_dir, qpos_dim=23, tracking_mode=tracking_mode)
        if self.qpos_source_mode in {"udp", "remote", "smart_glove", "smart-glove", "glove"}:
            host = str(os.environ.get("HAND_QPOS_UDP_HOST", "127.0.0.1")).strip() or "127.0.0.1"
            try:
                port = int(str(os.environ.get("HAND_QPOS_UDP_PORT", "15000")).strip())
            except Exception:
                port = 15000
            return UdpQposProvider(host=host, port=port, qpos_dim=23)
        return LiveLeapQposProvider(show_leap_view=False)

    def _qpos_provider_name(self) -> str:
        if self.qpos_source_mode in {"leapc", "direct", "leapc-direct", "leapc_direct"}:
            return "Direct LeapC"
        if self.qpos_source_mode in {"udp", "remote"}:
            return "UDP LeapC"
        if self.qpos_source_mode in {"smart_glove", "smart-glove", "glove"}:
            return "Smart glove UDP"
        return "Leap"

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
        messages.append("GUI idle. Leap viewer and MuJoCo viewer are opened from buttons.")
        self.leap_viewer_status_var.set(self.leap_viewer.status().message)
        if self.leap_provider is None:
            provider = self._create_qpos_provider()
            if provider.start():
                self.leap_provider = provider
                self.recorder.set_qpos_provider(provider)
                messages.append(f"{self._qpos_provider_name()} qpos provider ready")
                self.qpos_status_var.set(provider.status().message)
            else:
                messages.append(f"{self._qpos_provider_name()} qpos provider unavailable: {provider.message}")
                self.qpos_status_var.set(provider.message)
        self.preview_ready_var.set(" | ".join(messages))
        self._set_status(" | ".join(messages), "normal")
        self._refresh_replay_status()
        self._set_state(GuiState.IDLE)

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

    def _render_live_preview(self, qpos: Optional[object], *, message: str) -> bool:
        if qpos is None or not self._live_preview_allowed():
            return False
        return bool(self.replay.render_qpos(qpos, message=message))

    def _update_muscle_graph_viewer(self) -> None:
        if not self.muscle_graph.status().open:
            return
        replay_status = self.replay.status()
        if self.state == GuiState.RECORDING:
            mode = "RECORDING_LIVE"
        elif replay_status.replay_mode == ReplayMode.CONTROL.value:
            mode = "CONTROL_REPLAY"
        elif replay_status.replay_mode == ReplayMode.KINEMATIC.value:
            mode = "KINEMATIC_REPLAY"
        else:
            mode = "LIVE_PREVIEW"
        self.muscle_graph.update(
            mode=mode,
            ctrl_values=self.replay.last_ctrl_vector,
            frame_index=int(replay_status.frame_index),
            num_frames=int(replay_status.num_frames),
        )
        self._refresh_muscle_graph_status()

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
            provider = self._create_qpos_provider()
            if provider.start():
                self.leap_provider = provider
                self.recorder.set_qpos_provider(provider)
                provider_message = f"{self._qpos_provider_name()} qpos provider active."
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
            trial = self.recorder.stop_and_save(
                subject_id=self.subject_var.get(),
                motion_type=self.motion_var.get(),
                trial_label=internal_trial_label(self.trial_label_var.get()),
            )
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
        trial_label = display_trial_label(trial.meta.get("trial_label") or trial.meta.get("label") or "normal")
        label = f"{index:02d} | {trial_label} | {trial.trial_id} | {trial.num_frames} frames"
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
        if self.current_analysis_result is not None:
            self._set_state(GuiState.ANALYSIS_DONE)
            self._set_status(f"Loaded session trial with cached analysis: {trial.qpos_csv}", "done")
        else:
            self._set_state(GuiState.TRIAL_READY)
            self._set_status(f"Loaded session trial: {trial.qpos_csv}", "done")

    def load_existing_trial(self) -> None:
        selected = filedialog.askdirectory(
            title="Load trial folder containing qpos.csv",
            initialdir=str((Path("records") / internal_trial_label(self.trial_label_var.get()) / str(self.subject_var.get() or "subject_gui")).resolve()),
        )
        if not selected:
            return
        try:
            trial = load_trial(selected)
        except Exception as exc:
            self._set_status("Load failed.", "error")
            messagebox.showerror(
                "Load Failed",
                f"Select a trial folder that contains qpos.csv, for example:\n"
                f"records/subject_gui/20260611_094545_open_to_fist\n\n{exc}",
            )
            return
        self._set_trial(trial)
        if self.current_analysis_result is not None:
            self._set_state(GuiState.ANALYSIS_DONE)
            self._set_status(f"Loaded trial folder with cached analysis: {trial.trial_dir}", "done")
        else:
            self._set_state(GuiState.TRIAL_READY)
            self._set_status(f"Loaded trial folder: {trial.trial_dir}. Select QPOS or CTRL replay mode before pressing Play.", "done")

    def _set_trial(self, trial: TrialData) -> None:
        self.current_trial = trial
        self.current_analysis_result = None
        self.muscle_graph.set_analysis_result(None)
        self.path_var.set(str(trial.qpos_csv))
        self.trial_id_var.set(trial.trial_id)
        self.active_trial_label_var.set(display_trial_label(trial.meta.get("trial_label") or trial.meta.get("label") or "normal"))
        self.frame_count_var.set(str(trial.num_frames))
        self.summary_var.set("No activity-pattern result for this trial yet.")
        self.total_nodes_var.set("0")
        self.within_count_var.set("0")
        self.deviating_count_var.set("0")
        self.top_nodes_var.set("-")
        self.highest_region_var.set("-")
        self.score_var.set("-")
        if hasattr(self, "plot_controller"):
            self.plot_controller.draw_empty("Run analysis to show muscle activity-pattern plot.")
        self._clear_table()
        self.replay.set_trial(trial)
        self.progress_scale.configure(from_=0, to=max(0, trial.num_frames - 1))
        self._refresh_control_available_status()
        self._refresh_replay_status()
        cached_result = self._load_cached_analysis_result(trial)
        if cached_result is not None:
            self._show_result(cached_result)

    def _load_cached_analysis_result(self, trial: TrialData) -> Optional[dict]:
        result_path = trial.trial_dir / "result.json"
        if not result_path.exists():
            return None
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._set_status(f"Cached analysis exists but could not be loaded: {exc}", "error")
            return None
        if not isinstance(result, dict) or not result.get("nodes"):
            return None
        result.setdefault("artifact_paths", {})
        result["artifact_paths"].setdefault("result_json", str(result_path))
        result["artifact_paths"].setdefault("result_csv", str(trial.trial_dir / "result.csv"))
        return result

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
        self.muscle_graph.set_analysis_result(result)
        self._clear_table()
        summary = dict(result.get("summary") or {})
        affected_summary = dict((result.get("affected_muscle_analysis") or {}).get("summary") or {})
        top_nodes = list(summary.get("top_deviating_nodes") or [])
        top_affected = list(summary.get("top_affected_muscles") or affected_summary.get("top_affected_muscles") or [])
        num_deviating = int(affected_summary.get("deviating_muscles", summary.get("num_deviating", 0)))
        num_affected = int(affected_summary.get("affected_muscles", summary.get("num_affected", 0)))
        num_nodes = int(summary.get("num_nodes", 0))
        num_within = int(affected_summary.get("within_reference_muscles", max(0, num_nodes - num_deviating - num_affected)))
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
        affected_lines = [
            f"{idx}. {item.get('muscle_name', '')} <- {item.get('strongest_related_deviating_muscle', '')} "
            f"(skor relasi={float(item.get('relational_risk', 0.0)):.3f})"
            for idx, item in enumerate(top_affected[:5], start=1)
        ]
        top_text_parts = [
            "Pola Aktivitas Berbeda:",
            *([f"{idx}. {name}" for idx, name in enumerate(top_nodes[:5], start=1)] or ["-"]),
            "",
            "Terkait Secara Relasional:",
            *(affected_lines or ["-"]),
        ]
        self.top_nodes_var.set("\n".join(top_text_parts))
        self.score_var.set(
            f"otot={float(summary.get('overall_node_deviation_score', 0.0)):.3f} | relasi={float(summary.get('overall_edge_deviation_score', 0.0)):.3f}"
        )
        self.summary_var.set(
            f"Trial {result.get('trial_id')} | pola berbeda: {num_deviating}/{num_nodes} | "
            f"terkait relasional: {num_affected}/{num_nodes} | top: {', '.join(top_nodes[:5]) or '-'}"
        )
        for row in list(result.get("nodes") or []):
            label = str(row.get("predicted_label") or "")
            final_status = str(row.get("affected_final_status") or ("terindikasi_menyimpang" if label == "deviating" else "sesuai_acuan"))
            display_final_status = str(row.get("display_final_status") or display_status(final_status))
            display_predicted_label = str(row.get("display_predicted_label") or display_label(label))
            if final_status == "terindikasi_menyimpang":
                tag = "deviating"
            elif final_status == "berpotensi_terdampak":
                tag = "affected"
            else:
                tag = "within_reference"
            self.table.insert(
                "",
                tk.END,
                values=(
                    row.get("node_name", ""),
                    display_final_status,
                    display_predicted_label,
                    f"{float(row.get('probability_deviating', 0.0)):.3f}",
                    f"{float(row.get('relational_risk', 0.0)):.3f}",
                    f"{float(row.get('deviation_score', 0.0)):.3f}",
                ),
                tags=(tag,),
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
            rendered = self._render_live_preview(self.recorder.latest_qpos(), message="LIVE_PREVIEW recording qpos + ctrl estimate")
            if not rendered:
                self.preview_ready_var.set(f"QPOS recording live | MuJoCo preview not rendering: {self.replay.status().message}")
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
                    rendered = self._render_live_preview(latest_qpos, message="LIVE_PREVIEW qpos + ctrl estimate")
                    if rendered:
                        self.preview_ready_var.set(
                            f"{self._qpos_provider_name()} qpos provider live | MuJoCo live preview active | trial replay is manual"
                        )
                    else:
                        self.preview_ready_var.set(
                            f"{self._qpos_provider_name()} qpos provider live | MuJoCo preview not rendering: {self.replay.status().message}"
                        )
                else:
                    self.preview_ready_var.set("Qpos provider is stale or stuck. Reset QPOS Provider.")
            self._refresh_qpos_provider_status()
        if self.replay.playing:
            self.replay.step(1)
        self._refresh_replay_status()
        self._update_muscle_graph_viewer()
        self.leap_viewer_status_var.set(self.leap_viewer.status().message)
        self._handle_analysis_queue()
        self.after(33, self._tick)

    def _on_close(self) -> None:
        self._cancel_auto_stop_timer()
        self.muscle_graph.close()
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
    inference_config = BinaryGCNInferenceController().config
    return {
        "gui_backend": "tkinter_dark_dashboard",
        "leap_viewer": "external_opencv_skeleton_viewer_on_demand_from_local_leapc_python_bindings",
        "mujoco_viewer": "external_mujoco_kinematic_and_control_replay_on_request",
        "qpos_source_default": "leapc",
        "leap_tracking_mode_default": "auto",
        "ultraleap_sdk_resolved": str(resolve_ultraleap_sdk_dir()),
        "ctrl_preview_mode_default": "pd_dynamic",
        "prediction_deviating_threshold": f"{float(inference_config.prediction_deviating_threshold):.3f}",
        "deviating_threshold": f"{float(inference_config.threshold_deviating):.3f}",
        "affected_threshold": f"{float(inference_config.threshold_affected):.3f}",
        "model_path": str(inference_config.model_path),
        "reference_dir": str(inference_config.reference_dir),
    }


def main(*, show_external_leap_skeleton: bool = True) -> None:
    app = TrialReplayClassifierApp(show_external_leap_skeleton=show_external_leap_skeleton)
    app.mainloop()
