from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import leap

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from myosuite.utils import gym

    MYOSUITE_AVAILABLE = True
except Exception:
    gym = None
    MYOSUITE_AVAILABLE = False

try:
    import mujoco
except ImportError as e:
    raise ImportError("mujoco package required") from e

HAVE_OSQP = False
try:
    import osqp
    import scipy.sparse as spa

    HAVE_OSQP = True
except Exception:
    HAVE_OSQP = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

# ---------------------------------------------------------------------------
# Artifact Paths
# ---------------------------------------------------------------------------
def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_local_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_artifact_root() -> Path:
    load_dotenv()
    raw_root = os.getenv("ARTIFACT_ROOT")
    if raw_root:
        return resolve_local_path(raw_root)
    return PROJECT_ROOT / "artifacts" / "graphs"


class RollingRuntimeProfiler:
    FRAME_STAGE_NAMES = (
        "controller",
        "graph_update",
        "graph_canvas_draw",
        "main_env_render",
        "validation_env_render",
        "gui_refresh",
        "recording",
        "total_frame",
    )
    EVENT_STAGE_NAMES = (
        "tracking_input",
        "qpos_mapping",
    )

    def __init__(self, *, log_interval_seconds: float = 25.0, history_size: int = 60, log_spikes: bool = True):
        self.log_interval_seconds = max(1.0, float(log_interval_seconds))
        self.history_size = max(16, int(history_size))
        self.log_spikes = bool(log_spikes)
        self.lock = threading.Lock()
        self.last_log_time = time.time()
        self.stage_histories: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.history_size))
        self.stage_current: dict[str, float] = {}
        self.stage_spike_count: dict[str, int] = defaultdict(int)
        self.stage_worst: dict[str, tuple[float, int]] = {}
        self.frame_stage_accumulator: dict[str, float] = {}
        self.current_frame_index = -1
        self.current_runtime_flags: dict[str, Any] = {}
        self.worst_spike_stage = ""
        self.worst_spike_ms = 0.0
        self.worst_spike_frame = -1

    def set_runtime_flags(self, **flags: Any) -> None:
        with self.lock:
            self.current_runtime_flags.update(flags)

    def start_frame(self, frame_index: int) -> None:
        with self.lock:
            self.current_frame_index = int(frame_index)
            self.frame_stage_accumulator = {name: 0.0 for name in self.FRAME_STAGE_NAMES}

    def add(self, name: str, duration_ms: float) -> None:
        self.add_event(name, duration_ms)

    def add_event(self, name: str, duration_ms: float, *, frame_index: int | None = None) -> None:
        value = float(duration_ms)
        if not math.isfinite(value):
            return
        with self.lock:
            stage_name = str(name)
            self.stage_current[stage_name] = value
            self.stage_histories[stage_name].append(value)
            self._register_spike(stage_name, value, self.current_frame_index if frame_index is None else int(frame_index))

    def add_frame_stage(self, name: str, duration_ms: float) -> None:
        value = float(duration_ms)
        if not math.isfinite(value):
            return
        stage_name = str(name)
        with self.lock:
            self.frame_stage_accumulator[stage_name] = self.frame_stage_accumulator.get(stage_name, 0.0) + value

    def finish_frame(self, total_frame_ms: float) -> None:
        value = float(total_frame_ms)
        if not math.isfinite(value):
            return
        with self.lock:
            self.frame_stage_accumulator["total_frame"] = value
            for stage_name in self.FRAME_STAGE_NAMES:
                stage_value = float(self.frame_stage_accumulator.get(stage_name, 0.0))
                self.stage_current[stage_name] = stage_value
                self.stage_histories[stage_name].append(stage_value)
                self._register_spike(stage_name, stage_value, self.current_frame_index)

    def _register_spike(self, stage_name: str, value: float, frame_index: int) -> None:
        threshold = 8.0
        severe_threshold = None
        if stage_name == "total_frame":
            threshold = 16.7
            severe_threshold = 33.0
        if value <= threshold:
            return

        self.stage_spike_count[stage_name] += 1
        previous_worst = self.stage_worst.get(stage_name, (0.0, -1))
        if value > float(previous_worst[0]):
            self.stage_worst[stage_name] = (value, int(frame_index))
        if value > self.worst_spike_ms:
            self.worst_spike_stage = stage_name
            self.worst_spike_ms = value
            self.worst_spike_frame = int(frame_index)

        severity = "SEVERE" if severe_threshold is not None and value > severe_threshold else "WARN"
        flags = {
            "recording": bool(self.current_runtime_flags.get("recording_active", False)),
            "ctrl_validation": bool(self.current_runtime_flags.get("ctrl_validation_enabled", False)),
            "muscle_graph": bool(self.current_runtime_flags.get("muscle_graph_enabled", False)),
        }
        if self.log_spikes:
            print(
                f"[{severity}][STUTTER] frame={int(frame_index)} stage={stage_name} "
                f"elapsed_ms={value:.2f} recording={int(flags['recording'])} "
                f"ctrl_validation={int(flags['ctrl_validation'])} muscle_graph={int(flags['muscle_graph'])}"
            )

    def get_stage_stats(self, name: str) -> dict[str, float]:
        with self.lock:
            history = tuple(self.stage_histories.get(str(name), ()))
            current = float(self.stage_current.get(str(name), 0.0))
            spike_count = int(self.stage_spike_count.get(str(name), 0))
        if not history:
            return {"current_ms": current, "mean_ms": 0.0, "max_ms": 0.0, "spike_count": float(spike_count)}
        return {
            "current_ms": current,
            "mean_ms": float(np.mean(history)),
            "max_ms": float(np.max(history)),
            "spike_count": float(spike_count),
        }

    def overlay_snapshot(self) -> dict[str, float]:
        total_stats = self.get_stage_stats("total_frame")
        controller_stats = self.get_stage_stats("controller")
        graph_update_stats = self.get_stage_stats("graph_update")
        graph_draw_stats = self.get_stage_stats("graph_canvas_draw")
        render_ms = (
            self.get_stage_stats("main_env_render")["current_ms"]
            + self.get_stage_stats("validation_env_render")["current_ms"]
        )
        total_ms = float(total_stats["current_ms"])
        fps = 1000.0 / total_ms if total_ms > 1e-6 else 0.0
        return {
            "total_ms": total_ms,
            "controller_ms": float(controller_stats["current_ms"]),
            "graph_ms": float(graph_update_stats["current_ms"]),
            "draw_ms": float(graph_draw_stats["current_ms"]),
            "render_ms": render_ms,
            "fps": fps,
        }

    def summary_snapshot(self) -> dict[str, Any]:
        total_stats = self.get_stage_stats("total_frame")
        controller_stats = self.get_stage_stats("controller")
        graph_update_stats = self.get_stage_stats("graph_update")
        graph_draw_stats = self.get_stage_stats("graph_canvas_draw")
        main_render_stats = self.get_stage_stats("main_env_render")
        validation_render_stats = self.get_stage_stats("validation_env_render")
        render_mean = float(main_render_stats["mean_ms"] + validation_render_stats["mean_ms"])
        total_mean = float(total_stats["mean_ms"])
        return {
            "avg_total_frame_ms": total_mean,
            "avg_controller_ms": float(controller_stats["mean_ms"]),
            "avg_graph_ms": float(graph_update_stats["mean_ms"]),
            "avg_draw_ms": float(graph_draw_stats["mean_ms"]),
            "avg_render_ms": render_mean,
            "estimated_fps": (1000.0 / total_mean) if total_mean > 1e-6 else 0.0,
            "worst_spike_stage": self.worst_spike_stage,
            "worst_spike_ms": float(self.worst_spike_ms),
            "worst_spike_frame": int(self.worst_spike_frame),
        }

    def maybe_log(self, now: float | None = None) -> None:
        current_time = time.time() if now is None else float(now)
        if (current_time - self.last_log_time) < self.log_interval_seconds:
            return
        self.last_log_time = current_time
        summary = self.summary_snapshot()
        print(
            "[PROFILE] "
            f"avg_total={summary['avg_total_frame_ms']:.2f}ms | "
            f"avg_controller={summary['avg_controller_ms']:.2f}ms | "
            f"avg_graph={summary['avg_graph_ms']:.2f}ms | "
            f"avg_draw={summary['avg_draw_ms']:.2f}ms | "
            f"avg_render={summary['avg_render_ms']:.2f}ms | "
            f"fps={summary['estimated_fps']:.1f} | "
            f"worst={summary['worst_spike_stage'] or '-'}:{summary['worst_spike_ms']:.2f}ms"
        )


@dataclass
class RuntimeDiagnosticOptions:
    disable_graph_update: bool = False
    disable_graph_draw: bool = False
    disable_side_panels: bool = False
    disable_controller: bool = False
    disable_validation_window: bool = False
    disable_main_env_render: bool = False
    disable_cv_overlay: bool = False


def draw_runtime_profiler_overlay(image: np.ndarray, profiler: RollingRuntimeProfiler) -> None:
    snapshot = profiler.overlay_snapshot()
    x0, y0 = 12, 12
    x1, y1 = 232, 104
    cv2.rectangle(image, (x0, y0), (x1, y1), (16, 16, 20), -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (92, 100, 108), 1)
    lines = (
        f"frame {snapshot['total_ms']:.2f} ms  {snapshot['fps']:.1f} fps",
        f"ctrl  {snapshot['controller_ms']:.2f} ms",
        f"graph {snapshot['graph_ms']:.2f} ms",
        f"draw  {snapshot['draw_ms']:.2f} ms",
        f"render {snapshot['render_ms']:.2f} ms",
    )
    y = y0 + 18
    for line in lines:
        cv2.putText(image, line, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        y += 17


def build_diagnostic_ctrl_state(q_target, controller_context, previous_ctrl: np.ndarray | None = None) -> dict[str, Any]:
    q_target = np.asarray(q_target, dtype=float).reshape(-1)
    if previous_ctrl is None:
        previous_ctrl = controller_context.get("last_ctrl_analysis")
    if previous_ctrl is None:
        previous_ctrl = _zero_ctrl(controller_context)
    ctrl_vec = _sanitize_ctrl_signal(previous_ctrl)
    return {
        "ctrl_raw": ctrl_vec.copy(),
        "ctrl_smoothed": ctrl_vec.copy(),
        "ctrl_analysis": ctrl_vec.copy(),
        "ctrl_rolling_mean": ctrl_vec.copy(),
        "ctrl_rolling_median": ctrl_vec.copy(),
        "q_target_used": q_target.copy(),
    }


def apply_tracking_tuning_preset(
    preset_name: str | None,
    *,
    qpos_deadband_norm: float,
    ctrl_smoothing_alpha: float,
    hold_if_small_change: bool,
    hand_mapping_debug: bool,
) -> dict[str, Any]:
    preset = str(preset_name or DEFAULT_TRACKING_TUNING_PRESET).strip().lower()
    if preset not in TRACKING_TUNING_PRESETS:
        preset = DEFAULT_TRACKING_TUNING_PRESET
    overrides = dict(TRACKING_TUNING_PRESETS.get(preset) or {})
    return {
        "preset": preset,
        "qpos_deadband_norm": float(overrides.get("qpos_deadband_norm", qpos_deadband_norm)),
        "ctrl_smoothing_alpha": float(overrides.get("ctrl_smoothing_alpha", ctrl_smoothing_alpha)),
        "hold_if_small_change": bool(overrides.get("hold_if_small_change", hold_if_small_change)),
        "hand_mapping_debug": bool(overrides.get("mapping_debug", False) or hand_mapping_debug),
    }


def print_hand_mapping_debug(snapshot: MapperDebugSnapshot, *, preset_name: str, frame_hint: int) -> None:
    thumb = dict(snapshot.thumb or {})
    print(
        "[MAP] "
        f"preset={preset_name} frame={int(frame_hint)} "
        f"thumb cmc_abd={float(thumb.get('cmc_abduction', 0.0)):.3f} "
        f"cmc_flex={float(thumb.get('cmc_flexion', 0.0)):.3f} "
        f"mcp={float(thumb.get('mcp_flexion', 0.0)):.3f} "
        f"ip={float(thumb.get('ip_flexion', 0.0)):.3f}"
    )
    for finger_name in ("index", "middle", "ring", "little"):
        finger = dict((snapshot.fingers or {}).get(finger_name) or {})
        if not finger:
            continue
        print(
            "[MAP] "
            f"{finger_name} "
            f"mcp_raw={float(finger.get('mcp_flex_raw', 0.0)):.3f} "
            f"mcp_out={float(finger.get('mcp_flex', 0.0)):.3f} "
            f"pip_raw={float(finger.get('pip_flex_raw', 0.0)):.3f} "
            f"pip_direct={float(finger.get('pip_direct', 0.0)):.3f} "
            f"pip_coupled={float(finger.get('pip_coupled', 0.0)):.3f} "
            f"pip_out={float(finger.get('pip_flex', 0.0)):.3f} "
            f"dip_raw={float(finger.get('dip_flex_raw', 0.0)):.3f} "
            f"dip_direct={float(finger.get('dip_direct', 0.0)):.3f} "
            f"dip_coupled={float(finger.get('dip_coupled', 0.0)):.3f} "
            f"dip_out={float(finger.get('dip_flex', 0.0)):.3f} "
            f"pip_gain={float(finger.get('pip_gain', 0.0)):.2f}"
        )
# ---------------------------------------------------------------------------
# Ctrl Estimation Probe
# ---------------------------------------------------------------------------
DEFAULT_MYOSUITE_ENV_ID = "myoHandPoseRandom-v0"
ACTUATOR_AUDIT_STEPS = 20
DEFAULT_ACTUATOR_TORQUE_MATRIX_MODE = "activation_probe"
VALID_ACTUATOR_TORQUE_MATRIX_MODES = ("activation_probe",)
ACTUATOR_MATRIX_COL_NORM_EPS = 1e-8
ACTUATOR_MATRIX_FRO_NORM_EPS = 1e-8
ACTUATOR_MATRIX_COND_WARN = 1e12
CTRL_OUTPUT_DEADBAND = 1e-6
CTRL_QP_REG = 1e-4
CTRL_QP_MAX_ITERS = 400
CTRL_QP_TOL = 1e-8
ESTIMATE_CTRL_HOLD_PRIOR_WEIGHT = 5e-2

def _get_raw_model_data(model, data):
    """Return the underlying mujoco._structs.MjModel/MjData from possible wrappers.

    MyoSuite typically exposes dm_control-style wrapper objects with a .ptr
    attribute pointing to the low-level mujoco structs. We normalize that here.
    """
    m = getattr(model, "ptr", None)
    if m is None:
        m = getattr(model, "_model", None)
    if m is None:
        m = model

    d = getattr(data, "ptr", None)
    if d is None:
        d = getattr(data, "_data", None)
    if d is None:
        d = data

    return m, d


def set_pose(model, data, qpos_target):
    """Set the model state to a given target qpos (static, zero velocity)."""
    qpos_target = np.asarray(qpos_target, dtype=float)
    nq = model.nq

    # Build a full-length qpos vector; fill from qpos_target and zero the rest.
    q = np.zeros(nq, dtype=float)
    n = min(len(qpos_target), nq)
    q[:n] = qpos_target[:n]

    data.qpos[:] = q
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    m_raw, d_raw = _get_raw_model_data(model, data)
    mujoco.mj_forward(m_raw, d_raw)


def _normalize_actuator_torque_matrix_mode(mode):
    normalized = str(mode or DEFAULT_ACTUATOR_TORQUE_MATRIX_MODE).strip().lower()
    if normalized not in VALID_ACTUATOR_TORQUE_MATRIX_MODES:
        return DEFAULT_ACTUATOR_TORQUE_MATRIX_MODE
    return normalized


def _capture_probe_state(model, data):
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "qacc": data.qacc.copy(),
        "ctrl": data.ctrl.copy(),
        "act": data.act.copy() if int(getattr(model, "na", 0)) > 0 else None,
        "time": float(getattr(data, "time", 0.0)),
    }


def _restore_probe_state(model, data, state):
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    data.qacc[:] = state["qacc"]
    data.ctrl[:] = state["ctrl"]
    if state["act"] is not None:
        data.act[:] = state["act"]
    if hasattr(data, "time"):
        data.time = float(state["time"])


def _set_activation_from_ctrl(model, data, ctrl_values):
    if int(getattr(model, "na", 0)) <= 0 or len(getattr(data, "act", ())) == 0:
        return
    data.act[:] = 0.0
    actadr = getattr(model, "actuator_actadr", None)
    if actadr is None:
        return
    ctrl_arr = np.asarray(ctrl_values, dtype=float).reshape(-1)
    for j in range(min(int(model.nu), len(ctrl_arr))):
        a_idx = int(actadr[j])
        if 0 <= a_idx < len(data.act):
            data.act[a_idx] = max(data.act[a_idx], float(np.clip(ctrl_arr[j], 0.0, 1.0)))


def _column_norm_summary(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[1] == 0:
        return {
            "shape": [int(A.shape[0]) if A.ndim >= 1 else 0, 0],
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "fro_norm": 0.0,
            "num_near_zero": 0,
            "num_informative": 0,
        }
    col_norms = np.linalg.norm(A, axis=0)
    informative_mask = col_norms > float(ACTUATOR_MATRIX_COL_NORM_EPS)
    return {
        "shape": [int(A.shape[0]), int(A.shape[1])],
        "min": float(np.min(col_norms)) if col_norms.size else 0.0,
        "max": float(np.max(col_norms)) if col_norms.size else 0.0,
        "mean": float(np.mean(col_norms)) if col_norms.size else 0.0,
        "median": float(np.median(col_norms)) if col_norms.size else 0.0,
        "fro_norm": float(np.linalg.norm(A)),
        "num_near_zero": int(np.count_nonzero(~informative_mask)),
        "num_informative": int(np.count_nonzero(informative_mask)),
    }


def _assess_actuator_matrix(A):
    A = np.asarray(A, dtype=float)
    summary = _column_norm_summary(A)
    warnings: list[str] = []
    can_solve = True

    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0:
        warnings.append("actuator_torque_matrix_empty")
        can_solve = False
    elif not np.all(np.isfinite(A)):
        warnings.append("actuator_torque_matrix_non_finite")
        can_solve = False
    elif float(summary["fro_norm"]) <= float(ACTUATOR_MATRIX_FRO_NORM_EPS):
        warnings.append("actuator_torque_matrix_near_zero")
        can_solve = False
    elif int(summary["num_informative"]) <= 0:
        warnings.append("actuator_torque_matrix_uninformative")
        can_solve = False

    effective_rank = 0
    gram_condition = None
    if A.ndim == 2 and A.shape[0] > 0 and A.shape[1] > 0 and int(summary["num_informative"]) > 0:
        informative_cols = np.linalg.norm(A, axis=0) > float(ACTUATOR_MATRIX_COL_NORM_EPS)
        A_eff = A[:, informative_cols]
        if A_eff.size > 0:
            effective_rank = int(np.linalg.matrix_rank(A_eff, tol=float(ACTUATOR_MATRIX_COL_NORM_EPS)))
            if effective_rank <= 0:
                warnings.append("actuator_torque_matrix_rank_zero")
                can_solve = False
            elif effective_rank < min(A_eff.shape):
                warnings.append("actuator_torque_matrix_rank_deficient")
            gram = A_eff.T @ A_eff
            if gram.size > 0:
                try:
                    gram_condition = float(np.linalg.cond(gram))
                except np.linalg.LinAlgError:
                    gram_condition = math.inf
                if not np.isfinite(gram_condition):
                    warnings.append("actuator_torque_matrix_condition_non_finite")
                elif gram_condition > float(ACTUATOR_MATRIX_COND_WARN):
                    warnings.append("actuator_torque_matrix_ill_conditioned")

    return {
        "can_solve": bool(can_solve),
        "warnings": warnings,
        "column_norm_summary": summary,
        "effective_rank": int(effective_rank),
        "gram_condition": gram_condition,
    }


def build_actuator_torque_matrix_at_pose(model, data):
    """Build A such that A @ ctrl ≈ qfrc_actuator, at the current pose.

    Each column j is qfrc_actuator after a short activation rollout when only
    actuator j is driven with ctrl[j] = 1. This is necessary for muscle
    actuators whose force remains zero under a single static mj_forward call.
    """
    nv = model.nv
    nu = model.nu
    A = np.zeros((nv, nu), dtype=float)
    state0 = _capture_probe_state(model, data)
    m_raw, d_raw = _get_raw_model_data(model, data)

    try:
        for j in range(nu):
            data.qpos[:] = state0["qpos"]
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.ctrl[:] = 0.0
            if int(getattr(model, "na", 0)) > 0:
                data.act[:] = 0.0
            if hasattr(data, "time"):
                data.time = float(state0["time"])
            mujoco.mj_forward(m_raw, d_raw)
            for _ in range(int(ACTUATOR_AUDIT_STEPS)):
                data.ctrl[:] = 0.0
                data.ctrl[j] = 1.0
                mujoco.mj_step(m_raw, d_raw)

            A[:, j] = np.asarray(data.qfrc_actuator[:nv], dtype=float).copy()
        return A
    finally:
        _restore_probe_state(model, data, state0)
        mujoco.mj_forward(m_raw, d_raw)


def compute_required_torque(model, data):
    """Compute joint torques required to hold the current pose statically."""
    m_raw, d_raw = _get_raw_model_data(model, data)
    old_flags = int(m_raw.opt.disableflags)
    m_raw.opt.disableflags = old_flags | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    mujoco.mj_inverse(m_raw, d_raw)
    m_raw.opt.disableflags = old_flags
    return data.qfrc_inverse[: model.nv].copy()


def _solve_box_least_squares(A, b, *, reg=CTRL_QP_REG, u_prior=None, prior_weight=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = int(A.shape[1])
    if u_prior is None:
        u_prior_arr = np.zeros(n, dtype=float)
    else:
        u_prior_raw = np.asarray(u_prior, dtype=float).reshape(-1)
        u_prior_arr = np.zeros(n, dtype=float)
        u_prior_arr[: min(n, u_prior_raw.shape[0])] = u_prior_raw[: min(n, u_prior_raw.shape[0])]
    u_prior_arr = np.clip(np.where(np.isfinite(u_prior_arr), u_prior_arr, 0.0), 0.0, 1.0)
    prior_reg = float(reg if prior_weight is None else prior_weight)
    P = A.T @ A + float(prior_reg) * np.eye(n)
    q = -(A.T @ b + float(prior_reg) * u_prior_arr)
    x = u_prior_arr.copy()
    lipschitz = float(np.max(np.sum(np.abs(P), axis=1)))
    if not np.isfinite(lipschitz) or lipschitz <= 1e-12:
        lipschitz = 1.0
    alpha = 1.0 / lipschitz
    for _ in range(int(CTRL_QP_MAX_ITERS)):
        grad = P @ x + q
        x_new = np.clip(x - alpha * grad, 0.0, 1.0)
        if float(np.linalg.norm(x_new - x)) <= float(CTRL_QP_TOL):
            x = x_new
            break
        x = x_new
    return x


def validate_ctrl_forward(model, data, qpos_target, ctrl_est, n_steps=10):
    """Forward-validate an estimated ctrl from the target pose over a short rollout."""
    m_raw, _ = _get_raw_model_data(model, data)
    d_validate = mujoco.MjData(m_raw)
    q_target = np.zeros(int(m_raw.nq), dtype=float)
    q_in = np.asarray(qpos_target, dtype=float).reshape(-1)
    q_target[:min(len(q_target), len(q_in))] = q_in[:min(len(q_target), len(q_in))]
    ctrl_arr = np.clip(np.asarray(ctrl_est, dtype=float).reshape(-1), 0.0, 1.0)

    set_pose(m_raw, d_validate, q_target)
    d_validate.ctrl[:] = 0.0
    d_validate.ctrl[:min(int(m_raw.nu), len(ctrl_arr))] = ctrl_arr[:min(int(m_raw.nu), len(ctrl_arr))]
    _set_activation_from_ctrl(m_raw, d_validate, d_validate.ctrl)
    mujoco.mj_forward(m_raw, d_validate)

    pose_errors: list[float] = []
    for _ in range(max(0, int(n_steps))):
        d_validate.ctrl[:] = 0.0
        d_validate.ctrl[:min(int(m_raw.nu), len(ctrl_arr))] = ctrl_arr[:min(int(m_raw.nu), len(ctrl_arr))]
        mujoco.mj_step(m_raw, d_validate)
        pose_errors.append(float(np.linalg.norm(np.asarray(d_validate.qpos[: len(q_target)], dtype=float) - q_target)))

    return {
        "final_pose_error_norm": float(pose_errors[-1]) if pose_errors else 0.0,
        "mean_pose_error_norm": float(np.mean(pose_errors)) if pose_errors else 0.0,
        "final_qpos": np.asarray(d_validate.qpos, dtype=float).copy(),
    }


def _normalize_hold_exercise_name(exercise_name):
    label = str(exercise_name or "").strip().lower()
    if label in {"index_only", "index_bend", "index_flex"}:
        return "index_only"
    if label in {"making_o", "making-o", "make_o", "make-o"}:
        return "making_o"
    if label in {"claw", "claw_grasp", "claw_hold"}:
        return "claw"
    return label


def build_hold_prior(actuator_names, exercise_name, is_hold_mode):
    prior = np.zeros(len(actuator_names), dtype=float)
    if not bool(is_hold_mode):
        return prior
    hold_name = _normalize_hold_exercise_name(exercise_name)
    hold_weights = None
    if hold_name == "fist":
        hold_weights = {
            "FDS2": 0.18,
            "FDS3": 0.19,
            "FDS4": 0.19,
            "FDS5": 0.15,
            "FDP2": 0.16,
            "FDP3": 0.17,
            "FDP4": 0.17,
            "FDP5": 0.14,
            "FPL": 0.06,
        }
    elif hold_name == "index_only":
        hold_weights = {
            "FDS2": 0.26,
            "FDP2": 0.24,
            "EIP": 0.07,
            "EDC2": 0.05,
        }
    elif hold_name == "making_o":
        hold_weights = {
            "FDS2": 0.18,
            "FDP2": 0.17,
            "FPL": 0.12,
            "OP": 0.08,
            "EIP": 0.04,
            "EDC2": 0.03,
        }
    elif hold_name == "claw":
        hold_weights = {
            "FDP2": 0.17,
            "FDP3": 0.18,
            "FDP4": 0.18,
            "FDP5": 0.14,
            "FDS2": 0.10,
            "FDS3": 0.10,
            "FDS4": 0.10,
            "FDS5": 0.08,
        }
    if hold_weights is None:
        return prior
    for idx, actuator_name in enumerate(actuator_names):
        prior[idx] = float(hold_weights.get(str(actuator_name), 0.0))
    return np.clip(prior, 0.0, 1.0)


def _extract_hold_flexor_diagnostics(actuator_names, ctrl_vec):
    ctrl = np.asarray(ctrl_vec, dtype=float).reshape(-1)
    tracked_names = ("FDS2", "FDS3", "FDS4", "FDS5", "FDP2", "FDP3", "FDP4", "FDP5", "EIP", "EDC2")
    values = {}
    ranked: list[tuple[str, float]] = []
    for idx, actuator_name in enumerate(actuator_names):
        name = str(actuator_name)
        if name not in tracked_names or idx >= ctrl.shape[0]:
            continue
        value = float(ctrl[idx])
        values[name] = value
        ranked.append((name, value))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return {
        "hold_flexor_values": values,
        "hold_top_flexors": ranked[:4],
        "index_hold_values": {name: float(values.get(name, 0.0)) for name in ("FDS2", "FDP2", "EIP", "EDC2")},
    }


def estimate_ctrl_from_qpos(model, data, qpos_target, prev_ctrl=None, u_prior=None, exercise_name=None, actuator_names=None):
    """Estimate a ctrl vector that approximately realizes a given qpos.

    This solves a least-squares problem A @ ctrl ≈ tau_required at the given pose,
    where A is built from qfrc_actuator and tau_required from mj_inverse.

    Returns (ctrl_est, info_dict).
    """
    # Save original state
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()
    qacc_orig = data.qacc.copy()
    ctrl_orig = data.ctrl.copy()
    try:
        q_target_arr = np.asarray(qpos_target, dtype=float).reshape(-1)
        n_pose = min(len(q_target_arr), len(qpos_orig))
        pose_error_norm = float(np.linalg.norm(np.asarray(qpos_orig[:n_pose], dtype=float) - q_target_arr[:n_pose])) if n_pose > 0 else 0.0
        vel_norm = float(np.linalg.norm(np.asarray(qvel_orig[: model.nv], dtype=float)))
        is_hold_mode = bool(pose_error_norm <= float(HOLD_MODE_POSE_EPS) and vel_norm <= float(HOLD_MODE_VEL_EPS))
        actuator_names_arr = list(actuator_names) if actuator_names is not None else [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) or f"muscle_{idx}"
            for idx in range(int(model.nu))
        ]
        prev_ctrl_arr = None
        if prev_ctrl is not None:
            prev_ctrl_raw = np.asarray(prev_ctrl, dtype=float).reshape(-1)
            prev_ctrl_arr = np.full(int(model.nu), float(CTRL_BASELINE), dtype=float)
            prev_ctrl_arr[: min(int(model.nu), prev_ctrl_raw.shape[0])] = prev_ctrl_raw[: min(int(model.nu), prev_ctrl_raw.shape[0])]
            prev_ctrl_arr = np.clip(np.where(np.isfinite(prev_ctrl_arr), prev_ctrl_arr, float(CTRL_BASELINE)), 0.0, 1.0)
        hold_prior_label = _normalize_hold_exercise_name(exercise_name)
        hold_prior_arr = build_hold_prior(actuator_names_arr, hold_prior_label, is_hold_mode)
        hold_prior_active = bool(np.any(hold_prior_arr > 1e-8))
        fist_hold_prior_active = bool(hold_prior_active and hold_prior_label == "fist")
        index_hold_prior_active = bool(hold_prior_active and hold_prior_label == "index_only")
        if u_prior is None:
            prior_arr = np.maximum(np.full(int(model.nu), float(CTRL_BASELINE), dtype=float), hold_prior_arr)
            used_prior = hold_prior_label if hold_prior_active else "baseline"
        else:
            prior_raw = np.asarray(u_prior, dtype=float).reshape(-1)
            prior_arr = np.full(int(model.nu), float(CTRL_BASELINE), dtype=float)
            prior_arr[: min(int(model.nu), prior_raw.shape[0])] = prior_raw[: min(int(model.nu), prior_raw.shape[0])]
            prior_arr = np.clip(np.where(np.isfinite(prior_arr), prior_arr, float(CTRL_BASELINE)), 0.0, 1.0)
            prior_arr = np.maximum(prior_arr, hold_prior_arr)
            used_prior = "explicit"

        # 1) Set the pose and zero velocities/ctrl
        set_pose(model, data, qpos_target)

        # 2) Required torque to hold this pose
        tau_req = compute_required_torque(model, data)
        tau_required_norm = float(np.linalg.norm(tau_req))

        # 3) Actuator torque mapping at this pose
        A = build_actuator_torque_matrix_at_pose(model, data)

        # 4) Box-constrained regularized least-squares solution
        ctrl_new = _solve_box_least_squares(
            A,
            tau_req,
            u_prior=prior_arr,
            prior_weight=float(ESTIMATE_CTRL_HOLD_PRIOR_WEIGHT if hold_prior_active else CTRL_QP_REG),
        )
        used_prev_ctrl_blend = bool(is_hold_mode and prev_ctrl_arr is not None and prev_ctrl_arr.shape[0] == int(model.nu))
        if used_prev_ctrl_blend:
            prev_blend_alpha = float(ESTIMATE_CTRL_HOLD_BLEND_ALPHA)
            ctrl_est = prev_blend_alpha * prev_ctrl_arr + (1.0 - prev_blend_alpha) * ctrl_new
        else:
            ctrl_est = ctrl_new
        ctrl_est = np.clip(ctrl_est, 0.0, 1.0)
        ctrl_est[np.abs(ctrl_est) < float(CTRL_OUTPUT_DEADBAND)] = 0.0
        hold_diag = _extract_hold_flexor_diagnostics(actuator_names_arr, ctrl_est)

        # 5) Diagnostics
        tau_hat = A @ ctrl_est
        residual_vec = tau_hat - tau_req
        residual_norm = float(np.linalg.norm(residual_vec))

        info = {
            "residual_norm": residual_norm,
            "tau_required_norm": tau_required_norm,
            "ctrl_l1_norm": float(np.sum(np.abs(ctrl_est))),
            "ctrl_l2_norm": float(np.linalg.norm(ctrl_est)),
            "num_active_ctrl": int(np.count_nonzero(np.abs(ctrl_est) > float(CTRL_OUTPUT_DEADBAND))),
            "is_hold_mode": bool(is_hold_mode),
            "used_prev_ctrl_blend": bool(used_prev_ctrl_blend),
            "used_prior": used_prior,
            "hold_prior_label": hold_prior_label if hold_prior_active else "none",
            "hold_prior_active": bool(hold_prior_active),
            "fist_hold_prior_active": bool(fist_hold_prior_active),
            "index_hold_prior_active": bool(index_hold_prior_active),
            **hold_diag,
            "tau_required": tau_req,
            "tau_reconstructed": tau_hat,
        }
        return ctrl_est, info

    finally:
        # Restore original state
        data.qpos[:] = qpos_orig
        data.qvel[:] = qvel_orig
        data.qacc[:] = qacc_orig
        data.ctrl[:] = ctrl_orig
        m_raw, d_raw = _get_raw_model_data(model, data)
        mujoco.mj_forward(m_raw, d_raw)


def create_probe_env(env_id: str = DEFAULT_MYOSUITE_ENV_ID):
    """Create a MyoSuite env and return (env, model, data)."""
    env = gym.make(env_id)
    env.reset()
    try:
        env = env.unwrapped
    except Exception:
        pass
    sim = env.sim
    return env, sim.model, sim.data

# ---------------------------------------------------------------------------
# Realtime QP Controller
# ---------------------------------------------------------------------------
KP_GAIN = 50.0              # proportional gain for PD torque
KD_GAIN = 2.0               # derivative gain for PD torque
NNLS_REG = 1e-6             # regularization for NNLS QP
CTRL_EFFORT_WEIGHT = 0.025  # keep solutions compact instead of exciting many muscles uniformly
CTRL_BASELINE = 0.02        # retain a small tonic activation during hold
POSE_ERR_WEIGHT = 120.0
POSE_LOOKAHEAD_STEPS = 8
CTRL_CANDIDATE_BLEND = 0.10
TAUSMOOTH_OVERRIDE = 5.0
OSQP_REG = 1e-8
PGD_MAX_ITERS = 400
PGD_TOL = 1e-8
ACTUATION_AUDIT_STEPS = 20
CTRL_SMOOTH_WEIGHT = 0.08
TORQUE_WEIGHT_EXP = 0.5
CTRL_CANDIDATE_SHRINKS = ()
CTRL_CANDIDATE_BOOSTS = ()
CTRL_OUTPUT_DEADBAND = 1e-6
HOLD_TORQUE_BLEND = 0.6
TRACK_TORQUE_BLEND = 0.45
CTRL_EFFORT_SCORE_WEIGHT = 0.015
HOLD_MODE_POSE_EPS = 0.03
HOLD_MODE_VEL_EPS = 0.08
ESTIMATE_CTRL_HOLD_BLEND_ALPHA = 0.80
TARGET_FRAME_DT = 1.0 / 60.0
TARGET_VEL_BLEND = 0.45
TARGET_VEL_GAIN = 0.08
TARGET_VEL_CLIP = 12.0
TARGET_VEL_DEADBAND_NORM = 0.015
CTRL_GUIDANCE_WEIGHT = 0.06
CTRL_GUIDANCE_BLEND = 0.22
FAMILY_GUIDANCE_GAIN = 0.30
THUMB_TORQUE_WEIGHT_MULT = 1.50
THUMB_CMC_FLEX_GUIDANCE_GAIN = 0.90
THUMB_CMC_ABD_GUIDANCE_GAIN = 0.55
THUMB_MCP_GUIDANCE_GAIN = 0.75
THUMB_IP_GUIDANCE_GAIN = 0.90
THUMB_FLEXOR_BOOST_GAIN = 1.05
THUMB_OPPOSITION_BOOST_GAIN = 0.95
THUMB_ABDUCTOR_BOOST_GAIN = 0.55
THUMB_EXTENSOR_SUPPRESS_GAIN = 0.60
INDEX_PIP_GUIDANCE_GAIN = 0.95
INDEX_DIP_GUIDANCE_GAIN = 0.45
INDEX_FLEXOR_BOOST_GAIN = 1.05
INDEX_EXTENSOR_SUPPRESS_GAIN = 0.60
LITTLE_PIP_GUIDANCE_GAIN = 1.00
LITTLE_DIP_GUIDANCE_GAIN = 0.50
LITTLE_FLEXOR_BOOST_GAIN = 0.45
LITTLE_EXTENSOR_SUPPRESS_GAIN = 0.60
THUMB_CTRL_FLEXOR_PUSH_GAIN = 0.50
THUMB_CTRL_OPPOSITION_PUSH_GAIN = 0.42
THUMB_CTRL_ABDUCTOR_PUSH_GAIN = 0.24
THUMB_CTRL_EXTENSOR_DAMP_GAIN = 0.38
INDEX_CTRL_FLEXOR_PUSH_GAIN = 0.45
INDEX_CTRL_EXTENSOR_DAMP_GAIN = 0.35
LITTLE_CTRL_FLEXOR_PUSH_GAIN = 0.20
FIST_HOLD_NNLS_PRIOR_WEIGHT = 0.10
LITTLE_CTRL_EXTENSOR_DAMP_GAIN = 0.40
NOMINAL_TRACK_POSE_ERR_NORM = 0.75
NOMINAL_TRACK_WEIGHT_MIN = 0.35
NOMINAL_TRACK_WEIGHT_MAX = 0.62
BLENDED_CTRL_POSE_ERR_NORM = 0.45
PREV_CTRL_POSE_ERR_NORM = 0.15
POSE_SCORE_SMOOTH_MIN_SCALE = 0.20
POSE_SCORE_SMOOTH_ERR_GAIN = 2.0

CONTROLLER_NAME = "pd_muscle_controller"
CONTROLLER_DESCRIPTION = "PD torque mapped to muscle excitation with activation-aware dynamic actuation"


# ===================================================================
#  Low-level helpers (same as myo_ctrl_qp_track.py)
# ===================================================================

def _get_raw(model, data):
    m = getattr(model, "ptr", None) or getattr(model, "_model", None) or model
    d = getattr(data, "ptr", None) or getattr(data, "_data", None) or data
    return m, d


def _solve_box_qp(P, q, lb, ub, x0, reg=OSQP_REG):
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float).ravel()
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    x0 = np.asarray(x0, dtype=float).ravel()
    n = P.shape[0]
    P = P + reg * np.eye(n)

    if HAVE_OSQP:
        P_sp = spa.csc_matrix(P)
        A = spa.eye(n, format="csc")
        m = osqp.OSQP()
        m.setup(P=P_sp, q=q, A=A, l=lb, u=ub, verbose=False,
                max_iter=20000, eps_abs=1e-8, eps_rel=1e-8,
                polish=True, scaling=True)
        try:
            m.warm_start(x=x0)
        except Exception:
            pass
        res = m.solve()
        x = np.asarray(res.x, dtype=float)
        if x.shape[0] == n and np.all(np.isfinite(x)):
            return np.clip(x, lb, ub)

    # Fallback: projected gradient descent
    x = np.clip(x0, lb, ub)
    L = float(np.max(np.sum(np.abs(P), axis=1)))
    if not np.isfinite(L) or L <= 1e-12:
        L = 1.0
    alpha = 1.0 / L
    for _ in range(int(PGD_MAX_ITERS)):
        grad = P @ x + q
        x_new = np.clip(x - alpha * grad, lb, ub)
        if float(np.linalg.norm(x_new - x)) <= float(PGD_TOL):
            x = x_new
            break
        x = x_new
    return x


def _compute_tau(model, data, act_type="rise"):
    tau = np.zeros(model.nu, dtype=float)
    for i in range(model.nu):
        actnum = int(model.actuator_actnum[i])
        if actnum > 0:
            a = int(model.actuator_actadr[i])
            act_i = float(data.act[a])
            if act_type == "rise":
                tau[i] = float(model.actuator_dynprm[i, 0]) * (0.5 + 1.5 * act_i)
            else:
                tau[i] = float(model.actuator_dynprm[i, 1]) / max(1e-8, 0.5 + 1.5 * act_i)
    return tau


def _map_act(model, data):
    act = np.zeros(model.nu, dtype=float)
    for i in range(model.nu):
        if int(model.actuator_actnum[i]) > 0:
            act[i] = float(data.act[int(model.actuator_actadr[i])])
    return act


def _dense_moment(model, data):
    AM = np.zeros((model.nu, model.nv), dtype=float)
    mujoco.mju_sparse2dense(
        AM,
        data.actuator_moment.reshape(-1),
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind.reshape(-1),
    )
    return AM.T  # (nv, nu)


def _gain_bias(model, data_s1):
    gain = np.zeros(model.nu, dtype=float)
    bias = np.zeros(model.nu, dtype=float)
    for i in range(model.nu):
        length = float(data_s1.actuator_length[i])
        velocity = float(data_s1.actuator_velocity[i])
        lr = model.actuator_lengthrange[i]
        acc0 = float(model.actuator_acc0[i])
        prmb = model.actuator_biasprm[i, :9]
        prmg = model.actuator_gainprm[i, :9]
        try:
            bias[i] = float(mujoco.mju_muscleBias(length, lr, acc0, prmb))
            gain[i] = float(min(-1.0, mujoco.mju_muscleGain(length, velocity, lr, acc0, prmg)))
        except Exception:
            bias[i] = 0.0
            gain[i] = -1.0
    return gain, bias


def _clip_to_limits(model, qpos):
    q = qpos.copy()
    for j in range(int(model.njnt)):
        if int(model.jnt_limited[j]) == 0:
            continue
        jt = int(model.jnt_type[j])
        if jt not in (2, 3):
            continue
        adr = int(model.jnt_qposadr[j])
        lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
        if np.isfinite(lo) and np.isfinite(hi):
            q[adr] = float(np.clip(q[adr], lo, hi))
    return q


def _torque_weights(model):
    w = np.ones(int(model.nv), dtype=float)
    rng = np.asarray(model.jnt_range, dtype=float)
    for j in range(int(model.njnt)):
        if int(model.jnt_type[j]) not in (2, 3):
            continue
        qadr = int(model.jnt_qposadr[j])
        if qadr >= len(w):
            continue
        lo = float(rng[j, 0])
        hi = float(rng[j, 1])
        span = float(abs(hi - lo)) if np.isfinite(lo) and np.isfinite(hi) else 1.0
        span = max(span, 1e-3)
        w[qadr] = 1.0 / (span ** float(TORQUE_WEIGHT_EXP))
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if str(joint_name).startswith(("cmc_", "mp_", "ip_")):
            w[qadr] *= float(THUMB_TORQUE_WEIGHT_MULT)
    return w


def _compute_hold_torque(model, data):
    old_flags = int(model.opt.disableflags)
    try:
        model.opt.disableflags = old_flags | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        mujoco.mj_inverse(model, data)
        return np.asarray(data.qfrc_inverse[: model.nv], dtype=float).copy()
    finally:
        model.opt.disableflags = old_flags


def _joint_family(joint_name):
    name = str(joint_name)
    if name in ("pro_sup", "deviation", "flexion"):
        return "wrist"
    if name.startswith(("cmc_", "mp_", "ip_")):
        return "thumb"
    for digit, family in (("2", "index"), ("3", "middle"), ("4", "ring"), ("5", "little")):
        if digit in name:
            return family
    return "shared"


def _muscle_family(muscle_name):
    name = str(muscle_name)
    if name in {"ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"}:
        return "wrist"
    if name in {"EPL", "EPB", "FPL", "APL", "OP"}:
        return "thumb"
    for digit, family in (("2", "index"), ("3", "middle"), ("4", "ring"), ("5", "little")):
        if digit in name:
            return family
    return "shared"


def _normalize_nonnegative(vec):
    vec = np.asarray(vec, dtype=float).reshape(-1)
    vec = np.maximum(vec, 0.0)
    vmax = float(np.max(vec)) if vec.size else 0.0
    if vmax <= 1e-12 or not np.isfinite(vmax):
        return np.zeros_like(vec)
    return vec / vmax


def _positive_motion_demand(q_err, target_qvel, idx):
    if idx >= len(q_err):
        return 0.0
    err = max(0.0, float(q_err[idx]))
    vel = max(0.0, float(target_qvel[idx])) if idx < len(target_qvel) else 0.0
    return err + (0.20 * vel)


def _absolute_motion_demand(q_err, target_qvel, idx):
    if idx >= len(q_err):
        return 0.0
    err = abs(float(q_err[idx]))
    vel = abs(float(target_qvel[idx])) if idx < len(target_qvel) else 0.0
    return err + (0.15 * vel)


def _directed_motion_demand(q_err, target_qvel, idx, direction_sign):
    if idx >= len(q_err):
        return 0.0
    sign = 1.0 if float(direction_sign) >= 0.0 else -1.0
    err = max(0.0, sign * float(q_err[idx]))
    vel = 0.0
    if idx < len(target_qvel):
        vel = max(0.0, sign * float(target_qvel[idx]))
    return err + (0.20 * vel)


def _sanitize_ctrl(ctrl, lb=CTRL_BASELINE, ub=1.0, deadband=CTRL_OUTPUT_DEADBAND):
    ctrl = np.asarray(ctrl, dtype=float).reshape(-1)
    ctrl = np.where(np.isfinite(ctrl), ctrl, 0.0)
    ctrl = np.clip(ctrl, float(lb), float(ub))
    ctrl[np.abs(ctrl) < float(deadband)] = 0.0
    ctrl[np.abs(ctrl - float(ub)) < float(deadband)] = float(ub)
    return ctrl


# ===================================================================
#  Detect which DOFs are muscle-actuated
# ===================================================================

def _find_actuated_dofs(model):
    """Return boolean mask (nv,): True if DOF is affected by any muscle actuator."""
    d_tmp = mujoco.MjData(model)
    mask = np.zeros(model.nv, dtype=bool)
    for j in range(model.nu):
        d_tmp.qpos[:] = 0.0
        d_tmp.qvel[:] = 0.0
        d_tmp.ctrl[:] = 0.0
        try:
            if int(getattr(model, "na", 0)) > 0:
                d_tmp.act[:] = 0.0
        except Exception:
            pass
        mujoco.mj_forward(model, d_tmp)
        for _ in range(int(ACTUATION_AUDIT_STEPS)):
            d_tmp.ctrl[:] = 0.0
            d_tmp.ctrl[j] = 1.0
            mujoco.mj_step(model, d_tmp)
        mask |= (np.abs(d_tmp.qfrc_actuator[:model.nv]) > 1e-6)
    return mask


# ===================================================================
#  Stateful controller for real-time use
# ===================================================================

class RealtimeQPController:
    """Stateful realtime muscle controller with a private simulation state."""

    def __init__(self, model_wrapper, data_wrapper):
        """
        model_wrapper, data_wrapper: MyoSuite wrapper objects (or raw mujoco).
        """
        self.model_w = model_wrapper
        self.data_w = data_wrapper
        self.model, self.data_orig = _get_raw(model_wrapper, data_wrapper)

        # Detect which joints have muscle actuation
        self.actuated_mask = _find_actuated_dofs(self.model)  # bool[nv]
        self.actuated_dofs = np.where(self.actuated_mask)[0]
        self.unactuated_dofs = np.where(~self.actuated_mask)[0]
        n_act = int(self.actuated_mask.sum())
        print(f"[CTRL] Actuated DOFs: {n_act}/{self.model.nv}  "
              f"indices={self.actuated_dofs.tolist()}")

        # Private simulation copy for ctrl estimation
        self._data = mujoco.MjData(self.model)
        self._data.qpos[:] = self.data_orig.qpos
        self._data.qvel[:] = 0.0
        self._data.qacc[:] = 0.0
        self._data.ctrl[:] = 0.0
        try:
            if int(getattr(self.model, "na", 0)) > 0:
                self._data.act[:] = self.data_orig.act
        except Exception:
            pass
        mujoco.mj_forward(self.model, self._data)

        # Override tausmooth for better optimization conditioning
        if TAUSMOOTH_OVERRIDE is not None:
            try:
                self.model.actuator_dynprm[:, 2] = float(TAUSMOOTH_OVERRIDE)
            except Exception:
                pass

        self._prev_qpos = self._data.qpos.copy()
        self._last_target = self._data.qpos.copy()
        self._last_ctrl = np.zeros(int(self.model.nu), dtype=float)
        self._last_tau_des = np.zeros(int(self.model.nv), dtype=float)
        self._last_pose_error = np.zeros(int(self.model.nq), dtype=float)
        self._last_pose_error_norm = 0.0
        self._last_target_qvel = np.zeros(int(self.model.nv), dtype=float)
        self._tau_weights = _torque_weights(self.model)
        self._target_eval_data = mujoco.MjData(self.model)
        self._exercise_name: str | None = None
        self._actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) or f"muscle_{idx}"
            for idx in range(int(self.model.nu))
        ]
        self._joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, idx) or f"joint_{idx}"
            for idx in range(int(self.model.nv))
        ]
        self._last_hold_prior_label = "none"
        self._last_is_hold_mode = False
        self._last_fist_hold_prior_active = False
        self._last_index_hold_prior_active = False
        self._last_hold_flexor_values: dict[str, float] = {}
        self._last_hold_top_flexors: list[tuple[str, float]] = []
        self._last_index_hold_values: dict[str, float] = {}

    @property
    def name(self):
        return CONTROLLER_NAME

    @property
    def description(self):
        return CONTROLLER_DESCRIPTION

    def set_exercise_name(self, name):
        self._exercise_name = None if name is None else str(name)

    def get_info(self):
        return {
            "name": self.name,
            "description": self.description,
            "actuated_mask": self.actuated_mask.copy(),
            "actuated_dofs": self.actuated_dofs.copy(),
            "unactuated_dofs": self.unactuated_dofs.copy(),
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
        }

    def get_state(self):
        return {
            "qpos": np.asarray(self._data.qpos, dtype=float).copy(),
            "qvel": np.asarray(self._data.qvel, dtype=float).copy(),
            "ctrl": np.asarray(self._data.ctrl, dtype=float).copy(),
            "target_qpos": np.asarray(self._last_target, dtype=float).copy(),
            "tau_des": np.asarray(self._last_tau_des, dtype=float).copy(),
            "pose_error": np.asarray(self._last_pose_error, dtype=float).copy(),
            "pose_error_norm": float(self._last_pose_error_norm),
            "hold_prior_label": str(self._last_hold_prior_label),
            "is_hold_mode": bool(self._last_is_hold_mode),
            "fist_hold_prior_active": bool(self._last_fist_hold_prior_active),
            "index_hold_prior_active": bool(self._last_index_hold_prior_active),
            "hold_flexor_values": dict(self._last_hold_flexor_values),
            "hold_top_flexors": list(self._last_hold_top_flexors),
            "index_hold_values": dict(self._last_index_hold_values),
        }

    def step(self, qpos_target, sync_data=None):
        ctrl = self.estimate_ctrl(qpos_target, sync_data=sync_data)
        state = self.get_state()
        state["ctrl"] = _sanitize_ctrl(ctrl)
        return state

    def sync_from_data(self, data_source=None):
        """Copy an external MuJoCo/MyoSuite state into the private estimator state."""
        if data_source is None:
            data_source = self.data_orig

        _, data_src = _get_raw(self.model_w, data_source)
        self._data.qpos[:] = np.asarray(data_src.qpos, dtype=float).reshape(-1)
        self._data.qvel[:] = np.asarray(data_src.qvel, dtype=float).reshape(-1)
        self._data.qacc[:] = np.asarray(data_src.qacc, dtype=float).reshape(-1)
        self._data.ctrl[:] = np.asarray(data_src.ctrl, dtype=float).reshape(-1)
        try:
            if int(getattr(self.model, "na", 0)) > 0:
                self._data.act[:] = np.asarray(data_src.act, dtype=float).reshape(-1)
        except Exception:
            pass
        mujoco.mj_forward(self.model, self._data)

    # ---------------------------------------------------------------
    def estimate_ctrl(self, qpos_target, sync_data=None):
        """
        Given a desired qpos (from Leap), return ctrl ∈ [0,1]^nu.

        Simple PD-torque approach:
          1) Compute desired joint torque: τ = Kp*(q_target - q) - Kd*qdot
          2) Get actuator moment matrix AM (nv × nu) at current state
          3) Solve for ctrl: min ||AM @ (gain*ctrl + bias) - τ||²  s.t. 0 ≤ ctrl ≤ 1
          4) Step the internal sim with that ctrl

        All dynamically actuated joints participate in the muscle optimization.
        """
        if sync_data is not None:
            self.sync_from_data(sync_data)

        model = self.model
        data = self._data

        qpos_target = np.asarray(qpos_target, dtype=float).ravel()
        full_q = np.zeros(model.nq, dtype=float)
        n = min(len(qpos_target), model.nq)
        full_q[:n] = qpos_target[:n]
        full_q = _clip_to_limits(model, full_q)
        target_delta = full_q - self._last_target
        target_delta_norm = float(np.linalg.norm(target_delta))
        if target_delta_norm < float(TARGET_VEL_DEADBAND_NORM):
            raw_target_qvel = np.zeros(int(model.nv), dtype=float)
        else:
            raw_target_qvel = target_delta / float(TARGET_FRAME_DT)
        target_qvel = (1.0 - float(TARGET_VEL_BLEND)) * np.asarray(self._last_target_qvel, dtype=float) + float(TARGET_VEL_BLEND) * raw_target_qvel
        target_qvel = np.clip(target_qvel, -float(TARGET_VEL_CLIP), float(TARGET_VEL_CLIP))
        self._last_target = full_q.copy()
        self._last_target_qvel = np.asarray(target_qvel, dtype=float).copy()

        # --- Step 1: PD desired torque ---
        q_err = full_q - data.qpos
        tau_track = KP_GAIN * q_err + float(TARGET_VEL_GAIN) * target_qvel - KD_GAIN * data.qvel

        target_eval = self._target_eval_data
        target_eval.qpos[:] = full_q
        target_eval.qvel[:] = 0.0
        target_eval.qacc[:] = 0.0
        target_eval.ctrl[:] = 0.0
        try:
            if int(getattr(self.model, "na", 0)) > 0:
                target_eval.act[:] = np.asarray(data.act, dtype=float).reshape(-1)
        except Exception:
            pass
        mujoco.mj_forward(model, target_eval)
        mujoco.mj_step1(model, target_eval)
        gain_target, bias_target = _gain_bias(model, target_eval)
        AM_target = _dense_moment(model, target_eval)
        tau_hold = _compute_hold_torque(model, target_eval)

        mujoco.mj_step1(model, data)
        gain_current, bias_current = _gain_bias(model, data)
        AM_current = _dense_moment(model, data)

        tau_des = float(HOLD_TORQUE_BLEND) * tau_hold + float(TRACK_TORQUE_BLEND) * tau_track

        A_target = AM_target * gain_target[np.newaxis, :]
        guidance_target = self._build_ctrl_guidance(A_target, tau_des, q_err, target_qvel)
        b_target = tau_des - AM_target @ bias_target
        hold_prior = build_hold_prior(self._actuator_names, self._exercise_name, False)
        ctrl_hold = self._nnls_ctrl(A_target, b_target, guidance_target=guidance_target, u_prior=hold_prior, prior_weight=0.0)
        ctrl_hold = self._apply_edge_flexor_emphasis(ctrl_hold, q_err, target_qvel)

        A_current = AM_current * gain_current[np.newaxis, :]
        b_current = tau_des - AM_current @ bias_current
        ctrl_track = self._nnls_ctrl(A_current, b_current, guidance_target=guidance_target, u_prior=hold_prior, prior_weight=0.0)
        ctrl_track = self._apply_edge_flexor_emphasis(ctrl_track, q_err, target_qvel)

        pose_gap = q_err[self.actuated_mask] if len(q_err) == len(self.actuated_mask) else q_err
        pose_gap_norm = float(np.linalg.norm(pose_gap))
        track_mix = float(np.clip(pose_gap_norm / float(NOMINAL_TRACK_POSE_ERR_NORM), 0.0, 1.0))
        track_weight = float(
            NOMINAL_TRACK_WEIGHT_MIN
            + (NOMINAL_TRACK_WEIGHT_MAX - NOMINAL_TRACK_WEIGHT_MIN) * track_mix
        )
        hold_weight = float(np.clip(1.0 - track_weight, 0.0, 1.0))
        ctrl_nominal = _sanitize_ctrl(
            hold_weight * np.asarray(ctrl_hold, dtype=float)
            + track_weight * np.asarray(ctrl_track, dtype=float)
        )
        ctrl_nominal = _sanitize_ctrl(
            (1.0 - float(CTRL_GUIDANCE_BLEND)) * np.asarray(ctrl_nominal, dtype=float)
            + float(CTRL_GUIDANCE_BLEND) * np.asarray(guidance_target, dtype=float)
        )
        ctrl_nominal = self._apply_edge_flexor_emphasis(ctrl_nominal, q_err, target_qvel)
        ctrl_blended = np.clip(
            (1.0 - float(CTRL_CANDIDATE_BLEND)) * ctrl_nominal
            + float(CTRL_CANDIDATE_BLEND) * np.asarray(self._last_ctrl, dtype=float),
            float(CTRL_BASELINE),
            1.0,
        )
        ctrl_blended = self._apply_edge_flexor_emphasis(ctrl_blended, q_err, target_qvel)
        ctrl_mid = _sanitize_ctrl(0.5 * np.asarray(ctrl_hold, dtype=float) + 0.5 * np.asarray(ctrl_track, dtype=float))
        ctrl_mid = self._apply_edge_flexor_emphasis(ctrl_mid, q_err, target_qvel)
        ctrl_nominal = _sanitize_ctrl(ctrl_nominal)
        ctrl_blended = _sanitize_ctrl(ctrl_blended)
        vel_norm = float(np.linalg.norm(np.asarray(data.qvel, dtype=float).reshape(-1)))
        is_hold_mode = bool(pose_gap_norm <= float(HOLD_MODE_POSE_EPS) and vel_norm <= float(HOLD_MODE_VEL_EPS))
        hold_prior_label = _normalize_hold_exercise_name(self._exercise_name)
        hold_prior = build_hold_prior(self._actuator_names, hold_prior_label, is_hold_mode)
        hold_prior_active = bool(np.any(hold_prior > 1e-8))
        fist_hold_prior_active = bool(hold_prior_active and hold_prior_label == "fist")
        index_hold_prior_active = bool(hold_prior_active and hold_prior_label == "index_only")
        if hold_prior_active:
            ctrl_hold = self._nnls_ctrl(
                A_target,
                b_target,
                guidance_target=guidance_target,
                u_prior=hold_prior,
                prior_weight=float(FIST_HOLD_NNLS_PRIOR_WEIGHT),
            )
            ctrl_hold = self._apply_edge_flexor_emphasis(ctrl_hold, q_err, target_qvel)
            ctrl_track = self._nnls_ctrl(
                A_current,
                b_current,
                guidance_target=guidance_target,
                u_prior=hold_prior,
                prior_weight=float(FIST_HOLD_NNLS_PRIOR_WEIGHT),
            )
            ctrl_track = self._apply_edge_flexor_emphasis(ctrl_track, q_err, target_qvel)
            ctrl_nominal = _sanitize_ctrl(
                hold_weight * np.asarray(ctrl_hold, dtype=float)
                + track_weight * np.asarray(ctrl_track, dtype=float)
            )
            ctrl_nominal = _sanitize_ctrl(
                (1.0 - float(CTRL_GUIDANCE_BLEND)) * np.asarray(ctrl_nominal, dtype=float)
                + float(CTRL_GUIDANCE_BLEND) * np.asarray(guidance_target, dtype=float)
            )
            ctrl_nominal = self._apply_edge_flexor_emphasis(ctrl_nominal, q_err, target_qvel)
            ctrl_blended = np.clip(
                (1.0 - float(CTRL_CANDIDATE_BLEND)) * ctrl_nominal
                + float(CTRL_CANDIDATE_BLEND) * np.asarray(self._last_ctrl, dtype=float),
                float(CTRL_BASELINE),
                1.0,
            )
            ctrl_blended = self._apply_edge_flexor_emphasis(ctrl_blended, q_err, target_qvel)
            ctrl_mid = _sanitize_ctrl(0.5 * np.asarray(ctrl_hold, dtype=float) + 0.5 * np.asarray(ctrl_track, dtype=float))
            ctrl_mid = self._apply_edge_flexor_emphasis(ctrl_mid, q_err, target_qvel)
            ctrl_nominal = _sanitize_ctrl(ctrl_nominal)
            ctrl_blended = _sanitize_ctrl(ctrl_blended)
        if is_hold_mode:
            ctrl = ctrl_nominal
        else:
            ctrl = self._select_pose_best_ctrl(
                ctrl_nominal,
                ctrl_blended,
                ctrl_hold,
                ctrl_track,
                ctrl_mid,
                full_q,
                pose_gap_norm,
            )
        ctrl = _sanitize_ctrl(ctrl)
        self._last_tau_des = np.asarray(tau_des, dtype=float).copy()
        self._last_ctrl = _sanitize_ctrl(ctrl)
        hold_diag = _extract_hold_flexor_diagnostics(self._actuator_names, ctrl)
        self._last_hold_prior_label = hold_prior_label if hold_prior_active else "none"
        self._last_is_hold_mode = bool(is_hold_mode)
        self._last_fist_hold_prior_active = bool(fist_hold_prior_active)
        self._last_index_hold_prior_active = bool(index_hold_prior_active)
        self._last_hold_flexor_values = dict(hold_diag["hold_flexor_values"])
        self._last_hold_top_flexors = list(hold_diag["hold_top_flexors"])
        self._last_index_hold_values = dict(hold_diag["index_hold_values"])

        # --- Step 4: apply ctrl and step ---
        data.ctrl[:] = _sanitize_ctrl(ctrl)
        mujoco.mj_step(model, data)

        pose_err = np.asarray(full_q - data.qpos, dtype=float).copy()
        self._last_pose_error = pose_err
        self._last_pose_error_norm = float(np.linalg.norm(pose_err[self.actuated_mask])) if len(pose_err) == len(self.actuated_mask) else float(np.linalg.norm(pose_err))
        self._prev_qpos = data.qpos.copy()
        return _sanitize_ctrl(ctrl)

    def simulate_to_target(self, qpos_target, n_steps):
        ctrl = np.zeros(int(self.model.nu), dtype=float)
        for _ in range(int(n_steps)):
            ctrl = self.estimate_ctrl(qpos_target)
        state = self.get_state()
        state["ctrl"] = np.asarray(ctrl, dtype=float).copy()
        return state

    # ---------------------------------------------------------------
    def _nnls_ctrl(self, A, b, guidance_target=None, u_prior=None, prior_weight=0.0):
        """Solve min ||A@x - b||² + λ||x - baseline||²  s.t. 0 ≤ x ≤ 1.

        The spread term encourages ALL muscles to contribute rather than
        producing a sparse solution where many muscles are exactly zero.
        """
        nu = A.shape[1]
        W = self._tau_weights[:A.shape[0]]
        A = np.asarray(A, dtype=float) * W[:, np.newaxis]
        b = np.asarray(b, dtype=float) * W
        lam = float(CTRL_EFFORT_WEIGHT)
        guidance_weight = float(CTRL_GUIDANCE_WEIGHT)
        smooth = float(CTRL_SMOOTH_WEIGHT)
        baseline = np.full(nu, float(CTRL_BASELINE))
        prev = np.clip(np.asarray(self._last_ctrl, dtype=float).reshape(-1), 0.0, 1.0)
        guidance = np.zeros(nu, dtype=float) if guidance_target is None else np.clip(np.asarray(guidance_target, dtype=float).reshape(-1), 0.0, 1.0)
        prior = baseline.copy() if u_prior is None else np.clip(np.asarray(u_prior, dtype=float).reshape(-1), 0.0, 1.0)
        if prior.shape[0] != nu:
            prior_fixed = baseline.copy()
            prior_fixed[: min(nu, prior.shape[0])] = prior[: min(nu, prior.shape[0])]
            prior = prior_fixed
        prior_reg = max(0.0, float(prior_weight))
        # P = 2*(A^T A + (λ+μ) I),  q = -2*(A^T b + λ baseline + μ prev)
        P = 2.0 * (A.T @ A) + (NNLS_REG + 2.0 * lam + 2.0 * smooth + 2.0 * guidance_weight + 2.0 * prior_reg) * np.eye(nu)
        q = -2.0 * (A.T @ b) - 2.0 * lam * baseline - 2.0 * smooth * prev - 2.0 * guidance_weight * guidance - 2.0 * prior_reg * prior
        lb = np.full(nu, float(CTRL_BASELINE))
        ub = np.ones(nu)
        x0 = np.clip(0.5 * np.asarray(self._data.ctrl, dtype=float).copy() + 0.5 * prior, 0, 1)
        return _sanitize_ctrl(_solve_box_qp(P, q, lb, ub, x0))

    def _digit_curl_demand(self, q_err, target_qvel, digit_name, *, pip_gain, dip_gain, mcp_gain=0.10):
        base_idx = int(FINGER_BASE_IDX[str(digit_name)])
        return float(
            np.clip(
                float(pip_gain) * _positive_motion_demand(q_err, target_qvel, base_idx + 2)
                + float(dip_gain) * _positive_motion_demand(q_err, target_qvel, base_idx + 3)
                + float(mcp_gain) * _positive_motion_demand(q_err, target_qvel, base_idx),
                0.0,
                1.5,
            )
        )

    def _thumb_motion_demand(self, q_err, target_qvel):
        flex_demand = float(
            np.clip(
                float(THUMB_CMC_FLEX_GUIDANCE_GAIN) * _directed_motion_demand(q_err, target_qvel, THUMB_CMC_FLEX_IDX, +1.0)
                + float(THUMB_MCP_GUIDANCE_GAIN) * _directed_motion_demand(q_err, target_qvel, THUMB_MCP_IDX, -1.0)
                + float(THUMB_IP_GUIDANCE_GAIN) * _directed_motion_demand(q_err, target_qvel, THUMB_IP_IDX, -1.0),
                0.0,
                1.5,
            )
        )
        opposition_demand = float(
            np.clip(
                0.60 * float(THUMB_CMC_FLEX_GUIDANCE_GAIN) * _directed_motion_demand(q_err, target_qvel, THUMB_CMC_FLEX_IDX, +1.0)
                + float(THUMB_CMC_ABD_GUIDANCE_GAIN) * _directed_motion_demand(q_err, target_qvel, THUMB_CMC_ABD_IDX, -1.0),
                0.0,
                1.5,
            )
        )
        return flex_demand, opposition_demand

    def _apply_edge_flexor_emphasis(self, ctrl_vec, q_err, target_qvel):
        ctrl = np.clip(np.asarray(ctrl_vec, dtype=float).reshape(-1), 0.0, 1.0).copy()
        thumb_flex_demand, thumb_opposition_demand = self._thumb_motion_demand(q_err, target_qvel)
        index_curl_demand = self._digit_curl_demand(
            q_err,
            target_qvel,
            "index",
            pip_gain=INDEX_PIP_GUIDANCE_GAIN,
            dip_gain=INDEX_DIP_GUIDANCE_GAIN,
        )
        little_curl_demand = self._digit_curl_demand(
            q_err,
            target_qvel,
            "little",
            pip_gain=LITTLE_PIP_GUIDANCE_GAIN,
            dip_gain=LITTLE_DIP_GUIDANCE_GAIN,
        )
        for idx, muscle_name in enumerate(self._actuator_names):
            name = str(muscle_name)
            if thumb_flex_demand > 1e-6 or thumb_opposition_demand > 1e-6:
                if name == "FPL":
                    ctrl[idx] *= 1.0 + float(THUMB_CTRL_FLEXOR_PUSH_GAIN) * thumb_flex_demand
                elif name == "OP":
                    ctrl[idx] *= 1.0 + float(THUMB_CTRL_OPPOSITION_PUSH_GAIN) * thumb_opposition_demand
                elif name == "APL":
                    ctrl[idx] *= 1.0 + float(THUMB_CTRL_ABDUCTOR_PUSH_GAIN) * thumb_opposition_demand
                elif name in {"EPL", "EPB"}:
                    ctrl[idx] *= max(0.0, 1.0 - float(THUMB_CTRL_EXTENSOR_DAMP_GAIN) * thumb_flex_demand)
            if index_curl_demand > 1e-6:
                if name in {"FDS2", "FDP2"}:
                    ctrl[idx] *= 1.0 + float(INDEX_CTRL_FLEXOR_PUSH_GAIN) * index_curl_demand
                elif name in {"EDC2", "EIP"}:
                    ctrl[idx] *= max(0.0, 1.0 - float(INDEX_CTRL_EXTENSOR_DAMP_GAIN) * index_curl_demand)
            if little_curl_demand > 1e-6:
                if name in {"FDS5", "FDP5"}:
                    ctrl[idx] *= 1.0 + float(LITTLE_CTRL_FLEXOR_PUSH_GAIN) * little_curl_demand
                elif name in {"EDC5", "EDM"}:
                    ctrl[idx] *= max(0.0, 1.0 - float(LITTLE_CTRL_EXTENSOR_DAMP_GAIN) * little_curl_demand)
        return _sanitize_ctrl(ctrl)

    def _build_ctrl_guidance(self, A, tau_des, q_err, target_qvel):
        A = np.asarray(A, dtype=float)
        tau_des = np.asarray(tau_des, dtype=float).reshape(-1)
        q_err = np.asarray(q_err, dtype=float).reshape(-1)
        target_qvel = np.asarray(target_qvel, dtype=float).reshape(-1)

        torque_alignment = np.sum(np.maximum(0.0, A * tau_des[:, np.newaxis]), axis=0)
        torque_alignment = _normalize_nonnegative(torque_alignment)

        joint_motion = np.abs(q_err) + 0.25 * np.abs(target_qvel[: len(q_err)])
        family_demand = {}
        for idx, joint_name in enumerate(self._joint_names[: len(joint_motion)]):
            family = _joint_family(joint_name)
            family_demand[family] = family_demand.get(family, 0.0) + float(joint_motion[idx])
        family_max = max(family_demand.values()) if family_demand else 0.0
        if family_max > 1e-12:
            family_demand = {key: float(value) / float(family_max) for key, value in family_demand.items()}

        family_prior = np.zeros(int(self.model.nu), dtype=float)
        for idx, muscle_name in enumerate(self._actuator_names):
            family_prior[idx] = family_demand.get(_muscle_family(muscle_name), 0.0)
        family_prior = _normalize_nonnegative(family_prior)

        guidance = torque_alignment * (1.0 + float(FAMILY_GUIDANCE_GAIN) * family_prior)

        thumb_flex_demand, thumb_opposition_demand = self._thumb_motion_demand(q_err, target_qvel)
        if thumb_flex_demand > 1e-6 or thumb_opposition_demand > 1e-6:
            for idx, muscle_name in enumerate(self._actuator_names):
                name = str(muscle_name)
                if name == "FPL":
                    guidance[idx] *= 1.0 + float(THUMB_FLEXOR_BOOST_GAIN) * thumb_flex_demand
                elif name == "OP":
                    guidance[idx] *= 1.0 + float(THUMB_OPPOSITION_BOOST_GAIN) * thumb_opposition_demand
                elif name == "APL":
                    guidance[idx] *= 1.0 + float(THUMB_ABDUCTOR_BOOST_GAIN) * thumb_opposition_demand
                elif name in {"EPL", "EPB"}:
                    guidance[idx] *= max(0.0, 1.0 - float(THUMB_EXTENSOR_SUPPRESS_GAIN) * thumb_flex_demand)

        index_curl_demand = self._digit_curl_demand(
            q_err,
            target_qvel,
            "index",
            pip_gain=INDEX_PIP_GUIDANCE_GAIN,
            dip_gain=INDEX_DIP_GUIDANCE_GAIN,
        )
        if index_curl_demand > 1e-6:
            for idx, muscle_name in enumerate(self._actuator_names):
                name = str(muscle_name)
                if name in {"FDS2", "FDP2"}:
                    guidance[idx] *= 1.0 + float(INDEX_FLEXOR_BOOST_GAIN) * index_curl_demand
                elif name in {"EDC2", "EIP"}:
                    guidance[idx] *= max(0.0, 1.0 - float(INDEX_EXTENSOR_SUPPRESS_GAIN) * index_curl_demand)

        little_curl_demand = self._digit_curl_demand(
            q_err,
            target_qvel,
            "little",
            pip_gain=LITTLE_PIP_GUIDANCE_GAIN,
            dip_gain=LITTLE_DIP_GUIDANCE_GAIN,
        )
        if little_curl_demand > 1e-6:
            for idx, muscle_name in enumerate(self._actuator_names):
                name = str(muscle_name)
                if name in {"FDS5", "FDP5"}:
                    guidance[idx] *= 1.0 + float(LITTLE_FLEXOR_BOOST_GAIN) * little_curl_demand
                elif name in {"EDC5", "EDM"}:
                    guidance[idx] *= max(0.0, 1.0 - float(LITTLE_EXTENSOR_SUPPRESS_GAIN) * little_curl_demand)
        return _normalize_nonnegative(guidance)

    def _score_pose_candidate(self, data_candidate, target_qpos, ctrl_candidate):
        q_err = np.asarray(target_qpos, dtype=float).reshape(-1) - np.asarray(data_candidate.qpos, dtype=float).reshape(-1)
        if len(q_err) == len(self.actuated_mask):
            q_err_act = q_err[self.actuated_mask]
        else:
            q_err_act = q_err
        pose_cost = float(np.dot(q_err_act, q_err_act))
        current_q_err = np.asarray(target_qpos, dtype=float).reshape(-1) - np.asarray(self._data.qpos, dtype=float).reshape(-1)
        if len(current_q_err) == len(self.actuated_mask):
            current_q_err = current_q_err[self.actuated_mask]
        current_pose_norm = float(np.linalg.norm(current_q_err))
        smooth_scale = 1.0 / (1.0 + float(POSE_SCORE_SMOOTH_ERR_GAIN) * current_pose_norm)
        smooth_scale = float(np.clip(smooth_scale, float(POSE_SCORE_SMOOTH_MIN_SCALE), 1.0))
        ctrl_delta = np.asarray(ctrl_candidate, dtype=float).reshape(-1) - np.asarray(self._last_ctrl, dtype=float).reshape(-1)
        ctrl_cost = float(np.dot(ctrl_delta, ctrl_delta))
        ctrl_effort = float(np.dot(ctrl_candidate, ctrl_candidate))
        return (
            float(POSE_ERR_WEIGHT) * pose_cost
            + float(CTRL_SMOOTH_WEIGHT) * smooth_scale * ctrl_cost
            + float(CTRL_EFFORT_SCORE_WEIGHT) * ctrl_effort
        )

    def _rollout_candidate(self, ctrl_candidate, target_qpos):
        data_candidate = mujoco.MjData(self.model)
        data_candidate.qpos[:] = self._data.qpos
        data_candidate.qvel[:] = self._data.qvel
        data_candidate.qacc[:] = self._data.qacc
        data_candidate.ctrl[:] = np.asarray(ctrl_candidate, dtype=float).reshape(-1)
        try:
            if int(getattr(self.model, "na", 0)) > 0:
                data_candidate.act[:] = self._data.act
        except Exception:
            pass
        mujoco.mj_forward(self.model, data_candidate)
        for _ in range(int(max(1, POSE_LOOKAHEAD_STEPS))):
            data_candidate.ctrl[:] = np.asarray(ctrl_candidate, dtype=float).reshape(-1)
            mujoco.mj_step(self.model, data_candidate)
        return data_candidate

    def _select_pose_best_ctrl(self, ctrl_nominal, ctrl_blended, ctrl_hold, ctrl_track, ctrl_mid, target_qpos, pose_gap_norm):
        candidates = [
            _sanitize_ctrl(ctrl_nominal),
            _sanitize_ctrl(ctrl_hold),
            _sanitize_ctrl(ctrl_track),
            _sanitize_ctrl(ctrl_mid),
        ]
        ctrl_blended = _sanitize_ctrl(ctrl_blended)
        ctrl_prev = _sanitize_ctrl(self._last_ctrl)

        if (
            float(pose_gap_norm) <= float(BLENDED_CTRL_POSE_ERR_NORM)
            and not any(np.linalg.norm(ctrl_blended - candidate) <= 1e-8 for candidate in candidates)
        ):
            candidates.append(ctrl_blended)
        if (
            float(pose_gap_norm) <= float(PREV_CTRL_POSE_ERR_NORM)
            and not any(np.linalg.norm(ctrl_prev - candidate) <= 1e-8 for candidate in candidates)
        ):
            candidates.append(ctrl_prev)
        for shrink in CTRL_CANDIDATE_SHRINKS:
            shrunk = _sanitize_ctrl(float(shrink) * np.asarray(ctrl_nominal, dtype=float))
            if not any(np.linalg.norm(shrunk - candidate) <= 1e-8 for candidate in candidates):
                candidates.append(shrunk)
        for boost in CTRL_CANDIDATE_BOOSTS:
            boosted = _sanitize_ctrl(float(boost) * np.asarray(ctrl_nominal, dtype=float))
            if not any(np.linalg.norm(boosted - candidate) <= 1e-8 for candidate in candidates):
                candidates.append(boosted)

        best_ctrl = candidates[0]
        best_score = None
        for ctrl_candidate in candidates:
            data_candidate = self._rollout_candidate(ctrl_candidate, target_qpos)
            score = self._score_pose_candidate(data_candidate, target_qpos, ctrl_candidate)
            if best_score is None or score < best_score:
                best_score = score
                best_ctrl = ctrl_candidate
        return np.asarray(best_ctrl, dtype=float).copy()

    # ---------------------------------------------------------------
    def reset(self):
        """Reset private sim to rest pose."""
        self._data.qpos[:] = self.data_orig.qpos
        self._data.qvel[:] = 0.0
        self._data.qacc[:] = 0.0
        self._data.ctrl[:] = 0.0
        try:
            if int(getattr(self.model, "na", 0)) > 0:
                self._data.act[:] = self.data_orig.act
        except Exception:
            pass
        mujoco.mj_forward(self.model, self._data)
        self._prev_qpos = self._data.qpos.copy()
        self._last_target = self._data.qpos.copy()
        self._last_ctrl = np.zeros(int(self.model.nu), dtype=float)
        self._last_tau_des = np.zeros(int(self.model.nv), dtype=float)
        self._last_pose_error = np.zeros(int(self.model.nq), dtype=float)
        self._last_pose_error_norm = 0.0

# ---------------------------------------------------------------------------
# Dynamic Edge Scoring
# ---------------------------------------------------------------------------
DEFAULT_NODE_ACTIVATION_THRESHOLD = 0.15
DEFAULT_EDGE_SCORE_THRESHOLD = 0.12
DEFAULT_SCORING_MODE = "coactivation"
DEFAULT_WINDOW_SIZE = 10
DEFAULT_DEBUG_TOP_K = 5
DEFAULT_VISIBLE_EDGE_TOP_K = 20
DEFAULT_EDGE_RELATION_FILTER = "all_active"
SCORING_MODE_ALIASES = {
    "rolling_corr": "rolling_correlation",
}
VALID_SCORING_MODES = ("coactivation", "rolling_correlation", "hybrid")
EDGE_RELATION_FILTER_ALIASES = {
    "all": "all_active",
    "active": "all_active",
}
VALID_EDGE_RELATION_FILTERS = ("all_active", "synergist", "antagonist")


@dataclass
class DynamicEdgeScoringConfig:
    node_activation_threshold: float = DEFAULT_NODE_ACTIVATION_THRESHOLD
    edge_score_threshold: float = DEFAULT_EDGE_SCORE_THRESHOLD
    scoring_mode: str = DEFAULT_SCORING_MODE
    rolling_window_size: int = DEFAULT_WINDOW_SIZE
    debug_top_k: int = DEFAULT_DEBUG_TOP_K
    visible_edge_top_k: int = DEFAULT_VISIBLE_EDGE_TOP_K
    edge_relation_filter: str = DEFAULT_EDGE_RELATION_FILTER


def normalize_scoring_mode(scoring_mode: str) -> str:
    normalized = str(scoring_mode).strip().lower()
    return SCORING_MODE_ALIASES.get(normalized, normalized)


class DynamicEdgeScorer:
    def __init__(
        self,
        *,
        node_names: list[str],
        edge_index: np.ndarray,
        edge_type: np.ndarray,
        edge_weight: np.ndarray,
        edge_type_names: list[str],
        config: DynamicEdgeScoringConfig | None = None,
    ) -> None:
        self.node_names = list(node_names)
        self.edge_index = np.asarray(edge_index, dtype=np.int64)
        self.edge_type = np.asarray(edge_type, dtype=np.int64)
        self.edge_weight = np.asarray(edge_weight, dtype=np.float32)
        self.edge_type_names = list(edge_type_names)
        self.config = config or DynamicEdgeScoringConfig()
        self.config.scoring_mode = normalize_scoring_mode(self.config.scoring_mode)
        if self.config.scoring_mode not in VALID_SCORING_MODES:
            raise ValueError(f"Unsupported scoring_mode={self.config.scoring_mode!r}. Expected one of {VALID_SCORING_MODES}.")
        self.config.edge_relation_filter = self._normalize_edge_relation_filter(self.config.edge_relation_filter)
        if self.config.edge_relation_filter not in VALID_EDGE_RELATION_FILTERS:
            raise ValueError(
                f"Unsupported edge_relation_filter={self.config.edge_relation_filter!r}. "
                f"Expected one of {VALID_EDGE_RELATION_FILTERS}."
            )
        self._activation_history: deque[np.ndarray] = deque(maxlen=max(2, int(self.config.rolling_window_size)))

    def score(
        self,
        signal_values: np.ndarray,
        *,
        visibility_values: np.ndarray | None = None,
    ) -> dict[str, Any]:
        signal_vec = np.clip(np.asarray(signal_values, dtype=np.float32).reshape(-1), 0.0, 1.0)
        visibility_vec = signal_vec if visibility_values is None else np.clip(np.asarray(visibility_values, dtype=np.float32).reshape(-1), 0.0, 1.0)
        self._activation_history.append(signal_vec.copy())

        src = self.edge_index[0]
        dst = self.edge_index[1]
        src_signal = signal_vec[src]
        dst_signal = signal_vec[dst]
        coactivation = src_signal * dst_signal
        src_visibility = visibility_vec[src]
        dst_visibility = visibility_vec[dst]
        visibility_mask = (src_visibility >= float(self.config.node_activation_threshold)) & (
            dst_visibility >= float(self.config.node_activation_threshold)
        )

        rolling_corr = self._compute_rolling_corr(src, dst)
        score = self._combine_scores(coactivation, rolling_corr)
        relation_mask = self._relation_filter_mask()
        directed_relation_records = self._build_visible_edge_records(
            visible_mask=relation_mask,
            scores=score,
            coactivation=coactivation,
            rolling_corr=rolling_corr,
        )
        relation_edge_records = self._collapse_visible_edge_records(directed_relation_records)
        eligible_mask = visibility_mask & relation_mask
        directed_eligible_records = self._build_visible_edge_records(
            visible_mask=eligible_mask,
            scores=score,
            coactivation=coactivation,
            rolling_corr=rolling_corr,
        )
        eligible_edge_records = self._collapse_visible_edge_records(directed_eligible_records)
        candidate_edge_records = [
            item for item in eligible_edge_records if float(item["dynamic_score"]) >= float(self.config.edge_score_threshold)
        ]
        visible_edge_records = self._select_visible_edge_records(
            eligible_edge_records=eligible_edge_records,
            threshold_edge_records=candidate_edge_records,
            fallback_edge_records=relation_edge_records,
        )

        selected_edge_indices = np.asarray([int(item["edge_idx"]) for item in visible_edge_records], dtype=np.int64)
        visible_mask = np.zeros(score.shape[0], dtype=bool)
        if selected_edge_indices.size:
            visible_mask[selected_edge_indices] = True
        directed_visible_edge_records = [item for item in directed_relation_records if bool(visible_mask[int(item["edge_idx"])])]
        dynamic_edge_index = self.edge_index[:, visible_mask]
        dynamic_edge_weight = score[visible_mask].astype(np.float32, copy=False)
        dynamic_edge_type = self.edge_type[visible_mask].astype(np.int64, copy=False)

        return {
            "dynamic_edge_index": dynamic_edge_index,
            "dynamic_edge_weight": dynamic_edge_weight,
            "dynamic_edge_type": dynamic_edge_type,
            "dynamic_edge_mask": visible_mask.astype(bool, copy=False),
            "dynamic_edge_score": score.astype(np.float32, copy=False),
            "dynamic_edge_coactivation": coactivation.astype(np.float32, copy=False),
            "dynamic_edge_rolling_corr": rolling_corr.astype(np.float32, copy=False),
            "relation_edges": relation_edge_records,
            "eligible_edges": eligible_edge_records,
            "candidate_edges": candidate_edge_records,
            "visible_edges": visible_edge_records,
            "visible_edges_directed": directed_visible_edge_records,
            "top_dynamic_edges": visible_edge_records[: int(self.config.debug_top_k)],
            "config": {
                "node_activation_threshold": float(self.config.node_activation_threshold),
                "edge_score_threshold": float(self.config.edge_score_threshold),
                "scoring_mode": self.config.scoring_mode,
                "rolling_window_size": int(self.config.rolling_window_size),
                "visible_edge_top_k": int(self.config.visible_edge_top_k),
                "edge_relation_filter": str(self.config.edge_relation_filter),
            },
        }

    def _normalize_edge_relation_filter(self, value: str) -> str:
        normalized = str(value).strip().lower()
        return EDGE_RELATION_FILTER_ALIASES.get(normalized, normalized)

    def _relation_filter_mask(self) -> np.ndarray:
        relation_filter = str(self.config.edge_relation_filter)
        if relation_filter == "all_active":
            return np.ones(self.edge_type.shape[0], dtype=bool)
        selected_indices = {
            idx
            for idx, name in enumerate(self.edge_type_names)
            if str(name).strip().lower() == relation_filter
        }
        if not selected_indices:
            return np.zeros(self.edge_type.shape[0], dtype=bool)
        return np.isin(self.edge_type, np.asarray(sorted(selected_indices), dtype=np.int64))

    def _compute_rolling_corr(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if len(self._activation_history) < 2:
            return np.zeros(src.shape[0], dtype=np.float32)

        hist = np.stack(list(self._activation_history), axis=0)
        corr = np.zeros(src.shape[0], dtype=np.float32)
        for idx, (src_idx, dst_idx) in enumerate(zip(src.tolist(), dst.tolist())):
            a = hist[:, src_idx]
            b = hist[:, dst_idx]
            if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                corr[idx] = 0.0
                continue
            value = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(value):
                value = 0.0
            corr[idx] = float(np.clip(value, -1.0, 1.0))
        return corr

    def _select_visible_edge_records(
        self,
        *,
        eligible_edge_records: list[dict[str, Any]],
        threshold_edge_records: list[dict[str, Any]],
        fallback_edge_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        top_k = int(self.config.visible_edge_top_k)
        source_records = eligible_edge_records if eligible_edge_records else fallback_edge_records
        if not source_records:
            return []
        max_visible = len(source_records) if top_k <= 0 else min(top_k, len(source_records))
        if max_visible <= 0:
            return []

        if not threshold_edge_records:
            return source_records[: min(5, max_visible)]

        selected: list[dict[str, Any]] = list(threshold_edge_records[:max_visible])
        if len(selected) >= max_visible:
            return selected

        selected_pairs = {
            (str(item["source_name"]), str(item["target_name"]), str(item["edge_type"]))
            for item in selected
        }
        for item in source_records:
            key = (str(item["source_name"]), str(item["target_name"]), str(item["edge_type"]))
            if key in selected_pairs:
                continue
            selected.append(item)
            selected_pairs.add(key)
            if len(selected) >= max_visible:
                break
        return selected

    def _combine_scores(self, coactivation: np.ndarray, rolling_corr: np.ndarray) -> np.ndarray:
        weighted_coactivation = coactivation * self.edge_weight
        if self.config.scoring_mode == "coactivation":
            return weighted_coactivation
        if self.config.scoring_mode == "rolling_correlation":
            return np.clip(rolling_corr, 0.0, 1.0) * weighted_coactivation
        return weighted_coactivation * (0.5 + 0.5 * np.clip(rolling_corr, 0.0, 1.0))

    def _build_visible_edge_records(
        self,
        *,
        visible_mask: np.ndarray,
        scores: np.ndarray,
        coactivation: np.ndarray,
        rolling_corr: np.ndarray,
    ) -> list[dict[str, Any]]:
        visible_indices = np.where(visible_mask)[0]
        records: list[dict[str, Any]] = []
        for edge_idx in visible_indices.tolist():
            source_idx = int(self.edge_index[0, edge_idx])
            target_idx = int(self.edge_index[1, edge_idx])
            edge_type_index = int(self.edge_type[edge_idx])
            records.append(
                {
                    "edge_idx": edge_idx,
                    "source_name": self.node_names[source_idx],
                    "target_name": self.node_names[target_idx],
                    "edge_type": self.edge_type_names[edge_type_index] if 0 <= edge_type_index < len(self.edge_type_names) else str(edge_type_index),
                    "dynamic_score": float(scores[edge_idx]),
                    "coactivation_score": float(coactivation[edge_idx]),
                    "rolling_corr_score": float(rolling_corr[edge_idx]),
                    "prior_weight": float(self.edge_weight[edge_idx]),
                }
            )
        records.sort(key=lambda item: item["dynamic_score"], reverse=True)
        return records

    def _collapse_visible_edge_records(self, visible_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique_records: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in visible_edges:
            source_name = str(record["source_name"])
            target_name = str(record["target_name"])
            edge_type = str(record["edge_type"])
            pair = tuple(sorted((source_name, target_name)))
            key = (pair[0], pair[1], edge_type)
            existing = unique_records.get(key)
            candidate = {
                **record,
                "source_name": pair[0],
                "target_name": pair[1],
            }
            if existing is None or float(candidate["dynamic_score"]) > float(existing["dynamic_score"]):
                unique_records[key] = candidate
        collapsed = list(unique_records.values())
        collapsed.sort(key=lambda item: item["dynamic_score"], reverse=True)
        return collapsed

# ---------------------------------------------------------------------------
# Muscle Hand Layout
# ---------------------------------------------------------------------------
HAND_LAYOUT_REFERENCE_WIDTH = 1080
HAND_LAYOUT_REFERENCE_HEIGHT = 1080

_REFERENCE_X_CENTER = 540.0
_REFERENCE_Y_BASELINE = 980.0
_REFERENCE_X_SCALE = 390.0
_REFERENCE_Y_SCALE = 600.0

ZONE_DISPLAY_NAMES = {
    "wrist_forearm": "Wrist / Forearm",
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "little": "Little",
    "central_shared": "Palm / Intrinsics",
}

MUSCLE_VISUAL_ZONES = {
    "ECRL": "wrist_forearm",
    "ECRB": "wrist_forearm",
    "ECU": "wrist_forearm",
    "FCR": "wrist_forearm",
    "FCU": "wrist_forearm",
    "PL": "wrist_forearm",
    "PT": "wrist_forearm",
    "PQ": "wrist_forearm",
    "EPL": "thumb",
    "EPB": "thumb",
    "FPL": "thumb",
    "APL": "thumb",
    "FDS2": "index",
    "FDP2": "index",
    "EDC2": "index",
    "EIP": "index",
    "FDS3": "middle",
    "FDP3": "middle",
    "EDC3": "middle",
    "FDS4": "ring",
    "FDP4": "ring",
    "EDC4": "ring",
    "FDS5": "little",
    "FDP5": "little",
    "EDC5": "little",
    "EDM": "little",
    "OP": "central_shared",
    "RI2": "central_shared",
    "LU_RB2": "central_shared",
    "UI_UB2": "central_shared",
    "RI3": "central_shared",
    "LU_RB3": "central_shared",
    "UI_UB3": "central_shared",
    "RI4": "central_shared",
    "LU_RB4": "central_shared",
    "UI_UB4": "central_shared",
    "RI5": "central_shared",
    "LU_RB5": "central_shared",
    "UI_UB5": "central_shared",
}

NORMALIZED_NODE_LAYOUT = {
    # THUMB
    "APL": (-0.80, 0.70),
    "EPB": (-0.80, 0.90),
    "EPL": (-0.80, 1.10),
    "FPL": (-0.80, 0.50),
    # INDEX
    "FDP2": (-0.40, 0.40),
    "FDS2": (-0.40, 0.60),
    "EDC2": (-0.40, 0.85),
    "EIP": (-0.40, 1.10),
    # MIDDLE
    "FDP3": (-0.10, 0.40),
    "FDS3": (-0.10, 0.60),
    "EDC3": (-0.10, 0.85),
    # RING
    "FDP4": (0.20, 0.40),
    "FDS4": (0.20, 0.60),
    "EDC4": (0.20, 0.85),
    # LITTLE
    "FDP5": (0.50, 0.35),
    "FDS5": (0.50, 0.55),
    "EDC5": (0.50, 0.80),
    "EDM": (0.50, 1.05),
    # PALM / INTRINSICS
    "OP": (-0.30, 0.50),
    "UI_UB2": (-0.48, 0.48),
    "LU_RB2": (-0.40, 0.58),
    "RI2": (-0.30, 0.70),
    "UI_UB3": (-0.18, 0.48),
    "LU_RB3": (-0.10, 0.58),
    "RI3": (0.00, 0.70),
    "UI_UB4": (0.12, 0.48),
    "LU_RB4": (0.20, 0.58),
    "RI4": (0.30, 0.70),
    "UI_UB5": (0.42, 0.46),
    "LU_RB5": (0.50, 0.56),
    "RI5": (0.60, 0.70),
    # WRIST / FOREARM
    "ECU": (-0.30, 0.10),
    "PT": (-0.22, 0.02),
    "ECRB": (-0.10, 0.15),
    "ECRL": (0.00, 0.20),
    "PL": (0.08, 0.03),
    "FCR": (0.20, 0.15),
    "PQ": (0.26, 0.03),
    "FCU": (0.40, 0.10),
}


def _point(x: float, y: float) -> tuple[int, int]:
    return (int(round(x)), int(round(y)))


def _norm_to_reference(x: float, y: float) -> tuple[int, int]:
    return _point(
        _REFERENCE_X_CENTER + (float(x) * _REFERENCE_X_SCALE),
        _REFERENCE_Y_BASELINE - (float(y) * _REFERENCE_Y_SCALE),
    )


def _build_reference_coords() -> dict[str, tuple[int, int]]:
    return {
        name: _norm_to_reference(float(x), float(y))
        for name, (x, y) in NORMALIZED_NODE_LAYOUT.items()
    }


def _build_zone_label_positions() -> dict[str, tuple[int, int]]:
    return {
        "thumb": _norm_to_reference(-0.80, 1.20),
        "index": _norm_to_reference(-0.40, 1.20),
        "middle": _norm_to_reference(-0.10, 1.20),
        "ring": _norm_to_reference(0.20, 1.20),
        "little": _norm_to_reference(0.50, 1.20),
        "central_shared": _norm_to_reference(0.00, 0.76),
        "wrist_forearm": _norm_to_reference(0.00, 0.00),
    }


HAND_LAYOUT_COORDS = _build_reference_coords()
ZONE_LABEL_POSITIONS = _build_zone_label_positions()


def _resolve_scale_and_offset(
    *,
    width: int,
    height: int,
    bounds: tuple[int, int, int, int] | None = None,
) -> tuple[float, float, int, int]:
    if bounds is None:
        return (
            float(width) / float(HAND_LAYOUT_REFERENCE_WIDTH),
            float(height) / float(HAND_LAYOUT_REFERENCE_HEIGHT),
            0,
            0,
        )

    x0, y0, x1, y1 = bounds
    usable_width = max(1, int(x1) - int(x0))
    usable_height = max(1, int(y1) - int(y0))
    return (
        float(usable_width) / float(HAND_LAYOUT_REFERENCE_WIDTH),
        float(usable_height) / float(HAND_LAYOUT_REFERENCE_HEIGHT),
        int(x0),
        int(y0),
    )


def zone_for_muscle(muscle_name: str) -> str:
    return MUSCLE_VISUAL_ZONES.get(str(muscle_name), "central_shared")


def display_name_for_zone(zone_name: str) -> str:
    return ZONE_DISPLAY_NAMES.get(str(zone_name), str(zone_name).replace("_", " ").title())


def build_hand_layout_positions(
    node_names: Iterable[str],
    *,
    width: int = HAND_LAYOUT_REFERENCE_WIDTH,
    height: int = HAND_LAYOUT_REFERENCE_HEIGHT,
    bounds: tuple[int, int, int, int] | None = None,
) -> dict[str, tuple[int, int]]:
    scale_x, scale_y, offset_x, offset_y = _resolve_scale_and_offset(width=width, height=height, bounds=bounds)
    positions: dict[str, tuple[int, int]] = {}
    zone_members: dict[str, list[str]] = defaultdict(list)

    for node_name in node_names:
        name = str(node_name)
        zone_members[zone_for_muscle(name)].append(name)

    for zone_name, members in zone_members.items():
        zone_anchor = ZONE_LABEL_POSITIONS.get(zone_name, _norm_to_reference(0.0, 0.52))
        for member_idx, name in enumerate(sorted(members)):
            raw = HAND_LAYOUT_COORDS.get(name)
            if raw is None:
                dx = 24 * (member_idx % 3)
                dy = 32 * (member_idx // 3)
                raw = (zone_anchor[0] + dx, zone_anchor[1] + 72 + dy)
            positions[name] = (
                int(round(raw[0] * scale_x)) + offset_x,
                int(round(raw[1] * scale_y)) + offset_y,
            )
    return positions


def scaled_zone_label_positions(
    *,
    width: int,
    height: int,
    bounds: tuple[int, int, int, int] | None = None,
) -> dict[str, tuple[int, int]]:
    scale_x, scale_y, offset_x, offset_y = _resolve_scale_and_offset(width=width, height=height, bounds=bounds)
    return {
        zone_name: (int(round(x * scale_x)) + offset_x, int(round(y * scale_y)) + offset_y)
        for zone_name, (x, y) in ZONE_LABEL_POSITIONS.items()
    }

# ---------------------------------------------------------------------------
# Muscle Visualization Config
# ---------------------------------------------------------------------------
DEFAULT_SHOW_LABELS = True
DEFAULT_CANVAS_WIDTH = 1440
DEFAULT_CANVAS_HEIGHT = 1080
DEFAULT_HIGHLIGHT_TOP_NODES = 8
DEFAULT_VISUALIZATION_MODE = "dominant"
VALID_VISUALIZATION_MODES = ("dominant", "full", "synergy_only", "antagonist_only")
DEFAULT_DOMINANT_SELECTION_MODE = "functional_subset"
VALID_DOMINANT_SELECTION_MODES = ("functional_subset", "top_k_ctrl")
DEFAULT_DOMINANT_MODE = True
DEFAULT_DOMINANT_TOP_K = 8
DEFAULT_DOMINANT_EDGE_TOP_K = 16
DEFAULT_GHOST_NON_DOMINANT_NODES = True
DEFAULT_GRAPH_PRESET = "custom"
VALID_GRAPH_PRESETS = ("custom", "clean", "synergist_only", "antagonist_only")


GRAPH_PRESET_OVERRIDES = {
    "clean": {
        "edge_score_threshold": 0.12,
        "visible_edge_top_k": 16,
        "edge_relation_filter": "all_active",
        "show_labels": True,
        "highlight_top_nodes": 8,
    },
    "synergist_only": {
        "edge_score_threshold": 0.12,
        "visible_edge_top_k": 16,
        "edge_relation_filter": "synergist",
        "show_labels": True,
        "highlight_top_nodes": 8,
    },
    "antagonist_only": {
        "edge_score_threshold": 0.12,
        "visible_edge_top_k": 16,
        "edge_relation_filter": "antagonist",
        "show_labels": True,
        "highlight_top_nodes": 8,
    },
}


@dataclass
class MuscleGraphVisualizationConfig:
    visualization_mode: str = DEFAULT_VISUALIZATION_MODE
    dominant_selection_mode: str = DEFAULT_DOMINANT_SELECTION_MODE
    graph_preset: str = DEFAULT_GRAPH_PRESET
    node_activation_threshold: float = DEFAULT_NODE_ACTIVATION_THRESHOLD
    edge_score_threshold: float = DEFAULT_EDGE_SCORE_THRESHOLD
    scoring_mode: str = DEFAULT_SCORING_MODE
    rolling_window_size: int = DEFAULT_WINDOW_SIZE
    visible_edge_top_k: int = DEFAULT_VISIBLE_EDGE_TOP_K
    edge_relation_filter: str = DEFAULT_EDGE_RELATION_FILTER
    show_labels: bool = DEFAULT_SHOW_LABELS
    highlight_top_nodes: int = DEFAULT_HIGHLIGHT_TOP_NODES
    dominant_mode: bool = DEFAULT_DOMINANT_MODE
    dominant_top_k: int = DEFAULT_DOMINANT_TOP_K
    dominant_edge_top_k: int = DEFAULT_DOMINANT_EDGE_TOP_K
    include_thumb_in_dominant_subset: bool = True
    hide_inactive_dominant_nodes: bool = True
    ghost_non_dominant_nodes: bool = DEFAULT_GHOST_NON_DOMINANT_NODES
    canvas_width: int = DEFAULT_CANVAS_WIDTH
    canvas_height: int = DEFAULT_CANVAS_HEIGHT

    def to_dynamic_edge_config(self) -> DynamicEdgeScoringConfig:
        return DynamicEdgeScoringConfig(
            node_activation_threshold=float(self.node_activation_threshold),
            edge_score_threshold=float(self.edge_score_threshold),
            scoring_mode=str(self.scoring_mode),
            rolling_window_size=int(self.rolling_window_size),
            visible_edge_top_k=int(self.visible_edge_top_k),
            edge_relation_filter=str(self.edge_relation_filter),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_visualization_preset(
    config: MuscleGraphVisualizationConfig,
    preset_name: str | None,
) -> MuscleGraphVisualizationConfig:
    preset = str(preset_name or DEFAULT_GRAPH_PRESET).strip().lower()
    if preset not in VALID_GRAPH_PRESETS:
        preset = DEFAULT_GRAPH_PRESET
    if preset == DEFAULT_GRAPH_PRESET:
        config.graph_preset = DEFAULT_GRAPH_PRESET
        return config

    overrides = dict(GRAPH_PRESET_OVERRIDES.get(preset) or {})
    updated = asdict(config)
    updated.update(overrides)
    updated["graph_preset"] = preset
    return MuscleGraphVisualizationConfig(**updated)


def apply_visualization_mode(
    config: MuscleGraphVisualizationConfig,
    mode_name: str | None,
) -> MuscleGraphVisualizationConfig:
    mode = str(mode_name or DEFAULT_VISUALIZATION_MODE).strip().lower()
    if mode not in VALID_VISUALIZATION_MODES:
        mode = DEFAULT_VISUALIZATION_MODE

    updated = asdict(config)
    updated["visualization_mode"] = mode

    if mode == "full":
        updated["dominant_mode"] = False
        updated["edge_relation_filter"] = "all_active"
    elif mode == "synergy_only":
        updated["dominant_mode"] = True
        updated["edge_relation_filter"] = "synergist"
    elif mode == "antagonist_only":
        updated["dominant_mode"] = True
        updated["edge_relation_filter"] = "antagonist"
    else:
        updated["dominant_mode"] = True
        updated["edge_relation_filter"] = "all_active"

    return MuscleGraphVisualizationConfig(**updated)


def apply_dominant_selection_mode(
    config: MuscleGraphVisualizationConfig,
    selection_mode: str | None,
) -> MuscleGraphVisualizationConfig:
    mode = str(selection_mode or DEFAULT_DOMINANT_SELECTION_MODE).strip().lower()
    if mode not in VALID_DOMINANT_SELECTION_MODES:
        mode = DEFAULT_DOMINANT_SELECTION_MODE

    updated = asdict(config)
    updated["dominant_selection_mode"] = mode
    return MuscleGraphVisualizationConfig(**updated)

# ---------------------------------------------------------------------------
# Muscle Graph Spec
# ---------------------------------------------------------------------------
EDGE_TYPES = ("synergist", "antagonist", "functional_group", "data_driven")
NODE_FEATURE_NAMES = ("activation", "force", "length", "velocity")
MOMENT_EPS = 1e-6
SYNERGIST_THRESHOLD = 0.65
ANTAGONIST_THRESHOLD = -0.65
DATA_DRIVEN_THRESHOLD = 0.85

FALLBACK_MUSCLE_NAMES = [
    "ECRL",
    "ECRB",
    "ECU",
    "FCR",
    "FCU",
    "PL",
    "PT",
    "PQ",
    "FDS5",
    "FDS4",
    "FDS3",
    "FDS2",
    "FDP5",
    "FDP4",
    "FDP3",
    "FDP2",
    "EDC5",
    "EDC4",
    "EDC3",
    "EDC2",
    "EDM",
    "EIP",
    "EPL",
    "EPB",
    "FPL",
    "APL",
    "OP",
    "RI2",
    "LU_RB2",
    "UI_UB2",
    "RI3",
    "LU_RB3",
    "UI_UB3",
    "RI4",
    "LU_RB4",
    "UI_UB4",
    "RI5",
    "LU_RB5",
    "UI_UB5",
]

# Nodes below have only generic functional-group edges in the current graph
# and do not contribute meaningful mechanical structure in the zone view.
GRAPH_EXCLUDED_NODE_NAMES = frozenset(
    {
        "RI2",
        "RI3",
        "RI4",
        "RI5",
        "LU_RB2",
        "LU_RB3",
        "LU_RB4",
        "LU_RB5",
        "UI_UB2",
        "UI_UB3",
        "UI_UB4",
        "UI_UB5",
    }
)

FUNCTIONAL_GROUPS = {
    "wrist_extensors": ("ECRL", "ECRB", "ECU"),
    "wrist_flexors": ("FCR", "FCU", "PL"),
    "pronators": ("PT", "PQ"),
    "thumb": ("EPL", "EPB", "FPL", "APL", "OP"),
    "digit2_flexors": ("FDS2", "FDP2"),
    "digit3_flexors": ("FDS3", "FDP3"),
    "digit4_flexors": ("FDS4", "FDP4"),
    "digit5_flexors": ("FDS5", "FDP5"),
    "digit2_extensors": ("EDC2", "EIP"),
    "digit3_extensors": ("EDC3",),
    "digit4_extensors": ("EDC4",),
    "digit5_extensors": ("EDC5", "EDM"),
    "digit2_intrinsics": ("RI2", "LU_RB2", "UI_UB2"),
    "digit3_intrinsics": ("RI3", "LU_RB3", "UI_UB3"),
    "digit4_intrinsics": ("RI4", "LU_RB4", "UI_UB4"),
    "digit5_intrinsics": ("RI5", "LU_RB5", "UI_UB5"),
}

DOMINANT_MUSCLE_CORE_PATTERNS = (
    "FDP",
    "FDS",
    "EDC",
    "EIP",
    "EDM",
    "FCR",
    "FCU",
    "ECR",
    "ECU",
)
DOMINANT_MUSCLE_THUMB_PATTERNS = (
    "FPL",
    "EPL",
    "APL",
    "EPB",
)

MUSCLE_TO_GROUPS: dict[str, tuple[str, ...]] = {}
for group_name, muscles in FUNCTIONAL_GROUPS.items():
    for muscle_name in muscles:
        MUSCLE_TO_GROUPS.setdefault(muscle_name, [])
        MUSCLE_TO_GROUPS[muscle_name] = tuple(sorted(set((*MUSCLE_TO_GROUPS[muscle_name], group_name))))


@dataclass
class MuscleModelContext:
    env_id: str
    env: Any
    model: Any
    data: Any
    actuator_names: list[str]
    joint_names: list[str]
    moment_matrix: np.ndarray  # shape: (num_muscles, num_joints)


def create_muscle_model_context(env_id: str = DEFAULT_MYOSUITE_ENV_ID) -> MuscleModelContext:
    env, model, data = create_probe_env(env_id)
    m_raw, d_raw = _get_raw_model_data(model, data)
    mujoco.mj_forward(m_raw, d_raw)

    actuator_names = [
        mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) or f"muscle_{idx}"
        for idx in range(int(model.nu))
    ]
    joint_names = [
        mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_JOINT, idx) or f"joint_{idx}"
        for idx in range(int(model.nv))
    ]

    moment_matrix = np.zeros((int(model.nu), int(model.nv)), dtype=float)
    mujoco.mju_sparse2dense(
        moment_matrix,
        d_raw.actuator_moment.reshape(-1),
        d_raw.moment_rownnz,
        d_raw.moment_rowadr,
        d_raw.moment_colind.reshape(-1),
    )
    return MuscleModelContext(
        env_id=env_id,
        env=env,
        model=model,
        data=data,
        actuator_names=actuator_names,
        joint_names=joint_names,
        moment_matrix=moment_matrix,
    )


def muscle_groups_for_name(muscle_name: str) -> tuple[str, ...]:
    return MUSCLE_TO_GROUPS.get(muscle_name, ())


def muscle_matches_pattern(muscle_name: str, pattern: str) -> bool:
    name = str(muscle_name).strip().upper()
    token = str(pattern).strip().upper()
    if not token:
        return False
    if token in {"FDP", "FDS", "EDC", "ECR"}:
        return name.startswith(token)
    return name == token


def dominant_muscle_patterns(*, include_thumb: bool = True) -> tuple[str, ...]:
    if include_thumb:
        return tuple((*DOMINANT_MUSCLE_CORE_PATTERNS, *DOMINANT_MUSCLE_THUMB_PATTERNS))
    return tuple(DOMINANT_MUSCLE_CORE_PATTERNS)


def resolve_dominant_muscle_subset(
    muscle_names: list[str] | tuple[str, ...],
    *,
    include_thumb: bool = True,
) -> list[str]:
    patterns = dominant_muscle_patterns(include_thumb=include_thumb)
    resolved: list[str] = []
    for muscle_name in muscle_names:
        if any(muscle_matches_pattern(muscle_name, pattern) for pattern in patterns):
            resolved.append(str(muscle_name))
    return resolved


def build_moment_signature(moment_row: np.ndarray, joint_names: list[str], *, threshold: float = MOMENT_EPS) -> dict[str, float]:
    signature: dict[str, float] = {}
    for joint_name, value in zip(joint_names, moment_row.tolist()):
        if abs(value) > threshold:
            signature[joint_name] = float(value)
    return signature


def resolve_graph_node_names(context: MuscleModelContext | None = None) -> list[str]:
    muscle_names = context.actuator_names if context is not None else list(FALLBACK_MUSCLE_NAMES)
    return [str(name) for name in muscle_names if str(name) not in GRAPH_EXCLUDED_NODE_NAMES]


def _pair_cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float | None:
    mask = (np.abs(vec_a) > MOMENT_EPS) & (np.abs(vec_b) > MOMENT_EPS)
    if not np.any(mask):
        return None

    a = vec_a[mask]
    b = vec_b[mask]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return None
    return float(np.dot(a, b) / denom)


def build_muscle_nodes(context: MuscleModelContext | None = None) -> list[dict[str, Any]]:
    muscle_names = resolve_graph_node_names(context)
    joint_names = context.joint_names if context is not None else []
    moment_matrix = context.moment_matrix if context is not None else np.zeros((len(muscle_names), 0), dtype=float)
    actuator_index = {str(name): idx for idx, name in enumerate(context.actuator_names)} if context is not None else {str(name): idx for idx, name in enumerate(muscle_names)}

    nodes: list[dict[str, Any]] = []
    for node_index, muscle_name in enumerate(muscle_names):
        muscle_index = int(actuator_index.get(str(muscle_name), node_index))
        moment_row = moment_matrix[muscle_index] if muscle_index < len(moment_matrix) else np.zeros(0, dtype=float)
        nodes.append(
            {
                "node_name": muscle_name,
                "node_type": "muscle",
                "metadata": {
                    "muscle_index": muscle_index,
                    "functional_groups": list(muscle_groups_for_name(muscle_name)),
                    "moment_signature": build_moment_signature(moment_row, joint_names),
                },
            }
        )
    return nodes


def _append_bidirectional_edge(
    edges: list[dict[str, Any]],
    source_name: str,
    target_name: str,
    edge_type: str,
    edge_weight: float,
    metadata: dict[str, Any],
) -> None:
    for source_id, target_id in ((source_name, target_name), (target_name, source_name)):
        edges.append(
            {
                "source_name": source_id,
                "target_name": target_id,
                "edge_type": edge_type,
                "edge_weight": float(edge_weight),
                "metadata": metadata,
            }
        )


def build_muscle_edges(
    context: MuscleModelContext | None = None,
    *,
    include_data_driven: bool = False,
) -> list[dict[str, Any]]:
    muscle_names = resolve_graph_node_names(context)
    moment_matrix = context.moment_matrix if context is not None else np.zeros((len(muscle_names), 0), dtype=float)
    actuator_index = {str(name): idx for idx, name in enumerate(context.actuator_names)} if context is not None else {str(name): idx for idx, name in enumerate(muscle_names)}

    edges: list[dict[str, Any]] = []
    seen_relations: set[tuple[str, str, str]] = set()

    def register_pair(source_name: str, target_name: str, edge_type: str, edge_weight: float, metadata: dict[str, Any]) -> None:
        key = tuple(sorted((source_name, target_name)) + [edge_type])  # type: ignore[operator]
        if key in seen_relations:
            return
        seen_relations.add(key)
        _append_bidirectional_edge(edges, source_name, target_name, edge_type, edge_weight, metadata)

    for idx, source_name in enumerate(muscle_names):
        for jdx in range(idx + 1, len(muscle_names)):
            target_name = muscle_names[jdx]
            source_groups = set(muscle_groups_for_name(source_name))
            target_groups = set(muscle_groups_for_name(target_name))
            shared_groups = sorted(source_groups & target_groups)

            if shared_groups:
                register_pair(
                    source_name,
                    target_name,
                    "functional_group",
                    1.0,
                    {"shared_groups": shared_groups},
                )

            source_idx = int(actuator_index.get(str(source_name), idx))
            target_idx = int(actuator_index.get(str(target_name), jdx))
            cosine = _pair_cosine(moment_matrix[source_idx], moment_matrix[target_idx]) if context is not None else None
            if cosine is None:
                continue

            if cosine >= SYNERGIST_THRESHOLD:
                register_pair(
                    source_name,
                    target_name,
                    "synergist",
                    abs(cosine),
                    {"moment_cosine": cosine},
                )
                continue

            if cosine <= ANTAGONIST_THRESHOLD:
                register_pair(
                    source_name,
                    target_name,
                    "antagonist",
                    abs(cosine),
                    {"moment_cosine": cosine},
                )
                continue

            if include_data_driven and abs(cosine) >= DATA_DRIVEN_THRESHOLD:
                register_pair(
                    source_name,
                    target_name,
                    "data_driven",
                    abs(cosine),
                    {"moment_cosine": cosine},
                )

    for edge_id, edge in enumerate(edges, start=1):
        edge["edge_id"] = edge_id
    return edges


def build_muscle_graph_spec(
    *,
    context: MuscleModelContext | None = None,
    include_data_driven: bool = False,
) -> dict[str, Any]:
    nodes = build_muscle_nodes(context)
    edges = build_muscle_edges(context, include_data_driven=include_data_driven)
    return {
        "env_id": context.env_id if context is not None else DEFAULT_MYOSUITE_ENV_ID,
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_types": list(EDGE_TYPES),
        "nodes": nodes,
        "edges": edges,
    }


def extract_muscle_signals(
    context: MuscleModelContext | None,
    *,
    qpos_values: list[float],
    ctrl_values: list[float],
    muscle_names: list[str] | None = None,
) -> dict[str, list[float]]:
    all_muscle_names = context.actuator_names if context is not None else list(FALLBACK_MUSCLE_NAMES)
    selected_muscle_names = [str(name) for name in (muscle_names if muscle_names is not None else all_muscle_names)]
    num_muscles = len(all_muscle_names)
    ctrl_full = np.zeros(num_muscles, dtype=float)
    ctrl_full[: min(num_muscles, len(ctrl_values))] = np.asarray(ctrl_values[:num_muscles], dtype=float)

    if context is None:
        selected_num = len(selected_muscle_names)
        zeros = np.zeros(selected_num, dtype=float)
        name_to_index = {str(name): idx for idx, name in enumerate(all_muscle_names)}
        ctrl = np.asarray([ctrl_full[int(name_to_index[name])] for name in selected_muscle_names], dtype=float)
        return {
            "ctrl": ctrl.tolist(),
            "activation": ctrl.tolist(),
            "force": zeros.tolist(),
            "length": zeros.tolist(),
            "velocity": zeros.tolist(),
        }

    model = context.model
    data = context.data
    m_raw, d_raw = _get_raw_model_data(model, data)

    full_qpos = np.zeros(int(model.nq), dtype=float)
    full_qpos[: min(len(full_qpos), len(qpos_values))] = np.asarray(qpos_values[: len(full_qpos)], dtype=float)

    d_raw.qpos[:] = full_qpos
    d_raw.qvel[:] = 0.0
    d_raw.qacc[:] = 0.0
    d_raw.ctrl[:] = ctrl_full

    try:
        if int(getattr(model, "na", 0)) > 0:
            d_raw.act[:] = ctrl_full[: len(d_raw.act)]
    except Exception:
        pass

    mujoco.mj_forward(m_raw, d_raw)

    activation_full = np.zeros(num_muscles, dtype=float)
    for idx in range(num_muscles):
        if int(model.actuator_actnum[idx]) > 0:
            activation_full[idx] = float(d_raw.act[int(model.actuator_actadr[idx])])
        else:
            activation_full[idx] = float(ctrl_full[idx])

    name_to_index = {str(name): idx for idx, name in enumerate(all_muscle_names)}
    selected_indices = [int(name_to_index[name]) for name in selected_muscle_names if name in name_to_index]
    ctrl = np.asarray([ctrl_full[idx] for idx in selected_indices], dtype=float)
    activation = np.asarray([activation_full[idx] for idx in selected_indices], dtype=float)
    force = np.asarray([d_raw.actuator_force[idx] for idx in selected_indices], dtype=float)
    length = np.asarray([d_raw.actuator_length[idx] for idx in selected_indices], dtype=float)
    velocity = np.asarray([d_raw.actuator_velocity[idx] for idx in selected_indices], dtype=float)

    return {
        "ctrl": ctrl.tolist(),
        "activation": activation.tolist(),
        "force": force.tolist(),
        "length": length.tolist(),
        "velocity": velocity.tolist(),
    }

# ---------------------------------------------------------------------------
# Realtime Muscle Graph
# ---------------------------------------------------------------------------
class RealtimeMuscleGraph:
    def __init__(
        self,
        *,
        env_id: str = DEFAULT_MYOSUITE_ENV_ID,
        include_data_driven_edges: bool = False,
        dynamic_edge_config: DynamicEdgeScoringConfig | None = None,
        visualization_config: MuscleGraphVisualizationConfig | None = None,
        debug_dynamic_edges: bool = False,
    ):
        self.context = create_muscle_model_context(env_id)
        self.graph_spec = build_muscle_graph_spec(
            context=self.context,
            include_data_driven=include_data_driven_edges,
        )
        self.node_names = [str(node["node_name"]) for node in self.graph_spec["nodes"]]
        self.node_feature_names = list(self.graph_spec["node_feature_names"])
        self.edge_type_names = list(self.graph_spec["edge_types"])
        self.edge_index = np.asarray(
            [
                [self.node_names.index(edge["source_name"]) for edge in self.graph_spec["edges"]],
                [self.node_names.index(edge["target_name"]) for edge in self.graph_spec["edges"]],
            ],
            dtype=np.int64,
        )
        self.edge_type = np.asarray(
            [self.edge_type_names.index(str(edge["edge_type"])) for edge in self.graph_spec["edges"]],
            dtype=np.int64,
        )
        self.edge_weight = np.asarray([float(edge["edge_weight"]) for edge in self.graph_spec["edges"]], dtype=np.float32)
        self.node_index = {name: idx for idx, name in enumerate(self.node_names)}
        self._joint_qpos_names = list(self.context.joint_names)
        self._graph_edges = [dict(edge) for edge in self.graph_spec["edges"]]
        self.dynamic_edge_config = dynamic_edge_config or DynamicEdgeScoringConfig()
        self.visualization_config = visualization_config or MuscleGraphVisualizationConfig(
            node_activation_threshold=float(self.dynamic_edge_config.node_activation_threshold),
            edge_score_threshold=float(self.dynamic_edge_config.edge_score_threshold),
            scoring_mode=str(self.dynamic_edge_config.scoring_mode),
            rolling_window_size=int(self.dynamic_edge_config.rolling_window_size),
        )
        self.node_visual_zones = {name: zone_for_muscle(name) for name in self.node_names}
        self.debug_dynamic_edges = bool(debug_dynamic_edges)
        self.dynamic_edge_scorer = DynamicEdgeScorer(
            node_names=self.node_names,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_weight=self.edge_weight,
            edge_type_names=self.edge_type_names,
            config=self.dynamic_edge_config,
        )
        self._last_logged_frame_index: int | None = None

    def get_static_payload(self) -> dict[str, Any]:
        return {
            "graph_kind": "muscle_graph",
            "env_id": self.graph_spec["env_id"],
            "node_names": list(self.node_names),
            "node_feature_names": list(self.node_feature_names),
            "edge_type_names": list(self.edge_type_names),
            "edge_index": self.edge_index.copy(),
            "edge_type": self.edge_type.copy(),
            "edge_weight": self.edge_weight.copy(),
            "nodes": [dict(node) for node in self.graph_spec["nodes"]],
            "edges": list(self._graph_edges),
            "joint_qpos_names": list(self._joint_qpos_names),
            "dynamic_edge_config": {
                "node_activation_threshold": float(self.dynamic_edge_config.node_activation_threshold),
                "edge_score_threshold": float(self.dynamic_edge_config.edge_score_threshold),
                "scoring_mode": self.dynamic_edge_config.scoring_mode,
                "rolling_window_size": int(self.dynamic_edge_config.rolling_window_size),
                "visible_edge_top_k": int(self.dynamic_edge_config.visible_edge_top_k),
                "edge_relation_filter": str(self.dynamic_edge_config.edge_relation_filter),
            },
            "visualization_config": self.visualization_config.to_dict(),
            "node_visual_zones": dict(self.node_visual_zones),
        }

    def update(
        self,
        *,
        qpos_values: np.ndarray | list[float],
        ctrl_values: np.ndarray | list[float],
        ctrl_raw_values: np.ndarray | list[float] | None = None,
        ctrl_smoothed_values: np.ndarray | list[float] | None = None,
        ctrl_summary: dict[str, Any] | None = None,
        timestamp: float | None = None,
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        qpos = np.asarray(qpos_values, dtype=float).reshape(-1)
        ctrl = self._normalize_node_signal(ctrl_values)
        signals = extract_muscle_signals(
            self.context,
            qpos_values=qpos.tolist(),
            ctrl_values=ctrl.tolist(),
            muscle_names=self.node_names,
        )
        x = np.asarray(
            [
                [float(signals[feature_name][idx]) for feature_name in self.node_feature_names]
                for idx in range(len(self.node_names))
            ],
            dtype=np.float32,
        )
        activation_idx = self.node_feature_names.index("activation") if "activation" in self.node_feature_names else 0
        activations = np.asarray(x[:, activation_idx], dtype=float)
        dynamic_edges = self._score_dynamic_edges(ctrl, activations)
        analysis = self._analyze_state(x, dynamic_edges, ctrl)
        dynamic_edge_summary = self._build_dynamic_edge_summary(dynamic_edges=dynamic_edges, analysis=analysis)
        if self.debug_dynamic_edges:
            self._maybe_log_dynamic_edges(frame_index=frame_index, analysis=analysis)

        node_states = []
        for idx, node in enumerate(self.graph_spec["nodes"]):
            signal_values = {feature_name: float(signals[feature_name][idx]) for feature_name in self.node_feature_names}
            node_states.append(
                {
                    "node_name": node["node_name"],
                    "node_type": node["node_type"],
                    "metadata": {
                        **dict(node["metadata"]),
                        "feature_names": list(self.node_feature_names),
                        "signal_values": signal_values,
                    },
                }
            )

        return {
            "timestamp": float(timestamp) if timestamp is not None else None,
            "frame_index": int(frame_index) if frame_index is not None else None,
            "node_names": self.node_names,
            "node_feature_names": self.node_feature_names,
            "x": x,
            "node_states": node_states,
            "edge_index": self.edge_index,
            "edge_type": self.edge_type,
            "edge_weight": self.edge_weight,
            "edges": self._graph_edges,
            "dynamic_edge_index": dynamic_edges["dynamic_edge_index"],
            "dynamic_edge_weight": dynamic_edges["dynamic_edge_weight"],
            "dynamic_edge_type": dynamic_edges["dynamic_edge_type"],
            "dynamic_edge_mask": dynamic_edges["dynamic_edge_mask"],
            "dynamic_edge_score": dynamic_edges["dynamic_edge_score"],
            "visible_edges": dynamic_edges["visible_edges"],
            "dominant_visible_edges": list(analysis.get("dominant_visible_edges") or []),
            "dynamic_edge_summary": dynamic_edge_summary,
            "ctrl_analysis": ctrl.astype(np.float32),
            "ctrl_summary": dict(ctrl_summary or {}),
            "joint_qpos": qpos.astype(np.float32),
            "joint_qpos_names": self._joint_qpos_names,
            "analysis": analysis,
            "meta": {
                "graph_kind": "muscle_graph",
                "env_id": self.graph_spec["env_id"],
                "dynamic_edge_config": dict(dynamic_edges["config"]),
            },
        }

    def _maybe_log_dynamic_edges(self, *, frame_index: int | None, analysis: dict[str, Any]) -> None:
        frame_key = int(frame_index) if frame_index is not None else -1
        if self._last_logged_frame_index == frame_key:
            return
        self._last_logged_frame_index = frame_key

        top_k = int(self.dynamic_edge_config.debug_top_k)
        top_edges = list(analysis.get("debug_edges") or [])[:top_k]
        top_nodes = list(analysis.get("debug_nodes") or [])[:top_k]
        active_zones = list(analysis.get("active_zones") or [])[:4]

        parts = []
        if top_nodes:
            parts.append(
                "nodes="
                + ", ".join(
                    f"{item['node_name']}:{float(item.get('ctrl_analysis', item.get('activation', 0.0))):.3f}"
                    for item in top_nodes[:top_k]
                )
            )
        if active_zones:
            parts.append(
                "zones="
                + ", ".join(f"{item['zone_display']}:{item['mean_activation']:.3f}" for item in active_zones)
            )
        header = f"[DYN-EDGE] frame={frame_key}"
        if parts:
            header += " | " + " | ".join(parts)
        print(header)
        if not top_edges:
            print("  no visible edges")
            return
        for rank, item in enumerate(top_edges[:top_k], start=1):
            print(
                "  "
                f"{rank:02d}. {item['source_name']} <-> {item['target_name']} "
                f"[{item['edge_type']}] score={float(item['dynamic_score']):.3f}"
            )

    def _normalize_node_signal(self, values: np.ndarray | list[float]) -> np.ndarray:
        signal = np.asarray(values, dtype=float).reshape(-1)
        if signal.shape[0] < len(self.node_names):
            padded = np.zeros(len(self.node_names), dtype=float)
            padded[: signal.shape[0]] = signal
            signal = padded
        elif signal.shape[0] > len(self.node_names):
            signal = signal[: len(self.node_names)]
        return np.clip(signal, 0.0, 1.0)

    def _score_dynamic_edges(self, ctrl_signal: np.ndarray, activations: np.ndarray) -> dict[str, Any]:
        return self.dynamic_edge_scorer.score(
            ctrl_signal,
            visibility_values=np.maximum(np.asarray(ctrl_signal, dtype=float), np.asarray(activations, dtype=float)),
        )

    def _analyze_state(self, x: np.ndarray, dynamic_edges: dict[str, Any], ctrl_analysis: np.ndarray) -> dict[str, Any]:
        activation_idx = self.node_feature_names.index("activation") if "activation" in self.node_feature_names else 0
        activations = np.asarray(x[:, activation_idx], dtype=float)
        ctrl_signal = self._normalize_node_signal(ctrl_analysis)

        group_scores: dict[str, list[float]] = {}
        zone_scores: dict[str, list[float]] = {}
        for idx, node in enumerate(self.graph_spec["nodes"]):
            for group_name in node.get("metadata", {}).get("functional_groups", []):
                group_scores.setdefault(str(group_name), [])
                group_scores[str(group_name)].append(float(activations[idx]))
            zone_name = self.node_visual_zones.get(self.node_names[idx], "central_shared")
            zone_scores.setdefault(zone_name, [])
            zone_scores[zone_name].append(float(activations[idx]))

        dominant_groups = sorted(
            (
                {
                    "group": group_name,
                    "mean_activation": float(np.mean(values)),
                    "max_activation": float(np.max(values)),
                }
                for group_name, values in group_scores.items()
                if values
            ),
            key=lambda item: (item["mean_activation"], item["max_activation"]),
            reverse=True,
        )[:6]

        top_active_count = max(1, int(self.visualization_config.highlight_top_nodes))
        top_active_indices = np.argsort(activations)[-top_active_count:][::-1]
        top_active = [
            {
                "node_name": self.node_names[idx],
                "activation": float(activations[idx]),
                "ctrl_analysis": float(ctrl_signal[idx]),
                "zone": self.node_visual_zones.get(self.node_names[idx], "central_shared"),
            }
            for idx in top_active_indices
        ]

        dominant_selection_mode = str(self.visualization_config.dominant_selection_mode).strip().lower()
        if dominant_selection_mode == "functional_subset":
            dominant_base_names = resolve_dominant_muscle_subset(
                self.node_names,
                include_thumb=bool(self.visualization_config.include_thumb_in_dominant_subset),
            )
            dominant_indices = [self.node_index[name] for name in dominant_base_names if name in self.node_index]
        else:
            dominant_top_k = max(1, int(self.visualization_config.dominant_top_k))
            dominant_indices = np.argsort(ctrl_signal)[-dominant_top_k:][::-1].tolist()

        dominant_nodes = sorted(
            [
                {
                    "node_name": self.node_names[idx],
                    "ctrl_analysis": float(ctrl_signal[idx]),
                    "activation": float(activations[idx]),
                    "zone": self.node_visual_zones.get(self.node_names[idx], "central_shared"),
                }
                for idx in dominant_indices
            ],
            key=lambda item: (float(item["ctrl_analysis"]), float(item["activation"])),
            reverse=True,
        )
        dominant_node_names = {str(item["node_name"]) for item in dominant_nodes}
        dominant_edge_threshold = float(self.visualization_config.edge_score_threshold)
        dominant_edge_top_k = max(1, int(self.visualization_config.dominant_edge_top_k))
        eligible_edges = list(dynamic_edges.get("eligible_edges") or dynamic_edges.get("candidate_edges") or dynamic_edges["visible_edges"])
        relation_edges = list(dynamic_edges.get("relation_edges") or eligible_edges)
        dominant_ranked_edges = [
            item
            for item in eligible_edges
            if (
                str(item["source_name"]) in dominant_node_names
                and str(item["target_name"]) in dominant_node_names
            )
        ]
        if not dominant_ranked_edges:
            dominant_ranked_edges = [
                item
                for item in relation_edges
                if (
                    str(item["source_name"]) in dominant_node_names
                    and str(item["target_name"]) in dominant_node_names
                )
            ]
        dominant_ranked_edges.sort(key=lambda item: float(item["dynamic_score"]), reverse=True)
        unique_dominant_edges: dict[tuple[str, str], dict[str, Any]] = {}
        for item in dominant_ranked_edges:
            pair_key = tuple(sorted((str(item["source_name"]), str(item["target_name"]))))
            existing = unique_dominant_edges.get(pair_key)
            if existing is None or float(item["dynamic_score"]) > float(existing["dynamic_score"]):
                unique_dominant_edges[pair_key] = item
        dominant_ranked_edges = sorted(unique_dominant_edges.values(), key=lambda item: float(item["dynamic_score"]), reverse=True)
        dominant_threshold_edges = [
            item for item in dominant_ranked_edges if float(item["dynamic_score"]) >= dominant_edge_threshold
        ]
        dominant_visible_edges = self._select_ranked_edges(
            ranked_edges=dominant_ranked_edges,
            threshold_edges=dominant_threshold_edges,
            max_edges=dominant_edge_top_k,
            fallback_top_k=5,
        )
        dominant_connected_node_names = sorted(
            {
                str(item["source_name"])
                for item in dominant_visible_edges
            }
            | {
                str(item["target_name"])
                for item in dominant_visible_edges
            }
        )
        dominant_active_node_names = sorted(
            {
                str(self.node_names[idx])
                for idx in dominant_indices
                if float(activations[idx]) >= float(self.visualization_config.node_activation_threshold)
            }
        )
        dominant_visible_node_names = sorted(
            set(dominant_connected_node_names)
            | (
                set(dominant_active_node_names)
                if bool(self.visualization_config.hide_inactive_dominant_nodes)
                else set(dominant_node_names)
            )
        )
        if not dominant_visible_node_names and dominant_nodes:
            dominant_visible_node_names = [str(item["node_name"]) for item in dominant_nodes[: max(1, int(self.visualization_config.dominant_top_k))]]

        active_zones = sorted(
            (
                {
                    "zone": zone_name,
                    "zone_display": display_name_for_zone(zone_name),
                    "mean_activation": float(np.mean(values)),
                    "max_activation": float(np.max(values)),
                }
                for zone_name, values in zone_scores.items()
                if values
            ),
            key=lambda item: (item["mean_activation"], item["max_activation"]),
            reverse=True,
        )[:6]

        conflicts: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for edge in dynamic_edges["visible_edges"]:
            if edge["edge_type"] != "antagonist":
                continue
            source_name = str(edge["source_name"])
            target_name = str(edge["target_name"])
            pair_key = tuple(sorted((source_name, target_name)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            source_idx = self.node_index[source_name]
            target_idx = self.node_index[target_name]
            source_activation = float(activations[source_idx])
            target_activation = float(activations[target_idx])
            conflict_score = min(source_activation, target_activation) * float(edge["dynamic_score"])
            if conflict_score <= 0.05:
                continue
            conflicts.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "source_activation": source_activation,
                    "target_activation": target_activation,
                    "score": float(conflict_score),
                }
            )
        conflicts.sort(key=lambda item: item["score"], reverse=True)

        visible_edge_counts: dict[str, int] = {}
        for item in dynamic_edges["visible_edges"]:
            edge_type = str(item["edge_type"])
            visible_edge_counts[edge_type] = visible_edge_counts.get(edge_type, 0) + 1

        dominant_edge_counts: dict[str, int] = {}
        for item in dominant_visible_edges:
            edge_type = str(item["edge_type"])
            dominant_edge_counts[edge_type] = dominant_edge_counts.get(edge_type, 0) + 1

        debug_nodes = (
            [item for item in dominant_nodes if str(item["node_name"]) in set(dominant_visible_node_names)]
            if bool(self.visualization_config.dominant_mode)
            else top_active
        )
        debug_edges = dominant_visible_edges if bool(self.visualization_config.dominant_mode) else list(dynamic_edges["top_dynamic_edges"])

        return {
            "top_active": top_active,
            "dominant_nodes": dominant_nodes,
            "dominant_node_names": sorted(dominant_node_names),
            "dominant_active_node_names": dominant_active_node_names,
            "dominant_connected_node_names": dominant_connected_node_names,
            "dominant_visible_node_names": dominant_visible_node_names,
            "dominant_visible_edges": dominant_visible_edges,
            "dominant_groups": dominant_groups,
            "dominant_clusters": dominant_groups,
            "active_zones": active_zones,
            "antagonist_conflicts": conflicts[:6],
            "top_dynamic_edges": list(dynamic_edges["top_dynamic_edges"]),
            "visible_edge_counts": visible_edge_counts,
            "dominant_edge_counts": dominant_edge_counts,
            "debug_nodes": debug_nodes,
            "debug_edges": debug_edges,
            "dominant_selection_mode": dominant_selection_mode,
        }

    def _select_ranked_edges(
        self,
        *,
        ranked_edges: list[dict[str, Any]],
        threshold_edges: list[dict[str, Any]],
        max_edges: int,
        fallback_top_k: int,
    ) -> list[dict[str, Any]]:
        if not ranked_edges or max_edges <= 0:
            return []
        if not threshold_edges:
            return ranked_edges[: min(max_edges, max(1, fallback_top_k))]
        selected = list(threshold_edges[:max_edges])
        if len(selected) >= max_edges:
            return selected
        seen_pairs = {
            tuple(sorted((str(item["source_name"]), str(item["target_name"]))))
            for item in selected
        }
        for item in ranked_edges:
            pair_key = tuple(sorted((str(item["source_name"]), str(item["target_name"]))))
            if pair_key in seen_pairs:
                continue
            selected.append(item)
            seen_pairs.add(pair_key)
            if len(selected) >= max_edges:
                break
        return selected

    def _build_dynamic_edge_summary(self, *, dynamic_edges: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
        visible_edges = list(dynamic_edges.get("visible_edges") or [])
        return {
            "num_visible_edges": int(len(visible_edges)),
            "num_dominant_visible_edges": int(len(analysis.get("dominant_visible_edges") or [])),
            "visible_edge_counts": dict(analysis.get("visible_edge_counts") or {}),
            "dominant_edge_counts": dict(analysis.get("dominant_edge_counts") or {}),
            "top_dynamic_edges": list(analysis.get("top_dynamic_edges") or [])[:10],
            "dominant_nodes": list(analysis.get("dominant_nodes") or [])[:10],
            "dominant_active_node_names": list(analysis.get("dominant_active_node_names") or []),
            "dominant_connected_node_names": list(analysis.get("dominant_connected_node_names") or []),
            "dominant_visible_node_names": list(analysis.get("dominant_visible_node_names") or []),
            "active_zones": list(analysis.get("active_zones") or [])[:6],
            "dominant_groups": list(analysis.get("dominant_groups") or [])[:6],
            "dominant_clusters": list(analysis.get("dominant_clusters") or [])[:6],
        }

# ---------------------------------------------------------------------------
# Exercise Sequence
# ---------------------------------------------------------------------------
SEQUENCE_SCHEMA_VERSION = 1


class ExerciseSequenceBuffer:
    def __init__(self) -> None:
        self.active = False
        self.session_id: str | None = None
        self.exercise_name: str | None = None
        self.trial_id: str | None = None
        self.source_csv: str | None = None
        self.static_graph: dict[str, Any] | None = None
        self.frame_timestamps: list[float] = []
        self.frame_indices: list[int] = []
        self.x_seq: list[np.ndarray] = []
        self.joint_qpos_seq: list[np.ndarray] = []
        self.ctrl_raw_seq: list[np.ndarray] = []
        self.ctrl_smoothed_seq: list[np.ndarray] = []
        self.ctrl_analysis_seq: list[np.ndarray] = []
        self.ctrl_rolling_mean_seq: list[np.ndarray] = []
        self.ctrl_rolling_median_seq: list[np.ndarray] = []
        self.ctrl_summary_seq: list[dict[str, Any]] = []
        self.dynamic_edge_mask_seq: list[np.ndarray] = []
        self.dynamic_edge_weight_seq: list[np.ndarray] = []
        self.visible_edge_records_seq: list[list[dict[str, Any]]] = []
        self.dynamic_edge_summary_seq: list[dict[str, Any]] = []

    def start(
        self,
        *,
        session_id: str,
        exercise_name: str | None,
        trial_id: str | None,
        source_csv: str,
        static_graph: dict[str, Any],
    ) -> None:
        self.active = True
        self.session_id = session_id
        self.exercise_name = None if exercise_name is None else str(exercise_name)
        self.trial_id = None if trial_id is None else str(trial_id)
        self.source_csv = source_csv
        self.static_graph = static_graph
        self.frame_timestamps = []
        self.frame_indices = []
        self.x_seq = []
        self.joint_qpos_seq = []
        self.ctrl_raw_seq = []
        self.ctrl_smoothed_seq = []
        self.ctrl_analysis_seq = []
        self.ctrl_rolling_mean_seq = []
        self.ctrl_rolling_median_seq = []
        self.ctrl_summary_seq = []
        self.dynamic_edge_mask_seq = []
        self.dynamic_edge_weight_seq = []
        self.visible_edge_records_seq = []
        self.dynamic_edge_summary_seq = []

    def append(self, graph_state: dict[str, Any]) -> None:
        if not self.active:
            return
        ctrl_raw = graph_state.get("ctrl_raw")
        ctrl_smoothed = graph_state.get("ctrl_smoothed")
        ctrl_analysis = graph_state.get("ctrl_analysis")
        self.x_seq.append(np.asarray(graph_state["x"], dtype=np.float32))
        self.joint_qpos_seq.append(np.asarray(graph_state["joint_qpos"], dtype=np.float32))
        self.ctrl_raw_seq.append(np.asarray(ctrl_raw if ctrl_raw is not None else [], dtype=np.float32))
        self.ctrl_smoothed_seq.append(np.asarray(ctrl_smoothed if ctrl_smoothed is not None else [], dtype=np.float32))
        self.ctrl_analysis_seq.append(np.asarray(ctrl_analysis if ctrl_analysis is not None else [], dtype=np.float32))
        ctrl_summary = dict(graph_state.get("ctrl_summary") or {})
        ctrl_rolling_mean = ctrl_summary.get("ctrl_rolling_mean")
        ctrl_rolling_median = ctrl_summary.get("ctrl_rolling_median")
        self.ctrl_rolling_mean_seq.append(np.asarray(ctrl_rolling_mean if ctrl_rolling_mean is not None else [], dtype=np.float32))
        self.ctrl_rolling_median_seq.append(np.asarray(ctrl_rolling_median if ctrl_rolling_median is not None else [], dtype=np.float32))
        self.ctrl_summary_seq.append(ctrl_summary)
        self.dynamic_edge_mask_seq.append(np.asarray(graph_state["dynamic_edge_mask"], dtype=bool))
        self.dynamic_edge_weight_seq.append(np.asarray(graph_state["dynamic_edge_score"], dtype=np.float32))
        self.visible_edge_records_seq.append(list(graph_state.get("visible_edges") or []))
        self.dynamic_edge_summary_seq.append(dict(graph_state.get("dynamic_edge_summary") or {}))
        self.frame_timestamps.append(float(graph_state.get("timestamp") or 0.0))
        self.frame_indices.append(int(graph_state.get("frame_index") or len(self.frame_indices)))

    def has_frames(self) -> bool:
        return bool(self.x_seq)

    def build_artifact(self) -> dict[str, Any]:
        if self.static_graph is None or self.session_id is None:
            raise RuntimeError("ExerciseSequenceBuffer has not been started.")

        x_seq = np.stack(self.x_seq, axis=0) if self.x_seq else np.empty((0, 0, 0), dtype=np.float32)
        joint_qpos_seq = np.stack(self.joint_qpos_seq, axis=0) if self.joint_qpos_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_raw_seq = np.stack(self.ctrl_raw_seq, axis=0) if self.ctrl_raw_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_smoothed_seq = np.stack(self.ctrl_smoothed_seq, axis=0) if self.ctrl_smoothed_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_analysis_seq = np.stack(self.ctrl_analysis_seq, axis=0) if self.ctrl_analysis_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_rolling_mean_seq = np.stack(self.ctrl_rolling_mean_seq, axis=0) if self.ctrl_rolling_mean_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_rolling_median_seq = np.stack(self.ctrl_rolling_median_seq, axis=0) if self.ctrl_rolling_median_seq else np.empty((0, 0), dtype=np.float32)
        dynamic_edge_mask_seq = np.stack(self.dynamic_edge_mask_seq, axis=0) if self.dynamic_edge_mask_seq else np.empty((0, 0), dtype=bool)
        dynamic_edge_weight_seq = np.stack(self.dynamic_edge_weight_seq, axis=0) if self.dynamic_edge_weight_seq else np.empty((0, 0), dtype=np.float32)
        artifact: dict[str, Any] = {
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            "graph_kind": "muscle_graph_sequence",
            "session_id": self.session_id,
            "exercise_name": self.exercise_name,
            "trial_id": self.trial_id,
            "source_csv": self.source_csv,
            "num_frames": int(x_seq.shape[0]),
            "node_names": list(self.static_graph["node_names"]),
            "node_feature_names": list(self.static_graph["node_feature_names"]),
            "edge_type_names": list(self.static_graph["edge_type_names"]),
            "prior_edge_index": torch.tensor(self.static_graph["edge_index"], dtype=torch.long) if torch is not None else self.static_graph["edge_index"].tolist(),
            "prior_edge_type": torch.tensor(self.static_graph["edge_type"], dtype=torch.long) if torch is not None else self.static_graph["edge_type"].tolist(),
            "prior_edge_weight": torch.tensor(self.static_graph["edge_weight"], dtype=torch.float32) if torch is not None else self.static_graph["edge_weight"].tolist(),
            "prior_edges": list(self.static_graph["edges"]),
            "x_seq": torch.tensor(x_seq, dtype=torch.float32) if torch is not None else x_seq.tolist(),
            "joint_qpos_seq": torch.tensor(joint_qpos_seq, dtype=torch.float32) if torch is not None else joint_qpos_seq.tolist(),
            "ctrl_raw_seq": torch.tensor(ctrl_raw_seq, dtype=torch.float32) if torch is not None else ctrl_raw_seq.tolist(),
            "ctrl_smoothed_seq": torch.tensor(ctrl_smoothed_seq, dtype=torch.float32) if torch is not None else ctrl_smoothed_seq.tolist(),
            "ctrl_analysis_seq": torch.tensor(ctrl_analysis_seq, dtype=torch.float32) if torch is not None else ctrl_analysis_seq.tolist(),
            "ctrl_rolling_mean_seq": torch.tensor(ctrl_rolling_mean_seq, dtype=torch.float32) if torch is not None else ctrl_rolling_mean_seq.tolist(),
            "ctrl_rolling_median_seq": torch.tensor(ctrl_rolling_median_seq, dtype=torch.float32) if torch is not None else ctrl_rolling_median_seq.tolist(),
            "ctrl_summary_seq": self.ctrl_summary_seq,
            "dynamic_edge_mask_seq": torch.tensor(dynamic_edge_mask_seq.astype(np.int8), dtype=torch.int8) if torch is not None else dynamic_edge_mask_seq.astype(np.int8).tolist(),
            "dynamic_edge_weight_seq": torch.tensor(dynamic_edge_weight_seq, dtype=torch.float32) if torch is not None else dynamic_edge_weight_seq.tolist(),
            "visible_edge_records_seq": self.visible_edge_records_seq,
            "dynamic_edge_summary_seq": self.dynamic_edge_summary_seq,
            "joint_qpos_names": list(self.static_graph["joint_qpos_names"]),
            "timestamps": list(self.frame_timestamps),
            "frame_indices": list(self.frame_indices),
            "meta": {
                "env_id": self.static_graph["env_id"],
                "graph_kind": self.static_graph["graph_kind"],
                "dynamic_edge_config": dict(self.static_graph.get("dynamic_edge_config") or {}),
                "session_id": self.session_id,
                "exercise_name": self.exercise_name,
                "trial_id": self.trial_id,
            },
        }
        return artifact

    def stop(self) -> dict[str, Path] | None:
        if not self.active:
            return None

        self.active = False
        if not self.has_frames():
            self.session_id = None
            self.source_csv = None
            self.static_graph = None
            return None

        artifact = self.build_artifact()
        artifact_root = get_artifact_root() / "sequences"
        artifact_root.mkdir(parents=True, exist_ok=True)
        session_id = str(self.session_id)
        pt_path = artifact_root / f"exercise_{session_id}.pt"
        summary_path = artifact_root / f"exercise_{session_id}.summary.json"

        if torch is not None:
            torch.save(artifact, pt_path)

        summary = {
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            "graph_kind": "muscle_graph_sequence",
            "session_id": session_id,
            "exercise_name": self.exercise_name,
            "trial_id": self.trial_id,
            "source_csv": self.source_csv,
            "num_frames": artifact["num_frames"],
            "num_nodes": len(artifact["node_names"]),
            "num_prior_edges": int(np.asarray(self.static_graph["edge_weight"]).shape[0]),
            "mean_visible_edges": float(
                np.mean([int(np.asarray(mask, dtype=np.int8).sum()) for mask in self.dynamic_edge_mask_seq])
            )
            if self.dynamic_edge_mask_seq
            else 0.0,
            "node_feature_names": artifact["node_feature_names"],
            "joint_qpos_names": artifact["joint_qpos_names"],
            "dynamic_edge_config": artifact["meta"]["dynamic_edge_config"],
            "pt_path": str(pt_path) if torch is not None else None,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        self.session_id = None
        self.exercise_name = None
        self.trial_id = None
        self.source_csv = None
        self.static_graph = None
        self.frame_timestamps = []
        self.frame_indices = []
        self.x_seq = []
        self.joint_qpos_seq = []
        self.ctrl_raw_seq = []
        self.ctrl_smoothed_seq = []
        self.ctrl_analysis_seq = []
        self.ctrl_rolling_mean_seq = []
        self.ctrl_rolling_median_seq = []
        self.ctrl_summary_seq = []
        self.dynamic_edge_mask_seq = []
        self.dynamic_edge_weight_seq = []
        self.visible_edge_records_seq = []
        self.dynamic_edge_summary_seq = []
        return {"pt_path": pt_path, "summary_path": summary_path}

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
class RecordingManager:
    def __init__(self, env, static_graph=None, *, exercise_name: str | None = None, trial_id: str | None = None):
        self.env = env
        self.static_graph = static_graph
        self.exercise_name = None if exercise_name is None else str(exercise_name)
        self.trial_id = None if trial_id is None else str(trial_id)
        self.recording = False
        self.record_fh = None
        self.record_writer = None
        self.record_frame = 0
        self.record_path = None
        self.record_session_id = None
        self.exercise_buffer = ExerciseSequenceBuffer()

    def stop(self):
        if self.record_fh is None:
            self.recording = False
            self.record_writer = None
            return
        try:
            self.record_fh.flush()
        except Exception:
            pass
        try:
            self.record_fh.close()
        except Exception:
            pass
        self.recording = False
        self.record_fh = None
        self.record_writer = None
        saved_paths = self.exercise_buffer.stop()
        if saved_paths is not None:
            print(f"[INFO] Exercise sequence saved: {saved_paths.get('pt_path')}")
            print(f"[INFO] Exercise summary saved: {saved_paths.get('summary_path')}")
        self.record_path = None
        self.record_session_id = None
        print("[INFO] Recording stopped.")

    def start(self):
        if self.env is None:
            print("[WARN] Cannot start recording: env not created.")
            return

        os.makedirs("records", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.record_session_id = ts
        path = os.path.join("records", f"ctrl_est_{ts}.csv")
        self.record_path = path

        try:
            self.record_fh = open(path, "w", newline="")
            self.record_writer = csv.writer(self.record_fh)

            q_n = int(self.env.sim.data.qpos.shape[0])
            nu = int(self.env.sim.data.ctrl.shape[0])
            header = (
                ["t", "frame", "frame_index", "session_id", "exercise_name", "trial_id"]
                + [f"qpos_{i}" for i in range(q_n)]
                + [f"ctrl_{j}" for j in range(nu)]
                + [f"ctrl_raw_{j}" for j in range(nu)]
                + [f"ctrl_smoothed_{j}" for j in range(nu)]
                + [f"ctrl_analysis_{j}" for j in range(nu)]
            )
            self.record_writer.writerow(header)
            self.record_fh.flush()
            self.record_frame = 0
            self.recording = True
            if self.static_graph is not None:
                self.exercise_buffer.start(
                    session_id=ts,
                    exercise_name=self.exercise_name,
                    trial_id=self.trial_id,
                    source_csv=path,
                    static_graph=self.static_graph,
                )
            print(f"[INFO] Recording started: {path}")
        except Exception as e:
            print(f"[WARN] Failed to start recording: {e}")
            self.recording = False
            self.record_fh = None
            self.record_writer = None

    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

    def write_frame(self, ctrl_vec, graph_state=None):
        if not self.recording or self.record_writer is None or ctrl_vec is None:
            return

        try:
            q_vec = np.asarray(self.env.sim.data.qpos, dtype=float).reshape(-1)
            now_t = time.time()
            q_n = int(self.env.sim.data.qpos.shape[0])
            nu = int(self.env.sim.data.ctrl.shape[0])

            q_out = np.zeros(q_n, dtype=float)
            c_out = np.zeros(nu, dtype=float)
            c_raw = np.zeros(nu, dtype=float)
            c_smoothed = np.zeros(nu, dtype=float)
            c_analysis = np.zeros(nu, dtype=float)
            q_out[: min(q_n, len(q_vec))] = q_vec[: min(q_n, len(q_vec))]
            if isinstance(ctrl_vec, dict):
                ctrl_raw = np.asarray(ctrl_vec.get("ctrl_raw") if ctrl_vec.get("ctrl_raw") is not None else [], dtype=float).reshape(-1)
                ctrl_smoothed = np.asarray(ctrl_vec.get("ctrl_smoothed") if ctrl_vec.get("ctrl_smoothed") is not None else ctrl_raw, dtype=float).reshape(-1)
                ctrl_analysis = np.asarray(ctrl_vec.get("ctrl_analysis") if ctrl_vec.get("ctrl_analysis") is not None else ctrl_smoothed, dtype=float).reshape(-1)
            else:
                ctrl_analysis = np.asarray(ctrl_vec, dtype=float).reshape(-1)
                ctrl_smoothed = ctrl_analysis.copy()
                ctrl_raw = ctrl_analysis.copy()

            c_out[: min(nu, len(ctrl_analysis))] = ctrl_analysis[: min(nu, len(ctrl_analysis))]
            c_raw[: min(nu, len(ctrl_raw))] = ctrl_raw[: min(nu, len(ctrl_raw))]
            c_smoothed[: min(nu, len(ctrl_smoothed))] = ctrl_smoothed[: min(nu, len(ctrl_smoothed))]
            c_analysis[: min(nu, len(ctrl_analysis))] = ctrl_analysis[: min(nu, len(ctrl_analysis))]

            row = [
                float(now_t),
                int(self.record_frame),
                int(self.record_frame),
                str(self.record_session_id or ""),
                str(self.exercise_name or ""),
                str(self.trial_id or ""),
            ] + q_out.tolist() + c_out.tolist() + c_raw.tolist() + c_smoothed.tolist() + c_analysis.tolist()
            self.record_writer.writerow(row)
            if graph_state is not None:
                self.exercise_buffer.append(graph_state)
            self.record_frame += 1
            if (self.record_frame % 30) == 0:
                self.record_fh.flush()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Runtime Safety
# ---------------------------------------------------------------------------
AUTO_RESET_ON_TRACK_LOSS = True
AUTO_RESET_ON_INVALID_QPOS = True
TRACK_LOSS_RESET_SECONDS = 0.35
TRACK_REACQUIRE_RESET_SECONDS = 0.35
MIN_RESET_INTERVAL_SECONDS = 0.75


def update_listener_hud(listener, label, colour):
    if listener.canvas is not None:
        listener.canvas.set_hud(label, colour)


def clear_listener_outputs(listener):
    listener.latest_qpos = None
    listener.latest_ctrl = None
    listener.latest_ctrl_raw = None
    listener.latest_ctrl_smoothed = None
    listener.latest_ctrl_analysis = None
    listener.latest_ctrl_validation = None
    listener.latest_ctrl_summary = None


def reset_runtime_tracking(listener, reason, manual=False):
    listener.mapper.reset_tracking_state()
    clear_listener_outputs(listener)
    listener.last_auto_reset_time = time.time()
    if not listener.hand_present:
        listener.last_missing_time = None
    if listener.canvas is not None:
        listener.canvas.set_status(f"Reset: {reason}")
    update_listener_hud(listener, "AUTO RESET ACTIVE", (0, 255, 255))
    prefix = "manual reset" if manual else "auto-reset"
    print(f"[INFO] Tracking state {prefix}: {reason}")


def get_tracking_loss_reset_reason(listener, now):
    if (
        AUTO_RESET_ON_TRACK_LOSS
        and not listener.hand_present
        and listener.last_missing_time is not None
        and (now - float(listener.last_missing_time)) >= float(TRACK_LOSS_RESET_SECONDS)
        and (now - float(listener.last_auto_reset_time)) >= float(MIN_RESET_INTERVAL_SECONDS)
    ):
        return f"tracking lost for {(now - float(listener.last_missing_time)):.2f}s"
    return None


def get_invalid_qpos_reset_reason(q_target):
    if AUTO_RESET_ON_INVALID_QPOS and not np.all(np.isfinite(np.asarray(q_target, dtype=float))):
        return "invalid qpos detected"
    return None

# ---------------------------------------------------------------------------
# Runtime Loop
# ---------------------------------------------------------------------------
MYOSUITE_RENDER_FPS = 60
ENABLE_MYOSUITE_RENDER = True
CTRL_ENV_EPS = 0.015
CTRL_ENV_CTRL_NORM_EPS = 0.08
CTRL_ENV_SPEEDUP = 3


def unwrap_env(env):
    try:
        return env.unwrapped
    except Exception:
        return env


def create_myosuite_env(gym_module, myosuite_available, env_id, info_label, warn_label):
    if not myosuite_available:
        return None
    try:
        env = gym_module.make(env_id)
        env.reset()
        env = unwrap_env(env)
        print(f"[INFO] {info_label}")
        return env
    except Exception as e:
        print(f"[WARN] Failed to create {warn_label}: {e}")
        return None


def create_ctrl_env_context(env):
    context = {
        "env": env,
        "data_source": None,
        "steps_per_render": 1,
        "speedup": int(CTRL_ENV_SPEEDUP),
        "needs_reset": True,
    }
    if env is None:
        return context
    context["data_source"] = env.sim.data
    try:
        ctrl_dt = float(env.sim.model.opt.timestep)
        context["steps_per_render"] = max(1, int(round((1.0 / float(MYOSUITE_RENDER_FPS)) / ctrl_dt)))
    except Exception:
        context["steps_per_render"] = 5
    return context


def apply_qpos_to_view_env(env, q_target):
    qpos_view = env.sim.data.qpos
    n = min(len(qpos_view), len(q_target))
    qpos_view[:n] = q_target[:n]
    if n < len(qpos_view):
        qpos_view[n:] = 0.0


def update_main_env_ctrl_view(env, ctrl_vec):
    if ctrl_vec is None:
        return
    ctrl_arr = np.asarray(ctrl_vec, dtype=float).ravel()
    env.sim.data.ctrl[:min(len(ctrl_arr), int(env.sim.data.ctrl.shape[0]))] = ctrl_arr[:min(len(ctrl_arr), int(env.sim.data.ctrl.shape[0]))]


def _build_neutral_qpos(model):
    qpos = np.zeros(int(model.nq), dtype=float)
    for j in range(int(model.njnt)):
        jt = int(model.jnt_type[j])
        if jt not in (2, 3):
            continue
        adr = int(model.jnt_qposadr[j])
        lo = float(model.jnt_range[j, 0])
        hi = float(model.jnt_range[j, 1])
        if np.isfinite(lo) and np.isfinite(hi):
            qpos[adr] = 0.5 * (lo + hi)
    return qpos


def reset_ctrl_env_state(env_ctrl):
    if env_ctrl is None:
        return
    try:
        env_ctrl.reset()
        env_ctrl = unwrap_env(env_ctrl)
    except Exception:
        pass
    try:
        env_ctrl.sim.data.qpos[:] = _build_neutral_qpos(env_ctrl.sim.model)
        env_ctrl.sim.data.qvel[:] = 0.0
        env_ctrl.sim.data.qacc[:] = 0.0
        env_ctrl.sim.data.ctrl[:] = 0.0
        if hasattr(env_ctrl.sim.data, "act") and len(env_ctrl.sim.data.act) > 0:
            env_ctrl.sim.data.act[:] = 0.0
        env_ctrl.sim.forward()
    except Exception:
        pass


def apply_ctrl_to_env(env_ctrl, ctrl_vec):
    if env_ctrl is None:
        return np.zeros(0, dtype=float)
    if ctrl_vec is None:
        env_ctrl.sim.data.ctrl[:] = 0.0
        return np.zeros(int(env_ctrl.sim.data.ctrl.shape[0]), dtype=float)
    c_arr = np.asarray(ctrl_vec, dtype=float).ravel().copy()
    c_arr[np.abs(c_arr) < float(CTRL_ENV_EPS)] = 0.0
    env_ctrl.sim.data.ctrl[:min(len(c_arr), int(env_ctrl.sim.data.ctrl.shape[0]))] = c_arr[:min(len(c_arr), int(env_ctrl.sim.data.ctrl.shape[0]))]
    return c_arr


def print_ctrl_summary(ctrl_state):
    if isinstance(ctrl_state, dict):
        ctrl_raw_value = ctrl_state.get("ctrl_raw")
        ctrl_smoothed_value = ctrl_state.get("ctrl_smoothed")
        ctrl_analysis_value = ctrl_state.get("ctrl_analysis")
        ctrl_raw = np.asarray(ctrl_raw_value if ctrl_raw_value is not None else [], dtype=float)
        ctrl_smoothed = np.asarray(ctrl_smoothed_value if ctrl_smoothed_value is not None else ctrl_raw, dtype=float)
        ctrl_analysis = np.asarray(ctrl_analysis_value if ctrl_analysis_value is not None else ctrl_smoothed, dtype=float)
        ref = ctrl_analysis if ctrl_analysis.size else (ctrl_smoothed if ctrl_smoothed.size else ctrl_raw)
        if ref.size == 0:
            print("[CTRL] empty ctrl state")
            return
        active = int(np.sum(ref > 0.01))
        top5_idx = np.argsort(ref)[-5:][::-1]
        top5_str = " ".join([f"m{i}={ref[i]:.2f}" for i in top5_idx])
        flags = []
        if ctrl_state.get("deadband_applied"):
            flags.append("deadband")
        if ctrl_state.get("hold_applied"):
            flags.append("hold")
        if ctrl_state.get("freeze_applied"):
            flags.append("freeze")
        if ctrl_state.get("recomputed"):
            flags.append("recompute")
        flag_str = ",".join(flags) if flags else "live"
        print(
            "[CTRL] "
            f"mode={ctrl_state.get('mode', 'control')} flags={flag_str} "
            f"qdn={float(ctrl_state.get('qpos_delta_norm', 0.0)):.4f} "
            f"raw_dn={float(ctrl_state.get('ctrl_raw_delta_norm', 0.0)):.4f} "
            f"smooth_dn={float(ctrl_state.get('ctrl_smoothed_delta_norm', 0.0)):.4f} "
            f"analysis_dn={float(ctrl_state.get('ctrl_analysis_delta_norm', 0.0)):.4f} "
            f"act={active}/{len(ref)} mean={ref.mean():.3f} max={ref.max():.3f} | top5: {top5_str}"
        )
        if bool(ctrl_state.get("is_hold_mode")):
            hold_top = list(ctrl_state.get("hold_top_flexors") or [])
            hold_top_str = " ".join([f"{str(name)}={float(value):.2f}" for name, value in hold_top])
            hold_values = dict(ctrl_state.get("hold_flexor_values") or {})
            ordered_names = ("FDS2", "FDS3", "FDS4", "FDS5", "FDP2", "FDP3", "FDP4", "FDP5")
            hold_values_str = " ".join([f"{name}={float(hold_values.get(name, 0.0)):.2f}" for name in ordered_names])
            print(
                "[CTRL][HOLD] "
                f"prior={str(ctrl_state.get('hold_prior_label', 'none'))} "
                f"fist_hold_prior={int(bool(ctrl_state.get('fist_hold_prior_active')))} "
                f"top_flexors={hold_top_str or '-'} | {hold_values_str}"
            )
            index_values = dict(ctrl_state.get("index_hold_values") or {})
            print(
                "[CTRL][INDEX_HOLD] "
                f"active={int(bool(ctrl_state.get('index_hold_prior_active')))} "
                f"FDS2={float(index_values.get('FDS2', 0.0)):.2f} "
                f"FDP2={float(index_values.get('FDP2', 0.0)):.2f} "
                f"EIP={float(index_values.get('EIP', 0.0)):.2f} "
                f"EDC2={float(index_values.get('EDC2', 0.0)):.2f}"
            )
        return

    c = np.asarray(ctrl_state, dtype=float)
    active = int(np.sum(c > 0.01))
    top5_idx = np.argsort(c)[-5:][::-1]
    top5_str = " ".join([f"m{i}={c[i]:.2f}" for i in top5_idx])
    print(f"[CTRL] act={active}/{len(c)} mean={c.mean():.3f} max={c.max():.3f} | top5: {top5_str}")
def step_ctrl_env(env_ctrl, ctrl_steps_per_render, ctrl_vec, render=True):
    if env_ctrl is None:
        return
    c_arr = apply_ctrl_to_env(env_ctrl, ctrl_vec)
    ctrl_norm = float(np.linalg.norm(c_arr)) if c_arr.size else 0.0
    if ctrl_norm <= float(CTRL_ENV_CTRL_NORM_EPS):
        env_ctrl.sim.data.ctrl[:] = 0.0
    for _ in range(int(ctrl_steps_per_render)):
        env_ctrl.sim.advance()
    env_ctrl.sim.forward()
    if render:
        try:
            env_ctrl.mj_render()
        except Exception:
            pass


def handle_runtime_key(key, recorder, listener):
    if key == ord("x"):
        print("[INFO] Exit requested via key 'x'.")
        return False
    if key == ord("0"):
        reset_runtime_tracking(listener, "manual reset", manual=True)
        return True
    if key == ord("r"):
        recorder.toggle()
        return True
    return True

# ---------------------------------------------------------------------------
# Tracking Geometry
# ---------------------------------------------------------------------------
def vector_to_np(v, desktop_mode_transform=True):
    if desktop_mode_transform:
        return np.array([v.x, v.z, -v.y])
    return np.array([v.x, v.y, v.z])


def quaternion_to_yaw(q):
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def direction(a, b, desktop_mode_transform=True):
    v = vector_to_np(b, desktop_mode_transform) - vector_to_np(a, desktop_mode_transform)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)


def vector_angle_rad(v1, v2):
    v1 = unit(np.array(v1))
    v2 = unit(np.array(v2))
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.acos(dot)


def vector_angle_deg(v1, v2):
    v1 = unit(np.array(v1))
    v2 = unit(np.array(v2))
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def extract_palm_basis_from_hand(hand, desktop_mode_transform=True):
    dir_v = vector_to_np(hand.palm.direction, desktop_mode_transform)
    nrm_v = vector_to_np(hand.palm.normal, desktop_mode_transform)
    z = unit(dir_v)
    y = unit(nrm_v - np.dot(nrm_v, z) * z)
    x = unit(np.cross(y, z))
    return np.stack([x, y, z], axis=1)


def leap_hand_to_joint_angles(hand, desktop_mode_transform=True):
    angles = []
    for idx in [1, 2, 3, 4]:
        digit = hand.digits[idx]
        mcp = vector_angle_deg(
            vector_to_np(digit.metacarpal.next_joint, desktop_mode_transform) - vector_to_np(digit.metacarpal.prev_joint, desktop_mode_transform),
            vector_to_np(digit.proximal.next_joint, desktop_mode_transform) - vector_to_np(digit.proximal.prev_joint, desktop_mode_transform),
        )
        pip = vector_angle_deg(
            vector_to_np(digit.proximal.next_joint, desktop_mode_transform) - vector_to_np(digit.proximal.prev_joint, desktop_mode_transform),
            vector_to_np(digit.intermediate.next_joint, desktop_mode_transform) - vector_to_np(digit.intermediate.prev_joint, desktop_mode_transform),
        )
        dip = vector_angle_deg(
            vector_to_np(digit.intermediate.next_joint, desktop_mode_transform) - vector_to_np(digit.intermediate.prev_joint, desktop_mode_transform),
            vector_to_np(digit.distal.next_joint, desktop_mode_transform) - vector_to_np(digit.distal.prev_joint, desktop_mode_transform),
        )
        angles.extend([mcp, pip, dip])

    thumb = hand.digits[0]
    thumb_mcp = vector_angle_deg(
        vector_to_np(thumb.metacarpal.next_joint, desktop_mode_transform) - vector_to_np(thumb.metacarpal.prev_joint, desktop_mode_transform),
        vector_to_np(thumb.proximal.next_joint, desktop_mode_transform) - vector_to_np(thumb.proximal.prev_joint, desktop_mode_transform),
    )
    thumb_ip = vector_angle_deg(
        vector_to_np(thumb.proximal.next_joint, desktop_mode_transform) - vector_to_np(thumb.proximal.prev_joint, desktop_mode_transform),
        vector_to_np(thumb.distal.next_joint, desktop_mode_transform) - vector_to_np(thumb.distal.prev_joint, desktop_mode_transform),
    )
    angles.extend([thumb_mcp, thumb_ip, 0.0])
    return np.array(angles)

# ---------------------------------------------------------------------------
# Tracking State
# ---------------------------------------------------------------------------
MCP_ABD_BASE_DEFAULT = {
    1: 0.05095816096908771,
    2: -0.0006556564329452622,
    3: 0.0828368594413166,
    4: 0.08490408878999552,
}
THUMB_CMC_ABD0_RAD_DEFAULT = -0.3568619307393395
THUMB_CMC_FLEX0_RAW_DEFAULT = 1.5707963267948966


@dataclass
class RealtimeTrackingState:
    prev_yaw: float | None = None
    prev_qpos: np.ndarray | None = None
    wrist_prev_right: np.ndarray | None = None
    thumb_prev_normal: np.ndarray | None = None
    thumb_prev_right: np.ndarray | None = None
    calib_qpos: np.ndarray | None = None
    calib_index_signature: tuple[float, float, float] | None = None
    mcp_abd_base: dict = field(default_factory=lambda: MCP_ABD_BASE_DEFAULT.copy())
    thumb_cmc_abd0_rad: float = float(THUMB_CMC_ABD0_RAD_DEFAULT)
    thumb_cmc_flex0_raw: float = float(THUMB_CMC_FLEX0_RAW_DEFAULT)

    def reset(self):
        self.prev_yaw = None
        self.prev_qpos = None
        self.wrist_prev_right = None
        self.thumb_prev_normal = None
        self.thumb_prev_right = None
        self.calib_qpos = None
        self.calib_index_signature = None
        self.mcp_abd_base = MCP_ABD_BASE_DEFAULT.copy()
        self.thumb_cmc_abd0_rad = float(THUMB_CMC_ABD0_RAD_DEFAULT)
        self.thumb_cmc_flex0_raw = float(THUMB_CMC_FLEX0_RAW_DEFAULT)

# ---------------------------------------------------------------------------
# Tracking Mapper
# ---------------------------------------------------------------------------
QPOS_SIZE = 23

QPOS_SMOOTH_ALPHA = 0.35
INDEX_QPOS_SMOOTH_ALPHA = 0.50
QPOS_MAX_STEP_DEFAULT = 0.25
QPOS_MAX_STEP_YAW = 0.35
QPOS_MAX_STEP_WRIST_DEV = 0.12
QPOS_MAX_STEP_WRIST_FLEX = 0.30
QPOS_MAX_STEP_THUMB_CMC = 0.30
QPOS_MAX_STEP_MCP_ABD = 0.10
FINGER_FLEX_DEADZONE_RAD = 0.01
JOINT_ANGLE_OVERRIDE_BLEND = 0.70
MCP_ABD_LIMIT = 0.22
MCP_ABD_BASE_ADAPT_ALPHA = 0.08
MCP_ABD_BASE_OPEN_MCP_MAX = 0.45
MCP_ABD_BASE_OPEN_PIP_MAX = 0.35
MCP_ABD_BASE_OPEN_DIP_MAX = 0.25

WRIST_YAW_IDX = 0
WRIST_DEV_IDX = 1
WRIST_FLEX_IDX = 2

THUMB_CMC_ABD_IDX = 3
THUMB_CMC_FLEX_IDX = 4
THUMB_MCP_IDX = 5
THUMB_IP_IDX = 6

FINGER_BASE_IDX = {
    "index": 7,
    "middle": 11,
    "ring": 15,
    "little": 19,
}

FINGER_CONFIGS = (
    {"digit_index": 1, "name": "index", "flex_key": "index_flex_gain", "abd_key": "index_abd_gain", "base_idx": FINGER_BASE_IDX["index"]},
    {"digit_index": 2, "name": "middle", "flex_key": "middle_flex_gain", "abd_key": "middle_abd_gain", "base_idx": FINGER_BASE_IDX["middle"]},
    {"digit_index": 3, "name": "ring", "flex_key": "ring_flex_gain", "abd_key": "ring_abd_gain", "base_idx": FINGER_BASE_IDX["ring"]},
    {"digit_index": 4, "name": "little", "flex_key": "little_flex_gain", "abd_key": "little_abd_gain", "base_idx": FINGER_BASE_IDX["little"]},
)

FLEXION_JOINTS = (
    FINGER_BASE_IDX["index"],
    FINGER_BASE_IDX["index"] + 2,
    FINGER_BASE_IDX["index"] + 3,
    FINGER_BASE_IDX["middle"],
    FINGER_BASE_IDX["middle"] + 2,
    FINGER_BASE_IDX["middle"] + 3,
    FINGER_BASE_IDX["ring"],
    FINGER_BASE_IDX["ring"] + 2,
    FINGER_BASE_IDX["ring"] + 3,
    FINGER_BASE_IDX["little"],
    FINGER_BASE_IDX["little"] + 2,
    FINGER_BASE_IDX["little"] + 3,
)

INDEX_FLEXION_JOINTS = (
    FINGER_BASE_IDX["index"],
    FINGER_BASE_IDX["index"] + 2,
    FINGER_BASE_IDX["index"] + 3,
)

OVERRIDE_JOINTS = {
    FINGER_BASE_IDX["index"]: 0,
    FINGER_BASE_IDX["index"] + 2: 1,
    FINGER_BASE_IDX["index"] + 3: 2,
    FINGER_BASE_IDX["middle"]: 3,
    FINGER_BASE_IDX["middle"] + 2: 4,
    FINGER_BASE_IDX["middle"] + 3: 5,
    FINGER_BASE_IDX["ring"]: 6,
    FINGER_BASE_IDX["ring"] + 2: 7,
    FINGER_BASE_IDX["ring"] + 3: 8,
    FINGER_BASE_IDX["little"]: 9,
    FINGER_BASE_IDX["little"] + 2: 10,
    FINGER_BASE_IDX["little"] + 3: 11,
}

JOINT_TUNING = {
    "wrist_flex_gain": 1.2,
    "wrist_dev_gain": 0.35,
    "thumb_cmc_abd_gain": 1.05,
    "thumb_cmc_flex_gain": 1.72,
    "thumb_mcp_gain": 1.32,
    "thumb_ip_gain": 1.48,
    "mcp_abd_gain": 0.9,
    "mcp_abd_flex_suppress": 0.75,
    "index_flex_gain": 1.05,
    "index_pip_gain": 1.28,
    "index_pip_from_mcp_gain": 0.20,
    "index_dip_gain": 1.45,
    "index_dip_from_pip_gain": 0.45,
    "index_dip_from_mcp_gain": 0.10,
    "middle_flex_gain": 1.0,
    "middle_pip_gain": 1.05,
    "middle_pip_from_mcp_gain": 0.16,
    "middle_dip_gain": 1.15,
    "middle_dip_from_pip_gain": 0.50,
    "middle_dip_from_mcp_gain": 0.12,
    "ring_flex_gain": 1.0,
    "ring_pip_gain": 1.05,
    "ring_pip_from_mcp_gain": 0.16,
    "ring_dip_gain": 1.15,
    "ring_dip_from_pip_gain": 0.50,
    "ring_dip_from_mcp_gain": 0.12,
    # Keep little finger responsive, but reduce passive curl when the hand is open.
    "little_flex_gain": 1.02,
    "little_pip_gain": 1.18,
    "little_pip_from_mcp_gain": 0.24,
    "little_dip_gain": 1.32,
    "little_dip_from_pip_gain": 0.60,
    "little_dip_from_mcp_gain": 0.10,
    "finger_pip_gain": 1.15,
    "finger_pip_from_mcp_gain": 0.20,
    "finger_dip_gain": 1.30,
    "finger_dip_from_pip_gain": 0.55,
    "finger_dip_from_mcp_gain": 0.15,
    "index_abd_gain": 1.0,
    "middle_abd_gain": 1.0,
    "ring_abd_gain": 1.0,
    "little_abd_gain": 1.0,
}

TRACKING_TUNING_PRESETS = {
    "none": {},
    "grasp": {
        "qpos_deadband_norm": 0.01,
        "ctrl_smoothing_alpha": 0.25,
        "hold_if_small_change": False,
        "mapping_debug": True,
    },
    "index_grasp": {
        "qpos_deadband_norm": 0.01,
        "ctrl_smoothing_alpha": 0.25,
        "hold_if_small_change": False,
        "mapping_debug": True,
    },
    "thumb_opposition": {
        "qpos_deadband_norm": 0.01,
        "ctrl_smoothing_alpha": 0.25,
        "hold_if_small_change": False,
        "mapping_debug": True,
    },
    "pinch": {
        "qpos_deadband_norm": 0.01,
        "ctrl_smoothing_alpha": 0.25,
        "hold_if_small_change": False,
        "mapping_debug": True,
    },
}
DEFAULT_TRACKING_TUNING_PRESET = "none"
DEFAULT_HAND_MAPPING_DEBUG_INTERVAL = 0.35
DEFAULT_HAND_MAPPING_DEBUG_SECONDS = 8.0


@dataclass
class MapperDebugSnapshot:
    stage: str = "idle"
    wrist: dict = field(default_factory=dict)
    thumb: dict = field(default_factory=dict)
    fingers: dict = field(default_factory=dict)
    qpos_raw: np.ndarray | None = None
    qpos_output: np.ndarray | None = None
    used_previous_qpos: bool = False
    smoothing_applied: bool = False


class LeapToMyoSuiteMapper:
    def __init__(self, desktop_mode_transform=True, desktop_yaw_offset=-2.040):
        self.desktop_mode_transform = bool(desktop_mode_transform)
        self.desktop_yaw_offset = desktop_yaw_offset
        self.state = RealtimeTrackingState()
        self.last_debug = MapperDebugSnapshot()

    def reset_tracking_state(self):
        self.state.reset()
        self.last_debug = MapperDebugSnapshot(stage="reset")

    def get_last_debug_snapshot(self):
        return self.last_debug

    def _vector_to_np(self, v):
        return vector_to_np(v, self.desktop_mode_transform)

    def _bone_vector(self, bone):
        return self._vector_to_np(bone.next_joint) - self._vector_to_np(bone.prev_joint)

    def _proj_to_plane(self, v, n):
        n = unit(n)
        return v - np.dot(v, n) * n

    def _signed_angle_on_plane(self, a, b, n):
        a_u = unit(a)
        b_u = unit(b)
        n_u = unit(n)
        x = float(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
        y = float(np.dot(n_u, np.cross(a_u, b_u)))
        return math.atan2(y, x)

    def _stabilize_axis(self, current_axis, previous_axis):
        axis = unit(current_axis)
        if previous_axis is not None and float(np.dot(previous_axis, axis)) < 0.0:
            axis = -axis
        return axis

    def _unwrap_yaw(self, yaw):
        if self.state.prev_yaw is not None:
            while yaw - self.state.prev_yaw > math.pi:
                yaw -= 2 * math.pi
            while yaw - self.state.prev_yaw < -math.pi:
                yaw += 2 * math.pi
        self.state.prev_yaw = yaw
        return yaw

    def _is_left_hand(self, hand):
        if hand is None:
            return False
        is_left = getattr(hand, "is_left", None)
        if is_left is not None:
            try:
                return bool(is_left)
            except Exception:
                pass
        hand_type = getattr(hand, "type", None)
        if hand_type is not None:
            try:
                return "left" in str(hand_type).strip().lower()
            except Exception:
                pass
        return False

    def _map_wrist_yaw(self, arm):
        if arm is None or not getattr(arm, "rotation", None):
            return 0.0
        yaw = quaternion_to_yaw(arm.rotation)
        if self.desktop_mode_transform and self.desktop_yaw_offset is not None:
            yaw -= self.desktop_yaw_offset
        return float(np.clip(self._unwrap_yaw(yaw), -2.5, 2.5))

    def _map_wrist_deviation(self, hand, palm_normal, palm_direction):
        deviation_angle = math.atan2(palm_normal[0], palm_normal[1])
        have_arm = (
            hasattr(hand, "arm")
            and getattr(hand.arm, "prev_joint", None) is not None
            and getattr(hand.arm, "next_joint", None) is not None
        )
        if have_arm:
            arm_dir = direction(hand.arm.prev_joint, hand.arm.next_joint, self.desktop_mode_transform)
            arm_proj = self._proj_to_plane(arm_dir, palm_normal)
            palm_proj = self._proj_to_plane(palm_direction, palm_normal)
            if np.linalg.norm(arm_proj) > 1e-6 and np.linalg.norm(palm_proj) > 1e-6:
                deviation_angle = self._signed_angle_on_plane(arm_proj, palm_proj, palm_normal)

        dev_gain = float(JOINT_TUNING.get("wrist_dev_gain", 0.35))
        handedness_sign = 1.0 if self._is_left_hand(hand) else -1.0
        return float(np.clip(handedness_sign * deviation_angle * dev_gain, -0.174533, 0.436332))

    def _map_wrist_flexion(self, hand, palm_right, palm_direction):
        wf_angle = 0.0
        have_arm = (
            hasattr(hand, "arm")
            and getattr(hand.arm, "prev_joint", None) is not None
            and getattr(hand.arm, "next_joint", None) is not None
        )
        if have_arm:
            arm_dir = direction(hand.arm.prev_joint, hand.arm.next_joint, self.desktop_mode_transform)
            arm_proj = self._proj_to_plane(arm_dir, palm_right)
            palm_proj = self._proj_to_plane(palm_direction, palm_right)
            if np.linalg.norm(arm_proj) > 1e-6 and np.linalg.norm(palm_proj) > 1e-6:
                wf_angle = self._signed_angle_on_plane(arm_proj, palm_proj, palm_right)
        else:
            wf_angle = math.asin(float(np.clip(unit(palm_direction)[1], -1.0, 1.0)))
        gain = float(JOINT_TUNING.get("wrist_flex_gain", 0.9))
        return float(np.clip(-gain * wf_angle, -0.785398, 0.785398)), wf_angle

    def _map_wrist(self, hand, arm, basis, qpos):
        palm_right = self._stabilize_axis(basis[:, 0], self.state.wrist_prev_right)
        palm_normal = unit(basis[:, 1])
        palm_direction = unit(basis[:, 2])
        self.state.wrist_prev_right = palm_right

        qpos[WRIST_YAW_IDX] = self._map_wrist_yaw(arm)
        qpos[WRIST_DEV_IDX] = self._map_wrist_deviation(hand, palm_normal, palm_direction)
        qpos[WRIST_FLEX_IDX], raw_flex = self._map_wrist_flexion(hand, palm_right, palm_direction)

        self.last_debug.wrist = {
            "yaw": float(qpos[WRIST_YAW_IDX]),
            "deviation": float(qpos[WRIST_DEV_IDX]),
            "flexion": float(qpos[WRIST_FLEX_IDX]),
            "raw_flex_angle": float(raw_flex),
        }
        return palm_right, palm_normal, palm_direction

    def _map_thumb_axes(self, palm_right, palm_normal):
        n_palm = self._stabilize_axis(palm_normal, self.state.thumb_prev_normal)
        r_palm = self._stabilize_axis(palm_right, self.state.thumb_prev_right)
        self.state.thumb_prev_normal = n_palm
        self.state.thumb_prev_right = r_palm
        return r_palm, n_palm

    def _map_thumb_cmc_abduction(self, thumb_proximal, r_palm):
        radial = float(np.clip(np.dot(unit(thumb_proximal), r_palm), -1.0, 1.0))
        gain = float(JOINT_TUNING.get("thumb_cmc_abd_gain", 0.95))
        value = float(np.clip(-(radial - self.state.thumb_cmc_abd0_rad) * gain, -1.0, 1.0))
        return value, radial

    def _map_thumb_cmc_flexion(self, thumb_metacarpal, thumb_proximal):
        flex_raw = vector_angle_rad(thumb_metacarpal, thumb_proximal)
        gain = float(JOINT_TUNING.get("thumb_cmc_flex_gain", 1.55))
        value = float(np.clip(-(flex_raw - self.state.thumb_cmc_flex0_raw) * gain, -0.78, 0.85))
        return value, flex_raw

    def _map_thumb_mcp(self, palm_direction, thumb_proximal, r_palm):
        ref_flex = self._proj_to_plane(palm_direction, r_palm)
        thumb_proj = self._proj_to_plane(thumb_proximal, r_palm)
        gain = float(JOINT_TUNING.get("thumb_mcp_gain", 1.20))
        value = self._signed_angle_on_plane(ref_flex, thumb_proj, r_palm) * gain
        return float(np.clip(value, -0.698132, 0.698132))

    def _map_thumb_ip(self, thumb_proximal, thumb_distal, r_palm):
        prox_proj = self._proj_to_plane(thumb_proximal, r_palm)
        distal_proj = self._proj_to_plane(thumb_distal, r_palm)
        gain = float(JOINT_TUNING.get("thumb_ip_gain", 1.35))
        value = self._signed_angle_on_plane(prox_proj, distal_proj, r_palm) * gain
        return float(np.clip(value, -1.309, 0.436332))

    def _map_thumb(self, hand, palm_right, palm_normal, palm_direction, qpos):
        thumb = hand.digits[0]
        thumb_metacarpal = self._bone_vector(thumb.metacarpal)
        thumb_proximal = self._bone_vector(thumb.proximal)
        thumb_distal = self._bone_vector(thumb.distal)

        r_palm, n_palm = self._map_thumb_axes(palm_right, palm_normal)
        qpos[THUMB_CMC_ABD_IDX], radial = self._map_thumb_cmc_abduction(thumb_proximal, r_palm)
        qpos[THUMB_CMC_FLEX_IDX], flex_raw = self._map_thumb_cmc_flexion(thumb_metacarpal, thumb_proximal)
        qpos[THUMB_MCP_IDX] = self._map_thumb_mcp(palm_direction, thumb_proximal, r_palm)
        qpos[THUMB_IP_IDX] = self._map_thumb_ip(thumb_proximal, thumb_distal, r_palm)

        self.last_debug.thumb = {
            "cmc_abduction": float(qpos[THUMB_CMC_ABD_IDX]),
            "cmc_flexion": float(qpos[THUMB_CMC_FLEX_IDX]),
            "mcp_flexion": float(qpos[THUMB_MCP_IDX]),
            "ip_flexion": float(qpos[THUMB_IP_IDX]),
            "radial_projection": float(radial),
            "flex_raw": float(flex_raw),
            "cmc_abd_gain": float(JOINT_TUNING.get("thumb_cmc_abd_gain", 0.95)),
            "cmc_flex_gain": float(JOINT_TUNING.get("thumb_cmc_flex_gain", 1.55)),
            "mcp_gain": float(JOINT_TUNING.get("thumb_mcp_gain", 1.20)),
            "ip_gain": float(JOINT_TUNING.get("thumb_ip_gain", 1.35)),
        }
        return r_palm

    def _map_single_finger(self, digit, finger_config, r_palm, qpos):
        base = int(finger_config["base_idx"])
        metacarpal = self._bone_vector(digit.metacarpal)
        proximal = self._bone_vector(digit.proximal)
        intermediate = self._bone_vector(digit.intermediate)
        distal = self._bone_vector(digit.distal)

        mcp_flex = vector_angle_rad(metacarpal, proximal)
        pip_flex = vector_angle_rad(proximal, intermediate)
        dip_flex = vector_angle_rad(intermediate, distal)

        mcp_out = float(np.clip(mcp_flex * float(JOINT_TUNING.get(finger_config["flex_key"], 1.0)), 0.0, 1.5708))
        finger_name = str(finger_config["name"])
        pip_gain = float(JOINT_TUNING.get(f"{finger_name}_pip_gain", JOINT_TUNING.get("finger_pip_gain", 1.15)))
        pip_from_mcp_gain = float(
            JOINT_TUNING.get(
                f"{finger_name}_pip_from_mcp_gain",
                JOINT_TUNING.get("finger_pip_from_mcp_gain", 0.20),
            )
        )
        dip_gain = float(JOINT_TUNING.get(f"{finger_name}_dip_gain", JOINT_TUNING.get("finger_dip_gain", 1.30)))
        dip_from_pip_gain = float(
            JOINT_TUNING.get(f"{finger_name}_dip_from_pip_gain", JOINT_TUNING.get("finger_dip_from_pip_gain", 0.55))
        )
        dip_from_mcp_gain = float(
            JOINT_TUNING.get(f"{finger_name}_dip_from_mcp_gain", JOINT_TUNING.get("finger_dip_from_mcp_gain", 0.15))
        )
        pip_direct = float(pip_flex * pip_gain)
        pip_coupled = float(pip_from_mcp_gain * mcp_out)
        pip_out = float(np.clip(max(pip_direct, pip_coupled), 0.0, 1.5708))
        dip_direct = float(dip_flex * dip_gain)
        dip_coupled = float(
            dip_from_pip_gain * pip_out
            + dip_from_mcp_gain * mcp_out
        )
        dip_out = float(np.clip(max(dip_direct, dip_coupled), 0.0, 1.5708))

        qpos[base + 0] = mcp_out

        digit_index = int(finger_config["digit_index"])
        radial = float(np.clip(np.dot(unit(proximal), r_palm), -1.0, 1.0))
        if digit_index not in self.state.mcp_abd_base:
            self.state.mcp_abd_base[digit_index] = radial
        elif (
            float(mcp_flex) <= float(MCP_ABD_BASE_OPEN_MCP_MAX)
            and float(pip_flex) <= float(MCP_ABD_BASE_OPEN_PIP_MAX)
            and float(dip_flex) <= float(MCP_ABD_BASE_OPEN_DIP_MAX)
        ):
            alpha = float(np.clip(MCP_ABD_BASE_ADAPT_ALPHA, 0.0, 1.0))
            prev_base = float(self.state.mcp_abd_base.get(digit_index, radial))
            self.state.mcp_abd_base[digit_index] = (1.0 - alpha) * prev_base + alpha * radial

        gain_abd = float(JOINT_TUNING.get("mcp_abd_gain", 1.2)) * float(JOINT_TUNING.get(finger_config["abd_key"], 1.0))
        mcp_abd = float(np.clip(-(radial - self.state.mcp_abd_base.get(digit_index, radial)) * gain_abd, -MCP_ABD_LIMIT, MCP_ABD_LIMIT))
        flex_suppress = float(JOINT_TUNING.get("mcp_abd_flex_suppress", 0.4))
        curl_level = float(
            np.clip(
                (0.45 * float(mcp_flex) + 0.35 * float(pip_flex) + 0.20 * float(dip_flex)) / 1.1,
                0.0,
                1.0,
            )
        )
        qpos[base + 1] = mcp_abd * max(0.0, 1.0 - flex_suppress * curl_level)
        qpos[base + 2] = pip_out
        qpos[base + 3] = dip_out

        self.last_debug.fingers[finger_config["name"]] = {
            "mcp_flex_raw": float(mcp_flex),
            "mcp_flex": float(qpos[base + 0]),
            "mcp_abd": float(qpos[base + 1]),
            "pip_flex_raw": float(pip_flex),
            "pip_direct": float(pip_direct),
            "pip_coupled": float(pip_coupled),
            "pip_flex": float(qpos[base + 2]),
            "dip_flex_raw": float(dip_flex),
            "dip_direct": float(dip_direct),
            "dip_coupled": float(dip_coupled),
            "dip_flex": float(qpos[base + 3]),
            "pip_gain": float(pip_gain),
            "pip_from_mcp_gain": float(pip_from_mcp_gain),
            "dip_gain": float(dip_gain),
            "dip_from_pip_gain": float(dip_from_pip_gain),
            "dip_from_mcp_gain": float(dip_from_mcp_gain),
            "radial_projection": float(radial),
        }

    def _map_fingers(self, hand, r_palm, qpos):
        self.last_debug.fingers = {}
        for finger_config in FINGER_CONFIGS:
            digit = hand.digits[int(finger_config["digit_index"])]
            self._map_single_finger(digit, finger_config, r_palm, qpos)

    def _apply_joint_angle_overrides(self, hand, qpos):
        if not self.desktop_mode_transform:
            return
        try:
            joint_angles = leap_hand_to_joint_angles(hand, self.desktop_mode_transform)
            if joint_angles is None or len(joint_angles) < 15:
                return
            blend = float(np.clip(JOINT_ANGLE_OVERRIDE_BLEND, 0.0, 1.0))
            override_values = {
                THUMB_IP_IDX: float(
                    np.clip(
                        -math.radians(joint_angles[13] - 14.3) * float(JOINT_TUNING.get("thumb_ip_gain", 1.35)),
                        -1.309,
                        0.436332,
                    )
                )
            }
            finger_override_specs = (
                ("index", 0, 1, 2),
                ("middle", 3, 4, 5),
                ("ring", 6, 7, 8),
                ("little", 9, 10, 11),
            )
            for finger_name, mcp_idx, pip_idx, dip_idx in finger_override_specs:
                base = int(FINGER_BASE_IDX[finger_name])
                mcp_override = float(np.clip(math.radians(joint_angles[mcp_idx]), 0.0, 1.5708))
                pip_gain = float(
                    JOINT_TUNING.get(
                        f"{finger_name}_pip_gain",
                        JOINT_TUNING.get("finger_pip_gain", 1.15),
                    )
                )
                pip_from_mcp_gain = float(
                    JOINT_TUNING.get(
                        f"{finger_name}_pip_from_mcp_gain",
                        JOINT_TUNING.get("finger_pip_from_mcp_gain", 0.20),
                    )
                )
                dip_gain = float(
                    JOINT_TUNING.get(
                        f"{finger_name}_dip_gain",
                        JOINT_TUNING.get("finger_dip_gain", 1.30),
                    )
                )
                dip_from_pip_gain = float(
                    JOINT_TUNING.get(
                        f"{finger_name}_dip_from_pip_gain",
                        JOINT_TUNING.get("finger_dip_from_pip_gain", 0.55),
                    )
                )
                dip_from_mcp_gain = float(
                    JOINT_TUNING.get(
                        f"{finger_name}_dip_from_mcp_gain",
                        JOINT_TUNING.get("finger_dip_from_mcp_gain", 0.15),
                    )
                )
                pip_override = float(
                    np.clip(
                        max(
                            math.radians(joint_angles[pip_idx]) * pip_gain,
                            pip_from_mcp_gain * mcp_override,
                        ),
                        0.0,
                        1.5708,
                    )
                )
                dip_direct = float(math.radians(joint_angles[dip_idx]) * dip_gain)
                dip_coupled = float(
                    dip_from_pip_gain * pip_override
                    + dip_from_mcp_gain * mcp_override
                )
                override_values[base + 0] = mcp_override
                override_values[base + 2] = pip_override
                override_values[base + 3] = float(np.clip(max(dip_direct, dip_coupled), 0.0, 1.5708))
            for idx, value in override_values.items():
                if 0 <= idx < int(qpos.shape[0]) and np.isfinite(value):
                    qpos[idx] = (1.0 - blend) * float(qpos[idx]) + blend * float(value)
        except Exception:
            pass

    def _apply_calibration_correction(self, qpos):
        calib = self.state.calib_qpos
        if calib is None:
            return
        try:
            calib = np.asarray(calib, dtype=float)
            index_signature = None
            if int(calib.shape[0]) > int(INDEX_FLEXION_JOINTS[-1]):
                index_signature = tuple(float(calib[joint_idx]) for joint_idx in INDEX_FLEXION_JOINTS)
            if (
                index_signature is not None
                and index_signature != self.state.calib_index_signature
                and np.all(np.isfinite(index_signature))
            ):
                self.state.calib_index_signature = index_signature
                print(
                    "[TRACK][CALIB] open-hand index offsets "
                    f"MCP={index_signature[0]:.4f} "
                    f"PIP={index_signature[1]:.4f} "
                    f"DIP={index_signature[2]:.4f}"
                )
            for joint_idx in FLEXION_JOINTS:
                if (
                    0 <= joint_idx < int(qpos.shape[0])
                    and joint_idx < int(calib.shape[0])
                    and np.isfinite(qpos[joint_idx])
                    and np.isfinite(calib[joint_idx])
                ):
                    qpos[joint_idx] = qpos[joint_idx] - calib[joint_idx]
            deadzone = float(FINGER_FLEX_DEADZONE_RAD)
            for joint_idx in FLEXION_JOINTS:
                if 0 <= joint_idx < int(qpos.shape[0]) and np.isfinite(qpos[joint_idx]) and qpos[joint_idx] < deadzone:
                    qpos[joint_idx] = 0.0
        except Exception:
            pass

    def _max_step_vector(self, qpos_shape):
        max_step = np.ones(qpos_shape, dtype=float) * float(QPOS_MAX_STEP_DEFAULT)
        if qpos_shape[0] > WRIST_YAW_IDX:
            max_step[WRIST_YAW_IDX] = float(QPOS_MAX_STEP_YAW)
        if qpos_shape[0] > WRIST_DEV_IDX:
            max_step[WRIST_DEV_IDX] = float(QPOS_MAX_STEP_WRIST_DEV)
        if qpos_shape[0] > WRIST_FLEX_IDX:
            max_step[WRIST_FLEX_IDX] = float(QPOS_MAX_STEP_WRIST_FLEX)
        for joint_idx in (THUMB_CMC_ABD_IDX, THUMB_CMC_FLEX_IDX):
            if 0 <= joint_idx < int(qpos_shape[0]):
                max_step[joint_idx] = float(QPOS_MAX_STEP_THUMB_CMC)
        for joint_idx in (THUMB_CMC_ABD_IDX, FINGER_BASE_IDX["index"] + 1, FINGER_BASE_IDX["middle"] + 1, FINGER_BASE_IDX["ring"] + 1, FINGER_BASE_IDX["little"] + 1):
            if 0 <= joint_idx < int(qpos_shape[0]):
                max_step[joint_idx] = float(QPOS_MAX_STEP_MCP_ABD)
        return max_step

    def _smooth_and_clamp_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=float)
        prev_qpos = self.state.prev_qpos
        if prev_qpos is None:
            qpos = np.where(np.isfinite(qpos), qpos, 0.0)
            self.state.prev_qpos = qpos.copy()
            self.last_debug.smoothing_applied = False
            return qpos

        prev_qpos = np.asarray(prev_qpos, dtype=float)
        qpos = np.where(np.isfinite(qpos), qpos, prev_qpos)
        max_step = self._max_step_vector(qpos.shape)
        q_clamped = prev_qpos + np.clip(qpos - prev_qpos, -max_step, max_step)
        alpha_vec = np.ones(qpos.shape, dtype=float) * float(QPOS_SMOOTH_ALPHA)
        index_alpha = float(np.clip(INDEX_QPOS_SMOOTH_ALPHA, 0.0, 1.0))
        for joint_idx in INDEX_FLEXION_JOINTS:
            if 0 <= joint_idx < int(alpha_vec.shape[0]):
                alpha_vec[joint_idx] = index_alpha
        qpos = prev_qpos * (1.0 - alpha_vec) + q_clamped * alpha_vec
        self.state.prev_qpos = qpos.copy()
        self.last_debug.smoothing_applied = True
        return qpos

    def map_hand_to_qpos(self, hand, arm=None):
        qpos = np.zeros(QPOS_SIZE, dtype=float)
        self.last_debug = MapperDebugSnapshot(stage="mapping")
        try:
            basis = extract_palm_basis_from_hand(hand, self.desktop_mode_transform)
            palm_right, palm_normal, palm_direction = self._map_wrist(hand, arm, basis, qpos)
            r_palm = self._map_thumb(hand, palm_right, palm_normal, palm_direction, qpos)
            self._map_fingers(hand, r_palm, qpos)
            self._apply_joint_angle_overrides(hand, qpos)
            self._apply_calibration_correction(qpos)
            self.last_debug.qpos_raw = np.asarray(qpos, dtype=float).copy()
        except Exception:
            self.last_debug.stage = "fallback_prev_qpos"
            self.last_debug.used_previous_qpos = True
            try:
                qpos = np.asarray(self.state.prev_qpos, dtype=float).copy()
            except Exception:
                qpos = np.zeros(QPOS_SIZE, dtype=float)

        try:
            qpos_out = self._smooth_and_clamp_qpos(qpos)
        except Exception:
            self.last_debug.stage = "fallback_unsmoothed"
            try:
                self.state.prev_qpos = np.asarray(qpos, dtype=float).copy()
            except Exception:
                pass
            qpos_out = qpos

        self.last_debug.qpos_output = np.asarray(qpos_out, dtype=float).copy()
        return qpos_out

# ---------------------------------------------------------------------------
# Tracking Listener
# ---------------------------------------------------------------------------
class LeapDataListener(leap.Listener):
    def __init__(
        self,
        mapper,
        canvas=None,
        reset_callback=None,
        runtime_profiler=None,
        hand_mapping_debug=False,
        hand_mapping_debug_interval=DEFAULT_HAND_MAPPING_DEBUG_INTERVAL,
        hand_mapping_debug_seconds=DEFAULT_HAND_MAPPING_DEBUG_SECONDS,
        hand_mapping_debug_label=DEFAULT_TRACKING_TUNING_PRESET,
        auto_reset_on_track_loss=True,
        track_reacquire_reset_seconds=0.35,
        min_reset_interval_seconds=0.75,
    ):
        super().__init__()
        self.mapper = mapper
        self.canvas = canvas
        self.reset_callback = reset_callback
        self.runtime_profiler = runtime_profiler
        self.hand_mapping_debug = bool(hand_mapping_debug)
        self.hand_mapping_debug_interval = max(0.05, float(hand_mapping_debug_interval))
        self.hand_mapping_debug_until = time.time() + max(0.0, float(hand_mapping_debug_seconds)) if self.hand_mapping_debug else 0.0
        self.hand_mapping_debug_label = str(hand_mapping_debug_label or DEFAULT_TRACKING_TUNING_PRESET)
        self.last_hand_mapping_debug_time = 0.0
        self.auto_reset_on_track_loss = bool(auto_reset_on_track_loss)
        self.track_reacquire_reset_seconds = float(track_reacquire_reset_seconds)
        self.min_reset_interval_seconds = float(min_reset_interval_seconds)
        self.latest_qpos = None
        self.latest_ctrl = None
        self.latest_ctrl_raw = None
        self.latest_ctrl_smoothed = None
        self.latest_ctrl_analysis = None
        self.latest_ctrl_validation = None
        self.latest_ctrl_summary = None
        self.hand_present = False
        self.last_event_time = None
        self.last_hand_time = None
        self.last_missing_time = None
        self.last_auto_reset_time = 0.0
        self.runtime_frame_hint = 0

    def _set_hud(self, label, colour):
        if self.canvas is not None:
            self.canvas.set_hud(label, colour)

    def on_tracking_event(self, event):
        event_start = time.perf_counter()
        now = time.time()
        self.last_event_time = now
        has_hand = event is not None and len(event.hands) > 0

        if has_hand:
            if self.last_missing_time is not None:
                missing_duration = float(now - self.last_missing_time)
                if (
                    self.auto_reset_on_track_loss
                    and missing_duration >= self.track_reacquire_reset_seconds
                    and (now - float(self.last_auto_reset_time)) >= self.min_reset_interval_seconds
                    and self.reset_callback is not None
                ):
                    self.reset_callback(self, f"reacquired after gap {missing_duration:.2f}s")
            self.hand_present = True
            self.last_hand_time = now
            self.last_missing_time = None
            self._set_hud("TRACKING OK", (0, 220, 0))
            hand = event.hands[0]
            try:
                qpos_t0 = time.perf_counter()
                q = self.mapper.map_hand_to_qpos(hand, hand.arm)
                if self.runtime_profiler is not None:
                    self.runtime_profiler.add_event(
                        "qpos_mapping",
                        (time.perf_counter() - qpos_t0) * 1000.0,
                        frame_index=int(self.runtime_frame_hint),
                    )
            except Exception:
                q = None
            if q is not None:
                try:
                    self.latest_qpos = np.asarray(q, dtype=float)
                except Exception:
                    self.latest_qpos = None
                if (
                    self.hand_mapping_debug
                    and now <= float(self.hand_mapping_debug_until)
                    and (now - float(self.last_hand_mapping_debug_time)) >= self.hand_mapping_debug_interval
                ):
                    try:
                        print_hand_mapping_debug(
                            self.mapper.get_last_debug_snapshot(),
                            preset_name=self.hand_mapping_debug_label,
                            frame_hint=int(self.runtime_frame_hint),
                        )
                        self.last_hand_mapping_debug_time = now
                    except Exception:
                        pass
        else:
            self.hand_present = False
            self._set_hud("NO HAND", (0, 140, 255))
            if self.last_missing_time is None:
                self.last_missing_time = now

        if self.canvas is not None:
            try:
                self.canvas.render_hands(event)
            except Exception:
                pass
        if self.runtime_profiler is not None:
            self.runtime_profiler.add_event(
                "tracking_input",
                (time.perf_counter() - event_start) * 1000.0,
                frame_index=int(self.runtime_frame_hint),
            )


def setup_leap_motion_device(connection):
    t0 = time.time()
    while True:
        try:
            devices = list(connection.get_devices() or [])
        except Exception:
            devices = []
        if devices:
            break
        if (time.time() - t0) >= 2.0:
            print("[WARN] No Leap Motion device detected.")
            return False
        time.sleep(0.1)

    for device in devices:
        try:
            connection.subscribe_events(device)
        except Exception:
            continue
    return True

# ---------------------------------------------------------------------------
# Simple Canvas
# ---------------------------------------------------------------------------
VISUALIZE_LEAP = True


class Canvas:
    def __init__(self):
        self.name = "Leap Visualiser"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.hands_format = "Skeleton"
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.status_message = ""
        self.status_until = 0.0
        self.hud_message = "TRACKING INIT"
        self.hud_colour = (180, 180, 180)

    def set_status(self, message, duration=2.0):
        self.status_message = str(message)
        self.status_until = time.time() + float(duration)

    def set_hud(self, message, colour):
        self.hud_message = str(message)
        self.hud_colour = tuple(int(v) for v in colour)

    def draw_hud(self):
        text = self.hud_message or "TRACKING INIT"
        colour = self.hud_colour
        x0, y0 = 12, self.screen_size[0] - 44
        x1, y1 = 250, self.screen_size[0] - 12
        cv2.rectangle(self.output_image, (x0, y0), (x1, y1), (20, 20, 20), -1)
        cv2.rectangle(self.output_image, (x0, y0), (x1, y1), colour, 1)
        cv2.putText(self.output_image, text, (20, self.screen_size[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

    def draw_status(self):
        if not self.status_message or time.time() > float(self.status_until):
            return
        x0, y0 = 12, 12
        x1, y1 = self.screen_size[1] - 12, 54
        cv2.rectangle(self.output_image, (x0, y0), (x1, y1), (30, 30, 30), -1)
        cv2.rectangle(self.output_image, (x0, y0), (x1, y1), (0, 255, 255), 1)
        cv2.putText(self.output_image, self.status_message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    def get_joint_position(self, vec):
        if vec is None:
            return None
        half_w = self.screen_size[1] / 2
        half_h = self.screen_size[0] / 2
        return (int(vec.x + half_w), int(vec.z + half_h))

    def render_hands(self, event):
        self.output_image[:, :] = 0
        if event is None or len(event.hands) == 0:
            self.draw_hud()
            self.draw_status()
            return
        for hand in event.hands:
            wrist = self.get_joint_position(hand.arm.next_joint)
            elbow = self.get_joint_position(hand.arm.prev_joint)
            if wrist:
                cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)
            if elbow:
                cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)
            if wrist and elbow:
                cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)
            for index_digit in range(0, 5):
                digit = hand.digits[index_digit]
                for index_bone in range(0, 4):
                    bone = digit.bones[index_bone]
                    bone_start = self.get_joint_position(bone.prev_joint)
                    bone_end = self.get_joint_position(bone.next_joint)
                    if bone_start:
                        cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)
                    if bone_end:
                        cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)
                    if bone_start and bone_end:
                        cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)
        self.draw_hud()
        self.draw_status()

# ---------------------------------------------------------------------------
# Muscle Graph Canvas
# ---------------------------------------------------------------------------
EDGE_COLOURS = {
    "functional_group": (196, 183, 170),  # #AAB7C4
    "synergist": (110, 211, 110),         # #6ED36E
    "antagonist": (75, 106, 224),         # #E06A4B
    "data_driven": (162, 170, 180),
}

BACKGROUND_TOP = np.asarray((8, 11, 18), dtype=np.uint8)       # #120B08
BACKGROUND_BOTTOM = np.asarray((12, 16, 22), dtype=np.uint8)   # #16100C
PANEL_FILL = (22, 24, 30)
PANEL_BORDER = (64, 68, 74)
PANEL_HEADER = (178, 186, 196)
TEXT_PRIMARY = (223, 214, 202)
TEXT_SECONDARY = (172, 165, 154)
TEXT_MUTED = (126, 121, 113)
NODE_BORDER = (207, 227, 244)  # #F4E3CF
ZONE_LINE = (52, 58, 64)
ZONE_LABEL = (150, 146, 138)
DEFAULT_GRAPH_RENDER_FPS = 18.0
DEFAULT_GRAPH_PANEL_FPS = 6.0
DEFAULT_RUNTIME_PROFILE_INTERVAL = 25.0
GRAPH_RENDER_STALE_SECONDS = 0.75
GRAPH_SIGNATURE_CTRL_STEP = 0.02
GRAPH_SIGNATURE_SCORE_STEP = 0.02

class MuscleGraphCanvas:
    def __init__(
        self,
        static_graph: dict[str, Any],
        config: MuscleGraphVisualizationConfig | None = None,
        *,
        panel_update_fps: float = DEFAULT_GRAPH_PANEL_FPS,
        show_side_panels: bool = True,
    ):
        self.static_graph = static_graph
        self.config = config or self._load_config_from_static_graph(static_graph)
        self.name = "Muscle Graph"
        self.width = int(self.config.canvas_width)
        self.height = int(self.config.canvas_height)
        self.panel_update_interval = 1.0 / max(1.0, float(panel_update_fps))
        self.show_side_panels = bool(show_side_panels)
        self._window_prepared = False
        self.output_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.graph_bounds = self._graph_bounds()
        self.panel_layout = self._panel_layout()
        self.panel_region_left = int(self.panel_layout["header"][0][0]) if self.show_side_panels else self.width
        self.node_names = [str(name) for name in static_graph.get("node_names") or [node["node_name"] for node in static_graph["nodes"]]]
        self.node_zones = {
            str(name): str(zone)
            for name, zone in dict(static_graph.get("node_visual_zones") or {}).items()
        }
        self.node_positions = build_hand_layout_positions(
            self.node_names,
            width=self.width,
            height=self.height,
            bounds=self.graph_bounds,
        )
        self.zone_label_positions = scaled_zone_label_positions(width=self.width, height=self.height, bounds=self.graph_bounds)
        self._gradient_frame = self._build_gradient_frame()
        self._base_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._cached_panel_region: np.ndarray | None = None
        self._last_panel_signature: tuple[Any, ...] | None = None
        self._last_panel_refresh_time = 0.0
        self._last_render_signature: tuple[Any, ...] | None = None
        self._last_render_time = 0.0
        self._build_static_frame()

    def ensure_window(self) -> None:
        if self._window_prepared:
            return
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(self.name, self.width, self.height)
        self._window_prepared = True

    def _load_config_from_static_graph(self, static_graph: dict[str, Any]) -> MuscleGraphVisualizationConfig:
        config_dict = dict(static_graph.get("visualization_config") or {})
        allowed_keys = set(MuscleGraphVisualizationConfig.__dataclass_fields__.keys())
        filtered = {key: value for key, value in config_dict.items() if key in allowed_keys}
        return MuscleGraphVisualizationConfig(**filtered)

    def _graph_bounds(self) -> tuple[int, int, int, int]:
        left = max(28, int(round(self.width * 0.03)))
        top = max(28, int(round(self.height * 0.05)))
        right = min(self.width - 28, int(round(self.width * 0.68)))
        bottom = min(self.height - 28, int(round(self.height * 0.94)))
        return (left, top, right, bottom)

    def _panel_layout(self) -> dict[str, tuple[tuple[int, int], tuple[int, int], str]]:
        panel_left = max(16, int(round(self.width * 0.70)))
        right = self.width - 20
        return {
            "header": ((panel_left, 18), (right, 112), "Muscle Interaction Graph"),
            "top_active": ((panel_left, 122), (right, 332), "Top Active Muscles"),
            "edges": ((panel_left, 346), (right, 590), "Dominant Edges" if bool(self.config.dominant_mode) else "Visible Edges"),
            "zones": ((panel_left, 604), (right, min(self.height - 20, 850)), "Zones / Clusters"),
        }

    def _build_gradient_frame(self) -> np.ndarray:
        ramp = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None]
        gradient = ((1.0 - ramp) * BACKGROUND_TOP + ramp * BACKGROUND_BOTTOM).astype(np.uint8)
        return np.repeat(gradient[:, None, :], self.width, axis=1)

    def _build_static_frame(self) -> None:
        self._base_frame[:, :] = self._gradient_frame
        self._draw_zone_guides(image=self._base_frame)
        self._draw_zone_labels(image=self._base_frame)
        if self.show_side_panels:
            for top_left, bottom_right, title in self.panel_layout.values():
                self._draw_panel(top_left, bottom_right, title=title, image=self._base_frame)
        self.output_image[:, :] = self._base_frame
        self._cached_panel_region = self._base_frame[:, self.panel_region_left :].copy() if self.show_side_panels else None

    def render(self, graph_state: dict[str, Any], *, now: float | None = None, force: bool = False) -> bool:
        render_time = time.time() if now is None else float(now)
        node_feature_names = list(graph_state.get("node_feature_names") or [])
        activation_idx = node_feature_names.index("activation") if "activation" in node_feature_names else 0
        x = np.asarray(graph_state["x"], dtype=float)
        analysis = graph_state.get("analysis") or {}
        edge_list = list(graph_state.get("visible_edges") or [])
        visible_node_names = {str(name) for name in graph_state.get("node_names") or []}
        if bool(self.config.dominant_mode):
            edge_list = list(graph_state.get("dominant_visible_edges") or [])
            visible_node_names = {str(name) for name in list(analysis.get("dominant_visible_node_names") or [])}

        render_signature = self._build_render_signature(graph_state, analysis, edge_list, visible_node_names)
        if (
            not force
            and render_signature == self._last_render_signature
            and (render_time - self._last_render_time) < GRAPH_RENDER_STALE_SECONDS
        ):
            return False

        self.output_image[:, :] = self._base_frame
        self._draw_edges(edge_list)
        self._draw_nodes(graph_state, x, activation_idx, analysis, visible_node_names)
        if self.show_side_panels:
            self._draw_or_restore_panels(
                graph_state=graph_state,
                x=x,
                activation_idx=activation_idx,
                analysis=analysis,
                edge_list=edge_list,
                visible_node_names=visible_node_names,
                now=render_time,
            )
        self._last_render_signature = render_signature
        self._last_render_time = render_time
        return True

    def _draw_zone_guides(self, *, image: np.ndarray) -> None:
        x0, y0, x1, y1 = self.graph_bounds
        column_names = ("thumb", "index", "middle", "ring", "little")
        centers = [self.zone_label_positions[name][0] for name in column_names if name in self.zone_label_positions]
        separators = [int(round((centers[idx] + centers[idx + 1]) * 0.5)) for idx in range(len(centers) - 1)]
        guide_colour = self._mix_colour(BACKGROUND_BOTTOM, ZONE_LINE, 0.36)
        for x in separators:
            cv2.line(image, (x, y0 + 34), (x, y1 - 32), guide_colour, 1, cv2.LINE_AA)

        wrist_y = self.zone_label_positions.get("wrist_forearm", (0, y1 - 24))[1]
        palm_y = self.zone_label_positions.get("central_shared", (0, int(round((y0 + y1) * 0.55))))[1]
        cv2.line(image, (x0 + 16, palm_y + 26), (x1 - 16, palm_y + 26), guide_colour, 1, cv2.LINE_AA)
        cv2.line(image, (x0 + 16, wrist_y - 34), (x1 - 16, wrist_y - 34), guide_colour, 1, cv2.LINE_AA)

    def _draw_zone_labels(self, *, image: np.ndarray) -> None:
        for zone_name, pos in self.zone_label_positions.items():
            label = display_name_for_zone(zone_name)
            font_scale = 0.48 if zone_name != "central_shared" else 0.36
            thickness = 1
            (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            origin = (int(round(pos[0] - (text_width * 0.5))), int(round(pos[1])))
            cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, ZONE_LABEL, thickness, cv2.LINE_AA)

    def _build_render_signature(
        self,
        graph_state: dict[str, Any],
        analysis: dict[str, Any],
        edge_list: list[dict[str, Any]],
        visible_node_names: set[str],
    ) -> tuple[Any, ...]:
        ctrl = np.asarray(graph_state.get("ctrl_analysis") if graph_state.get("ctrl_analysis") is not None else [], dtype=float)
        if ctrl.size:
            top_ctrl_indices = np.argsort(ctrl)[-6:][::-1]
            ctrl_signature = tuple(
                (str(graph_state["node_names"][idx]), int(round(float(ctrl[idx]) / GRAPH_SIGNATURE_CTRL_STEP)))
                for idx in top_ctrl_indices
                if float(ctrl[idx]) >= GRAPH_SIGNATURE_CTRL_STEP
            )
        else:
            ctrl_signature = ()

        edge_signature = tuple(
            (
                str(item.get("source_name")),
                str(item.get("target_name")),
                str(item.get("edge_type")),
                int(round(float(item.get("dynamic_score", 0.0)) / GRAPH_SIGNATURE_SCORE_STEP)),
            )
            for item in edge_list[:10]
        )
        dominant_nodes = tuple(sorted(str(name) for name in list(analysis.get("dominant_visible_node_names") or [])))
        active_zones = tuple(
            (
                str(item.get("zone_display")),
                int(round(float(item.get("mean_activation", 0.0)) / GRAPH_SIGNATURE_CTRL_STEP)),
            )
            for item in list(analysis.get("active_zones") or [])[:4]
        )
        return (
            ctrl_signature,
            edge_signature,
            dominant_nodes,
            active_zones,
            len(visible_node_names),
            int(bool(self.config.dominant_mode)),
        )

    def _draw_edges(self, edge_list: list[dict[str, Any]]) -> None:
        if not edge_list:
            return

        min_score = min(float(edge.get("dynamic_score", 0.0)) for edge in edge_list)
        max_score = max(float(edge.get("dynamic_score", 0.0)) for edge in edge_list)
        score_floor = float(self.config.edge_score_threshold)
        score_span = max(1e-6, max_score - min_score)
        strong_span = max(1e-6, max_score - score_floor)

        for edge in edge_list:
            source = str(edge["source_name"])
            target = str(edge["target_name"])
            p0 = self.node_positions.get(source)
            p1 = self.node_positions.get(target)
            if p0 is None or p1 is None:
                continue

            edge_type = str(edge["edge_type"])
            colour = EDGE_COLOURS.get(edge_type, (164, 168, 172))
            dynamic_score = float(edge.get("dynamic_score", 0.0))
            relative_strength = float(np.clip((dynamic_score - min_score) / score_span, 0.0, 1.0))
            threshold_strength = float(np.clip((dynamic_score - score_floor) / strong_span, 0.0, 1.0))
            visual_strength = threshold_strength if dynamic_score >= score_floor else relative_strength * 0.45
            thickness = 1 + int(round(4.0 * visual_strength))
            glow_colour = self._mix_colour(BACKGROUND_BOTTOM, colour, 0.24 + (0.24 * visual_strength))
            core_colour = self._mix_colour((40, 46, 56), colour, 0.72 + (0.22 * visual_strength))
            cv2.line(self.output_image, p0, p1, glow_colour, thickness + 2, cv2.LINE_8)
            cv2.line(self.output_image, p0, p1, core_colour, thickness, cv2.LINE_AA)

    def _draw_nodes(
        self,
        graph_state: dict[str, Any],
        x: np.ndarray,
        activation_idx: int,
        analysis: dict[str, Any],
        visible_node_names: set[str],
    ) -> None:
        top_active = list(analysis.get("top_active") or [])
        top_indices = np.argsort(x[:, activation_idx])[-max(1, int(self.config.highlight_top_nodes)) :][::-1] if len(x) else []
        highlight_names = {str(item["node_name"]) for item in top_active} if top_active else {str(graph_state["node_names"][idx]) for idx in top_indices}
        dominant_names = {str(name) for name in list(analysis.get("dominant_visible_node_names") or [])}
        dominant_ctrl = {
            str(item["node_name"]): float(item.get("ctrl_analysis", 0.0))
            for item in list(analysis.get("dominant_nodes") or [])
        }

        for idx, node_name in enumerate(graph_state["node_names"]):
            name = str(node_name)
            pos = self.node_positions.get(name)
            if pos is None:
                continue

            activation = float(np.clip(x[idx, activation_idx], 0.0, 1.0))
            ctrl_strength = float(np.clip(dominant_ctrl.get(name, activation), 0.0, 1.0))
            display_strength = float(max(activation, ctrl_strength))
            is_dominant = (name in dominant_names) if bool(self.config.dominant_mode) else (name in highlight_names)
            if bool(self.config.dominant_mode) and name not in visible_node_names and not bool(self.config.ghost_non_dominant_nodes):
                continue

            if bool(self.config.dominant_mode):
                radius = 6 + int(round(13.0 * display_strength)) if is_dominant else 4 + int(round(4.0 * display_strength))
            else:
                radius = 5 + int(round(11.0 * display_strength))
                if is_dominant:
                    radius += 2

            fill = self._node_fill_colour(display_strength)
            if bool(self.config.dominant_mode) and not is_dominant:
                fill = tuple(int(round(value * 0.55)) for value in fill)

            if is_dominant:
                halo_radius = radius + 5
                halo_colour = self._mix_colour(BACKGROUND_BOTTOM, (112, 178, 244), 0.18 + (0.18 * display_strength))
                cv2.circle(self.output_image, pos, halo_radius, halo_colour, -1, cv2.LINE_8)

            cv2.circle(self.output_image, (pos[0] + 2, pos[1] + 2), radius + 1, (10, 11, 14), -1, cv2.LINE_8)
            node_alpha = 0.92 if is_dominant else (0.38 if bool(self.config.dominant_mode) else 0.72)
            self._blend_circle(pos, radius, fill, node_alpha, outline_colour=NODE_BORDER)

            show_label = bool(self.config.show_labels) and (
                (bool(self.config.dominant_mode) and name in dominant_names)
                or ((not bool(self.config.dominant_mode)) and name in highlight_names)
            )
            if show_label:
                self._draw_node_label(name, pos, radius)

    def _blend_circle(
        self,
        pos: tuple[int, int],
        radius: int,
        colour: tuple[int, int, int],
        alpha: float,
        *,
        outline_colour: tuple[int, int, int] | None = None,
    ) -> None:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha <= 0.0 or radius <= 0:
            return
        x0 = max(0, int(pos[0] - radius - 3))
        y0 = max(0, int(pos[1] - radius - 3))
        x1 = min(self.width, int(pos[0] + radius + 4))
        y1 = min(self.height, int(pos[1] + radius + 4))
        if x0 >= x1 or y0 >= y1:
            return
        roi = self.output_image[y0:y1, x0:x1]
        overlay = roi.copy()
        center = (int(pos[0] - x0), int(pos[1] - y0))
        cv2.circle(overlay, center, radius, colour, -1, cv2.LINE_AA)
        if outline_colour is not None:
            cv2.circle(overlay, center, radius + 1, outline_colour, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)

    def _node_fill_colour(self, strength: float) -> tuple[int, int, int]:
        low = np.asarray((57, 98, 140), dtype=float)    # #8C6239
        mid = np.asarray((91, 155, 216), dtype=float)   # #D89B5B
        high = np.asarray((99, 154, 247), dtype=float)  # #F79A63
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength < 0.55:
            mix = strength / 0.55 if strength > 0.0 else 0.0
            colour = (1.0 - mix) * low + mix * mid
        else:
            mix = (strength - 0.55) / 0.45
            colour = (1.0 - mix) * mid + mix * high
        return tuple(int(round(value)) for value in colour.tolist())

    def _draw_node_label(self, node_name: str, pos: tuple[int, int], radius: int) -> None:
        zone_name = self.node_zones.get(node_name, zone_for_muscle(node_name))
        if zone_name == "thumb":
            text_origin = (pos[0] - 62, pos[1] - radius - 8)
        else:
            text_origin = (pos[0] + 10, pos[1] - radius - 8)

        self._draw_text_chip(node_name, text_origin, font_scale=0.40, text_colour=TEXT_PRIMARY)

    def _build_panel_signature(
        self,
        graph_state: dict[str, Any],
        analysis: dict[str, Any],
        edge_list: list[dict[str, Any]],
        visible_node_names: set[str],
    ) -> tuple[Any, ...]:
        meta = graph_state.get("meta") or {}
        top_active = tuple(
            (
                str(item.get("node_name")),
                int(round(float(item.get("ctrl_analysis", item.get("activation", 0.0))) / GRAPH_SIGNATURE_CTRL_STEP)),
            )
            for item in list(analysis.get("top_active") or [])[:6]
        )
        edge_sig = tuple(
            (
                str(item.get("source_name")),
                str(item.get("target_name")),
                int(round(float(item.get("dynamic_score", 0.0)) / GRAPH_SIGNATURE_SCORE_STEP)),
            )
            for item in edge_list[:8]
        )
        cluster_sig = tuple(
            (
                str(item.get("group")),
                int(round(float(item.get("mean_activation", 0.0)) / GRAPH_SIGNATURE_CTRL_STEP)),
            )
            for item in list(analysis.get("dominant_clusters") or [])[:3]
        )
        return (
            len(visible_node_names),
            len(edge_list),
            str(self.config.visualization_mode),
            str(meta.get("dynamic_edge_config", {}).get("edge_relation_filter", "all_active")),
            top_active,
            cluster_sig,
            edge_sig,
        )

    def _draw_or_restore_panels(
        self,
        *,
        graph_state: dict[str, Any],
        x: np.ndarray,
        activation_idx: int,
        analysis: dict[str, Any],
        edge_list: list[dict[str, Any]],
        visible_node_names: set[str],
        now: float,
    ) -> None:
        panel_signature = self._build_panel_signature(graph_state, analysis, edge_list, visible_node_names)
        refresh_panels = (
            self._cached_panel_region is None
            or panel_signature != self._last_panel_signature
            or (now - self._last_panel_refresh_time) >= self.panel_update_interval
        )
        if not refresh_panels:
            self.output_image[:, self.panel_region_left :] = self._cached_panel_region
            return

        self._draw_header_metrics(graph_state, edge_list, visible_node_names)
        self._draw_activity_panels(graph_state, x, activation_idx, analysis, edge_list)
        self._cached_panel_region = self.output_image[:, self.panel_region_left :].copy()
        self._last_panel_signature = panel_signature
        self._last_panel_refresh_time = now

    def _draw_header_metrics(self, graph_state: dict[str, Any], edge_list: list[dict[str, Any]], visible_node_names: set[str]) -> None:
        top_left, _, _ = self.panel_layout["header"]
        frame_index = graph_state.get("frame_index")
        timestamp = graph_state.get("timestamp")
        meta = graph_state.get("meta") or {}
        line1 = [
            f"frame {frame_index if frame_index is not None else '-'}",
            f"time {timestamp:.1f}" if timestamp is not None else "time -",
            f"nodes {len(visible_node_names)}",
            f"edges {len(edge_list)}",
        ]
        line2 = [
            f"mode {self.config.visualization_mode}",
            f"filter {meta.get('dynamic_edge_config', {}).get('edge_relation_filter', 'all_active')}",
        ]
        x = top_left[0] + 12
        y = top_left[1] + 46
        for line in line1:
            cv2.putText(self.output_image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_PRIMARY, 1, cv2.LINE_AA)
            x += 88
        x = top_left[0] + 12
        y += 22
        for line in line2:
            cv2.putText(self.output_image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_SECONDARY, 1, cv2.LINE_AA)
            x += 170

    def _draw_activity_panels(
        self,
        graph_state: dict[str, Any],
        x: np.ndarray,
        activation_idx: int,
        analysis: dict[str, Any],
        edge_list: list[dict[str, Any]],
    ) -> None:
        top_active_panel = self.panel_layout["top_active"]
        edges_panel = self.panel_layout["edges"]
        zones_panel = self.panel_layout["zones"]

        top_active = list(analysis.get("top_active") or [])
        if bool(self.config.dominant_mode):
            visible_dominant_names = {str(name) for name in list(analysis.get("dominant_visible_node_names") or [])}
            top_active = [
                item
                for item in list(analysis.get("dominant_nodes") or [])
                if str(item.get("node_name")) in visible_dominant_names
            ]
            if not top_active:
                top_active = list(analysis.get("dominant_nodes") or [])
        if not top_active and len(x):
            top_indices = np.argsort(x[:, activation_idx])[-6:][::-1]
            top_active = [
                {"node_name": graph_state["node_names"][idx], "activation": float(x[idx, activation_idx]), "zone": zone_for_muscle(graph_state["node_names"][idx])}
                for idx in top_indices[:6]
            ]

        y_cursor = top_active_panel[0][1] + 40
        for rank, item in enumerate(top_active[:6], start=1):
            zone_text = display_name_for_zone(str(item.get("zone") or zone_for_muscle(item["node_name"])))
            strength = float(item.get("ctrl_analysis", item.get("activation", 0.0)))
            cv2.putText(
                self.output_image,
                f"{rank}. {item['node_name']}",
                (top_active_panel[0][0] + 12, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                TEXT_PRIMARY,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                self.output_image,
                f"{zone_text}  {strength:.3f}",
                (top_active_panel[0][0] + 120, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                TEXT_SECONDARY,
                1,
                cv2.LINE_AA,
            )
            y_cursor += 28

        edge_y = edges_panel[0][1] + 40
        top_edges = list(analysis.get("top_dynamic_edges") or [])
        if bool(self.config.dominant_mode):
            top_edges = edge_list[:8]
        if not top_edges:
            cv2.putText(self.output_image, "No visible edges", (edges_panel[0][0] + 12, edge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_MUTED, 1, cv2.LINE_AA)
        else:
            for item in top_edges[:8]:
                colour = EDGE_COLOURS.get(str(item["edge_type"]), TEXT_SECONDARY)
                cv2.putText(
                    self.output_image,
                    f"{item['source_name']} <-> {item['target_name']}",
                    (edges_panel[0][0] + 12, edge_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    colour,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    self.output_image,
                    f"{item['edge_type']}  score {float(item['dynamic_score']):.3f}",
                    (edges_panel[0][0] + 12, edge_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    TEXT_SECONDARY,
                    1,
                    cv2.LINE_AA,
                )
                edge_y += 27

        zone_y = zones_panel[0][1] + 40
        for item in list(analysis.get("active_zones") or [])[:4]:
            zone_display = str(item["zone_display"])
            mean_activation = float(item["mean_activation"])
            max_activation = float(item["max_activation"])
            bar_x = zones_panel[0][0] + 12
            bar_y = zone_y + 8
            bar_width = max(120, (zones_panel[1][0] - zones_panel[0][0]) - 28)
            cv2.putText(self.output_image, zone_display, (bar_x, zone_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_PRIMARY, 1, cv2.LINE_AA)
            self._blend_rect((bar_x, bar_y), (bar_x + bar_width, bar_y + 12), (34, 38, 42), 0.80, image=self.output_image)
            fill_width = int(round(bar_width * np.clip(mean_activation, 0.0, 1.0)))
            self._blend_rect((bar_x, bar_y), (bar_x + fill_width, bar_y + 12), self._node_fill_colour(mean_activation), 0.90, image=self.output_image)
            cv2.putText(
                self.output_image,
                f"mean {mean_activation:.2f}  max {max_activation:.2f}",
                (bar_x, zone_y + 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                TEXT_SECONDARY,
                1,
                cv2.LINE_AA,
            )
            zone_y += 48

        cluster_y = min(zones_panel[1][1] - 12, zone_y + 18)
        cv2.putText(self.output_image, "Clusters", (zones_panel[0][0] + 12, cluster_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_PRIMARY, 1, cv2.LINE_AA)
        cluster_y += 22
        for item in list(analysis.get("dominant_clusters") or [])[:3]:
            cv2.putText(
                self.output_image,
                f"{item['group']}  {float(item['mean_activation']):.3f}",
                (zones_panel[0][0] + 12, cluster_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                TEXT_SECONDARY,
                1,
                cv2.LINE_AA,
            )
            cluster_y += 20

    def _draw_panel(
        self,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        *,
        title: str,
        image: np.ndarray,
    ) -> None:
        self._blend_rect(top_left, bottom_right, PANEL_FILL, 0.86, image=image)
        cv2.rectangle(image, top_left, bottom_right, PANEL_BORDER, 1)
        cv2.putText(
            image,
            title,
            (top_left[0] + 12, top_left[1] + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            PANEL_HEADER,
            1,
            cv2.LINE_AA,
        )

    def _draw_text_chip(
        self,
        text: str,
        origin: tuple[int, int],
        *,
        font_scale: float,
        text_colour: tuple[int, int, int],
    ) -> None:
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top_left = (origin[0] - 4, origin[1] - text_height - 4)
        bottom_right = (origin[0] + text_width + 4, origin[1] + 4)
        self._blend_rect(top_left, bottom_right, (18, 18, 20), 0.62, image=self.output_image)
        cv2.putText(self.output_image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_colour, 1, cv2.LINE_AA)

    def _blend_rect(
        self,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        colour: tuple[int, int, int],
        alpha: float,
        *,
        image: np.ndarray,
    ) -> None:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return
        x0 = max(0, min(int(top_left[0]), int(bottom_right[0])))
        y0 = max(0, min(int(top_left[1]), int(bottom_right[1])))
        x1 = min(image.shape[1], max(int(top_left[0]), int(bottom_right[0])))
        y1 = min(image.shape[0], max(int(top_left[1]), int(bottom_right[1])))
        if x0 >= x1 or y0 >= y1:
            return
        roi = image[y0:y1, x0:x1]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x1 - x0 - 1, y1 - y0 - 1), colour, -1)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)

    def _mix_colour(
        self,
        base_colour: tuple[int, int, int] | np.ndarray,
        highlight_colour: tuple[int, int, int] | np.ndarray,
        mix: float,
    ) -> tuple[int, int, int]:
        base = np.asarray(base_colour, dtype=float)
        highlight = np.asarray(highlight_colour, dtype=float)
        alpha = float(np.clip(mix, 0.0, 1.0))
        colour = ((1.0 - alpha) * base) + (alpha * highlight)
        return tuple(int(round(value)) for value in colour.tolist())

# ---------------------------------------------------------------------------
# Controller Bridge
# ---------------------------------------------------------------------------
CTRL_OUTPUT_DEADBAND = 1e-6
DEFAULT_QPOS_DEADBAND_NORM = 0.03
DEFAULT_CTRL_SMOOTHING_ALPHA = 0.18
DEFAULT_HOLD_IF_SMALL_CHANGE = True
DEFAULT_ROLLING_WINDOW = 7
DEFAULT_CTRL_MODE = "control"
VALID_CTRL_MODES = ("control", "analysis")

CTRL_ESTIMATION_AVAILABLE = True
CONTROLLER_AVAILABLE = True


@dataclass
class CtrlEstimationConfig:
    qpos_deadband_norm: float = DEFAULT_QPOS_DEADBAND_NORM
    ctrl_smoothing_alpha: float = DEFAULT_CTRL_SMOOTHING_ALPHA
    hold_if_small_change: bool = DEFAULT_HOLD_IF_SMALL_CHANGE
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    mode: str = DEFAULT_CTRL_MODE
    freeze_mode: bool = False

    def normalized(self) -> "CtrlEstimationConfig":
        mode = str(self.mode).strip().lower()
        if mode not in VALID_CTRL_MODES:
            mode = DEFAULT_CTRL_MODE
        deadband = max(0.0, float(self.qpos_deadband_norm))
        alpha = float(np.clip(self.ctrl_smoothing_alpha, 0.0, 1.0))
        rolling_window = max(1, int(self.rolling_window))
        if mode == "analysis":
            deadband = max(deadband, 0.03)
            alpha = min(alpha, 0.10)
            rolling_window = max(rolling_window, 10)
        return CtrlEstimationConfig(
            qpos_deadband_norm=deadband,
            ctrl_smoothing_alpha=alpha,
            hold_if_small_change=bool(self.hold_if_small_change),
            rolling_window=rolling_window,
            mode=mode,
            freeze_mode=bool(self.freeze_mode),
        )


def _sanitize_ctrl_signal(ctrl):
    ctrl = np.asarray(ctrl, dtype=float).reshape(-1)
    ctrl = np.where(np.isfinite(ctrl), ctrl, 0.0)
    ctrl[np.abs(ctrl) < float(CTRL_OUTPUT_DEADBAND)] = 0.0
    return np.clip(ctrl, 0.0, 1.0)


def _zero_ctrl(controller_context):
    probe_model = controller_context.get("probe_model")
    nu = int(getattr(probe_model, "nu", 0)) if probe_model is not None else 0
    return np.zeros(max(0, nu), dtype=float)


def _estimate_ctrl_raw(q_target, controller_context, ctrl_data_source, ctrl_log_state):
    probe_model = controller_context["probe_model"]
    probe_data = controller_context["probe_data"]
    controller = controller_context["controller"]
    if not (CTRL_ESTIMATION_AVAILABLE and probe_model is not None):
        return None

    ctrl_ok = False
    ctrl_est = None
    hold_debug = None
    if controller is not None:
        try:
            if hasattr(controller, "set_exercise_name"):
                controller.set_exercise_name(controller_context.get("exercise_name"))
            ctrl_state = controller.step(q_target, sync_data=ctrl_data_source)
            ctrl_est = np.asarray(ctrl_state["ctrl"], dtype=float)
            ctrl_ok = True
            hold_debug = {
                "hold_prior_label": str(ctrl_state.get("hold_prior_label", "none")),
                "is_hold_mode": bool(ctrl_state.get("is_hold_mode", False)),
                "fist_hold_prior_active": bool(ctrl_state.get("fist_hold_prior_active", False)),
                "index_hold_prior_active": bool(ctrl_state.get("index_hold_prior_active", False)),
                "hold_flexor_values": dict(ctrl_state.get("hold_flexor_values") or {}),
                "hold_top_flexors": list(ctrl_state.get("hold_top_flexors") or []),
                "index_hold_values": dict(ctrl_state.get("index_hold_values") or {}),
            }
            if not ctrl_log_state["controller_logged"]:
                print(f"[INFO] ctrl estimated via controller: {controller.name}")
                ctrl_log_state["controller_logged"] = True
        except Exception as e:
            if not ctrl_log_state["fallback_logged"]:
                print(f"[WARN] Controller failed, trying least-squares: {type(e).__name__}: {e}")

    if not ctrl_ok and probe_data is not None:
        try:
            if ctrl_data_source is not None:
                _, data_src = _get_raw_model_data(probe_model, ctrl_data_source)
                probe_data.qpos[:] = np.asarray(data_src.qpos, dtype=float).reshape(-1)
                probe_data.qvel[:] = np.asarray(data_src.qvel, dtype=float).reshape(-1)
                probe_data.qacc[:] = np.asarray(data_src.qacc, dtype=float).reshape(-1)
                probe_data.ctrl[:] = np.asarray(data_src.ctrl, dtype=float).reshape(-1)
                if hasattr(probe_data, "act") and int(getattr(probe_model, "na", 0)) > 0:
                    probe_data.act[:] = np.asarray(data_src.act, dtype=float).reshape(-1)
                m_raw, d_raw = _get_raw_model_data(probe_model, probe_data)
                mujoco.mj_forward(m_raw, d_raw)
            full_qpos = np.zeros(probe_model.nq, dtype=float)
            full_qpos[:min(len(full_qpos), len(q_target))] = q_target[:min(len(full_qpos), len(q_target))]
            prev_ctrl = controller_context.get("last_ctrl_raw")
            ctrl_est, info = estimate_ctrl_from_qpos(
                probe_model,
                probe_data,
                full_qpos,
                prev_ctrl=prev_ctrl,
                exercise_name=controller_context.get("exercise_name"),
            )
            hold_debug = {
                "hold_prior_label": str(info.get("hold_prior_label", "none")),
                "is_hold_mode": bool(info.get("is_hold_mode", False)),
                "fist_hold_prior_active": bool(info.get("fist_hold_prior_active", False)),
                "index_hold_prior_active": bool(info.get("index_hold_prior_active", False)),
                "hold_flexor_values": dict(info.get("hold_flexor_values") or {}),
                "hold_top_flexors": list(info.get("hold_top_flexors") or []),
                "index_hold_values": dict(info.get("index_hold_values") or {}),
            }
            if not ctrl_log_state["fallback_logged"]:
                print("[INFO] ctrl estimated via least-squares fallback.")
                ctrl_log_state["fallback_logged"] = True
        except Exception as e:
            if not ctrl_log_state["fallback_logged"]:
                print(f"[WARN] ctrl estimation failed: {type(e).__name__}: {e}")
                ctrl_log_state["fallback_logged"] = True
    controller_context["last_hold_debug"] = dict(hold_debug) if hold_debug is not None else None
    if ctrl_est is None:
        return None
    return _sanitize_ctrl_signal(ctrl_est)


def _rolling_stats(history: deque[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    if not history:
        empty = np.empty((0,), dtype=float)
        return empty, empty
    arr = np.stack(list(history)[-max(1, int(width)):], axis=0)
    return np.mean(arr, axis=0), np.median(arr, axis=0)


def reset_controller_context_state(controller_context, data_source=None):
    controller_context["last_input_q_target"] = None
    controller_context["stable_q_target"] = None
    controller_context["last_ctrl_raw"] = None
    controller_context["last_ctrl_smoothed"] = None
    controller_context["last_ctrl_analysis"] = None
    controller_context["last_ctrl_rolling_mean"] = None
    controller_context["last_ctrl_rolling_median"] = None
    history = controller_context.get("ctrl_smoothed_history")
    if history is not None:
        history.clear()
    controller_context["last_ctrl_summary"] = None
    controller_context["last_hold_debug"] = None

    controller = controller_context.get("controller")
    if controller is None:
        return
    try:
        if data_source is not None and hasattr(controller, "sync_from_data"):
            controller.sync_from_data(data_source)
        nu = int(getattr(controller.model, "nu", 0))
        nq = int(getattr(controller.model, "nq", 0))
        nv = int(getattr(controller.model, "nv", 0))
        qpos_now = np.asarray(getattr(controller, "_data").qpos, dtype=float).copy()
        controller._last_ctrl = np.zeros(nu, dtype=float)
        controller._last_target = np.zeros(nq, dtype=float)
        controller._last_target[: min(nq, len(qpos_now))] = qpos_now[: min(nq, len(qpos_now))]
        controller._last_target_qvel = np.zeros(nv, dtype=float)
        controller._last_tau_des = np.zeros(nv, dtype=float)
        controller._last_pose_error = np.zeros(nq, dtype=float)
        controller._last_pose_error_norm = 0.0
        controller._prev_qpos = qpos_now.copy()
    except Exception:
        pass


def create_controller_context(env_id=DEFAULT_MYOSUITE_ENV_ID, estimation_config: CtrlEstimationConfig | None = None, exercise_name: str | None = None):
    config = (estimation_config or CtrlEstimationConfig()).normalized()
    context = {
        "probe_model": None,
        "probe_data": None,
        "controller": None,
        "exercise_name": None if exercise_name is None else str(exercise_name),
        "estimation_config": config,
        "last_input_q_target": None,
        "stable_q_target": None,
        "last_ctrl_raw": None,
        "last_ctrl_smoothed": None,
        "last_ctrl_analysis": None,
        "last_ctrl_rolling_mean": None,
        "last_ctrl_rolling_median": None,
        "ctrl_smoothed_history": deque(maxlen=max(1, int(config.rolling_window))),
        "last_ctrl_summary": None,
        "last_hold_debug": None,
    }
    if not CTRL_ESTIMATION_AVAILABLE:
        return context
    try:
        _, probe_model, probe_data = create_probe_env(env_id)
        context["probe_model"] = probe_model
        context["probe_data"] = probe_data
        print("[INFO] Probe env for ctrl estimation created.")
        if CONTROLLER_AVAILABLE:
            try:
                controller = RealtimeQPController(probe_model, probe_data)
                if hasattr(controller, "set_exercise_name"):
                    controller.set_exercise_name(context["exercise_name"])
                context["controller"] = controller
                ctrl_info = controller.get_info()
                print(f"[INFO] Controller initialized: {ctrl_info['name']}")
            except Exception as e:
                print(f"[WARN] Controller init failed, will use least-squares fallback: {e}")
    except Exception as e:
        print(f"[WARN] Failed to create probe env: {e}")
    return context


def prepare_target_qpos(q_target, controller_context):
    q_target = np.asarray(q_target, dtype=float).reshape(-1)
    probe_model = controller_context.get("probe_model")
    if _clip_to_limits is None or probe_model is None:
        return q_target
    full_qpos = np.zeros(int(probe_model.nq), dtype=float)
    full_qpos[:min(len(full_qpos), len(q_target))] = q_target[:min(len(full_qpos), len(q_target))]
    clipped = _clip_to_limits(probe_model, full_qpos)
    return clipped[: len(q_target)] if len(q_target) < len(clipped) else clipped


def estimate_ctrl_signals(q_target, controller_context, ctrl_data_source, ctrl_log_state):
    probe_model = controller_context["probe_model"]
    if not (CTRL_ESTIMATION_AVAILABLE and probe_model is not None):
        return None

    config = controller_context.get("estimation_config") or CtrlEstimationConfig()
    config = config.normalized() if hasattr(config, "normalized") else CtrlEstimationConfig().normalized()
    controller_context["estimation_config"] = config

    q_target = np.asarray(q_target, dtype=float).reshape(-1)
    prev_input = controller_context.get("last_input_q_target")
    stable_q = controller_context.get("stable_q_target")
    prev_raw = controller_context.get("last_ctrl_raw")
    prev_smoothed = controller_context.get("last_ctrl_smoothed")
    prev_analysis = controller_context.get("last_ctrl_analysis")
    history = controller_context.get("ctrl_smoothed_history")
    if history is None or getattr(history, "maxlen", None) != max(1, int(config.rolling_window)):
        history = deque(maxlen=max(1, int(config.rolling_window)))
        controller_context["ctrl_smoothed_history"] = history

    qpos_delta_norm = float(np.linalg.norm(q_target - stable_q)) if stable_q is not None and len(stable_q) == len(q_target) else 0.0
    input_qpos_delta_norm = float(np.linalg.norm(q_target - prev_input)) if prev_input is not None and len(prev_input) == len(q_target) else 0.0
    deadband_applied = False
    freeze_applied = False
    hold_applied = False
    recomputed = False

    if stable_q is None or prev_raw is None:
        effective_q_target = q_target.copy()
        controller_context["stable_q_target"] = q_target.copy()
    elif bool(config.freeze_mode):
        effective_q_target = np.asarray(stable_q, dtype=float).copy()
        freeze_applied = True
        hold_applied = True
    elif bool(config.hold_if_small_change) and qpos_delta_norm < float(config.qpos_deadband_norm):
        effective_q_target = np.asarray(stable_q, dtype=float).copy()
        deadband_applied = True
        hold_applied = True
    else:
        effective_q_target = q_target.copy()
        controller_context["stable_q_target"] = q_target.copy()

    if prev_raw is None or not hold_applied:
        ctrl_raw = _estimate_ctrl_raw(effective_q_target, controller_context, ctrl_data_source, ctrl_log_state)
        if ctrl_raw is None:
            ctrl_raw = _zero_ctrl(controller_context)
        recomputed = True
    else:
        ctrl_raw = np.asarray(prev_raw, dtype=float).copy()

    ctrl_raw = _sanitize_ctrl_signal(ctrl_raw)
    if prev_smoothed is None:
        ctrl_smoothed = ctrl_raw.copy()
    elif recomputed:
        alpha = float(config.ctrl_smoothing_alpha)
        ctrl_smoothed = _sanitize_ctrl_signal((1.0 - alpha) * np.asarray(prev_smoothed, dtype=float) + alpha * ctrl_raw)
    else:
        ctrl_smoothed = np.asarray(prev_smoothed, dtype=float).copy()

    history.append(np.asarray(ctrl_smoothed, dtype=float).copy())
    rolling_mean, rolling_median = _rolling_stats(history, int(config.rolling_window))
    if rolling_mean.size == 0:
        rolling_mean = ctrl_smoothed.copy()
        rolling_median = ctrl_smoothed.copy()
    ctrl_analysis = rolling_median if str(config.mode) == "analysis" else rolling_mean
    ctrl_analysis = _sanitize_ctrl_signal(ctrl_analysis)

    ctrl_raw_delta_norm = float(np.linalg.norm(ctrl_raw - prev_raw)) if prev_raw is not None and len(prev_raw) == len(ctrl_raw) else 0.0
    ctrl_smoothed_delta_norm = float(np.linalg.norm(ctrl_smoothed - prev_smoothed)) if prev_smoothed is not None and len(prev_smoothed) == len(ctrl_smoothed) else 0.0
    ctrl_analysis_delta_norm = float(np.linalg.norm(ctrl_analysis - prev_analysis)) if prev_analysis is not None and len(prev_analysis) == len(ctrl_analysis) else 0.0

    controller_context["last_input_q_target"] = q_target.copy()
    controller_context["last_ctrl_raw"] = ctrl_raw.copy()
    controller_context["last_ctrl_smoothed"] = ctrl_smoothed.copy()
    controller_context["last_ctrl_analysis"] = ctrl_analysis.copy()
    controller_context["last_ctrl_rolling_mean"] = np.asarray(rolling_mean, dtype=float).copy()
    controller_context["last_ctrl_rolling_median"] = np.asarray(rolling_median, dtype=float).copy()

    result = {
        "q_target_input": q_target.copy(),
        "q_target_used": np.asarray(effective_q_target, dtype=float).copy(),
        "ctrl_raw": ctrl_raw.copy(),
        "ctrl_smoothed": ctrl_smoothed.copy(),
        "ctrl_analysis": ctrl_analysis.copy(),
        "ctrl_rolling_mean": np.asarray(rolling_mean, dtype=float).copy(),
        "ctrl_rolling_median": np.asarray(rolling_median, dtype=float).copy(),
        "qpos_delta_norm": float(qpos_delta_norm),
        "input_qpos_delta_norm": float(input_qpos_delta_norm),
        "ctrl_raw_delta_norm": float(ctrl_raw_delta_norm),
        "ctrl_smoothed_delta_norm": float(ctrl_smoothed_delta_norm),
        "ctrl_analysis_delta_norm": float(ctrl_analysis_delta_norm),
        "deadband_applied": bool(deadband_applied),
        "hold_applied": bool(hold_applied),
        "freeze_applied": bool(freeze_applied),
        "recomputed": bool(recomputed),
        "mode": str(config.mode),
        "rolling_window": int(config.rolling_window),
    }
    hold_debug = controller_context.get("last_hold_debug")
    if isinstance(hold_debug, dict):
        result.update(dict(hold_debug))
    controller_context["last_ctrl_summary"] = dict(result)
    return result


def estimate_ctrl_vector(q_target, controller_context, ctrl_data_source, ctrl_log_state):
    result = estimate_ctrl_signals(
        q_target=q_target,
        controller_context=controller_context,
        ctrl_data_source=ctrl_data_source,
        ctrl_log_state=ctrl_log_state,
    )
    if result is None:
        return None
    return np.asarray(result["ctrl_analysis"], dtype=float).copy()


def _artifact_array(value: Any, *, dtype: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _indexed_columns(field_names: Iterable[str], prefix: str) -> list[str]:
    columns: list[tuple[int, str]] = []
    for name in field_names:
        label = str(name)
        if not label.startswith(prefix):
            continue
        try:
            index = int(label[len(prefix) :])
        except ValueError:
            continue
        columns.append((index, label))
    columns.sort(key=lambda item: item[0])
    return [label for _, label in columns]


def _load_graph_replay_series_from_csv(path: Path) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        field_names = list(reader.fieldnames or [])
        if not field_names:
            raise ValueError(f"Recording CSV has no header: {path}")

        qpos_columns = _indexed_columns(field_names, "qpos_")
        ctrl_columns = _indexed_columns(field_names, "ctrl_analysis_")
        if not ctrl_columns:
            ctrl_columns = _indexed_columns(field_names, "ctrl_smoothed_")
        if not ctrl_columns:
            ctrl_columns = _indexed_columns(field_names, "ctrl_raw_")
        if not ctrl_columns:
            ctrl_columns = _indexed_columns(field_names, "ctrl_")

        if not qpos_columns:
            raise ValueError(f"Recording CSV has no qpos_* columns: {path}")
        if not ctrl_columns:
            raise ValueError(f"Recording CSV has no ctrl columns: {path}")

        timestamps: list[float] = []
        frame_indices: list[int] = []
        qpos_rows: list[list[float]] = []
        ctrl_rows: list[list[float]] = []
        for row_idx, row in enumerate(reader):
            timestamps.append(float(row.get("t") or row_idx))
            frame_indices.append(int(float(row.get("frame") or row_idx)))
            qpos_rows.append([float(row.get(name) or 0.0) for name in qpos_columns])
            ctrl_rows.append([float(row.get(name) or 0.0) for name in ctrl_columns])

    qpos_seq = np.asarray(qpos_rows, dtype=np.float32)
    ctrl_seq = np.asarray(ctrl_rows, dtype=np.float32)
    if qpos_seq.ndim != 2 or ctrl_seq.ndim != 2 or qpos_seq.shape[0] == 0 or ctrl_seq.shape[0] == 0:
        raise ValueError(f"Recording CSV does not contain any replayable frames: {path}")
    return {
        "path": str(path),
        "label": path.name,
        "timestamps": timestamps,
        "frame_indices": frame_indices,
        "joint_qpos_seq": qpos_seq,
        "ctrl_analysis_seq": ctrl_seq,
        "source_kind": "csv",
    }


def _resolve_sequence_pt_path(path: Path) -> Path:
    if path.suffix.lower() == ".pt":
        return path
    if path.suffix.lower() == ".json":
        summary = json.loads(path.read_text(encoding="utf-8"))
        pt_value = summary.get("pt_path")
        if not pt_value:
            raise ValueError(f"Sequence summary does not reference a .pt artifact: {path}")
        return resolve_local_path(str(pt_value))
    raise ValueError(f"Unsupported sequence artifact: {path}")


def _load_graph_replay_series_from_sequence(path: Path) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Replay from .pt sequence requires torch to be installed.")
    pt_path = _resolve_sequence_pt_path(path)
    artifact = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError(f"Unexpected sequence artifact payload: {pt_path}")
    if str(artifact.get("graph_kind") or "") != "muscle_graph_sequence":
        raise ValueError(f"Unsupported graph artifact kind: {artifact.get('graph_kind')}")

    qpos_seq = _artifact_array(artifact.get("joint_qpos_seq"), dtype=np.float32)
    ctrl_seq = _artifact_array(artifact.get("ctrl_analysis_seq"), dtype=np.float32)
    if qpos_seq.ndim != 2 or ctrl_seq.ndim != 2 or qpos_seq.shape[0] == 0 or ctrl_seq.shape[0] == 0:
        raise ValueError(f"Sequence artifact has no replayable frames: {pt_path}")

    timestamps = [float(item) for item in list(artifact.get("timestamps") or range(int(qpos_seq.shape[0])))]
    frame_indices = [int(item) for item in list(artifact.get("frame_indices") or range(int(qpos_seq.shape[0])))]
    return {
        "path": str(pt_path),
        "label": pt_path.name,
        "timestamps": timestamps,
        "frame_indices": frame_indices,
        "joint_qpos_seq": qpos_seq,
        "ctrl_analysis_seq": ctrl_seq,
        "source_kind": "sequence",
    }


def load_graph_replay_series(record_path: str) -> dict[str, Any]:
    path = resolve_local_path(record_path)
    if not path.exists():
        raise FileNotFoundError(f"Replay source not found: {path}")
    if path.suffix.lower() == ".csv":
        return _load_graph_replay_series_from_csv(path)
    if path.suffix.lower() in {".pt", ".json"}:
        return _load_graph_replay_series_from_sequence(path)
    raise ValueError(f"Unsupported replay source: {path}")


def replay_recorded_muscle_graph(
    record_path: str,
    *,
    show_ctrl_validation_window: bool = False,
    show_muscle_graph_window: bool = True,
    dynamic_edge_mode: str,
    node_activation_threshold: float,
    edge_score_threshold: float,
    dynamic_edge_window: int,
    visible_edge_top_k: int,
    edge_relation_filter: str,
    highlight_top_nodes: int,
    visualization_mode: str,
    dominant_selection_mode: str,
    dominant_mode: bool,
    dominant_top_k: int,
    dominant_edge_top_k: int,
    include_thumb_in_dominant_subset: bool,
    hide_inactive_dominant_nodes: bool,
    ghost_non_dominant_nodes: bool,
    graph_preset: str,
    debug_dynamic_edges: bool,
    show_labels: bool,
    ctrl_mode: str = DEFAULT_CTRL_MODE,
    qpos_deadband_norm: float = DEFAULT_QPOS_DEADBAND_NORM,
    ctrl_smoothing_alpha: float = DEFAULT_CTRL_SMOOTHING_ALPHA,
    ctrl_rolling_window: int = DEFAULT_ROLLING_WINDOW,
    hold_if_small_change: bool = True,
    freeze_control: bool = False,
    tracking_tuning_preset: str = DEFAULT_TRACKING_TUNING_PRESET,
    hand_mapping_debug: bool = False,
    hand_mapping_debug_interval: float = DEFAULT_HAND_MAPPING_DEBUG_INTERVAL,
    hand_mapping_debug_seconds: float = DEFAULT_HAND_MAPPING_DEBUG_SECONDS,
    graph_render_fps: float = DEFAULT_GRAPH_RENDER_FPS,
    graph_panel_fps: float = DEFAULT_GRAPH_PANEL_FPS,
    replay_speed: float = 1.0,
    replay_loop: bool = False,
    diagnostic_options: RuntimeDiagnosticOptions | None = None,
) -> int:
    _ = (
        show_ctrl_validation_window,
        show_muscle_graph_window,
        ctrl_mode,
        qpos_deadband_norm,
        ctrl_smoothing_alpha,
        ctrl_rolling_window,
        hold_if_small_change,
        freeze_control,
        tracking_tuning_preset,
        hand_mapping_debug,
        hand_mapping_debug_interval,
        hand_mapping_debug_seconds,
    )
    replay_data = load_graph_replay_series(record_path)
    qpos_seq = np.asarray(replay_data["joint_qpos_seq"], dtype=np.float32)
    ctrl_seq = np.asarray(replay_data["ctrl_analysis_seq"], dtype=np.float32)
    num_frames = int(min(qpos_seq.shape[0], ctrl_seq.shape[0]))
    if num_frames <= 0:
        raise ValueError(f"No replayable frames found in {record_path}")

    timestamps = [float(item) for item in list(replay_data.get("timestamps") or [])][:num_frames]
    if len(timestamps) < num_frames:
        timestamps = [float(idx) / max(1.0, float(MYOSUITE_RENDER_FPS)) for idx in range(num_frames)]
    frame_indices = [int(item) for item in list(replay_data.get("frame_indices") or [])][:num_frames]
    if len(frame_indices) < num_frames:
        frame_indices = list(range(num_frames))

    visualization_config = MuscleGraphVisualizationConfig(
        visualization_mode=str(visualization_mode),
        dominant_selection_mode=str(dominant_selection_mode),
        graph_preset=str(graph_preset),
        node_activation_threshold=float(node_activation_threshold),
        edge_score_threshold=float(edge_score_threshold),
        scoring_mode=str(dynamic_edge_mode),
        rolling_window_size=int(dynamic_edge_window),
        visible_edge_top_k=int(visible_edge_top_k),
        edge_relation_filter=str(edge_relation_filter),
        show_labels=bool(show_labels),
        highlight_top_nodes=int(highlight_top_nodes),
        dominant_mode=bool(dominant_mode),
        dominant_top_k=int(dominant_top_k),
        dominant_edge_top_k=int(dominant_edge_top_k),
        include_thumb_in_dominant_subset=bool(include_thumb_in_dominant_subset),
        hide_inactive_dominant_nodes=bool(hide_inactive_dominant_nodes),
        ghost_non_dominant_nodes=bool(ghost_non_dominant_nodes),
    )
    visualization_config = apply_visualization_preset(visualization_config, graph_preset)
    visualization_config = apply_visualization_mode(visualization_config, visualization_mode)
    visualization_config = apply_dominant_selection_mode(visualization_config, dominant_selection_mode)

    diag = diagnostic_options or RuntimeDiagnosticOptions()

    def _build_runtime() -> RealtimeMuscleGraph:
        return RealtimeMuscleGraph(
            env_id=DEFAULT_MYOSUITE_ENV_ID,
            dynamic_edge_config=visualization_config.to_dynamic_edge_config(),
            visualization_config=visualization_config,
            debug_dynamic_edges=bool(debug_dynamic_edges),
        )

    runtime = _build_runtime()
    canvas = MuscleGraphCanvas(
        runtime.get_static_payload(),
        config=visualization_config,
        panel_update_fps=float(graph_panel_fps),
        show_side_panels=not bool(diag.disable_side_panels),
    )
    canvas.name = "Muscle Graph Replay"
    canvas.ensure_window()

    print(f"[INFO] Replay source: {replay_data['path']} ({replay_data['source_kind']}, {num_frames} frames)")
    print("[INFO] Replay controls: SPACE pause/resume, R restart, Q or ESC exit.")

    frame_ptr = 0
    paused = False
    speed = float(max(0.05, replay_speed))
    try:
        while True:
            graph_state = runtime.update(
                qpos_values=qpos_seq[frame_ptr],
                ctrl_values=ctrl_seq[frame_ptr],
                ctrl_summary={"ctrl_analysis": np.asarray(ctrl_seq[frame_ptr], dtype=np.float32)},
                timestamp=timestamps[frame_ptr],
                frame_index=frame_indices[frame_ptr],
            )
            canvas.render(graph_state, now=timestamps[frame_ptr], force=True)

            x1 = min(canvas.width - 14, 860)
            cv2.rectangle(canvas.output_image, (14, canvas.height - 42), (x1, canvas.height - 12), (20, 20, 20), -1)
            cv2.rectangle(canvas.output_image, (14, canvas.height - 42), (x1, canvas.height - 12), (0, 255, 255), 1)
            cv2.putText(
                canvas.output_image,
                f"{replay_data['label']} | frame {frame_ptr + 1}/{num_frames} | {'paused' if paused else 'playing'} | x{speed:.2f}",
                (24, canvas.height - 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(canvas.name, canvas.output_image)

            if paused or frame_ptr >= (num_frames - 1):
                wait_ms = 30
            else:
                dt = max(1.0 / max(1.0, float(graph_render_fps)), float(timestamps[frame_ptr + 1]) - float(timestamps[frame_ptr]))
                wait_ms = max(1, int(round((dt / speed) * 1000.0)))

            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key == ord("r"):
                runtime = _build_runtime()
                frame_ptr = 0
                paused = False
                continue

            if paused:
                continue
            if frame_ptr >= (num_frames - 1):
                if bool(replay_loop):
                    runtime = _build_runtime()
                    frame_ptr = 0
                    continue
                paused = True
                continue
            frame_ptr += 1
    finally:
        try:
            cv2.destroyWindow(canvas.name)
        except Exception:
            pass

    return 0

# ---------------------------------------------------------------------------
# Runtime Entry
# ---------------------------------------------------------------------------
def parse_runtime_args():
    parser = argparse.ArgumentParser(description="Leap Motion to MyoSuite realtime hand mapping.")
    parser.add_argument(
        "--ctrl-validation-window",
        action="store_true",
        help="Open a second MyoSuite window driven by estimated muscle ctrl for validation.",
    )
    parser.add_argument(
        "--muscle-graph-window",
        action="store_true",
        help="Open a live muscle graph window with per-frame node updates.",
    )
    parser.add_argument("--dynamic-edge-mode", default=DEFAULT_SCORING_MODE, help="Dynamic edge scoring mode: coactivation, rolling_correlation, or hybrid. rolling_corr is accepted as an alias.")
    parser.add_argument("--node-activation-threshold", type=float, default=DEFAULT_NODE_ACTIVATION_THRESHOLD, help="Minimum activation required for both endpoint nodes to show an edge.")
    parser.add_argument("--edge-score-threshold", type=float, default=DEFAULT_EDGE_SCORE_THRESHOLD, help="Minimum dynamic edge score required to render an edge.")
    parser.add_argument("--dynamic-edge-window", type=int, default=DEFAULT_WINDOW_SIZE, help="Rolling window size for dynamic edge scoring modes.")
    parser.add_argument("--visible-edge-top-k", type=int, default=DEFAULT_VISIBLE_EDGE_TOP_K, help="Maximum number of strongest visible edges to render per frame. Use <=0 to disable the cap.")
    parser.add_argument("--edge-relation-filter", default=DEFAULT_EDGE_RELATION_FILTER, choices=("all_active", "synergist", "antagonist"), help="Filter visible edges by relation type.")
    parser.add_argument("--highlight-top-nodes", type=int, default=DEFAULT_HIGHLIGHT_TOP_NODES, help="Number of most active nodes to emphasize visually.")
    parser.add_argument("--visualization-mode", default=DEFAULT_VISUALIZATION_MODE, choices=VALID_VISUALIZATION_MODES, help="Visualization mode: dominant, full, synergy_only, or antagonist_only.")
    parser.add_argument("--dominant-selection-mode", default=DEFAULT_DOMINANT_SELECTION_MODE, choices=VALID_DOMINANT_SELECTION_MODES, help="How dominant nodes are chosen: functional_subset or top_k_ctrl.")
    parser.add_argument("--dominant-mode", action="store_const", const="dominant", dest="visualization_mode", help="Alias for --visualization-mode dominant.")
    parser.add_argument("--full-mode", action="store_const", const="full", dest="visualization_mode", help="Show the full debug graph without dominant filtering.")
    parser.add_argument("--synergy-only-mode", action="store_const", const="synergy_only", dest="visualization_mode", help="Dominant-mode rendering restricted to synergist edges.")
    parser.add_argument("--antagonist-only-mode", action="store_const", const="antagonist_only", dest="visualization_mode", help="Dominant-mode rendering restricted to antagonist edges.")
    parser.add_argument("--dominant-top-k", type=int, default=DEFAULT_DOMINANT_TOP_K, help="Number of dominant muscles to keep prominent in dominant mode.")
    parser.add_argument("--dominant-edge-top-k", type=int, default=DEFAULT_DOMINANT_EDGE_TOP_K, help="Maximum number of dominant-mode edges to render per frame.")
    parser.add_argument("--no-thumb-dominant-subset", action="store_true", help="Exclude thumb muscles from the functional dominant subset.")
    parser.add_argument("--show-inactive-dominant-nodes", action="store_true", help="Keep dominant-subset nodes visible even when they are inactive and disconnected.")
    parser.add_argument("--no-ghost-non-dominant-nodes", action="store_true", help="Hide non-dominant nodes instead of rendering them as dim ghost nodes.")
    parser.add_argument("--graph-preset", default=DEFAULT_GRAPH_PRESET, choices=VALID_GRAPH_PRESETS, help="Apply a locked visualization preset for cleaner analysis figures.")
    parser.add_argument("--debug-dynamic-edges", action="store_true", help="Print top active dynamic edges during runtime.")
    parser.add_argument("--hide-labels", action="store_true", help="Hide node labels in the muscle graph window.")
    parser.add_argument("--ctrl-mode", default=DEFAULT_CTRL_MODE, choices=("control", "analysis"), help="Realtime ctrl post-processing mode.")
    parser.add_argument("--qpos-deadband-norm", type=float, default=DEFAULT_QPOS_DEADBAND_NORM, help="If qpos target change is below this norm, reuse the last stable ctrl.")
    parser.add_argument("--ctrl-smoothing-alpha", type=float, default=DEFAULT_CTRL_SMOOTHING_ALPHA, help="EMA alpha for ctrl smoothing.")
    parser.add_argument("--ctrl-rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW, help="Rolling window for ctrl analysis outputs.")
    parser.add_argument("--no-hold-small-change", action="store_true", help="Disable deadband-based hold when qpos changes are tiny.")
    parser.add_argument("--freeze-control", action="store_true", help="Freeze q_target and ctrl for static pose inspection.")
    parser.add_argument("--tracking-tuning-preset", default=DEFAULT_TRACKING_TUNING_PRESET, choices=tuple(TRACKING_TUNING_PRESETS.keys()), help="Focused tuning preset for grasp, thumb opposition, or pinch testing.")
    parser.add_argument("--hand-mapping-debug", action="store_true", help="Print focused thumb and distal-joint mapping debug lines for a short tuning session.")
    parser.add_argument("--hand-mapping-debug-interval", type=float, default=DEFAULT_HAND_MAPPING_DEBUG_INTERVAL, help="Seconds between hand-mapping debug prints.")
    parser.add_argument("--hand-mapping-debug-seconds", type=float, default=DEFAULT_HAND_MAPPING_DEBUG_SECONDS, help="How long hand-mapping debug prints stay enabled after startup.")
    parser.add_argument("--graph-render-fps", type=float, default=DEFAULT_GRAPH_RENDER_FPS, help="Throttle the muscle graph redraw cadence without slowing controller updates.")
    parser.add_argument("--graph-panel-fps", type=float, default=DEFAULT_GRAPH_PANEL_FPS, help="Throttle right-side graph panel text and bars separately from node/edge redraw.")
    parser.add_argument("--profile-interval", type=float, default=DEFAULT_RUNTIME_PROFILE_INTERVAL, help="Seconds between rolling runtime timing summaries.")
    parser.add_argument("--no-runtime-profiler", action="store_true", help="Disable runtime timing instrumentation and overlay.")
    parser.add_argument("--exercise-name", type=str, choices=("fist", "claw_stretch", "making_o"), default=None, help="Exercise label to attach to recordings.")
    parser.add_argument("--trial-id", type=str, default=None, help="Trial identifier to attach to recordings.")
    parser.add_argument("--replay-record", type=str, default=None, help="Replay a muscle graph from a recorded .csv, .pt, or .summary.json artifact.")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Playback speed multiplier for --replay-record.")
    parser.add_argument("--replay-loop", action="store_true", help="Loop playback when a replay reaches the final frame.")
    parser.add_argument("--diagnostic-mode", action="store_true", help="Run an automated bottleneck experiment instead of the live Leap runtime.")
    parser.add_argument("--diagnostic-seconds", type=float, default=2.5, help="Duration per automated bottleneck experiment mode.")
    parser.add_argument("--diag-disable-graph-update", action="store_true", help="Diagnostic toggle: skip realtime graph update work.")
    parser.add_argument("--diag-disable-graph-draw", action="store_true", help="Diagnostic toggle: skip graph canvas drawing work.")
    parser.add_argument("--diag-disable-side-panels", action="store_true", help="Diagnostic toggle: hide graph side panels to isolate node/edge draw cost.")
    parser.add_argument("--diag-disable-controller", action="store_true", help="Diagnostic toggle: reuse previous ctrl instead of recomputing controller output.")
    parser.add_argument("--diag-disable-validation-window", action="store_true", help="Diagnostic toggle: skip validation env stepping/rendering.")
    parser.add_argument("--diag-disable-main-render", action="store_true", help="Diagnostic toggle: skip main env mj_render work.")
    parser.add_argument("--diag-disable-cv-overlay", action="store_true", help="Diagnostic toggle: disable the lightweight on-screen profiler overlay.")
    return parser.parse_args()


def main(
    show_ctrl_validation_window=False,
    show_muscle_graph_window=False,
    dynamic_edge_mode=DEFAULT_SCORING_MODE,
    node_activation_threshold=DEFAULT_NODE_ACTIVATION_THRESHOLD,
    edge_score_threshold=DEFAULT_EDGE_SCORE_THRESHOLD,
    dynamic_edge_window=DEFAULT_WINDOW_SIZE,
    visible_edge_top_k=DEFAULT_VISIBLE_EDGE_TOP_K,
    edge_relation_filter=DEFAULT_EDGE_RELATION_FILTER,
    highlight_top_nodes=DEFAULT_HIGHLIGHT_TOP_NODES,
    visualization_mode=DEFAULT_VISUALIZATION_MODE,
    dominant_selection_mode=DEFAULT_DOMINANT_SELECTION_MODE,
    dominant_mode=DEFAULT_DOMINANT_MODE,
    dominant_top_k=DEFAULT_DOMINANT_TOP_K,
    dominant_edge_top_k=DEFAULT_DOMINANT_EDGE_TOP_K,
    include_thumb_in_dominant_subset=True,
    hide_inactive_dominant_nodes=True,
    ghost_non_dominant_nodes=DEFAULT_GHOST_NON_DOMINANT_NODES,
    graph_preset=DEFAULT_GRAPH_PRESET,
    debug_dynamic_edges=False,
    show_labels=DEFAULT_SHOW_LABELS,
    ctrl_mode=DEFAULT_CTRL_MODE,
    qpos_deadband_norm=DEFAULT_QPOS_DEADBAND_NORM,
    ctrl_smoothing_alpha=DEFAULT_CTRL_SMOOTHING_ALPHA,
    ctrl_rolling_window=DEFAULT_ROLLING_WINDOW,
    hold_if_small_change=DEFAULT_HOLD_IF_SMALL_CHANGE,
    freeze_control=False,
    tracking_tuning_preset=DEFAULT_TRACKING_TUNING_PRESET,
    hand_mapping_debug=False,
    hand_mapping_debug_interval=DEFAULT_HAND_MAPPING_DEBUG_INTERVAL,
    hand_mapping_debug_seconds=DEFAULT_HAND_MAPPING_DEBUG_SECONDS,
    graph_render_fps=DEFAULT_GRAPH_RENDER_FPS,
    graph_panel_fps=DEFAULT_GRAPH_PANEL_FPS,
    exercise_name=None,
    trial_id=None,
    profile_interval=DEFAULT_RUNTIME_PROFILE_INTERVAL,
    runtime_profiler_enabled=True,
    diagnostic_options: RuntimeDiagnosticOptions | None = None,
):
    diagnostic_options = diagnostic_options or RuntimeDiagnosticOptions()
    tuning_config = apply_tracking_tuning_preset(
        tracking_tuning_preset,
        qpos_deadband_norm=float(qpos_deadband_norm),
        ctrl_smoothing_alpha=float(ctrl_smoothing_alpha),
        hold_if_small_change=bool(hold_if_small_change),
        hand_mapping_debug=bool(hand_mapping_debug),
    )
    qpos_deadband_norm = float(tuning_config["qpos_deadband_norm"])
    ctrl_smoothing_alpha = float(tuning_config["ctrl_smoothing_alpha"])
    hold_if_small_change = bool(tuning_config["hold_if_small_change"])
    hand_mapping_debug = bool(tuning_config["hand_mapping_debug"])
    if str(tuning_config["preset"]) != DEFAULT_TRACKING_TUNING_PRESET:
        print(
            "[INFO] Tracking tuning preset "
            f"{tuning_config['preset']} | qdn={qpos_deadband_norm:.3f} "
            f"alpha={ctrl_smoothing_alpha:.2f} hold={int(bool(hold_if_small_change))} "
            f"mapping_debug={int(bool(hand_mapping_debug))}"
        )
    tracking_mode = leap.TrackingMode.Desktop
    connection = leap.Connection()
    canvas = Canvas() if VISUALIZE_LEAP else None
    runtime_profiler = RollingRuntimeProfiler(log_interval_seconds=float(profile_interval)) if bool(runtime_profiler_enabled) else None
    mapper = LeapToMyoSuiteMapper(desktop_mode_transform=tracking_mode == leap.TrackingMode.Desktop)

    if not MYOSUITE_AVAILABLE:
        print("[WARN] MyoSuite not available.")

    env = create_myosuite_env(
        gym_module=gym,
        myosuite_available=MYOSUITE_AVAILABLE,
        env_id=DEFAULT_MYOSUITE_ENV_ID,
        info_label="MyoSuite env created.",
        warn_label="MyoSuite env",
    )

    enable_ctrl_env = bool(show_ctrl_validation_window) and not bool(diagnostic_options.disable_validation_window)
    env_ctrl = None
    if enable_ctrl_env and MYOSUITE_AVAILABLE:
        env_ctrl = create_myosuite_env(
            gym_module=gym,
            myosuite_available=MYOSUITE_AVAILABLE,
            env_id=DEFAULT_MYOSUITE_ENV_ID,
            info_label="Ctrl-driven validation env created.",
            warn_label="ctrl env",
        )
    elif MYOSUITE_AVAILABLE:
        print("[INFO] Ctrl-driven validation window disabled. Use --ctrl-validation-window to enable it.")

    ctrl_env_context = create_ctrl_env_context(env_ctrl)
    if ctrl_env_context["data_source"] is None and env is not None:
        ctrl_env_context["data_source"] = env.sim.data

    controller_context = create_controller_context(
        DEFAULT_MYOSUITE_ENV_ID,
        estimation_config=CtrlEstimationConfig(
            qpos_deadband_norm=float(qpos_deadband_norm),
            ctrl_smoothing_alpha=float(ctrl_smoothing_alpha),
            hold_if_small_change=bool(hold_if_small_change),
            rolling_window=int(ctrl_rolling_window),
            mode=str(ctrl_mode),
            freeze_mode=bool(freeze_control),
        ),
        exercise_name=exercise_name,
    )
    validation_controller_context = None
    if env_ctrl is not None:
        validation_controller_context = create_controller_context(
            DEFAULT_MYOSUITE_ENV_ID,
            estimation_config=CtrlEstimationConfig(
                qpos_deadband_norm=0.0,
                ctrl_smoothing_alpha=float(ctrl_smoothing_alpha),
                hold_if_small_change=False,
                rolling_window=max(1, min(3, int(ctrl_rolling_window))),
                mode="control",
                freeze_mode=False,
            ),
            exercise_name=exercise_name,
        )
    muscle_graph_runtime = None
    muscle_graph_canvas = None
    visualization_config = MuscleGraphVisualizationConfig(
        visualization_mode=str(visualization_mode),
        dominant_selection_mode=str(dominant_selection_mode),
        graph_preset=str(graph_preset),
        node_activation_threshold=float(node_activation_threshold),
        edge_score_threshold=float(edge_score_threshold),
        scoring_mode=str(dynamic_edge_mode),
        rolling_window_size=int(dynamic_edge_window),
        visible_edge_top_k=int(visible_edge_top_k),
        edge_relation_filter=str(edge_relation_filter),
        show_labels=bool(show_labels),
        highlight_top_nodes=int(highlight_top_nodes),
        dominant_mode=bool(dominant_mode),
        dominant_top_k=int(dominant_top_k),
        dominant_edge_top_k=int(dominant_edge_top_k),
        include_thumb_in_dominant_subset=bool(include_thumb_in_dominant_subset),
        hide_inactive_dominant_nodes=bool(hide_inactive_dominant_nodes),
        ghost_non_dominant_nodes=bool(ghost_non_dominant_nodes),
    )
    visualization_config = apply_visualization_preset(visualization_config, graph_preset)
    visualization_config = apply_visualization_mode(visualization_config, visualization_mode)
    visualization_config = apply_dominant_selection_mode(visualization_config, dominant_selection_mode)
    try:
        muscle_graph_runtime = RealtimeMuscleGraph(
            env_id=DEFAULT_MYOSUITE_ENV_ID,
            dynamic_edge_config=visualization_config.to_dynamic_edge_config(),
            visualization_config=visualization_config,
            debug_dynamic_edges=bool(debug_dynamic_edges),
        )
        if show_muscle_graph_window:
            muscle_graph_canvas = MuscleGraphCanvas(
                muscle_graph_runtime.get_static_payload(),
                config=visualization_config,
                panel_update_fps=float(graph_panel_fps),
                show_side_panels=not bool(diagnostic_options.disable_side_panels),
            )
            muscle_graph_canvas.ensure_window()
        print("[INFO] Realtime muscle graph initialized.")
    except Exception as e:
        print(f"[WARN] Failed to initialize realtime muscle graph: {e}")

    recorder = RecordingManager(
        env,
        static_graph=muscle_graph_runtime.get_static_payload() if muscle_graph_runtime is not None else None,
        exercise_name=exercise_name,
        trial_id=trial_id,
    )
    listener = LeapDataListener(
        mapper=mapper,
        canvas=canvas,
        reset_callback=reset_runtime_tracking,
        runtime_profiler=runtime_profiler,
        hand_mapping_debug=bool(hand_mapping_debug),
        hand_mapping_debug_interval=float(hand_mapping_debug_interval),
        hand_mapping_debug_seconds=float(hand_mapping_debug_seconds),
        hand_mapping_debug_label=str(tuning_config["preset"]),
        auto_reset_on_track_loss=AUTO_RESET_ON_TRACK_LOSS,
        track_reacquire_reset_seconds=TRACK_REACQUIRE_RESET_SECONDS,
        min_reset_interval_seconds=MIN_RESET_INTERVAL_SECONDS,
    )
    connection.add_listener(listener)

    try:
        with connection.open():
            connection.set_tracking_mode(tracking_mode)
            mapper.desktop_mode_transform = tracking_mode == leap.TrackingMode.Desktop

            leap_ok = setup_leap_motion_device(connection)
            print(f"[INFO] Leap device available: {bool(leap_ok)}")
            if not leap_ok:
                return

            running = True
            last_render_time = time.time()
            last_ctrl_print_time = 0.0
            ctrl_print_interval = 1.0
            live_frame_index = 0
            graph_render_interval = 1.0 / max(1.0, float(graph_render_fps))
            next_graph_render_time = 0.0
            last_graph_show_time = 0.0
            graph_window_dirty = bool(muscle_graph_canvas is not None)
            latest_graph_state = None
            ctrl_log_state = {
                "controller_logged": False,
                "fallback_logged": False,
            }
            validation_ctrl_log_state = {
                "controller_logged": False,
                "fallback_logged": False,
            }

            print("[INFO] Entering main loop. Press 'x' in the visualizer window to exit.")

            while running:
                now = time.time()
                if runtime_profiler is not None:
                    runtime_profiler.set_runtime_flags(
                        recording_active=bool(recorder.recording),
                        ctrl_validation_enabled=bool(enable_ctrl_env),
                        muscle_graph_enabled=bool(show_muscle_graph_window),
                    )
                listener.runtime_frame_hint = int(live_frame_index)
                frame_started = False
                frame_start_perf = 0.0
                reset_reason = get_tracking_loss_reset_reason(listener, now)
                if reset_reason is not None:
                    reset_runtime_tracking(listener, reset_reason)
                    if env_ctrl is not None:
                        reset_ctrl_env_state(env_ctrl)
                        ctrl_env_context["needs_reset"] = False
                    if validation_controller_context is not None:
                        reset_controller_context_state(validation_controller_context, data_source=env_ctrl.sim.data if env_ctrl is not None else None)

                if env is not None and ENABLE_MYOSUITE_RENDER and (now - last_render_time) >= (1.0 / float(MYOSUITE_RENDER_FPS)):
                    current_frame_index = int(live_frame_index)
                    frame_started = True
                    frame_start_perf = time.perf_counter()
                    if runtime_profiler is not None:
                        runtime_profiler.start_frame(current_frame_index)
                    q_target = getattr(listener, "latest_qpos", None)
                    if q_target is not None:
                        q_target = np.asarray(q_target, dtype=float)
                        q_target = prepare_target_qpos(q_target, controller_context)
                        invalid_reason = get_invalid_qpos_reset_reason(q_target)
                        if invalid_reason is not None:
                            update_listener_hud(listener, "INVALID POSE", (0, 165, 255))
                            reset_runtime_tracking(listener, invalid_reason)
                            if env_ctrl is not None:
                                reset_ctrl_env_state(env_ctrl)
                                ctrl_env_context["needs_reset"] = False
                            if validation_controller_context is not None:
                                reset_controller_context_state(validation_controller_context, data_source=env_ctrl.sim.data if env_ctrl is not None else None)
                            q_target = None

                    if q_target is not None:
                        ctrl_state = None
                        controller_t0 = time.perf_counter()
                        if bool(diagnostic_options.disable_controller):
                            ctrl_state = build_diagnostic_ctrl_state(q_target, controller_context)
                        elif env_ctrl is not None and validation_controller_context is not None:
                            if bool(ctrl_env_context.get("needs_reset")):
                                reset_ctrl_env_state(env_ctrl)
                                ctrl_env_context["needs_reset"] = False
                                reset_controller_context_state(validation_controller_context, data_source=env_ctrl.sim.data)
                            steps_per_render = int(ctrl_env_context["steps_per_render"]) * int(ctrl_env_context.get("speedup", 1))
                            for _ in range(max(1, steps_per_render)):
                                ctrl_state_iter = estimate_ctrl_signals(
                                    q_target=q_target,
                                    controller_context=validation_controller_context,
                                    ctrl_data_source=env_ctrl.sim.data,
                                    ctrl_log_state=validation_ctrl_log_state,
                                )
                                if ctrl_state_iter is None:
                                    continue
                                ctrl_state = dict(ctrl_state_iter)
                                step_ctrl_env(
                                    env_ctrl=env_ctrl,
                                    ctrl_steps_per_render=1,
                                    ctrl_vec=ctrl_state_iter["ctrl_raw"],
                                    render=False,
                                )
                            try:
                                env_ctrl_render_t0 = time.perf_counter()
                                env_ctrl.sim.forward()
                                env_ctrl.mj_render()
                                if runtime_profiler is not None:
                                    runtime_profiler.add_frame_stage("validation_env_render", (time.perf_counter() - env_ctrl_render_t0) * 1000.0)
                            except Exception:
                                pass
                        else:
                            ctrl_state = estimate_ctrl_signals(
                                q_target=q_target,
                                controller_context=controller_context,
                                ctrl_data_source=ctrl_env_context["data_source"],
                                ctrl_log_state=ctrl_log_state,
                            )
                        if runtime_profiler is not None:
                            runtime_profiler.add_frame_stage("controller", (time.perf_counter() - controller_t0) * 1000.0)
                        if ctrl_state is not None:
                            listener.latest_ctrl_raw = np.asarray(ctrl_state["ctrl_raw"], dtype=float)
                            listener.latest_ctrl_smoothed = np.asarray(ctrl_state["ctrl_smoothed"], dtype=float)
                            listener.latest_ctrl_analysis = np.asarray(ctrl_state["ctrl_analysis"], dtype=float)
                            ctrl_validation = ctrl_state["ctrl_raw"]
                            listener.latest_ctrl_validation = np.asarray(ctrl_validation, dtype=float) if ctrl_validation is not None else None
                            listener.latest_ctrl = np.asarray(ctrl_state["ctrl_analysis"], dtype=float)
                            listener.latest_ctrl_summary = dict(ctrl_state)
                            q_target = np.asarray(ctrl_state["q_target_used"], dtype=float)
                        apply_qpos_to_view_env(env, q_target)

                        ctrl_state_now = getattr(listener, "latest_ctrl_summary", None)
                        if ctrl_state_now is not None and (now - last_ctrl_print_time) >= ctrl_print_interval:
                            print_ctrl_summary(ctrl_state_now)
                            last_ctrl_print_time = now

                    ctrl_for_view = getattr(listener, "latest_ctrl_analysis", None)
                    graph_state = None
                    if (
                        q_target is not None
                        and ctrl_for_view is not None
                        and muscle_graph_runtime is not None
                        and not bool(diagnostic_options.disable_graph_update)
                    ):
                        try:
                            graph_update_t0 = time.perf_counter()
                            graph_state = muscle_graph_runtime.update(
                                qpos_values=q_target,
                                ctrl_values=ctrl_for_view,
                                ctrl_summary=getattr(listener, "latest_ctrl_summary", None),
                                timestamp=now,
                                frame_index=current_frame_index,
                            )
                            if runtime_profiler is not None:
                                runtime_profiler.add_frame_stage("graph_update", (time.perf_counter() - graph_update_t0) * 1000.0)
                            listener.latest_muscle_graph = graph_state
                            latest_graph_state = graph_state
                            if (
                                muscle_graph_canvas is not None
                                and now >= next_graph_render_time
                                and not bool(diagnostic_options.disable_graph_draw)
                            ):
                                canvas_draw_t0 = time.perf_counter()
                                graph_window_dirty = bool(muscle_graph_canvas.render(graph_state, now=now)) or graph_window_dirty
                                if runtime_profiler is not None:
                                    runtime_profiler.add_frame_stage("graph_canvas_draw", (time.perf_counter() - canvas_draw_t0) * 1000.0)
                                next_graph_render_time = now + graph_render_interval
                        except Exception as e:
                            print(f"[WARN] Failed to update realtime muscle graph: {type(e).__name__}: {e}")
                    update_main_env_ctrl_view(env, ctrl_for_view)

                    if not bool(diagnostic_options.disable_main_env_render):
                        render_t0 = time.perf_counter()
                        env.sim.forward()
                        env.mj_render()
                        if runtime_profiler is not None:
                            runtime_profiler.add_frame_stage("main_env_render", (time.perf_counter() - render_t0) * 1000.0)
                    if env_ctrl is not None and q_target is None and not bool(diagnostic_options.disable_validation_window):
                        validation_render_t0 = time.perf_counter()
                        step_ctrl_env(
                            env_ctrl=env_ctrl,
                            ctrl_steps_per_render=1,
                            ctrl_vec=np.zeros(int(env_ctrl.sim.data.ctrl.shape[0]), dtype=float),
                            render=True,
                        )
                        if runtime_profiler is not None:
                            runtime_profiler.add_frame_stage("validation_env_render", (time.perf_counter() - validation_render_t0) * 1000.0)

                    recording_t0 = time.perf_counter()
                    recorder.write_frame(getattr(listener, "latest_ctrl_summary", None), graph_state=graph_state)
                    if runtime_profiler is not None:
                        runtime_profiler.add_frame_stage("recording", (time.perf_counter() - recording_t0) * 1000.0)
                    last_render_time = now
                    live_frame_index += 1

                imshow_t0 = time.perf_counter()
                if runtime_profiler is not None and not bool(diagnostic_options.disable_cv_overlay):
                    if canvas is not None:
                        draw_runtime_profiler_overlay(canvas.output_image, runtime_profiler)
                    if muscle_graph_canvas is not None:
                        draw_runtime_profiler_overlay(muscle_graph_canvas.output_image, runtime_profiler)
                if canvas is not None:
                    cv2.imshow(canvas.name, canvas.output_image)
                if muscle_graph_canvas is not None and (graph_window_dirty or (now - last_graph_show_time) >= 0.25):
                    cv2.imshow(muscle_graph_canvas.name, muscle_graph_canvas.output_image)
                    graph_window_dirty = False
                    last_graph_show_time = now

                key = cv2.waitKey(1) & 0xFF if (canvas is not None or muscle_graph_canvas is not None) else 255
                if runtime_profiler is not None and frame_started:
                    runtime_profiler.add_frame_stage("gui_refresh", (time.perf_counter() - imshow_t0) * 1000.0)
                    runtime_profiler.finish_frame((time.perf_counter() - frame_start_perf) * 1000.0)
                    runtime_profiler.maybe_log(now)
                running = handle_runtime_key(key=key, recorder=recorder, listener=listener)
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            connection.remove_listener(listener)
        except Exception:
            pass
        recorder.stop()
        if VISUALIZE_LEAP or muscle_graph_canvas is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def _synthetic_q_target(frame_index: int, width: int) -> np.ndarray:
    t = float(frame_index) / max(1.0, float(MYOSUITE_RENDER_FPS))
    q_target = np.zeros(max(0, int(width)), dtype=float)
    if q_target.size == 0:
        return q_target
    for idx in range(q_target.size):
        freq = 0.7 + (0.03 * float(idx))
        phase = 0.18 * float(idx)
        q_target[idx] = 0.22 * math.sin((t * freq * 2.0 * math.pi) + phase)
    return q_target


def _merge_diagnostic_options(base: RuntimeDiagnosticOptions, **overrides: Any) -> RuntimeDiagnosticOptions:
    merged = RuntimeDiagnosticOptions(**asdict(base))
    for key, value in overrides.items():
        if hasattr(merged, key):
            setattr(merged, key, bool(value))
    return merged


def _run_single_diagnostic_trial(
    *,
    label: str,
    duration_seconds: float,
    show_ctrl_validation_window: bool,
    show_muscle_graph_window: bool,
    dynamic_edge_mode: str,
    node_activation_threshold: float,
    edge_score_threshold: float,
    dynamic_edge_window: int,
    visible_edge_top_k: int,
    edge_relation_filter: str,
    highlight_top_nodes: int,
    visualization_mode: str,
    dominant_selection_mode: str,
    dominant_mode: bool,
    dominant_top_k: int,
    dominant_edge_top_k: int,
    include_thumb_in_dominant_subset: bool,
    hide_inactive_dominant_nodes: bool,
    ghost_non_dominant_nodes: bool,
    graph_preset: str,
    debug_dynamic_edges: bool,
    show_labels: bool,
    ctrl_mode: str,
    qpos_deadband_norm: float,
    ctrl_smoothing_alpha: float,
    ctrl_rolling_window: int,
    hold_if_small_change: bool,
    freeze_control: bool,
    tracking_tuning_preset: str,
    hand_mapping_debug: bool,
    hand_mapping_debug_interval: float,
    hand_mapping_debug_seconds: float,
    graph_render_fps: float,
    graph_panel_fps: float,
    diagnostic_options: RuntimeDiagnosticOptions,
) -> dict[str, Any]:
    runtime_profiler = RollingRuntimeProfiler(
        log_interval_seconds=max(60.0, float(duration_seconds) * 10.0),
        log_spikes=False,
    )
    runtime_profiler.set_runtime_flags(
        recording_active=False,
        ctrl_validation_enabled=bool(show_ctrl_validation_window and not diagnostic_options.disable_validation_window),
        muscle_graph_enabled=bool(show_muscle_graph_window),
    )
    env = create_myosuite_env(
        gym_module=gym,
        myosuite_available=MYOSUITE_AVAILABLE,
        env_id=DEFAULT_MYOSUITE_ENV_ID,
        info_label=f"Diagnostic env created ({label}).",
        warn_label=f"diagnostic env {label}",
    )
    env_ctrl = None
    if bool(show_ctrl_validation_window) and not bool(diagnostic_options.disable_validation_window):
        env_ctrl = create_myosuite_env(
            gym_module=gym,
            myosuite_available=MYOSUITE_AVAILABLE,
            env_id=DEFAULT_MYOSUITE_ENV_ID,
            info_label=f"Diagnostic ctrl env created ({label}).",
            warn_label=f"diagnostic ctrl env {label}",
        )
    ctrl_env_context = create_ctrl_env_context(env_ctrl)
    if ctrl_env_context["data_source"] is None and env is not None:
        ctrl_env_context["data_source"] = env.sim.data

    controller_context = create_controller_context(
        DEFAULT_MYOSUITE_ENV_ID,
        estimation_config=CtrlEstimationConfig(
            qpos_deadband_norm=float(qpos_deadband_norm),
            ctrl_smoothing_alpha=float(ctrl_smoothing_alpha),
            hold_if_small_change=bool(hold_if_small_change),
            rolling_window=int(ctrl_rolling_window),
            mode=str(ctrl_mode),
            freeze_mode=bool(freeze_control),
        ),
        exercise_name=exercise_name,
    )
    validation_controller_context = None
    if env_ctrl is not None:
        validation_controller_context = create_controller_context(
            DEFAULT_MYOSUITE_ENV_ID,
            estimation_config=CtrlEstimationConfig(
                qpos_deadband_norm=0.0,
                ctrl_smoothing_alpha=float(ctrl_smoothing_alpha),
                hold_if_small_change=False,
                rolling_window=max(1, min(3, int(ctrl_rolling_window))),
                mode="control",
                freeze_mode=False,
            ),
            exercise_name=exercise_name,
        )

    visualization_config = MuscleGraphVisualizationConfig(
        visualization_mode=str(visualization_mode),
        dominant_selection_mode=str(dominant_selection_mode),
        graph_preset=str(graph_preset),
        node_activation_threshold=float(node_activation_threshold),
        edge_score_threshold=float(edge_score_threshold),
        scoring_mode=str(dynamic_edge_mode),
        rolling_window_size=int(dynamic_edge_window),
        visible_edge_top_k=int(visible_edge_top_k),
        edge_relation_filter=str(edge_relation_filter),
        show_labels=bool(show_labels),
        highlight_top_nodes=int(highlight_top_nodes),
        dominant_mode=bool(dominant_mode),
        dominant_top_k=int(dominant_top_k),
        dominant_edge_top_k=int(dominant_edge_top_k),
        include_thumb_in_dominant_subset=bool(include_thumb_in_dominant_subset),
        hide_inactive_dominant_nodes=bool(hide_inactive_dominant_nodes),
        ghost_non_dominant_nodes=bool(ghost_non_dominant_nodes),
    )
    visualization_config = apply_visualization_preset(visualization_config, graph_preset)
    visualization_config = apply_visualization_mode(visualization_config, visualization_mode)
    visualization_config = apply_dominant_selection_mode(visualization_config, dominant_selection_mode)

    muscle_graph_runtime = RealtimeMuscleGraph(
        env_id=DEFAULT_MYOSUITE_ENV_ID,
        dynamic_edge_config=visualization_config.to_dynamic_edge_config(),
        visualization_config=visualization_config,
        debug_dynamic_edges=bool(debug_dynamic_edges),
    )
    muscle_graph_canvas = None
    if bool(show_muscle_graph_window):
        muscle_graph_canvas = MuscleGraphCanvas(
            muscle_graph_runtime.get_static_payload(),
            config=visualization_config,
            panel_update_fps=float(graph_panel_fps),
            show_side_panels=not bool(diagnostic_options.disable_side_panels),
        )

    frame_count = max(8, int(round(max(0.10, float(duration_seconds)) * float(MYOSUITE_RENDER_FPS))))
    graph_render_interval = 1.0 / max(1.0, float(graph_render_fps))
    next_graph_render_time = 0.0
    ctrl_log_state = {"controller_logged": False, "fallback_logged": False}
    validation_ctrl_log_state = {"controller_logged": False, "fallback_logged": False}
    previous_ctrl = None

    for frame_index in range(frame_count):
        runtime_profiler.start_frame(frame_index)
        frame_now = float(frame_index) / max(1.0, float(MYOSUITE_RENDER_FPS))
        frame_start_perf = time.perf_counter()

        qpos_t0 = time.perf_counter()
        q_target = _synthetic_q_target(frame_index, 23)
        q_target = prepare_target_qpos(q_target, controller_context)
        runtime_profiler.add_event("qpos_mapping", (time.perf_counter() - qpos_t0) * 1000.0, frame_index=frame_index)

        controller_t0 = time.perf_counter()
        ctrl_state = None
        if bool(diagnostic_options.disable_controller):
            ctrl_state = build_diagnostic_ctrl_state(q_target, controller_context, previous_ctrl)
        elif env_ctrl is not None and validation_controller_context is not None:
            steps_per_render = int(ctrl_env_context["steps_per_render"]) * int(ctrl_env_context.get("speedup", 1))
            for _ in range(max(1, steps_per_render)):
                ctrl_state_iter = estimate_ctrl_signals(
                    q_target=q_target,
                    controller_context=validation_controller_context,
                    ctrl_data_source=env_ctrl.sim.data,
                    ctrl_log_state=validation_ctrl_log_state,
                )
                if ctrl_state_iter is None:
                    continue
                ctrl_state = dict(ctrl_state_iter)
                step_ctrl_env(
                    env_ctrl=env_ctrl,
                    ctrl_steps_per_render=1,
                    ctrl_vec=ctrl_state_iter["ctrl_raw"],
                    render=False,
                )
            if env_ctrl is not None:
                validation_render_t0 = time.perf_counter()
                env_ctrl.sim.forward()
                env_ctrl.mj_render()
                runtime_profiler.add_frame_stage("validation_env_render", (time.perf_counter() - validation_render_t0) * 1000.0)
        else:
            ctrl_state = estimate_ctrl_signals(
                q_target=q_target,
                controller_context=controller_context,
                ctrl_data_source=ctrl_env_context["data_source"],
                ctrl_log_state=ctrl_log_state,
            )
        runtime_profiler.add_frame_stage("controller", (time.perf_counter() - controller_t0) * 1000.0)
        if ctrl_state is None:
            ctrl_state = build_diagnostic_ctrl_state(q_target, controller_context, previous_ctrl)
        previous_ctrl = np.asarray(ctrl_state["ctrl_analysis"], dtype=float)

        ctrl_for_view = np.asarray(ctrl_state["ctrl_analysis"], dtype=float)
        graph_state = None
        if not bool(diagnostic_options.disable_graph_update):
            graph_update_t0 = time.perf_counter()
            graph_state = muscle_graph_runtime.update(
                qpos_values=q_target,
                ctrl_values=ctrl_for_view,
                ctrl_summary=ctrl_state,
                timestamp=frame_now,
                frame_index=frame_index,
            )
            runtime_profiler.add_frame_stage("graph_update", (time.perf_counter() - graph_update_t0) * 1000.0)

        if (
            muscle_graph_canvas is not None
            and graph_state is not None
            and not bool(diagnostic_options.disable_graph_draw)
            and frame_now >= next_graph_render_time
        ):
            draw_t0 = time.perf_counter()
            muscle_graph_canvas.render(graph_state, now=frame_now, force=True)
            runtime_profiler.add_frame_stage("graph_canvas_draw", (time.perf_counter() - draw_t0) * 1000.0)
            next_graph_render_time = frame_now + graph_render_interval

        update_main_env_ctrl_view(env, ctrl_for_view)
        if not bool(diagnostic_options.disable_main_env_render):
            render_t0 = time.perf_counter()
            env.sim.forward()
            env.mj_render()
            runtime_profiler.add_frame_stage("main_env_render", (time.perf_counter() - render_t0) * 1000.0)

        runtime_profiler.add_frame_stage("gui_refresh", 0.0)
        runtime_profiler.finish_frame((time.perf_counter() - frame_start_perf) * 1000.0)

    summary = runtime_profiler.summary_snapshot()
    result = {
        "label": label,
        "frame_count": frame_count,
        "summary": summary,
        "diagnostic_options": asdict(diagnostic_options),
    }
    print(
        "[DIAG] "
        f"{label}: avg_total={summary['avg_total_frame_ms']:.2f}ms "
        f"ctrl={summary['avg_controller_ms']:.2f}ms "
        f"graph={summary['avg_graph_ms']:.2f}ms "
        f"draw={summary['avg_draw_ms']:.2f}ms "
        f"render={summary['avg_render_ms']:.2f}ms "
        f"fps={summary['estimated_fps']:.1f}"
    )
    for candidate in (env_ctrl, env):
        if candidate is None:
            continue
        try:
            candidate.close()
        except Exception:
            pass
    return result


def run_bottleneck_diagnostic(
    *,
    duration_seconds: float,
    show_ctrl_validation_window: bool,
    show_muscle_graph_window: bool,
    dynamic_edge_mode: str,
    node_activation_threshold: float,
    edge_score_threshold: float,
    dynamic_edge_window: int,
    visible_edge_top_k: int,
    edge_relation_filter: str,
    highlight_top_nodes: int,
    visualization_mode: str,
    dominant_selection_mode: str,
    dominant_mode: bool,
    dominant_top_k: int,
    dominant_edge_top_k: int,
    include_thumb_in_dominant_subset: bool,
    hide_inactive_dominant_nodes: bool,
    ghost_non_dominant_nodes: bool,
    graph_preset: str,
    debug_dynamic_edges: bool,
    show_labels: bool,
    ctrl_mode: str,
    qpos_deadband_norm: float,
    ctrl_smoothing_alpha: float,
    ctrl_rolling_window: int,
    hold_if_small_change: bool,
    freeze_control: bool,
    tracking_tuning_preset: str,
    hand_mapping_debug: bool,
    hand_mapping_debug_interval: float,
    hand_mapping_debug_seconds: float,
    graph_render_fps: float,
    graph_panel_fps: float,
    base_diagnostic_options: RuntimeDiagnosticOptions,
) -> list[dict[str, Any]]:
    experiments = [
        ("baseline_full", base_diagnostic_options),
        ("no_graph_draw", _merge_diagnostic_options(base_diagnostic_options, disable_graph_draw=True)),
        ("no_graph_update", _merge_diagnostic_options(base_diagnostic_options, disable_graph_update=True)),
        ("no_validation_render", _merge_diagnostic_options(base_diagnostic_options, disable_validation_window=True)),
        ("no_controller_recompute", _merge_diagnostic_options(base_diagnostic_options, disable_controller=True)),
        ("no_cv_overlay", _merge_diagnostic_options(base_diagnostic_options, disable_cv_overlay=True)),
    ]
    results = []
    for label, diagnostic_options in experiments:
        results.append(
            _run_single_diagnostic_trial(
                label=label,
                duration_seconds=float(duration_seconds),
                show_ctrl_validation_window=bool(show_ctrl_validation_window),
                show_muscle_graph_window=bool(show_muscle_graph_window),
                dynamic_edge_mode=str(dynamic_edge_mode),
                node_activation_threshold=float(node_activation_threshold),
                edge_score_threshold=float(edge_score_threshold),
                dynamic_edge_window=int(dynamic_edge_window),
                visible_edge_top_k=int(visible_edge_top_k),
                edge_relation_filter=str(edge_relation_filter),
                highlight_top_nodes=int(highlight_top_nodes),
                visualization_mode=str(visualization_mode),
                dominant_selection_mode=str(dominant_selection_mode),
                dominant_mode=bool(dominant_mode),
                dominant_top_k=int(dominant_top_k),
                dominant_edge_top_k=int(dominant_edge_top_k),
                include_thumb_in_dominant_subset=bool(include_thumb_in_dominant_subset),
                hide_inactive_dominant_nodes=bool(hide_inactive_dominant_nodes),
                ghost_non_dominant_nodes=bool(ghost_non_dominant_nodes),
                graph_preset=str(graph_preset),
                debug_dynamic_edges=bool(debug_dynamic_edges),
                show_labels=bool(show_labels),
                ctrl_mode=str(ctrl_mode),
                qpos_deadband_norm=float(qpos_deadband_norm),
                ctrl_smoothing_alpha=float(ctrl_smoothing_alpha),
                ctrl_rolling_window=int(ctrl_rolling_window),
                hold_if_small_change=bool(hold_if_small_change),
                freeze_control=bool(freeze_control),
                tracking_tuning_preset=str(tracking_tuning_preset),
                hand_mapping_debug=bool(hand_mapping_debug),
                hand_mapping_debug_interval=float(hand_mapping_debug_interval),
                hand_mapping_debug_seconds=float(hand_mapping_debug_seconds),
                graph_render_fps=float(graph_render_fps),
                graph_panel_fps=float(graph_panel_fps),
                diagnostic_options=diagnostic_options,
            )
        )

    baseline = next((item for item in results if item["label"] == "baseline_full"), None)
    baseline_total = float(baseline["summary"]["avg_total_frame_ms"]) if baseline is not None else 0.0
    ranked = sorted(results, key=lambda item: float(item["summary"]["avg_total_frame_ms"]))
    print("[DIAG] Ranked average total frame time:")
    for item in ranked:
        avg_total = float(item["summary"]["avg_total_frame_ms"])
        delta = avg_total - baseline_total
        print(
            f"  {item['label']}: {avg_total:.2f}ms "
            f"(delta {delta:+.2f}ms, fps {float(item['summary']['estimated_fps']):.1f})"
        )
    return results


def _cli_main() -> int:
    args = parse_runtime_args()
    diagnostic_options = RuntimeDiagnosticOptions(
        disable_graph_update=bool(args.diag_disable_graph_update),
        disable_graph_draw=bool(args.diag_disable_graph_draw),
        disable_side_panels=bool(args.diag_disable_side_panels),
        disable_controller=bool(args.diag_disable_controller),
        disable_validation_window=bool(args.diag_disable_validation_window),
        disable_main_env_render=bool(args.diag_disable_main_render),
        disable_cv_overlay=bool(args.diag_disable_cv_overlay),
    )
    common_kwargs = dict(
        show_ctrl_validation_window=bool(args.ctrl_validation_window),
        show_muscle_graph_window=bool(args.muscle_graph_window),
        dynamic_edge_mode=str(args.dynamic_edge_mode),
        node_activation_threshold=float(args.node_activation_threshold),
        edge_score_threshold=float(args.edge_score_threshold),
        dynamic_edge_window=int(args.dynamic_edge_window),
        visible_edge_top_k=int(args.visible_edge_top_k),
        edge_relation_filter=str(args.edge_relation_filter),
        highlight_top_nodes=int(args.highlight_top_nodes),
        visualization_mode=str(args.visualization_mode),
        dominant_selection_mode=str(args.dominant_selection_mode),
        dominant_mode=str(args.visualization_mode).strip().lower() != "full",
        dominant_top_k=int(args.dominant_top_k),
        dominant_edge_top_k=int(args.dominant_edge_top_k),
        include_thumb_in_dominant_subset=not bool(args.no_thumb_dominant_subset),
        hide_inactive_dominant_nodes=not bool(args.show_inactive_dominant_nodes),
        ghost_non_dominant_nodes=not bool(args.no_ghost_non_dominant_nodes),
        graph_preset=str(args.graph_preset),
        debug_dynamic_edges=bool(args.debug_dynamic_edges),
        show_labels=not bool(args.hide_labels),
        ctrl_mode=str(args.ctrl_mode),
        qpos_deadband_norm=float(args.qpos_deadband_norm),
        ctrl_smoothing_alpha=float(args.ctrl_smoothing_alpha),
        ctrl_rolling_window=int(args.ctrl_rolling_window),
        hold_if_small_change=not bool(args.no_hold_small_change),
        freeze_control=bool(args.freeze_control),
        tracking_tuning_preset=str(args.tracking_tuning_preset),
        hand_mapping_debug=bool(args.hand_mapping_debug),
        hand_mapping_debug_interval=float(args.hand_mapping_debug_interval),
        hand_mapping_debug_seconds=float(args.hand_mapping_debug_seconds),
        graph_render_fps=float(args.graph_render_fps),
        graph_panel_fps=float(args.graph_panel_fps),
        exercise_name=None if args.exercise_name is None else str(args.exercise_name),
        trial_id=None if args.trial_id is None else str(args.trial_id),
    )
    if bool(args.diagnostic_mode):
        run_bottleneck_diagnostic(
            duration_seconds=float(args.diagnostic_seconds),
            base_diagnostic_options=diagnostic_options,
            **common_kwargs,
        )
        return 0
    if args.replay_record:
        return replay_recorded_muscle_graph(
            record_path=str(args.replay_record),
            replay_speed=float(args.replay_speed),
            replay_loop=bool(args.replay_loop),
            diagnostic_options=diagnostic_options,
            **common_kwargs,
        )
    main(
        profile_interval=float(args.profile_interval),
        runtime_profiler_enabled=not bool(args.no_runtime_profiler),
        diagnostic_options=diagnostic_options,
        **common_kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
