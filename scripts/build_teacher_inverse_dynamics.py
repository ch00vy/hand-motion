from __future__ import annotations

"""Compatibility backend for offline inverse-dynamics biomech references.

Canonical users should call scripts/offline_biomech/build_biomech_ctrl_reference.py.
This module keeps legacy teacher names because other compatibility scripts still
import its solver helpers.

The output is a simulation-consistent biomechanical ctrl_ref for later offline
analysis. It is not biological ground truth and must not be treated as a
clinical label source.
"""

import argparse
import copy
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import (  # noqa: E402
    DEFAULT_MYOSUITE_ENV_ID,
    _dense_moment,
    _gain_bias,
    _get_raw_model_data,
    _solve_box_qp,
    _torque_weights,
    create_probe_env,
    estimate_ctrl_from_qpos,
)


DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "teacher_inverse_dynamics_summary.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "teacher_inverse_dynamics"


def indexed_columns(columns: list[str], prefix: str) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for name in columns:
        label = str(name)
        if not label.startswith(prefix):
            continue
        try:
            idx = int(label[len(prefix) :])
        except ValueError:
            continue
        indexed.append((idx, label))
    indexed.sort(key=lambda item: item[0])
    return [label for _, label in indexed]


def unique_nonempty(series: pd.Series) -> str | None:
    values = sorted({str(item).strip() for item in series.fillna("").tolist() if str(item).strip()})
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"Inconsistent metadata values: {values}")
    return values[0]


def discover_recording_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("ctrl_est_*.csv") if path.is_file())


def load_recorded_trial(path: Path) -> dict[str, Any] | None:
    df = pd.read_csv(path)
    if df.empty:
        return None
    qpos_cols = indexed_columns(df.columns.tolist(), "qpos_")
    if not qpos_cols:
        return None
    exercise_name = unique_nonempty(df.get("exercise_name", pd.Series(dtype=str)))
    trial_id = unique_nonempty(df.get("trial_id", pd.Series(dtype=str)))
    session_id = unique_nonempty(df.get("session_id", pd.Series(dtype=str)))
    if exercise_name is None or trial_id is None:
        return None
    timestamps = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float) if "t" in df.columns else None
    frame_indices = (
        pd.to_numeric(df["frame_index"], errors="coerce")
        .fillna(pd.to_numeric(df.get("frame"), errors="coerce"))
        .fillna(0)
        .to_numpy(dtype=int)
        if "frame_index" in df.columns or "frame" in df.columns
        else np.arange(len(df), dtype=int)
    )
    qpos_raw = df.loc[:, qpos_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return {
        "source_csv": str(path),
        "exercise_name": exercise_name,
        "trial_id": trial_id,
        "session_id": session_id or "",
        "timestamps": timestamps,
        "frame_indices": frame_indices,
        "qpos_raw": qpos_raw,
        "qpos_columns": qpos_cols,
        "num_frames": int(len(df)),
    }


def smooth_signal(sequence: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(sequence, dtype=float)
    if values.ndim != 2 or values.shape[0] <= 2 or int(window) <= 1:
        return values.copy()
    radius = max(1, int(window) // 2)
    kernel = np.ones((2 * radius + 1,), dtype=float)
    kernel /= float(np.sum(kernel))
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.empty_like(values)
    for j in range(values.shape[1]):
        smoothed[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    return smoothed


def smooth_signal_savgol(sequence: np.ndarray, window: int, polyorder: int = 3) -> np.ndarray:
    values = np.asarray(sequence, dtype=float)
    if values.ndim != 2 or values.shape[0] <= 3 or int(window) <= 2:
        return values.copy()
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return smooth_signal(values, window)

    n = int(values.shape[0])
    win = max(3, int(window))
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 3:
        return values.copy()
    order = max(1, min(int(polyorder), win - 1))
    return np.asarray(savgol_filter(values, window_length=win, polyorder=order, axis=0, mode="interp"), dtype=float)


def smooth_signal_by_method(sequence: np.ndarray, window: int, *, method: str = "moving_average", polyorder: int = 3) -> np.ndarray:
    name = str(method or "moving_average").strip().lower()
    if name in {"savgol", "savitzky_golay", "savitzky-golay"}:
        return smooth_signal_savgol(sequence, window, polyorder=polyorder)
    return smooth_signal(sequence, window)


def clip_signal(values: np.ndarray, clip_value: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(float(clip_value)) or float(clip_value) <= 0.0:
        return np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(np.where(np.isfinite(arr), arr, 0.0), -float(clip_value), float(clip_value))


def normalize_timestamps(
    timestamps: np.ndarray | None,
    num_frames: int,
    fallback_dt: float,
    *,
    ignore_timestamps: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if bool(ignore_timestamps) or timestamps is None or len(timestamps) != num_frames:
        times = np.arange(num_frames, dtype=float) * float(fallback_dt)
    else:
        times = np.asarray(timestamps, dtype=float).copy()
        finite_mask = np.isfinite(times)
        if not np.any(finite_mask):
            times = np.arange(num_frames, dtype=float) * float(fallback_dt)
        else:
            first_valid = float(times[np.where(finite_mask)[0][0]])
            times = np.where(finite_mask, times, first_valid)
            times -= float(times[0])
            for i in range(1, num_frames):
                if not np.isfinite(times[i]) or times[i] <= times[i - 1]:
                    times[i] = times[i - 1] + float(fallback_dt)
    dt = np.diff(times, prepend=times[0])
    if num_frames > 1:
        median_dt = float(np.median(np.diff(times)))
    else:
        median_dt = float(fallback_dt)
    if not np.isfinite(median_dt) or median_dt <= 1e-6:
        median_dt = float(fallback_dt)
    dt[0] = median_dt
    dt = np.where(np.isfinite(dt) & (dt > 1e-6), dt, median_dt)
    return times, dt


def estimate_kinematics(
    qpos_raw: np.ndarray,
    timestamps: np.ndarray | None,
    *,
    fallback_dt: float,
    qpos_smoothing_window: int,
    qvel_smoothing_window: int,
    qacc_smoothing_window: int,
    qvel_clip: float,
    qacc_clip: float,
    use_raw_qpos_target: bool,
    kinematic_filter: str = "moving_average",
    savgol_polyorder: int = 3,
    ignore_timestamps: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one consistent kinematic target trajectory for the offline teacher.

    The returned `(qpos_target, qvel_target, qacc_target, dt)` must be used
    consistently throughout the inverse-dynamics, actuator solve, and forward
    validation path. Raw qpos remains available for logging only unless
    `use_raw_qpos_target` is explicitly enabled.
    """

    qpos_raw = np.asarray(qpos_raw, dtype=float)
    times, dt = normalize_timestamps(timestamps, qpos_raw.shape[0], fallback_dt, ignore_timestamps=bool(ignore_timestamps))
    if bool(use_raw_qpos_target):
        qpos_target = qpos_raw.copy()
    else:
        qpos_target = smooth_signal_by_method(
            qpos_raw,
            qpos_smoothing_window,
            method=kinematic_filter,
            polyorder=int(savgol_polyorder),
        )
    if qpos_raw.shape[0] <= 2:
        qvel_target = np.zeros_like(qpos_target)
        qacc_target = np.zeros_like(qpos_target)
    else:
        qvel_target = np.gradient(qpos_target, times, axis=0, edge_order=1)
        qvel_target = smooth_signal_by_method(
            qvel_target,
            qvel_smoothing_window,
            method=kinematic_filter,
            polyorder=int(savgol_polyorder),
        )
        qacc_target = np.gradient(qvel_target, times, axis=0, edge_order=1)
        qacc_target = smooth_signal_by_method(
            qacc_target,
            qacc_smoothing_window,
            method=kinematic_filter,
            polyorder=int(savgol_polyorder),
        )
    qvel_target = clip_signal(qvel_target, qvel_clip)
    qacc_target = clip_signal(qacc_target, qacc_clip)
    qpos_target = np.where(np.isfinite(qpos_target), qpos_target, 0.0)
    qvel_target = np.where(np.isfinite(qvel_target), qvel_target, 0.0)
    qacc_target = np.where(np.isfinite(qacc_target), qacc_target, 0.0)
    return qpos_target, qvel_target, qacc_target, dt


def set_full_state(model: Any, data: Any, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray, *, ctrl: np.ndarray | None = None, zero_act: bool = True) -> None:
    q = np.zeros(int(model.nq), dtype=float)
    v = np.zeros(int(model.nv), dtype=float)
    a = np.zeros(int(model.nv), dtype=float)
    q_in = np.asarray(qpos, dtype=float).reshape(-1)
    v_in = np.asarray(qvel, dtype=float).reshape(-1)
    a_in = np.asarray(qacc, dtype=float).reshape(-1)
    q[: min(len(q), len(q_in))] = q_in[: min(len(q), len(q_in))]
    v[: min(len(v), len(v_in))] = v_in[: min(len(v), len(v_in))]
    a[: min(len(a), len(a_in))] = a_in[: min(len(a), len(a_in))]
    data.qpos[:] = q
    data.qvel[:] = v
    data.qacc[:] = a
    data.ctrl[:] = 0.0
    if ctrl is not None:
        ctrl_in = np.asarray(ctrl, dtype=float).reshape(-1)
        data.ctrl[: min(int(model.nu), len(ctrl_in))] = ctrl_in[: min(int(model.nu), len(ctrl_in))]
    if int(getattr(model, "na", 0)) > 0 and zero_act:
        data.act[:] = 0.0
    m_raw, d_raw = _get_raw_model_data(model, data)
    mujoco.mj_forward(m_raw, d_raw)


def compute_inverse_dynamics_tau(
    model: Any,
    data: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    *,
    disable_constraints: bool,
    tau_clip: float,
) -> np.ndarray:
    set_full_state(model, data, qpos, qvel, qacc, zero_act=True)
    m_raw, d_raw = _get_raw_model_data(model, data)
    old_flags = int(m_raw.opt.disableflags)
    try:
        if disable_constraints:
            m_raw.opt.disableflags = old_flags | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        mujoco.mj_inverse(m_raw, d_raw)
        tau = np.asarray(d_raw.qfrc_inverse[: m_raw.nv], dtype=float).copy()
        tau = clip_signal(tau, tau_clip)
        return tau
    finally:
        m_raw.opt.disableflags = old_flags


def compute_vector_norm_stats(sequence: np.ndarray) -> dict[str, float]:
    arr = np.asarray(sequence, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {"mean_norm": 0.0, "max_norm": 0.0}
    norms = np.linalg.norm(arr, axis=1)
    return {
        "mean_norm": float(np.mean(norms)) if norms.size else 0.0,
        "max_norm": float(np.max(norms)) if norms.size else 0.0,
    }


def maybe_build_static_prior(
    teacher_mode: str,
    teacher_prior_mode: str,
    static_prior_weight: float,
    model: Any,
    data: Any,
    qpos_target: np.ndarray,
    prev_ctrl: np.ndarray | None,
    exercise_name: str | None,
    actuator_names: list[str],
) -> np.ndarray | None:
    use_static_prior = False
    if teacher_mode in {"dynamic_with_static_prior", "dynamic_with_fallback"}:
        use_static_prior = True
    elif teacher_mode == "dynamic_only" and static_prior_weight > 0.0 and str(teacher_prior_mode).strip().lower() != "zero":
        use_static_prior = True
    if not use_static_prior or str(teacher_prior_mode).strip().lower() == "zero":
        return None
    try:
        prior_ctrl, _ = estimate_ctrl_from_qpos(
            model,
            data,
            qpos_target,
            prev_ctrl=prev_ctrl,
            exercise_name=exercise_name,
            actuator_names=actuator_names,
        )
        return np.asarray(prior_ctrl, dtype=float).reshape(-1)
    except Exception:
        return None


def solve_teacher_ctrl_linearized(
    model: Any,
    data: Any,
    tau_target: np.ndarray,
    qpos_target: np.ndarray,
    qvel_target: np.ndarray,
    *,
    prev_ctrl: np.ndarray | None,
    static_prior_ctrl: np.ndarray | None,
    ctrl_lb: float,
    ctrl_ub: float,
    qp_reg: float,
    prev_ctrl_weight: float,
    static_prior_weight: float,
    prior_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve bounded actuator controls with separated penalties.

    Objective:
    - track inverse-dynamics torque target,
    - stay close to previous control,
    - stay close to a static qpos-derived prior,
    - remain regularized.
    """

    set_full_state(model, data, qpos_target, qvel_target, np.zeros(int(model.nv), dtype=float), zero_act=True)
    m_raw, d_raw = _get_raw_model_data(model, data)
    mujoco.mj_step1(m_raw, d_raw)
    moment = _dense_moment(m_raw, d_raw)
    gain, bias = _gain_bias(m_raw, d_raw)
    A = moment * gain[np.newaxis, :]
    tau_target = np.asarray(tau_target, dtype=float).reshape(-1)
    b = tau_target - moment @ bias
    W = _torque_weights(m_raw)[: A.shape[0]]
    A_w = A * W[:, np.newaxis]
    b_w = b * W

    nu = int(m_raw.nu)
    prev = np.full(nu, float(ctrl_lb), dtype=float) if prev_ctrl is None else np.clip(np.asarray(prev_ctrl, dtype=float).reshape(-1), ctrl_lb, ctrl_ub)
    if prev.shape[0] != nu:
        prev_fixed = np.full(nu, float(ctrl_lb), dtype=float)
        prev_fixed[: min(prev.shape[0], nu)] = prev[: min(prev.shape[0], nu)]
        prev = prev_fixed
    static_prior = np.full(nu, float(prior_value), dtype=float)
    if static_prior_ctrl is not None:
        static_prior_in = np.clip(np.asarray(static_prior_ctrl, dtype=float).reshape(-1), ctrl_lb, ctrl_ub)
        static_prior[: min(nu, static_prior_in.shape[0])] = static_prior_in[: min(nu, static_prior_in.shape[0])]

    P = 2.0 * (A_w.T @ A_w) + (2.0 * float(qp_reg) + 2.0 * float(prev_ctrl_weight) + 2.0 * float(static_prior_weight)) * np.eye(nu)
    q = -2.0 * (A_w.T @ b_w) - 2.0 * float(prev_ctrl_weight) * prev - 2.0 * float(static_prior_weight) * static_prior
    lb = np.full(nu, float(ctrl_lb), dtype=float)
    ub = np.full(nu, float(ctrl_ub), dtype=float)
    x0 = np.clip(0.5 * prev + 0.5 * static_prior, ctrl_lb, ctrl_ub)
    ctrl = np.asarray(_solve_box_qp(P, q, lb, ub, x0), dtype=float).reshape(-1)
    ctrl = np.clip(np.where(np.isfinite(ctrl), ctrl, 0.0), ctrl_lb, ctrl_ub)
    tau_hat = A @ ctrl + moment @ bias
    residual = tau_hat - tau_target
    ctrl_norm = float(np.linalg.norm(ctrl))
    prev_distance = float(np.linalg.norm(ctrl - prev))
    static_prior_distance = float(np.linalg.norm(ctrl - static_prior))
    lower_sat = float(np.mean(ctrl <= (float(ctrl_lb) + 1e-6))) if ctrl.size else 0.0
    upper_sat = float(np.mean(ctrl >= (float(ctrl_ub) - 1e-6))) if ctrl.size else 0.0
    col_norms = np.linalg.norm(A, axis=0)
    return ctrl, {
        "solver_mode": "dynamic_linearized_qp",
        "tau_reconstructed": tau_hat,
        "residual_norm": float(np.linalg.norm(residual)),
        "relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(tau_target), 1e-8)),
        "matrix_rank": int(np.linalg.matrix_rank(A)) if A.size else 0,
        "matrix_fro_norm": float(np.linalg.norm(A)),
        "matrix_col_norm_mean": float(np.mean(col_norms)) if col_norms.size else 0.0,
        "ctrl_norm": ctrl_norm,
        "prev_distance": prev_distance,
        "static_prior_distance": static_prior_distance,
        "lower_saturation_fraction": lower_sat,
        "upper_saturation_fraction": upper_sat,
        "tau_target_norm": float(np.linalg.norm(tau_target)),
        "tau_hat_norm": float(np.linalg.norm(tau_hat)),
    }


def solve_teacher_ctrl_frame(
    model: Any,
    data: Any,
    tau_target: np.ndarray,
    qpos_target: np.ndarray,
    qvel_target: np.ndarray,
    *,
    prev_ctrl: np.ndarray | None,
    static_prior_ctrl: np.ndarray | None,
    ctrl_lb: float,
    ctrl_ub: float,
    qp_reg: float,
    prev_ctrl_weight: float,
    static_prior_weight: float,
    prior_value: float,
    teacher_mode: str,
    exercise_name: str | None,
    actuator_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    allow_fallback = str(teacher_mode).strip().lower() == "dynamic_with_fallback"
    use_static_prior = str(teacher_mode).strip().lower() in {"dynamic_with_static_prior", "dynamic_with_fallback"} or (
        str(teacher_mode).strip().lower() == "dynamic_only" and static_prior_ctrl is not None and static_prior_weight > 0.0
    )
    static_prior = static_prior_ctrl if use_static_prior else None
    try:
        ctrl, info = solve_teacher_ctrl_linearized(
            model,
            data,
            tau_target,
            qpos_target,
            qvel_target,
            prev_ctrl=prev_ctrl,
            static_prior_ctrl=static_prior,
            ctrl_lb=ctrl_lb,
            ctrl_ub=ctrl_ub,
            qp_reg=qp_reg,
            prev_ctrl_weight=prev_ctrl_weight,
            static_prior_weight=static_prior_weight if use_static_prior else 0.0,
            prior_value=prior_value,
        )
        info["solver_mode"] = f"{teacher_mode}:dynamic_linearized_qp"
        return ctrl, info
    except Exception as exc:
        if not allow_fallback:
            raise
        ctrl, fallback_info = estimate_ctrl_from_qpos(
            model,
            data,
            qpos_target,
            prev_ctrl=prev_ctrl,
            exercise_name=exercise_name,
            actuator_names=actuator_names,
        )
        ctrl_arr = np.clip(np.asarray(ctrl, dtype=float).reshape(-1), ctrl_lb, ctrl_ub)
        info = {
            "solver_mode": f"{teacher_mode}:static_fallback_estimate_ctrl_from_qpos",
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "tau_reconstructed": np.asarray(fallback_info.get("tau_reconstructed"), dtype=float),
            "residual_norm": float(fallback_info.get("residual_norm", 0.0)),
            "relative_residual": float(fallback_info.get("relative_residual", 0.0)),
            "matrix_rank": int(fallback_info.get("effective_rank", 0)),
            "matrix_fro_norm": float(fallback_info.get("column_norm_summary", {}).get("fro_norm", 0.0)),
            "matrix_col_norm_mean": float(fallback_info.get("column_norm_summary", {}).get("mean", 0.0)),
            "ctrl_norm": float(np.linalg.norm(ctrl_arr)),
            "prev_distance": float(np.linalg.norm(ctrl_arr - (np.asarray(prev_ctrl, dtype=float).reshape(-1) if prev_ctrl is not None else np.zeros_like(ctrl_arr)))),
            "static_prior_distance": float(np.linalg.norm(ctrl_arr - (np.asarray(static_prior, dtype=float).reshape(-1) if static_prior is not None else np.full_like(ctrl_arr, prior_value)))),
            "lower_saturation_fraction": float(np.mean(ctrl_arr <= (float(ctrl_lb) + 1e-6))),
            "upper_saturation_fraction": float(np.mean(ctrl_arr >= (float(ctrl_ub) - 1e-6))),
            "tau_target_norm": float(np.linalg.norm(tau_target)),
            "tau_hat_norm": float(np.linalg.norm(np.asarray(fallback_info.get("tau_reconstructed"), dtype=float))) if fallback_info.get("tau_reconstructed") is not None else 0.0,
        }
        return ctrl_arr, info


def build_substeps_from_dt(dt: np.ndarray, sim_dt: float, max_substeps: int) -> np.ndarray:
    steps = np.maximum(1, np.rint(np.asarray(dt, dtype=float) / max(float(sim_dt), 1e-8)).astype(int))
    if int(max_substeps) > 0:
        steps = np.clip(steps, 1, int(max_substeps))
    return steps


def compute_window_rmse(frame_errors: np.ndarray, window_size: int) -> list[float]:
    if int(window_size) <= 1 or frame_errors.size == 0:
        return []
    values: list[float] = []
    for start in range(0, len(frame_errors), int(window_size)):
        chunk = frame_errors[start : start + int(window_size)]
        if chunk.size == 0:
            continue
        values.append(float(math.sqrt(np.mean(np.square(chunk)))))
    return values


def forward_validate_sequence(
    model: Any,
    qpos_target_seq: np.ndarray,
    qvel_target_seq: np.ndarray,
    ctrl_seq: np.ndarray,
    dt_seq: np.ndarray,
    *,
    max_substeps: int,
    validation_window_size: int,
) -> dict[str, Any]:
    m_raw, _ = _get_raw_model_data(model, model.data if hasattr(model, "data") else None) if hasattr(model, "data") else (model, None)
    d_val = mujoco.MjData(m_raw)
    qpos_target_seq = np.asarray(qpos_target_seq, dtype=float)
    qvel_target_seq = np.asarray(qvel_target_seq, dtype=float)
    ctrl_seq = np.asarray(ctrl_seq, dtype=float)
    dt_seq = np.asarray(dt_seq, dtype=float)
    if qpos_target_seq.shape[0] <= 1:
        return {
            "mean_joint_error": 0.0,
            "rmse_forward_error": 0.0,
            "mean_pose_error_norm": 0.0,
            "median_frame_error": 0.0,
            "p95_frame_error": 0.0,
            "max_frame_error": 0.0,
            "per_joint_rmse": [0.0] * int(qpos_target_seq.shape[1]),
            "frame_errors": [0.0] * int(qpos_target_seq.shape[0]),
            "qpos_forward_final": np.asarray(qpos_target_seq[-1] if qpos_target_seq.shape[0] else [], dtype=float).tolist(),
            "qvel_forward_final": np.asarray(qvel_target_seq[-1] if qvel_target_seq.shape[0] else [], dtype=float).tolist(),
            "num_windows": 0,
            "worst_window_rmse": 0.0,
            "median_window_rmse": 0.0,
            "per_window_rmse": [],
        }

    set_full_state(m_raw, d_val, qpos_target_seq[0], qvel_target_seq[0], np.zeros(int(m_raw.nv), dtype=float), ctrl=ctrl_seq[0], zero_act=True)
    substeps = build_substeps_from_dt(dt_seq, float(m_raw.opt.timestep), max_substeps)
    frame_errors: list[float] = [0.0]
    per_joint_sqerr = np.zeros(int(m_raw.nq), dtype=float)
    per_joint_count = np.zeros(int(m_raw.nq), dtype=float)
    forward_qpos_seq = [np.asarray(d_val.qpos, dtype=float).copy()]
    forward_qvel_seq = [np.asarray(d_val.qvel, dtype=float).copy()]

    for i in range(qpos_target_seq.shape[0] - 1):
        ctrl = np.asarray(ctrl_seq[i], dtype=float).reshape(-1)
        for _ in range(int(substeps[min(i + 1, len(substeps) - 1)])):
            d_val.ctrl[:] = 0.0
            d_val.ctrl[: min(int(m_raw.nu), len(ctrl))] = ctrl[: min(int(m_raw.nu), len(ctrl))]
            mujoco.mj_step(m_raw, d_val)
        target_next = np.asarray(qpos_target_seq[i + 1], dtype=float).reshape(-1)
        nq = min(int(m_raw.nq), len(target_next))
        joint_err = np.asarray(d_val.qpos[:nq], dtype=float) - target_next[:nq]
        per_joint_sqerr[:nq] += joint_err ** 2
        per_joint_count[:nq] += 1.0
        frame_errors.append(float(np.linalg.norm(joint_err)))
        forward_qpos_seq.append(np.asarray(d_val.qpos, dtype=float).copy())
        forward_qvel_seq.append(np.asarray(d_val.qvel, dtype=float).copy())

    frame_error_arr = np.asarray(frame_errors[1:], dtype=float) if len(frame_errors) > 1 else np.zeros(0, dtype=float)
    mean_joint_error = float(np.mean(np.sqrt(np.divide(per_joint_sqerr, np.maximum(per_joint_count, 1.0), out=np.zeros_like(per_joint_sqerr), where=per_joint_count > 0))))
    rmse_forward_error = float(math.sqrt(np.mean(np.square(frame_error_arr)))) if frame_error_arr.size else 0.0
    per_joint_rmse = np.sqrt(np.divide(per_joint_sqerr, np.maximum(per_joint_count, 1.0), out=np.zeros_like(per_joint_sqerr), where=per_joint_count > 0))
    window_rmses = compute_window_rmse(frame_error_arr, validation_window_size)
    return {
        "mean_joint_error": mean_joint_error,
        "rmse_forward_error": rmse_forward_error,
        "mean_pose_error_norm": float(np.mean(frame_error_arr)) if frame_error_arr.size else 0.0,
        "median_frame_error": float(np.median(frame_error_arr)) if frame_error_arr.size else 0.0,
        "p95_frame_error": float(np.percentile(frame_error_arr, 95.0)) if frame_error_arr.size else 0.0,
        "max_frame_error": float(np.max(frame_error_arr)) if frame_error_arr.size else 0.0,
        "per_joint_rmse": per_joint_rmse.tolist(),
        "frame_errors": frame_errors,
        "qpos_forward_final": np.asarray(forward_qpos_seq[-1], dtype=float).tolist(),
        "qvel_forward_final": np.asarray(forward_qvel_seq[-1], dtype=float).tolist(),
        "forward_qpos_seq": [np.asarray(item, dtype=float).tolist() for item in forward_qpos_seq],
        "forward_qvel_seq": [np.asarray(item, dtype=float).tolist() for item in forward_qvel_seq],
        "num_windows": int(len(window_rmses)),
        "worst_window_rmse": float(max(window_rmses)) if window_rmses else 0.0,
        "median_window_rmse": float(np.median(window_rmses)) if window_rmses else 0.0,
        "per_window_rmse": window_rmses,
    }


def compute_quality_score(quality: dict[str, Any]) -> float:
    rel_res = abs(float(quality.get("relative_residual_mean", 0.0)))
    score = 1.0 / (
        1.0
        + float(quality.get("mean_joint_error", 0.0))
        + 0.5 * float(quality.get("rmse_forward_error", 0.0))
        + 0.20 * float(quality.get("ctrl_delta_mean", 0.0))
        + 0.25 * float(quality.get("upper_saturation_fraction", 0.0))
        + 0.15 * float(quality.get("lower_saturation_fraction", 0.0))
        + 0.10 * math.log1p(rel_res)
    )
    return float(score)


def summarize_trial_quality(
    trial: dict[str, Any],
    ctrl_seq: np.ndarray,
    solver_infos: list[dict[str, Any]],
    validation: dict[str, Any],
    *,
    qvel_target: np.ndarray,
    qacc_target: np.ndarray,
    tau_seq: np.ndarray,
    max_mean_joint_error: float,
    max_rmse_forward_error: float,
    max_upper_saturation_fraction: float,
    max_ctrl_delta_mean: float,
) -> tuple[bool, dict[str, Any]]:
    ctrl = np.asarray(ctrl_seq, dtype=float)
    ctrl_delta = np.diff(ctrl, axis=0) if ctrl.shape[0] > 1 else np.zeros((0, ctrl.shape[1] if ctrl.ndim == 2 else 0), dtype=float)
    ctrl_delta_mean = float(np.mean(np.linalg.norm(ctrl_delta, axis=1))) if ctrl_delta.size else 0.0
    upper_saturation_fraction = float(np.mean(ctrl >= 0.98)) if ctrl.size else 0.0
    lower_saturation_fraction = float(np.mean(ctrl <= 0.02)) if ctrl.size else 0.0
    fallback_fraction = float(np.mean([1.0 if "fallback" in str(item.get("solver_mode", "")).lower() else 0.0 for item in solver_infos])) if solver_infos else 0.0
    rel_residual_mean = float(np.mean([float(item.get("relative_residual", 0.0)) for item in solver_infos])) if solver_infos else 0.0
    ctrl_norm_mean = float(np.mean([float(item.get("ctrl_norm", np.linalg.norm(ctrl[i]))) for i, item in enumerate(solver_infos)])) if solver_infos else (float(np.mean(np.linalg.norm(ctrl, axis=1))) if ctrl.ndim == 2 and ctrl.shape[0] else 0.0)
    qvel_stats = compute_vector_norm_stats(qvel_target)
    qacc_stats = compute_vector_norm_stats(qacc_target)
    tau_stats = compute_vector_norm_stats(tau_seq)
    quality = {
        "source_csv": str(trial["source_csv"]),
        "exercise_name": str(trial["exercise_name"]),
        "trial_id": str(trial["trial_id"]),
        "num_frames": int(ctrl.shape[0]),
        "mean_joint_error": float(validation["mean_joint_error"]),
        "rmse_forward_error": float(validation["rmse_forward_error"]),
        "mean_pose_error_norm": float(validation["mean_pose_error_norm"]),
        "median_frame_error": float(validation.get("median_frame_error", 0.0)),
        "p95_frame_error": float(validation.get("p95_frame_error", 0.0)),
        "max_frame_error": float(validation.get("max_frame_error", 0.0)),
        "worst_window_rmse": float(validation.get("worst_window_rmse", 0.0)),
        "median_window_rmse": float(validation.get("median_window_rmse", 0.0)),
        "per_joint_rmse": list(validation.get("per_joint_rmse", [])),
        "upper_saturation_fraction": upper_saturation_fraction,
        "lower_saturation_fraction": lower_saturation_fraction,
        "ctrl_delta_mean": ctrl_delta_mean,
        "ctrl_norm_mean": ctrl_norm_mean,
        "relative_residual_mean": rel_residual_mean,
        "fallback_fraction": fallback_fraction,
        "mean_qvel_norm": float(qvel_stats["mean_norm"]),
        "max_qvel_norm": float(qvel_stats["max_norm"]),
        "mean_qacc_norm": float(qacc_stats["mean_norm"]),
        "max_qacc_norm": float(qacc_stats["max_norm"]),
        "mean_tau_norm": float(tau_stats["mean_norm"]),
        "max_tau_norm": float(tau_stats["max_norm"]),
        "num_windows": int(validation.get("num_windows", 0)),
    }
    keep = True
    reasons: list[str] = []
    if quality["mean_joint_error"] > float(max_mean_joint_error):
        keep = False
        reasons.append("forward_mean_joint_error")
    if quality["rmse_forward_error"] > float(max_rmse_forward_error):
        keep = False
        reasons.append("forward_rmse")
    if quality["upper_saturation_fraction"] > float(max_upper_saturation_fraction):
        keep = False
        reasons.append("upper_saturation")
    if quality["ctrl_delta_mean"] > float(max_ctrl_delta_mean):
        keep = False
        reasons.append("ctrl_jitter")
    quality["quality_score"] = compute_quality_score(quality)
    quality["keep_for_training"] = bool(keep)
    quality["reject_reasons"] = reasons
    return keep, quality


def build_frame_table(
    trial: dict[str, Any],
    times: np.ndarray,
    dt: np.ndarray,
    qpos_raw: np.ndarray,
    qpos_target: np.ndarray,
    qvel_target: np.ndarray,
    qacc_target: np.ndarray,
    tau_seq: np.ndarray,
    ctrl_seq: np.ndarray,
    solver_infos: list[dict[str, Any]],
    validation: dict[str, Any],
    *,
    suspicious_frame_error: float,
    suspicious_qacc_norm: float,
    suspicious_tau_norm: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame_errors = list(validation.get("frame_errors", []))
    for i in range(qpos_target.shape[0]):
        ctrl_frame = np.asarray(ctrl_seq[i], dtype=float)
        tau_frame = np.asarray(tau_seq[i], dtype=float)
        qvel_frame = np.asarray(qvel_target[i], dtype=float)
        qacc_frame = np.asarray(qacc_target[i], dtype=float)
        tau_norm = float(np.linalg.norm(tau_frame))
        ctrl_norm = float(np.linalg.norm(ctrl_frame))
        qvel_norm = float(np.linalg.norm(qvel_frame))
        qacc_norm = float(np.linalg.norm(qacc_frame))
        lower_sat_frame = float(np.mean(ctrl_frame <= 0.02)) if ctrl_frame.size else 0.0
        upper_sat_frame = float(np.mean(ctrl_frame >= 0.98)) if ctrl_frame.size else 0.0
        frame_error = float(frame_errors[i]) if i < len(frame_errors) else np.nan
        suspicious = False
        if np.isfinite(frame_error) and float(suspicious_frame_error) > 0.0 and frame_error > float(suspicious_frame_error):
            suspicious = True
        if float(suspicious_qacc_norm) > 0.0 and qacc_norm > float(suspicious_qacc_norm):
            suspicious = True
        if float(suspicious_tau_norm) > 0.0 and tau_norm > float(suspicious_tau_norm):
            suspicious = True
        if upper_sat_frame > 0.8 or lower_sat_frame > 0.95:
            suspicious = True
        row: dict[str, Any] = {
            "source_csv": str(trial["source_csv"]),
            "session_id": str(trial["session_id"]),
            "exercise_name": str(trial["exercise_name"]),
            "trial_id": str(trial["trial_id"]),
            "frame_index": int(trial["frame_indices"][i]) if i < len(trial["frame_indices"]) else int(i),
            "time_from_start": float(times[i]),
            "frame_dt": float(dt[i]),
            "solver_mode": str(solver_infos[i].get("solver_mode", "")),
            "relative_residual": float(solver_infos[i].get("relative_residual", 0.0)),
            "residual_norm": float(solver_infos[i].get("residual_norm", 0.0)),
            "tau_norm": tau_norm,
            "ctrl_norm": ctrl_norm,
            "qvel_norm": qvel_norm,
            "qacc_norm": qacc_norm,
            "lower_saturation_fraction_frame": lower_sat_frame,
            "upper_saturation_fraction_frame": upper_sat_frame,
            "forward_error_to_next": frame_error,
            "is_suspicious_frame": bool(suspicious),
        }
        for j in range(qpos_raw.shape[1]):
            row[f"qpos_raw_{j}"] = float(qpos_raw[i, j])
            row[f"qpos_target_{j}"] = float(qpos_target[i, j])
            row[f"qvel_target_{j}"] = float(qvel_target[i, j])
            row[f"qacc_target_{j}"] = float(qacc_target[i, j])
            row[f"qfrc_inverse_{j}"] = float(tau_seq[i, j])
        for j in range(ctrl_seq.shape[1]):
            row[f"ctrl_teacher_{j}"] = float(ctrl_seq[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def save_trial_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def dataframe_to_markdown_simple(df: pd.DataFrame) -> str:
    """Render a small dataframe as markdown without requiring tabulate."""

    if df.empty:
        return "_empty_"
    headers = [str(col) for col in df.columns.tolist()]
    rows = [[str(item) for item in row] for row in df.astype(object).where(pd.notnull(df), "").values.tolist()]
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def build_per_gesture_summary(trial_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_gesture: dict[str, list[dict[str, Any]]] = {}
    for item in trial_summaries:
        by_gesture.setdefault(str(item["exercise_name"]), []).append(item)
    out: dict[str, Any] = {}
    for gesture, rows in sorted(by_gesture.items()):
        passed = [row for row in rows if bool(row.get("keep_for_training", False))]
        out[gesture] = {
            "num_trials": int(len(rows)),
            "num_kept": int(len(passed)),
            "num_rejected": int(len(rows) - len(passed)),
            "pass_rate": float(len(passed) / max(len(rows), 1)),
            "mean_joint_error": distribution_stats([float(row["mean_joint_error"]) for row in rows]),
            "rmse_forward_error": distribution_stats([float(row["rmse_forward_error"]) for row in rows]),
            "ctrl_delta_mean": distribution_stats([float(row["ctrl_delta_mean"]) for row in rows]),
            "upper_saturation_fraction": distribution_stats([float(row["upper_saturation_fraction"]) for row in rows]),
            "relative_residual_mean": distribution_stats([float(row["relative_residual_mean"]) for row in rows]),
        }
    return out


def write_summary_report(
    report_path: Path,
    *,
    records_found: list[Path],
    trial_summaries: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    passed = [item for item in trial_summaries if bool(item.get("keep_for_training", False))]
    failed = [item for item in trial_summaries if not bool(item.get("keep_for_training", False))]
    best_trials = sorted(trial_summaries, key=lambda item: float(item.get("quality_score", 0.0)), reverse=True)[:5]
    worst_trials = sorted(trial_summaries, key=lambda item: float(item.get("quality_score", 0.0)))[:5]
    reject_counter = Counter(reason for item in failed for reason in item.get("reject_reasons", []))
    fallback_fraction_avg = float(np.mean([float(item.get("fallback_fraction", 0.0)) for item in trial_summaries])) if trial_summaries else 0.0
    per_gesture = build_per_gesture_summary(trial_summaries)
    lines = [
        "# Teacher Inverse Dynamics Summary",
        "",
        "This dataset is a model-based teacher reference derived from the MyoHand / MyoSuite simulation.",
        "It is not biological ground truth and should not be interpreted as clinical truth.",
        "",
        "## Repository Inspection",
        "",
        f"- Recorded qpos trajectories were loaded from `ctrl_est_*.csv` files under `{Path(args.input_path).resolve()}`.",
        "- The recording CSV schema was confirmed to contain frame-wise `qpos_*`, `exercise_name`, and `trial_id` columns.",
        f"- The MuJoCo / MyoHand environment was loaded through `create_probe_env()` using env id `{args.env_id}`.",
        "- The existing online qpos-to-ctrl path was found in `estimate_ctrl_from_qpos()` and `RealtimeQPController` inside `app/main.py`.",
        "",
        "## Teacher Label Construction",
        "",
        f"- qpos target source: `{'raw_qpos' if bool(args.use_raw_qpos_target) else 'smoothed_qpos_target'}`.",
        f"- qpos / qvel / qacc smoothing windows: `{int(args.qpos_smoothing_window)}`, `{int(args.qvel_smoothing_window)}`, `{int(args.qacc_smoothing_window)}`.",
        f"- qvel clip: `{float(args.qvel_clip):.6f}` | qacc clip: `{float(args.qacc_clip):.6f}` | tau clip: `{float(args.tau_clip):.6f}`.",
        f"- MuJoCo inverse dynamics was called per frame with constraints disabled = `{bool(args.disable_constraints)}`.",
        f"- Teacher mode: `{str(args.teacher_mode)}`.",
        f"- Teacher prior mode: `{str(args.teacher_prior_mode)}`.",
        f"- QP reg: `{float(args.qp_reg):.6f}` | prev ctrl weight: `{float(args.prev_ctrl_weight):.6f}` | static prior weight: `{float(args.static_prior_weight):.6f}`.",
        f"- Control bounds were clamped to `{float(args.ctrl_lb):.3f} <= u <= {float(args.ctrl_ub):.3f}`.",
        "",
        "## Validation and Filtering",
        "",
        f"- Forward replay used up to `{int(args.max_substeps_per_frame)}` MuJoCo steps per recorded frame interval.",
        f"- Validation window size: `{int(args.validation_window_size)}`.",
        f"- Mean joint error threshold: `{float(args.max_mean_joint_error):.6f}`.",
        f"- Forward RMSE threshold: `{float(args.max_rmse_forward_error):.6f}`.",
        f"- Upper saturation threshold: `{float(args.max_upper_saturation_fraction):.6f}`.",
        f"- Control jitter threshold: `{float(args.max_ctrl_delta_mean):.6f}`.",
        "",
        "## Trial Summary",
        "",
        f"- CSV files discovered: `{len(records_found)}`",
        f"- Trials processed: `{len(trial_summaries)}`",
        f"- Trials accepted: `{len(passed)}`",
        f"- Trials rejected: `{len(failed)}`",
        f"- Fallback usage mean fraction: `{fallback_fraction_avg:.6f}`",
        "",
        "## Per-Gesture Summary",
        "",
    ]
    for gesture, summary in per_gesture.items():
        lines.append(
            f"- `{gesture}`: pass-rate `{summary['pass_rate']:.3f}` "
            f"({summary['num_kept']}/{summary['num_trials']}) | "
            f"mean joint error `{summary['mean_joint_error']['mean']:.4f}` | "
            f"mean RMSE `{summary['rmse_forward_error']['mean']:.4f}`"
        )
    if best_trials:
        lines.extend(["", "## Best Trials by Quality Score", ""])
        for item in best_trials:
            lines.append(
                f"- `{item['trial_id']}` ({item['exercise_name']}): "
                f"score `{float(item['quality_score']):.6f}`, "
                f"joint `{float(item['mean_joint_error']):.4f}`, "
                f"rmse `{float(item['rmse_forward_error']):.4f}`"
            )
    if worst_trials:
        lines.extend(["", "## Worst Trials by Quality Score", ""])
        for item in worst_trials:
            lines.append(
                f"- `{item['trial_id']}` ({item['exercise_name']}): "
                f"score `{float(item['quality_score']):.6f}`, "
                f"joint `{float(item['mean_joint_error']):.4f}`, "
                f"rmse `{float(item['rmse_forward_error']):.4f}`"
            )
    if reject_counter:
        lines.extend(["", "## Reject Reason Counts", ""])
        for reason, count in sorted(reject_counter.items()):
            lines.append(f"- `{reason}`: `{int(count)}`")
    lines.extend(
        [
            "",
            "## Metric Distributions",
            "",
        ]
    )
    for metric in ("mean_joint_error", "rmse_forward_error", "ctrl_delta_mean", "upper_saturation_fraction", "relative_residual_mean"):
        stats = distribution_stats([float(item.get(metric, 0.0)) for item in trial_summaries])
        lines.append(
            f"- `{metric}`: mean `{stats['mean']:.6f}`, median `{stats['median']:.6f}`, "
            f"p75 `{stats['p75']:.6f}`, p95 `{stats['p95']:.6f}`"
        )
    lines.extend(
        [
            "",
            "## Output Layout",
            "",
            f"- Accepted per-trial CSVs: `{output_dir / 'accepted_trials'}`",
            f"- Rejected per-trial CSVs: `{output_dir / 'rejected_trials'}`",
            f"- Trial summary CSV: `{output_dir / 'teacher_trial_summary.csv'}`",
            f"- Dataset summary JSON: `{output_dir / 'teacher_dataset_summary.json'}`",
            "",
            "## Assumptions",
            "",
            "- The teacher labels are consistent with the MyoHand model and the chosen actuator inversion approximation.",
            "- Exact actuator inversion is underdetermined under bounded muscle controls, so the final teacher uses a linearized inverse-dynamics solve stabilized by a qpos-derived static prior when configured.",
            "- Recorded Leap trajectories are treated as target kinematics and are not assumed to be dynamically exact.",
            "- When timestamps are irregular, finite-difference kinematics and forward replay are based on the observed frame intervals.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(float(item.strip())) for item in str(raw).split(",") if item.strip()]


def build_anchor_audit(
    output_dir: Path,
    summary_df: pd.DataFrame,
    *,
    anchor_trial_id: str,
    anchor_exercise_name: str | None,
    compare_top_k_failed: int,
) -> None:
    anchor_df = summary_df.loc[summary_df["trial_id"].astype(str) == str(anchor_trial_id)].copy()
    if anchor_exercise_name:
        anchor_df = anchor_df.loc[anchor_df["exercise_name"].astype(str) == str(anchor_exercise_name)]
    if anchor_df.empty:
        return
    anchor = anchor_df.iloc[0]
    failed = summary_df.loc[summary_df["keep_for_training"] == False].copy()  # noqa: E712
    failed = failed.sort_values(by=["quality_score", "rmse_forward_error"], ascending=[False, True]).head(max(1, int(compare_top_k_failed)))
    compare_cols = [
        "trial_id",
        "exercise_name",
        "keep_for_training",
        "mean_joint_error",
        "rmse_forward_error",
        "median_frame_error",
        "p95_frame_error",
        "mean_qvel_norm",
        "mean_qacc_norm",
        "mean_tau_norm",
        "upper_saturation_fraction",
        "ctrl_delta_mean",
        "relative_residual_mean",
        "quality_score",
    ]
    rows = [anchor[compare_cols].to_dict()]
    rows.extend(failed.loc[:, compare_cols].to_dict(orient="records"))
    audit_df = pd.DataFrame(rows)
    csv_path = output_dir / "anchor_trial_audit.csv"
    md_path = output_dir / "anchor_trial_audit.md"
    audit_df.to_csv(csv_path, index=False)
    lines = [
        "# Anchor Trial Audit",
        "",
        f"- Anchor trial: `{anchor_trial_id}`",
        f"- Anchor exercise: `{anchor_exercise_name or anchor['exercise_name']}`",
        f"- Compared failed trials: `{len(failed)}`",
        "",
        dataframe_to_markdown_simple(audit_df),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def build_frame_sequences(
    trial: dict[str, Any],
    qpos_target: np.ndarray,
    qvel_target: np.ndarray,
    qacc_target: np.ndarray,
    dt: np.ndarray,
    args: argparse.Namespace,
    model: Any,
    data: Any,
    actuator_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    m_raw, _ = _get_raw_model_data(model, data)
    tau_seq = np.zeros((qpos_target.shape[0], int(m_raw.nv)), dtype=float)
    ctrl_seq = np.zeros((qpos_target.shape[0], int(m_raw.nu)), dtype=float)
    solver_infos: list[dict[str, Any]] = []
    prev_ctrl = None
    for i in range(qpos_target.shape[0]):
        static_prior_ctrl = maybe_build_static_prior(
            args.teacher_mode,
            args.teacher_prior_mode,
            float(args.static_prior_weight),
            m_raw,
            data,
            qpos_target[i],
            prev_ctrl,
            str(trial["exercise_name"]),
            actuator_names,
        )
        tau_target = compute_inverse_dynamics_tau(
            m_raw,
            data,
            qpos_target[i],
            qvel_target[i],
            qacc_target[i],
            disable_constraints=bool(args.disable_constraints),
            tau_clip=float(args.tau_clip),
        )
        ctrl_teacher, info = solve_teacher_ctrl_frame(
            m_raw,
            data,
            tau_target,
            qpos_target[i],
            qvel_target[i],
            prev_ctrl=prev_ctrl,
            static_prior_ctrl=static_prior_ctrl,
            ctrl_lb=float(args.ctrl_lb),
            ctrl_ub=float(args.ctrl_ub),
            qp_reg=float(args.qp_reg),
            prev_ctrl_weight=float(args.prev_ctrl_weight),
            static_prior_weight=float(args.static_prior_weight),
            prior_value=float(args.prior_value),
            teacher_mode=str(args.teacher_mode),
            exercise_name=str(trial["exercise_name"]),
            actuator_names=actuator_names,
        )
        tau_seq[i] = tau_target
        ctrl_seq[i] = ctrl_teacher
        solver_infos.append(info)
        prev_ctrl = ctrl_teacher.copy()
    return tau_seq, ctrl_seq, solver_infos


def summarize_dataset_json(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    records: list[Path],
    trial_summaries: list[dict[str, Any]],
    actuator_names: list[str],
    combined_csv: Path | None,
) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "env_id": str(args.env_id),
        "teacher_mode": str(args.teacher_mode),
        "teacher_prior_mode": str(args.teacher_prior_mode),
        "use_raw_qpos_target": bool(args.use_raw_qpos_target),
        "actuator_names": actuator_names,
        "num_csv_discovered": int(len(records)),
        "num_trials_processed": int(len(trial_summaries)),
        "num_trials_kept": int(sum(1 for item in trial_summaries if bool(item["keep_for_training"]))),
        "num_trials_rejected": int(sum(1 for item in trial_summaries if not bool(item["keep_for_training"]))),
        "combined_dataset_csv": str(combined_csv) if combined_csv is not None else "",
        "per_gesture_summary": build_per_gesture_summary(trial_summaries),
        "trial_summaries": trial_summaries,
    }


def run_teacher_pipeline(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    report_path: Path,
    emit_datasets: bool,
    emit_reports: bool,
    summary_csv_name: str,
    summary_json_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_path = Path(args.input_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if emit_datasets:
        (output_dir / "accepted_trials").mkdir(parents=True, exist_ok=True)
        (output_dir / "rejected_trials").mkdir(parents=True, exist_ok=True)

    records = discover_recording_csvs(input_path)
    if not records:
        raise SystemExit(f"No ctrl_est_*.csv recordings found under {input_path}")
    if int(args.max_trials) > 0:
        records = records[: int(args.max_trials)]

    trial_summaries: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    kept_frame_tables: list[pd.DataFrame] = []
    env = None
    try:
        env, model, data = create_probe_env(str(args.env_id))
        m_raw, _ = _get_raw_model_data(model, data)
        actuator_names = [mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"muscle_{i}" for i in range(int(m_raw.nu))]
        for csv_path in records:
            trial = load_recorded_trial(csv_path)
            if trial is None:
                print(f"[WARN] Skipping {csv_path.name}: missing qpos or metadata.")
                continue
            qpos_raw = np.asarray(trial["qpos_raw"], dtype=float)
            if int(args.max_frames_per_trial) > 0:
                cap = int(args.max_frames_per_trial)
                qpos_raw = qpos_raw[:cap]
                trial["frame_indices"] = np.asarray(trial["frame_indices"], dtype=int)[:cap]
                if trial["timestamps"] is not None:
                    trial["timestamps"] = np.asarray(trial["timestamps"], dtype=float)[:cap]
            qpos_target, qvel_target, qacc_target, dt = estimate_kinematics(
                qpos_raw,
                trial["timestamps"],
                fallback_dt=float(args.timestep),
                qpos_smoothing_window=int(args.qpos_smoothing_window),
                qvel_smoothing_window=int(args.qvel_smoothing_window),
                qacc_smoothing_window=int(args.qacc_smoothing_window),
                qvel_clip=float(args.qvel_clip),
                qacc_clip=float(args.qacc_clip),
                use_raw_qpos_target=bool(args.use_raw_qpos_target),
            )
            times, _ = normalize_timestamps(trial["timestamps"], qpos_target.shape[0], float(args.timestep))
            print(f"[INFO] Teacher solve {trial['exercise_name']} / {trial['trial_id']} | frames={qpos_target.shape[0]}")
            tau_seq, ctrl_seq, solver_infos = build_frame_sequences(
                trial,
                qpos_target,
                qvel_target,
                qacc_target,
                dt,
                args,
                m_raw,
                data,
                actuator_names,
            )
            validation = forward_validate_sequence(
                m_raw,
                qpos_target,
                qvel_target,
                ctrl_seq,
                dt,
                max_substeps=int(args.max_substeps_per_frame),
                validation_window_size=int(args.validation_window_size),
            )
            keep, quality = summarize_trial_quality(
                trial,
                ctrl_seq,
                solver_infos,
                validation,
                qvel_target=qvel_target,
                qacc_target=qacc_target,
                tau_seq=tau_seq,
                max_mean_joint_error=float(args.max_mean_joint_error),
                max_rmse_forward_error=float(args.max_rmse_forward_error),
                max_upper_saturation_fraction=float(args.max_upper_saturation_fraction),
                max_ctrl_delta_mean=float(args.max_ctrl_delta_mean),
            )
            quality["kept_dataset_path"] = ""
            frame_table = build_frame_table(
                trial,
                times,
                dt,
                qpos_raw,
                qpos_target,
                qvel_target,
                qacc_target,
                tau_seq,
                ctrl_seq,
                solver_infos,
                validation,
                suspicious_frame_error=float(args.suspicious_frame_error),
                suspicious_qacc_norm=float(args.suspicious_qacc_norm),
                suspicious_tau_norm=float(args.suspicious_tau_norm),
            )
            dst_path = None
            if emit_datasets:
                dst_dir = output_dir / ("accepted_trials" if keep else "rejected_trials")
                dst_path = dst_dir / f"{trial['exercise_name']}__{trial['trial_id']}__teacher.csv"
                save_trial_dataset(frame_table, dst_path)
                quality["kept_dataset_path"] = str(dst_path)
            trial_summaries.append(quality)
            if keep:
                kept_frame_tables.append(frame_table)
            dataset_rows.append(
                {
                    "trial_id": str(trial["trial_id"]),
                    "exercise_name": str(trial["exercise_name"]),
                    "source_csv": str(trial["source_csv"]),
                    "keep_for_training": bool(keep),
                    **{k: v for k, v in quality.items() if k not in {"source_csv", "exercise_name", "trial_id", "reject_reasons", "per_joint_rmse", "kept_dataset_path"}},
                    "reject_reasons": "|".join(quality["reject_reasons"]),
                    "dataset_path": str(dst_path) if dst_path is not None else "",
                }
            )
        if not trial_summaries:
            raise SystemExit("No valid trials were processed.")

        summary_df = pd.DataFrame(dataset_rows)
        summary_df = summary_df.sort_values(by=["exercise_name", "trial_id"]).reset_index(drop=True)
        summary_csv = output_dir / summary_csv_name
        summary_df.to_csv(summary_csv, index=False)

        combined_csv_path: Path | None = None
        if emit_datasets and kept_frame_tables:
            combined_csv_path = output_dir / "teacher_dataset_combined.csv"
            pd.concat(kept_frame_tables, axis=0, ignore_index=True).to_csv(combined_csv_path, index=False)

        summary_json = summarize_dataset_json(
            input_path,
            output_dir,
            args,
            records,
            trial_summaries,
            actuator_names,
            combined_csv_path,
        )
        (output_dir / summary_json_name).write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

        if emit_reports:
            write_summary_report(report_path, records_found=records, trial_summaries=trial_summaries, output_dir=output_dir, args=args)
            if str(args.anchor_trial_id or "").strip():
                build_anchor_audit(
                    output_dir,
                    summary_df,
                    anchor_trial_id=str(args.anchor_trial_id),
                    anchor_exercise_name=str(args.anchor_exercise_name) if args.anchor_exercise_name else None,
                    compare_top_k_failed=int(args.compare_top_k_failed),
                )
        return summary_df, summary_json
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def run_sweep(args: argparse.Namespace) -> int:
    sweep_rows: list[dict[str, Any]] = []
    qpos_windows = parse_int_list(args.sweep_qpos_windows)
    qvel_windows = parse_int_list(args.sweep_qvel_windows)
    qacc_windows = parse_int_list(args.sweep_qacc_windows)
    qp_regs = parse_float_list(args.sweep_qp_regs)
    prev_weights = parse_float_list(args.sweep_prev_ctrl_weights)
    static_weights = parse_float_list(args.sweep_static_prior_weights)
    substeps_values = parse_int_list(args.sweep_max_substeps)
    sweep_dir = Path(args.output_path).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    for qpos_w, qvel_w, qacc_w, qp_reg, prev_w, static_w, substeps in itertools.product(
        qpos_windows,
        qvel_windows,
        qacc_windows,
        qp_regs,
        prev_weights,
        static_weights,
        substeps_values,
    ):
        cfg = copy.deepcopy(args)
        cfg.run_sweep = False
        cfg.qpos_smoothing_window = int(qpos_w)
        cfg.qvel_smoothing_window = int(qvel_w)
        cfg.qacc_smoothing_window = int(qacc_w)
        cfg.qp_reg = float(qp_reg)
        cfg.prev_ctrl_weight = float(prev_w)
        cfg.static_prior_weight = float(static_w)
        cfg.max_substeps_per_frame = int(substeps)
        cfg.anchor_trial_id = ""
        cfg.anchor_exercise_name = ""
        summary_df, summary_json = run_teacher_pipeline(
            cfg,
            output_dir=sweep_dir / "_sweep_tmp",
            report_path=sweep_dir / "_sweep_tmp" / "report.md",
            emit_datasets=False,
            emit_reports=False,
            summary_csv_name="_tmp_summary.csv",
            summary_json_name="_tmp_summary.json",
        )
        per_gesture = summary_json["per_gesture_summary"]
        sweep_rows.append(
            {
                "qpos_smoothing_window": int(qpos_w),
                "qvel_smoothing_window": int(qvel_w),
                "qacc_smoothing_window": int(qacc_w),
                "qp_reg": float(qp_reg),
                "prev_ctrl_weight": float(prev_w),
                "static_prior_weight": float(static_w),
                "max_substeps_per_frame": int(substeps),
                "accepted_count": int(summary_json["num_trials_kept"]),
                "rejected_count": int(summary_json["num_trials_rejected"]),
                "pass_rate": float(summary_json["num_trials_kept"] / max(summary_json["num_trials_processed"], 1)),
                "mean_rmse_forward_error": float(summary_df["rmse_forward_error"].mean()),
                "median_rmse_forward_error": float(summary_df["rmse_forward_error"].median()),
                "mean_joint_error_mean": float(summary_df["mean_joint_error"].mean()),
                "median_mean_joint_error": float(summary_df["mean_joint_error"].median()),
                "open_pass_rate": float(per_gesture.get("open", {}).get("pass_rate", 0.0)),
                "fist_pass_rate": float(per_gesture.get("fist", {}).get("pass_rate", 0.0)),
            }
        )
    sweep_df = pd.DataFrame(sweep_rows).sort_values(by=["pass_rate", "mean_rmse_forward_error", "mean_joint_error_mean"], ascending=[False, True, True]).reset_index(drop=True)
    sweep_csv = sweep_dir / "sweep_summary.csv"
    sweep_md = sweep_dir / "sweep_summary.md"
    sweep_df.to_csv(sweep_csv, index=False)
    lines = [
        "# Teacher Inverse Dynamics Sweep Summary",
        "",
        f"- Configurations evaluated: `{len(sweep_df)}`",
        "",
        dataframe_to_markdown_simple(sweep_df.head(20)),
        "",
    ]
    sweep_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote: {sweep_csv}")
    print(f"[INFO] Wrote: {sweep_md}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline inverse-dynamics teacher labels from recorded qpos trajectories.")
    parser.add_argument("--input-path", type=str, default="records", help="CSV file or directory containing ctrl_est_*.csv recordings.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save teacher-label datasets.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown report path.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id.")
    parser.add_argument("--timestep", type=float, default=1.0 / 60.0, help="Fallback timestep when CSV timestamps are missing or invalid.")
    parser.add_argument("--smoothing-window", type=int, default=None, help="Legacy compatibility option. If provided, maps to qpos smoothing only.")
    parser.add_argument("--qpos-smoothing-window", type=int, default=5, help="Moving-average window for qpos target generation.")
    parser.add_argument("--qvel-smoothing-window", type=int, default=3, help="Optional smoothing window for qvel target.")
    parser.add_argument("--qacc-smoothing-window", type=int, default=3, help="Optional smoothing window for qacc target.")
    parser.add_argument("--use-raw-qpos-target", action="store_true", help="Use raw recorded qpos as the teacher target instead of the smoothed qpos target.")
    parser.add_argument("--qvel-clip", type=float, default=0.0, help="Elementwise clip for qvel target. Use <=0 to disable.")
    parser.add_argument("--qacc-clip", type=float, default=0.0, help="Elementwise clip for qacc target. Use <=0 to disable.")
    parser.add_argument("--tau-clip", type=float, default=0.0, help="Elementwise clip for inverse-dynamics tau target. Use <=0 to disable.")
    parser.add_argument("--disable-constraints", action="store_true", help="Disable MuJoCo constraints during inverse dynamics.")
    parser.add_argument("--ctrl-lb", type=float, default=0.0, help="Lower bound for teacher control.")
    parser.add_argument("--ctrl-ub", type=float, default=1.0, help="Upper bound for teacher control.")
    parser.add_argument("--qp-reg", type=float, default=1e-3, help="Regularization weight for actuator-space solve.")
    parser.add_argument("--prev-ctrl-weight", type=float, default=5e-2, help="Penalty weight for staying close to the previous control.")
    parser.add_argument("--static-prior-weight", type=float, default=1e-1, help="Penalty weight for staying close to the static qpos-derived prior.")
    parser.add_argument("--prior-value", type=float, default=0.0, help="Fallback prior value when no static prior is used.")
    parser.add_argument("--teacher-prior-mode", type=str, default="static_qpos", choices=("zero", "static_qpos"), help="Source of static prior before dynamic correction.")
    parser.add_argument("--teacher-mode", type=str, default="dynamic_with_fallback", choices=("dynamic_only", "dynamic_with_static_prior", "dynamic_with_fallback"), help="Teacher solve strategy.")
    parser.add_argument("--max-mean-joint-error", type=float, default=0.20, help="Reject trial if mean joint error exceeds this threshold.")
    parser.add_argument("--max-rmse-forward-error", type=float, default=0.75, help="Reject trial if forward RMSE exceeds this threshold.")
    parser.add_argument("--max-upper-saturation-fraction", type=float, default=0.35, help="Reject trial if too many teacher controls saturate near 1.0.")
    parser.add_argument("--max-ctrl-delta-mean", type=float, default=0.30, help="Reject trial if mean frame-to-frame control change is too large.")
    parser.add_argument("--max-substeps-per-frame", type=int, default=150, help="Maximum MuJoCo steps used per recorded frame interval during validation.")
    parser.add_argument("--validation-window-size", type=int, default=0, help="Window size for replay RMSE diagnostics. Use <=0 to disable.")
    parser.add_argument("--suspicious-frame-error", type=float, default=1.0, help="Flag frame as suspicious when forward error exceeds this threshold.")
    parser.add_argument("--suspicious-qacc-norm", type=float, default=10.0, help="Flag frame as suspicious when qacc norm exceeds this threshold.")
    parser.add_argument("--suspicious-tau-norm", type=float, default=10.0, help="Flag frame as suspicious when tau norm exceeds this threshold.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap for testing. Use <=0 for all trials.")
    parser.add_argument("--max-frames-per-trial", type=int, default=0, help="Optional frame cap per trial. Use <=0 for all frames.")
    parser.add_argument("--anchor-trial-id", type=str, default="", help="Optional anchor trial id for comparison audit.")
    parser.add_argument("--anchor-exercise-name", type=str, default="", help="Optional gesture label for anchor trial filtering.")
    parser.add_argument("--compare-top-k-failed", type=int, default=5, help="Compare anchor trial against top-K failed trials.")
    parser.add_argument("--run-sweep", action="store_true", help="Run an internal parameter sweep instead of a single dataset build.")
    parser.add_argument("--sweep-qpos-windows", type=str, default="3,5,7", help="Comma-separated qpos smoothing windows for sweep mode.")
    parser.add_argument("--sweep-qvel-windows", type=str, default="1,3", help="Comma-separated qvel smoothing windows for sweep mode.")
    parser.add_argument("--sweep-qacc-windows", type=str, default="1,3", help="Comma-separated qacc smoothing windows for sweep mode.")
    parser.add_argument("--sweep-qp-regs", type=str, default="0.001,0.01", help="Comma-separated qp_reg values for sweep mode.")
    parser.add_argument("--sweep-prev-ctrl-weights", type=str, default="0.01,0.05", help="Comma-separated prev ctrl weights for sweep mode.")
    parser.add_argument("--sweep-static-prior-weights", type=str, default="0.05,0.1", help="Comma-separated static prior weights for sweep mode.")
    parser.add_argument("--sweep-max-substeps", type=str, default="100,150", help="Comma-separated max substeps values for sweep mode.")
    args = parser.parse_args()
    if args.smoothing_window is not None:
        args.qpos_smoothing_window = int(args.smoothing_window)
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_path).resolve()
    report_path = Path(args.report_path).resolve()
    if bool(args.run_sweep):
        return run_sweep(args)
    summary_df, summary_json = run_teacher_pipeline(
        args,
        output_dir=output_dir,
        report_path=report_path,
        emit_datasets=True,
        emit_reports=True,
        summary_csv_name="teacher_trial_summary.csv",
        summary_json_name="teacher_dataset_summary.json",
    )
    print(f"[INFO] Wrote: {output_dir / 'teacher_trial_summary.csv'}")
    if (output_dir / 'teacher_dataset_combined.csv').exists():
        print(f"[INFO] Wrote: {output_dir / 'teacher_dataset_combined.csv'}")
    print(f"[INFO] Wrote: {output_dir / 'teacher_dataset_summary.json'}")
    print(f"[INFO] Wrote: {report_path}")
    if str(args.anchor_trial_id or "").strip():
        if (output_dir / "anchor_trial_audit.csv").exists():
            print(f"[INFO] Wrote: {output_dir / 'anchor_trial_audit.csv'}")
            print(f"[INFO] Wrote: {output_dir / 'anchor_trial_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
