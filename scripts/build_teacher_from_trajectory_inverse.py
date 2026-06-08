from __future__ import annotations

"""Compatibility backend for offline biomechanical ctrl_ref construction.

Canonical users should call scripts/offline_biomech/build_biomech_ctrl_reference.py.
This module keeps legacy function/file names for backward compatibility.

Pipeline:
    recorded qpos_t -> qvel_t / qacc_t from trajectory
    -> mj_inverse -> qfrc_inverse_t
    -> bounded actuator-space solve -> ctrl_ref_t

This is an offline, model-derived biomechanical reference for the MyoHand simulator.
It is not biological ground truth.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import DEFAULT_MYOSUITE_ENV_ID, _dense_moment, _gain_bias, _get_raw_model_data, _torque_weights, create_probe_env  # noqa: E402
from scripts.build_teacher_from_qpos import repository_has_qfrc_scaler  # noqa: E402
from scripts.build_teacher_inverse_dynamics import (  # noqa: E402
    build_per_gesture_summary,
    build_substeps_from_dt,
    build_frame_table,
    clip_signal,
    compute_inverse_dynamics_tau,
    dataframe_to_markdown_simple,
    discover_recording_csvs,
    estimate_kinematics,
    forward_validate_sequence,
    load_recorded_trial,
    maybe_build_static_prior,
    normalize_timestamps,
    save_trial_dataset,
    solve_teacher_ctrl_frame,
    summarize_trial_quality,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "teacher_from_trajectory_inverse"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "trajectory_inverse_teacher_summary.md"


def build_trajectory_teacher_frame_table(
    trial: dict[str, Any],
    *,
    times: np.ndarray,
    dt: np.ndarray,
    qpos_raw: np.ndarray,
    qpos_target: np.ndarray,
    qvel_traj: np.ndarray,
    qacc_traj: np.ndarray,
    tau_seq: np.ndarray,
    ctrl_seq: np.ndarray,
    solver_infos: list[dict[str, Any]],
    validation: dict[str, Any],
    suspicious_frame_error: float,
    suspicious_qacc_norm: float,
    suspicious_tau_norm: float,
) -> pd.DataFrame:
    df = build_frame_table(
        trial,
        times,
        dt,
        qpos_raw,
        qpos_target,
        qvel_traj,
        qacc_traj,
        tau_seq,
        ctrl_seq,
        solver_infos,
        validation,
        suspicious_frame_error=suspicious_frame_error,
        suspicious_qacc_norm=suspicious_qacc_norm,
        suspicious_tau_norm=suspicious_tau_norm,
    )
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if col.startswith("qvel_target_"):
            rename_map[col] = col.replace("qvel_target_", "qvel_traj_")
        elif col.startswith("qacc_target_"):
            rename_map[col] = col.replace("qacc_target_", "qacc_traj_")
    if rename_map:
        df = df.rename(columns=rename_map)
    if "qvel_norm" in df.columns:
        df = df.rename(columns={"qvel_norm": "qvel_traj_norm"})
    if "qacc_norm" in df.columns:
        df = df.rename(columns={"qacc_norm": "qacc_traj_norm"})
    return df


def evaluate_ctrl_diagnostic(
    model: Any,
    data: Any,
    tau_target: np.ndarray,
    qpos_ref: np.ndarray,
    qvel_ref: np.ndarray,
    ctrl: np.ndarray,
    *,
    prev_ctrl: np.ndarray | None,
    static_prior_ctrl: np.ndarray | None,
    ctrl_lb: float,
    ctrl_ub: float,
    solver_mode: str,
) -> dict[str, Any]:
    """Recompute linearized actuator diagnostics for an already chosen control.

    This is used after short-horizon optimization so residual metrics reflect
    the final sequence, not only the warm-start framewise solve.
    """

    set_state_qacc = np.zeros(int(model.nv), dtype=float)
    from scripts.build_teacher_inverse_dynamics import set_full_state  # noqa: E402

    set_full_state(model, data, qpos_ref, qvel_ref, set_state_qacc, zero_act=True)
    m_raw, d_raw = _get_raw_model_data(model, data)
    mujoco.mj_step1(m_raw, d_raw)
    moment = _dense_moment(m_raw, d_raw)
    gain, bias = _gain_bias(m_raw, d_raw)
    A = moment * gain[np.newaxis, :]
    tau_target = np.asarray(tau_target, dtype=float).reshape(-1)
    ctrl = np.clip(np.asarray(ctrl, dtype=float).reshape(-1), float(ctrl_lb), float(ctrl_ub))
    tau_hat = A @ ctrl + moment @ bias
    residual = tau_hat - tau_target
    prev = np.zeros_like(ctrl) if prev_ctrl is None else np.asarray(prev_ctrl, dtype=float).reshape(-1)
    prev = np.pad(prev[: ctrl.shape[0]], (0, max(0, ctrl.shape[0] - prev.shape[0])), constant_values=0.0)
    static_prior = np.full_like(ctrl, float(ctrl_lb))
    if static_prior_ctrl is not None:
        static_prior_in = np.asarray(static_prior_ctrl, dtype=float).reshape(-1)
        static_prior[: min(len(static_prior), len(static_prior_in))] = static_prior_in[: min(len(static_prior), len(static_prior_in))]
    col_norms = np.linalg.norm(A, axis=0) if A.size else np.zeros(0, dtype=float)
    return {
        "solver_mode": solver_mode,
        "tau_reconstructed": tau_hat,
        "residual_norm": float(np.linalg.norm(residual)),
        "relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(tau_target), 1e-8)),
        "matrix_rank": int(np.linalg.matrix_rank(A)) if A.size else 0,
        "matrix_fro_norm": float(np.linalg.norm(A)) if A.size else 0.0,
        "matrix_col_norm_mean": float(np.mean(col_norms)) if col_norms.size else 0.0,
        "ctrl_norm": float(np.linalg.norm(ctrl)),
        "prev_distance": float(np.linalg.norm(ctrl - prev)),
        "static_prior_distance": float(np.linalg.norm(ctrl - static_prior)),
        "lower_saturation_fraction": float(np.mean(ctrl <= (float(ctrl_lb) + 1e-6))) if ctrl.size else 0.0,
        "upper_saturation_fraction": float(np.mean(ctrl >= (float(ctrl_ub) - 1e-6))) if ctrl.size else 0.0,
        "tau_target_norm": float(np.linalg.norm(tau_target)),
        "tau_hat_norm": float(np.linalg.norm(tau_hat)),
    }


def compute_tracking_tau_assist(
    *,
    qpos_current: np.ndarray,
    qvel_current: np.ndarray,
    qpos_next: np.ndarray,
    qvel_next: np.ndarray,
    nv: int,
    kp: float,
    kv: float,
) -> np.ndarray:
    """Small offline tracking torque assist for forward-replay ctrl references.

    Pure inverse dynamics can be very weak for slow recorded trajectories,
    especially legacy CSVs with large timestamp gaps. This term keeps the
    target explicitly tied to the next recorded posture while remaining a
    generalized torque target for the same local actuator-space solve.
    """

    out = np.zeros(max(0, int(nv)), dtype=float)
    if float(kp) == 0.0 and float(kv) == 0.0:
        return out
    q0 = np.asarray(qpos_current, dtype=float).reshape(-1)
    q1 = np.asarray(qpos_next, dtype=float).reshape(-1)
    v0 = np.asarray(qvel_current, dtype=float).reshape(-1)
    v1 = np.asarray(qvel_next, dtype=float).reshape(-1)
    nq = min(len(out), len(q0), len(q1))
    nv_count = min(len(out), len(v0), len(v1))
    if nq:
        out[:nq] += float(kp) * (q1[:nq] - q0[:nq])
    if nv_count:
        out[:nv_count] += float(kv) * (v1[:nv_count] - v0[:nv_count])
    return np.where(np.isfinite(out), out, 0.0)


FINGER_FLEXION_QPOS_INDICES = (7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22)
DISTAL_FLEXION_QPOS_INDICES = (9, 10, 13, 14, 17, 18, 21, 22)
FINGER_QPOS_GROUPS = {
    "thumb": (3, 4, 5, 6),
    "index": (7, 9, 10),
    "middle": (11, 13, 14),
    "ring": (15, 17, 18),
    "pinky": (19, 21, 22),
}


def build_qpos_feedback_weights(
    nv: int,
    *,
    finger_flexion_weight: float,
    distal_flexion_weight: float,
    adaptive_qpos_weights: np.ndarray | None = None,
) -> np.ndarray:
    weights = np.ones(max(0, int(nv)), dtype=float)
    for idx in FINGER_FLEXION_QPOS_INDICES:
        if idx < len(weights):
            weights[idx] *= max(0.0, float(finger_flexion_weight))
    for idx in DISTAL_FLEXION_QPOS_INDICES:
        if idx < len(weights):
            weights[idx] *= max(0.0, float(distal_flexion_weight))
    if adaptive_qpos_weights is not None:
        adaptive = np.asarray(adaptive_qpos_weights, dtype=float).reshape(-1)
        weights[: min(len(weights), len(adaptive))] *= adaptive[: min(len(weights), len(adaptive))]
    return weights


def build_adaptive_finger_feedback_weights(
    *,
    qpos_target: np.ndarray,
    forward_qpos_seq: np.ndarray,
    nv: int,
    enabled: bool,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    weights = np.ones(max(0, int(nv)), dtype=float)
    if not bool(enabled):
        return weights
    target = np.asarray(qpos_target, dtype=float)
    forward = np.asarray(forward_qpos_seq, dtype=float)
    if target.ndim != 2 or forward.ndim != 2 or target.shape[0] == 0 or forward.shape[0] == 0:
        return weights
    n = min(target.shape[0], forward.shape[0])
    width = min(target.shape[1], forward.shape[1], len(weights))
    if n <= 0 or width <= 0:
        return weights
    err = forward[:n, :width] - target[:n, :width]
    group_scores: dict[str, float] = {}
    for finger, indices in FINGER_QPOS_GROUPS.items():
        valid = [idx for idx in indices if idx < width]
        if not valid:
            continue
        group_scores[finger] = float(np.sqrt(np.mean(np.square(err[:, valid]))))
    if not group_scores:
        return weights
    baseline = float(np.median([value for value in group_scores.values() if np.isfinite(value)]))
    baseline = max(baseline, 1e-6)
    lo = max(0.0, float(min_weight))
    hi = max(lo, float(max_weight))
    for finger, score in group_scores.items():
        mult = float(np.clip(score / baseline, lo, hi))
        for idx in FINGER_QPOS_GROUPS[finger]:
            if idx < len(weights):
                weights[idx] *= mult
    return weights


def parse_feedback_alphas(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            continue
        if value > 0.0:
            values.append(float(value))
    return values or [1.0]


def compute_qpos_mismatch_feedback_tau(
    *,
    qpos_target: np.ndarray,
    qvel_target: np.ndarray,
    forward_qpos_seq: np.ndarray,
    forward_qvel_seq: np.ndarray,
    nv: int,
    kp: float,
    kv: float,
    max_tau: float,
    finger_flexion_weight: float,
    distal_flexion_weight: float,
    adaptive_qpos_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Convert forward qpos mismatch into an additional tau target sequence.

    ctrl[i] advances the simulated state from frame i toward frame i + 1, so
    feedback for ctrl[i] is based on the qpos mismatch at frame i + 1. This is
    intentionally simple and local: it makes the existing inverse-dynamics/OSQP
    solve aware of the actual forward replay error without adding a new model.
    """

    qpos_target = np.asarray(qpos_target, dtype=float)
    qvel_target = np.asarray(qvel_target, dtype=float)
    forward = np.asarray(forward_qpos_seq, dtype=float)
    forward_vel = np.asarray(forward_qvel_seq, dtype=float)
    n = int(qpos_target.shape[0])
    feedback = np.zeros((n, max(0, int(nv))), dtype=float)
    if n <= 1 or forward.ndim != 2 or forward.shape[0] < n or (float(kp) == 0.0 and float(kv) == 0.0):
        return feedback
    weights = build_qpos_feedback_weights(
        int(nv),
        finger_flexion_weight=float(finger_flexion_weight),
        distal_flexion_weight=float(distal_flexion_weight),
        adaptive_qpos_weights=adaptive_qpos_weights,
    )
    for i in range(n - 1):
        target_idx = i + 1
        qerr = np.zeros(int(nv), dtype=float)
        verr = np.zeros(int(nv), dtype=float)
        width = min(int(nv), qpos_target.shape[1], forward.shape[1])
        vwidth = min(int(nv), qvel_target.shape[1], forward_vel.shape[1]) if forward_vel.ndim == 2 and forward_vel.shape[0] > target_idx else 0
        if width <= 0 and vwidth <= 0:
            continue
        if width > 0:
            qerr[:width] = qpos_target[target_idx, :width] - forward[target_idx, :width]
        if vwidth > 0:
            verr[:vwidth] = qvel_target[target_idx, :vwidth] - forward_vel[target_idx, :vwidth]
        feedback[i] = weights * ((float(kp) * qerr) + (float(kv) * verr))
    if float(max_tau) > 0.0:
        feedback = np.clip(feedback, -float(max_tau), float(max_tau))
    return np.where(np.isfinite(feedback), feedback, 0.0)


def compute_weighted_qpos_mismatch_objective(
    *,
    qpos_target: np.ndarray,
    qvel_target: np.ndarray,
    validation: dict[str, Any],
    ctrl_seq: np.ndarray,
    nv: int,
    qpos_weight: float,
    qvel_weight: float,
    finger_flexion_weight: float,
    distal_flexion_weight: float,
    final_weight: float,
    control_weight: float,
    smoothness_weight: float,
) -> float:
    forward = np.asarray(validation.get("forward_qpos_seq", []), dtype=float)
    forward_vel = np.asarray(validation.get("forward_qvel_seq", []), dtype=float)
    target = np.asarray(qpos_target, dtype=float)
    target_vel = np.asarray(qvel_target, dtype=float)
    if forward.ndim != 2 or target.ndim != 2 or forward.shape[0] == 0 or target.shape[0] == 0:
        return float(validation.get("rmse_forward_error", 0.0))
    n = min(forward.shape[0], target.shape[0])
    width = min(forward.shape[1], target.shape[1], int(nv))
    if n <= 0 or width <= 0:
        return float(validation.get("rmse_forward_error", 0.0))
    weights = build_qpos_feedback_weights(
        int(nv),
        finger_flexion_weight=float(finger_flexion_weight),
        distal_flexion_weight=float(distal_flexion_weight),
    )[:width]
    err = forward[:n, :width] - target[:n, :width]
    mean_term = float(np.mean(np.square(err) * weights[np.newaxis, :]))
    velocity_term = 0.0
    if forward_vel.ndim == 2 and target_vel.ndim == 2 and forward_vel.shape[0] > 0 and target_vel.shape[0] > 0:
        vn = min(forward_vel.shape[0], target_vel.shape[0], n)
        vwidth = min(forward_vel.shape[1], target_vel.shape[1], width)
        if vn > 0 and vwidth > 0:
            verr = forward_vel[:vn, :vwidth] - target_vel[:vn, :vwidth]
            velocity_term = float(np.mean(np.square(verr) * weights[:vwidth][np.newaxis, :]))
    final_err = err[-1]
    final_term = float(np.mean(np.square(final_err) * weights))
    ctrl_arr = np.asarray(ctrl_seq, dtype=float)
    control_term = float(np.mean(np.square(ctrl_arr))) if ctrl_arr.size else 0.0
    if ctrl_arr.ndim == 2 and ctrl_arr.shape[0] > 1:
        smoothness_term = float(np.mean(np.square(np.diff(ctrl_arr, axis=0))))
    else:
        smoothness_term = 0.0
    return float(
        max(0.0, float(qpos_weight)) * mean_term
        + max(0.0, float(qvel_weight)) * velocity_term
        + max(0.0, float(final_weight)) * final_term
        + max(0.0, float(control_weight)) * control_term
        + max(0.0, float(smoothness_weight)) * smoothness_term
    )


def solve_ctrl_sequence_from_tau(
    trial: dict[str, Any],
    *,
    model: Any,
    data: Any,
    actuator_names: list[str],
    qpos_target: np.ndarray,
    qvel_traj: np.ndarray,
    dt: np.ndarray,
    tau_seq: np.ndarray,
    static_prior_seq: np.ndarray | None,
    args: argparse.Namespace,
    feedback_iter_index: int = 0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    m_raw, _ = _get_raw_model_data(model, data)
    warm_ctrl_seq = np.zeros((qpos_target.shape[0], int(m_raw.nu)), dtype=float)
    warm_solver_infos: list[dict[str, Any]] = []
    prev_ctrl = None
    warm_teacher_mode = "dynamic_with_static_prior" if float(args.static_prior_weight) > 0.0 else "dynamic_only"
    for i in range(qpos_target.shape[0]):
        static_prior_ctrl = static_prior_seq[i] if static_prior_seq is not None and i < static_prior_seq.shape[0] else None
        ctrl_teacher, info = solve_teacher_ctrl_frame(
            m_raw,
            data,
            tau_seq[i],
            qpos_target[i],
            qvel_traj[i],
            prev_ctrl=prev_ctrl,
            static_prior_ctrl=static_prior_ctrl,
            ctrl_lb=float(args.ctrl_lb),
            ctrl_ub=float(args.ctrl_ub),
            qp_reg=float(args.qp_reg),
            prev_ctrl_weight=float(args.prev_ctrl_weight),
            static_prior_weight=float(args.static_prior_weight),
            prior_value=float(args.prior_value),
            teacher_mode=warm_teacher_mode,
            exercise_name=str(trial["exercise_name"]),
            actuator_names=actuator_names,
        )
        info["mismatch_feedback_iter"] = int(feedback_iter_index)
        warm_ctrl_seq[i] = np.asarray(ctrl_teacher, dtype=float).reshape(-1)
        warm_solver_infos.append(info)
        prev_ctrl = warm_ctrl_seq[i].copy()

    if str(args.teacher_mode).strip().lower() == "short_horizon_rollout":
        ctrl_seq, solver_infos = solve_teacher_short_horizon(
            trial,
            model=m_raw,
            data=data,
            actuator_names=actuator_names,
            qpos_target=qpos_target,
            qvel_traj=qvel_traj,
            dt=dt,
            tau_seq=tau_seq,
            warm_ctrl_seq=warm_ctrl_seq,
            static_prior_seq=static_prior_seq,
            args=args,
        )
        for info in solver_infos:
            info["mismatch_feedback_iter"] = int(feedback_iter_index)
        return ctrl_seq, solver_infos
    return warm_ctrl_seq, warm_solver_infos


def rollout_control_sequence(
    model: Any,
    *,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl_seq: np.ndarray,
    dt_seq: np.ndarray,
    max_substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the model forward for a short horizon and return the full state path."""

    from scripts.build_teacher_inverse_dynamics import set_full_state  # noqa: E402

    m_raw, _ = _get_raw_model_data(model, model.data if hasattr(model, "data") else None) if hasattr(model, "data") else (model, None)
    d_roll = mujoco.MjData(m_raw)
    ctrl_seq = np.asarray(ctrl_seq, dtype=float)
    dt_seq = np.asarray(dt_seq, dtype=float)
    qacc0 = np.zeros(int(m_raw.nv), dtype=float)
    set_full_state(m_raw, d_roll, qpos0, qvel0, qacc0, ctrl=ctrl_seq[0] if ctrl_seq.shape[0] else None, zero_act=True)
    substeps = build_substeps_from_dt(dt_seq, float(m_raw.opt.timestep), max_substeps)
    qpos_roll = [np.asarray(d_roll.qpos, dtype=float).copy()]
    qvel_roll = [np.asarray(d_roll.qvel, dtype=float).copy()]
    for i in range(ctrl_seq.shape[0]):
        ctrl = np.asarray(ctrl_seq[i], dtype=float).reshape(-1)
        d_roll.ctrl[:] = 0.0
        d_roll.ctrl[: min(int(m_raw.nu), len(ctrl))] = ctrl[: min(int(m_raw.nu), len(ctrl))]
        for _ in range(int(substeps[min(i, len(substeps) - 1)])):
            mujoco.mj_step(m_raw, d_roll)
        qpos_roll.append(np.asarray(d_roll.qpos, dtype=float).copy())
        qvel_roll.append(np.asarray(d_roll.qvel, dtype=float).copy())
    return np.asarray(qpos_roll, dtype=float), np.asarray(qvel_roll, dtype=float)


def compute_rollout_objective(
    *,
    qpos_roll: np.ndarray,
    qvel_roll: np.ndarray,
    qpos_target_seq: np.ndarray,
    qvel_target_seq: np.ndarray,
    ctrl_seq: np.ndarray,
    prior_seq: np.ndarray,
    prev_ctrl_anchor: np.ndarray | None,
    qpos_weight: float,
    qvel_weight: float,
    final_qpos_weight: float,
    finger_flexion_weight: float,
    distal_flexion_weight: float,
    u_weight: float,
    delta_u_weight: float,
    prior_weight: float,
) -> tuple[float, dict[str, float]]:
    """Compute a short-horizon rollout objective over the whole window."""

    qpos_roll = np.asarray(qpos_roll, dtype=float)
    qvel_roll = np.asarray(qvel_roll, dtype=float)
    qpos_target_seq = np.asarray(qpos_target_seq, dtype=float)
    qvel_target_seq = np.asarray(qvel_target_seq, dtype=float)
    ctrl_seq = np.asarray(ctrl_seq, dtype=float)
    prior_seq = np.asarray(prior_seq, dtype=float)

    qn = min(max(0, qpos_roll.shape[0] - 1), qpos_target_seq.shape[0])
    qwidth = min(qpos_roll.shape[1] if qpos_roll.ndim == 2 else 0, qpos_target_seq.shape[1] if qpos_target_seq.ndim == 2 else 0)
    qpos_err = np.empty((0, 0), dtype=float)
    pose_term = 0.0
    final_pose_term = 0.0
    if qn > 0 and qwidth > 0:
        q_weights = build_qpos_feedback_weights(
            qwidth,
            finger_flexion_weight=float(finger_flexion_weight),
            distal_flexion_weight=float(distal_flexion_weight),
        )[:qwidth]
        qpos_err = qpos_roll[1 : 1 + qn, :qwidth] - qpos_target_seq[:qn, :qwidth]
        pose_term = float(np.mean(np.sum(np.square(qpos_err) * q_weights[np.newaxis, :], axis=1))) if qpos_err.size else 0.0
        final_pose_term = float(np.sum(np.square(qpos_err[-1]) * q_weights)) if qpos_err.size else 0.0
    vn = min(max(0, qvel_roll.shape[0] - 1), qvel_target_seq.shape[0])
    vwidth = min(qvel_roll.shape[1] if qvel_roll.ndim == 2 else 0, qvel_target_seq.shape[1] if qvel_target_seq.ndim == 2 else 0, qwidth)
    vel_term = 0.0
    if vn > 0 and vwidth > 0:
        v_weights = build_qpos_feedback_weights(
            vwidth,
            finger_flexion_weight=float(finger_flexion_weight),
            distal_flexion_weight=float(distal_flexion_weight),
        )[:vwidth]
        qvel_err = qvel_roll[1 : 1 + vn, :vwidth] - qvel_target_seq[:vn, :vwidth]
        vel_term = float(np.mean(np.sum(np.square(qvel_err) * v_weights[np.newaxis, :], axis=1))) if qvel_err.size else 0.0
    u_term = float(np.mean(np.sum(np.square(ctrl_seq), axis=1))) if ctrl_seq.size else 0.0
    if ctrl_seq.shape[0] > 1:
        delta_term = float(np.mean(np.sum(np.square(np.diff(ctrl_seq, axis=0)), axis=1)))
    else:
        delta_term = 0.0
    if prev_ctrl_anchor is not None and ctrl_seq.shape[0] > 0:
        prev_anchor = np.asarray(prev_ctrl_anchor, dtype=float).reshape(-1)
        delta0 = float(np.sum(np.square(ctrl_seq[0] - prev_anchor[: ctrl_seq.shape[1]])))
        delta_term += delta0
    prior_term = float(np.mean(np.sum(np.square(ctrl_seq - prior_seq), axis=1))) if ctrl_seq.size else 0.0
    total = (
        float(qpos_weight) * pose_term
        + float(qvel_weight) * vel_term
        + float(final_qpos_weight) * final_pose_term
        + float(u_weight) * u_term
        + float(delta_u_weight) * delta_term
        + float(prior_weight) * prior_term
    )
    return total, {
        "pose_term": pose_term,
        "velocity_term": vel_term,
        "final_pose_term": final_pose_term,
        "u_term": u_term,
        "delta_u_term": delta_term,
        "prior_term": prior_term,
    }


def apply_activation_preemphasis(
    ctrl_seq: np.ndarray,
    *,
    rise_gain: float,
    ctrl_lb: float,
    ctrl_ub: float,
) -> np.ndarray:
    """Boost rising control edges so activation dynamics can catch up.

    MyoSuite muscle actuators do not instantly convert ctrl to force. This
    light-weight post-process is intentionally conservative and is only kept
    by the caller if forward validation improves.
    """

    ctrl = np.asarray(ctrl_seq, dtype=float)
    if ctrl.ndim != 2 or ctrl.shape[0] == 0 or float(rise_gain) <= 0.0:
        return np.clip(ctrl, float(ctrl_lb), float(ctrl_ub))
    out = ctrl.copy()
    prev = np.zeros(ctrl.shape[1], dtype=float)
    for idx in range(ctrl.shape[0]):
        delta = ctrl[idx] - prev
        out[idx] = ctrl[idx] + (float(rise_gain) * np.maximum(delta, 0.0))
        prev = ctrl[idx]
    return np.clip(np.where(np.isfinite(out), out, 0.0), float(ctrl_lb), float(ctrl_ub))


def optimize_short_horizon_controls(
    model: Any,
    data: Any,
    *,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    qpos_target_seq: np.ndarray,
    qvel_target_seq: np.ndarray,
    dt_seq: np.ndarray,
    tau_target_seq: np.ndarray,
    warm_ctrl_seq: np.ndarray,
    prior_ctrl_seq: np.ndarray,
    actuator_names: list[str],
    prev_ctrl_anchor: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize a short horizon using rollout-aware local updates.

    The framewise teacher solve provides the warm start. Each local update
    proposes a new control using the dynamic linearized solve at the current
    rolled-out state, then accepts only if the horizon rollout objective
    improves.
    """

    seq = np.asarray(warm_ctrl_seq, dtype=float).copy()
    prior_seq = np.asarray(prior_ctrl_seq, dtype=float).copy()
    qpos_roll, qvel_roll = rollout_control_sequence(
        model,
        qpos0=qpos0,
        qvel0=qvel0,
        ctrl_seq=seq,
        dt_seq=dt_seq,
        max_substeps=int(args.max_substeps_per_frame),
    )
    current_obj, current_terms = compute_rollout_objective(
        qpos_roll=qpos_roll,
        qvel_roll=qvel_roll,
        qpos_target_seq=qpos_target_seq,
        qvel_target_seq=qvel_target_seq,
        ctrl_seq=seq,
        prior_seq=prior_seq,
        prev_ctrl_anchor=prev_ctrl_anchor,
        qpos_weight=float(args.rollout_qpos_weight),
        qvel_weight=float(args.rollout_qvel_weight),
        final_qpos_weight=float(getattr(args, "rollout_final_qpos_weight", 0.0)),
        finger_flexion_weight=float(getattr(args, "finger_flexion_feedback_weight", 1.0)),
        distal_flexion_weight=float(getattr(args, "distal_flexion_feedback_weight", 1.0)),
        u_weight=float(args.rollout_u_weight),
        delta_u_weight=float(args.rollout_delta_u_weight),
        prior_weight=float(args.rollout_prior_weight),
    )
    accepted_updates = 0
    blend_alphas = [0.35, 0.65, 1.0]
    teacher_mode_warm = "dynamic_with_static_prior" if float(args.static_prior_weight) > 0.0 else "dynamic_only"
    for _ in range(max(1, int(args.horizon_iters))):
        improved = False
        for j in range(seq.shape[0]):
            prev_for_step = prev_ctrl_anchor if j == 0 else seq[j - 1]
            static_prior = prior_seq[j] if prior_seq.size else None
            try:
                candidate_ctrl, _ = solve_teacher_ctrl_frame(
                    model,
                    data,
                    tau_target_seq[j],
                    qpos_roll[j],
                    qvel_roll[j],
                    prev_ctrl=prev_for_step,
                    static_prior_ctrl=static_prior,
                    ctrl_lb=float(args.ctrl_lb),
                    ctrl_ub=float(args.ctrl_ub),
                    qp_reg=float(args.qp_reg),
                    prev_ctrl_weight=float(args.prev_ctrl_weight),
                    static_prior_weight=float(args.static_prior_weight),
                    prior_value=float(args.prior_value),
                    teacher_mode=teacher_mode_warm,
                    exercise_name=None,
                    actuator_names=actuator_names,
                )
            except Exception:
                continue
            base_seq = seq.copy()
            best_local_obj = current_obj
            best_local_seq = seq
            best_local_roll = (qpos_roll, qvel_roll)
            best_local_terms = current_terms
            for alpha in blend_alphas:
                proposal = np.clip((1.0 - float(alpha)) * seq[j] + float(alpha) * candidate_ctrl, float(args.ctrl_lb), float(args.ctrl_ub))
                trial_seq = base_seq.copy()
                trial_seq[j] = proposal
                trial_qpos_roll, trial_qvel_roll = rollout_control_sequence(
                    model,
                    qpos0=qpos0,
                    qvel0=qvel0,
                    ctrl_seq=trial_seq,
                    dt_seq=dt_seq,
                    max_substeps=int(args.max_substeps_per_frame),
                )
                trial_obj, trial_terms = compute_rollout_objective(
                    qpos_roll=trial_qpos_roll,
                    qvel_roll=trial_qvel_roll,
                    qpos_target_seq=qpos_target_seq,
                    qvel_target_seq=qvel_target_seq,
                    ctrl_seq=trial_seq,
                    prior_seq=prior_seq,
                    prev_ctrl_anchor=prev_ctrl_anchor,
                    qpos_weight=float(args.rollout_qpos_weight),
                    qvel_weight=float(args.rollout_qvel_weight),
                    final_qpos_weight=float(getattr(args, "rollout_final_qpos_weight", 0.0)),
                    finger_flexion_weight=float(getattr(args, "finger_flexion_feedback_weight", 1.0)),
                    distal_flexion_weight=float(getattr(args, "distal_flexion_feedback_weight", 1.0)),
                    u_weight=float(args.rollout_u_weight),
                    delta_u_weight=float(args.rollout_delta_u_weight),
                    prior_weight=float(args.rollout_prior_weight),
                )
                if trial_obj + 1e-9 < best_local_obj:
                    best_local_obj = trial_obj
                    best_local_seq = trial_seq
                    best_local_roll = (trial_qpos_roll, trial_qvel_roll)
                    best_local_terms = trial_terms
            if best_local_obj + 1e-9 < current_obj:
                seq = best_local_seq
                qpos_roll, qvel_roll = best_local_roll
                current_obj = best_local_obj
                current_terms = best_local_terms
                accepted_updates += 1
                improved = True
        if not improved:
            break
    return seq, {
        "objective": float(current_obj),
        "accepted_updates": int(accepted_updates),
        **{f"objective_{key}": float(value) for key, value in current_terms.items()},
    }


def solve_teacher_short_horizon(
    trial: dict[str, Any],
    *,
    model: Any,
    data: Any,
    actuator_names: list[str],
    qpos_target: np.ndarray,
    qvel_traj: np.ndarray,
    dt: np.ndarray,
    tau_seq: np.ndarray,
    warm_ctrl_seq: np.ndarray,
    static_prior_seq: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Solve a biomech reference control sequence with receding-horizon rollout fitting."""

    n = int(qpos_target.shape[0])
    if n <= 1:
        prev_ctrl = warm_ctrl_seq[0].copy() if warm_ctrl_seq.shape[0] else None
        static_prior = static_prior_seq[0] if static_prior_seq is not None and static_prior_seq.shape[0] else None
        info = evaluate_ctrl_diagnostic(
            model,
            data,
            tau_seq[0] if tau_seq.shape[0] else np.zeros(int(model.nv), dtype=float),
            qpos_target[0] if qpos_target.shape[0] else np.zeros(int(model.nq), dtype=float),
            qvel_traj[0] if qvel_traj.shape[0] else np.zeros(int(model.nv), dtype=float),
            warm_ctrl_seq[0] if warm_ctrl_seq.shape[0] else np.zeros(int(model.nu), dtype=float),
            prev_ctrl=prev_ctrl,
            static_prior_ctrl=static_prior,
            ctrl_lb=float(args.ctrl_lb),
            ctrl_ub=float(args.ctrl_ub),
            solver_mode="short_horizon_rollout:degenerate",
        )
        return warm_ctrl_seq.copy(), [info]

    m_raw, _ = _get_raw_model_data(model, data)
    d_state = mujoco.MjData(m_raw)
    from scripts.build_teacher_inverse_dynamics import set_full_state  # noqa: E402

    set_full_state(m_raw, d_state, qpos_target[0], qvel_traj[0], np.zeros(int(m_raw.nv), dtype=float), zero_act=True)
    ctrl_final = warm_ctrl_seq.copy()
    solver_infos: list[dict[str, Any]] = [dict() for _ in range(n)]
    prev_committed_ctrl = None
    commit_frames = max(1, int(args.commit_frames))
    t = 0
    while t < (n - 1):
        horizon = min(int(args.horizon_size), n - 1 - t)
        if horizon <= 0:
            break
        qpos0 = np.asarray(d_state.qpos, dtype=float).copy()
        qvel0 = np.asarray(d_state.qvel, dtype=float).copy()
        warm_window = np.asarray(warm_ctrl_seq[t : t + horizon], dtype=float).copy()
        if warm_window.shape[0] < horizon:
            break
        if static_prior_seq is None:
            prior_window = warm_window.copy()
        else:
            prior_window = np.asarray(static_prior_seq[t : t + horizon], dtype=float).copy()
            if prior_window.shape[0] < horizon:
                prior_window = warm_window.copy()
        opt_seq, opt_info = optimize_short_horizon_controls(
            model,
            data,
            qpos0=qpos0,
            qvel0=qvel0,
            qpos_target_seq=qpos_target[t + 1 : t + 1 + horizon],
            qvel_target_seq=qvel_traj[t + 1 : t + 1 + horizon],
            dt_seq=dt[t + 1 : t + 1 + horizon],
            tau_target_seq=tau_seq[t : t + horizon],
            warm_ctrl_seq=warm_window,
            prior_ctrl_seq=prior_window,
            actuator_names=actuator_names,
            prev_ctrl_anchor=prev_committed_ctrl,
            args=args,
        )
        qpos_roll, qvel_roll = rollout_control_sequence(
            m_raw,
            qpos0=qpos0,
            qvel0=qvel0,
            ctrl_seq=opt_seq,
            dt_seq=dt[t + 1 : t + 1 + horizon],
            max_substeps=int(args.max_substeps_per_frame),
        )
        n_commit = min(commit_frames, horizon)
        for j in range(n_commit):
            idx = t + j
            ctrl_final[idx] = opt_seq[j]
            static_prior = prior_window[j] if prior_window.shape[0] > j else None
            info = evaluate_ctrl_diagnostic(
                m_raw,
                data,
                tau_seq[idx],
                qpos_roll[j],
                qvel_roll[j],
                ctrl_final[idx],
                prev_ctrl=prev_committed_ctrl if j == 0 else ctrl_final[idx - 1],
                static_prior_ctrl=static_prior,
                ctrl_lb=float(args.ctrl_lb),
                ctrl_ub=float(args.ctrl_ub),
                solver_mode="short_horizon_rollout",
            )
            info.update(
                {
                    "window_start_index": int(t),
                    "window_horizon": int(horizon),
                    "window_objective": float(opt_info.get("objective", 0.0)),
                    "window_accepted_updates": int(opt_info.get("accepted_updates", 0)),
                    "window_pose_term": float(opt_info.get("objective_pose_term", 0.0)),
                    "window_velocity_term": float(opt_info.get("objective_velocity_term", 0.0)),
                    "window_final_pose_term": float(opt_info.get("objective_final_pose_term", 0.0)),
                    "window_u_term": float(opt_info.get("objective_u_term", 0.0)),
                    "window_delta_u_term": float(opt_info.get("objective_delta_u_term", 0.0)),
                    "window_prior_term": float(opt_info.get("objective_prior_term", 0.0)),
                    "warm_start_distance": float(np.linalg.norm(opt_seq[j] - warm_window[j])),
                }
            )
            solver_infos[idx] = info
            prev_committed_ctrl = ctrl_final[idx].copy()
        substeps = build_substeps_from_dt(dt[t + 1 : t + 1 + n_commit], float(m_raw.opt.timestep), int(args.max_substeps_per_frame))
        for j in range(n_commit):
            ctrl = np.asarray(ctrl_final[t + j], dtype=float).reshape(-1)
            d_state.ctrl[:] = 0.0
            d_state.ctrl[: min(int(m_raw.nu), len(ctrl))] = ctrl[: min(int(m_raw.nu), len(ctrl))]
            for _ in range(int(substeps[min(j, len(substeps) - 1)])):
                mujoco.mj_step(m_raw, d_state)
        t += n_commit
    for idx in range(max(0, n - 1), n):
        prev_ctrl = ctrl_final[idx - 1] if idx > 0 else prev_committed_ctrl
        static_prior = static_prior_seq[idx] if static_prior_seq is not None and idx < static_prior_seq.shape[0] else None
        if not solver_infos[idx]:
            solver_infos[idx] = evaluate_ctrl_diagnostic(
                m_raw,
                data,
                tau_seq[idx],
                qpos_target[idx],
                qvel_traj[idx],
                ctrl_final[idx],
                prev_ctrl=prev_ctrl,
                static_prior_ctrl=static_prior,
                ctrl_lb=float(args.ctrl_lb),
                ctrl_ub=float(args.ctrl_ub),
                solver_mode="short_horizon_rollout:tail",
            )
    return ctrl_final, solver_infos


def run_trajectory_teacher_trial(
    trial: dict[str, Any],
    *,
    model: Any,
    data: Any,
    actuator_names: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    qpos_raw = np.asarray(trial["qpos_raw"], dtype=float)
    qpos_target, qvel_traj, qacc_traj, dt = estimate_kinematics(
        qpos_raw,
        trial["timestamps"],
        fallback_dt=float(args.timestep),
        qpos_smoothing_window=int(args.qpos_smoothing_window),
        qvel_smoothing_window=int(args.qvel_smoothing_window),
        qacc_smoothing_window=int(args.qacc_smoothing_window),
        qvel_clip=float(args.qvel_clip),
        qacc_clip=float(args.qacc_clip),
        use_raw_qpos_target=bool(args.use_raw_qpos_target),
        kinematic_filter=str(getattr(args, "kinematic_filter", "moving_average")),
        savgol_polyorder=int(getattr(args, "savgol_polyorder", 3)),
        ignore_timestamps=bool(getattr(args, "ignore_recorded_timestamps", False)),
    )
    times, _ = normalize_timestamps(
        trial["timestamps"],
        qpos_target.shape[0],
        float(args.timestep),
        ignore_timestamps=bool(getattr(args, "ignore_recorded_timestamps", False)),
    )
    m_raw, d_raw = _get_raw_model_data(model, data)
    tau_seq = np.zeros((qpos_target.shape[0], int(m_raw.nv)), dtype=float)
    warm_ctrl_seq = np.zeros((qpos_target.shape[0], int(m_raw.nu)), dtype=float)
    static_prior_seq: np.ndarray | None = None
    static_prior_list: list[np.ndarray] = []
    warm_solver_infos: list[dict[str, Any]] = []
    prev_ctrl = None
    warm_teacher_mode = "dynamic_with_static_prior" if float(args.static_prior_weight) > 0.0 else "dynamic_only"
    for i in range(qpos_target.shape[0]):
        tau_inverse = compute_inverse_dynamics_tau(
            m_raw,
            d_raw,
            qpos_target[i],
            qvel_traj[i],
            qacc_traj[i],
            disable_constraints=bool(args.disable_constraints),
            tau_clip=float(args.tau_clip),
        )
        next_idx = min(i + 1, qpos_target.shape[0] - 1)
        tau_track = compute_tracking_tau_assist(
            qpos_current=qpos_target[i],
            qvel_current=qvel_traj[i],
            qpos_next=qpos_target[next_idx],
            qvel_next=qvel_traj[next_idx],
            nv=int(m_raw.nv),
            kp=float(getattr(args, "tracking_tau_kp", 0.0)),
            kv=float(getattr(args, "tracking_tau_kv", 0.0)),
        )
        tau_target = tau_inverse + tau_track
        tau_target = clip_signal(tau_target, float(args.tau_clip))
        static_prior_ctrl = maybe_build_static_prior(
            warm_teacher_mode,
            str(args.teacher_prior_mode),
            float(args.static_prior_weight),
            m_raw,
            d_raw,
            qpos_target[i],
            prev_ctrl,
            str(trial["exercise_name"]),
            actuator_names,
        )
        ctrl_teacher, info = solve_teacher_ctrl_frame(
            m_raw,
            d_raw,
            tau_target,
            qpos_target[i],
            qvel_traj[i],
            prev_ctrl=prev_ctrl,
            static_prior_ctrl=static_prior_ctrl,
            ctrl_lb=float(args.ctrl_lb),
            ctrl_ub=float(args.ctrl_ub),
            qp_reg=float(args.qp_reg),
            prev_ctrl_weight=float(args.prev_ctrl_weight),
            static_prior_weight=float(args.static_prior_weight),
            prior_value=float(args.prior_value),
            teacher_mode=warm_teacher_mode,
            exercise_name=str(trial["exercise_name"]),
            actuator_names=actuator_names,
        )
        tau_seq[i] = tau_target
        warm_ctrl_seq[i] = np.asarray(ctrl_teacher, dtype=float).reshape(-1)
        warm_solver_infos.append(info)
        if static_prior_ctrl is None:
            static_prior_list.append(np.full(int(m_raw.nu), float(args.prior_value), dtype=float))
        else:
            static_arr = np.asarray(static_prior_ctrl, dtype=float).reshape(-1)
            padded = np.full(int(m_raw.nu), float(args.prior_value), dtype=float)
            padded[: min(int(m_raw.nu), len(static_arr))] = static_arr[: min(int(m_raw.nu), len(static_arr))]
            static_prior_list.append(padded)
        prev_ctrl = warm_ctrl_seq[i].copy()

    if static_prior_list:
        static_prior_seq = np.asarray(static_prior_list, dtype=float)

    if str(args.teacher_mode).strip().lower() == "short_horizon_rollout":
        ctrl_seq, solver_infos = solve_teacher_short_horizon(
            trial,
            model=m_raw,
            data=d_raw,
            actuator_names=actuator_names,
            qpos_target=qpos_target,
            qvel_traj=qvel_traj,
            dt=dt,
            tau_seq=tau_seq,
            warm_ctrl_seq=warm_ctrl_seq,
            static_prior_seq=static_prior_seq,
            args=args,
        )
    else:
        ctrl_seq = warm_ctrl_seq
        solver_infos = warm_solver_infos

    validation = forward_validate_sequence(
        m_raw,
        qpos_target,
        qvel_traj,
        ctrl_seq,
        dt,
        max_substeps=int(args.max_substeps_per_frame),
        validation_window_size=int(args.validation_window_size),
    )
    feedback_history: list[dict[str, Any]] = []
    activation_preemphasis_history: list[dict[str, Any]] = []
    best_ctrl_seq = ctrl_seq
    best_solver_infos = solver_infos
    best_validation = validation
    best_tau_seq = tau_seq
    best_mismatch_objective = compute_weighted_qpos_mismatch_objective(
        qpos_target=qpos_target,
        qvel_target=qvel_traj,
        validation=best_validation,
        ctrl_seq=best_ctrl_seq,
        nv=int(m_raw.nv),
        qpos_weight=float(getattr(args, "mismatch_feedback_qpos_weight", 1.0)),
        qvel_weight=float(getattr(args, "mismatch_feedback_qvel_weight", 0.0)),
        finger_flexion_weight=float(getattr(args, "finger_flexion_feedback_weight", 1.0)),
        distal_flexion_weight=float(getattr(args, "distal_flexion_feedback_weight", 1.0)),
        final_weight=float(getattr(args, "mismatch_feedback_final_weight", 0.5)),
        control_weight=float(getattr(args, "mismatch_feedback_control_weight", 0.0)),
        smoothness_weight=float(getattr(args, "mismatch_feedback_smoothness_weight", 0.0)),
    )
    feedback_alphas = parse_feedback_alphas(str(getattr(args, "mismatch_feedback_alphas", "1.0")))
    for feedback_iter in range(max(0, int(getattr(args, "mismatch_feedback_iters", 0)))):
        forward_qpos_seq = np.asarray(best_validation.get("forward_qpos_seq", []), dtype=float)
        forward_qvel_seq = np.asarray(best_validation.get("forward_qvel_seq", []), dtype=float)
        adaptive_qpos_weights = build_adaptive_finger_feedback_weights(
            qpos_target=qpos_target,
            forward_qpos_seq=forward_qpos_seq,
            nv=int(m_raw.nv),
            enabled=bool(getattr(args, "adaptive_finger_feedback", False)),
            min_weight=float(getattr(args, "adaptive_feedback_min_weight", 0.75)),
            max_weight=float(getattr(args, "adaptive_feedback_max_weight", 1.75)),
        )
        feedback_tau = compute_qpos_mismatch_feedback_tau(
            qpos_target=qpos_target,
            qvel_target=qvel_traj,
            forward_qpos_seq=forward_qpos_seq,
            forward_qvel_seq=forward_qvel_seq,
            nv=int(m_raw.nv),
            kp=float(getattr(args, "mismatch_feedback_kp", 0.0)),
            kv=float(getattr(args, "mismatch_feedback_kv", 0.0)),
            max_tau=float(getattr(args, "mismatch_feedback_tau_clip", 0.0)),
            finger_flexion_weight=float(getattr(args, "finger_flexion_feedback_weight", 1.0)),
            distal_flexion_weight=float(getattr(args, "distal_flexion_feedback_weight", 1.0)),
            adaptive_qpos_weights=adaptive_qpos_weights,
        )
        feedback_norm = float(np.mean(np.linalg.norm(feedback_tau, axis=1))) if feedback_tau.size else 0.0
        if feedback_norm <= 1e-10:
            break
        old_rmse = float(best_validation.get("rmse_forward_error", 0.0))
        best_candidate: dict[str, Any] | None = None
        for alpha in feedback_alphas:
            candidate_tau_seq = best_tau_seq + (float(alpha) * feedback_tau)
            candidate_tau_seq = clip_signal(candidate_tau_seq, float(args.tau_clip))
            candidate_ctrl_seq, candidate_solver_infos = solve_ctrl_sequence_from_tau(
                trial,
                model=m_raw,
                data=d_raw,
                actuator_names=actuator_names,
                qpos_target=qpos_target,
                qvel_traj=qvel_traj,
                dt=dt,
                tau_seq=candidate_tau_seq,
                static_prior_seq=static_prior_seq,
                args=args,
                feedback_iter_index=feedback_iter + 1,
            )
            candidate_validation = forward_validate_sequence(
                m_raw,
                qpos_target,
                qvel_traj,
                candidate_ctrl_seq,
                dt,
                max_substeps=int(args.max_substeps_per_frame),
                validation_window_size=int(args.validation_window_size),
            )
            candidate_objective = compute_weighted_qpos_mismatch_objective(
                qpos_target=qpos_target,
                qvel_target=qvel_traj,
                validation=candidate_validation,
                ctrl_seq=candidate_ctrl_seq,
                nv=int(m_raw.nv),
                qpos_weight=float(getattr(args, "mismatch_feedback_qpos_weight", 1.0)),
                qvel_weight=float(getattr(args, "mismatch_feedback_qvel_weight", 0.0)),
                finger_flexion_weight=float(getattr(args, "finger_flexion_feedback_weight", 1.0)),
                distal_flexion_weight=float(getattr(args, "distal_flexion_feedback_weight", 1.0)),
                final_weight=float(getattr(args, "mismatch_feedback_final_weight", 0.5)),
                control_weight=float(getattr(args, "mismatch_feedback_control_weight", 0.0)),
                smoothness_weight=float(getattr(args, "mismatch_feedback_smoothness_weight", 0.0)),
            )
            if best_candidate is None or float(candidate_objective) < float(best_candidate["objective"]):
                best_candidate = {
                    "alpha": float(alpha),
                    "tau_seq": candidate_tau_seq,
                    "ctrl_seq": candidate_ctrl_seq,
                    "solver_infos": candidate_solver_infos,
                    "validation": candidate_validation,
                    "objective": float(candidate_objective),
                }
        if best_candidate is None:
            break
        candidate_validation = best_candidate["validation"]
        new_rmse = float(candidate_validation.get("rmse_forward_error", 0.0))
        new_mismatch_objective = float(best_candidate["objective"])
        accepted = bool(new_mismatch_objective <= best_mismatch_objective + 1e-12)
        feedback_history.append(
            {
                "iter": int(feedback_iter + 1),
                "alpha": float(best_candidate["alpha"]),
                "feedback_tau_norm_mean": feedback_norm,
                "old_rmse_forward_error": old_rmse,
                "new_rmse_forward_error": new_rmse,
                "old_qpos_mismatch_objective": float(best_mismatch_objective),
                "new_qpos_mismatch_objective": float(new_mismatch_objective),
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        best_ctrl_seq = best_candidate["ctrl_seq"]
        best_solver_infos = best_candidate["solver_infos"]
        best_validation = candidate_validation
        best_tau_seq = best_candidate["tau_seq"]
        best_mismatch_objective = new_mismatch_objective

    activation_gain = max(0.0, float(getattr(args, "activation_preemphasis_gain", 0.0)))
    if activation_gain > 0.0:
        old_rmse = float(best_validation.get("rmse_forward_error", 0.0))
        best_candidate: dict[str, Any] | None = None
        for alpha in parse_feedback_alphas(str(getattr(args, "activation_preemphasis_alphas", "1.0"))):
            candidate_ctrl_seq = apply_activation_preemphasis(
                best_ctrl_seq,
                rise_gain=activation_gain * float(alpha),
                ctrl_lb=float(args.ctrl_lb),
                ctrl_ub=float(args.ctrl_ub),
            )
            candidate_validation = forward_validate_sequence(
                m_raw,
                qpos_target,
                qvel_traj,
                candidate_ctrl_seq,
                dt,
                max_substeps=int(args.max_substeps_per_frame),
                validation_window_size=int(args.validation_window_size),
            )
            candidate_objective = compute_weighted_qpos_mismatch_objective(
                qpos_target=qpos_target,
                qvel_target=qvel_traj,
                validation=candidate_validation,
                ctrl_seq=candidate_ctrl_seq,
                nv=int(m_raw.nv),
                qpos_weight=float(getattr(args, "mismatch_feedback_qpos_weight", 1.0)),
                qvel_weight=float(getattr(args, "mismatch_feedback_qvel_weight", 0.0)),
                finger_flexion_weight=float(getattr(args, "finger_flexion_feedback_weight", 1.0)),
                distal_flexion_weight=float(getattr(args, "distal_flexion_feedback_weight", 1.0)),
                final_weight=float(getattr(args, "mismatch_feedback_final_weight", 0.5)),
                control_weight=float(getattr(args, "mismatch_feedback_control_weight", 0.0)),
                smoothness_weight=float(getattr(args, "mismatch_feedback_smoothness_weight", 0.0)),
            )
            if best_candidate is None or float(candidate_objective) < float(best_candidate["objective"]):
                best_candidate = {
                    "alpha": float(alpha),
                    "ctrl_seq": candidate_ctrl_seq,
                    "validation": candidate_validation,
                    "objective": float(candidate_objective),
                }
        if best_candidate is not None:
            new_mismatch_objective = float(best_candidate["objective"])
            new_rmse = float(best_candidate["validation"].get("rmse_forward_error", 0.0))
            accepted = bool(new_mismatch_objective <= best_mismatch_objective + 1e-12)
            activation_preemphasis_history.append(
                {
                    "alpha": float(best_candidate["alpha"]),
                    "rise_gain": activation_gain * float(best_candidate["alpha"]),
                    "old_rmse_forward_error": old_rmse,
                    "new_rmse_forward_error": new_rmse,
                    "old_qpos_mismatch_objective": float(best_mismatch_objective),
                    "new_qpos_mismatch_objective": float(new_mismatch_objective),
                    "accepted": accepted,
                }
            )
            if accepted:
                best_ctrl_seq = best_candidate["ctrl_seq"]
                best_validation = best_candidate["validation"]
                recomputed_infos: list[dict[str, Any]] = []
                prev_ctrl_for_diag = None
                for diag_idx in range(best_ctrl_seq.shape[0]):
                    static_prior = static_prior_seq[diag_idx] if static_prior_seq is not None and diag_idx < static_prior_seq.shape[0] else None
                    info = evaluate_ctrl_diagnostic(
                        m_raw,
                        data,
                        best_tau_seq[diag_idx],
                        qpos_target[diag_idx],
                        qvel_traj[diag_idx],
                        best_ctrl_seq[diag_idx],
                        prev_ctrl=prev_ctrl_for_diag,
                        static_prior_ctrl=static_prior,
                        ctrl_lb=float(args.ctrl_lb),
                        ctrl_ub=float(args.ctrl_ub),
                        solver_mode="activation_preemphasis",
                    )
                    recomputed_infos.append(info)
                    prev_ctrl_for_diag = best_ctrl_seq[diag_idx].copy()
                best_solver_infos = recomputed_infos
                best_mismatch_objective = new_mismatch_objective

    ctrl_seq = best_ctrl_seq
    solver_infos = best_solver_infos
    validation = best_validation
    tau_seq = best_tau_seq
    keep, quality = summarize_trial_quality(
        trial,
        ctrl_seq,
        solver_infos,
        validation,
        qvel_target=qvel_traj,
        qacc_target=qacc_traj,
        tau_seq=tau_seq,
        max_mean_joint_error=float(args.max_mean_joint_error),
        max_rmse_forward_error=float(args.max_rmse_forward_error),
        max_upper_saturation_fraction=float(args.max_upper_saturation_fraction),
        max_ctrl_delta_mean=float(args.max_ctrl_delta_mean),
    )
    quality.update(
        {
            "teacher_source": "trajectory_inverse_dynamics",
            "teacher_mode_used": str(args.teacher_mode),
            "qfrc_scaler_used": False,
            "trajectory_qvel_source": "finite_difference",
            "trajectory_qacc_source": "finite_difference",
            "kinematic_filter": str(getattr(args, "kinematic_filter", "moving_average")),
            "ignore_recorded_timestamps": bool(getattr(args, "ignore_recorded_timestamps", False)),
            "tracking_tau_kp": float(getattr(args, "tracking_tau_kp", 0.0)),
            "tracking_tau_kv": float(getattr(args, "tracking_tau_kv", 0.0)),
            "mismatch_feedback_iters_requested": int(getattr(args, "mismatch_feedback_iters", 0)),
            "mismatch_feedback_iters_accepted": int(sum(1 for item in feedback_history if bool(item.get("accepted", False)))),
            "mismatch_feedback_kp": float(getattr(args, "mismatch_feedback_kp", 0.0)),
            "mismatch_feedback_kv": float(getattr(args, "mismatch_feedback_kv", 0.0)),
            "mismatch_feedback_qpos_weight": float(getattr(args, "mismatch_feedback_qpos_weight", 1.0)),
            "mismatch_feedback_qvel_weight": float(getattr(args, "mismatch_feedback_qvel_weight", 0.0)),
            "mismatch_feedback_final_weight": float(getattr(args, "mismatch_feedback_final_weight", 0.5)),
            "mismatch_feedback_control_weight": float(getattr(args, "mismatch_feedback_control_weight", 0.0)),
            "mismatch_feedback_smoothness_weight": float(getattr(args, "mismatch_feedback_smoothness_weight", 0.0)),
            "mismatch_feedback_alphas": str(getattr(args, "mismatch_feedback_alphas", "1.0")),
            "adaptive_finger_feedback": bool(getattr(args, "adaptive_finger_feedback", False)),
            "mismatch_feedback_history": json.dumps(feedback_history, ensure_ascii=False),
            "activation_preemphasis_gain": float(getattr(args, "activation_preemphasis_gain", 0.0)),
            "activation_preemphasis_accepted": int(sum(1 for item in activation_preemphasis_history if bool(item.get("accepted", False)))),
            "activation_preemphasis_history": json.dumps(activation_preemphasis_history, ensure_ascii=False),
        }
    )
    reject_reasons = quality.get("reject_reasons", [])
    if isinstance(reject_reasons, (list, tuple)):
        quality["reject_reasons"] = "|".join(str(item) for item in reject_reasons if str(item).strip())
    frame_table = build_trajectory_teacher_frame_table(
        trial,
        times=times,
        dt=dt,
        qpos_raw=qpos_raw,
        qpos_target=qpos_target,
        qvel_traj=qvel_traj,
        qacc_traj=qacc_traj,
        tau_seq=tau_seq,
        ctrl_seq=ctrl_seq,
        solver_infos=solver_infos,
        validation=validation,
        suspicious_frame_error=float(args.suspicious_frame_error),
        suspicious_qacc_norm=float(args.suspicious_qacc_norm),
        suspicious_tau_norm=float(args.suspicious_tau_norm),
    )
    return frame_table, quality, validation


def write_trajectory_report(
    report_path: Path,
    *,
    input_path: Path,
    args: argparse.Namespace,
    summary_df: pd.DataFrame,
    output_dir: Path,
    qfrc_scaler_found: bool,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    passed = summary_df.loc[summary_df["keep_for_training"] == True]  # noqa: E712
    failed = summary_df.loc[summary_df["keep_for_training"] == False]  # noqa: E712
    per_gesture = build_per_gesture_summary(summary_df.to_dict(orient="records"))
    lines = [
        "# Trajectory Inverse Teacher Summary",
        "",
        "This pipeline builds a trajectory-driven offline biomechanical ctrl reference from recorded qpos trajectories.",
        "The outputs are model-derived reference controls for the MyoHand simulator, not biological ground truth.",
        "",
        "## Repository Inspection",
        "",
        f"- Recordings loaded from `{input_path}` using `ctrl_est_*.csv` discovery.",
        "- CSV metadata used: `exercise_name`, `trial_id`, `t`, `frame_index`, and frame-wise `qpos_*`.",
        f"- Environment loader: `create_probe_env()` with env id `{args.env_id}`.",
        "- Current realtime runtime was not modified.",
        f"- `qfrc_scaler` logic found in runtime code: `{qfrc_scaler_found}`.",
        "- This baseline bypasses qfrc scaling and uses raw `qfrc_inverse` from `mj_inverse`.",
        "",
        "## Trajectory Kinematics",
        "",
        "- `qvel_traj` is estimated by temporal finite difference from the filtered recorded trajectory.",
        "- `qacc_traj` is estimated by finite difference from `qvel_traj`.",
        f"- Timestamp policy: `{'fallback_uniform_dt' if bool(getattr(args, 'ignore_recorded_timestamps', False)) else 'recorded_timestamps'}`.",
        f"- Kinematic filter: `{str(getattr(args, 'kinematic_filter', 'moving_average'))}` | Savitzky-Golay polyorder `{int(getattr(args, 'savgol_polyorder', 3))}`.",
        f"- qpos / qvel / qacc smoothing windows: `{int(args.qpos_smoothing_window)}`, `{int(args.qvel_smoothing_window)}`, `{int(args.qacc_smoothing_window)}`.",
        f"- Raw qpos target override enabled: `{bool(args.use_raw_qpos_target)}`.",
        f"- Tracking torque assist: kp=`{float(getattr(args, 'tracking_tau_kp', 0.0)):.4g}`, kv=`{float(getattr(args, 'tracking_tau_kv', 0.0)):.4g}`.",
        f"- Qpos/qvel mismatch feedback: iters=`{int(getattr(args, 'mismatch_feedback_iters', 0))}`, kp=`{float(getattr(args, 'mismatch_feedback_kp', 0.0)):.4g}`, kv=`{float(getattr(args, 'mismatch_feedback_kv', 0.0)):.4g}`, tau_clip=`{float(getattr(args, 'mismatch_feedback_tau_clip', 0.0)):.4g}`.",
        f"- Feedback objective weights: qpos=`{float(getattr(args, 'mismatch_feedback_qpos_weight', 1.0)):.4g}`, qvel=`{float(getattr(args, 'mismatch_feedback_qvel_weight', 0.0)):.4g}`, final=`{float(getattr(args, 'mismatch_feedback_final_weight', 0.5)):.4g}`, u=`{float(getattr(args, 'mismatch_feedback_control_weight', 0.0)):.4g}`, delta_u=`{float(getattr(args, 'mismatch_feedback_smoothness_weight', 0.0)):.4g}`.",
        f"- Finger feedback weights: flexion=`{float(getattr(args, 'finger_flexion_feedback_weight', 1.0)):.4g}`, distal=`{float(getattr(args, 'distal_flexion_feedback_weight', 1.0)):.4g}`, adaptive=`{bool(getattr(args, 'adaptive_finger_feedback', False))}`, alphas=`{str(getattr(args, 'mismatch_feedback_alphas', '1.0'))}`.",
        f"- Activation pre-emphasis: gain=`{float(getattr(args, 'activation_preemphasis_gain', 0.0)):.4g}`, alphas=`{str(getattr(args, 'activation_preemphasis_alphas', '1.0'))}`.",
        "",
        "## Actuator-Space Solve",
        "",
        f"- Reference mode: `{args.teacher_mode}`",
        f"- Short-horizon rollout config: horizon=`{int(args.horizon_size)}`, iterations=`{int(args.horizon_iters)}`, commit=`{int(args.commit_frames)}`",
        f"- Rollout objective weights: qpos=`{float(args.rollout_qpos_weight):.4g}`, qvel=`{float(args.rollout_qvel_weight):.4g}`, final=`{float(getattr(args, 'rollout_final_qpos_weight', 0.0)):.4g}`, u=`{float(args.rollout_u_weight):.4g}`, delta_u=`{float(args.rollout_delta_u_weight):.4g}`, prior=`{float(args.rollout_prior_weight):.4g}`",
        f"- Prior mode: `{args.teacher_prior_mode}`",
        f"- Regularization: qp_reg=`{float(args.qp_reg):.4g}`, prev_ctrl_weight=`{float(args.prev_ctrl_weight):.4g}`, static_prior_weight=`{float(args.static_prior_weight):.4g}`",
        "",
        "## Results",
        "",
        f"- Trials processed: `{len(summary_df)}`",
        f"- Trials accepted: `{len(passed)}`",
        f"- Trials rejected: `{len(failed)}`",
        "",
        "## Per-Gesture Summary",
        "",
    ]
    for gesture, summary in per_gesture.items():
        lines.append(
            f"- `{gesture}`: pass-rate `{summary['pass_rate']:.3f}` ({summary['num_kept']}/{summary['num_trials']}) | "
            f"mean joint error `{summary['mean_joint_error']['mean']:.4f}` | mean RMSE `{summary['rmse_forward_error']['mean']:.4f}` | "
            f"mean ctrl delta `{summary['ctrl_delta_mean']['mean']:.4f}`"
        )
    best = summary_df.sort_values(by=["quality_score"], ascending=False).head(10)
    if not best.empty:
        lines.extend(
            [
                "",
                "## Best Trials",
                "",
                dataframe_to_markdown_simple(best[["trial_id", "exercise_name", "quality_score", "mean_joint_error", "rmse_forward_error"]]),
                "",
            ]
        )
    lines.extend(
        [
            "## Output Files",
            "",
            f"- Summary CSV: `{output_dir / 'teacher_trial_summary.csv'}`",
            f"- Summary JSON: `{output_dir / 'teacher_dataset_summary.json'}`",
            f"- Accepted trial CSVs: `{output_dir / 'accepted_trials'}`",
            f"- Rejected trial CSVs: `{output_dir / 'rejected_trials'}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline biomechanical ctrl_ref dataset from recorded qpos trajectories.")
    parser.add_argument("--input-path", type=str, default="records", help="CSV file or directory containing ctrl_est_*.csv recordings.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for trajectory-inverse biomech reference datasets.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown report path.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id.")
    parser.add_argument("--timestep", type=float, default=1.0 / 60.0, help="Fallback timestep when recording timestamps are missing.")
    parser.add_argument("--qpos-smoothing-window", type=int, default=5, help="Moving-average window for qpos smoothing before differentiation.")
    parser.add_argument("--qvel-smoothing-window", type=int, default=3, help="Optional smoothing window for differentiated qvel.")
    parser.add_argument("--qacc-smoothing-window", type=int, default=3, help="Optional smoothing window for differentiated qacc.")
    parser.add_argument("--kinematic-filter", type=str, default="moving_average", choices=("moving_average", "savgol"), help="Filter used before finite-difference kinematics.")
    parser.add_argument("--savgol-polyorder", type=int, default=3, help="Savitzky-Golay polynomial order when --kinematic-filter savgol is used.")
    parser.add_argument("--ignore-recorded-timestamps", action="store_true", help="Use uniform --timestep instead of CSV timestamps. Useful for legacy recordings with unreliable wall-clock gaps.")
    parser.add_argument("--use-raw-qpos-target", action="store_true", help="Use raw qpos target instead of smoothed qpos target.")
    parser.add_argument("--qvel-clip", type=float, default=0.0, help="Elementwise clip for differentiated qvel.")
    parser.add_argument("--qacc-clip", type=float, default=0.0, help="Elementwise clip for differentiated qacc.")
    parser.add_argument("--tau-clip", type=float, default=0.0, help="Elementwise clip for qfrc_inverse.")
    parser.add_argument("--tracking-tau-kp", type=float, default=0.0, help="Optional posture tracking torque assist toward the next qpos frame.")
    parser.add_argument("--tracking-tau-kv", type=float, default=0.0, help="Optional velocity tracking torque assist toward the next qvel frame.")
    parser.add_argument("--mismatch-feedback-iters", type=int, default=0, help="Iteratively add qpos-forward mismatch feedback into tau target and re-solve controls.")
    parser.add_argument("--mismatch-feedback-kp", type=float, default=0.0, help="Gain converting qpos replay mismatch into tau feedback.")
    parser.add_argument("--mismatch-feedback-kv", type=float, default=0.0, help="Gain converting qvel replay mismatch into tau feedback.")
    parser.add_argument("--mismatch-feedback-tau-clip", type=float, default=0.0, help="Optional elementwise clip for mismatch feedback tau before re-solving.")
    parser.add_argument("--mismatch-feedback-qpos-weight", type=float, default=1.0, help="Objective weight for qpos mismatch during feedback refinement.")
    parser.add_argument("--mismatch-feedback-qvel-weight", type=float, default=0.0, help="Objective weight for qvel mismatch during feedback refinement.")
    parser.add_argument("--mismatch-feedback-final-weight", type=float, default=0.5, help="Extra acceptance weight for final qpos mismatch during feedback refinement.")
    parser.add_argument("--mismatch-feedback-control-weight", type=float, default=0.0, help="Objective weight penalizing large controls during feedback refinement.")
    parser.add_argument("--mismatch-feedback-smoothness-weight", type=float, default=0.0, help="Objective weight penalizing control changes during feedback refinement.")
    parser.add_argument("--mismatch-feedback-alphas", type=str, default="1.0", help="Comma-separated trust-region scales for mismatch feedback tau updates.")
    parser.add_argument("--finger-flexion-feedback-weight", type=float, default=1.0, help="Extra mismatch feedback weight for finger flexion joints.")
    parser.add_argument("--distal-flexion-feedback-weight", type=float, default=1.0, help="Extra mismatch feedback weight for PIP/DIP finger flexion joints.")
    parser.add_argument("--adaptive-finger-feedback", action="store_true", help="Scale finger feedback weights from the previous forward mismatch per finger.")
    parser.add_argument("--adaptive-feedback-min-weight", type=float, default=0.75, help="Minimum adaptive per-finger feedback multiplier.")
    parser.add_argument("--adaptive-feedback-max-weight", type=float, default=1.75, help="Maximum adaptive per-finger feedback multiplier.")
    parser.add_argument("--activation-preemphasis-gain", type=float, default=0.0, help="Optional rising-edge ctrl boost to compensate activation dynamics; accepted only if replay objective improves.")
    parser.add_argument("--activation-preemphasis-alphas", type=str, default="1.0", help="Comma-separated trial scales for activation pre-emphasis.")
    parser.add_argument("--ctrl-lb", type=float, default=0.0, help="Lower bound for reference control.")
    parser.add_argument("--ctrl-ub", type=float, default=1.0, help="Upper bound for reference control.")
    parser.add_argument("--qp-reg", type=float, default=1e-3, help="Regularization for actuator-space solve.")
    parser.add_argument("--prev-ctrl-weight", type=float, default=5e-2, help="Smoothness weight toward previous control.")
    parser.add_argument("--static-prior-weight", type=float, default=1e-1, help="Weight toward a static prior.")
    parser.add_argument("--prior-value", type=float, default=0.0, help="Fallback prior value if no static prior is used.")
    parser.add_argument("--reference-prior-mode", dest="teacher_prior_mode", type=str, default="static_qpos", choices=("zero", "static_qpos"), help="Static prior source for the biomech reference solve.")
    parser.add_argument("--teacher-prior-mode", dest="teacher_prior_mode", type=str, choices=("zero", "static_qpos"), help=argparse.SUPPRESS)
    parser.add_argument("--reference-mode", dest="teacher_mode", type=str, default="dynamic_with_static_prior", choices=("dynamic_only", "dynamic_with_static_prior", "dynamic_with_fallback", "short_horizon_rollout"), help="Actuator solve mode for the biomech reference. `short_horizon_rollout` uses the framewise dynamic solve as a warm start, then refines controls against a short rollout objective.")
    parser.add_argument("--teacher-mode", dest="teacher_mode", type=str, choices=("dynamic_only", "dynamic_with_static_prior", "dynamic_with_fallback", "short_horizon_rollout"), help=argparse.SUPPRESS)
    parser.add_argument("--disable-constraints", action="store_true", help="Disable MuJoCo constraints during inverse dynamics.")
    parser.add_argument("--horizon-size", type=int, default=5, help="Short-horizon rollout length in frames when --reference-mode short_horizon_rollout is used.")
    parser.add_argument("--horizon-iters", type=int, default=5, help="Number of local improvement passes for the short-horizon rollout optimizer.")
    parser.add_argument("--commit-frames", type=int, default=1, help="Number of frames to commit before sliding the short-horizon window forward.")
    parser.add_argument("--rollout-qpos-weight", type=float, default=1.0, help="Weight on rollout pose tracking error in short-horizon mode.")
    parser.add_argument("--rollout-qvel-weight", type=float, default=0.25, help="Weight on rollout velocity tracking error in short-horizon mode.")
    parser.add_argument("--rollout-final-qpos-weight", type=float, default=0.0, help="Extra final-posture qpos weight in short-horizon mode.")
    parser.add_argument("--rollout-u-weight", type=float, default=0.01, help="Weight on control magnitude in short-horizon mode.")
    parser.add_argument("--rollout-delta-u-weight", type=float, default=0.10, help="Weight on control smoothness in short-horizon mode.")
    parser.add_argument("--rollout-prior-weight", type=float, default=0.10, help="Weight on staying close to the warm/static prior sequence in short-horizon mode.")
    parser.add_argument("--max-mean-joint-error", type=float, default=0.20, help="Reject trial if mean joint error exceeds this threshold.")
    parser.add_argument("--max-rmse-forward-error", type=float, default=0.75, help="Reject trial if forward RMSE exceeds this threshold.")
    parser.add_argument("--max-upper-saturation-fraction", type=float, default=0.35, help="Reject trial if controls saturate near the upper bound too often.")
    parser.add_argument("--max-ctrl-delta-mean", type=float, default=0.30, help="Reject trial if control jitter is too large.")
    parser.add_argument("--max-substeps-per-frame", type=int, default=150, help="Maximum MuJoCo steps between recorded frames during replay validation.")
    parser.add_argument("--validation-window-size", type=int, default=0, help="Optional window size for replay diagnostics.")
    parser.add_argument("--suspicious-frame-error", type=float, default=1.0, help="Threshold used to flag suspicious replay frames in per-frame export.")
    parser.add_argument("--suspicious-qacc-norm", type=float, default=1000.0, help="Threshold used to flag suspicious qacc frames in per-frame export.")
    parser.add_argument("--suspicious-tau-norm", type=float, default=1000.0, help="Threshold used to flag suspicious tau frames in per-frame export.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on the number of trials to process.")
    parser.add_argument("--max-frames-per-trial", type=int, default=0, help="Optional cap on the number of frames per trial.")
    parser.add_argument("--gesture-select", type=str, default="", help="Optional comma-separated gesture labels to include.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_path).resolve()
    report_path = Path(args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "accepted_trials").mkdir(parents=True, exist_ok=True)
    (output_dir / "rejected_trials").mkdir(parents=True, exist_ok=True)

    records = discover_recording_csvs(input_path)
    if not records:
        raise SystemExit(f"No ctrl_est_*.csv recordings found under {input_path}")
    allowed = {item.strip().lower() for item in str(args.gesture_select).split(",") if item.strip()}
    if allowed:
        records = [path for path in records if (lambda t: t is not None and str(t["exercise_name"]).strip().lower() in allowed)(load_recorded_trial(path))]
    if int(args.max_trials) > 0:
        records = records[: int(args.max_trials)]

    env = None
    qfrc_scaler_found = repository_has_qfrc_scaler(PROJECT_ROOT)
    trial_rows: list[dict[str, Any]] = []
    try:
        env, model, data = create_probe_env(str(args.env_id))
        m_raw, d_raw = _get_raw_model_data(model, data)
        actuator_names = [mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"muscle_{i}" for i in range(int(m_raw.nu))]
        for csv_path in records:
            trial = load_recorded_trial(csv_path)
            if trial is None:
                continue
            cap = int(args.max_frames_per_trial) if int(args.max_frames_per_trial) > 0 else 0
            if cap > 0:
                trial["qpos_raw"] = np.asarray(trial["qpos_raw"], dtype=float)[:cap]
                trial["frame_indices"] = np.asarray(trial["frame_indices"], dtype=int)[:cap]
                if trial["timestamps"] is not None:
                    trial["timestamps"] = np.asarray(trial["timestamps"], dtype=float)[:cap]
            print(f"[INFO] Offline biomech reference {trial['exercise_name']} / {trial['trial_id']} | frames={np.asarray(trial['qpos_raw']).shape[0]}")
            frame_table, quality, _ = run_trajectory_teacher_trial(
                trial,
                model=m_raw,
                data=d_raw,
                actuator_names=actuator_names,
                args=args,
            )
            keep = bool(quality["keep_for_training"])
            dst_dir = output_dir / ("accepted_trials" if keep else "rejected_trials")
            dst_path = dst_dir / f"{trial['exercise_name']}__{trial['trial_id']}__teacher_from_trajectory_inverse.csv"
            save_trial_dataset(frame_table, dst_path)
            quality["dataset_path"] = str(dst_path)
            trial_rows.append(quality)

        if not trial_rows:
            raise SystemExit("No valid trials were processed.")
        summary_df = pd.DataFrame(trial_rows).sort_values(by=["exercise_name", "trial_id"]).reset_index(drop=True)
        summary_csv = output_dir / "teacher_trial_summary.csv"
        summary_json = output_dir / "teacher_dataset_summary.json"
        summary_df.to_csv(summary_csv, index=False)
        summary_json.write_text(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "env_id": str(args.env_id),
                    "teacher_source": "trajectory_inverse_dynamics",
                    "qfrc_scaler_used": False,
                    "qfrc_scaler_found": qfrc_scaler_found,
                    "num_trials_processed": int(len(summary_df)),
                    "num_trials_kept": int(summary_df["keep_for_training"].sum()),
                    "num_trials_rejected": int((~summary_df["keep_for_training"]).sum()),
                    "per_gesture_summary": build_per_gesture_summary(summary_df.to_dict(orient="records")),
                    "records_used": [str(path) for path in records],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_trajectory_report(
            report_path,
            input_path=input_path,
            args=args,
            summary_df=summary_df,
            output_dir=output_dir,
            qfrc_scaler_found=qfrc_scaler_found,
        )
        print(f"[INFO] Wrote: {summary_csv}")
        print(f"[INFO] Wrote: {summary_json}")
        print(f"[INFO] Wrote: {report_path}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
