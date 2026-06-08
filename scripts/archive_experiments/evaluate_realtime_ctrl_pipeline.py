from __future__ import annotations

"""Evaluate legacy solver vs fast predictor offline on recorded trajectories."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.fast_ctrl_predictor import load_fast_ctrl_predictor_runtime  # noqa: E402
from app.main import DEFAULT_MYOSUITE_ENV_ID, _get_raw_model_data, create_probe_env, estimate_ctrl_from_qpos  # noqa: E402
from scripts.build_teacher_inverse_dynamics import build_per_gesture_summary, dataframe_to_markdown_simple, discover_recording_csvs, estimate_kinematics, forward_validate_sequence, load_recorded_trial  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evaluate_realtime_ctrl_pipeline"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "evaluate_realtime_ctrl_pipeline.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate legacy realtime ctrl estimation against the fast predictor offline.")
    parser.add_argument("--input-path", type=str, default="records", help="Recording CSV or records directory.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown report path.")
    parser.add_argument("--fast-predictor-checkpoint", type=str, required=True, help="Checkpoint produced by train_fast_ctrl_predictor.py.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id.")
    parser.add_argument("--gesture-filter", type=str, default="", help="Optional comma-separated gesture filter.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on number of trials.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on frames per trial.")
    parser.add_argument("--timestep", type=float, default=1.0 / 60.0, help="Fallback timestep.")
    parser.add_argument("--qpos-smoothing-window", type=int, default=5, help="qpos smoothing window.")
    parser.add_argument("--qvel-smoothing-window", type=int, default=3, help="qvel smoothing window.")
    parser.add_argument("--qacc-smoothing-window", type=int, default=1, help="qacc smoothing window.")
    parser.add_argument("--qvel-clip", type=float, default=0.0, help="Optional qvel clip.")
    parser.add_argument("--use-raw-qpos-target", action="store_true", help="Use raw qpos target.")
    parser.add_argument("--max-substeps-per-frame", type=int, default=150, help="Maximum MuJoCo steps between frames.")
    parser.add_argument("--validation-window-size", type=int, default=0, help="Optional validation chunk size.")
    parser.add_argument("--predictor-ema-alpha", type=float, default=1.0, help="Optional EMA smoothing for fast predictor output. 1.0 disables EMA.")
    parser.add_argument("--predictor-max-step", type=float, default=0.0, help="Optional per-frame max ctrl step for fast predictor output. 0 disables the constraint.")
    return parser.parse_args()


def gesture_filter_set(raw_value: str) -> set[str]:
    return {item.strip().lower() for item in str(raw_value).split(",") if item.strip()}


def compute_ctrl_stats(ctrl_seq: np.ndarray, timings_ms: list[float]) -> dict[str, float]:
    ctrl = np.asarray(ctrl_seq, dtype=float)
    delta = np.diff(ctrl, axis=0) if ctrl.shape[0] > 1 else np.zeros((0, ctrl.shape[1] if ctrl.ndim == 2 else 0), dtype=float)
    timings = np.asarray(timings_ms, dtype=float)
    return {
        "ctrl_delta_mean": float(np.mean(np.linalg.norm(delta, axis=1))) if delta.size else 0.0,
        "ctrl_norm_mean": float(np.mean(np.linalg.norm(ctrl, axis=1))) if ctrl.ndim == 2 and ctrl.shape[0] else 0.0,
        "upper_saturation_fraction": float(np.mean(ctrl >= 0.98)) if ctrl.size else 0.0,
        "lower_saturation_fraction": float(np.mean(ctrl <= 0.02)) if ctrl.size else 0.0,
        "latency_mean_ms": float(np.mean(timings)) if timings.size else 0.0,
        "latency_p95_ms": float(np.percentile(timings, 95.0)) if timings.size else 0.0,
    }


def postprocess_predictor_ctrl(
    ctrl_pred: np.ndarray,
    *,
    prev_output: np.ndarray | None,
    ema_alpha: float,
    max_step: float,
) -> np.ndarray:
    ctrl_out = np.asarray(ctrl_pred, dtype=float).reshape(-1)
    if prev_output is not None and prev_output.shape == ctrl_out.shape:
        alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        if alpha < 1.0:
            ctrl_out = alpha * ctrl_out + (1.0 - alpha) * np.asarray(prev_output, dtype=float)
        step_limit = float(max_step)
        if step_limit > 0.0:
            delta = np.clip(ctrl_out - np.asarray(prev_output, dtype=float), -step_limit, step_limit)
            ctrl_out = np.asarray(prev_output, dtype=float) + delta
    return np.clip(ctrl_out, 0.0, 1.0)


def run_legacy_trial(
    trial: dict[str, Any],
    *,
    model: Any,
    data: Any,
    actuator_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    qpos_raw = np.asarray(trial["qpos_raw"], dtype=float)
    qpos_target, qvel_target, _qacc_target, dt = estimate_kinematics(
        qpos_raw,
        trial["timestamps"],
        fallback_dt=float(args.timestep),
        qpos_smoothing_window=int(args.qpos_smoothing_window),
        qvel_smoothing_window=int(args.qvel_smoothing_window),
        qacc_smoothing_window=int(args.qacc_smoothing_window),
        qvel_clip=float(args.qvel_clip),
        qacc_clip=0.0,
        use_raw_qpos_target=bool(args.use_raw_qpos_target),
    )
    m_raw, d_raw = _get_raw_model_data(model, data)
    ctrl_seq = np.zeros((qpos_target.shape[0], int(m_raw.nu)), dtype=float)
    prev_ctrl = None
    timings_ms: list[float] = []
    for i in range(qpos_target.shape[0]):
        t0 = time.perf_counter()
        ctrl, _info = estimate_ctrl_from_qpos(
            m_raw,
            d_raw,
            qpos_target[i],
            prev_ctrl=prev_ctrl,
            exercise_name=str(trial["exercise_name"]),
            actuator_names=actuator_names,
        )
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
        ctrl_seq[i] = np.asarray(ctrl, dtype=float)
        prev_ctrl = ctrl_seq[i].copy()
    validation = forward_validate_sequence(
        m_raw,
        qpos_target,
        qvel_target,
        ctrl_seq,
        dt,
        max_substeps=int(args.max_substeps_per_frame),
        validation_window_size=int(args.validation_window_size),
    )
    stats = compute_ctrl_stats(ctrl_seq, timings_ms)
    return {
        "trial_id": str(trial["trial_id"]),
        "exercise_name": str(trial["exercise_name"]),
        "source_csv": str(trial["source_csv"]),
        "mode": "legacy_solver",
        "mean_joint_error": float(validation["mean_joint_error"]),
        "rmse_forward_error": float(validation["rmse_forward_error"]),
        "median_frame_error": float(validation.get("median_frame_error", 0.0)),
        "p95_frame_error": float(validation.get("p95_frame_error", 0.0)),
        "relative_residual_mean": 0.0,
        "keep_for_training": True,
        **stats,
    }


def run_predictor_trial(
    trial: dict[str, Any],
    *,
    model: Any,
    data: Any,
    predictor_checkpoint: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    qpos_raw = np.asarray(trial["qpos_raw"], dtype=float)
    qpos_target, qvel_target, _qacc_target, dt = estimate_kinematics(
        qpos_raw,
        trial["timestamps"],
        fallback_dt=float(args.timestep),
        qpos_smoothing_window=int(args.qpos_smoothing_window),
        qvel_smoothing_window=int(args.qvel_smoothing_window),
        qacc_smoothing_window=int(args.qacc_smoothing_window),
        qvel_clip=float(args.qvel_clip),
        qacc_clip=0.0,
        use_raw_qpos_target=bool(args.use_raw_qpos_target),
    )
    predictor = load_fast_ctrl_predictor_runtime(predictor_checkpoint, device="cpu")
    predictor.reset()
    m_raw, _ = _get_raw_model_data(model, data)
    ctrl_seq = np.zeros((qpos_target.shape[0], int(m_raw.nu)), dtype=float)
    prev_ctrl = np.zeros(int(m_raw.nu), dtype=float)
    prev_output = np.zeros(int(m_raw.nu), dtype=float)
    timings_ms: list[float] = []
    for i in range(qpos_target.shape[0]):
        t0 = time.perf_counter()
        ctrl = predictor.predict(
            qpos=qpos_target[i],
            qvel=qvel_target[i],
            prev_ctrl=prev_ctrl,
            gesture_label=str(trial["exercise_name"]),
        )
        ctrl = postprocess_predictor_ctrl(
            ctrl,
            prev_output=prev_output,
            ema_alpha=float(args.predictor_ema_alpha),
            max_step=float(args.predictor_max_step),
        )
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
        ctrl_seq[i] = np.asarray(ctrl, dtype=float)
        prev_ctrl = ctrl_seq[i].copy()
        prev_output = ctrl_seq[i].copy()
    validation = forward_validate_sequence(
        m_raw,
        qpos_target,
        qvel_target,
        ctrl_seq,
        dt,
        max_substeps=int(args.max_substeps_per_frame),
        validation_window_size=int(args.validation_window_size),
    )
    stats = compute_ctrl_stats(ctrl_seq, timings_ms)
    return {
        "trial_id": str(trial["trial_id"]),
        "exercise_name": str(trial["exercise_name"]),
        "source_csv": str(trial["source_csv"]),
        "mode": "fast_predictor",
        "mean_joint_error": float(validation["mean_joint_error"]),
        "rmse_forward_error": float(validation["rmse_forward_error"]),
        "median_frame_error": float(validation.get("median_frame_error", 0.0)),
        "p95_frame_error": float(validation.get("p95_frame_error", 0.0)),
        "relative_residual_mean": 0.0,
        "keep_for_training": True,
        "predictor_ema_alpha": float(args.predictor_ema_alpha),
        "predictor_max_step": float(args.predictor_max_step),
        **stats,
    }


def write_report(report_path: Path, *, merged_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    better_rmse = int((merged_df["rmse_forward_error_fast_predictor"] < merged_df["rmse_forward_error_legacy_solver"]).sum()) if not merged_df.empty else 0
    better_latency = int((merged_df["latency_mean_ms_fast_predictor"] < merged_df["latency_mean_ms_legacy_solver"]).sum()) if not merged_df.empty else 0
    mode_notes: list[str] = []
    if not merged_df.empty and "predictor_ema_alpha_fast_predictor" in merged_df.columns:
        mode_notes.append(f"- Fast predictor EMA alpha: `{merged_df['predictor_ema_alpha_fast_predictor'].iloc[0]}`")
    if not merged_df.empty and "predictor_max_step_fast_predictor" in merged_df.columns:
        mode_notes.append(f"- Fast predictor max step: `{merged_df['predictor_max_step_fast_predictor'].iloc[0]}`")
    lines = [
        "# Ctrl-First Realtime Pipeline Evaluation",
        "",
        "This evaluation is ctrl-first: the main question is whether predicted ctrl rolls out correctly against the target qpos trajectory.",
        "Torque-space metrics are intentionally not primary here.",
        "",
        f"- Trials compared: `{len(merged_df)}`",
        f"- Fast predictor better RMSE: `{better_rmse}`",
        f"- Fast predictor faster mean latency: `{better_latency}`",
        "",
    ]
    lines.extend(mode_notes)
    if mode_notes:
        lines.append("")
    lines.extend(
        [
            "## Primary Metrics",
            "",
            "- `rmse_forward_error`",
            "- `mean_joint_error`",
            "- `p95_frame_error`",
            "- `ctrl_delta_mean`",
            "- `latency_mean_ms / latency_p95_ms`",
            "",
        ]
    )
    if not summary_df.empty:
        lines.extend(
            [
                "## Per-Gesture Summary",
                "",
                dataframe_to_markdown_simple(summary_df),
                "",
            ]
        )
    lines.extend(
        [
            "## Output Files",
            "",
            f"- Trial summary: `{output_dir / 'trial_summary.csv'}`",
            f"- Comparison table: `{output_dir / 'comparison.csv'}`",
            f"- Gesture summary: `{output_dir / 'gesture_summary.csv'}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor_checkpoint = Path(args.fast_predictor_checkpoint).resolve()
    selected_gestures = gesture_filter_set(args.gesture_filter)
    paths = discover_recording_csvs(input_path)
    if int(args.max_trials) > 0:
        paths = paths[: int(args.max_trials)]
    env, model, data = create_probe_env(args.env_id)
    _ = env
    m_raw, _ = _get_raw_model_data(model, data)
    actuator_names = [
        str(mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) or f"actuator_{idx}")
        for idx in range(int(m_raw.nu))
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        trial = load_recorded_trial(path)
        if trial is None:
            continue
        if selected_gestures and str(trial["exercise_name"]).strip().lower() not in selected_gestures:
            continue
        if int(args.max_frames) > 0:
            trial = dict(trial)
            trial["qpos_raw"] = np.asarray(trial["qpos_raw"], dtype=float)[: int(args.max_frames)]
            if trial.get("timestamps") is not None:
                trial["timestamps"] = np.asarray(trial["timestamps"], dtype=float)[: int(args.max_frames)]
        rows.append(run_legacy_trial(trial, model=model, data=data, actuator_names=actuator_names, args=args))
        rows.append(run_predictor_trial(trial, model=model, data=data, predictor_checkpoint=predictor_checkpoint, args=args))
    trial_df = pd.DataFrame(rows)
    legacy_df = trial_df.loc[trial_df["mode"] == "legacy_solver"].copy()
    predictor_df = trial_df.loc[trial_df["mode"] == "fast_predictor"].copy()
    merged = legacy_df.merge(
        predictor_df,
        on=["trial_id", "exercise_name", "source_csv"],
        suffixes=("_legacy_solver", "_fast_predictor"),
    )
    gesture_rows: list[dict[str, Any]] = []
    for gesture, group in merged.groupby("exercise_name"):
        gesture_rows.append(
            {
                "exercise_name": str(gesture),
                "legacy_rmse_mean": float(group["rmse_forward_error_legacy_solver"].mean()),
                "fast_rmse_mean": float(group["rmse_forward_error_fast_predictor"].mean()),
                "legacy_latency_mean_ms": float(group["latency_mean_ms_legacy_solver"].mean()),
                "fast_latency_mean_ms": float(group["latency_mean_ms_fast_predictor"].mean()),
            }
        )
    gesture_df = pd.DataFrame(gesture_rows)
    trial_df.to_csv(output_dir / "trial_summary.csv", index=False)
    merged.to_csv(output_dir / "comparison.csv", index=False)
    gesture_df.to_csv(output_dir / "gesture_summary.csv", index=False)
    dataset_summary = {
        "num_trials_compared": int(len(merged)),
        "per_mode": build_per_gesture_summary(trial_df.to_dict(orient="records")),
    }
    (output_dir / "summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    write_report(Path(args.report_path).resolve(), merged_df=merged, summary_df=gesture_df, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
