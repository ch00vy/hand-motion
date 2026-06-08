from __future__ import annotations

"""Phase 2: compare the offline teacher baseline against the current solver.

The comparison is offline only and does not modify the current realtime path.
Teacher labels are treated as model-based teacher references, not biological
ground truth.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import DEFAULT_MYOSUITE_ENV_ID, _get_raw_model_data, create_probe_env, estimate_ctrl_from_qpos  # noqa: E402
from scripts.build_teacher_from_qpos import repository_has_qfrc_scaler  # noqa: E402
from scripts.build_teacher_inverse_dynamics import (  # noqa: E402
    build_per_gesture_summary,
    dataframe_to_markdown_simple,
    discover_recording_csvs,
    estimate_kinematics,
    forward_validate_sequence,
    load_recorded_trial,
    normalize_timestamps,
)


DEFAULT_COMPARE_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "teacher_vs_current_solver"
DEFAULT_COMPARE_REPORT_PATH = PROJECT_ROOT / "reports" / "teacher_solver_phase1_phase2_summary.md"


def compute_ctrl_metrics(ctrl_seq: np.ndarray) -> dict[str, float]:
    ctrl = np.asarray(ctrl_seq, dtype=float)
    if ctrl.ndim != 2 or ctrl.shape[0] == 0:
        return {
            "ctrl_delta_mean": 0.0,
            "ctrl_norm_mean": 0.0,
            "upper_saturation_fraction": 0.0,
            "lower_saturation_fraction": 0.0,
        }
    delta = np.diff(ctrl, axis=0) if ctrl.shape[0] > 1 else np.zeros((0, ctrl.shape[1]), dtype=float)
    return {
        "ctrl_delta_mean": float(np.mean(np.linalg.norm(delta, axis=1))) if delta.size else 0.0,
        "ctrl_norm_mean": float(np.mean(np.linalg.norm(ctrl, axis=1))),
        "upper_saturation_fraction": float(np.mean(ctrl >= 0.98)),
        "lower_saturation_fraction": float(np.mean(ctrl <= 0.02)),
    }


def run_current_solver_trial(
    trial: dict[str, Any],
    *,
    model: Any,
    data: Any,
    actuator_names: list[str],
    args: argparse.Namespace,
    max_frames: int = 0,
) -> dict[str, Any]:
    qpos_raw = np.asarray(trial["qpos_raw"], dtype=float)
    if int(max_frames) > 0:
        qpos_raw = qpos_raw[: int(max_frames)]
        if trial.get("timestamps") is not None:
            timestamps = np.asarray(trial["timestamps"], dtype=float)[: int(max_frames)]
        else:
            timestamps = None
    else:
        timestamps = trial["timestamps"]
    qpos_target, qvel_target, _qacc_unused, dt = estimate_kinematics(
        qpos_raw,
        timestamps,
        fallback_dt=float(args.timestep),
        qpos_smoothing_window=int(args.qpos_smoothing_window),
        qvel_smoothing_window=int(args.qvel_smoothing_window),
        qacc_smoothing_window=int(args.qacc_smoothing_window),
        qvel_clip=float(args.qvel_clip),
        qacc_clip=0.0,
        use_raw_qpos_target=bool(args.use_raw_qpos_target),
    )
    m_raw, _ = _get_raw_model_data(model, data)
    ctrl_seq = np.zeros((qpos_target.shape[0], int(m_raw.nu)), dtype=float)
    prev_ctrl = None
    relative_residuals: list[float] = []
    for i in range(qpos_target.shape[0]):
        ctrl, _info = estimate_ctrl_from_qpos(
            m_raw,
            data,
            qpos_target[i],
            prev_ctrl=prev_ctrl,
            exercise_name=str(trial["exercise_name"]),
            actuator_names=actuator_names,
        )
        ctrl_seq[i] = np.asarray(ctrl, dtype=float).reshape(-1)
        prev_ctrl = ctrl_seq[i].copy()
        try:
            relative_residuals.append(float(_info.get("relative_residual", 0.0)))
        except Exception:
            relative_residuals.append(0.0)
    validation = forward_validate_sequence(
        m_raw,
        qpos_target,
        qvel_target,
        ctrl_seq,
        dt,
        max_substeps=int(args.max_substeps_per_frame),
        validation_window_size=int(args.validation_window_size),
    )
    metrics = compute_ctrl_metrics(ctrl_seq)
    return {
        "trial_id": str(trial["trial_id"]),
        "exercise_name": str(trial["exercise_name"]),
        "source_csv": str(trial["source_csv"]),
        "mean_joint_error": float(validation["mean_joint_error"]),
        "rmse_forward_error": float(validation["rmse_forward_error"]),
        "mean_pose_error_norm": float(validation["mean_pose_error_norm"]),
        "median_frame_error": float(validation.get("median_frame_error", 0.0)),
        "p95_frame_error": float(validation.get("p95_frame_error", 0.0)),
        "max_frame_error": float(validation.get("max_frame_error", 0.0)),
        "per_joint_rmse": list(validation.get("per_joint_rmse", [])),
        **metrics,
        "relative_residual_mean": float(np.mean(relative_residuals)) if relative_residuals else 0.0,
        "fallback_fraction": 0.0,
        "quality_score": float(
            1.0
            / (
                1.0
                + float(validation["rmse_forward_error"])
                + float(validation["mean_joint_error"])
                + float(metrics["ctrl_delta_mean"])
            )
        ),
        "keep_for_training": True,
        "reject_reasons": "",
        "solver_name": "current_solver",
    }


def join_teacher_and_current(teacher_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    merged = teacher_df.merge(
        current_df,
        on=["trial_id", "exercise_name", "source_csv"],
        how="inner",
        suffixes=("_teacher", "_current"),
    )
    if merged.empty:
        return merged
    merged["mean_joint_error_delta"] = merged["mean_joint_error_current"] - merged["mean_joint_error_teacher"]
    merged["rmse_forward_error_delta"] = merged["rmse_forward_error_current"] - merged["rmse_forward_error_teacher"]
    merged["ctrl_delta_mean_delta"] = merged["ctrl_delta_mean_current"] - merged["ctrl_delta_mean_teacher"]
    merged["upper_saturation_fraction_delta"] = merged["upper_saturation_fraction_current"] - merged["upper_saturation_fraction_teacher"]
    return merged


def write_phase2_report(
    report_path: Path,
    *,
    args: argparse.Namespace,
    teacher_df: pd.DataFrame,
    current_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    qfrc_scaler_found = repository_has_qfrc_scaler(PROJECT_ROOT)
    teacher_summary = build_per_gesture_summary(teacher_df.to_dict(orient="records"))
    current_summary = build_per_gesture_summary(current_df.to_dict(orient="records"))
    better_teacher_rmse = int((merged_df["rmse_forward_error_teacher"] < merged_df["rmse_forward_error_current"]).sum()) if not merged_df.empty else 0
    better_teacher_joint = int((merged_df["mean_joint_error_teacher"] < merged_df["mean_joint_error_current"]).sum()) if not merged_df.empty else 0
    lines = [
        "# Teacher Solver Phase 1 + Phase 2 Summary",
        "",
        "This comparison evaluates a model-based offline teacher baseline against the current qpos-to-ctrl solver.",
        "The teacher labels remain model-derived references, not biological ground truth.",
        "",
        "## Phase 1 Teacher Baseline",
        "",
        f"- Teacher summary loaded from `{Path(args.teacher_output_path).resolve()}`.",
        f"- Teacher formulation: `qacc_des = (qpos_target - qpos_current) * {float(args.kp):.3f} - qvel_current * {float(args.kv):.3f}`.",
        f"- `qfrc_scaler` found in repository: `{qfrc_scaler_found}`.",
        "- The baseline comparison bypasses qfrc scaling and uses raw inverse-dynamics generalized force output.",
        "",
        "## Phase 2 Comparison",
        "",
        f"- Trials compared: `{len(merged_df)}`",
        f"- Teacher better in mean joint error: `{better_teacher_joint}` trials",
        f"- Teacher better in RMSE forward error: `{better_teacher_rmse}` trials",
        "",
        "## Per-Gesture Teacher Summary",
        "",
    ]
    for gesture, summary in teacher_summary.items():
        lines.append(
            f"- `{gesture}` teacher: mean joint `{summary['mean_joint_error']['mean']:.4f}` | "
            f"mean RMSE `{summary['rmse_forward_error']['mean']:.4f}` | "
            f"mean ctrl delta `{summary['ctrl_delta_mean']['mean']:.4f}`"
        )
    lines.extend(["", "## Per-Gesture Current Solver Summary", ""])
    for gesture, summary in current_summary.items():
        lines.append(
            f"- `{gesture}` current: mean joint `{summary['mean_joint_error']['mean']:.4f}` | "
            f"mean RMSE `{summary['rmse_forward_error']['mean']:.4f}` | "
            f"mean ctrl delta `{summary['ctrl_delta_mean']['mean']:.4f}`"
        )
    if not merged_df.empty:
        best = merged_df.sort_values(by=["rmse_forward_error_delta"], ascending=False).head(10)
        lines.extend(["", "## Trials Where Teacher Most Improved RMSE", "", dataframe_to_markdown_simple(best[[
            "trial_id",
            "exercise_name",
            "rmse_forward_error_teacher",
            "rmse_forward_error_current",
            "rmse_forward_error_delta",
            "mean_joint_error_teacher",
            "mean_joint_error_current",
        ]]), ""])
    lines.extend(
        [
            "## Output Files",
            "",
            f"- Teacher metrics used: `{Path(args.teacher_output_path).resolve() / 'teacher_trial_summary.csv'}`",
            f"- Current solver metrics: `{output_dir / 'current_solver_trial_summary.csv'}`",
            f"- Merged comparison table: `{output_dir / 'teacher_vs_current_comparison.csv'}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 comparison between teacher baseline and current solver.")
    parser.add_argument("--input-path", type=str, default="records", help="CSV file or directory containing ctrl_est_*.csv recordings.")
    parser.add_argument("--teacher-output-path", type=str, required=True, help="Output directory from build_teacher_from_qpos.py.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_COMPARE_OUTPUT_DIR), help="Directory for comparison outputs.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_COMPARE_REPORT_PATH), help="Markdown report path.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id.")
    parser.add_argument("--kp", type=float, default=1000.0, help="Teacher kp used for documentation.")
    parser.add_argument("--kv", type=float, default=100.0, help="Teacher kv used for documentation.")
    parser.add_argument("--timestep", type=float, default=1.0 / 60.0, help="Fallback timestep when CSV timestamps are missing.")
    parser.add_argument("--qpos-smoothing-window", type=int, default=5, help="qpos smoothing window used for target generation.")
    parser.add_argument("--qvel-smoothing-window", type=int, default=3, help="qvel smoothing window used for target generation.")
    parser.add_argument("--qacc-smoothing-window", type=int, default=1, help="qacc smoothing window used for target generation.")
    parser.add_argument("--use-raw-qpos-target", action="store_true", help="Use raw qpos target instead of smoothed qpos target.")
    parser.add_argument("--qvel-clip", type=float, default=0.0, help="Elementwise clip for qvel target.")
    parser.add_argument("--max-substeps-per-frame", type=int, default=150, help="Maximum MuJoCo steps between recorded frames.")
    parser.add_argument("--validation-window-size", type=int, default=0, help="Optional validation window size.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on number of trials to compare.")
    parser.add_argument("--gesture-select", type=str, default="", help="Optional comma-separated gesture labels to include.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    teacher_output_path = Path(args.teacher_output_path).resolve()
    output_dir = Path(args.output_path).resolve()
    report_path = Path(args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_summary_csv = teacher_output_path / "teacher_trial_summary.csv"
    if not teacher_summary_csv.exists():
        raise SystemExit(f"Teacher summary not found: {teacher_summary_csv}")
    teacher_df = pd.read_csv(teacher_summary_csv)
    records = discover_recording_csvs(input_path)
    if not records:
        raise SystemExit(f"No ctrl_est_*.csv recordings found under {input_path}")
    allowed = {item.strip().lower() for item in str(args.gesture_select).split(",") if item.strip()}
    trial_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for path in records:
        trial = load_recorded_trial(path)
        if trial is None:
            continue
        gesture = str(trial["exercise_name"]).strip().lower()
        if allowed and gesture not in allowed:
            continue
        trial_by_key[(str(trial["exercise_name"]), str(trial["trial_id"]))] = trial

    teacher_rows = []
    for row in teacher_df.to_dict(orient="records"):
        key = (str(row["exercise_name"]), str(row["trial_id"]))
        if key in trial_by_key:
            teacher_rows.append(row)
    teacher_df = pd.DataFrame(teacher_rows)
    if teacher_df.empty:
        raise SystemExit("No overlapping trials found between teacher output and input recordings.")
    if int(args.max_trials) > 0:
        teacher_df = teacher_df.head(int(args.max_trials)).copy()

    env = None
    current_rows: list[dict[str, Any]] = []
    try:
        env, model, data = create_probe_env(str(args.env_id))
        m_raw, _ = _get_raw_model_data(model, data)
        actuator_names = [mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"muscle_{i}" for i in range(int(m_raw.nu))]
        for row in teacher_df.to_dict(orient="records"):
            trial = trial_by_key[(str(row["exercise_name"]), str(row["trial_id"]))]
            compare_frames = int(row.get("num_frames", 0)) if int(row.get("num_frames", 0)) > 0 else int(np.asarray(trial["qpos_raw"]).shape[0])
            print(f"[INFO] Compare current solver {trial['exercise_name']} / {trial['trial_id']} | frames={compare_frames}")
            current_rows.append(
                run_current_solver_trial(
                    trial,
                    model=m_raw,
                    data=data,
                    actuator_names=actuator_names,
                    args=args,
                    max_frames=int(row.get("num_frames", 0)),
                )
            )
        current_df = pd.DataFrame(current_rows).sort_values(by=["exercise_name", "trial_id"]).reset_index(drop=True)
        merged_df = join_teacher_and_current(teacher_df, current_df)
        current_csv = output_dir / "current_solver_trial_summary.csv"
        merged_csv = output_dir / "teacher_vs_current_comparison.csv"
        current_df.to_csv(current_csv, index=False)
        merged_df.to_csv(merged_csv, index=False)
        write_phase2_report(
            report_path,
            args=args,
            teacher_df=teacher_df,
            current_df=current_df,
            merged_df=merged_df,
            output_dir=output_dir,
        )
        print(f"[INFO] Wrote: {current_csv}")
        print(f"[INFO] Wrote: {merged_csv}")
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
