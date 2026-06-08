from __future__ import annotations

"""Compare pure inverse-dynamics joint torque targets against the previous qpos->ctrl solver.

This is an offline analysis only. It does not modify the realtime runtime.
The goal is to determine whether the legacy solver already lands close to the
inverse-dynamics target in joint-torque space, even when replay consistency is
still imperfect.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import DEFAULT_MYOSUITE_ENV_ID, _get_raw_model_data, create_probe_env, estimate_ctrl_from_qpos  # noqa: E402
from scripts.build_teacher_inverse_dynamics import (  # noqa: E402
    compute_inverse_dynamics_tau,
    dataframe_to_markdown_simple,
    discover_recording_csvs,
    estimate_kinematics,
    load_recorded_trial,
    normalize_timestamps,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "compare_inverse_vs_previous_solver"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "compare_inverse_vs_previous_solver.md"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_vec = np.asarray(a, dtype=float).reshape(-1)
    b_vec = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom <= 1e-8:
        return 1.0 if float(np.linalg.norm(a_vec - b_vec)) <= 1e-8 else 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def build_trial_frame_table(
    trial: dict[str, Any],
    *,
    times: np.ndarray,
    qpos_target: np.ndarray,
    qvel_target: np.ndarray,
    qacc_target: np.ndarray,
    tau_inverse_seq: np.ndarray,
    tau_prev_seq: np.ndarray,
) -> pd.DataFrame:
    num_frames = int(qpos_target.shape[0])
    tau_error_seq = tau_prev_seq - tau_inverse_seq
    rows: list[dict[str, Any]] = []
    for i in range(num_frames):
        tau_inverse = np.asarray(tau_inverse_seq[i], dtype=float)
        tau_prev = np.asarray(tau_prev_seq[i], dtype=float)
        tau_error = np.asarray(tau_error_seq[i], dtype=float)
        row: dict[str, Any] = {
            "source_csv": str(trial["source_csv"]),
            "exercise_name": str(trial["exercise_name"]),
            "trial_id": str(trial["trial_id"]),
            "frame_index": int(np.asarray(trial["frame_indices"], dtype=int)[i]) if i < len(trial["frame_indices"]) else int(i),
            "time_from_start": float(times[i]) if i < len(times) else float(i),
            "tau_inverse_norm": float(np.linalg.norm(tau_inverse)),
            "tau_prev_norm": float(np.linalg.norm(tau_prev)),
            "tau_error_norm": float(np.linalg.norm(tau_error)),
            "tau_relative_error": float(np.linalg.norm(tau_error) / max(np.linalg.norm(tau_inverse), 1e-8)),
            "tau_cosine_similarity": cosine_similarity(tau_prev, tau_inverse),
        }
        for j, value in enumerate(np.asarray(qpos_target[i], dtype=float).reshape(-1)):
            row[f"qpos_target_{j}"] = float(value)
        for j, value in enumerate(np.asarray(qvel_target[i], dtype=float).reshape(-1)):
            row[f"qvel_target_{j}"] = float(value)
        for j, value in enumerate(np.asarray(qacc_target[i], dtype=float).reshape(-1)):
            row[f"qacc_target_{j}"] = float(value)
        for j, value in enumerate(tau_inverse.reshape(-1)):
            row[f"tau_inverse_{j}"] = float(value)
        for j, value in enumerate(tau_prev.reshape(-1)):
            row[f"tau_prev_{j}"] = float(value)
        for j, value in enumerate(tau_error.reshape(-1)):
            row[f"tau_error_{j}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_trial(
    frame_df: pd.DataFrame,
    *,
    trial: dict[str, Any],
    tau_error_seq: np.ndarray,
) -> dict[str, Any]:
    abs_error = np.abs(np.asarray(tau_error_seq, dtype=float))
    signed_error = np.asarray(tau_error_seq, dtype=float)
    mean_per_joint_abs_error = np.mean(abs_error, axis=0) if abs_error.size else np.zeros(0, dtype=float)
    mean_per_joint_signed_error = np.mean(signed_error, axis=0) if signed_error.size else np.zeros(0, dtype=float)
    row = {
        "source_csv": str(trial["source_csv"]),
        "exercise_name": str(trial["exercise_name"]),
        "trial_id": str(trial["trial_id"]),
        "num_frames": int(len(frame_df)),
        "mean_tau_relative_error": float(frame_df["tau_relative_error"].mean()) if not frame_df.empty else 0.0,
        "median_tau_relative_error": float(frame_df["tau_relative_error"].median()) if not frame_df.empty else 0.0,
        "p95_tau_relative_error": float(frame_df["tau_relative_error"].quantile(0.95)) if not frame_df.empty else 0.0,
        "mean_tau_cosine_similarity": float(frame_df["tau_cosine_similarity"].mean()) if not frame_df.empty else 0.0,
        "worst_frame_tau_error_norm": float(frame_df["tau_error_norm"].max()) if not frame_df.empty else 0.0,
        "mean_tau_inverse_norm": float(frame_df["tau_inverse_norm"].mean()) if not frame_df.empty else 0.0,
        "mean_tau_prev_norm": float(frame_df["tau_prev_norm"].mean()) if not frame_df.empty else 0.0,
        "mean_per_joint_abs_error": json.dumps(mean_per_joint_abs_error.tolist()),
        "mean_per_joint_signed_error": json.dumps(mean_per_joint_signed_error.tolist()),
    }
    return row


def build_per_gesture_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gesture, sub in sorted(trial_df.groupby("exercise_name"), key=lambda item: str(item[0])):
        best_idx = sub["mean_tau_relative_error"].idxmin()
        worst_idx = sub["mean_tau_relative_error"].idxmax()
        best_trial = sub.loc[best_idx]
        worst_trial = sub.loc[worst_idx]
        rows.append(
            {
                "exercise_name": str(gesture),
                "num_trials": int(len(sub)),
                "mean_tau_relative_error": float(sub["mean_tau_relative_error"].mean()),
                "median_tau_relative_error": float(sub["mean_tau_relative_error"].median()),
                "p95_tau_relative_error": float(sub["mean_tau_relative_error"].quantile(0.95)),
                "mean_tau_cosine_similarity": float(sub["mean_tau_cosine_similarity"].mean()),
                "best_trial_id": str(best_trial["trial_id"]),
                "best_trial_mean_tau_relative_error": float(best_trial["mean_tau_relative_error"]),
                "best_trial_mean_tau_cosine_similarity": float(best_trial["mean_tau_cosine_similarity"]),
                "worst_trial_id": str(worst_trial["trial_id"]),
                "worst_trial_mean_tau_relative_error": float(worst_trial["mean_tau_relative_error"]),
                "worst_trial_mean_tau_cosine_similarity": float(worst_trial["mean_tau_cosine_similarity"]),
            }
        )
    return pd.DataFrame(rows)


def interpret_results(trial_df: pd.DataFrame, gesture_df: pd.DataFrame) -> tuple[str, list[str]]:
    mean_rel = float(trial_df["mean_tau_relative_error"].mean()) if not trial_df.empty else float("nan")
    median_rel = float(trial_df["median_tau_relative_error"].median()) if not trial_df.empty else float("nan")
    mean_cos = float(trial_df["mean_tau_cosine_similarity"].mean()) if not trial_df.empty else float("nan")
    good_fraction = float(
        (
            (trial_df["mean_tau_cosine_similarity"] >= 0.7)
            & (trial_df["median_tau_relative_error"] <= 0.5)
        ).mean()
    ) if not trial_df.empty else float("nan")
    gesture_lines: list[str] = []
    for _, row in gesture_df.iterrows():
        gesture_lines.append(
            f"- `{row['exercise_name']}`: mean tau_relative_error `{row['mean_tau_relative_error']:.4f}`, "
            f"median `{row['median_tau_relative_error']:.4f}`, mean cosine `{row['mean_tau_cosine_similarity']:.4f}`"
        )
    if np.isfinite(median_rel) and median_rel <= 0.5 and np.isfinite(mean_cos) and mean_cos >= 0.8 and np.isfinite(good_fraction) and good_fraction >= 0.5:
        headline = (
            "The previous solver appears torque-space close to the inverse-dynamics target on average. "
            "This supports prioritizing direct ctrl prediction as the next stage, while keeping inverse-dynamics as a reference check."
        )
    elif np.isfinite(median_rel) and median_rel <= 1.0 and np.isfinite(mean_cos) and mean_cos >= 0.6 and np.isfinite(good_fraction) and good_fraction >= 0.2:
        headline = (
            "The previous solver is partially close to the inverse-dynamics target, but the similarity is not uniformly strong. "
            "Direct ctrl prediction is promising, yet inverse-dynamics still provides useful additional structure."
        )
    else:
        headline = (
            "The previous solver is not consistently close to the inverse-dynamics target in torque space. "
            "Inverse-dynamics remains an informative teacher/reference path rather than a redundant duplication of the existing solver."
        )
    return headline, gesture_lines


def write_report(
    report_path: Path,
    *,
    input_path: Path,
    args: argparse.Namespace,
    trial_df: pd.DataFrame,
    gesture_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    headline, gesture_lines = interpret_results(trial_df, gesture_df)
    best_trials = trial_df.sort_values(by=["mean_tau_relative_error", "mean_tau_cosine_similarity"], ascending=[True, False]).head(10)
    worst_trials = trial_df.sort_values(by=["mean_tau_relative_error", "mean_tau_cosine_similarity"], ascending=[False, True]).head(10)
    lines = [
        "# Inverse Dynamics vs Previous Solver",
        "",
        "This report compares the pure inverse-dynamics torque target against the previous qpos->ctrl solver in joint-torque space.",
        "The inverse-dynamics target is a model-based teacher reference, not biological ground truth.",
        "",
        "## Repository / Data Inputs",
        "",
        f"- Recordings loaded from `{input_path}` using `ctrl_est_*.csv` discovery.",
        "- Frame-wise `qpos_*`, `exercise_name`, `trial_id`, `t`, and `frame_index` were used.",
        f"- Environment loader: `create_probe_env()` with env id `{args.env_id}`.",
        "- Kinematic preprocessing matches the offline teacher pipeline: qpos smoothing, finite-difference qvel, finite-difference qacc.",
        "",
        "## Comparison Method",
        "",
        "- `tau_inverse` is computed from `(qpos_target, qvel_target, qacc_target)` using `mj_inverse`.",
        "- The previous solver is run through `estimate_ctrl_from_qpos(...)` on the same target trajectory.",
        "- `tau_prev` is taken from the previous solver's reconstructed torque output `tau_reconstructed`, i.e. the torque-space quantity that the legacy solver itself is trying to match.",
        "- `tau_relative_error` can become numerically large when `||tau_inverse||` is very small, so the interpretation also relies on cosine similarity and median relative error rather than the raw mean alone.",
        "",
        "## Main Answer",
        "",
        headline,
        "",
        "## Per-Gesture Summary",
        "",
        *gesture_lines,
        "",
        "## Best Trials",
        "",
        dataframe_to_markdown_simple(
            best_trials[
                [
                    "trial_id",
                    "exercise_name",
                    "mean_tau_relative_error",
                    "median_tau_relative_error",
                    "mean_tau_cosine_similarity",
                    "worst_frame_tau_error_norm",
                ]
            ]
        ),
        "",
        "## Worst Trials",
        "",
        dataframe_to_markdown_simple(
            worst_trials[
                [
                    "trial_id",
                    "exercise_name",
                    "mean_tau_relative_error",
                    "median_tau_relative_error",
                    "mean_tau_cosine_similarity",
                    "worst_frame_tau_error_norm",
                ]
            ]
        ),
        "",
        "## Output Files",
        "",
        f"- Per-trial frame CSVs: `{output_dir / 'trial_frames'}`",
        f"- Trial summary CSV: `{output_dir / 'trial_summary.csv'}`",
        f"- Gesture summary CSV: `{output_dir / 'gesture_summary.csv'}`",
        f"- Dataset summary JSON: `{output_dir / 'dataset_summary.json'}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare inverse-dynamics joint torque targets against the previous qpos->ctrl solver.")
    parser.add_argument("--input-path", type=str, default="records", help="CSV file or directory containing ctrl_est_*.csv recordings.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for comparison outputs.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown report path.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite environment id.")
    parser.add_argument("--timestep", type=float, default=1.0 / 60.0, help="Fallback timestep when CSV timestamps are missing.")
    parser.add_argument("--qpos-smoothing-window", type=int, default=5, help="qpos smoothing window used before differentiation.")
    parser.add_argument("--qvel-smoothing-window", type=int, default=3, help="qvel smoothing window used after differentiation.")
    parser.add_argument("--qacc-smoothing-window", type=int, default=3, help="qacc smoothing window used after differentiating qvel.")
    parser.add_argument("--use-raw-qpos-target", action="store_true", help="Use raw qpos target instead of smoothed qpos target.")
    parser.add_argument("--qvel-clip", type=float, default=0.0, help="Elementwise clip for qvel target.")
    parser.add_argument("--qacc-clip", type=float, default=0.0, help="Elementwise clip for qacc target.")
    parser.add_argument("--tau-clip", type=float, default=0.0, help="Elementwise clip for tau_inverse.")
    parser.add_argument("--disable-constraints", action="store_true", help="Disable MuJoCo constraints during inverse-dynamics target generation.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on number of trials.")
    parser.add_argument("--max-frames-per-trial", type=int, default=0, help="Optional cap on frames per trial.")
    parser.add_argument("--gesture-select", type=str, default="", help="Optional comma-separated gesture labels to include.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_path).resolve()
    report_path = Path(args.report_path).resolve()
    trial_frames_dir = output_dir / "trial_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_frames_dir.mkdir(parents=True, exist_ok=True)

    records = discover_recording_csvs(input_path)
    if not records:
        raise SystemExit(f"No ctrl_est_*.csv recordings found under {input_path}")
    allowed = {item.strip().lower() for item in str(args.gesture_select).split(",") if item.strip()}
    if allowed:
        filtered: list[Path] = []
        for path in records:
            trial = load_recorded_trial(path)
            if trial is None:
                continue
            if str(trial["exercise_name"]).strip().lower() in allowed:
                filtered.append(path)
        records = filtered
    if int(args.max_trials) > 0:
        records = records[: int(args.max_trials)]

    env = None
    trial_rows: list[dict[str, Any]] = []
    frame_csv_paths: list[str] = []
    try:
        env, model, data = create_probe_env(str(args.env_id))
        m_raw, d_raw = _get_raw_model_data(model, data)
        actuator_names = [mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"muscle_{i}" for i in range(int(m_raw.nu))]

        for csv_path in records:
            trial = load_recorded_trial(csv_path)
            if trial is None:
                continue
            qpos_raw = np.asarray(trial["qpos_raw"], dtype=float)
            cap = int(args.max_frames_per_trial) if int(args.max_frames_per_trial) > 0 else 0
            if cap > 0:
                qpos_raw = qpos_raw[:cap]
                trial["frame_indices"] = np.asarray(trial["frame_indices"], dtype=int)[:cap]
                if trial["timestamps"] is not None:
                    trial["timestamps"] = np.asarray(trial["timestamps"], dtype=float)[:cap]
            qpos_target, qvel_target, qacc_target, _dt = estimate_kinematics(
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
            tau_inverse_seq = np.zeros((qpos_target.shape[0], int(m_raw.nv)), dtype=float)
            tau_prev_seq = np.zeros((qpos_target.shape[0], int(m_raw.nv)), dtype=float)
            prev_ctrl = None
            for i in range(qpos_target.shape[0]):
                tau_inverse = compute_inverse_dynamics_tau(
                    m_raw,
                    d_raw,
                    qpos_target[i],
                    qvel_target[i],
                    qacc_target[i],
                    disable_constraints=bool(args.disable_constraints),
                    tau_clip=float(args.tau_clip),
                )
                ctrl_prev, info = estimate_ctrl_from_qpos(
                    m_raw,
                    d_raw,
                    qpos_target[i],
                    prev_ctrl=prev_ctrl,
                    exercise_name=str(trial["exercise_name"]),
                    actuator_names=actuator_names,
                )
                tau_prev = np.asarray(info.get("tau_reconstructed"), dtype=float).reshape(-1)
                tau_inverse_seq[i] = tau_inverse
                tau_prev_seq[i] = tau_prev[: int(m_raw.nv)]
                prev_ctrl = np.asarray(ctrl_prev, dtype=float).reshape(-1)

            frame_df = build_trial_frame_table(
                trial,
                times=times,
                qpos_target=qpos_target,
                qvel_target=qvel_target,
                qacc_target=qacc_target,
                tau_inverse_seq=tau_inverse_seq,
                tau_prev_seq=tau_prev_seq,
            )
            frame_csv_path = trial_frames_dir / f"{trial['exercise_name']}__{trial['trial_id']}__inverse_vs_previous_solver.csv"
            frame_df.to_csv(frame_csv_path, index=False)
            frame_csv_paths.append(str(frame_csv_path))
            tau_error_seq = tau_prev_seq - tau_inverse_seq
            row = summarize_trial(frame_df, trial=trial, tau_error_seq=tau_error_seq)
            row["frame_csv_path"] = str(frame_csv_path)
            trial_rows.append(row)
            print(f"[INFO] Compared torque {trial['exercise_name']} / {trial['trial_id']} | frames={qpos_target.shape[0]}")

        if not trial_rows:
            raise SystemExit("No valid trials were processed.")

        trial_df = pd.DataFrame(trial_rows).sort_values(by=["exercise_name", "trial_id"]).reset_index(drop=True)
        gesture_df = build_per_gesture_summary(trial_df)
        trial_csv = output_dir / "trial_summary.csv"
        gesture_csv = output_dir / "gesture_summary.csv"
        summary_json = output_dir / "dataset_summary.json"
        trial_df.to_csv(trial_csv, index=False)
        gesture_df.to_csv(gesture_csv, index=False)
        headline, _gesture_lines = interpret_results(trial_df, gesture_df)
        summary_json.write_text(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "env_id": str(args.env_id),
                    "num_trials_processed": int(len(trial_df)),
                    "records_used": [str(path) for path in records],
                    "frame_csv_paths": frame_csv_paths,
                    "overall_mean_tau_relative_error": float(trial_df["mean_tau_relative_error"].mean()),
                    "overall_median_tau_relative_error": float(trial_df["mean_tau_relative_error"].median()),
                    "overall_mean_tau_cosine_similarity": float(trial_df["mean_tau_cosine_similarity"].mean()),
                    "interpretation": headline,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_report(
            report_path,
            input_path=input_path,
            args=args,
            trial_df=trial_df,
            gesture_df=gesture_df,
            output_dir=output_dir,
        )
        print(f"[INFO] Wrote: {trial_csv}")
        print(f"[INFO] Wrote: {gesture_csv}")
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
