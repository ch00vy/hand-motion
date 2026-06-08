from __future__ import annotations

"""Offline 3D musculoskeletal replay viewer for teacher-control trajectories.

This tool replays saved `ctrl_teacher` sequences on the MyoHand MuJoCo model
for visual inspection. It does not modify the realtime runtime path and should
be used only as an offline, model-derived visualization/debugging utility.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import DEFAULT_MYOSUITE_ENV_ID, _get_raw_model_data, create_probe_env  # noqa: E402
from scripts.build_teacher_inverse_dynamics import (  # noqa: E402
    build_substeps_from_dt,
    dataframe_to_markdown_simple,
    distribution_stats,
    normalize_timestamps,
    set_full_state,
)
from scripts.archive_experiments.replay_teacher_controls import discover_teacher_csvs, load_teacher_trial  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "teacher_replay_videos"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "teacher_musculoskeletal_replay_summary.md"


@dataclass
class ReplayState:
    paused: bool = False
    step_once: bool = False
    restart_requested: bool = False
    playback_speed: float = 1.0


def build_vis_option() -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    try:
        option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        option.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
    except Exception:
        pass
    return option


def clamp_render_size(model: Any, width: int, height: int) -> tuple[int, int]:
    try:
        offw = int(getattr(model.vis.global_, "offwidth", width))
        offh = int(getattr(model.vis.global_, "offheight", height))
    except Exception:
        return int(width), int(height)
    return max(1, min(int(width), offw)), max(1, min(int(height), offh))


def compute_trial_times_and_steps(trial: dict[str, Any], fallback_dt: float, max_substeps: int, sim_dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_target = np.asarray(trial["qpos_target"], dtype=float)
    times = None if trial["time_from_start"] is None else np.asarray(trial["time_from_start"], dtype=float)
    norm_times, dt = normalize_timestamps(times, qpos_target.shape[0], fallback_dt)
    if trial["frame_dt"] is not None and len(trial["frame_dt"]) == qpos_target.shape[0]:
        dt_candidate = np.asarray(trial["frame_dt"], dtype=float)
        dt = np.where(np.isfinite(dt_candidate) & (dt_candidate > 1e-6), dt_candidate, dt)
    substeps = build_substeps_from_dt(dt, float(sim_dt), int(max_substeps))
    return norm_times, dt, substeps


def prepare_replay_state(model: Any, data: Any, trial: dict[str, Any], ctrl_seq: np.ndarray) -> None:
    m_raw, d_raw = _get_raw_model_data(model, data)
    qpos_target = np.asarray(trial["qpos_target"], dtype=float)
    qvel_init_all = np.asarray(trial["qvel_init"], dtype=float) if trial["qvel_init"] is not None else None
    if qvel_init_all is not None and qvel_init_all.shape[0] > 0:
        qvel_init = qvel_init_all[0]
    else:
        qvel_init = np.zeros(int(m_raw.nv), dtype=float)
    set_full_state(
        m_raw,
        d_raw,
        qpos_target[0],
        qvel_init,
        np.zeros(int(m_raw.nv), dtype=float),
        ctrl=ctrl_seq[0] if ctrl_seq.shape[0] > 0 else None,
        zero_act=True,
    )


def step_forward_one_frame(model: Any, data: Any, ctrl: np.ndarray, num_substeps: int) -> None:
    m_raw, d_raw = _get_raw_model_data(model, data)
    ctrl_arr = np.asarray(ctrl, dtype=float).reshape(-1)
    for _ in range(int(num_substeps)):
        d_raw.ctrl[:] = 0.0
        d_raw.ctrl[: min(int(m_raw.nu), ctrl_arr.shape[0])] = ctrl_arr[: min(int(m_raw.nu), ctrl_arr.shape[0])]
        mujoco.mj_step(m_raw, d_raw)


def render_rgb(renderer: mujoco.Renderer, model: Any, data: Any, option: mujoco.MjvOption) -> np.ndarray:
    renderer.update_scene(data, scene_option=option)
    frame = renderer.render()
    return np.asarray(frame, dtype=np.uint8).copy()


def add_overlay(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    x = 12
    y = 24
    for idx, text in enumerate(lines):
        y_line = y + idx * 22
        cv2.putText(frame_bgr, text, (x, y_line), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (x, y_line), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (17, 24, 39), 1, cv2.LINE_AA)
    return frame_bgr


def compute_joint_error(forward_qpos: np.ndarray, target_qpos: np.ndarray) -> tuple[np.ndarray, float]:
    fq = np.asarray(forward_qpos, dtype=float).reshape(-1)
    tq = np.asarray(target_qpos, dtype=float).reshape(-1)
    nq = min(fq.shape[0], tq.shape[0])
    joint_err = fq[:nq] - tq[:nq]
    return joint_err, float(np.linalg.norm(joint_err))


def build_overlay_lines(trial: dict[str, Any], frame_idx: int, frame_error: float, mean_error_so_far: float, playback_speed: float, label: str) -> list[str]:
    return [
        f"{label}",
        f"trial: {trial['trial_id']} | gesture: {trial['exercise_name']}",
        f"frame: {frame_idx + 1}/{trial['num_frames']} | speed: {playback_speed:.2f}x",
        f"frame error: {frame_error:.4f}",
        f"mean error so far: {mean_error_so_far:.4f}",
    ]


def export_video_for_trial(
    trial: dict[str, Any],
    *,
    model: Any,
    data: Any,
    output_dir: Path,
    fps: float,
    max_substeps: int,
    playback_speed: float,
    side_by_side: bool,
    max_frames: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    m_raw, d_raw = _get_raw_model_data(model, data)
    option = build_vis_option()
    ctrl_seq = np.asarray(trial["ctrl_teacher"], dtype=float)
    qpos_target = np.asarray(trial["qpos_target"], dtype=float)
    if int(max_frames) > 0:
        ctrl_seq = ctrl_seq[: int(max_frames)]
        qpos_target = qpos_target[: int(max_frames)]
        trial = dict(trial)
        trial["num_frames"] = int(min(int(trial["num_frames"]), int(max_frames)))
        if trial["time_from_start"] is not None:
            trial["time_from_start"] = np.asarray(trial["time_from_start"], dtype=float)[: int(max_frames)]
        if trial["frame_dt"] is not None:
            trial["frame_dt"] = np.asarray(trial["frame_dt"], dtype=float)[: int(max_frames)]
        trial["frame_indices"] = np.asarray(trial["frame_indices"], dtype=int)[: int(max_frames)]
    times, dt, substeps = compute_trial_times_and_steps(trial, 1.0 / max(float(fps), 1e-6), max_substeps, float(m_raw.opt.timestep))

    forward_data = mujoco.MjData(m_raw)
    target_data = mujoco.MjData(m_raw)
    prepare_replay_state(m_raw, forward_data, trial, ctrl_seq)
    prepare_replay_state(m_raw, target_data, trial, ctrl_seq)

    render_width, render_height = clamp_render_size(m_raw, int(width), int(height))
    renderer = mujoco.Renderer(m_raw, height=int(render_height), width=int(render_width))
    out_trial_dir = output_dir / f"{trial['exercise_name']}__{trial['trial_id']}"
    out_trial_dir.mkdir(parents=True, exist_ok=True)
    forward_path = out_trial_dir / "teacher_forward_replay.mp4"
    target_path = out_trial_dir / "target_qpos_replay.mp4"
    combined_path = out_trial_dir / "side_by_side_replay.mp4"
    table_path = out_trial_dir / "replay_summary.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_forward = cv2.VideoWriter(str(forward_path), fourcc, float(fps), (int(render_width), int(render_height)))
    writer_target = cv2.VideoWriter(str(target_path), fourcc, float(fps), (int(render_width), int(render_height)))
    writer_combined = None
    if bool(side_by_side):
        writer_combined = cv2.VideoWriter(str(combined_path), fourcc, float(fps), (int(render_width) * 2, int(render_height)))

    rows: list[dict[str, Any]] = []
    frame_error_history: list[float] = []
    try:
        for i in range(qpos_target.shape[0]):
            target_q = np.asarray(qpos_target[i], dtype=float)
            set_full_state(m_raw, target_data, target_q, np.zeros(int(m_raw.nv), dtype=float), np.zeros(int(m_raw.nv), dtype=float), zero_act=True)
            forward_q = np.asarray(forward_data.qpos[: int(m_raw.nq)], dtype=float).copy()
            joint_err, frame_error = compute_joint_error(forward_q, target_q)
            frame_error_history.append(frame_error)
            mean_error_so_far = float(np.mean(frame_error_history))

            rgb_target = render_rgb(renderer, m_raw, target_data, option)
            rgb_forward = render_rgb(renderer, m_raw, forward_data, option)
            bgr_target = add_overlay(rgb_target, build_overlay_lines(trial, i, frame_error, mean_error_so_far, playback_speed, "Target trajectory"))
            bgr_forward = add_overlay(rgb_forward, build_overlay_lines(trial, i, frame_error, mean_error_so_far, playback_speed, "Forward replay from teacher ctrl"))

            writer_target.write(bgr_target)
            writer_forward.write(bgr_forward)
            if writer_combined is not None:
                combined = np.concatenate([bgr_target, bgr_forward], axis=1)
                writer_combined.write(combined)

            row: dict[str, Any] = {
                "trial_id": str(trial["trial_id"]),
                "exercise_name": str(trial["exercise_name"]),
                "frame_index": int(trial["frame_indices"][i]) if i < len(trial["frame_indices"]) else int(i),
                "time_from_start": float(times[i]) if i < len(times) else float(i),
                "frame_error_norm": frame_error,
                "mean_error_so_far": mean_error_so_far,
            }
            for j in range(min(target_q.shape[0], forward_q.shape[0])):
                row[f"qpos_target_{j}"] = float(target_q[j])
                row[f"qpos_forward_{j}"] = float(forward_q[j])
                row[f"joint_error_{j}"] = float(joint_err[j])
            rows.append(row)

            if i < qpos_target.shape[0] - 1:
                step_forward_one_frame(m_raw, forward_data, ctrl_seq[i], int(substeps[min(i + 1, len(substeps) - 1)]))
    finally:
        writer_forward.release()
        writer_target.release()
        if writer_combined is not None:
            writer_combined.release()

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(table_path, index=False)
    per_joint_cols = [col for col in summary_df.columns if col.startswith("joint_error_")]
    per_joint_rmse = {}
    for col in per_joint_cols:
        per_joint_rmse[col.replace("joint_error_", "qpos_")] = float(math.sqrt(np.mean(np.square(summary_df[col].to_numpy(dtype=float))))) if len(summary_df) else 0.0
    largest_joint = max(per_joint_rmse.items(), key=lambda item: item[1])[0] if per_joint_rmse else "qpos_0"
    largest_joint_rmse = per_joint_rmse.get(largest_joint, 0.0)
    return {
        "trial_id": str(trial["trial_id"]),
        "exercise_name": str(trial["exercise_name"]),
        "num_frames": int(qpos_target.shape[0]),
        "mean_joint_error": float(np.mean(list(per_joint_rmse.values()))) if per_joint_rmse else 0.0,
        "rmse_forward_error": float(math.sqrt(np.mean(np.square(summary_df["frame_error_norm"].to_numpy(dtype=float))))) if len(summary_df) else 0.0,
        "mean_pose_error_norm": float(summary_df["frame_error_norm"].mean()) if len(summary_df) else 0.0,
        "largest_error_joint": largest_joint,
        "largest_error_joint_rmse": largest_joint_rmse,
        "render_width": int(render_width),
        "render_height": int(render_height),
        "forward_video_path": str(forward_path),
        "target_video_path": str(target_path),
        "combined_video_path": str(combined_path) if writer_combined is not None else "",
        "replay_table_path": str(table_path),
    }


def run_interactive_viewer_for_trial(
    trial: dict[str, Any],
    *,
    model: Any,
    data: Any,
    fps: float,
    max_substeps: int,
    playback_speed: float,
    max_frames: int,
) -> None:
    import mujoco.viewer

    m_raw, d_raw = _get_raw_model_data(model, data)
    ctrl_seq = np.asarray(trial["ctrl_teacher"], dtype=float)
    qpos_target = np.asarray(trial["qpos_target"], dtype=float)
    if int(max_frames) > 0:
        ctrl_seq = ctrl_seq[: int(max_frames)]
        qpos_target = qpos_target[: int(max_frames)]
        trial = dict(trial)
        trial["num_frames"] = int(min(int(trial["num_frames"]), int(max_frames)))
        if trial["time_from_start"] is not None:
            trial["time_from_start"] = np.asarray(trial["time_from_start"], dtype=float)[: int(max_frames)]
        if trial["frame_dt"] is not None:
            trial["frame_dt"] = np.asarray(trial["frame_dt"], dtype=float)[: int(max_frames)]
        trial["frame_indices"] = np.asarray(trial["frame_indices"], dtype=int)[: int(max_frames)]
    _times, dt, substeps = compute_trial_times_and_steps(trial, 1.0 / max(float(fps), 1e-6), max_substeps, float(m_raw.opt.timestep))
    state = ReplayState(playback_speed=float(playback_speed))

    def on_key(keycode: int) -> None:
        if keycode == 32:  # space
            state.paused = not state.paused
        elif keycode in (ord("N"), ord("n")):
            state.step_once = True
            state.paused = True
        elif keycode in (ord("R"), ord("r")):
            state.restart_requested = True
        elif keycode == ord("]"):
            state.playback_speed = min(4.0, state.playback_speed * 1.25)
        elif keycode == ord("["):
            state.playback_speed = max(0.1, state.playback_speed / 1.25)

    prepare_replay_state(m_raw, d_raw, trial, ctrl_seq)
    frame_idx = 0
    frame_errors: list[float] = []
    option = build_vis_option()
    with mujoco.viewer.launch_passive(m_raw, d_raw, key_callback=on_key, show_left_ui=True, show_right_ui=True) as viewer:
        try:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = option.flags[mujoco.mjtVisFlag.mjVIS_TENDON]
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR]
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = option.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION]
        except Exception:
            pass

        last_time = time.perf_counter()
        while viewer.is_running():
            if state.restart_requested:
                prepare_replay_state(m_raw, d_raw, trial, ctrl_seq)
                frame_idx = 0
                frame_errors.clear()
                state.restart_requested = False

            target_q = np.asarray(qpos_target[min(frame_idx, qpos_target.shape[0] - 1)], dtype=float)
            forward_q = np.asarray(d_raw.qpos[: int(m_raw.nq)], dtype=float).copy()
            _joint_err, frame_error = compute_joint_error(forward_q, target_q)
            frame_errors.append(frame_error)
            mean_error = float(np.mean(frame_errors)) if frame_errors else 0.0
            viewer.sync()

            if frame_idx >= qpos_target.shape[0] - 1:
                time.sleep(0.01)
                continue
            if state.paused and not state.step_once:
                time.sleep(0.01)
                continue

            wait_seconds = max(0.0, float(dt[min(frame_idx + 1, len(dt) - 1)]) / max(state.playback_speed, 1e-6))
            elapsed = time.perf_counter() - last_time
            if elapsed < wait_seconds:
                time.sleep(wait_seconds - elapsed)
            last_time = time.perf_counter()

            step_forward_one_frame(m_raw, d_raw, ctrl_seq[frame_idx], int(substeps[min(frame_idx + 1, len(substeps) - 1)]))
            frame_idx += 1
            state.step_once = False
            if frame_idx % 10 == 0:
                print(
                    f"[VIEW] {trial['exercise_name']} / {trial['trial_id']} "
                    f"frame={frame_idx + 1}/{qpos_target.shape[0]} "
                    f"frame_error={frame_error:.4f} mean_error={mean_error:.4f} speed={state.playback_speed:.2f}x"
                )


def write_summary_report(report_path: Path, summaries: list[dict[str, Any]], output_dir: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Teacher Musculoskeletal Replay Summary",
        "",
        "This report summarizes offline 3D replays of saved teacher-control trajectories on the MyoHand MuJoCo model.",
        "The replay is a visualization and consistency check for model-derived teacher references, not clinical truth.",
        "",
    ]
    if not summaries:
        lines.append("No replay outputs were produced.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    df = pd.DataFrame(summaries)
    lines.extend(
        [
            f"- Trials processed: `{len(df)}`",
            f"- Mean joint error (avg): `{float(df['mean_joint_error'].mean()):.4f}`",
            f"- RMSE forward error (avg): `{float(df['rmse_forward_error'].mean()):.4f}`",
            "",
            "## Aggregate Metrics",
            "",
            dataframe_to_markdown_simple(
                pd.DataFrame(
                    [
                        {"metric": "mean_joint_error", **distribution_stats(df["mean_joint_error"].astype(float).tolist())},
                        {"metric": "rmse_forward_error", **distribution_stats(df["rmse_forward_error"].astype(float).tolist())},
                        {"metric": "mean_pose_error_norm", **distribution_stats(df["mean_pose_error_norm"].astype(float).tolist())},
                    ]
                )
            ),
            "",
            "## Per-Trial Replay Summary",
            "",
            dataframe_to_markdown_simple(
                df[
                    [
                        "trial_id",
                        "exercise_name",
                        "mean_joint_error",
                        "rmse_forward_error",
                        "largest_error_joint",
                        "largest_error_joint_rmse",
                        "forward_video_path",
                        "target_video_path",
                    ]
                ]
            ),
            "",
            "## Notes",
            "",
            "- Side-by-side visualization is exported as a combined MP4 when `--side-by-side` is enabled.",
            "- Interactive controls: `Space` pause/resume, `N` single-step, `R` restart, `[` slower, `]` faster.",
            "",
            f"- Output directory: `{output_dir}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View or export offline 3D musculoskeletal replays for teacher-control trials.")
    parser.add_argument("--input-path", type=str, required=True, help="Teacher trial CSV file or directory containing *teacher_from_qpos.csv files.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for replay videos and summaries.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown replay summary path.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback and export FPS.")
    parser.add_argument("--substeps", type=int, default=150, help="Maximum MuJoCo substeps per recorded frame.")
    parser.add_argument("--mode", type=str, default="video", choices=("interactive", "video", "both"), help="Replay mode.")
    parser.add_argument("--side-by-side", action="store_true", help="Export a combined target/forward side-by-side video in video mode.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Interactive playback speed multiplier.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on frames per trial.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on number of teacher trial CSVs.")
    parser.add_argument("--width", type=int, default=960, help="Render width for exported frames.")
    parser.add_argument("--height", type=int, default=720, help="Render height for exported frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_csvs = discover_teacher_csvs(input_path)
    if not teacher_csvs:
        raise SystemExit(f"No teacher trial CSVs found under {input_path}")
    if int(args.max_trials) > 0:
        teacher_csvs = teacher_csvs[: int(args.max_trials)]

    env = None
    summaries: list[dict[str, Any]] = []
    try:
        env, model, data = create_probe_env(str(args.env_id))
        m_raw, d_raw = _get_raw_model_data(model, data)
        for csv_path in teacher_csvs:
            trial = load_teacher_trial(csv_path)
            if trial is None:
                continue
            if str(args.mode) in ("video", "both"):
                print(f"[INFO] Export replay video {trial['exercise_name']} / {trial['trial_id']} | frames={trial['num_frames']}")
                summaries.append(
                    export_video_for_trial(
                        trial,
                        model=m_raw,
                        data=d_raw,
                        output_dir=output_dir,
                        fps=float(args.fps),
                        max_substeps=int(args.substeps),
                        playback_speed=float(args.playback_speed),
                        side_by_side=bool(args.side_by_side),
                        max_frames=int(args.max_frames),
                        width=int(args.width),
                        height=int(args.height),
                    )
                )
            if str(args.mode) in ("interactive", "both"):
                print(f"[INFO] Launch interactive replay {trial['exercise_name']} / {trial['trial_id']} | frames={trial['num_frames']}")
                run_interactive_viewer_for_trial(
                    trial,
                    model=m_raw,
                    data=d_raw,
                    fps=float(args.fps),
                    max_substeps=int(args.substeps),
                    playback_speed=float(args.playback_speed),
                    max_frames=int(args.max_frames),
                )

        summary_df = pd.DataFrame(summaries).sort_values(by=["exercise_name", "trial_id"]).reset_index(drop=True) if summaries else pd.DataFrame()
        summary_csv = output_dir / "teacher_musculoskeletal_replay_summary.csv"
        summary_json = output_dir / "teacher_musculoskeletal_replay_summary.json"
        summary_df.to_csv(summary_csv, index=False)
        summary_json.write_text(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "env_id": str(args.env_id),
                    "mode": str(args.mode),
                    "num_trials_processed": int(len(summary_df)),
                    "summaries": summary_df.to_dict(orient="records"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_summary_report(report_path, summaries, output_dir)
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
