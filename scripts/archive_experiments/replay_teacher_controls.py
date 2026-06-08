from __future__ import annotations

"""Replay offline teacher controls through MyoHand and validate forward pose.

This script is fully offline. It replays saved `ctrl_teacher` / `u_teacher`
sequences through the MuJoCo hand model, compares the forward pose against the
saved `qpos_target` trajectory, and writes debugging-friendly exports.

The replayed controls remain model-derived teacher references, not biological
ground truth.
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

from app.main import DEFAULT_MYOSUITE_ENV_ID, _get_raw_model_data, create_probe_env  # noqa: E402
from scripts.build_teacher_inverse_dynamics import (  # noqa: E402
    build_substeps_from_dt,
    dataframe_to_markdown_simple,
    distribution_stats,
    indexed_columns,
    normalize_timestamps,
    save_trial_dataset,
    set_full_state,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "teacher_replay_validation"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "teacher_replay_validation_summary.md"


def discover_teacher_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*__teacher_*.csv") if path.is_file())


def _first_matching_indexed_columns(columns: list[str], prefixes: tuple[str, ...]) -> list[str]:
    for prefix in prefixes:
        cols = indexed_columns(columns, prefix)
        if cols:
            return cols
    return []


def load_teacher_trial(path: Path) -> dict[str, Any] | None:
    df = pd.read_csv(path)
    if df.empty:
        return None
    qpos_target_cols = indexed_columns(df.columns.tolist(), "qpos_target_")
    ctrl_cols = indexed_columns(df.columns.tolist(), "ctrl_teacher_")
    qvel_cols = _first_matching_indexed_columns(df.columns.tolist(), ("qvel_current_", "qvel_traj_", "qvel_target_"))
    if not qpos_target_cols or not ctrl_cols:
        return None
    qpos_forward_hint_cols = indexed_columns(df.columns.tolist(), "qpos_current_")
    frame_dt = pd.to_numeric(df.get("frame_dt", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float) if "frame_dt" in df.columns else None
    times = pd.to_numeric(df.get("time_from_start", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float) if "time_from_start" in df.columns else None
    frame_series = pd.to_numeric(df.get("frame_index", pd.Series(dtype=float)), errors="coerce")
    if len(frame_series) != len(df):
        frame_series = pd.Series(np.arange(len(df), dtype=float))
    frame_series = frame_series.fillna(pd.Series(np.arange(len(df), dtype=float)))
    frame_indices = frame_series.to_numpy(dtype=int)
    return {
        "source_teacher_csv": str(path),
        "source_csv": str(df["source_csv"].iloc[0]) if "source_csv" in df.columns else "",
        "session_id": str(df["session_id"].iloc[0]) if "session_id" in df.columns else "",
        "exercise_name": str(df["exercise_name"].iloc[0]) if "exercise_name" in df.columns else "",
        "trial_id": str(df["trial_id"].iloc[0]) if "trial_id" in df.columns else path.stem,
        "frame_indices": frame_indices,
        "time_from_start": times,
        "frame_dt": frame_dt,
        "qpos_target": df.loc[:, qpos_target_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float),
        "qvel_init": (
            df.loc[:, qvel_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            if qvel_cols
            else None
        ),
        "ctrl_teacher": df.loc[:, ctrl_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float),
        "qpos_forward_hint": (
            df.loc[:, qpos_forward_hint_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            if qpos_forward_hint_cols
            else None
        ),
        "num_frames": int(len(df)),
    }


def replay_teacher_sequence(
    model: Any,
    data: Any,
    trial: dict[str, Any],
    *,
    fallback_dt: float,
    max_substeps: int,
) -> dict[str, Any]:
    m_raw, d_raw = _get_raw_model_data(model, data)
    qpos_target = np.asarray(trial["qpos_target"], dtype=float)
    ctrl_seq = np.asarray(trial["ctrl_teacher"], dtype=float)
    qvel_init_all = np.asarray(trial["qvel_init"], dtype=float) if trial["qvel_init"] is not None else None
    if qvel_init_all is not None and qvel_init_all.shape[0] > 0:
        qvel_init = qvel_init_all[0]
    else:
        qvel_init = np.zeros(int(m_raw.nv), dtype=float)
    times = None if trial["time_from_start"] is None else np.asarray(trial["time_from_start"], dtype=float)
    _, dt = normalize_timestamps(times, qpos_target.shape[0], fallback_dt)
    if trial["frame_dt"] is not None and len(trial["frame_dt"]) == qpos_target.shape[0]:
        dt_candidate = np.asarray(trial["frame_dt"], dtype=float)
        dt = np.where(np.isfinite(dt_candidate) & (dt_candidate > 1e-6), dt_candidate, dt)
    set_full_state(
        m_raw,
        d_raw,
        qpos_target[0],
        qvel_init,
        np.zeros(int(m_raw.nv), dtype=float),
        ctrl=ctrl_seq[0] if ctrl_seq.shape[0] > 0 else None,
        zero_act=True,
    )

    substeps = build_substeps_from_dt(dt, float(m_raw.opt.timestep), int(max_substeps))
    forward_qpos_seq = [np.asarray(d_raw.qpos[: int(m_raw.nq)], dtype=float).copy()]
    frame_error_norms = [0.0]
    per_joint_sqerr = np.zeros(int(m_raw.nq), dtype=float)
    per_joint_count = np.zeros(int(m_raw.nq), dtype=float)

    for i in range(qpos_target.shape[0] - 1):
        ctrl = np.asarray(ctrl_seq[i], dtype=float).reshape(-1)
        for _ in range(int(substeps[min(i + 1, len(substeps) - 1)])):
            d_raw.ctrl[:] = 0.0
            d_raw.ctrl[: min(int(m_raw.nu), ctrl.shape[0])] = ctrl[: min(int(m_raw.nu), ctrl.shape[0])]
            mujoco.mj_step(m_raw, d_raw)
        forward = np.asarray(d_raw.qpos[: int(m_raw.nq)], dtype=float).copy()
        target_next = np.asarray(qpos_target[i + 1], dtype=float).reshape(-1)
        nq = min(int(m_raw.nq), target_next.shape[0], forward.shape[0])
        joint_err = forward[:nq] - target_next[:nq]
        per_joint_sqerr[:nq] += joint_err ** 2
        per_joint_count[:nq] += 1.0
        frame_error_norms.append(float(np.linalg.norm(joint_err)))
        forward_qpos_seq.append(forward)

    forward_arr = np.asarray(forward_qpos_seq, dtype=float)
    nq = min(forward_arr.shape[1], qpos_target.shape[1]) if forward_arr.ndim == 2 else qpos_target.shape[1]
    joint_error_seq = forward_arr[:, :nq] - qpos_target[:, :nq]
    per_joint_rmse = np.sqrt(
        np.divide(
            per_joint_sqerr[:nq],
            np.maximum(per_joint_count[:nq], 1.0),
            out=np.zeros(nq, dtype=float),
            where=per_joint_count[:nq] > 0,
        )
    )
    frame_err_arr = np.asarray(frame_error_norms[1:], dtype=float) if len(frame_error_norms) > 1 else np.zeros(0, dtype=float)
    mean_joint_error = float(np.mean(per_joint_rmse)) if per_joint_rmse.size else 0.0
    rmse_forward_error = float(math.sqrt(np.mean(np.square(frame_err_arr)))) if frame_err_arr.size else 0.0
    return {
        "qpos_target": qpos_target,
        "qpos_forward": forward_arr[:, :nq],
        "joint_error_seq": joint_error_seq,
        "frame_error_norms": frame_error_norms,
        "per_joint_rmse": per_joint_rmse,
        "mean_joint_error": mean_joint_error,
        "rmse_forward_error": rmse_forward_error,
        "mean_pose_error_norm": float(np.mean(frame_err_arr)) if frame_err_arr.size else 0.0,
        "max_frame_error": float(np.max(frame_err_arr)) if frame_err_arr.size else 0.0,
        "dt": dt,
        "substeps": substeps,
    }


def build_replay_table(trial: dict[str, Any], replay: dict[str, Any]) -> pd.DataFrame:
    qpos_target = np.asarray(replay["qpos_target"], dtype=float)
    qpos_forward = np.asarray(replay["qpos_forward"], dtype=float)
    joint_error = np.asarray(replay["joint_error_seq"], dtype=float)
    frame_error_norms = list(replay["frame_error_norms"])
    dt = np.asarray(replay["dt"], dtype=float)
    frame_indices = np.asarray(trial["frame_indices"], dtype=int)
    times = np.asarray(trial["time_from_start"], dtype=float) if trial["time_from_start"] is not None else np.cumsum(dt) - dt[0]

    rows: list[dict[str, Any]] = []
    for i in range(qpos_target.shape[0]):
        row: dict[str, Any] = {
            "source_teacher_csv": str(trial["source_teacher_csv"]),
            "source_csv": str(trial["source_csv"]),
            "session_id": str(trial["session_id"]),
            "exercise_name": str(trial["exercise_name"]),
            "trial_id": str(trial["trial_id"]),
            "frame_index": int(frame_indices[i]) if i < frame_indices.shape[0] else int(i),
            "time_from_start": float(times[i]) if i < times.shape[0] else float(i),
            "frame_dt": float(dt[i]) if i < dt.shape[0] else np.nan,
            "frame_error_norm": float(frame_error_norms[i]) if i < len(frame_error_norms) else np.nan,
        }
        for j in range(qpos_target.shape[1]):
            row[f"qpos_target_{j}"] = float(qpos_target[i, j])
            row[f"qpos_forward_{j}"] = float(qpos_forward[i, j])
            row[f"joint_error_{j}"] = float(joint_error[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def maybe_plot_replay(trial: dict[str, Any], replay: dict[str, Any], out_png: Path, out_svg: Path, top_k_joints: int) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib_not_available"

    qpos_target = np.asarray(replay["qpos_target"], dtype=float)
    qpos_forward = np.asarray(replay["qpos_forward"], dtype=float)
    per_joint_rmse = np.asarray(replay["per_joint_rmse"], dtype=float)
    frame_error_norms = np.asarray(replay["frame_error_norms"], dtype=float)
    times = np.asarray(trial["time_from_start"], dtype=float) if trial["time_from_start"] is not None else np.arange(qpos_target.shape[0], dtype=float)
    top_k = max(1, min(int(top_k_joints), qpos_target.shape[1]))
    top_idx = np.argsort(per_joint_rmse)[::-1][:top_k]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for joint_idx in top_idx:
        axes[0].plot(times, qpos_target[:, joint_idx], linestyle="-", linewidth=1.8, label=f"target q{joint_idx}")
        axes[0].plot(times, qpos_forward[:, joint_idx], linestyle="--", linewidth=1.2, label=f"forward q{joint_idx}")
    axes[0].set_title(f"Teacher Replay vs Target: {trial['exercise_name']} / {trial['trial_id']}")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("qpos")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(times, frame_error_norms, color="tab:red", linewidth=1.8)
    axes[1].set_title("Per-frame forward pose error norm")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("error norm")
    axes[1].grid(True, alpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    return None


def write_summary_report(
    report_path: Path,
    *,
    trial_summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trial_summaries)
    lines = [
        "# Teacher Replay Validation Summary",
        "",
        "This replay validates saved teacher controls by forwarding them through the MyoHand MuJoCo model.",
        "The replay remains a model-derived consistency check, not biological ground truth.",
        "",
    ]
    if df.empty:
        lines.append("No trials were processed.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

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
            "## Per-Trial Summary",
            "",
            dataframe_to_markdown_simple(
                df[
                    [
                        "trial_id",
                        "exercise_name",
                        "mean_joint_error",
                        "rmse_forward_error",
                        "mean_pose_error_norm",
                        "largest_error_joint",
                        "largest_error_joint_rmse",
                        "close_enough",
                    ]
                ]
            ),
            "",
            "## Output Files",
            "",
            f"- Replay tables: `{output_dir / 'replayed_trials'}`",
            f"- Trial summary CSV: `{output_dir / 'teacher_replay_validation_summary.csv'}`",
            f"- Trial summary JSON: `{output_dir / 'teacher_replay_validation_summary.json'}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved teacher controls and validate forward pose against qpos_target.")
    parser.add_argument("--input-path", type=str, required=True, help="Teacher trial CSV file or directory containing *teacher_from_qpos.csv.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for replay validation outputs.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown summary path.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id.")
    parser.add_argument("--timestep", type=float, default=1.0 / 60.0, help="Fallback timestep when frame_dt/time is missing.")
    parser.add_argument("--max-substeps-per-frame", type=int, default=150, help="Maximum MuJoCo steps between recorded frames.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on number of teacher trial CSVs.")
    parser.add_argument("--max-mean-joint-error", type=float, default=0.20, help="Threshold used to judge whether replay is close enough.")
    parser.add_argument("--max-rmse-forward-error", type=float, default=0.75, help="Threshold used to judge whether replay is close enough.")
    parser.add_argument("--save-plots", action="store_true", help="Save matplotlib comparison plots if matplotlib is available.")
    parser.add_argument("--plot-top-k-joints", type=int, default=6, help="Number of highest-error joints to visualize.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_path).resolve()
    report_path = Path(args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = output_dir / "replayed_trials"
    replay_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    if bool(args.save_plots):
        plot_dir.mkdir(parents=True, exist_ok=True)

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
            print(f"[INFO] Replay teacher {trial['exercise_name']} / {trial['trial_id']} | frames={trial['num_frames']}")
            replay = replay_teacher_sequence(
                m_raw,
                d_raw,
                trial,
                fallback_dt=float(args.timestep),
                max_substeps=int(args.max_substeps_per_frame),
            )
            replay_df = build_replay_table(trial, replay)
            out_csv = replay_dir / f"{trial['exercise_name']}__{trial['trial_id']}__replay_validation.csv"
            save_trial_dataset(replay_df, out_csv)

            largest_joint_idx = int(np.argmax(np.asarray(replay["per_joint_rmse"], dtype=float))) if len(replay["per_joint_rmse"]) else 0
            largest_joint_rmse = float(np.max(np.asarray(replay["per_joint_rmse"], dtype=float))) if len(replay["per_joint_rmse"]) else 0.0
            close_enough = bool(
                float(replay["mean_joint_error"]) <= float(args.max_mean_joint_error)
                and float(replay["rmse_forward_error"]) <= float(args.max_rmse_forward_error)
            )
            summary_row = {
                "source_teacher_csv": str(csv_path),
                "source_csv": str(trial["source_csv"]),
                "exercise_name": str(trial["exercise_name"]),
                "trial_id": str(trial["trial_id"]),
                "num_frames": int(trial["num_frames"]),
                "mean_joint_error": float(replay["mean_joint_error"]),
                "rmse_forward_error": float(replay["rmse_forward_error"]),
                "mean_pose_error_norm": float(replay["mean_pose_error_norm"]),
                "largest_error_joint": f"qpos_{largest_joint_idx}",
                "largest_error_joint_rmse": largest_joint_rmse,
                "close_enough": close_enough,
                "replay_table_path": str(out_csv),
            }
            if bool(args.save_plots):
                out_png = plot_dir / f"{trial['exercise_name']}__{trial['trial_id']}__replay_plot.png"
                out_svg = plot_dir / f"{trial['exercise_name']}__{trial['trial_id']}__replay_plot.svg"
                plot_warning = maybe_plot_replay(trial, replay, out_png, out_svg, int(args.plot_top_k_joints))
                summary_row["plot_warning"] = plot_warning or ""
                if plot_warning is None:
                    summary_row["plot_png"] = str(out_png)
                    summary_row["plot_svg"] = str(out_svg)
            summaries.append(summary_row)

        summary_df = pd.DataFrame(summaries).sort_values(by=["exercise_name", "trial_id"]).reset_index(drop=True)
        summary_csv = output_dir / "teacher_replay_validation_summary.csv"
        summary_json = output_dir / "teacher_replay_validation_summary.json"
        summary_df.to_csv(summary_csv, index=False)
        summary_json.write_text(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "env_id": str(args.env_id),
                    "num_trials_processed": int(len(summary_df)),
                    "num_close_enough": int(summary_df["close_enough"].sum()) if not summary_df.empty else 0,
                    "num_not_close_enough": int((~summary_df["close_enough"]).sum()) if not summary_df.empty else 0,
                    "trials": summary_df.to_dict(orient="records"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_summary_report(report_path, trial_summaries=summaries, output_dir=output_dir)
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
