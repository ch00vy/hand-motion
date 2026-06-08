from __future__ import annotations

"""Validate teacher controls frame-by-frame with mj_forward.

This script checks whether saved teacher controls are dynamically consistent
with the recorded trajectory state on a per-frame basis. The validation is
model-derived and intended for offline audit only.
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
    dataframe_to_markdown_simple,
    distribution_stats,
    indexed_columns,
    save_trial_dataset,
    set_full_state,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "teacher_mj_forward_validation"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "trajectory_inverse_teacher_forward_consistency.md"


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
    cols = df.columns.tolist()
    qpos_cols = _first_matching_indexed_columns(cols, ("qpos_target_",))
    qvel_cols = _first_matching_indexed_columns(cols, ("qvel_traj_", "qvel_target_", "qvel_current_"))
    qacc_cols = _first_matching_indexed_columns(cols, ("qacc_traj_", "qacc_target_", "qacc_des_"))
    tau_cols = _first_matching_indexed_columns(cols, ("qfrc_inverse_",))
    ctrl_cols = _first_matching_indexed_columns(cols, ("ctrl_teacher_",))
    if not qpos_cols or not tau_cols or not ctrl_cols:
        return None
    return {
        "source_teacher_csv": str(path),
        "source_csv": str(df["source_csv"].iloc[0]) if "source_csv" in df.columns else "",
        "exercise_name": str(df["exercise_name"].iloc[0]) if "exercise_name" in df.columns else "",
        "trial_id": str(df["trial_id"].iloc[0]) if "trial_id" in df.columns else path.stem,
        "frame_index": pd.to_numeric(df.get("frame_index", pd.Series(np.arange(len(df), dtype=float))), errors="coerce").fillna(pd.Series(np.arange(len(df), dtype=float))).to_numpy(dtype=int),
        "time_from_start": pd.to_numeric(df.get("time_from_start", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float) if "time_from_start" in df.columns else None,
        "qpos_target": df.loc[:, qpos_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float),
        "qvel_ref": df.loc[:, qvel_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float) if qvel_cols else None,
        "qacc_ref": df.loc[:, qacc_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float) if qacc_cols else None,
        "tau_target": df.loc[:, tau_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float),
        "ctrl_teacher": df.loc[:, ctrl_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float),
        "num_frames": int(len(df)),
    }


def validate_trial_with_mj_forward(
    model: Any,
    data: Any,
    trial: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    m_raw, d_raw = _get_raw_model_data(model, data)
    qpos_seq = np.asarray(trial["qpos_target"], dtype=float)
    qvel_seq = np.asarray(trial["qvel_ref"], dtype=float) if trial["qvel_ref"] is not None else np.zeros((qpos_seq.shape[0], int(m_raw.nv)), dtype=float)
    qacc_seq = np.asarray(trial["qacc_ref"], dtype=float) if trial["qacc_ref"] is not None else np.zeros((qpos_seq.shape[0], int(m_raw.nv)), dtype=float)
    tau_target_seq = np.asarray(trial["tau_target"], dtype=float)
    ctrl_seq = np.asarray(trial["ctrl_teacher"], dtype=float)

    rows: list[dict[str, Any]] = []
    tau_residual_norms: list[float] = []
    tau_relative_residuals: list[float] = []
    qacc_error_norms: list[float] = []

    for i in range(qpos_seq.shape[0]):
        qpos_t = qpos_seq[i]
        qvel_t = qvel_seq[i] if i < qvel_seq.shape[0] else np.zeros(int(m_raw.nv), dtype=float)
        qacc_t = qacc_seq[i] if i < qacc_seq.shape[0] else np.zeros(int(m_raw.nv), dtype=float)
        ctrl_t = ctrl_seq[i]
        tau_target = tau_target_seq[i]

        set_full_state(
            m_raw,
            d_raw,
            qpos_t,
            qvel_t,
            np.zeros(int(m_raw.nv), dtype=float),
            ctrl=ctrl_t,
            zero_act=True,
        )
        mujoco.mj_forward(m_raw, d_raw)

        tau_act = np.asarray(d_raw.qfrc_actuator[: int(m_raw.nv)], dtype=float).copy()
        tau_residual = tau_act - tau_target[: int(m_raw.nv)]
        tau_residual_norm = float(np.linalg.norm(tau_residual))
        tau_relative_residual = float(tau_residual_norm / max(float(np.linalg.norm(tau_target[: int(m_raw.nv)])), 1e-8))
        qacc_forward = np.asarray(d_raw.qacc[: int(m_raw.nv)], dtype=float).copy()
        qacc_compare = qacc_t[: int(m_raw.nv)]
        qacc_error = qacc_forward - qacc_compare
        qacc_error_norm = float(np.linalg.norm(qacc_error))

        tau_residual_norms.append(tau_residual_norm)
        tau_relative_residuals.append(tau_relative_residual)
        qacc_error_norms.append(qacc_error_norm)

        row: dict[str, Any] = {
            "source_teacher_csv": str(trial["source_teacher_csv"]),
            "source_csv": str(trial["source_csv"]),
            "exercise_name": str(trial["exercise_name"]),
            "trial_id": str(trial["trial_id"]),
            "frame_index": int(trial["frame_index"][i]) if i < len(trial["frame_index"]) else int(i),
            "time_from_start": float(trial["time_from_start"][i]) if trial["time_from_start"] is not None and i < len(trial["time_from_start"]) else float(i),
            "tau_target_norm": float(np.linalg.norm(tau_target[: int(m_raw.nv)])),
            "tau_actuator_norm": float(np.linalg.norm(tau_act)),
            "tau_residual_norm": tau_residual_norm,
            "tau_relative_residual": tau_relative_residual,
            "qacc_traj_norm": float(np.linalg.norm(qacc_compare)),
            "qacc_forward_norm": float(np.linalg.norm(qacc_forward)),
            "qacc_error_norm": qacc_error_norm,
        }
        for j in range(min(int(m_raw.nv), tau_target.shape[0])):
            row[f"qfrc_inverse_{j}"] = float(tau_target[j])
            row[f"qfrc_actuator_{j}"] = float(tau_act[j])
            row[f"tau_residual_{j}"] = float(tau_residual[j])
            row[f"qacc_traj_{j}"] = float(qacc_compare[j])
            row[f"qacc_forward_{j}"] = float(qacc_forward[j])
            row[f"qacc_error_{j}"] = float(qacc_error[j])
        rows.append(row)

    frame_df = pd.DataFrame(rows)
    summary = {
        "source_teacher_csv": str(trial["source_teacher_csv"]),
        "source_csv": str(trial["source_csv"]),
        "exercise_name": str(trial["exercise_name"]),
        "trial_id": str(trial["trial_id"]),
        "num_frames": int(qpos_seq.shape[0]),
        "mean_tau_residual_norm": float(np.mean(tau_residual_norms)) if tau_residual_norms else 0.0,
        "rmse_tau_residual_norm": float(math.sqrt(np.mean(np.square(tau_residual_norms)))) if tau_residual_norms else 0.0,
        "mean_tau_relative_residual": float(np.mean(tau_relative_residuals)) if tau_relative_residuals else 0.0,
        "mean_qacc_error_norm": float(np.mean(qacc_error_norms)) if qacc_error_norms else 0.0,
        "rmse_qacc_error_norm": float(math.sqrt(np.mean(np.square(qacc_error_norms)))) if qacc_error_norms else 0.0,
    }
    return frame_df, summary


def write_summary_report(report_path: Path, summaries: list[dict[str, Any]], output_dir: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Teacher mj_forward Consistency Summary",
        "",
        "This report checks saved teacher controls frame-by-frame using `mj_forward` at the recorded trajectory state.",
        "The main consistency metric is the residual between the inverse-dynamics target `qfrc_inverse` and the actuator-generated generalized force `qfrc_actuator` under the saved control.",
        "When available, forward dynamic acceleration is also compared against the trajectory-derived `qacc_traj`.",
        "",
    ]
    if not summaries:
        lines.append("No teacher trials were processed.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    df = pd.DataFrame(summaries)
    lines.extend(
        [
            f"- Trials processed: `{len(df)}`",
            f"- Mean tau residual norm (avg): `{float(df['mean_tau_residual_norm'].mean()):.4f}`",
            f"- Mean tau relative residual (avg): `{float(df['mean_tau_relative_residual'].mean()):.4f}`",
            f"- Mean qacc error norm (avg): `{float(df['mean_qacc_error_norm'].mean()):.4f}`",
            "",
            "## Aggregate Metrics",
            "",
            dataframe_to_markdown_simple(
                pd.DataFrame(
                    [
                        {"metric": "mean_tau_residual_norm", **distribution_stats(df["mean_tau_residual_norm"].astype(float).tolist())},
                        {"metric": "mean_tau_relative_residual", **distribution_stats(df["mean_tau_relative_residual"].astype(float).tolist())},
                        {"metric": "mean_qacc_error_norm", **distribution_stats(df["mean_qacc_error_norm"].astype(float).tolist())},
                    ]
                )
            ),
            "",
            "## Per-Trial Consistency Summary",
            "",
            dataframe_to_markdown_simple(
                df[
                    [
                        "trial_id",
                        "exercise_name",
                        "mean_tau_residual_norm",
                        "rmse_tau_residual_norm",
                        "mean_tau_relative_residual",
                        "mean_qacc_error_norm",
                    ]
                ]
            ),
            "",
            f"- Output directory: `{output_dir}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate saved teacher controls with frame-wise mj_forward consistency checks.")
    parser.add_argument("--input-path", type=str, required=True, help="Teacher CSV file or directory containing *__teacher_*.csv files.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for frame-wise consistency outputs.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown summary path.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on number of teacher trials.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frame_consistency"
    frame_dir.mkdir(parents=True, exist_ok=True)

    teacher_csvs = discover_teacher_csvs(input_path)
    if not teacher_csvs:
        raise SystemExit(f"No teacher CSVs found under {input_path}")
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
            print(f"[INFO] mj_forward consistency {trial['exercise_name']} / {trial['trial_id']} | frames={trial['num_frames']}")
            frame_df, summary = validate_trial_with_mj_forward(m_raw, d_raw, trial)
            out_csv = frame_dir / f"{trial['exercise_name']}__{trial['trial_id']}__mj_forward_consistency.csv"
            save_trial_dataset(frame_df, out_csv)
            summary["frame_csv_path"] = str(out_csv)
            summaries.append(summary)

        summary_df = pd.DataFrame(summaries).sort_values(by=["exercise_name", "trial_id"]).reset_index(drop=True) if summaries else pd.DataFrame()
        summary_csv = output_dir / "teacher_mj_forward_consistency_summary.csv"
        summary_json = output_dir / "teacher_mj_forward_consistency_summary.json"
        summary_df.to_csv(summary_csv, index=False)
        summary_json.write_text(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "env_id": str(args.env_id),
                    "num_trials_processed": int(len(summary_df)),
                    "trials": summary_df.to_dict(orient="records"),
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
