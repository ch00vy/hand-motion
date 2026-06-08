from __future__ import annotations

"""Optional legacy baseline for offline qpos-to-control reference generation.

Baseline teacher formulation:
    qacc_des = (qpos_target - qpos_current) * kp - qvel_current * kv

Canonical users should call scripts/offline_biomech/build_biomech_ctrl_reference.py.
This file remains because the trajectory backend imports small helper utilities.

This output is a model-based simulation reference, not biological ground truth.
It is designed for offline analysis and later model training only.
"""

import argparse
import json
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

from scripts.build_teacher_inverse_dynamics import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_PATH,
    build_per_gesture_summary,
    build_substeps_from_dt,
    clip_signal,
    compute_vector_norm_stats,
    dataframe_to_markdown_simple,
    discover_recording_csvs,
    distribution_stats,
    estimate_kinematics,
    forward_validate_sequence,
    load_recorded_trial,
    maybe_build_static_prior,
    normalize_timestamps,
    save_trial_dataset,
    set_full_state,
    solve_teacher_ctrl_frame,
    summarize_trial_quality,
    write_summary_report,
)
from app.main import DEFAULT_MYOSUITE_ENV_ID, _get_raw_model_data, create_probe_env  # noqa: E402


DEFAULT_PHASE1_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "teacher_from_qpos"
DEFAULT_PHASE1_REPORT_PATH = PROJECT_ROOT / "reports" / "teacher_solver_phase1_phase2_summary.md"


def repository_has_qfrc_scaler(root: Path) -> bool:
    """Inspect core runtime code for explicit qfrc scaling logic.

    The teacher baseline should bypass any qfrc scaling used by the current
    runtime solver. We intentionally restrict the search to the main runtime
    package so this detector is not polluted by report strings or helper
    scripts that merely mention `qfrc_scaler`.
    """

    search_roots = [root / "app"]
    for base in search_roots:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "venv" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if "qfrc_scaler" in text:
                return True
    return False


def compute_qacc_des(
    qpos_target: np.ndarray,
    qpos_current: np.ndarray,
    qvel_current: np.ndarray,
    *,
    kp: float,
    kv: float,
    qacc_clip: float,
) -> np.ndarray:
    qpos_target = np.asarray(qpos_target, dtype=float).reshape(-1)
    qpos_current = np.asarray(qpos_current, dtype=float).reshape(-1)
    qvel_current = np.asarray(qvel_current, dtype=float).reshape(-1)
    n = min(qpos_target.shape[0], qpos_current.shape[0], qvel_current.shape[0])
    qacc_des = np.zeros_like(qvel_current, dtype=float)
    if n > 0:
        qacc_des[:n] = (qpos_target[:n] - qpos_current[:n]) * float(kp) - qvel_current[:n] * float(kv)
    return clip_signal(qacc_des, qacc_clip)


def compute_inverse_dynamics_from_current_state(
    model: Any,
    data: Any,
    qpos_current: np.ndarray,
    qvel_current: np.ndarray,
    qacc_des: np.ndarray,
    *,
    disable_constraints: bool,
    tau_clip: float,
) -> np.ndarray:
    set_full_state(model, data, qpos_current, qvel_current, qacc_des, zero_act=True)
    m_raw, d_raw = _get_raw_model_data(model, data)
    old_flags = int(m_raw.opt.disableflags)
    try:
        if disable_constraints:
            m_raw.opt.disableflags = old_flags | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        mujoco.mj_inverse(m_raw, d_raw)
        tau = np.asarray(d_raw.qfrc_inverse[: m_raw.nv], dtype=float).copy()
        return clip_signal(tau, tau_clip)
    finally:
        m_raw.opt.disableflags = old_flags


def build_teacher_frame_table(
    trial: dict[str, Any],
    *,
    times: np.ndarray,
    dt: np.ndarray,
    qpos_raw: np.ndarray,
    qpos_target: np.ndarray,
    qpos_current_seq: np.ndarray,
    qvel_current_seq: np.ndarray,
    qacc_des_seq: np.ndarray,
    tau_seq: np.ndarray,
    ctrl_seq: np.ndarray,
    validation: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame_errors = list(validation.get("frame_errors", []))
    for i in range(qpos_target.shape[0]):
        row: dict[str, Any] = {
            "source_csv": str(trial["source_csv"]),
            "session_id": str(trial["session_id"]),
            "exercise_name": str(trial["exercise_name"]),
            "trial_id": str(trial["trial_id"]),
            "frame_index": int(trial["frame_indices"][i]) if i < len(trial["frame_indices"]) else int(i),
            "time_from_start": float(times[i]),
            "frame_dt": float(dt[i]),
            "forward_error_to_next": float(frame_errors[i]) if i < len(frame_errors) else np.nan,
            "tau_norm": float(np.linalg.norm(tau_seq[i])),
            "ctrl_norm": float(np.linalg.norm(ctrl_seq[i])),
            "qvel_current_norm": float(np.linalg.norm(qvel_current_seq[i])),
            "qacc_des_norm": float(np.linalg.norm(qacc_des_seq[i])),
        }
        for j in range(qpos_target.shape[1]):
            row[f"qpos_raw_{j}"] = float(qpos_raw[i, j])
            row[f"qpos_target_{j}"] = float(qpos_target[i, j])
            row[f"qpos_current_{j}"] = float(qpos_current_seq[i, j])
            row[f"qvel_current_{j}"] = float(qvel_current_seq[i, j])
            row[f"qacc_des_{j}"] = float(qacc_des_seq[i, j])
            row[f"qfrc_inverse_{j}"] = float(tau_seq[i, j])
        for j in range(ctrl_seq.shape[1]):
            row[f"ctrl_teacher_{j}"] = float(ctrl_seq[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def run_teacher_trial(
    trial: dict[str, Any],
    *,
    model: Any,
    data: Any,
    actuator_names: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    qpos_raw = np.asarray(trial["qpos_raw"], dtype=float)
    qpos_target, qvel_target, _qacc_target_unused, dt = estimate_kinematics(
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
    m_raw, d_raw = _get_raw_model_data(model, data)
    qvel_seed = qvel_target[0] if qvel_target.shape[0] > 0 else np.zeros(int(m_raw.nv), dtype=float)
    set_full_state(m_raw, d_raw, qpos_target[0], qvel_seed, np.zeros(int(m_raw.nv), dtype=float), zero_act=True)

    qpos_current_seq = np.zeros_like(qpos_target)
    qvel_current_seq = np.zeros((qpos_target.shape[0], int(m_raw.nv)), dtype=float)
    qacc_des_seq = np.zeros((qpos_target.shape[0], int(m_raw.nv)), dtype=float)
    tau_seq = np.zeros((qpos_target.shape[0], int(m_raw.nv)), dtype=float)
    ctrl_seq = np.zeros((qpos_target.shape[0], int(m_raw.nu)), dtype=float)
    solver_infos: list[dict[str, Any]] = []
    prev_ctrl = None
    substeps = build_substeps_from_dt(dt, float(m_raw.opt.timestep), int(args.max_substeps_per_frame))

    for i in range(qpos_target.shape[0]):
        qpos_current = np.asarray(d_raw.qpos[: int(m_raw.nq)], dtype=float).copy()
        qvel_current = np.asarray(d_raw.qvel[: int(m_raw.nv)], dtype=float).copy()
        qacc_des = compute_qacc_des(
            qpos_target[i],
            qpos_current,
            qvel_current,
            kp=float(args.kp),
            kv=float(args.kv),
            qacc_clip=float(args.qacc_clip),
        )
        tau_target = compute_inverse_dynamics_from_current_state(
            m_raw,
            data,
            qpos_current,
            qvel_current,
            qacc_des,
            disable_constraints=bool(args.disable_constraints),
            tau_clip=float(args.tau_clip),
        )
        static_prior_ctrl = maybe_build_static_prior(
            str(args.teacher_mode),
            str(args.teacher_prior_mode),
            float(args.static_prior_weight),
            m_raw,
            data,
            qpos_target[i],
            prev_ctrl,
            str(trial["exercise_name"]),
            actuator_names,
        )
        ctrl_teacher, info = solve_teacher_ctrl_frame(
            m_raw,
            data,
            tau_target,
            qpos_current,
            qvel_current,
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
        qpos_current_seq[i] = qpos_current[: qpos_target.shape[1]]
        qvel_current_seq[i] = qvel_current
        qacc_des_seq[i] = qacc_des
        tau_seq[i] = tau_target
        ctrl_seq[i] = ctrl_teacher
        solver_infos.append(info)
        prev_ctrl = ctrl_teacher.copy()
        if i < qpos_target.shape[0] - 1:
            for _ in range(int(substeps[min(i + 1, len(substeps) - 1)])):
                d_raw.ctrl[:] = 0.0
                d_raw.ctrl[: min(int(m_raw.nu), ctrl_teacher.shape[0])] = ctrl_teacher[: min(int(m_raw.nu), ctrl_teacher.shape[0])]
                mujoco.mj_step(m_raw, d_raw)

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
        qvel_target=qvel_current_seq,
        qacc_target=qacc_des_seq,
        tau_seq=tau_seq,
        max_mean_joint_error=float(args.max_mean_joint_error),
        max_rmse_forward_error=float(args.max_rmse_forward_error),
        max_upper_saturation_fraction=float(args.max_upper_saturation_fraction),
        max_ctrl_delta_mean=float(args.max_ctrl_delta_mean),
    )
    quality.update(
        {
            "teacher_kp": float(args.kp),
            "teacher_kv": float(args.kv),
            "qfrc_scaler_used": False,
        }
    )
    reject_reasons = quality.get("reject_reasons", [])
    if isinstance(reject_reasons, (list, tuple)):
        quality["reject_reasons"] = "|".join(str(item) for item in reject_reasons if str(item).strip())
    frame_table = build_teacher_frame_table(
        trial,
        times=times,
        dt=dt,
        qpos_raw=qpos_raw,
        qpos_target=qpos_target,
        qpos_current_seq=qpos_current_seq,
        qvel_current_seq=qvel_current_seq,
        qacc_des_seq=qacc_des_seq,
        tau_seq=tau_seq,
        ctrl_seq=ctrl_seq,
        validation=validation,
    )
    return frame_table, quality, validation


def write_phase1_report(
    report_path: Path,
    *,
    input_path: Path,
    args: argparse.Namespace,
    records_found: list[Path],
    summary_df: pd.DataFrame,
    output_dir: Path,
    qfrc_scaler_found: bool,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    passed = summary_df.loc[summary_df["keep_for_training"] == True]  # noqa: E712
    failed = summary_df.loc[summary_df["keep_for_training"] == False]  # noqa: E712
    per_gesture = build_per_gesture_summary(summary_df.to_dict(orient="records"))
    reject_counter = Counter()
    for value in summary_df["reject_reasons"].fillna("").astype(str).tolist():
        for item in [part for part in value.split("|") if part]:
            reject_counter[item] += 1
    lines = [
        "# Teacher Solver Phase 1 Summary",
        "",
        "This teacher dataset is a model-based simulation reference derived from MyoHand / MuJoCo.",
        "It is not biological ground truth.",
        "",
        "## Repository Inspection",
        "",
        f"- Recorded trajectories were loaded from `ctrl_est_*.csv` under `{input_path}`.",
        "- The recording CSVs contain frame-wise `qpos_*`, `exercise_name`, `trial_id`, `t`, and `frame_index` metadata.",
        f"- The MyoHand environment was loaded via `create_probe_env()` with env id `{args.env_id}`.",
        "- The current online solver path remains unchanged and lives in `estimate_ctrl_from_qpos()` / `RealtimeQPController` in `app/main.py`.",
        f"- `qfrc_scaler` logic found in repo: `{qfrc_scaler_found}`.",
        "- The baseline teacher bypasses qfrc scaling and uses raw inverse-dynamics generalized force output directly.",
        "",
        "## Teacher Formulation",
        "",
        f"- Desired acceleration formula: `qacc_des = (qpos_target - qpos_current) * {float(args.kp):.3f} - qvel_current * {float(args.kv):.3f}`.",
        f"- qpos target source: `{'raw_qpos' if bool(args.use_raw_qpos_target) else 'smoothed_qpos_target'}`.",
        f"- qpos / qvel / qacc smoothing windows: `{int(args.qpos_smoothing_window)}`, `{int(args.qvel_smoothing_window)}`, `{int(args.qacc_smoothing_window)}`.",
        f"- Constraints disabled during inverse dynamics: `{bool(args.disable_constraints)}`.",
        f"- Teacher mode: `{str(args.teacher_mode)}`.",
        f"- Teacher prior mode: `{str(args.teacher_prior_mode)}`.",
        "",
        "## Phase 1 Results",
        "",
        f"- CSV files discovered: `{len(records_found)}`",
        f"- Trials processed: `{len(summary_df)}`",
        f"- Trials accepted: `{int(len(passed))}`",
        f"- Trials rejected: `{int(len(failed))}`",
        "",
        "## Per-Gesture Summary",
        "",
    ]
    for gesture, summary in per_gesture.items():
        lines.append(
            f"- `{gesture}`: pass-rate `{summary['pass_rate']:.3f}` ({summary['num_kept']}/{summary['num_trials']}) | "
            f"mean joint error `{summary['mean_joint_error']['mean']:.4f}` | mean RMSE `{summary['rmse_forward_error']['mean']:.4f}`"
        )
    if reject_counter:
        lines.extend(["", "## Reject Reason Counts", ""])
        for reason, count in sorted(reject_counter.items()):
            lines.append(f"- `{reason}`: `{int(count)}`")
    best = summary_df.sort_values(by=["quality_score"], ascending=False).head(5)
    if not best.empty:
        lines.extend(["", "## Best Trials", "", dataframe_to_markdown_simple(best[["trial_id", "exercise_name", "quality_score", "mean_joint_error", "rmse_forward_error"]]), ""])
    lines.extend(
        [
            "## Output Files",
            "",
            f"- Trial summary: `{output_dir / 'teacher_trial_summary.csv'}`",
            f"- Metadata summary: `{output_dir / 'teacher_dataset_summary.json'}`",
            f"- Accepted trials: `{output_dir / 'accepted_trials'}`",
            f"- Rejected trials: `{output_dir / 'rejected_trials'}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 offline teacher generation from recorded qpos trajectories.")
    parser.add_argument("--input-path", type=str, default="records", help="CSV file or directory containing ctrl_est_*.csv recordings.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_PHASE1_OUTPUT_DIR), help="Output directory for teacher datasets.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_PHASE1_REPORT_PATH), help="Markdown report output path.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id.")
    parser.add_argument("--kp", type=float, default=1000.0, help="Proportional gain used in qacc_des.")
    parser.add_argument("--kv", type=float, default=100.0, help="Velocity gain used in qacc_des.")
    parser.add_argument("--timestep", type=float, default=1.0 / 60.0, help="Fallback timestep when CSV timestamps are missing.")
    parser.add_argument("--qpos-smoothing-window", type=int, default=5, help="Moving-average window for qpos target generation.")
    parser.add_argument("--qvel-smoothing-window", type=int, default=3, help="Optional smoothing window for qvel target estimation.")
    parser.add_argument("--qacc-smoothing-window", type=int, default=1, help="Optional smoothing window for qacc-related diagnostics.")
    parser.add_argument("--use-raw-qpos-target", action="store_true", help="Use raw qpos as teacher target instead of smoothed qpos.")
    parser.add_argument("--qvel-clip", type=float, default=0.0, help="Elementwise clip for qvel target. Use <=0 to disable.")
    parser.add_argument("--qacc-clip", type=float, default=0.0, help="Elementwise clip for qacc_des. Use <=0 to disable.")
    parser.add_argument("--tau-clip", type=float, default=0.0, help="Elementwise clip for qfrc_inverse. Use <=0 to disable.")
    parser.add_argument("--ctrl-lb", type=float, default=0.0, help="Lower bound for teacher controls.")
    parser.add_argument("--ctrl-ub", type=float, default=1.0, help="Upper bound for teacher controls.")
    parser.add_argument("--qp-reg", type=float, default=1e-3, help="Regularization for actuator-space solve.")
    parser.add_argument("--prev-ctrl-weight", type=float, default=5e-2, help="Penalty weight toward previous control.")
    parser.add_argument("--static-prior-weight", type=float, default=1e-1, help="Penalty weight toward the static qpos prior.")
    parser.add_argument("--prior-value", type=float, default=0.0, help="Fallback prior value when no static prior is used.")
    parser.add_argument("--teacher-prior-mode", type=str, default="static_qpos", choices=("zero", "static_qpos"), help="Static prior source.")
    parser.add_argument("--teacher-mode", type=str, default="dynamic_with_fallback", choices=("dynamic_only", "dynamic_with_static_prior", "dynamic_with_fallback"), help="Teacher solve mode.")
    parser.add_argument("--disable-constraints", action="store_true", help="Disable MuJoCo constraints during inverse dynamics.")
    parser.add_argument("--max-mean-joint-error", type=float, default=0.20, help="Reject trial if mean joint error exceeds this threshold.")
    parser.add_argument("--max-rmse-forward-error", type=float, default=0.75, help="Reject trial if forward RMSE exceeds this threshold.")
    parser.add_argument("--max-upper-saturation-fraction", type=float, default=0.35, help="Reject trial if too many controls saturate near 1.")
    parser.add_argument("--max-ctrl-delta-mean", type=float, default=0.30, help="Reject trial if control jitter is too large.")
    parser.add_argument("--max-substeps-per-frame", type=int, default=150, help="Maximum MuJoCo steps used between recorded frames.")
    parser.add_argument("--validation-window-size", type=int, default=0, help="Optional window size for forward replay diagnostics.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on number of trials.")
    parser.add_argument("--max-frames-per-trial", type=int, default=0, help="Optional cap on frames per trial.")
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
        filtered: list[Path] = []
        for path in records:
            trial = load_recorded_trial(path)
            if trial is not None and str(trial["exercise_name"]).strip().lower() in allowed:
                filtered.append(path)
        records = filtered
    if int(args.max_trials) > 0:
        records = records[: int(args.max_trials)]

    env = None
    trial_rows: list[dict[str, Any]] = []
    qfrc_scaler_found = repository_has_qfrc_scaler(PROJECT_ROOT)
    try:
        env, model, data = create_probe_env(str(args.env_id))
        m_raw, _ = _get_raw_model_data(model, data)
        actuator_names = [mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"muscle_{i}" for i in range(int(m_raw.nu))]
        for csv_path in records:
            trial = load_recorded_trial(csv_path)
            if trial is None:
                continue
            if int(args.max_frames_per_trial) > 0:
                cap = int(args.max_frames_per_trial)
            else:
                cap = 0
            if cap > 0:
                trial["qpos_raw"] = np.asarray(trial["qpos_raw"], dtype=float)[:cap]
                trial["frame_indices"] = np.asarray(trial["frame_indices"], dtype=int)[:cap]
                if trial["timestamps"] is not None:
                    trial["timestamps"] = np.asarray(trial["timestamps"], dtype=float)[:cap]
            print(f"[INFO] Teacher phase1 {trial['exercise_name']} / {trial['trial_id']} | frames={np.asarray(trial['qpos_raw']).shape[0]}")
            frame_table, quality, _ = run_teacher_trial(
                trial,
                model=m_raw,
                data=data,
                actuator_names=actuator_names,
                args=args,
            )
            keep = bool(quality["keep_for_training"])
            dst_dir = output_dir / ("accepted_trials" if keep else "rejected_trials")
            dst_path = dst_dir / f"{trial['exercise_name']}__{trial['trial_id']}__teacher_from_qpos.csv"
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
                    "kp": float(args.kp),
                    "kv": float(args.kv),
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
        write_phase1_report(
            report_path,
            input_path=input_path,
            args=args,
            records_found=records,
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
