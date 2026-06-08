from __future__ import annotations

"""Sweep short-horizon rollout parameters on selected offline teacher trials.

This helper keeps the realtime runtime untouched. It reuses the trajectory
inverse teacher builder offline and compares:
- baseline framewise trajectory-inverse solve
- short-horizon rollout solve

The default sweep strategy is one-at-a-time so the search remains practical for
small subsets such as three open trials and three fist trials.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import DEFAULT_MYOSUITE_ENV_ID, _get_raw_model_data, create_probe_env  # noqa: E402
from scripts.build_teacher_from_trajectory_inverse import run_trajectory_teacher_trial  # noqa: E402
from scripts.build_teacher_inverse_dynamics import dataframe_to_markdown_simple, load_recorded_trial  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "teacher_short_horizon_sweeps"


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(float(item.strip())) for item in str(raw).split(",") if item.strip()]


def build_base_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        timestep=float(cli_args.timestep),
        qpos_smoothing_window=int(cli_args.qpos_smoothing_window),
        qvel_smoothing_window=int(cli_args.qvel_smoothing_window),
        qacc_smoothing_window=int(cli_args.qacc_smoothing_window),
        use_raw_qpos_target=bool(cli_args.use_raw_qpos_target),
        qvel_clip=float(cli_args.qvel_clip),
        qacc_clip=float(cli_args.qacc_clip),
        tau_clip=float(cli_args.tau_clip),
        ctrl_lb=float(cli_args.ctrl_lb),
        ctrl_ub=float(cli_args.ctrl_ub),
        qp_reg=float(cli_args.qp_reg),
        prev_ctrl_weight=float(cli_args.prev_ctrl_weight),
        static_prior_weight=float(cli_args.static_prior_weight),
        prior_value=float(cli_args.prior_value),
        teacher_prior_mode=str(cli_args.teacher_prior_mode),
        teacher_mode=str(cli_args.teacher_mode),
        disable_constraints=bool(cli_args.disable_constraints),
        max_mean_joint_error=float(cli_args.max_mean_joint_error),
        max_rmse_forward_error=float(cli_args.max_rmse_forward_error),
        max_upper_saturation_fraction=float(cli_args.max_upper_saturation_fraction),
        max_ctrl_delta_mean=float(cli_args.max_ctrl_delta_mean),
        max_substeps_per_frame=int(cli_args.max_substeps_per_frame),
        validation_window_size=int(cli_args.validation_window_size),
        suspicious_frame_error=float(cli_args.suspicious_frame_error),
        suspicious_qacc_norm=float(cli_args.suspicious_qacc_norm),
        suspicious_tau_norm=float(cli_args.suspicious_tau_norm),
        horizon_size=int(cli_args.horizon_size),
        horizon_iters=int(cli_args.horizon_iters),
        commit_frames=int(cli_args.commit_frames),
        rollout_qpos_weight=float(cli_args.rollout_qpos_weight),
        rollout_qvel_weight=float(cli_args.rollout_qvel_weight),
        rollout_u_weight=float(cli_args.rollout_u_weight),
        rollout_delta_u_weight=float(cli_args.rollout_delta_u_weight),
        rollout_prior_weight=float(cli_args.rollout_prior_weight),
    )


def select_trials(
    baseline_summary_csv: Path,
    *,
    gesture: str,
    trial_count: int,
    min_frames: int,
) -> pd.DataFrame:
    df = pd.read_csv(baseline_summary_csv)
    sub = df.loc[(df["exercise_name"].astype(str).str.lower() == gesture.lower()) & (df["num_frames"].astype(int) >= int(min_frames))].copy()
    sub = sub.sort_values(by=["rmse_forward_error", "mean_joint_error", "num_frames"], ascending=[True, True, False]).head(int(trial_count)).reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"No trials found for gesture={gesture} with min_frames={min_frames} in {baseline_summary_csv}")
    return sub


def evaluate_config_on_trials(
    trial_paths: list[Path],
    *,
    args_obj: argparse.Namespace,
    model: Any,
    data: Any,
    actuator_names: list[str],
    config_id: str,
    gesture_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial_path in trial_paths:
        trial = load_recorded_trial(trial_path)
        if trial is None:
            continue
        frame_table, quality, _validation = run_trajectory_teacher_trial(
            trial,
            model=model,
            data=data,
            actuator_names=actuator_names,
            args=args_obj,
        )
        row = dict(quality)
        row["source_csv"] = str(trial_path)
        row["config_id"] = str(config_id)
        row["gesture_sweep"] = str(gesture_label)
        row["teacher_mode_used"] = str(args_obj.teacher_mode)
        row["rollout_qvel_weight"] = float(args_obj.rollout_qvel_weight)
        row["rollout_delta_u_weight"] = float(args_obj.rollout_delta_u_weight)
        row["rollout_prior_weight"] = float(args_obj.rollout_prior_weight)
        row["horizon_size"] = int(args_obj.horizon_size)
        row["horizon_iters"] = int(args_obj.horizon_iters)
        row["commit_frames"] = int(args_obj.commit_frames)
        row["p95_frame_error"] = float(row.get("p95_frame_error", 0.0))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No trials evaluated for config {config_id}")
    summary = {
        "config_id": str(config_id),
        "gesture_sweep": str(gesture_label),
        "teacher_mode_used": str(args_obj.teacher_mode),
        "num_trials": int(len(df)),
        "num_kept": int(df["keep_for_training"].sum()) if "keep_for_training" in df.columns else 0,
        "pass_rate": float(df["keep_for_training"].mean()) if "keep_for_training" in df.columns else 0.0,
        "mean_rmse_forward_error": float(df["rmse_forward_error"].mean()),
        "mean_mean_joint_error": float(df["mean_joint_error"].mean()),
        "mean_p95_frame_error": float(df["p95_frame_error"].mean()),
        "mean_ctrl_delta_mean": float(df["ctrl_delta_mean"].mean()),
        "rollout_qvel_weight": float(args_obj.rollout_qvel_weight),
        "rollout_delta_u_weight": float(args_obj.rollout_delta_u_weight),
        "rollout_prior_weight": float(args_obj.rollout_prior_weight),
        "horizon_size": int(args_obj.horizon_size),
        "horizon_iters": int(args_obj.horizon_iters),
        "commit_frames": int(args_obj.commit_frames),
    }
    return df, summary


def choose_best_summary(summaries_df: pd.DataFrame) -> pd.Series:
    ordered = summaries_df.sort_values(
        by=[
            "mean_rmse_forward_error",
            "mean_mean_joint_error",
            "mean_p95_frame_error",
            "mean_ctrl_delta_mean",
        ],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return ordered.iloc[0]


def run_one_at_a_time_sweep(
    trial_paths: list[Path],
    *,
    gesture_label: str,
    base_args: argparse.Namespace,
    model: Any,
    data: Any,
    actuator_names: list[str],
    qvel_values: list[float],
    delta_u_values: list[float],
    prior_values: list[float],
    horizon_sizes: list[int],
    horizon_iters_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    baseline_args = copy.deepcopy(base_args)
    baseline_args.teacher_mode = "dynamic_with_static_prior"
    baseline_trials_df, baseline_summary = evaluate_config_on_trials(
        trial_paths,
        args_obj=baseline_args,
        model=model,
        data=data,
        actuator_names=actuator_names,
        config_id="baseline_framewise",
        gesture_label=gesture_label,
    )

    selected_args = copy.deepcopy(base_args)
    selected_args.teacher_mode = "short_horizon_rollout"
    step_rows: list[dict[str, Any]] = [dict(baseline_summary, sweep_stage="baseline", tuned_param="none", candidate_value="baseline")]
    per_trial_rows: list[pd.DataFrame] = [baseline_trials_df]

    sweep_plan = [
        ("rollout_qvel_weight", qvel_values),
        ("rollout_delta_u_weight", delta_u_values),
        ("rollout_prior_weight", prior_values),
        ("horizon_size", horizon_sizes),
        ("horizon_iters", horizon_iters_values),
    ]

    for param_name, values in sweep_plan:
        candidate_summaries: list[dict[str, Any]] = []
        candidate_trials: list[pd.DataFrame] = []
        for value in values:
            cfg = copy.deepcopy(selected_args)
            setattr(cfg, param_name, value)
            config_id = f"{gesture_label}__{param_name}__{value}"
            trial_df, summary = evaluate_config_on_trials(
                trial_paths,
                args_obj=cfg,
                model=model,
                data=data,
                actuator_names=actuator_names,
                config_id=config_id,
                gesture_label=gesture_label,
            )
            summary["sweep_stage"] = "one_at_a_time"
            summary["tuned_param"] = param_name
            summary["candidate_value"] = value
            candidate_summaries.append(summary)
            candidate_trials.append(trial_df)
        stage_df = pd.DataFrame(candidate_summaries)
        best = choose_best_summary(stage_df)
        setattr(selected_args, param_name, best[param_name])
        step_rows.extend(candidate_summaries)
        best_config_id = str(best["config_id"])
        for trial_df in candidate_trials:
            if str(trial_df["config_id"].iloc[0]) == best_config_id:
                per_trial_rows.append(trial_df)
                break

    final_args = copy.deepcopy(selected_args)
    final_args.teacher_mode = "short_horizon_rollout"
    final_trials_df, final_summary = evaluate_config_on_trials(
        trial_paths,
        args_obj=final_args,
        model=model,
        data=data,
        actuator_names=actuator_names,
        config_id=f"{gesture_label}__final_short_horizon",
        gesture_label=gesture_label,
    )
    final_summary["sweep_stage"] = "final"
    final_summary["tuned_param"] = "all"
    final_summary["candidate_value"] = "final_short_horizon"
    step_rows.append(final_summary)
    per_trial_rows.append(final_trials_df)

    baseline_merge = baseline_trials_df[["trial_id", "rmse_forward_error", "mean_joint_error", "p95_frame_error", "ctrl_delta_mean"]].rename(
        columns={
            "rmse_forward_error": "baseline_rmse_forward_error",
            "mean_joint_error": "baseline_mean_joint_error",
            "p95_frame_error": "baseline_p95_frame_error",
            "ctrl_delta_mean": "baseline_ctrl_delta_mean",
        }
    )
    final_compare = final_trials_df.merge(baseline_merge, on="trial_id", how="left")
    final_compare["delta_rmse_forward_error"] = final_compare["rmse_forward_error"] - final_compare["baseline_rmse_forward_error"]
    final_compare["delta_mean_joint_error"] = final_compare["mean_joint_error"] - final_compare["baseline_mean_joint_error"]
    final_compare["delta_p95_frame_error"] = final_compare["p95_frame_error"] - final_compare["baseline_p95_frame_error"]
    final_compare["delta_ctrl_delta_mean"] = final_compare["ctrl_delta_mean"] - final_compare["baseline_ctrl_delta_mean"]

    return pd.concat(per_trial_rows, ignore_index=True), pd.DataFrame(step_rows), final_summary, final_compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep short-horizon rollout parameters on selected trajectory-inverse teacher trials.")
    parser.add_argument("--baseline-summary-csv", type=str, default=str(PROJECT_ROOT / "artifacts" / "teacher_from_trajectory_inverse_full" / "teacher_trial_summary.csv"), help="Baseline trajectory-inverse summary used to select representative trials.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for sweep outputs.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite environment id.")
    parser.add_argument("--trial-count", type=int, default=3, help="Number of trials to select per gesture.")
    parser.add_argument("--min-frames", type=int, default=20, help="Minimum frame count when selecting candidate trials from the baseline summary.")
    parser.add_argument("--run-open", action="store_true", help="Run the open-gesture sweep.")
    parser.add_argument("--run-fist", action="store_true", help="Run the fist-gesture sweep.")
    parser.add_argument("--timestep", type=float, default=1.0 / 60.0)
    parser.add_argument("--qpos-smoothing-window", type=int, default=5)
    parser.add_argument("--qvel-smoothing-window", type=int, default=3)
    parser.add_argument("--qacc-smoothing-window", type=int, default=3)
    parser.add_argument("--use-raw-qpos-target", action="store_true")
    parser.add_argument("--qvel-clip", type=float, default=0.0)
    parser.add_argument("--qacc-clip", type=float, default=0.0)
    parser.add_argument("--tau-clip", type=float, default=0.0)
    parser.add_argument("--ctrl-lb", type=float, default=0.0)
    parser.add_argument("--ctrl-ub", type=float, default=1.0)
    parser.add_argument("--qp-reg", type=float, default=1e-3)
    parser.add_argument("--prev-ctrl-weight", type=float, default=5e-2)
    parser.add_argument("--static-prior-weight", type=float, default=1e-1)
    parser.add_argument("--prior-value", type=float, default=0.0)
    parser.add_argument("--teacher-prior-mode", type=str, default="static_qpos", choices=("zero", "static_qpos"))
    parser.add_argument("--teacher-mode", type=str, default="short_horizon_rollout")
    parser.add_argument("--disable-constraints", action="store_true")
    parser.add_argument("--max-mean-joint-error", type=float, default=0.20)
    parser.add_argument("--max-rmse-forward-error", type=float, default=0.75)
    parser.add_argument("--max-upper-saturation-fraction", type=float, default=0.35)
    parser.add_argument("--max-ctrl-delta-mean", type=float, default=0.30)
    parser.add_argument("--max-substeps-per-frame", type=int, default=150)
    parser.add_argument("--validation-window-size", type=int, default=0)
    parser.add_argument("--suspicious-frame-error", type=float, default=1.0)
    parser.add_argument("--suspicious-qacc-norm", type=float, default=1000.0)
    parser.add_argument("--suspicious-tau-norm", type=float, default=1000.0)
    parser.add_argument("--horizon-size", type=int, default=5)
    parser.add_argument("--horizon-iters", type=int, default=5)
    parser.add_argument("--commit-frames", type=int, default=1)
    parser.add_argument("--rollout-qpos-weight", type=float, default=1.0)
    parser.add_argument("--rollout-qvel-weight", type=float, default=0.25)
    parser.add_argument("--rollout-u-weight", type=float, default=0.01)
    parser.add_argument("--rollout-delta-u-weight", type=float, default=0.10)
    parser.add_argument("--rollout-prior-weight", type=float, default=0.10)
    parser.add_argument("--open-qvel-values", type=str, default="0.1,0.25,0.5")
    parser.add_argument("--open-delta-u-values", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--open-prior-values", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--open-horizon-sizes", type=str, default="3,5,7")
    parser.add_argument("--open-horizon-iters", type=str, default="3,5")
    parser.add_argument("--fist-qvel-values", type=str, default="0.25,0.5,1.0")
    parser.add_argument("--fist-delta-u-values", type=str, default="0.02,0.05,0.1")
    parser.add_argument("--fist-prior-values", type=str, default="0.02,0.05,0.1")
    parser.add_argument("--fist-horizon-sizes", type=str, default="5,7,9")
    parser.add_argument("--fist-horizon-iters", type=str, default="3,5")
    return parser.parse_args()


def write_report(output_dir: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Short-Horizon Teacher Sweep Summary",
        "",
        "This sweep compares the baseline framewise trajectory-inverse teacher against a short-horizon rollout refinement on selected offline trials.",
        "",
    ]
    for gesture, payload in results.items():
        selected_df = payload["selected_trials"]
        summary_df = payload["step_summary"]
        final_compare_df = payload["final_compare"]
        best = payload["best_summary"]
        lines.extend(
            [
                f"## {gesture.capitalize()}",
                "",
                "### Selected Trials",
                "",
                dataframe_to_markdown_simple(selected_df[["trial_id", "source_csv", "num_frames", "rmse_forward_error", "mean_joint_error"]]),
                "",
                "### Best Short-Horizon Configuration",
                "",
                f"- rollout_qvel_weight: `{best['rollout_qvel_weight']}`",
                f"- rollout_delta_u_weight: `{best['rollout_delta_u_weight']}`",
                f"- rollout_prior_weight: `{best['rollout_prior_weight']}`",
                f"- horizon_size: `{best['horizon_size']}`",
                f"- horizon_iters: `{best['horizon_iters']}`",
                f"- mean RMSE forward error: `{best['mean_rmse_forward_error']:.4f}`",
                f"- mean joint error: `{best['mean_mean_joint_error']:.4f}`",
                f"- mean p95 frame error: `{best['mean_p95_frame_error']:.4f}`",
                f"- mean ctrl delta: `{best['mean_ctrl_delta_mean']:.4f}`",
                "",
                "### Final Trial-Level Delta vs Baseline",
                "",
                dataframe_to_markdown_simple(final_compare_df[["trial_id", "baseline_rmse_forward_error", "rmse_forward_error", "delta_rmse_forward_error", "baseline_mean_joint_error", "mean_joint_error", "delta_mean_joint_error"]]),
                "",
                "### Sweep Steps",
                "",
                dataframe_to_markdown_simple(summary_df[["config_id", "tuned_param", "candidate_value", "mean_rmse_forward_error", "mean_mean_joint_error", "mean_p95_frame_error", "mean_ctrl_delta_mean"]]),
                "",
            ]
        )
    (output_dir / "sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary_csv = Path(args.baseline_summary_csv).resolve()
    run_open = bool(args.run_open) or (not bool(args.run_open) and not bool(args.run_fist))
    run_fist = bool(args.run_fist) or (not bool(args.run_open) and not bool(args.run_fist))
    base_args = build_base_args(args)

    env = None
    results: dict[str, Any] = {}
    try:
        env, model, data = create_probe_env(str(args.env_id))
        m_raw, d_raw = _get_raw_model_data(model, data)
        actuator_names = [f"muscle_{i}" for i in range(int(m_raw.nu))]
        for i in range(int(m_raw.nu)):
            name = mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_ACTUATOR, i)  # type: ignore[name-defined]
            if name:
                actuator_names[i] = name

        if run_open:
            selected_open = select_trials(baseline_summary_csv, gesture="open", trial_count=int(args.trial_count), min_frames=int(args.min_frames))
            open_paths = [Path(item) for item in selected_open["source_csv"].tolist()]
            per_trial_df, step_df, best_summary, final_compare = run_one_at_a_time_sweep(
                open_paths,
                gesture_label="open",
                base_args=base_args,
                model=m_raw,
                data=d_raw,
                actuator_names=actuator_names,
                qvel_values=parse_float_list(args.open_qvel_values),
                delta_u_values=parse_float_list(args.open_delta_u_values),
                prior_values=parse_float_list(args.open_prior_values),
                horizon_sizes=parse_int_list(args.open_horizon_sizes),
                horizon_iters_values=parse_int_list(args.open_horizon_iters),
            )
            results["open"] = {
                "selected_trials": selected_open,
                "per_trial": per_trial_df,
                "step_summary": step_df,
                "best_summary": best_summary,
                "final_compare": final_compare,
            }
            selected_open.to_csv(output_dir / "open_selected_trials.csv", index=False)
            per_trial_df.to_csv(output_dir / "open_sweep_trials.csv", index=False)
            step_df.to_csv(output_dir / "open_sweep_summary.csv", index=False)
            final_compare.to_csv(output_dir / "open_final_compare.csv", index=False)
            (output_dir / "open_best_config.json").write_text(json.dumps(best_summary, indent=2, ensure_ascii=False), encoding="utf-8")

        if run_fist:
            selected_fist = select_trials(baseline_summary_csv, gesture="fist", trial_count=int(args.trial_count), min_frames=int(args.min_frames))
            fist_paths = [Path(item) for item in selected_fist["source_csv"].tolist()]
            per_trial_df, step_df, best_summary, final_compare = run_one_at_a_time_sweep(
                fist_paths,
                gesture_label="fist",
                base_args=base_args,
                model=m_raw,
                data=d_raw,
                actuator_names=actuator_names,
                qvel_values=parse_float_list(args.fist_qvel_values),
                delta_u_values=parse_float_list(args.fist_delta_u_values),
                prior_values=parse_float_list(args.fist_prior_values),
                horizon_sizes=parse_int_list(args.fist_horizon_sizes),
                horizon_iters_values=parse_int_list(args.fist_horizon_iters),
            )
            results["fist"] = {
                "selected_trials": selected_fist,
                "per_trial": per_trial_df,
                "step_summary": step_df,
                "best_summary": best_summary,
                "final_compare": final_compare,
            }
            selected_fist.to_csv(output_dir / "fist_selected_trials.csv", index=False)
            per_trial_df.to_csv(output_dir / "fist_sweep_trials.csv", index=False)
            step_df.to_csv(output_dir / "fist_sweep_summary.csv", index=False)
            final_compare.to_csv(output_dir / "fist_final_compare.csv", index=False)
            (output_dir / "fist_best_config.json").write_text(json.dumps(best_summary, indent=2, ensure_ascii=False), encoding="utf-8")

        combined_rows = []
        for gesture, payload in results.items():
            row = dict(payload["best_summary"])
            row["gesture"] = gesture
            combined_rows.append(row)
        if combined_rows:
            pd.DataFrame(combined_rows).to_csv(output_dir / "best_configs_combined.csv", index=False)
        write_report(output_dir, results)
        print(f"[INFO] Wrote sweep outputs to {output_dir}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    import mujoco  # local import to avoid unused import until main

    raise SystemExit(main())
