from __future__ import annotations

"""Compare tracking-style and trajectory-inverse teacher pipelines offline."""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_teacher_inverse_dynamics import dataframe_to_markdown_simple  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "teacher_mode_comparison"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "trajectory_inverse_teacher_summary.md"


def load_summary_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ("trial_id", "exercise_name", "source_csv"):
        if col not in df.columns:
            raise ValueError(f"Required column missing in {path}: {col}")
    return df


def load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _parse_rmse_list(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [float(x) for x in parsed]
    except Exception:
        return []
    return []


def add_derived_metrics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    per_joint_col = "per_joint_rmse"
    if per_joint_col in out.columns:
        out[f"{prefix}max_per_joint_rmse"] = [
            float(max(_parse_rmse_list(value))) if _parse_rmse_list(value) else 0.0
            for value in out[per_joint_col].tolist()
        ]
    else:
        out[f"{prefix}max_per_joint_rmse"] = 0.0
    return out


def merge_mode_tables(
    tracking_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    tracking_replay_df: pd.DataFrame | None,
    trajectory_replay_df: pd.DataFrame | None,
    tracking_consistency_df: pd.DataFrame | None,
    trajectory_consistency_df: pd.DataFrame | None,
) -> pd.DataFrame:
    base_cols = ["trial_id", "exercise_name", "source_csv"]
    trk = add_derived_metrics(tracking_df, "tracking_").copy()
    traj = add_derived_metrics(trajectory_df, "trajectory_").copy()
    merged = trk.merge(traj, on=base_cols, suffixes=("_tracking", "_trajectory"), how="inner")

    if tracking_replay_df is not None:
        merged = merged.merge(
            tracking_replay_df.rename(
                columns={
                    "mean_joint_error": "replay_mean_joint_error_tracking",
                    "rmse_forward_error": "replay_rmse_forward_error_tracking",
                    "mean_pose_error_norm": "replay_mean_pose_error_norm_tracking",
                    "largest_error_joint": "largest_error_joint_tracking",
                    "largest_error_joint_rmse": "largest_error_joint_rmse_tracking",
                }
            )[["trial_id", "exercise_name", "source_csv", "replay_mean_joint_error_tracking", "replay_rmse_forward_error_tracking", "replay_mean_pose_error_norm_tracking", "largest_error_joint_tracking", "largest_error_joint_rmse_tracking"]],
            on=base_cols,
            how="left",
        )
    if trajectory_replay_df is not None:
        merged = merged.merge(
            trajectory_replay_df.rename(
                columns={
                    "mean_joint_error": "replay_mean_joint_error_trajectory",
                    "rmse_forward_error": "replay_rmse_forward_error_trajectory",
                    "mean_pose_error_norm": "replay_mean_pose_error_norm_trajectory",
                    "largest_error_joint": "largest_error_joint_trajectory",
                    "largest_error_joint_rmse": "largest_error_joint_rmse_trajectory",
                }
            )[["trial_id", "exercise_name", "source_csv", "replay_mean_joint_error_trajectory", "replay_rmse_forward_error_trajectory", "replay_mean_pose_error_norm_trajectory", "largest_error_joint_trajectory", "largest_error_joint_rmse_trajectory"]],
            on=base_cols,
            how="left",
        )
    if tracking_consistency_df is not None:
        merged = merged.merge(
            tracking_consistency_df.rename(
                columns={
                    "mean_tau_residual_norm": "mean_tau_residual_norm_tracking",
                    "mean_tau_relative_residual": "mean_tau_relative_residual_tracking",
                    "mean_qacc_error_norm": "mean_qacc_error_norm_tracking",
                }
            )[["trial_id", "exercise_name", "source_csv", "mean_tau_residual_norm_tracking", "mean_tau_relative_residual_tracking", "mean_qacc_error_norm_tracking"]],
            on=base_cols,
            how="left",
        )
    if trajectory_consistency_df is not None:
        merged = merged.merge(
            trajectory_consistency_df.rename(
                columns={
                    "mean_tau_residual_norm": "mean_tau_residual_norm_trajectory",
                    "mean_tau_relative_residual": "mean_tau_relative_residual_trajectory",
                    "mean_qacc_error_norm": "mean_qacc_error_norm_trajectory",
                }
            )[["trial_id", "exercise_name", "source_csv", "mean_tau_residual_norm_trajectory", "mean_tau_relative_residual_trajectory", "mean_qacc_error_norm_trajectory"]],
            on=base_cols,
            how="left",
        )

    for metric in (
        "mean_joint_error",
        "rmse_forward_error",
        "ctrl_delta_mean",
        "upper_saturation_fraction",
        "max_per_joint_rmse",
        "replay_mean_joint_error",
        "replay_rmse_forward_error",
        "mean_tau_relative_residual",
        "mean_qacc_error_norm",
    ):
        left = f"{metric}_tracking"
        right = f"{metric}_trajectory"
        if left in merged.columns and right in merged.columns:
            merged[f"{metric}_delta"] = merged[left].astype(float) - merged[right].astype(float)
    return merged


def write_report(
    report_path: Path,
    *,
    merged_df: pd.DataFrame,
    tracking_summary_path: Path,
    trajectory_summary_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Trajectory Inverse Teacher Summary",
        "",
        "This report compares two offline teacher-control pipelines:",
        "- tracking-style teacher: qacc_des from qpos tracking",
        "- pure trajectory inverse teacher: qvel/qacc derived directly from the recorded trajectory",
        "",
        "Both are model-derived teacher references for MyoHand, not biological ground truth.",
        "",
        "## Inputs",
        "",
        f"- Tracking-style summary: `{tracking_summary_path}`",
        f"- Trajectory-inverse summary: `{trajectory_summary_path}`",
        "",
    ]
    if merged_df.empty:
        lines.append("No overlapping trials were found.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    teacher_better_joint = int((merged_df["mean_joint_error_delta"] > 0).sum()) if "mean_joint_error_delta" in merged_df.columns else 0
    teacher_better_rmse = int((merged_df["rmse_forward_error_delta"] > 0).sum()) if "rmse_forward_error_delta" in merged_df.columns else 0
    replay_better = int((merged_df["replay_rmse_forward_error_delta"] > 0).sum()) if "replay_rmse_forward_error_delta" in merged_df.columns else 0
    lines.extend(
        [
            "## Overall Comparison",
            "",
            f"- Overlapping trials: `{len(merged_df)}`",
            f"- Trajectory-inverse better in build-time mean joint error: `{teacher_better_joint}` trials",
            f"- Trajectory-inverse better in build-time RMSE forward error: `{teacher_better_rmse}` trials",
            f"- Trajectory-inverse better in replay RMSE: `{replay_better}` trials",
            "",
        ]
    )
    for gesture, group in merged_df.groupby("exercise_name"):
        lines.append(
            f"- `{gesture}`: "
            f"mean joint delta (tracking - trajectory) `{float(group['mean_joint_error_delta'].mean()):.4f}` | "
            f"RMSE delta `{float(group['rmse_forward_error_delta'].mean()):.4f}`"
        )
    best = merged_df.sort_values(by=["rmse_forward_error_delta"], ascending=False).head(10)
    lines.extend(
        [
            "",
            "## Trials Where Trajectory-Inverse Most Improved RMSE",
            "",
            dataframe_to_markdown_simple(
                best[
                    [
                        "trial_id",
                        "exercise_name",
                        "rmse_forward_error_tracking",
                        "rmse_forward_error_trajectory",
                        "rmse_forward_error_delta",
                        "mean_joint_error_tracking",
                        "mean_joint_error_trajectory",
                    ]
                ]
            ),
            "",
            "## Interpretation",
            "",
            "- The trajectory-inverse pipeline follows the recorded motion directly through qvel/qacc estimated from the trajectory.",
            "- The tracking-style pipeline remains useful as a baseline, but it injects controller-style acceleration targets.",
            "- Replay metrics and mj_forward consistency should be used together when judging whether the resulting `u_teacher` is meaningful enough for later training.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tracking-style and trajectory-inverse offline teacher pipelines.")
    parser.add_argument("--tracking-summary-csv", type=str, required=True, help="teacher_trial_summary.csv from the tracking-style teacher output.")
    parser.add_argument("--trajectory-summary-csv", type=str, required=True, help="teacher_trial_summary.csv from the trajectory-inverse teacher output.")
    parser.add_argument("--tracking-replay-summary-csv", type=str, default="", help="Optional replay summary CSV for the tracking-style teacher.")
    parser.add_argument("--trajectory-replay-summary-csv", type=str, default="", help="Optional replay summary CSV for the trajectory-inverse teacher.")
    parser.add_argument("--tracking-consistency-summary-csv", type=str, default="", help="Optional mj_forward consistency summary CSV for the tracking-style teacher.")
    parser.add_argument("--trajectory-consistency-summary-csv", type=str, default="", help="Optional mj_forward consistency summary CSV for the trajectory-inverse teacher.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for merged comparison outputs.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_summary_path = Path(args.tracking_summary_csv).resolve()
    trajectory_summary_path = Path(args.trajectory_summary_csv).resolve()
    tracking_df = load_summary_csv(tracking_summary_path)
    trajectory_df = load_summary_csv(trajectory_summary_path)
    tracking_replay_df = load_optional_csv(Path(args.tracking_replay_summary_csv).resolve()) if str(args.tracking_replay_summary_csv).strip() else None
    trajectory_replay_df = load_optional_csv(Path(args.trajectory_replay_summary_csv).resolve()) if str(args.trajectory_replay_summary_csv).strip() else None
    tracking_consistency_df = load_optional_csv(Path(args.tracking_consistency_summary_csv).resolve()) if str(args.tracking_consistency_summary_csv).strip() else None
    trajectory_consistency_df = load_optional_csv(Path(args.trajectory_consistency_summary_csv).resolve()) if str(args.trajectory_consistency_summary_csv).strip() else None

    merged_df = merge_mode_tables(
        tracking_df,
        trajectory_df,
        tracking_replay_df,
        trajectory_replay_df,
        tracking_consistency_df,
        trajectory_consistency_df,
    )
    merged_csv = output_dir / "teacher_mode_comparison.csv"
    merged_json = output_dir / "teacher_mode_comparison.json"
    merged_df.to_csv(merged_csv, index=False)
    merged_json.write_text(
        json.dumps(
            {
                "num_trials_compared": int(len(merged_df)),
                "trials": merged_df.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report(
        report_path,
        merged_df=merged_df,
        tracking_summary_path=tracking_summary_path,
        trajectory_summary_path=trajectory_summary_path,
    )
    print(f"[INFO] Wrote: {merged_csv}")
    print(f"[INFO] Wrote: {merged_json}")
    print(f"[INFO] Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
