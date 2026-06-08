from __future__ import annotations

"""Build a supervised realtime-ctrl dataset from offline teacher controls.

Offline heavy steps are reused here to produce `ctrl_ref` targets. The dataset
is intended to train a lightweight realtime predictor, not to run inverse
dynamics in the realtime loop.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import DEFAULT_MYOSUITE_ENV_ID, _get_raw_model_data, create_probe_env  # noqa: E402
from scripts.build_teacher_from_trajectory_inverse import run_trajectory_teacher_trial  # noqa: E402
from scripts.build_teacher_inverse_dynamics import dataframe_to_markdown_simple, discover_recording_csvs, load_recorded_trial  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "realtime_ctrl_teacher_dataset"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "realtime_ctrl_teacher_dataset_summary.md"

GESTURE_SHORT_HORIZON_DEFAULTS = {
    "open": {
        "rollout_qvel_weight": 0.5,
        "rollout_delta_u_weight": 0.05,
        "rollout_prior_weight": 0.05,
        "horizon_size": 3,
        "horizon_iters": 3,
        "commit_frames": 1,
    },
    "fist": {
        "rollout_qvel_weight": 1.0,
        "rollout_delta_u_weight": 0.02,
        "rollout_prior_weight": 0.02,
        "horizon_size": 5,
        "horizon_iters": 3,
        "commit_frames": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a realtime ctrl teacher dataset from offline trajectory-based teacher controls.")
    parser.add_argument("--input-path", type=str, default="records", help="Recording CSV or records directory.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for dataset CSVs.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown report path.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id.")
    parser.add_argument("--gesture-filter", type=str, default="", help="Optional comma-separated gestures to include.")
    parser.add_argument("--only-kept-trials", action="store_true", help="Use only trials that pass the teacher quality filter for train/val/test splits.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on number of trials.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on frames per trial before teacher generation.")
    parser.add_argument("--history-length", type=int, default=3, help="Number of previous frames to embed in each training row.")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for trial split shuffling.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio at the trial level.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio at the trial level.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio at the trial level.")
    parser.add_argument("--use-gesture-defaults", action="store_true", help="Use gesture-tuned short-horizon defaults discovered from the open/fist sweeps.")
    parser.add_argument("--teacher-mode", type=str, default="short_horizon_rollout", choices=("dynamic_only", "dynamic_with_static_prior", "dynamic_with_fallback", "short_horizon_rollout"), help="Offline teacher mode used to generate ctrl_target.")
    parser.add_argument("--teacher-prior-mode", type=str, default="static_qpos", choices=("zero", "static_qpos"), help="Static prior source for the teacher solve.")
    parser.add_argument("--timestep", type=float, default=1.0 / 60.0, help="Fallback timestep when timestamps are missing.")
    parser.add_argument("--qpos-smoothing-window", type=int, default=5, help="qpos smoothing window before finite differencing.")
    parser.add_argument("--qvel-smoothing-window", type=int, default=3, help="qvel smoothing window.")
    parser.add_argument("--qacc-smoothing-window", type=int, default=3, help="qacc smoothing window.")
    parser.add_argument("--qvel-clip", type=float, default=0.0, help="Optional qvel clip.")
    parser.add_argument("--qacc-clip", type=float, default=0.0, help="Optional qacc clip.")
    parser.add_argument("--tau-clip", type=float, default=0.0, help="Optional inverse-dynamics tau clip.")
    parser.add_argument("--use-raw-qpos-target", action="store_true", help="Use raw qpos target instead of smoothed qpos target.")
    parser.add_argument("--ctrl-lb", type=float, default=0.0, help="Lower bound for ctrl target.")
    parser.add_argument("--ctrl-ub", type=float, default=1.0, help="Upper bound for ctrl target.")
    parser.add_argument("--qp-reg", type=float, default=1e-3, help="Actuator solve regularization.")
    parser.add_argument("--prev-ctrl-weight", type=float, default=5e-2, help="Prev-ctrl penalty for framewise teacher solve.")
    parser.add_argument("--static-prior-weight", type=float, default=1e-1, help="Static prior penalty for framewise teacher solve.")
    parser.add_argument("--prior-value", type=float, default=0.0, help="Fallback prior value.")
    parser.add_argument("--disable-constraints", action="store_true", help="Disable MuJoCo constraints during inverse dynamics.")
    parser.add_argument("--horizon-size", type=int, default=5, help="Short-horizon rollout size.")
    parser.add_argument("--horizon-iters", type=int, default=5, help="Short-horizon local-optimization iterations.")
    parser.add_argument("--commit-frames", type=int, default=1, help="Receding-horizon commit size.")
    parser.add_argument("--rollout-qpos-weight", type=float, default=1.0, help="Rollout qpos objective weight.")
    parser.add_argument("--rollout-qvel-weight", type=float, default=0.25, help="Rollout qvel objective weight.")
    parser.add_argument("--rollout-u-weight", type=float, default=0.01, help="Rollout ctrl magnitude penalty.")
    parser.add_argument("--rollout-delta-u-weight", type=float, default=0.10, help="Rollout ctrl smoothness penalty.")
    parser.add_argument("--rollout-prior-weight", type=float, default=0.10, help="Rollout prior-closeness penalty.")
    parser.add_argument("--max-substeps-per-frame", type=int, default=150, help="Maximum MuJoCo steps between recorded frames.")
    parser.add_argument("--validation-window-size", type=int, default=0, help="Optional validation chunk size.")
    parser.add_argument("--max-mean-joint-error", type=float, default=0.20, help="Quality filter threshold.")
    parser.add_argument("--max-rmse-forward-error", type=float, default=1.00, help="Quality filter threshold.")
    parser.add_argument("--max-upper-saturation-fraction", type=float, default=0.35, help="Quality filter threshold.")
    parser.add_argument("--max-ctrl-delta-mean", type=float, default=0.25, help="Quality filter threshold.")
    parser.add_argument("--suspicious-frame-error", type=float, default=0.75, help="Debug export threshold.")
    parser.add_argument("--suspicious-qacc-norm", type=float, default=5.0, help="Debug export threshold.")
    parser.add_argument("--suspicious-tau-norm", type=float, default=2.0, help="Debug export threshold.")
    return parser.parse_args()


def gesture_filter_set(raw_value: str) -> set[str]:
    return {item.strip().lower() for item in str(raw_value).split(",") if item.strip()}


def build_teacher_args(args: argparse.Namespace, gesture: str) -> argparse.Namespace:
    payload = dict(vars(args))
    if bool(args.use_gesture_defaults):
        overrides = GESTURE_SHORT_HORIZON_DEFAULTS.get(str(gesture).strip().lower(), {})
        payload.update(overrides)
    return SimpleNamespace(**payload)


def actuator_names_from_model(model: Any) -> list[str]:
    m_raw, _ = _get_raw_model_data(model, None)
    return [
        str(mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) or f"actuator_{idx}")
        for idx in range(int(m_raw.nu))
    ]


def ctrl_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols: list[tuple[int, str]] = []
    for name in df.columns:
        label = str(name)
        if not label.startswith(prefix):
            continue
        try:
            idx = int(label[len(prefix) :])
        except ValueError:
            continue
        cols.append((idx, label))
    cols.sort(key=lambda item: item[0])
    return [label for _, label in cols]


def attach_history_rows(df: pd.DataFrame, *, history_length: int) -> pd.DataFrame:
    out = df.copy()
    history_length = max(0, int(history_length))
    if history_length <= 0 or out.empty:
        return out
    qpos_cols = ctrl_columns(out, "qpos_")
    qvel_cols = ctrl_columns(out, "qvel_")
    target_ctrl_cols = ctrl_columns(out, "ctrl_target_")
    grouped = out.groupby("trial_id", sort=False)
    history_frames: list[pd.DataFrame] = []
    for h in range(1, history_length + 1):
        chunk_cols: dict[str, pd.Series] = {}
        for source_col in qpos_cols:
            idx = source_col.split("_", 1)[1]
            chunk_cols[f"qpos_hist{h}_{idx}"] = grouped[source_col].shift(h).fillna(0.0)
        for source_col in qvel_cols:
            idx = source_col.split("_", 1)[1]
            chunk_cols[f"qvel_hist{h}_{idx}"] = grouped[source_col].shift(h).fillna(0.0)
        for source_col in target_ctrl_cols:
            idx = source_col.split("_", 2)[2]
            chunk_cols[f"ctrl_hist{h}_{idx}"] = grouped[source_col].shift(h).fillna(0.0)
        history_frames.append(pd.DataFrame(chunk_cols, index=out.index))
    if not history_frames:
        return out
    return pd.concat([out] + history_frames, axis=1)


def frame_table_to_dataset_rows(
    frame_table: pd.DataFrame,
    *,
    quality: dict[str, Any],
    teacher_mode_used: str,
) -> pd.DataFrame:
    qpos_cols = ctrl_columns(frame_table, "qpos_target_")
    qvel_cols = ctrl_columns(frame_table, "qvel_traj_")
    ctrl_cols = ctrl_columns(frame_table, "ctrl_teacher_")
    if not qpos_cols or not qvel_cols or not ctrl_cols:
        raise ValueError("Teacher frame table is missing qpos/qvel/ctrl columns.")
    row_count = int(len(frame_table))
    result_payload: dict[str, Any] = {
        "source_csv": np.full(row_count, str(quality["source_csv"]), dtype=object),
        "exercise_name": np.full(row_count, str(quality["exercise_name"]), dtype=object),
        "trial_id": np.full(row_count, str(quality["trial_id"]), dtype=object),
        "frame_index": frame_table["frame_index"].astype(int).to_numpy(),
        "time_from_start": pd.to_numeric(frame_table["time_from_start"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        "teacher_mode_used": np.full(row_count, str(teacher_mode_used), dtype=object),
        "quality_score": np.full(row_count, float(quality.get("quality_score", 0.0)), dtype=float),
        "keep_for_training": np.full(row_count, bool(quality.get("keep_for_training", False)), dtype=bool),
    }
    qpos_arr = frame_table.loc[:, qpos_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    qvel_arr = frame_table.loc[:, qvel_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for idx in range(qpos_arr.shape[1]):
        result_payload[f"qpos_{idx}"] = qpos_arr[:, idx]
    for idx in range(qvel_arr.shape[1]):
        result_payload[f"qvel_{idx}"] = qvel_arr[:, idx]
    ctrl_arr = frame_table.loc[:, ctrl_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    prev_ctrl = np.vstack([np.zeros((1, ctrl_arr.shape[1]), dtype=float), ctrl_arr[:-1]]) if ctrl_arr.shape[0] else np.zeros_like(ctrl_arr)
    for idx in range(ctrl_arr.shape[1]):
        result_payload[f"prev_ctrl_{idx}"] = prev_ctrl[:, idx]
        result_payload[f"ctrl_target_{idx}"] = ctrl_arr[:, idx]
    return pd.DataFrame(result_payload)


def split_trials(
    dataset_df: pd.DataFrame,
    *,
    only_kept_trials: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    working = dataset_df.loc[dataset_df["keep_for_training"] == True].copy() if only_kept_trials else dataset_df.copy()  # noqa: E712
    rng = random.Random(int(seed))
    assignments: dict[str, str] = {}
    split_summary: dict[str, Any] = {}
    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if total_ratio <= 0.0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio = float(train_ratio) / total_ratio
    val_ratio = float(val_ratio) / total_ratio
    for gesture, gesture_df in working.groupby("exercise_name"):
        trial_ids = sorted({str(item) for item in gesture_df["trial_id"].tolist()})
        rng.shuffle(trial_ids)
        n = len(trial_ids)
        n_train = min(n, max(1 if n > 0 else 0, int(round(n * train_ratio))))
        n_val = min(max(0, n - n_train), int(round(n * val_ratio)))
        if (n_train + n_val) > n:
            n_val = max(0, n - n_train)
        train_ids = trial_ids[:n_train]
        val_ids = trial_ids[n_train : n_train + n_val]
        test_ids = trial_ids[n_train + n_val :]
        for item in train_ids:
            assignments[item] = "train"
        for item in val_ids:
            assignments[item] = "val"
        for item in test_ids:
            assignments[item] = "test"
        split_summary[str(gesture)] = {
            "num_trials_total": int(n),
            "num_trials_train": int(len(train_ids)),
            "num_trials_val": int(len(val_ids)),
            "num_trials_test": int(len(test_ids)),
        }
    working["split"] = working["trial_id"].map(assignments).fillna("test")
    train_df = working.loc[working["split"] == "train"].copy()
    val_df = working.loc[working["split"] == "val"].copy()
    test_df = working.loc[working["split"] == "test"].copy()
    split_summary["global"] = {
        "num_rows_total": int(len(working)),
        "num_rows_train": int(len(train_df)),
        "num_rows_val": int(len(val_df)),
        "num_rows_test": int(len(test_df)),
        "only_kept_trials": bool(only_kept_trials),
    }
    return train_df, val_df, test_df, split_summary


def build_report(
    report_path: Path,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    dataset_df: pd.DataFrame,
    trial_quality_df: pd.DataFrame,
    split_summary: dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    kept = trial_quality_df.loc[trial_quality_df["keep_for_training"] == True].copy()  # noqa: E712
    lines = [
        "# Realtime Ctrl Teacher Dataset Summary",
        "",
        "This dataset is built offline from model-derived teacher controls. It is intended to train a lightweight realtime ctrl predictor.",
        "",
        "## Configuration",
        "",
        f"- Input path: `{Path(args.input_path).resolve()}`",
        f"- Teacher mode: `{args.teacher_mode}`",
        f"- Use gesture defaults: `{bool(args.use_gesture_defaults)}`",
        f"- History length: `{int(args.history_length)}`",
        f"- Only kept trials in splits: `{bool(args.only_kept_trials)}`",
        "",
        "## Dataset Size",
        "",
        f"- Rows (all): `{len(dataset_df)}`",
        f"- Trials (all): `{trial_quality_df['trial_id'].nunique() if not trial_quality_df.empty else 0}`",
        f"- Trials kept: `{len(kept)}`",
        "",
        "## Split Summary",
        "",
        "```json",
        json.dumps(split_summary, indent=2),
        "```",
        "",
    ]
    if not trial_quality_df.empty:
        best = trial_quality_df.sort_values(by=["quality_score"], ascending=False).head(10)
        lines.extend(
            [
                "## Best Teacher Trials",
                "",
                dataframe_to_markdown_simple(best[["trial_id", "exercise_name", "quality_score", "mean_joint_error", "rmse_forward_error"]]),
                "",
            ]
        )
    lines.extend(
        [
            "## Output Files",
            "",
            f"- Combined dataset: `{output_dir / 'dataset.csv'}`",
            f"- Train split: `{output_dir / 'train.csv'}`",
            f"- Validation split: `{output_dir / 'val.csv'}`",
            f"- Test split: `{output_dir / 'test.csv'}`",
            f"- Trial quality: `{output_dir / 'trial_quality_summary.csv'}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_gestures = gesture_filter_set(args.gesture_filter)
    record_paths = discover_recording_csvs(input_path)
    if selected_gestures:
        record_paths = [path for path in record_paths if any(part.lower() in selected_gestures for part in path.parts)]
    if int(args.max_trials) > 0:
        record_paths = record_paths[: int(args.max_trials)]
    if not record_paths:
        raise SystemExit("No recording CSVs matched the requested dataset build.")

    env, model, data = create_probe_env(args.env_id)
    _ = env
    actuator_names = actuator_names_from_model(model)
    trial_rows: list[pd.DataFrame] = []
    trial_quality_rows: list[dict[str, Any]] = []
    for path in record_paths:
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
        teacher_args = build_teacher_args(args, str(trial["exercise_name"]))
        frame_table, quality, _validation = run_trajectory_teacher_trial(
            trial,
            model=model,
            data=data,
            actuator_names=actuator_names,
            args=teacher_args,
        )
        quality["teacher_mode_used"] = str(teacher_args.teacher_mode)
        trial_quality_rows.append(dict(quality))
        dataset_rows = frame_table_to_dataset_rows(
            frame_table,
            quality=quality,
            teacher_mode_used=str(teacher_args.teacher_mode),
        )
        trial_rows.append(dataset_rows)

    if not trial_rows:
        raise SystemExit("Teacher dataset build produced no rows.")

    dataset_df = pd.concat(trial_rows, axis=0, ignore_index=True)
    dataset_df = attach_history_rows(dataset_df, history_length=int(args.history_length))
    trial_quality_df = pd.DataFrame(trial_quality_rows)
    train_df, val_df, test_df, split_summary = split_trials(
        dataset_df,
        only_kept_trials=bool(args.only_kept_trials),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.random_seed),
    )

    dataset_df.to_csv(output_dir / "dataset.csv", index=False)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    trial_quality_df.to_csv(output_dir / "trial_quality_summary.csv", index=False)
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    build_report(
        Path(args.report_path).resolve(),
        args=args,
        output_dir=output_dir,
        dataset_df=dataset_df,
        trial_quality_df=trial_quality_df,
        split_summary=split_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
