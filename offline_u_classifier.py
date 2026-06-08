#!/usr/bin/env python3
"""Offline classifier for validating muscle-activation representations.

This script uses simple offline classification as a representation check, not
as the final goal. If the muscle-activation signal `u` separates gestures with
simple trial-level features and a linear classifier, that is evidence that `u`
contains meaningful structure.

The pipeline focuses on the terminal hold segment because that is the most
stable part of each gesture. Unstable hold segments inject noise into the
feature vector, which can make the confusion matrix less reliable and obscure
whether `u` is actually a useful representation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CSV_GLOB = "ctrl_est_*.csv"
DEFAULT_HOLD_TAIL_RATIO = 0.40
DEFAULT_HOLD_FRAME_CAP = 20
DEFAULT_MAX_HOLD_STD = 0.08
DEFAULT_MAX_HOLD_DELTA = 0.12
DEFAULT_MIN_HOLD_ACTIVATION = 0.05
CLASS_ORDER = ("open", "fist", "claw")
DEFAULT_EDGE_ACTIVE_THRESHOLD = 0.02
PROPOSAL_FILTERED_MUSCLES = (
    "FDS2",
    "FDS3",
    "FDS4",
    "FDS5",
    "FDP2",
    "FDP3",
    "FDP4",
    "FDP5",
    "EDC2",
    "EDC3",
    "EDC4",
    "EDC5",
    "EIP",
    "EDM",
    "FPL",
    "EPL",
    "EPB",
    "APL",
    "OP",
)
PROPOSAL_FILTERED_CLASSES = ("open", "fist", "claw", "thumb", "index", "middle")
PROPOSAL_CORE_FILENAMES = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "summary_node_only.json",
    "summary_node_edge.json",
    "muscle_profile_per_class_filtered.csv",
    "top_muscles_per_class_filtered.csv",
    "graph_edges_open.csv",
    "graph_edges_fist.csv",
    "graph_edges_claw.csv",
    "graph_edges_thumb.csv",
    "graph_edges_index.csv",
    "graph_edges_middle.csv",
    "trial_quality.csv",
)

# Local fallback graph metadata mirrors the runtime graph enough for offline
# interpretation when importing the full realtime stack is not practical.
FALLBACK_MUSCLE_NAMES = [
    "ECRL",
    "ECRB",
    "ECU",
    "FCR",
    "FCU",
    "PL",
    "PT",
    "PQ",
    "FDS5",
    "FDS4",
    "FDS3",
    "FDS2",
    "FDP5",
    "FDP4",
    "FDP3",
    "FDP2",
    "EDC5",
    "EDC4",
    "EDC3",
    "EDC2",
    "EDM",
    "EIP",
    "EPL",
    "EPB",
    "FPL",
    "APL",
    "OP",
    "RI2",
    "LU_RB2",
    "UI_UB2",
    "RI3",
    "LU_RB3",
    "UI_UB3",
    "RI4",
    "LU_RB4",
    "UI_UB4",
    "RI5",
    "LU_RB5",
    "UI_UB5",
]
GRAPH_EXCLUDED_NODE_NAMES = frozenset(
    {
        "RI2",
        "RI3",
        "RI4",
        "RI5",
        "LU_RB2",
        "LU_RB3",
        "LU_RB4",
        "LU_RB5",
        "UI_UB2",
        "UI_UB3",
        "UI_UB4",
        "UI_UB5",
    }
)
FUNCTIONAL_GROUPS = {
    "wrist_extensors": ("ECRL", "ECRB", "ECU"),
    "wrist_flexors": ("FCR", "FCU", "PL"),
    "pronators": ("PT", "PQ"),
    "thumb": ("EPL", "EPB", "FPL", "APL", "OP"),
    "digit2_flexors": ("FDS2", "FDP2"),
    "digit3_flexors": ("FDS3", "FDP3"),
    "digit4_flexors": ("FDS4", "FDP4"),
    "digit5_flexors": ("FDS5", "FDP5"),
    "digit2_extensors": ("EDC2", "EIP"),
    "digit3_extensors": ("EDC3",),
    "digit4_extensors": ("EDC4",),
    "digit5_extensors": ("EDC5", "EDM"),
    "digit2_intrinsics": ("RI2", "LU_RB2", "UI_UB2"),
    "digit3_intrinsics": ("RI3", "LU_RB3", "UI_UB3"),
    "digit4_intrinsics": ("RI4", "LU_RB4", "UI_UB4"),
    "digit5_intrinsics": ("RI5", "LU_RB5", "UI_UB5"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline validation pipeline for muscle-activation (u) recordings."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing ctrl_est_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where outputs will be written. Default: <input_dir>/offline_u_classifier_output",
    )
    parser.add_argument(
        "--hold-tail-ratio",
        type=float,
        default=DEFAULT_HOLD_TAIL_RATIO,
        help="Use the last fraction of frames as the hold segment candidate window.",
    )
    parser.add_argument(
        "--hold-frame-cap",
        type=int,
        default=DEFAULT_HOLD_FRAME_CAP,
        help="Cap the terminal hold window to this many frames.",
    )
    parser.add_argument(
        "--enable-quality-filter",
        action="store_true",
        help="Discard trials whose hold segment is too unstable before classification.",
    )
    parser.add_argument(
        "--max-hold-std",
        type=float,
        default=DEFAULT_MAX_HOLD_STD,
        help="Maximum allowed mean per-muscle std inside the hold segment.",
    )
    parser.add_argument(
        "--max-hold-delta",
        type=float,
        default=DEFAULT_MAX_HOLD_DELTA,
        help="Maximum allowed mean frame-to-frame L2 delta inside the hold segment.",
    )
    parser.add_argument(
        "--min-hold-activation",
        type=float,
        default=DEFAULT_MIN_HOLD_ACTIVATION,
        help="Minimum required max activation inside the hold segment before marking the trial as weak.",
    )
    parser.add_argument(
        "--edge-active-threshold",
        type=float,
        default=DEFAULT_EDGE_ACTIVE_THRESHOLD,
        help="Threshold used when counting active edge coactivation summaries.",
    )
    return parser.parse_args()


def normalize_label(label: Any) -> str:
    value = str(label or "").strip().lower()
    if value == "claw_stretch":
        return "claw"
    return value


def sort_signal_columns(columns: list[str], prefix: str) -> list[str]:
    selected = [col for col in columns if str(col).startswith(prefix)]

    def sort_key(name: str) -> tuple[int, str]:
        suffix = str(name).split(prefix, 1)[-1]
        try:
            return (int(suffix), str(name))
        except ValueError:
            return (10**9, str(name))

    return sorted(selected, key=sort_key)


def find_ctrl_analysis_columns(columns: list[str]) -> list[str]:
    return sort_signal_columns(columns, "ctrl_analysis_")


def find_qpos_columns(columns: list[str]) -> list[str]:
    return sort_signal_columns(columns, "qpos_")


def sanitize_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def choose_trial_id(trial_df: pd.DataFrame, csv_path: Path) -> str:
    if "trial_id" not in trial_df.columns:
        print(f"[WARN] {csv_path.name}: missing trial_id column, falling back to filename.")
        return csv_path.stem
    series = trial_df["trial_id"].dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        print(f"[WARN] {csv_path.name}: empty trial_id values, falling back to filename.")
        return csv_path.stem
    return str(series.iloc[-1])


def choose_exercise_name(trial_df: pd.DataFrame) -> str:
    if "exercise_name" not in trial_df.columns:
        return ""
    series = trial_df["exercise_name"].dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return ""
    return normalize_label(series.iloc[-1])


def select_hold_segment(
    trial_df: pd.DataFrame,
    *,
    hold_tail_ratio: float,
    hold_frame_cap: int,
) -> pd.DataFrame:
    n_frames = len(trial_df)
    if n_frames == 0:
        return trial_df.iloc[0:0]

    hold_tail_ratio = float(np.clip(hold_tail_ratio, 0.0, 1.0))
    hold_frame_cap = max(1, int(hold_frame_cap))

    # The hold segment is where the gesture should be most stationary. That
    # makes it the right place to summarize `u` for representation validation.
    hold_start = int(np.floor(n_frames * (1.0 - hold_tail_ratio)))
    hold_candidate = trial_df.iloc[hold_start:].copy()
    if hold_candidate.empty:
        hold_candidate = trial_df.copy()
    return hold_candidate.iloc[-hold_frame_cap:].copy()


def compute_frame_delta_metrics(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0, 0.0
    deltas = np.diff(values, axis=0)
    delta_norms = np.linalg.norm(deltas, axis=1)
    if delta_norms.size == 0:
        return 0.0, 0.0
    return float(np.mean(delta_norms)), float(np.max(delta_norms))


def compute_quality_score(
    *,
    hold_ctrl_std_mean: float,
    hold_frame_to_frame_delta_mean: float,
    hold_activation_max: float,
    min_hold_activation: float,
) -> float:
    std_component = 1.0 / (1.0 + max(0.0, float(hold_ctrl_std_mean)))
    delta_component = 1.0 / (1.0 + max(0.0, float(hold_frame_to_frame_delta_mean)))
    denom = max(float(min_hold_activation), 1e-6)
    activation_component = float(np.clip(float(hold_activation_max) / denom, 0.0, 1.0))
    score = 0.45 * std_component + 0.35 * delta_component + 0.20 * activation_component
    return float(np.clip(score, 0.0, 1.0))


def compute_trial_quality_metrics(
    hold_ctrl_df: pd.DataFrame,
    *,
    hold_qpos_df: pd.DataFrame | None,
    max_hold_std: float,
    max_hold_delta: float,
    min_hold_activation: float,
) -> dict[str, Any]:
    ctrl_values = hold_ctrl_df.to_numpy(dtype=float, copy=True)
    std_per_muscle = np.nanstd(ctrl_values, axis=0) if ctrl_values.size else np.zeros(0, dtype=float)
    hold_ctrl_std_mean = float(np.mean(std_per_muscle)) if std_per_muscle.size else 0.0
    hold_ctrl_std_max = float(np.max(std_per_muscle)) if std_per_muscle.size else 0.0
    hold_frame_to_frame_delta_mean, hold_frame_to_frame_delta_max = compute_frame_delta_metrics(ctrl_values)
    hold_activation_mean = float(np.mean(ctrl_values)) if ctrl_values.size else 0.0
    hold_activation_max = float(np.max(ctrl_values)) if ctrl_values.size else 0.0

    qpos_std_mean = np.nan
    qpos_delta_mean = np.nan
    if hold_qpos_df is not None and not hold_qpos_df.empty:
        qpos_values = hold_qpos_df.to_numpy(dtype=float, copy=True)
        qpos_std = np.nanstd(qpos_values, axis=0) if qpos_values.size else np.zeros(0, dtype=float)
        qpos_std_mean = float(np.mean(qpos_std)) if qpos_std.size else 0.0
        qpos_delta_mean, _ = compute_frame_delta_metrics(qpos_values)

    unstable_hold = bool(
        hold_ctrl_std_mean > float(max_hold_std)
        or hold_frame_to_frame_delta_mean > float(max_hold_delta)
    )
    weak_activation = bool(hold_activation_max < float(min_hold_activation))
    quality_score = compute_quality_score(
        hold_ctrl_std_mean=hold_ctrl_std_mean,
        hold_frame_to_frame_delta_mean=hold_frame_to_frame_delta_mean,
        hold_activation_max=hold_activation_max,
        min_hold_activation=min_hold_activation,
    )

    return {
        "hold_ctrl_std_mean": hold_ctrl_std_mean,
        "hold_ctrl_std_max": hold_ctrl_std_max,
        "hold_frame_to_frame_delta_mean": hold_frame_to_frame_delta_mean,
        "hold_frame_to_frame_delta_max": hold_frame_to_frame_delta_max,
        "hold_activation_mean": hold_activation_mean,
        "hold_activation_max": hold_activation_max,
        "qpos_std_mean": qpos_std_mean,
        "qpos_delta_mean": qpos_delta_mean,
        "quality_score": quality_score,
        "unstable_hold": unstable_hold,
        "weak_activation": weak_activation,
        "keep_for_classification": not unstable_hold,
    }


def build_trial_record(
    trial_df: pd.DataFrame,
    *,
    csv_path: Path,
    ctrl_cols: list[str],
    qpos_cols: list[str],
    hold_tail_ratio: float,
    hold_frame_cap: int,
    max_hold_std: float,
    max_hold_delta: float,
    min_hold_activation: float,
) -> dict[str, Any] | None:
    if trial_df.empty:
        return None

    trial_id = choose_trial_id(trial_df, csv_path)
    label = choose_exercise_name(trial_df)
    if not label:
        print(f"[WARN] {csv_path.name}: skipping trial {trial_id!r} because exercise_name is missing.")
        return None

    hold_df = select_hold_segment(
        trial_df,
        hold_tail_ratio=hold_tail_ratio,
        hold_frame_cap=hold_frame_cap,
    )
    if hold_df.empty:
        print(f"[WARN] {csv_path.name}: skipping trial {trial_id!r} because hold segment is empty.")
        return None

    hold_ctrl_df = sanitize_numeric_frame(hold_df.loc[:, ctrl_cols])
    if hold_ctrl_df.empty:
        print(f"[WARN] {csv_path.name}: skipping trial {trial_id!r} because ctrl feature block is empty.")
        return None

    hold_qpos_df = sanitize_numeric_frame(hold_df.loc[:, qpos_cols]) if qpos_cols else None
    ctrl_values = hold_ctrl_df.to_numpy(dtype=float, copy=True)
    mean_vec = np.nanmean(ctrl_values, axis=0)
    std_vec = np.nanstd(ctrl_values, axis=0)
    max_vec = np.nanmax(ctrl_values, axis=0)
    quality_metrics = compute_trial_quality_metrics(
        hold_ctrl_df,
        hold_qpos_df=hold_qpos_df,
        max_hold_std=max_hold_std,
        max_hold_delta=max_hold_delta,
        min_hold_activation=min_hold_activation,
    )

    row: dict[str, Any] = {
        "source_file": csv_path.name,
        "trial_id": trial_id,
        "exercise_name": label,
        "num_total_frames": int(len(trial_df)),
        "num_hold_frames": int(len(hold_ctrl_df)),
    }
    row.update(quality_metrics)
    for col, value in zip(ctrl_cols, mean_vec.tolist()):
        row[f"mean_{col}"] = float(value)
    for col, value in zip(ctrl_cols, std_vec.tolist()):
        row[f"std_{col}"] = float(value)
    for col, value in zip(ctrl_cols, max_vec.tolist()):
        row[f"max_{col}"] = float(value)
    return row


def extract_trial_records(
    input_dir: Path,
    *,
    hold_tail_ratio: float,
    hold_frame_cap: int,
    max_hold_std: float,
    max_hold_delta: float,
    min_hold_activation: float,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    trial_records: list[dict[str, Any]] = []
    ctrl_cols_reference: list[str] = []
    qpos_cols_reference: list[str] = []
    skipped: list[dict[str, str]] = []

    csv_paths = sorted(input_dir.glob(CSV_GLOB))
    if not csv_paths:
        raise FileNotFoundError(f"No files matching {CSV_GLOB!r} were found in {input_dir}.")

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            skipped.append({"file": csv_path.name, "reason": f"read_error: {type(exc).__name__}"})
            print(f"[WARN] {csv_path.name}: failed to read ({type(exc).__name__}: {exc}).")
            continue

        if df.empty:
            skipped.append({"file": csv_path.name, "reason": "empty_csv"})
            print(f"[WARN] {csv_path.name}: skipped because CSV is empty.")
            continue

        ctrl_cols = find_ctrl_analysis_columns(df.columns.tolist())
        if not ctrl_cols:
            skipped.append({"file": csv_path.name, "reason": "missing_ctrl_analysis_columns"})
            print(f"[WARN] {csv_path.name}: skipped because no ctrl_analysis_* columns were found.")
            continue

        qpos_cols = find_qpos_columns(df.columns.tolist())

        if not ctrl_cols_reference:
            ctrl_cols_reference = list(ctrl_cols)
        elif ctrl_cols != ctrl_cols_reference:
            missing = [col for col in ctrl_cols_reference if col not in ctrl_cols]
            extra = [col for col in ctrl_cols if col not in ctrl_cols_reference]
            if missing:
                for col in missing:
                    df[col] = 0.0
            if extra:
                print(f"[WARN] {csv_path.name}: ignoring unexpected ctrl columns: {extra}")
            ctrl_cols = list(ctrl_cols_reference)

        if not qpos_cols_reference:
            qpos_cols_reference = list(qpos_cols)
        elif qpos_cols_reference:
            missing_qpos = [col for col in qpos_cols_reference if col not in qpos_cols]
            for col in missing_qpos:
                df[col] = 0.0
            qpos_cols = list(qpos_cols_reference)

        if "trial_id" in df.columns:
            grouped = df.groupby("trial_id", sort=False, dropna=False)
            trial_dfs = [trial_group.copy() for _, trial_group in grouped]
        else:
            trial_dfs = [df.copy()]

        for trial_df in trial_dfs:
            row = build_trial_record(
                trial_df,
                csv_path=csv_path,
                ctrl_cols=ctrl_cols_reference,
                qpos_cols=qpos_cols_reference,
                hold_tail_ratio=hold_tail_ratio,
                hold_frame_cap=hold_frame_cap,
                max_hold_std=max_hold_std,
                max_hold_delta=max_hold_delta,
                min_hold_activation=min_hold_activation,
            )
            if row is None:
                skipped.append({"file": csv_path.name, "reason": "trial_skipped"})
                continue
            trial_records.append(row)

    return trial_records, ctrl_cols_reference, skipped


def build_trial_dataframe(trial_records: list[dict[str, Any]]) -> pd.DataFrame:
    trials_df = pd.DataFrame(trial_records)
    if trials_df.empty:
        raise ValueError("No valid trial rows were extracted.")
    return trials_df


def feature_columns_from_trials(trials_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in trials_df.columns
        if str(col).startswith(("mean_ctrl_analysis_", "std_ctrl_analysis_", "max_ctrl_analysis_"))
    ]


def resolve_muscle_names(ctrl_cols: list[str]) -> list[str]:
    names: list[str] = []
    for ctrl_col in ctrl_cols:
        suffix = str(ctrl_col).split("ctrl_analysis_", 1)[-1]
        if suffix.isdigit():
            index = int(suffix)
            if 0 <= index < len(FALLBACK_MUSCLE_NAMES):
                names.append(str(FALLBACK_MUSCLE_NAMES[index]))
                continue
        names.append(str(suffix))
    return names


def mean_feature_columns_from_ctrl_cols(ctrl_cols: list[str]) -> list[str]:
    return [f"mean_{col}" for col in ctrl_cols]


def safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def ordered_class_names(labels: list[str] | pd.Series | np.ndarray) -> list[str]:
    seen = {str(label) for label in list(labels)}
    ordered = [name for name in CLASS_ORDER if name in seen]
    extras = sorted(name for name in seen if name not in CLASS_ORDER)
    return [name for name in ordered + extras if name]


def quality_report_columns() -> list[str]:
    return [
        "source_file",
        "trial_id",
        "exercise_name",
        "num_total_frames",
        "num_hold_frames",
        "hold_ctrl_std_mean",
        "hold_ctrl_std_max",
        "hold_frame_to_frame_delta_mean",
        "hold_frame_to_frame_delta_max",
        "hold_activation_mean",
        "hold_activation_max",
        "qpos_std_mean",
        "qpos_delta_mean",
        "quality_score",
        "unstable_hold",
        "weak_activation",
        "keep_for_classification",
    ]


def features_dataframe_from_trials(trials_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    metadata_cols = [
        "source_file",
        "trial_id",
        "exercise_name",
        "num_total_frames",
        "num_hold_frames",
        "quality_score",
        "unstable_hold",
        "weak_activation",
        "keep_for_classification",
    ]
    selected_cols = [col for col in metadata_cols if col in trials_df.columns] + list(feature_cols)
    features_df = trials_df.loc[:, selected_cols].copy()
    if feature_cols:
        features_df.loc[:, feature_cols] = sanitize_numeric_frame(features_df.loc[:, feature_cols])
    return features_df


def build_trial_activation_dataframe(
    trials_df: pd.DataFrame,
    *,
    ctrl_cols: list[str],
    muscle_names: list[str],
) -> pd.DataFrame:
    mean_feature_cols = mean_feature_columns_from_ctrl_cols(ctrl_cols)
    available_pairs = [
        (feature_col, muscle_name)
        for feature_col, muscle_name in zip(mean_feature_cols, muscle_names)
        if feature_col in trials_df.columns
    ]
    metadata_cols = ["source_file", "trial_id", "exercise_name", "num_hold_frames", "quality_score"]
    activation_df = trials_df.loc[:, [col for col in metadata_cols if col in trials_df.columns]].copy()
    for feature_col, muscle_name in available_pairs:
        activation_df[muscle_name] = pd.to_numeric(trials_df[feature_col], errors="coerce").fillna(0.0)
    return activation_df


def build_class_profile_table(
    trial_activation_df: pd.DataFrame,
    *,
    muscle_names: list[str],
) -> pd.DataFrame:
    if trial_activation_df.empty:
        return pd.DataFrame(columns=["muscle"])

    grouped = trial_activation_df.groupby("exercise_name", sort=False)
    class_order = ordered_class_names(trial_activation_df["exercise_name"])
    rows: list[dict[str, Any]] = []
    for muscle_name in muscle_names:
        if muscle_name not in trial_activation_df.columns:
            continue
        row: dict[str, Any] = {"muscle": muscle_name}
        for class_name in class_order:
            series = pd.to_numeric(grouped.get_group(class_name)[muscle_name], errors="coerce") if class_name in grouped.groups else pd.Series(dtype=float)
            row[f"{class_name}_mean"] = float(series.mean()) if not series.empty else np.nan
            row[f"{class_name}_std"] = float(series.std(ddof=0)) if not series.empty else np.nan
        rows.append(row)

    profile_df = pd.DataFrame(rows)
    if profile_df.empty:
        return pd.DataFrame(columns=["muscle"])

    preferred_columns = ["muscle"]
    for class_name in ordered_class_names(trial_activation_df["exercise_name"]):
        preferred_columns.extend([f"{class_name}_mean", f"{class_name}_std"])
    for class_name in CLASS_ORDER:
        for suffix in ("mean", "std"):
            column = f"{class_name}_{suffix}"
            if column not in profile_df.columns:
                profile_df[column] = np.nan
    ordered_columns = ["muscle"]
    for class_name in CLASS_ORDER:
        ordered_columns.extend([f"{class_name}_mean", f"{class_name}_std"])
    extra_columns = [col for col in preferred_columns if col not in ordered_columns]
    return profile_df.loc[:, ordered_columns + extra_columns].copy()


def build_top_muscles_table(
    class_profile_df: pd.DataFrame,
    *,
    class_names: list[str],
    top_k: int = 8,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_name in class_names:
        mean_col = f"{class_name}_mean"
        std_col = f"{class_name}_std"
        if mean_col not in class_profile_df.columns:
            continue
        ranked_df = class_profile_df.loc[:, ["muscle", mean_col] + ([std_col] if std_col in class_profile_df.columns else [])].copy()
        ranked_df[mean_col] = pd.to_numeric(ranked_df[mean_col], errors="coerce").fillna(0.0)
        if std_col in ranked_df.columns:
            ranked_df[std_col] = pd.to_numeric(ranked_df[std_col], errors="coerce").fillna(0.0)
        ranked_df = ranked_df.sort_values(by=mean_col, ascending=False).head(int(top_k)).reset_index(drop=True)
        for rank, (_, muscle_row) in enumerate(ranked_df.iterrows(), start=1):
            rows.append(
                {
                    "class_name": class_name,
                    "rank": rank,
                    "muscle": str(muscle_row["muscle"]),
                    "mean_activation": float(muscle_row[mean_col]),
                    "std_activation": float(muscle_row[std_col]) if std_col in ranked_df.columns else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=["class_name", "rank", "muscle", "mean_activation", "std_activation"])


def print_top_muscles_summary(top_muscles_df: pd.DataFrame, *, top_k: int = 5) -> None:
    if top_muscles_df.empty:
        print("No top-muscle summary available.")
        return

    class_names = ordered_class_names(top_muscles_df["class_name"])
    print("\nTop muscles per class:")
    for class_name in class_names:
        class_df = top_muscles_df.loc[top_muscles_df["class_name"] == class_name].copy().head(int(top_k))
        if class_df.empty:
            continue
        summary = ", ".join(
            f"{row['muscle']}={float(row['mean_activation']):.3f}"
            for _, row in class_df.iterrows()
        )
        print(f"  {class_name}: {summary}")


def load_runtime_graph_edges(
    *,
    muscle_names: list[str],
) -> tuple[list[dict[str, Any]], str]:
    try:
        from app import main as runtime_main

        context = runtime_main.create_muscle_model_context()
        raw_edges = runtime_main.build_muscle_edges(context, include_data_driven=False)
        allowed = set(str(name) for name in muscle_names)
        unique_edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for edge in raw_edges:
            source_name = str(edge.get("source_name", ""))
            target_name = str(edge.get("target_name", ""))
            edge_type = str(edge.get("edge_type", ""))
            if source_name not in allowed or target_name not in allowed:
                continue
            if source_name == target_name:
                continue
            pair_key = (min(source_name, target_name), max(source_name, target_name), edge_type)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            unique_edges.append(
                {
                    "source_name": source_name,
                    "target_name": target_name,
                    "edge_type": edge_type,
                    "edge_weight": float(edge.get("edge_weight", 1.0)),
                }
            )
        if unique_edges:
            return unique_edges, "runtime_graph"
    except Exception as exc:
        print(f"[WARN] Could not load runtime graph edges; using functional-group fallback ({type(exc).__name__}: {exc}).")
    return build_fallback_graph_edges(muscle_names), "functional_group_fallback"


def build_fallback_graph_edges(muscle_names: list[str]) -> list[dict[str, Any]]:
    allowed = {str(name) for name in muscle_names if str(name) not in GRAPH_EXCLUDED_NODE_NAMES}
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group_name, group_muscles in FUNCTIONAL_GROUPS.items():
        available = [str(name) for name in group_muscles if str(name) in allowed]
        for idx, source_name in enumerate(available):
            for target_name in available[idx + 1 :]:
                pair_key = (min(source_name, target_name), max(source_name, target_name), "functional_group")
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                edges.append(
                    {
                        "source_name": source_name,
                        "target_name": target_name,
                        "edge_type": "functional_group",
                        "edge_weight": 1.0,
                        "group_name": group_name,
                    }
                )
    return edges


def build_class_graph_edges(
    class_profile_df: pd.DataFrame,
    *,
    static_edges: list[dict[str, Any]],
    class_names: list[str],
    top_k: int = 15,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if class_profile_df.empty:
        empty_columns = ["class_name", "source_name", "target_name", "edge_type", "score", "rank_within_class"]
        empty_df = pd.DataFrame(columns=empty_columns)
        return empty_df, {class_name: empty_df.copy() for class_name in class_names}

    profile_indexed = class_profile_df.set_index("muscle")
    combined_rows: list[dict[str, Any]] = []
    per_class_tables: dict[str, pd.DataFrame] = {}

    for class_name in class_names:
        mean_col = f"{class_name}_mean"
        if mean_col not in profile_indexed.columns:
            per_class_tables[class_name] = pd.DataFrame(columns=["source_name", "target_name", "edge_type", "score", "rank_within_class"])
            continue
        edge_rows: list[dict[str, Any]] = []
        for edge in static_edges:
            source_name = str(edge["source_name"])
            target_name = str(edge["target_name"])
            if source_name not in profile_indexed.index or target_name not in profile_indexed.index:
                continue
            u_i = safe_float(profile_indexed.at[source_name, mean_col], default=0.0)
            u_j = safe_float(profile_indexed.at[target_name, mean_col], default=0.0)
            score = float(u_i * u_j)
            edge_rows.append(
                {
                    "class_name": class_name,
                    "source_name": source_name,
                    "target_name": target_name,
                    "edge_type": str(edge["edge_type"]),
                    "score": score,
                }
            )
        class_edges_df = pd.DataFrame(edge_rows)
        if class_edges_df.empty:
            per_class_tables[class_name] = pd.DataFrame(columns=["source_name", "target_name", "edge_type", "score", "rank_within_class"])
            continue
        class_edges_df = class_edges_df.sort_values(by=["score", "source_name", "target_name"], ascending=[False, True, True]).reset_index(drop=True)
        class_edges_df["rank_within_class"] = np.arange(1, len(class_edges_df) + 1)
        top_edges_df = class_edges_df.head(int(top_k)).copy()
        per_class_tables[class_name] = top_edges_df.loc[:, ["source_name", "target_name", "edge_type", "score", "rank_within_class"]].copy()
        combined_rows.extend(top_edges_df.to_dict(orient="records"))

    combined_df = pd.DataFrame(
        combined_rows,
        columns=["class_name", "source_name", "target_name", "edge_type", "score", "rank_within_class"],
    )
    return combined_df, per_class_tables


def compute_trial_edge_summary_features(
    trial_activation_df: pd.DataFrame,
    *,
    static_edges: list[dict[str, Any]],
    edge_active_threshold: float,
) -> pd.DataFrame:
    metadata_cols = ["source_file", "trial_id", "exercise_name"]
    edge_summary_df = trial_activation_df.loc[:, [col for col in metadata_cols if col in trial_activation_df.columns]].copy()

    synergy_edges = [edge for edge in static_edges if str(edge.get("edge_type", "")) == "synergist"]
    antagonist_edges = [edge for edge in static_edges if str(edge.get("edge_type", "")) == "antagonist"]
    functional_group_edges = [edge for edge in static_edges if str(edge.get("edge_type", "")) == "functional_group"]

    # Edge features represent coactivation. If two muscles are both active
    # during the hold segment, u_i * u_j grows even when the individual
    # activations are uneven, so it captures pair-level structure that node
    # features alone do not encode. Classification here is still only a
    # representation check: we use it to ask whether u plus inter-muscle
    # coordination summaries separate gestures better than node features alone.
    def summarize_row(row: pd.Series, edges: list[dict[str, Any]]) -> tuple[float, float, int, float]:
        scores: list[float] = []
        for edge in edges:
            source_name = str(edge["source_name"])
            target_name = str(edge["target_name"])
            u_i = safe_float(row.get(source_name, 0.0), default=0.0)
            u_j = safe_float(row.get(target_name, 0.0), default=0.0)
            scores.append(float(u_i * u_j))
        if not scores:
            return 0.0, 0.0, 0, 0.0
        scores_array = np.asarray(scores, dtype=float)
        active_count = int(np.sum(scores_array > float(edge_active_threshold)))
        top_k = min(5, int(scores_array.size))
        if top_k > 0:
            top_scores = np.sort(scores_array)[-top_k:]
            top_sum = float(np.sum(top_scores))
        else:
            top_sum = 0.0
        return float(np.mean(scores_array)), float(np.max(scores_array)), active_count, top_sum

    synergy_metrics = trial_activation_df.apply(lambda row: summarize_row(row, synergy_edges), axis=1)
    antagonist_metrics = trial_activation_df.apply(lambda row: summarize_row(row, antagonist_edges), axis=1)
    functional_group_metrics = trial_activation_df.apply(lambda row: summarize_row(row, functional_group_edges), axis=1)

    edge_summary_df["mean_synergy_score"] = [item[0] for item in synergy_metrics]
    edge_summary_df["max_synergy_score"] = [item[1] for item in synergy_metrics]
    edge_summary_df["num_synergy_edges_active"] = [item[2] for item in synergy_metrics]
    edge_summary_df["sum_top5_synergy_scores"] = [item[3] for item in synergy_metrics]
    edge_summary_df["mean_antagonist_score"] = [item[0] for item in antagonist_metrics]
    edge_summary_df["max_antagonist_score"] = [item[1] for item in antagonist_metrics]
    edge_summary_df["num_antagonist_edges_active"] = [item[2] for item in antagonist_metrics]
    edge_summary_df["sum_top5_antagonist_scores"] = [item[3] for item in antagonist_metrics]
    edge_summary_df["mean_functional_group_score"] = [item[0] for item in functional_group_metrics]
    return edge_summary_df


def evaluate_feature_set(
    output_dir: Path,
    *,
    filename_stem: str,
    features_df: pd.DataFrame,
    feature_cols: list[str],
    extra_summary: dict[str, Any],
) -> dict[str, Any]:
    save_dataframe(features_df, output_dir / f"features_{filename_stem}.csv")
    try:
        predictions_df, confusion_df, summary, report_text = evaluate_classifier(features_df, feature_cols)
    except Exception as exc:
        summary = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
            **extra_summary,
        }
        save_dataframe(pd.DataFrame(columns=["source_file", "trial_id", "y_true", "y_pred", "correct"]), output_dir / f"predictions_{filename_stem}.csv")
        save_confusion_dataframe(pd.DataFrame(), output_dir / f"confusion_matrix_{filename_stem}.csv")
        save_json(summary, output_dir / f"summary_{filename_stem}.json")
        print(f"[WARN] {filename_stem} evaluation skipped: {type(exc).__name__}: {exc}")
        return summary

    summary.update(extra_summary)
    summary["feature_columns"] = list(feature_cols)
    summary["n_feature_columns"] = int(len(feature_cols))
    save_dataframe(predictions_df, output_dir / f"predictions_{filename_stem}.csv")
    save_confusion_dataframe(confusion_df, output_dir / f"confusion_matrix_{filename_stem}.csv")
    save_json(summary, output_dir / f"summary_{filename_stem}.json")
    save_text(report_text, output_dir / f"classification_report_{filename_stem}.txt")
    return summary


def run_node_edge_model_comparison(
    output_dir: Path,
    *,
    analysis_trials_df: pd.DataFrame,
    ctrl_cols: list[str],
    edge_active_threshold: float,
) -> None:
    if analysis_trials_df.empty or not ctrl_cols:
        empty_summary = {
            "status": "skipped",
            "error": "No trials available for node/edge comparison.",
        }
        save_dataframe(pd.DataFrame(), output_dir / "features_node_edge.csv")
        save_dataframe(pd.DataFrame(), output_dir / "predictions_node_only.csv")
        save_dataframe(pd.DataFrame(), output_dir / "predictions_node_edge.csv")
        save_confusion_dataframe(pd.DataFrame(), output_dir / "confusion_matrix_node_only.csv")
        save_confusion_dataframe(pd.DataFrame(), output_dir / "confusion_matrix_node_edge.csv")
        save_json(empty_summary, output_dir / "summary_node_only.json")
        save_json(empty_summary, output_dir / "summary_node_edge.json")
        print("[WARN] Node/edge comparison skipped because no analysis trials are available.")
        return

    muscle_names = resolve_muscle_names(ctrl_cols)
    trial_activation_df = build_trial_activation_dataframe(
        analysis_trials_df,
        ctrl_cols=ctrl_cols,
        muscle_names=muscle_names,
    )
    static_edges, edge_source = load_runtime_graph_edges(muscle_names=muscle_names)
    edge_summary_df = compute_trial_edge_summary_features(
        trial_activation_df,
        static_edges=static_edges,
        edge_active_threshold=float(edge_active_threshold),
    )

    node_feature_cols = feature_columns_from_trials(analysis_trials_df)
    node_feature_df = features_dataframe_from_trials(analysis_trials_df, node_feature_cols)

    join_keys = [col for col in ["source_file", "trial_id", "exercise_name"] if col in node_feature_df.columns and col in edge_summary_df.columns]
    comparison_df = node_feature_df.merge(edge_summary_df, on=join_keys, how="left", suffixes=("", "_edge"))
    comparison_df = comparison_df.fillna(0.0)

    edge_feature_cols = [
        "mean_synergy_score",
        "max_synergy_score",
        "num_synergy_edges_active",
        "mean_antagonist_score",
        "max_antagonist_score",
        "num_antagonist_edges_active",
        "mean_functional_group_score",
        "sum_top5_synergy_scores",
        "sum_top5_antagonist_scores",
    ]
    node_edge_feature_cols = node_feature_cols + [col for col in edge_feature_cols if col in comparison_df.columns]

    metadata_extra = {
        "n_trials": int(len(comparison_df)),
        "edge_source": edge_source,
        "edge_active_threshold": float(edge_active_threshold),
        "n_static_edges": int(len(static_edges)),
        "n_synergist_edges": int(sum(1 for edge in static_edges if str(edge.get("edge_type")) == "synergist")),
        "n_antagonist_edges": int(sum(1 for edge in static_edges if str(edge.get("edge_type")) == "antagonist")),
        "n_functional_group_edges": int(sum(1 for edge in static_edges if str(edge.get("edge_type")) == "functional_group")),
        "n_edge_features_added": int(len(edge_feature_cols)),
    }

    print("\nNode-only feature evaluation:")
    node_only_summary = evaluate_feature_set(
        output_dir,
        filename_stem="node_only",
        features_df=comparison_df.loc[:, ["source_file", "trial_id", "exercise_name"] + node_feature_cols].copy(),
        feature_cols=node_feature_cols,
        extra_summary={**metadata_extra, "feature_mode": "node_only"},
    )

    print("\nNode + edge-summary feature evaluation:")
    node_edge_summary = evaluate_feature_set(
        output_dir,
        filename_stem="node_edge",
        features_df=comparison_df.loc[:, ["source_file", "trial_id", "exercise_name"] + node_edge_feature_cols].copy(),
        feature_cols=node_edge_feature_cols,
        extra_summary={**metadata_extra, "feature_mode": "node_plus_edge_summary"},
    )
    save_dataframe(comparison_df.loc[:, ["source_file", "trial_id", "exercise_name"] + node_edge_feature_cols].copy(), output_dir / "features_node_edge.csv")

    class_profile_df = build_class_profile_table(
        trial_activation_df,
        muscle_names=muscle_names,
    )
    class_names = ordered_class_names(trial_activation_df["exercise_name"])
    graph_edges_combined_df, _ = build_class_graph_edges(
        class_profile_df,
        static_edges=static_edges,
        class_names=class_names,
        top_k=max(15, len(static_edges)),
    )
    if graph_edges_combined_df.empty:
        top_coordination_edges_df = pd.DataFrame(columns=["class_name", "edge_type", "source_name", "target_name", "score", "rank_within_class_type"])
    else:
        top_coordination_rows: list[dict[str, Any]] = []
        filtered_graph_edges_df = graph_edges_combined_df.loc[
            graph_edges_combined_df["edge_type"].isin(["synergist", "antagonist"])
        ].copy()
        for (class_name, edge_type), group_df in filtered_graph_edges_df.groupby(["class_name", "edge_type"], sort=False):
            top_df = group_df.sort_values(by=["score", "source_name", "target_name"], ascending=[False, True, True]).head(10).reset_index(drop=True)
            for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
                top_coordination_rows.append(
                    {
                        "class_name": str(class_name),
                        "edge_type": str(edge_type),
                        "source_name": str(row["source_name"]),
                        "target_name": str(row["target_name"]),
                        "score": float(row["score"]),
                        "rank_within_class_type": rank,
                    }
                )
        top_coordination_edges_df = pd.DataFrame(
            top_coordination_rows,
            columns=["class_name", "edge_type", "source_name", "target_name", "score", "rank_within_class_type"],
        )
    save_dataframe(top_coordination_edges_df, output_dir / "top_coordination_edges_per_class.csv")

    print("\nNode/edge comparison summary:")
    print(f"  retained trials: {int(len(comparison_df))}")
    print(f"  edge features added: {int(len(edge_feature_cols))}")
    print(f"  node-only accuracy: {safe_float(node_only_summary.get('accuracy'), default=float('nan')):.4f}" if "accuracy" in node_only_summary else "  node-only accuracy: n/a")
    print(f"  node+edge accuracy: {safe_float(node_edge_summary.get('accuracy'), default=float('nan')):.4f}" if "accuracy" in node_edge_summary else "  node+edge accuracy: n/a")


def save_class_analysis_outputs(
    output_dir: Path,
    *,
    class_profile_df: pd.DataFrame,
    top_muscles_df: pd.DataFrame,
    graph_edges_combined_df: pd.DataFrame,
    graph_edges_per_class: dict[str, pd.DataFrame],
) -> None:
    save_dataframe(class_profile_df, output_dir / "muscle_profile_per_class.csv")
    save_dataframe(top_muscles_df, output_dir / "top_muscles_per_class.csv")
    save_dataframe(graph_edges_combined_df, output_dir / "graph_edges_per_class.csv")

    all_class_names = list(dict.fromkeys([*CLASS_ORDER, *graph_edges_per_class.keys()]))
    empty_edge_df = pd.DataFrame(columns=["source_name", "target_name", "edge_type", "score", "rank_within_class"])
    for class_name in all_class_names:
        class_edges_df = graph_edges_per_class.get(class_name, empty_edge_df)
        save_dataframe(class_edges_df, output_dir / f"graph_edges_{class_name}.csv")


def build_filtered_class_profile_table(
    class_profile_df: pd.DataFrame,
    *,
    filtered_muscles: tuple[str, ...] = PROPOSAL_FILTERED_MUSCLES,
) -> pd.DataFrame:
    if class_profile_df.empty:
        return pd.DataFrame(columns=class_profile_df.columns.tolist() or ["muscle"])
    filtered_set = {str(name) for name in filtered_muscles}
    filtered_df = class_profile_df.loc[class_profile_df["muscle"].astype(str).isin(filtered_set)].copy()
    if filtered_df.empty:
        return pd.DataFrame(columns=class_profile_df.columns.tolist())
    order_map = {name: idx for idx, name in enumerate(filtered_muscles)}
    filtered_df["_order"] = filtered_df["muscle"].astype(str).map(lambda name: order_map.get(name, 10**9))
    filtered_df = filtered_df.sort_values(by=["_order", "muscle"]).drop(columns=["_order"]).reset_index(drop=True)
    return filtered_df


def save_filtered_interpretation_outputs(
    output_dir: Path,
    *,
    class_profile_df: pd.DataFrame,
    class_names: list[str],
) -> None:
    filtered_profile_df = build_filtered_class_profile_table(class_profile_df)
    filtered_top_muscles_df = build_top_muscles_table(
        filtered_profile_df,
        class_names=class_names,
        top_k=8,
    )
    save_dataframe(filtered_profile_df, output_dir / "muscle_profile_per_class_filtered.csv")
    save_dataframe(filtered_top_muscles_df, output_dir / "top_muscles_per_class_filtered.csv")

    empty_top_df = pd.DataFrame(columns=["class_name", "rank", "muscle", "mean_activation", "std_activation"])
    for class_name in PROPOSAL_FILTERED_CLASSES:
        class_df = filtered_top_muscles_df.loc[filtered_top_muscles_df["class_name"] == class_name].copy()
        save_dataframe(class_df if not class_df.empty else empty_top_df.copy(), output_dir / f"top_muscles_{class_name}_filtered.csv")


def write_proposal_notes(output_dir: Path) -> None:
    notes = (
        "Proposal notes\n"
        "==============\n\n"
        "- Full muscle outputs are kept unchanged for transparency and reproducibility.\n"
        "- The filtered muscle subset is used only for interpretation, so proposal figures focus on gesture-relevant muscles instead of every available actuator.\n"
        "- Global gestures such as open, fist, and claw provide the strongest graph-level evidence because their activation patterns are the most stable and separable across trials.\n"
        "- Finger-specific gestures are interpreted with the filtered muscle subset because stabilizer and wrist-support muscles can contaminate rankings if every muscle is shown at once.\n"
    )
    (output_dir / "proposal_notes.txt").write_text(notes, encoding="utf-8")


def build_proposal_core_package(output_dir: Path) -> None:
    proposal_core_dir = output_dir / "proposal_core"
    proposal_core_dir.mkdir(parents=True, exist_ok=True)
    for child in proposal_core_dir.iterdir():
        if child.is_file():
            child.unlink()

    for filename in PROPOSAL_CORE_FILENAMES:
        source_path = output_dir / filename
        if not source_path.exists():
            print(f"[WARN] proposal_core missing source file: {source_path.name}")
            continue
        shutil.copy2(source_path, proposal_core_dir / filename)

def try_generate_plots(
    output_dir: Path,
    *,
    class_profile_df: pd.DataFrame,
    top_muscles_df: pd.DataFrame,
    class_names: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Plot generation skipped because matplotlib is unavailable ({type(exc).__name__}: {exc}).")
        return

    mean_columns = [f"{class_name}_mean" for class_name in class_names if f"{class_name}_mean" in class_profile_df.columns]
    if mean_columns:
        heatmap_df = class_profile_df.loc[:, ["muscle"] + mean_columns].copy().set_index("muscle")
        heatmap_values = heatmap_df.to_numpy(dtype=float, copy=True)
        fig_height = max(4.0, 0.30 * max(1, len(heatmap_df)))
        fig_width = max(5.0, 1.8 * max(1, len(mean_columns)))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(heatmap_values, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(mean_columns)))
        ax.set_xticklabels([col.replace("_mean", "") for col in mean_columns], rotation=0)
        ax.set_yticks(np.arange(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index.tolist())
        ax.set_title("Mean Muscle Activation Per Class")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean activation")
        fig.tight_layout()
        fig.savefig(output_dir / "muscle_profile_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    for class_name in class_names:
        class_top_df = top_muscles_df.loc[top_muscles_df["class_name"] == class_name].copy().sort_values(by="rank")
        if class_top_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.bar(class_top_df["muscle"], class_top_df["mean_activation"], color="#1f77b4")
        ax.set_title(f"Top Muscles: {class_name}")
        ax.set_xlabel("Muscle")
        ax.set_ylabel("Mean activation")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / f"top_muscles_{class_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def run_interpretation_analysis(
    output_dir: Path,
    *,
    analysis_trials_df: pd.DataFrame,
    ctrl_cols: list[str],
) -> None:
    if analysis_trials_df.empty or not ctrl_cols:
        print("[WARN] Interpretation analysis skipped because no trial activations are available.")
        empty_profile_df = pd.DataFrame(columns=["muscle", "open_mean", "open_std", "fist_mean", "fist_std", "claw_mean", "claw_std"])
        save_class_analysis_outputs(
            output_dir,
            class_profile_df=empty_profile_df,
            top_muscles_df=pd.DataFrame(columns=["class_name", "rank", "muscle", "mean_activation", "std_activation"]),
            graph_edges_combined_df=pd.DataFrame(columns=["class_name", "source_name", "target_name", "edge_type", "score", "rank_within_class"]),
            graph_edges_per_class={class_name: pd.DataFrame(columns=["source_name", "target_name", "edge_type", "score", "rank_within_class"]) for class_name in CLASS_ORDER},
        )
        save_filtered_interpretation_outputs(
            output_dir,
            class_profile_df=empty_profile_df,
            class_names=list(PROPOSAL_FILTERED_CLASSES),
        )
        write_proposal_notes(output_dir)
        return

    muscle_names = resolve_muscle_names(ctrl_cols)
    trial_activation_df = build_trial_activation_dataframe(
        analysis_trials_df,
        ctrl_cols=ctrl_cols,
        muscle_names=muscle_names,
    )
    class_names = ordered_class_names(trial_activation_df["exercise_name"])
    class_profile_df = build_class_profile_table(
        trial_activation_df,
        muscle_names=muscle_names,
    )
    top_muscles_df = build_top_muscles_table(
        class_profile_df,
        class_names=class_names,
        top_k=8,
    )
    print_top_muscles_summary(top_muscles_df, top_k=5)

    static_edges, edge_source = load_runtime_graph_edges(muscle_names=muscle_names)
    print(f"Graph edge source: {edge_source} ({len(static_edges)} static edges)")
    graph_edges_combined_df, graph_edges_per_class = build_class_graph_edges(
        class_profile_df,
        static_edges=static_edges,
        class_names=class_names,
        top_k=15,
    )
    save_class_analysis_outputs(
        output_dir,
        class_profile_df=class_profile_df,
        top_muscles_df=top_muscles_df,
        graph_edges_combined_df=graph_edges_combined_df,
        graph_edges_per_class=graph_edges_per_class,
    )
    save_filtered_interpretation_outputs(
        output_dir,
        class_profile_df=class_profile_df,
        class_names=class_names,
    )
    write_proposal_notes(output_dir)
    try_generate_plots(
        output_dir,
        class_profile_df=class_profile_df,
        top_muscles_df=top_muscles_df,
        class_names=class_names,
    )


def print_trial_summary(trials_df: pd.DataFrame, *, prefix: str = "") -> None:
    class_counts = trials_df["exercise_name"].value_counts().sort_index()
    hold_frame_counts = sorted(trials_df["num_hold_frames"].astype(int).unique().tolist())

    title_prefix = f"{prefix} " if prefix else ""
    print(f"{title_prefix}loaded trials: {len(trials_df)}")
    print(f"{title_prefix}hold frame counts used: {hold_frame_counts}")
    print(f"{title_prefix}trials per class:")
    for label, count in class_counts.items():
        print(f"  {label}: {int(count)}")


def evaluate_classifier(
    features_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    X = features_df.loc[:, feature_cols].to_numpy(dtype=float, copy=True)
    y = features_df["exercise_name"].astype(str).to_numpy()

    unique_labels = sorted(pd.unique(y).tolist())
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 classes for classification, found {unique_labels}.")
    class_counts = pd.Series(y).value_counts()
    if int(class_counts.min()) < 2:
        raise ValueError(
            "Need at least 2 samples per class for stratified train-test split, "
            f"found counts {class_counts.to_dict()}."
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    indices = np.arange(len(features_df), dtype=int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions_df = pd.DataFrame(
        {
            "source_file": features_df.iloc[idx_test]["source_file"].astype(str).to_numpy(),
            "trial_id": features_df.iloc[idx_test]["trial_id"].astype(str).to_numpy(),
            "y_true": y_test.astype(str),
            "y_pred": y_pred.astype(str),
        }
    )
    predictions_df["correct"] = predictions_df["y_true"] == predictions_df["y_pred"]

    label_order = sorted(pd.unique(np.concatenate([y_test.astype(str), y_pred.astype(str)])).tolist())
    conf = confusion_matrix(y_test, y_pred, labels=label_order)
    confusion_df = pd.DataFrame(conf, index=label_order, columns=label_order)
    confusion_df.index.name = "y_true"
    confusion_df.columns.name = "y_pred"

    report_dict = classification_report(y_test, y_pred, labels=label_order, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, labels=label_order, zero_division=0)
    accuracy = float(accuracy_score(y_test, y_pred))

    summary = {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "label_order": label_order,
        "n_trials": int(len(features_df)),
        "n_features": int(len(feature_cols)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "split_scheme": "StratifiedTrainTestSplit",
        "test_size_fraction": 0.3,
        "random_state": 42,
        "feature_source": "ctrl_analysis_*",
    }

    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion matrix:")
    print(confusion_df.to_string())
    print("\nClassification report:")
    print(report_text)

    return predictions_df, confusion_df, summary, report_text


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_confusion_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def plot_confusion_matrix(cm_path: Path, output_dir: Path) -> bool:
    if not cm_path.exists():
        print(f"[WARN] Confusion matrix plot skipped because file does not exist: {cm_path}")
        return False

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        print(f"[WARN] Confusion matrix plotting skipped because seaborn/matplotlib is unavailable ({type(exc).__name__}: {exc}).")
        return False

    try:
        cm_df = pd.read_csv(cm_path, index_col=0)
    except Exception as exc:
        print(f"[WARN] Confusion matrix plotting skipped because CSV could not be read ({type(exc).__name__}: {exc}).")
        return False

    if cm_df.empty:
        print(f"[WARN] Confusion matrix plotting skipped because matrix is empty: {cm_path}")
        return False

    cm_df = cm_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy = 0.0
    total = int(cm_df.to_numpy(dtype=int, copy=True).sum())
    if total > 0:
        accuracy = float(np.trace(cm_df.to_numpy(dtype=float, copy=True)) / float(total))

    class_labels = [str(label) for label in cm_df.columns.tolist()]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        square=True,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Muscle Activation Classification)")
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels([str(label) for label in cm_df.index.tolist()], rotation=0)
    fig.tight_layout()
    cm_image_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    cm_values = cm_df.to_numpy(dtype=float, copy=True)
    row_sums = cm_values.sum(axis=1, keepdims=True)
    normalized_values = np.divide(
        cm_values,
        row_sums,
        out=np.zeros_like(cm_values, dtype=float),
        where=row_sums > 0,
    )
    cm_normalized_df = pd.DataFrame(normalized_values, index=cm_df.index, columns=cm_df.columns)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_normalized_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        square=True,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels([str(label) for label in cm_normalized_df.index.tolist()], rotation=0)
    fig.tight_layout()
    cm_normalized_image_path = output_dir / "confusion_matrix_normalized.png"
    fig.savefig(cm_normalized_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix image: {cm_image_path}")
    print(f"Saved normalized confusion matrix image: {cm_normalized_image_path}")
    print(f"Confusion matrix accuracy: {accuracy:.4f}")
    return True


def plot_best_available_confusion_matrix(output_dir: Path) -> bool:
    filtered_path = output_dir / "filtered_confusion_matrix.csv"
    base_path = output_dir / "confusion_matrix.csv"
    if filtered_path.exists():
        return plot_confusion_matrix(filtered_path, output_dir)
    return plot_confusion_matrix(base_path, output_dir)


def run_classification_bundle(
    output_dir: Path,
    *,
    prefix: str,
    features_df: pd.DataFrame,
    feature_cols: list[str],
    extra_summary: dict[str, Any],
) -> bool:
    save_dataframe(features_df, output_dir / f"{prefix}features.csv")
    try:
        predictions_df, confusion_df, summary, report_text = evaluate_classifier(features_df, feature_cols)
    except Exception as exc:
        summary = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
            **extra_summary,
        }
        save_dataframe(pd.DataFrame(columns=["source_file", "trial_id", "y_true", "y_pred", "correct"]), output_dir / f"{prefix}predictions.csv")
        save_confusion_dataframe(pd.DataFrame(), output_dir / f"{prefix}confusion_matrix.csv")
        save_json(summary, output_dir / f"{prefix}summary.json")
        print(f"[WARN] Classification skipped for {prefix or 'base'}set: {type(exc).__name__}: {exc}")
        return False

    summary.update(extra_summary)
    save_dataframe(predictions_df, output_dir / f"{prefix}predictions.csv")
    save_confusion_dataframe(confusion_df, output_dir / f"{prefix}confusion_matrix.csv")
    save_json(summary, output_dir / f"{prefix}summary.json")
    if prefix == "":
        save_confusion_dataframe(confusion_df, output_dir / "confusion_matrix_split.csv")
        save_text(report_text, output_dir / "classification_report_split.txt")
    return True


def filter_trials_for_classification(
    trials_df: pd.DataFrame,
    *,
    enable_quality_filter: bool,
) -> pd.DataFrame:
    if not enable_quality_filter:
        return trials_df.copy()

    kept_df = trials_df.loc[trials_df["keep_for_classification"].astype(bool)].copy()
    dropped_df = trials_df.loc[~trials_df["keep_for_classification"].astype(bool)].copy()
    if not dropped_df.empty:
        print("Dropped trials per class:")
        drop_counts = dropped_df["exercise_name"].value_counts().sort_index()
        for label, count in drop_counts.items():
            print(f"  {label}: {int(count)}")
    else:
        print("Dropped trials per class:\n  none")

    kept_counts = kept_df["exercise_name"].value_counts().sort_index() if not kept_df.empty else pd.Series(dtype=int)
    original_labels = sorted(trials_df["exercise_name"].astype(str).unique().tolist())
    for label in original_labels:
        remaining = int(kept_counts.get(label, 0))
        if remaining == 0:
            print(f"[WARN] Filtering removed all trials for class {label!r}.")
        elif remaining < 2:
            print(f"[WARN] Class {label!r} has fewer than 2 trials after filtering ({remaining}).")
    return kept_df


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "offline_u_classifier_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_records, ctrl_cols, skipped = extract_trial_records(
        input_dir,
        hold_tail_ratio=float(args.hold_tail_ratio),
        hold_frame_cap=int(args.hold_frame_cap),
        max_hold_std=float(args.max_hold_std),
        max_hold_delta=float(args.max_hold_delta),
        min_hold_activation=float(args.min_hold_activation),
    )
    trials_df = build_trial_dataframe(trial_records)
    feature_cols = feature_columns_from_trials(trials_df)

    print(f"Matched ctrl columns: {len(ctrl_cols)}")
    print(f"Feature count: {len(feature_cols)}")
    print_trial_summary(trials_df)
    if skipped:
        print(f"Skipped items: {len(skipped)}")

    quality_df = trials_df.loc[:, [col for col in quality_report_columns() if col in trials_df.columns]].copy()
    save_dataframe(quality_df, output_dir / "trial_quality.csv")

    base_features_df = features_dataframe_from_trials(trials_df, feature_cols)
    base_summary_extra = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "csv_pattern": CSV_GLOB,
        "hold_tail_ratio": float(args.hold_tail_ratio),
        "hold_frame_cap": int(args.hold_frame_cap),
        "class_counts": {
            str(label): int(count)
            for label, count in trials_df["exercise_name"].value_counts().sort_index().items()
        },
        "hold_frame_counts_used": sorted(trials_df["num_hold_frames"].astype(int).unique().tolist()),
        "skipped": skipped,
        "quality_filter_enabled": False,
    }
    run_classification_bundle(
        output_dir,
        prefix="",
        features_df=base_features_df,
        feature_cols=feature_cols,
        extra_summary=base_summary_extra,
    )

    analysis_trials_df = trials_df.copy()
    if not bool(args.enable_quality_filter):
        run_interpretation_analysis(
            output_dir,
            analysis_trials_df=analysis_trials_df,
            ctrl_cols=ctrl_cols,
        )
        run_node_edge_model_comparison(
            output_dir,
            analysis_trials_df=analysis_trials_df,
            ctrl_cols=ctrl_cols,
            edge_active_threshold=float(args.edge_active_threshold),
        )
        plot_best_available_confusion_matrix(output_dir)
        build_proposal_core_package(output_dir)
        print(f"\nSaved outputs to: {output_dir}")
        return

    filtered_trials_df = filter_trials_for_classification(
        trials_df,
        enable_quality_filter=bool(args.enable_quality_filter),
    )
    if filtered_trials_df.empty:
        filtered_summary = {
            "status": "skipped",
            "error": "All trials were removed by quality filtering.",
            "quality_filter_enabled": True,
            "max_hold_std": float(args.max_hold_std),
            "max_hold_delta": float(args.max_hold_delta),
            "min_hold_activation": float(args.min_hold_activation),
            "n_trials_before_filter": int(len(trials_df)),
            "n_trials_after_filter": 0,
        }
        save_json(filtered_summary, output_dir / "filtered_summary.json")
        run_interpretation_analysis(
            output_dir,
            analysis_trials_df=filtered_trials_df,
            ctrl_cols=ctrl_cols,
        )
        run_node_edge_model_comparison(
            output_dir,
            analysis_trials_df=filtered_trials_df,
            ctrl_cols=ctrl_cols,
            edge_active_threshold=float(args.edge_active_threshold),
        )
        plot_best_available_confusion_matrix(output_dir)
        build_proposal_core_package(output_dir)
        print("[WARN] Quality filtering removed every trial. Saved quality report and filtered_summary.json.")
        print(f"\nSaved outputs to: {output_dir}")
        return

    print_trial_summary(filtered_trials_df, prefix="Filtered")
    filtered_features_df = features_dataframe_from_trials(filtered_trials_df, feature_cols)
    filtered_summary_extra = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "csv_pattern": CSV_GLOB,
        "hold_tail_ratio": float(args.hold_tail_ratio),
        "hold_frame_cap": int(args.hold_frame_cap),
        "quality_filter_enabled": True,
        "max_hold_std": float(args.max_hold_std),
        "max_hold_delta": float(args.max_hold_delta),
        "min_hold_activation": float(args.min_hold_activation),
        "n_trials_before_filter": int(len(trials_df)),
        "n_trials_after_filter": int(len(filtered_trials_df)),
        "class_counts_after_filter": {
            str(label): int(count)
            for label, count in filtered_trials_df["exercise_name"].value_counts().sort_index().items()
        },
    }
    run_classification_bundle(
        output_dir,
        prefix="filtered_",
        features_df=filtered_features_df,
        feature_cols=feature_cols,
        extra_summary=filtered_summary_extra,
    )
    analysis_trials_df = filtered_trials_df.copy()
    run_interpretation_analysis(
        output_dir,
        analysis_trials_df=analysis_trials_df,
        ctrl_cols=ctrl_cols,
    )
    run_node_edge_model_comparison(
        output_dir,
        analysis_trials_df=analysis_trials_df,
        ctrl_cols=ctrl_cols,
        edge_active_threshold=float(args.edge_active_threshold),
    )
    plot_best_available_confusion_matrix(output_dir)
    build_proposal_core_package(output_dir)
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
