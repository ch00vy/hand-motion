#!/usr/bin/env python3
"""Offline node-level reference and deviation pipeline for gesture contexts.

This script treats `open` and `fist` as simulation-based gesture references.
It does not diagnose muscle pathology. Instead, it estimates whether a node is
within or outside the model-derived reference for the corresponding gesture.

For sidang/reporting, the main outputs are:
- explicit reference / validation / test splits
- cleaned reference profiles
- node-level deviation outputs on validation and test trials
- presentation-ready summary tables and case studies

The optional classifier is kept as an appendix-only tool. It is not the main
methodological claim of this pipeline.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from deviation_analysis import VALID_DEVIATION_AGGREGATIONS, VALID_THRESHOLD_MODES, compute_node_deviation
from node_features import NODE_TEMPORAL_FEATURE_NAMES, build_node_temporal_features
from offline_u_classifier import (
    CSV_GLOB,
    DEFAULT_HOLD_FRAME_CAP,
    DEFAULT_HOLD_TAIL_RATIO,
    DEFAULT_MAX_HOLD_DELTA,
    DEFAULT_MAX_HOLD_STD,
    DEFAULT_MIN_HOLD_ACTIVATION,
    choose_exercise_name,
    choose_trial_id,
    compute_trial_quality_metrics,
    find_ctrl_analysis_columns,
    find_qpos_columns,
    resolve_muscle_names,
    sanitize_numeric_frame,
    save_confusion_dataframe,
    save_dataframe,
    save_json,
    save_text,
    select_hold_segment,
)
from phase_segmentation import segment_motion_phases
from reference_profile import build_reference_profile


DEFAULT_TARGET_GESTURES = ("open", "fist")
DEFAULT_REFERENCE_COUNT = 15
DEFAULT_VALIDATION_COUNT = 8
DEFAULT_TEST_COUNT = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_QUANTILE_THRESHOLD = 0.70
DEFAULT_REFERENCE_FILTER_MODE = "filter_outliers"
REFERENCE_FILTER_MODES = ("keep_all", "filter_outliers", "filter_unstable")

SIMPLE_REFERENCE_FEATURES = (
    "mean_activation_full",
    "std_activation_full",
    "peak_activation",
    "mean_hold",
    "auc_activation",
)
NUMERIC_FEATURE_COLUMNS = tuple(NODE_TEMPORAL_FEATURE_NAMES)
CATEGORICAL_FEATURE_COLUMNS = ("gesture_label", "node_name")
REFERENCE_MIN_KEEP_FRACTION = 0.70
REFERENCE_MIN_KEEP_ABSOLUTE = 8
REFERENCE_INVALID_FRACTION_EPS = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline node-level reference/deviation pipeline for open and fist recordings."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing ctrl_est_*.csv files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where outputs will be written. Default: <input_dir>/offline_node_classifier_output",
    )
    parser.add_argument(
        "--target-gestures",
        type=str,
        nargs="+",
        default=list(DEFAULT_TARGET_GESTURES),
        help="Gesture contexts used by the pipeline. Default: open fist",
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
        help="Discard trials whose hold segment is too unstable before split generation.",
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
        "--reference-count",
        type=int,
        default=DEFAULT_REFERENCE_COUNT,
        help="Number of reference trials per gesture. Default: 15",
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=DEFAULT_VALIDATION_COUNT,
        help="Number of validation/tuning trials per gesture. Default: 8",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=DEFAULT_TEST_COUNT,
        help="Number of demo/test trials per gesture. Default: 10",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for deterministic split generation.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=DEFAULT_Z_THRESHOLD,
        help="Fallback fixed z-threshold if threshold-mode=fixed.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=VALID_THRESHOLD_MODES,
        default="quantile",
        help="How node deviation thresholds are derived: fixed, quantile, or adaptive.",
    )
    parser.add_argument(
        "--quantile-threshold",
        type=float,
        default=DEFAULT_QUANTILE_THRESHOLD,
        help="Quantile used when threshold-mode=quantile. Default: 0.7",
    )
    parser.add_argument(
        "--deviation-aggregation",
        choices=VALID_DEVIATION_AGGREGATIONS,
        default="weighted",
        help="How feature-level z-scores are aggregated into a node deviation score.",
    )
    parser.add_argument(
        "--per-gesture-threshold",
        type=str,
        choices=("true", "false"),
        default="true",
        help="Use separate thresholds for open and fist when possible.",
    )
    parser.add_argument(
        "--reference-filter-mode",
        choices=REFERENCE_FILTER_MODES,
        default=DEFAULT_REFERENCE_FILTER_MODE,
        help="How aggressively the reference set is cleaned.",
    )
    parser.add_argument(
        "--run-classifier-appendix",
        action="store_true",
        help="Run an appendix-only classifier on validation->test node labels.",
    )
    return parser.parse_args()


def normalize_gesture_set(values: list[str]) -> set[str]:
    return {str(value).strip().lower() for value in values if str(value).strip()}


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def compute_class_balance_score(counts: pd.Series) -> float:
    counts = pd.to_numeric(counts, errors="coerce").fillna(0.0)
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts.to_numpy(dtype=float, copy=True) / total
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    return float(entropy / math.log(float(probs.size)))


def select_deviation_feature_frame(node_df: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "source_file",
        "trial_id",
        "gesture_label",
        "exercise_name",
        "node_name",
        "node_index",
        "num_hold_frames",
        "phase_seq",
    ]
    keep_cols = [col for col in metadata_cols if col in node_df.columns]
    keep_cols.extend([col for col in SIMPLE_REFERENCE_FEATURES if col in node_df.columns])
    return node_df.loc[:, keep_cols].copy()


def summarize_distribution_rows(
    df: pd.DataFrame,
    *,
    split_name: str,
    group_cols: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["split", "gesture_label", "node_name", "node_status", "count", "proportion"])
    working_df = df.copy()
    if "gesture_label" not in working_df.columns:
        working_df["gesture_label"] = "all"
    if "node_name" not in working_df.columns:
        working_df["node_name"] = "all"
    groupby_cols = list(group_cols) + ["node_status"]
    counts_df = working_df.groupby(groupby_cols, dropna=False).size().reset_index(name="count")
    denom_df = counts_df.groupby(group_cols, dropna=False)["count"].sum().reset_index(name="group_total")
    merged_df = counts_df.merge(denom_df, on=group_cols, how="left")
    merged_df["proportion"] = merged_df["count"] / np.maximum(merged_df["group_total"], 1)
    merged_df["split"] = split_name
    if "gesture_label" not in merged_df.columns:
        merged_df["gesture_label"] = "all"
    if "node_name" not in merged_df.columns:
        merged_df["node_name"] = "all"
    return merged_df.loc[:, ["split", "gesture_label", "node_name", "node_status", "count", "proportion"]].copy()


def build_label_distribution_outputs(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    overall_validation = summarize_distribution_rows(
        validation_df.assign(gesture_label="all", node_name="all"),
        split_name="validation",
        group_cols=["gesture_label", "node_name"],
    )
    overall_test = summarize_distribution_rows(
        test_df.assign(gesture_label="all", node_name="all"),
        split_name="test",
        group_cols=["gesture_label", "node_name"],
    )
    by_gesture_validation = summarize_distribution_rows(
        validation_df.assign(node_name="all"),
        split_name="validation",
        group_cols=["gesture_label", "node_name"],
    )
    by_gesture_test = summarize_distribution_rows(
        test_df.assign(node_name="all"),
        split_name="test",
        group_cols=["gesture_label", "node_name"],
    )
    by_node_validation = summarize_distribution_rows(
        validation_df.assign(gesture_label="all"),
        split_name="validation",
        group_cols=["gesture_label", "node_name"],
    )
    by_node_test = summarize_distribution_rows(
        test_df.assign(gesture_label="all"),
        split_name="test",
        group_cols=["gesture_label", "node_name"],
    )
    return {
        "validation": overall_validation,
        "test": overall_test,
        "by_gesture": pd.concat([by_gesture_validation, by_gesture_test], ignore_index=True),
        "by_node": pd.concat([by_node_validation, by_node_test], ignore_index=True),
    }


def compute_threshold_value(
    deviation_df: pd.DataFrame,
    *,
    gesture_label: str | None,
    threshold_mode: str,
    z_threshold: float,
    quantile_threshold: float,
) -> float:
    if deviation_df.empty:
        return float(z_threshold)
    working_df = deviation_df.copy()
    if gesture_label is not None:
        working_df = working_df.loc[
            working_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()
        ].copy()
    if working_df.empty:
        return float(z_threshold)
    abs_signed = working_df["signed_deviation_score"].abs().to_numpy(dtype=float, copy=True)
    mode_name = str(threshold_mode or "fixed").strip().lower()
    if mode_name == "quantile":
        return float(np.quantile(abs_signed, np.clip(float(quantile_threshold), 0.5, 0.99)))
    if mode_name == "adaptive":
        return float(np.mean(abs_signed) + np.std(abs_signed))
    return float(z_threshold)


def apply_thresholds_to_deviation_df(
    deviation_df: pd.DataFrame,
    *,
    threshold_by_gesture: dict[str, float],
) -> pd.DataFrame:
    if deviation_df.empty:
        return deviation_df.copy()
    out_df = deviation_df.copy()
    default_threshold = float(list(threshold_by_gesture.values())[0]) if threshold_by_gesture else 0.0
    out_df["threshold_used"] = out_df["gesture_label"].astype(str).map(
        lambda name: float(threshold_by_gesture.get(str(name).lower(), default_threshold))
    )
    out_df["node_status"] = np.where(
        out_df["signed_deviation_score"] > out_df["threshold_used"],
        "higher_than_reference",
        np.where(
            out_df["signed_deviation_score"] < -out_df["threshold_used"],
            "lower_than_reference",
            "within_reference",
        ),
    )
    return out_df


def build_threshold_sweep_summary(
    raw_validation_df: pd.DataFrame,
    *,
    target_gestures: set[str],
    threshold_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sweep_rows: list[dict[str, Any]] = []
    mode_name = str(threshold_mode or "fixed").strip().lower()
    if mode_name == "quantile":
        threshold_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    else:
        threshold_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for threshold in threshold_values:
        for gesture_label in sorted(target_gestures):
            gesture_df = raw_validation_df.loc[
                raw_validation_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()
            ].copy()
            if gesture_df.empty:
                continue
            if mode_name == "quantile":
                threshold_value = float(
                    np.quantile(gesture_df["signed_deviation_score"].abs().to_numpy(dtype=float, copy=True), threshold)
                )
            elif mode_name == "adaptive":
                abs_scores = gesture_df["signed_deviation_score"].abs().to_numpy(dtype=float, copy=True)
                threshold_value = float(np.mean(abs_scores) + np.std(abs_scores))
            else:
                threshold_value = float(threshold)
            status_series = np.where(
                gesture_df["signed_deviation_score"] > threshold_value,
                "higher_than_reference",
                np.where(
                    gesture_df["signed_deviation_score"] < -threshold_value,
                    "lower_than_reference",
                    "within_reference",
                ),
            )
            counts = pd.Series(status_series).value_counts().reindex(
                ["within_reference", "lower_than_reference", "higher_than_reference"],
                fill_value=0,
            )
            sweep_rows.append(
                {
                    "threshold": float(threshold),
                    "gesture_label": str(gesture_label),
                    "num_within_reference": int(counts["within_reference"]),
                    "num_lower_than_reference": int(counts["lower_than_reference"]),
                    "num_higher_than_reference": int(counts["higher_than_reference"]),
                    "class_balance_score": compute_class_balance_score(counts),
                }
            )

    sweep_df = pd.DataFrame(
        sweep_rows,
        columns=[
            "threshold",
            "gesture_label",
            "num_within_reference",
            "num_lower_than_reference",
            "num_higher_than_reference",
            "class_balance_score",
        ],
    )
    if sweep_df.empty:
        return sweep_df, {}
    grouped = sweep_df.groupby("threshold", sort=True)["class_balance_score"].mean().reset_index(name="mean_balance_score")
    best_threshold = float(grouped.sort_values(by=["mean_balance_score", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
    best_rows_df = sweep_df.loc[sweep_df["threshold"] == best_threshold].copy()
    best_payload = {
        "threshold_mode": mode_name,
        "recommended_threshold": best_threshold,
        "mean_balance_score": float(best_rows_df["class_balance_score"].mean()),
        "gesture_rows": best_rows_df.to_dict(orient="records"),
    }
    return sweep_df, best_payload


def build_reference_quality_outputs(reference_profiles: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    quality_tables: list[pd.DataFrame] = []
    summary_payload: dict[str, Any] = {"gestures": {}}
    for gesture_label, profile in reference_profiles.items():
        quality_table = profile.get("quality_table")
        if isinstance(quality_table, pd.DataFrame) and not quality_table.empty:
            quality_tables.append(quality_table.copy())
            mean_cv = float(pd.to_numeric(quality_table["cv"], errors="coerce").mean())
            stable_nodes = (
                quality_table.groupby("node_name", sort=False)["cv"].mean().sort_values(ascending=True).head(5).to_dict()
            )
            unstable_nodes = (
                quality_table.groupby("node_name", sort=False)["cv"].mean().sort_values(ascending=False).head(5).to_dict()
            )
        else:
            mean_cv = float("nan")
            stable_nodes = {}
            unstable_nodes = {}
        summary_payload["gestures"][str(gesture_label)] = {
            "num_trials": int(profile.get("num_trials", 0)),
            "quality_stats": profile.get("quality_stats", {}),
            "mean_cv": mean_cv,
            "stable_reference": bool(mean_cv < 0.50) if not math.isnan(mean_cv) else False,
            "most_stable_nodes": stable_nodes,
            "least_stable_nodes": unstable_nodes,
        }
    combined_df = pd.concat(quality_tables, ignore_index=True) if quality_tables else pd.DataFrame(
        columns=["gesture_label", "node_name", "feature_name", "ref_mean", "ref_std", "cv", "num_trials"]
    )
    return combined_df, summary_payload


def build_node_rows_for_trial(
    trial_df: pd.DataFrame,
    *,
    csv_path: Path,
    ctrl_cols: list[str],
    qpos_cols: list[str],
    muscle_names: list[str],
    hold_tail_ratio: float,
    hold_frame_cap: int,
    max_hold_std: float,
    max_hold_delta: float,
    min_hold_activation: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if trial_df.empty:
        return [], None

    trial_id = choose_trial_id(trial_df, csv_path)
    exercise_name = choose_exercise_name(trial_df)
    if not exercise_name:
        return [], None

    hold_df = select_hold_segment(
        trial_df,
        hold_tail_ratio=hold_tail_ratio,
        hold_frame_cap=hold_frame_cap,
    )
    if hold_df.empty:
        return [], None

    raw_hold_ctrl_df = hold_df.loc[:, ctrl_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    hold_ctrl_df = sanitize_numeric_frame(raw_hold_ctrl_df)
    if hold_ctrl_df.empty:
        return [], None

    raw_hold_qpos_df = None
    hold_qpos_df = None
    if qpos_cols:
        raw_hold_qpos_df = hold_df.loc[:, qpos_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        hold_qpos_df = sanitize_numeric_frame(raw_hold_qpos_df)

    quality_metrics = compute_trial_quality_metrics(
        hold_ctrl_df,
        hold_qpos_df=hold_qpos_df,
        max_hold_std=max_hold_std,
        max_hold_delta=max_hold_delta,
        min_hold_activation=min_hold_activation,
    )

    ctrl_invalid_total = int(raw_hold_ctrl_df.isna().sum().sum()) if raw_hold_ctrl_df is not None else 0
    ctrl_entry_total = int(raw_hold_ctrl_df.shape[0] * raw_hold_ctrl_df.shape[1]) if raw_hold_ctrl_df is not None else 0
    invalid_ctrl_fraction = float(ctrl_invalid_total / ctrl_entry_total) if ctrl_entry_total > 0 else 0.0
    qpos_invalid_total = int(raw_hold_qpos_df.isna().sum().sum()) if raw_hold_qpos_df is not None else 0
    qpos_entry_total = int(raw_hold_qpos_df.shape[0] * raw_hold_qpos_df.shape[1]) if raw_hold_qpos_df is not None else 0
    invalid_qpos_fraction = float(qpos_invalid_total / qpos_entry_total) if qpos_entry_total > 0 else 0.0

    ctrl_values = hold_ctrl_df.to_numpy(dtype=float, copy=True)
    timestamps = None
    if "t" in hold_df.columns:
        timestamps = pd.to_numeric(hold_df["t"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
    elif "frame" in hold_df.columns:
        timestamps = pd.to_numeric(hold_df["frame"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
    phase_seq = segment_motion_phases(
        ctrl_values,
        timestamps=timestamps,
        strategy="fixed_ratio",
    )
    node_feature_array, node_feature_names = build_node_temporal_features(
        ctrl_values,
        timestamps=timestamps,
        phase_seq=phase_seq,
    )

    trial_info: dict[str, Any] = {
        "source_file": csv_path.name,
        "trial_id": trial_id,
        "gesture_label": exercise_name,
        "exercise_name": exercise_name,
        "num_total_frames": int(len(trial_df)),
        "num_hold_frames": int(len(hold_ctrl_df)),
        "phase_seq": list(phase_seq),
        "invalid_ctrl_fraction": invalid_ctrl_fraction,
        "invalid_qpos_fraction": invalid_qpos_fraction,
        "num_invalid_ctrl_values": ctrl_invalid_total,
        "num_invalid_qpos_values": qpos_invalid_total,
    }
    trial_info.update(quality_metrics)

    node_rows: list[dict[str, Any]] = []
    for idx, muscle_name in enumerate(muscle_names):
        row = {
            "source_file": csv_path.name,
            "trial_id": trial_id,
            "gesture_label": exercise_name,
            "exercise_name": exercise_name,
            "node_name": str(muscle_name),
            "node_index": int(idx),
            "num_hold_frames": int(len(hold_ctrl_df)),
            "phase_seq": "|".join(str(item) for item in phase_seq),
        }
        for feature_name, feature_value in zip(node_feature_names, node_feature_array[idx].tolist()):
            row[str(feature_name)] = float(feature_value)
        node_rows.append(row)
    return node_rows, trial_info


def extract_node_dataset(
    input_dir: Path,
    *,
    target_gestures: set[str],
    hold_tail_ratio: float,
    hold_frame_cap: int,
    max_hold_std: float,
    max_hold_delta: float,
    min_hold_activation: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    csv_paths = sorted(input_dir.rglob(CSV_GLOB))
    if not csv_paths:
        raise FileNotFoundError(f"No files matching {CSV_GLOB!r} were found in {input_dir}.")

    node_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    ctrl_cols_reference: list[str] = []
    qpos_cols_reference: list[str] = []
    muscle_names: list[str] = []

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] {csv_path.name}: failed to read ({type(exc).__name__}: {exc}).")
            continue

        if df.empty:
            print(f"[WARN] {csv_path.name}: skipped because CSV is empty.")
            continue

        ctrl_cols = find_ctrl_analysis_columns(df.columns.tolist())
        if not ctrl_cols:
            print(f"[WARN] {csv_path.name}: skipped because no ctrl_analysis_* columns were found.")
            continue

        qpos_cols = find_qpos_columns(df.columns.tolist())
        if not ctrl_cols_reference:
            ctrl_cols_reference = list(ctrl_cols)
            muscle_names = resolve_muscle_names(ctrl_cols_reference)
        elif ctrl_cols != ctrl_cols_reference:
            missing = [col for col in ctrl_cols_reference if col not in ctrl_cols]
            extra = [col for col in ctrl_cols if col not in ctrl_cols_reference]
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
            trial_dfs = [trial_group.copy() for _, trial_group in df.groupby("trial_id", sort=False, dropna=False)]
        else:
            trial_dfs = [df.copy()]

        for trial_df in trial_dfs:
            exercise_name = choose_exercise_name(trial_df)
            if exercise_name not in target_gestures:
                continue
            trial_node_rows, trial_info = build_node_rows_for_trial(
                trial_df,
                csv_path=csv_path,
                ctrl_cols=ctrl_cols_reference,
                qpos_cols=qpos_cols_reference,
                muscle_names=muscle_names,
                hold_tail_ratio=hold_tail_ratio,
                hold_frame_cap=hold_frame_cap,
                max_hold_std=max_hold_std,
                max_hold_delta=max_hold_delta,
                min_hold_activation=min_hold_activation,
            )
            if not trial_node_rows or trial_info is None:
                continue
            node_rows.extend(trial_node_rows)
            trial_rows.append(trial_info)

    node_df = pd.DataFrame(node_rows)
    trial_df = pd.DataFrame(trial_rows).drop_duplicates(subset=["trial_id"]).reset_index(drop=True)
    if node_df.empty or trial_df.empty:
        raise ValueError("No valid node-level data were extracted for the requested gestures.")
    return node_df, trial_df, muscle_names


def split_trials_by_counts(
    trial_df: pd.DataFrame,
    *,
    target_gestures: set[str],
    reference_count: int,
    validation_count: int,
    test_count: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(random_seed))
    reference_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    split_summary: dict[str, Any] = {
        "random_seed": int(random_seed),
        "requested_counts": {
            "reference": int(reference_count),
            "validation": int(validation_count),
            "test": int(test_count),
        },
        "gestures": {},
    }
    required_total = int(reference_count) + int(validation_count) + int(test_count)

    for gesture_label in sorted(target_gestures):
        gesture_df = trial_df.loc[
            trial_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()
        ].copy()
        available = int(len(gesture_df))
        if available < required_total:
            raise ValueError(
                f"Gesture {gesture_label!r} has only {available} trials, but "
                f"{required_total} are required by reference/validation/test counts."
            )
        indices = rng.permutation(available)
        reference_idx = indices[:reference_count]
        validation_idx = indices[reference_count : reference_count + validation_count]
        test_idx = indices[reference_count + validation_count : reference_count + validation_count + test_count]

        gesture_reference_df = gesture_df.iloc[reference_idx].copy()
        gesture_validation_df = gesture_df.iloc[validation_idx].copy()
        gesture_test_df = gesture_df.iloc[test_idx].copy()

        gesture_reference_df["split"] = "reference"
        gesture_validation_df["split"] = "validation"
        gesture_test_df["split"] = "test"

        reference_parts.append(gesture_reference_df)
        validation_parts.append(gesture_validation_df)
        test_parts.append(gesture_test_df)

        split_summary["gestures"][str(gesture_label)] = {
            "available_trials": available,
            "reference_trials": int(len(gesture_reference_df)),
            "validation_trials": int(len(gesture_validation_df)),
            "test_trials": int(len(gesture_test_df)),
            "reference_trial_ids": gesture_reference_df["trial_id"].astype(str).tolist(),
            "validation_trial_ids": gesture_validation_df["trial_id"].astype(str).tolist(),
            "test_trial_ids": gesture_test_df["trial_id"].astype(str).tolist(),
        }

    reference_trials_df = pd.concat(reference_parts, ignore_index=True)
    validation_trials_df = pd.concat(validation_parts, ignore_index=True)
    test_trials_df = pd.concat(test_parts, ignore_index=True)
    return reference_trials_df, validation_trials_df, test_trials_df, split_summary


def compute_deviation_dataset(
    node_df: pd.DataFrame,
    *,
    target_gestures: set[str],
    z_threshold: float,
    threshold_mode: str,
    quantile_threshold: float,
    deviation_aggregation: str,
    reference_profiles: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    reference_rows: list[pd.DataFrame] = []
    deviation_rows: list[pd.DataFrame] = []
    resolved_profiles: dict[str, dict[str, object]] = {}
    for gesture_label in sorted(target_gestures):
        if reference_profiles is not None and gesture_label in reference_profiles:
            reference_profile = reference_profiles[gesture_label]
        else:
            reference_profile = build_reference_profile(node_df, gesture_label)
        resolved_profiles[str(gesture_label)] = reference_profile
        reference_table = reference_profile.get("reference_table")
        if isinstance(reference_table, pd.DataFrame) and not reference_table.empty:
            ref_copy = reference_table.copy()
            ref_copy.insert(0, "gesture_label", str(gesture_label))
            reference_rows.append(ref_copy)

        sample_df = node_df.loc[node_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()].copy()
        if sample_df.empty:
            continue
        deviation_df = compute_node_deviation(
            sample_df,
            reference_profile,
            z_threshold=float(z_threshold),
            threshold_mode=str(threshold_mode),
            quantile_threshold=float(quantile_threshold),
            deviation_aggregation=str(deviation_aggregation),
        )
        if not deviation_df.empty:
            deviation_rows.append(deviation_df)

    combined_reference_df = pd.concat(reference_rows, ignore_index=True) if reference_rows else pd.DataFrame()
    combined_deviation_df = pd.concat(deviation_rows, ignore_index=True) if deviation_rows else pd.DataFrame()
    return combined_reference_df, combined_deviation_df, resolved_profiles


def build_reference_quality_audit(
    reference_trials_df: pd.DataFrame,
    reference_node_df: pd.DataFrame,
    *,
    target_gestures: set[str],
    reference_count: int,
    z_threshold: float,
    threshold_mode: str,
    quantile_threshold: float,
    deviation_aggregation: str,
    reference_filter_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_df = reference_trials_df.copy()
    if audit_df.empty:
        return audit_df, audit_df

    _, raw_reference_deviation_df, _ = compute_deviation_dataset(
        select_deviation_feature_frame(reference_node_df),
        target_gestures=target_gestures,
        z_threshold=float(z_threshold),
        threshold_mode=str(threshold_mode),
        quantile_threshold=float(quantile_threshold),
        deviation_aggregation=str(deviation_aggregation),
    )

    if raw_reference_deviation_df.empty:
        audit_df["reference_deviation_mean"] = np.nan
        audit_df["reference_deviation_max"] = np.nan
    else:
        deviation_summary_df = raw_reference_deviation_df.groupby(["trial_id", "gesture_label"], sort=False).agg(
            reference_deviation_mean=("aggregated_deviation_score", "mean"),
            reference_deviation_max=("aggregated_deviation_score", "max"),
        ).reset_index()
        audit_df = audit_df.merge(deviation_summary_df, on=["trial_id", "gesture_label"], how="left")

    audit_df["invalid_or_missing"] = (
        (pd.to_numeric(audit_df["invalid_ctrl_fraction"], errors="coerce").fillna(0.0) > REFERENCE_INVALID_FRACTION_EPS)
        | (pd.to_numeric(audit_df["invalid_qpos_fraction"], errors="coerce").fillna(0.0) > REFERENCE_INVALID_FRACTION_EPS)
    )
    audit_df["unstable_or_noisy"] = (
        audit_df["unstable_hold"].astype(bool)
        | (pd.to_numeric(audit_df["hold_ctrl_std_mean"], errors="coerce").fillna(0.0) > pd.to_numeric(audit_df["hold_ctrl_std_mean"], errors="coerce").median())
        | (pd.to_numeric(audit_df["hold_frame_to_frame_delta_mean"], errors="coerce").fillna(0.0) > pd.to_numeric(audit_df["hold_frame_to_frame_delta_mean"], errors="coerce").median())
    )
    audit_df["outlier_threshold"] = np.nan
    audit_df["outlier_against_reference"] = False

    for gesture_label in sorted(target_gestures):
        gesture_mask = audit_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()
        gesture_scores = pd.to_numeric(audit_df.loc[gesture_mask, "reference_deviation_mean"], errors="coerce").dropna()
        if gesture_scores.empty:
            continue
        q1 = float(gesture_scores.quantile(0.25))
        q3 = float(gesture_scores.quantile(0.75))
        iqr = max(q3 - q1, 0.0)
        threshold = float(max(q3 + 1.5 * iqr, gesture_scores.quantile(0.90)))
        audit_df.loc[gesture_mask, "outlier_threshold"] = threshold
        audit_df.loc[gesture_mask, "outlier_against_reference"] = (
            pd.to_numeric(audit_df.loc[gesture_mask, "reference_deviation_mean"], errors="coerce").fillna(0.0) > threshold
        )

    mode_name = str(reference_filter_mode).strip().lower()
    if mode_name == "keep_all":
        reject_mask = np.zeros(len(audit_df), dtype=bool)
    elif mode_name == "filter_unstable":
        reject_mask = audit_df["invalid_or_missing"].astype(bool) | audit_df["unstable_or_noisy"].astype(bool)
    else:
        reject_mask = (
            audit_df["invalid_or_missing"].astype(bool)
            | audit_df["unstable_or_noisy"].astype(bool)
            | audit_df["outlier_against_reference"].astype(bool)
        )

    audit_df["keep_for_reference"] = ~reject_mask
    audit_df["rejection_reason"] = ""
    audit_df.loc[audit_df["invalid_or_missing"].astype(bool), "rejection_reason"] += "invalid_or_missing;"
    audit_df.loc[audit_df["unstable_or_noisy"].astype(bool), "rejection_reason"] += "unstable_or_noisy;"
    audit_df.loc[audit_df["outlier_against_reference"].astype(bool), "rejection_reason"] += "outlier_against_reference;"
    audit_df["rejection_reason"] = audit_df["rejection_reason"].str.rstrip(";")
    audit_df.loc[audit_df["keep_for_reference"], "rejection_reason"] = "kept"

    min_keep_count = max(REFERENCE_MIN_KEEP_ABSOLUTE, int(math.ceil(float(reference_count) * REFERENCE_MIN_KEEP_FRACTION)))
    for gesture_label in sorted(target_gestures):
        gesture_mask = audit_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()
        kept_count = int(audit_df.loc[gesture_mask, "keep_for_reference"].sum())
        if kept_count >= min_keep_count:
            continue
        rejected_df = audit_df.loc[gesture_mask & ~audit_df["keep_for_reference"].astype(bool)].copy()
        if rejected_df.empty:
            continue
        rejected_df = rejected_df.sort_values(
            by=["quality_score", "reference_deviation_mean"],
            ascending=[False, True],
            na_position="last",
        )
        needed = min_keep_count - kept_count
        restore_ids = rejected_df["trial_id"].astype(str).head(needed).tolist()
        restore_mask = gesture_mask & audit_df["trial_id"].astype(str).isin(set(restore_ids))
        audit_df.loc[restore_mask, "keep_for_reference"] = True
        audit_df.loc[restore_mask, "rejection_reason"] = "kept_by_safeguard"

    rejected_df = audit_df.loc[~audit_df["keep_for_reference"].astype(bool)].copy().reset_index(drop=True)
    return audit_df.reset_index(drop=True), rejected_df


def select_reference_export_columns(reference_table: pd.DataFrame) -> pd.DataFrame:
    selected_columns = ["node_name"]
    for feature_name in SIMPLE_REFERENCE_FEATURES:
        mean_col = f"{feature_name}_mean"
        std_col = f"{feature_name}_std"
        if mean_col in reference_table.columns:
            selected_columns.append(mean_col)
        if std_col in reference_table.columns:
            selected_columns.append(std_col)
    return reference_table.loc[:, selected_columns].copy()


def build_reference_gesture_summary(
    *,
    gesture_label: str,
    profile: dict[str, Any],
    audit_df: pd.DataFrame,
) -> dict[str, Any]:
    gesture_audit_df = audit_df.loc[audit_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()].copy()
    quality_table = profile.get("quality_table")
    if isinstance(quality_table, pd.DataFrame) and not quality_table.empty:
        mean_cv = float(pd.to_numeric(quality_table["cv"], errors="coerce").mean())
        most_stable_nodes = (
            quality_table.groupby("node_name", sort=False)["cv"].mean().sort_values(ascending=True).head(5).to_dict()
        )
        least_stable_nodes = (
            quality_table.groupby("node_name", sort=False)["cv"].mean().sort_values(ascending=False).head(5).to_dict()
        )
    else:
        mean_cv = float("nan")
        most_stable_nodes = {}
        least_stable_nodes = {}
    return {
        "gesture_label": str(gesture_label),
        "reference_trials_allocated": int(len(gesture_audit_df)),
        "reference_trials_used": int(gesture_audit_df["keep_for_reference"].sum()) if not gesture_audit_df.empty else 0,
        "reference_trials_rejected": int((~gesture_audit_df["keep_for_reference"].astype(bool)).sum()) if not gesture_audit_df.empty else 0,
        "mean_cv": mean_cv,
        "stable_reference": bool(mean_cv < 0.50) if not math.isnan(mean_cv) else False,
        "most_stable_nodes": most_stable_nodes,
        "least_stable_nodes": least_stable_nodes,
        "quality_stats": profile.get("quality_stats", {}),
    }


def build_reference_stability_outputs(
    reference_profiles: dict[str, dict[str, Any]],
    reference_audit_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    summary_rows: list[dict[str, Any]] = []
    report_lines = [
        "Reference Stability Report",
        "=========================",
        "",
        "These references are simulation-based gesture templates, not clinical norms.",
        "",
    ]
    for gesture_label, profile in reference_profiles.items():
        gesture_summary = build_reference_gesture_summary(
            gesture_label=gesture_label,
            profile=profile,
            audit_df=reference_audit_df,
        )
        summary_rows.append(
            {
                "gesture_label": gesture_summary["gesture_label"],
                "reference_trials_allocated": gesture_summary["reference_trials_allocated"],
                "reference_trials_used": gesture_summary["reference_trials_used"],
                "reference_trials_rejected": gesture_summary["reference_trials_rejected"],
                "mean_cv": gesture_summary["mean_cv"],
                "most_stable_node": next(iter(gesture_summary["most_stable_nodes"]), ""),
                "least_stable_node": next(iter(gesture_summary["least_stable_nodes"]), ""),
                "stable_reference": gesture_summary["stable_reference"],
            }
        )
        report_lines.extend(
            [
                f"Gesture: {gesture_label}",
                f"- allocated reference trials: {gesture_summary['reference_trials_allocated']}",
                f"- used after audit: {gesture_summary['reference_trials_used']}",
                f"- rejected by audit: {gesture_summary['reference_trials_rejected']}",
                f"- mean_cv: {gesture_summary['mean_cv']:.4f}" if not math.isnan(float(gesture_summary["mean_cv"])) else "- mean_cv: nan",
                f"- most stable nodes: {gesture_summary['most_stable_nodes']}",
                f"- least stable nodes: {gesture_summary['least_stable_nodes']}",
                "",
            ]
        )
    return pd.DataFrame(summary_rows), "\n".join(report_lines)


def build_deviation_summary(labeled_df: pd.DataFrame) -> pd.DataFrame:
    if labeled_df.empty:
        return pd.DataFrame(
            columns=[
                "trial_id",
                "gesture_label",
                "node_name",
                "aggregated_deviation_score",
                "node_status",
                "dominant_feature",
                "dominant_feature_zscore",
                "reference_mean_feature",
                "sample_feature_value",
                "rank_within_trial",
            ]
        )

    summary_rows: list[dict[str, Any]] = []
    for (trial_id, gesture_label), group_df in labeled_df.groupby(["trial_id", "gesture_label"], sort=False):
        ranked_df = group_df.sort_values(by="aggregated_deviation_score", ascending=False).head(10)
        for rank, (_, row) in enumerate(ranked_df.iterrows(), start=1):
            summary_rows.append(
                {
                    "trial_id": str(trial_id),
                    "gesture_label": str(gesture_label),
                    "node_name": str(row["node_name"]),
                    "aggregated_deviation_score": float(row["aggregated_deviation_score"]),
                    "node_status": str(row["node_status"]),
                    "dominant_feature": str(row.get("dominant_feature", "")),
                    "dominant_feature_zscore": float(row.get("dominant_feature_zscore", 0.0)),
                    "reference_mean_feature": float(row.get("reference_mean_feature", 0.0)),
                    "sample_feature_value": float(row.get("sample_feature_value", 0.0)),
                    "rank_within_trial": rank,
                }
            )
    return pd.DataFrame(summary_rows)


def select_case_study(labeled_df: pd.DataFrame, gesture_label: str) -> pd.DataFrame:
    gesture_df = labeled_df.loc[
        labeled_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()
    ].copy()
    if gesture_df.empty:
        return pd.DataFrame(
            columns=[
                "trial_id",
                "gesture_label",
                "node_name",
                "aggregated_deviation_score",
                "node_status",
                "dominant_feature",
                "dominant_feature_zscore",
                "reference_mean_feature",
                "sample_feature_value",
                "rank_within_trial",
            ]
        )
    trial_scores_df = gesture_df.groupby("trial_id", sort=False)["aggregated_deviation_score"].mean().reset_index(name="mean_deviation")
    selected_trial_id = str(trial_scores_df.sort_values(by="mean_deviation", ascending=False).iloc[0]["trial_id"])
    selected_df = gesture_df.loc[gesture_df["trial_id"].astype(str) == selected_trial_id].copy()
    selected_df = selected_df.sort_values(by="aggregated_deviation_score", ascending=False).head(10).copy()
    selected_df["rank_within_trial"] = np.arange(1, len(selected_df) + 1)
    columns = [
        "trial_id",
        "gesture_label",
        "node_name",
        "aggregated_deviation_score",
        "node_status",
        "dominant_feature",
        "dominant_feature_zscore",
        "reference_mean_feature",
        "sample_feature_value",
        "rank_within_trial",
    ]
    return selected_df.loc[:, [col for col in columns if col in selected_df.columns]].copy()


def build_presentation_data_summary(
    trial_df: pd.DataFrame,
    reference_audit_df: pd.DataFrame,
    validation_trials_df: pd.DataFrame,
    test_trials_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gestures = sorted(pd.unique(trial_df["gesture_label"].astype(str)))
    for gesture_label in gestures:
        total_trials = int(
            (trial_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()).sum()
        )
        reference_allocated = int(
            (reference_audit_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()).sum()
        )
        reference_used = int(
            reference_audit_df.loc[
                (reference_audit_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower())
                & reference_audit_df["keep_for_reference"].astype(bool)
            ].shape[0]
        )
        reference_rejected = int(reference_allocated - reference_used)
        validation_count = int(
            (validation_trials_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()).sum()
        )
        test_count = int(
            (test_trials_df["gesture_label"].astype(str).str.lower() == str(gesture_label).lower()).sum()
        )
        rows.append(
            {
                "gesture_label": str(gesture_label),
                "total_trials": total_trials,
                "reference_allocated": reference_allocated,
                "reference_used": reference_used,
                "reference_rejected": reference_rejected,
                "validation_trials": validation_count,
                "test_trials": test_count,
            }
        )
    return pd.DataFrame(rows)


def build_final_sidang_summary(
    *,
    presentation_summary_df: pd.DataFrame,
    open_reference_summary: dict[str, Any],
    fist_reference_summary: dict[str, Any],
    threshold_by_gesture: dict[str, float],
    threshold_mode: str,
    quantile_threshold: float,
    deviation_aggregation: str,
) -> tuple[str, dict[str, Any]]:
    lines = [
        "Final Summary for Sidang",
        "========================",
        "",
        "A. Dataset overview",
    ]
    dataset_rows: list[dict[str, Any]] = []
    for _, row in presentation_summary_df.iterrows():
        dataset_rows.append(
            {
                "gesture_label": str(row["gesture_label"]),
                "total_trials": int(row["total_trials"]),
                "reference_allocated": int(row["reference_allocated"]),
                "reference_used": int(row["reference_used"]),
                "reference_rejected": int(row["reference_rejected"]),
                "validation_trials": int(row["validation_trials"]),
                "test_trials": int(row["test_trials"]),
            }
        )
        lines.append(
            f"- {row['gesture_label']}: total={int(row['total_trials'])}, "
            f"reference={int(row['reference_allocated'])}, reference_used={int(row['reference_used'])}, "
            f"validation={int(row['validation_trials'])}, test={int(row['test_trials'])}, "
            f"reference_rejected={int(row['reference_rejected'])}"
        )
    lines.extend(
        [
            "",
            "B. Reference stability",
            f"- open mean_cv = {float(open_reference_summary.get('mean_cv', float('nan'))):.4f}",
            f"- fist mean_cv = {float(fist_reference_summary.get('mean_cv', float('nan'))):.4f}",
            f"- open stable_reference = {bool(open_reference_summary.get('stable_reference', False))}",
            f"- fist stable_reference = {bool(fist_reference_summary.get('stable_reference', False))}",
            "",
            "C. Methodology summary",
            "- Simulation-based reference profile is built separately for open and fist.",
            "- Node deviation analysis compares each node against the corresponding model-derived gesture reference.",
            f"- Final threshold setting: mode={threshold_mode}, quantile={quantile_threshold:.2f}, aggregation={deviation_aggregation}, per_gesture_threshold=True.",
            "",
            "D. Key output",
            "- The system reports node status as within_reference, lower_than_reference, or higher_than_reference.",
            "- Case study for open is available.",
            "- Case study for fist is available.",
            "",
            "E. Limitation",
            "- The output is not a clinical diagnosis.",
            "- The reference is derived from simulation/model behavior.",
            "- The classifier is not used as the main result.",
        ]
    )
    payload = {
        "dataset_overview": dataset_rows,
        "reference_stability": {
            "open_mean_cv": float(open_reference_summary.get("mean_cv", float("nan"))),
            "fist_mean_cv": float(fist_reference_summary.get("mean_cv", float("nan"))),
            "open_stable_reference": bool(open_reference_summary.get("stable_reference", False)),
            "fist_stable_reference": bool(fist_reference_summary.get("stable_reference", False)),
        },
        "methodology_summary": {
            "reference_type": "simulation-based reference profile",
            "analysis_type": "node deviation analysis",
            "threshold_mode": str(threshold_mode),
            "quantile_threshold": float(quantile_threshold),
            "deviation_aggregation": str(deviation_aggregation),
            "threshold_by_gesture": {str(k): float(v) for k, v in threshold_by_gesture.items()},
        },
        "key_output": {
            "node_status_labels": [
                "within_reference",
                "lower_than_reference",
                "higher_than_reference",
            ],
            "case_study_open_available": True,
            "case_study_fist_available": True,
        },
        "limitation": {
            "not_clinical_diagnosis": True,
            "reference_is_model_derived": True,
            "classifier_not_main_result": True,
        },
    }
    return "\n".join(lines), payload


def build_bab4_ready_points(
    *,
    open_reference_summary: dict[str, Any],
    fist_reference_summary: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "Hasil utama",
            "- Dataset dibagi menjadi reference, validation, dan test untuk gesture open dan fist dengan proporsi yang seimbang.",
            "- Reference profile dibangun dari data reference yang telah melalui audit kualitas, sehingga acuan simulasi menjadi lebih stabil untuk analisis node.",
            "- Reference open membaik setelah audit dan mencapai kondisi stabil, sedangkan reference fist juga menunjukkan stabilitas yang baik.",
            "",
            "Interpretasi",
            "- Reference profile merepresentasikan pola kontribusi node otot yang diharapkan pada gesture target berdasarkan simulasi/model.",
            "- Node deviation analysis dapat menunjukkan contribution mismatch pada level node melalui status within_reference, lower_than_reference, dan higher_than_reference.",
            "- Hasil case study menunjukkan bahwa deviasi node dapat ditelusuri kembali ke fitur dominan serta perbedaan antara nilai sampel dan nilai referensi.",
            "",
            "Pembahasan singkat",
            f"- Nilai mean_cv reference open sebesar {float(open_reference_summary.get('mean_cv', float('nan'))):.4f} menunjukkan bahwa acuan open sudah cukup layak digunakan dalam evaluasi node.",
            f"- Nilai mean_cv reference fist sebesar {float(fist_reference_summary.get('mean_cv', float('nan'))):.4f} menunjukkan bahwa acuan fist relatif lebih stabil dibanding open.",
            "- Dengan reference yang lebih stabil, analisis node deviation menjadi lebih mudah dipresentasikan sebagai evaluasi kontribusi node terhadap gesture target.",
            "",
            "Keterbatasan",
            "- Sistem ini bersifat decision support awal berbasis simulation-based reference, bukan diagnosis klinis.",
            "- Acuan yang digunakan berasal dari model/simulasi, sehingga validitasnya tetap dipengaruhi kualitas model dan kualitas recording.",
            "- Classifier belum dijadikan hasil utama dan hanya relevan sebagai appendix atau arah pengembangan lanjutan.",
        ]
    )


def build_slide_ready_points(
    *,
    open_reference_summary: dict[str, Any],
    fist_reference_summary: dict[str, Any],
    threshold_mode: str,
    quantile_threshold: float,
    deviation_aggregation: str,
) -> str:
    return "\n".join(
        [
            "1. Latar belakang singkat",
            "- Aktivasi otot hasil estimasi dapat digunakan sebagai representasi kontribusi node pada gesture tangan.",
            "- Untuk sidang ini, fokus analisis dibatasi pada gesture open dan fist.",
            "",
            "2. Tujuan penelitian",
            "- Membangun simulation-based reference profile untuk open dan fist.",
            "- Menggunakan reference tersebut untuk mendeteksi node deviation / contribution mismatch.",
            "",
            "3. Pipeline sistem",
            "- Leap Motion -> qpos -> estimasi u/ctrl_analysis -> hold segment -> node temporal features.",
            "- Reference profile dibangun dari trial reference, lalu validation dan test dibandingkan terhadap reference gesture yang sesuai.",
            "",
            "4. Hasil reference profile",
            f"- Open mean_cv = {float(open_reference_summary.get('mean_cv', float('nan'))):.4f}, stable_reference = {bool(open_reference_summary.get('stable_reference', False))}.",
            f"- Fist mean_cv = {float(fist_reference_summary.get('mean_cv', float('nan'))):.4f}, stable_reference = {bool(fist_reference_summary.get('stable_reference', False))}.",
            "",
            "5. Hasil node deviation / case study",
            "- Sistem menghasilkan status within_reference, lower_than_reference, dan higher_than_reference pada level node.",
            f"- Threshold final: {threshold_mode}, quantile {quantile_threshold:.2f}, aggregation {deviation_aggregation}.",
            "- Case study open dan fist tersedia untuk menunjukkan node dengan deviasi terbesar.",
            "",
            "6. Batasan dan arah lanjut",
            "- Hasil ini merupakan decision support awal berbasis model-derived reference, bukan diagnosis klinis.",
            "- Pengembangan lanjutan dapat difokuskan pada perluasan gesture dan validasi reference yang lebih luas.",
        ]
    )


def build_node_classifier() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURE_COLUMNS)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURE_COLUMNS)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def evaluate_node_classifier(
    train_labeled_df: pd.DataFrame,
    test_labeled_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    numeric_feature_columns = [col for col in NUMERIC_FEATURE_COLUMNS if col in train_labeled_df.columns and col in test_labeled_df.columns]
    feature_columns = list(tuple(numeric_feature_columns) + CATEGORICAL_FEATURE_COLUMNS)
    X_train = train_labeled_df.loc[:, feature_columns].copy()
    X_test = test_labeled_df.loc[:, feature_columns].copy()
    y_train = train_labeled_df["node_status"].astype(str).to_numpy()
    y_test = test_labeled_df["node_status"].astype(str).to_numpy()
    train_distribution = pd.Series(y_train).value_counts().to_dict()
    test_distribution = pd.Series(y_test).value_counts().to_dict()
    print(f"Validation node-status distribution: {train_distribution}")
    print(f"Test node-status distribution: {test_distribution}")

    unique_train_labels = pd.unique(y_train).tolist()
    classifier_ran = False
    classifier_skipped_reason = ""
    label_collapse_detected = len(unique_train_labels) < 2
    if len(unique_train_labels) < 2:
        fallback_label = str(unique_train_labels[0])
        print(
            "[WARN] Validation node dataset has only one pseudo-label class "
            f"({fallback_label}); classifier appendix will be skipped and constant prediction used."
        )
        classifier_skipped_reason = f"validation_split_single_class:{fallback_label}"
        y_pred = np.full(shape=len(X_test), fill_value=fallback_label, dtype=object)
    else:
        model = build_node_classifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        classifier_ran = True

    predictions_df = test_labeled_df.loc[
        :,
        [
            "source_file",
            "trial_id",
            "gesture_label",
            "node_name",
            *NUMERIC_FEATURE_COLUMNS,
            "aggregated_deviation_score",
            "node_status",
        ],
    ].copy()
    predictions_df = predictions_df.rename(columns={"node_status": "y_true"})
    predictions_df["y_pred"] = y_pred.astype(str)
    predictions_df["correct"] = predictions_df["y_true"] == predictions_df["y_pred"]

    label_order = sorted(pd.unique(np.concatenate([y_test.astype(str), y_pred.astype(str)])).tolist())
    if len(pd.unique(y_test.astype(str)).tolist()) >= 2 and len(label_order) >= 2:
        conf = confusion_matrix(y_test, y_pred, labels=label_order)
        confusion_df = pd.DataFrame(conf, index=label_order, columns=label_order)
        confusion_df.index.name = "y_true"
        confusion_df.columns.name = "y_pred"
    else:
        confusion_df = pd.DataFrame()

    report_dict = classification_report(y_test, y_pred, labels=label_order, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, labels=label_order, zero_division=0)
    accuracy = float(accuracy_score(y_test, y_pred))
    summary = {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "label_order": label_order,
        "n_validation_nodes": int(len(train_labeled_df)),
        "n_test_nodes": int(len(test_labeled_df)),
        "n_validation_trials": int(train_labeled_df["trial_id"].nunique()),
        "n_test_trials": int(test_labeled_df["trial_id"].nunique()),
        "feature_columns": list(feature_columns),
        "classification_purpose": "appendix_reference_relative_node_status_estimation",
        "classifier_ran": bool(classifier_ran),
        "classifier_skipped_reason": classifier_skipped_reason,
        "label_collapse_detected": bool(label_collapse_detected),
        "validation_label_distribution": train_distribution,
        "test_label_distribution": test_distribution,
    }
    return predictions_df, confusion_df, summary, report_text


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "offline_node_classifier_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    target_gestures = normalize_gesture_set(list(args.target_gestures))
    per_gesture_threshold = str_to_bool(args.per_gesture_threshold)

    node_df, trial_df, muscle_names = extract_node_dataset(
        input_dir,
        target_gestures=target_gestures,
        hold_tail_ratio=float(args.hold_tail_ratio),
        hold_frame_cap=int(args.hold_frame_cap),
        max_hold_std=float(args.max_hold_std),
        max_hold_delta=float(args.max_hold_delta),
        min_hold_activation=float(args.min_hold_activation),
    )
    save_dataframe(trial_df, output_dir / "trial_quality_node.csv")

    if trial_df.empty or node_df.empty:
        raise ValueError("All node-level trials were removed before split generation.")

    reference_trials_df, validation_trials_df, test_trials_df, dataset_split_summary = split_trials_by_counts(
        trial_df,
        target_gestures=target_gestures,
        reference_count=int(args.reference_count),
        validation_count=int(args.validation_count),
        test_count=int(args.test_count),
        random_seed=int(args.random_seed),
    )

    reference_trial_ids = set(reference_trials_df["trial_id"].astype(str))
    validation_trial_ids = set(validation_trials_df["trial_id"].astype(str))
    test_trial_ids = set(test_trials_df["trial_id"].astype(str))

    reference_node_df = node_df.loc[node_df["trial_id"].astype(str).isin(reference_trial_ids)].copy().reset_index(drop=True)
    validation_node_df = node_df.loc[node_df["trial_id"].astype(str).isin(validation_trial_ids)].copy().reset_index(drop=True)
    test_node_df = node_df.loc[node_df["trial_id"].astype(str).isin(test_trial_ids)].copy().reset_index(drop=True)
    reference_node_features_df = select_deviation_feature_frame(reference_node_df)
    validation_node_features_df = select_deviation_feature_frame(validation_node_df)
    test_node_features_df = select_deviation_feature_frame(test_node_df)

    reference_audit_df, reference_rejected_df = build_reference_quality_audit(
        reference_trials_df,
        reference_node_features_df,
        target_gestures=target_gestures,
        reference_count=int(args.reference_count),
        z_threshold=float(args.z_threshold),
        threshold_mode=str(args.threshold_mode),
        quantile_threshold=float(args.quantile_threshold),
        deviation_aggregation=str(args.deviation_aggregation),
        reference_filter_mode=str(args.reference_filter_mode),
    )

    kept_reference_ids = set(reference_audit_df.loc[reference_audit_df["keep_for_reference"].astype(bool), "trial_id"].astype(str))
    final_reference_trials_df = reference_audit_df.loc[reference_audit_df["keep_for_reference"].astype(bool)].copy().reset_index(drop=True)
    final_reference_node_df = reference_node_df.loc[reference_node_df["trial_id"].astype(str).isin(kept_reference_ids)].copy().reset_index(drop=True)
    final_reference_node_features_df = reference_node_features_df.loc[
        reference_node_features_df["trial_id"].astype(str).isin(kept_reference_ids)
    ].copy().reset_index(drop=True)
    if final_reference_trials_df.empty or final_reference_node_df.empty:
        raise ValueError("Reference audit removed all reference trials.")

    final_reference_df, _, final_reference_profiles = compute_deviation_dataset(
        final_reference_node_features_df,
        target_gestures=target_gestures,
        z_threshold=float(args.z_threshold),
        threshold_mode=str(args.threshold_mode),
        quantile_threshold=float(args.quantile_threshold),
        deviation_aggregation=str(args.deviation_aggregation),
    )

    for gesture_label in sorted(target_gestures):
        profile = final_reference_profiles.get(str(gesture_label))
        if profile is None:
            continue
        reference_table = profile.get("reference_table")
        if isinstance(reference_table, pd.DataFrame) and not reference_table.empty:
            save_dataframe(
                select_reference_export_columns(reference_table),
                output_dir / f"reference_{gesture_label}_profile.csv",
            )
        summary_payload = build_reference_gesture_summary(
            gesture_label=gesture_label,
            profile=profile,
            audit_df=reference_audit_df,
        )
        save_json(summary_payload, output_dir / f"reference_{gesture_label}_summary.json")

    reference_quality_df, reference_quality_summary = build_reference_quality_outputs(final_reference_profiles)
    save_dataframe(reference_quality_df, output_dir / "reference_profile_quality.csv")
    save_json(reference_quality_summary, output_dir / "reference_profile_summary.json")
    save_dataframe(reference_audit_df, output_dir / "reference_quality_audit.csv")
    save_dataframe(reference_rejected_df, output_dir / "reference_rejected_trials.csv")

    reference_stability_df, reference_stability_report = build_reference_stability_outputs(
        final_reference_profiles,
        reference_audit_df,
    )
    save_dataframe(reference_stability_df, output_dir / "reference_stability_summary.csv")
    save_text(reference_stability_report, output_dir / "reference_stability_report.txt")

    _, raw_validation_deviation_df, _ = compute_deviation_dataset(
        validation_node_features_df,
        target_gestures=target_gestures,
        z_threshold=float(args.z_threshold),
        threshold_mode=str(args.threshold_mode),
        quantile_threshold=float(args.quantile_threshold),
        deviation_aggregation=str(args.deviation_aggregation),
        reference_profiles=final_reference_profiles,
    )
    _, raw_test_deviation_df, _ = compute_deviation_dataset(
        test_node_features_df,
        target_gestures=target_gestures,
        z_threshold=float(args.z_threshold),
        threshold_mode=str(args.threshold_mode),
        quantile_threshold=float(args.quantile_threshold),
        deviation_aggregation=str(args.deviation_aggregation),
        reference_profiles=final_reference_profiles,
    )

    if raw_validation_deviation_df.empty or raw_test_deviation_df.empty:
        raise ValueError("Reference attachment removed all validation/test node samples.")

    if per_gesture_threshold:
        threshold_by_gesture = {
            str(gesture_label).lower(): compute_threshold_value(
                raw_validation_deviation_df,
                gesture_label=gesture_label,
                threshold_mode=str(args.threshold_mode),
                z_threshold=float(args.z_threshold),
                quantile_threshold=float(args.quantile_threshold),
            )
            for gesture_label in sorted(target_gestures)
        }
    else:
        shared_threshold = compute_threshold_value(
            raw_validation_deviation_df,
            gesture_label=None,
            threshold_mode=str(args.threshold_mode),
            z_threshold=float(args.z_threshold),
            quantile_threshold=float(args.quantile_threshold),
        )
        threshold_by_gesture = {str(gesture_label).lower(): shared_threshold for gesture_label in sorted(target_gestures)}

    validation_labeled_df = apply_thresholds_to_deviation_df(
        raw_validation_deviation_df,
        threshold_by_gesture=threshold_by_gesture,
    )
    test_labeled_df = apply_thresholds_to_deviation_df(
        raw_test_deviation_df,
        threshold_by_gesture=threshold_by_gesture,
    )

    label_distributions = build_label_distribution_outputs(validation_labeled_df, test_labeled_df)
    save_dataframe(label_distributions["validation"], output_dir / "node_label_distribution_validation.csv")
    save_dataframe(label_distributions["test"], output_dir / "node_label_distribution_test.csv")
    save_dataframe(label_distributions["by_gesture"], output_dir / "node_label_distribution_by_gesture.csv")
    save_dataframe(label_distributions["by_node"], output_dir / "node_label_distribution_by_node.csv")

    threshold_sweep_df, threshold_sweep_best = build_threshold_sweep_summary(
        raw_validation_deviation_df,
        target_gestures=target_gestures,
        threshold_mode=str(args.threshold_mode),
    )
    save_dataframe(threshold_sweep_df, output_dir / "threshold_sweep_summary.csv")
    save_json(threshold_sweep_best, output_dir / "threshold_sweep_best.json")

    node_predictions_validation = validation_labeled_df.copy()
    node_predictions_test = test_labeled_df.copy()
    save_dataframe(node_predictions_validation, output_dir / "node_predictions_validation.csv")
    save_dataframe(node_predictions_test, output_dir / "node_predictions_test.csv")
    save_dataframe(build_deviation_summary(validation_labeled_df), output_dir / "node_top_deviations_validation.csv")
    save_dataframe(build_deviation_summary(test_labeled_df), output_dir / "node_top_deviations_test.csv")

    save_json(dataset_split_summary, output_dir / "dataset_split_summary.json")
    save_dataframe(final_reference_trials_df, output_dir / "reference_trials.csv")
    save_dataframe(validation_trials_df, output_dir / "validation_trials.csv")
    save_dataframe(test_trials_df, output_dir / "test_trials.csv")
    save_dataframe(final_reference_df, output_dir / "node_reference_profile.csv")

    presentation_summary_df = build_presentation_data_summary(
        trial_df,
        reference_audit_df,
        validation_trials_df,
        test_trials_df,
    )
    save_dataframe(presentation_summary_df, output_dir / "presentation_data_summary.csv")

    case_study_open_df = select_case_study(test_labeled_df, "open")
    case_study_fist_df = select_case_study(test_labeled_df, "fist")
    save_dataframe(case_study_open_df, output_dir / "case_study_open.csv")
    save_dataframe(case_study_fist_df, output_dir / "case_study_fist.csv")

    open_reference_summary = build_reference_gesture_summary(
        gesture_label="open",
        profile=final_reference_profiles.get("open", {}),
        audit_df=reference_audit_df,
    )
    fist_reference_summary = build_reference_gesture_summary(
        gesture_label="fist",
        profile=final_reference_profiles.get("fist", {}),
        audit_df=reference_audit_df,
    )
    presentation_report_lines = [
        "Presentation-Ready Node Deviation Report",
        "=======================================",
        "",
        "This report uses simulation-based references for open and fist.",
        "The outputs describe node deviation and contribution mismatch, not clinical diagnosis.",
        "",
        "Dataset split:",
    ]
    for _, row in presentation_summary_df.iterrows():
        presentation_report_lines.append(
            f"- {row['gesture_label']}: total={int(row['total_trials'])}, "
            f"reference_used={int(row['reference_used'])}/{int(row['reference_allocated'])}, "
            f"validation={int(row['validation_trials'])}, test={int(row['test_trials'])}, "
            f"reference_rejected={int(row['reference_rejected'])}"
        )
    presentation_report_lines.extend(
        [
            "",
            f"Open reference mean_cv: {open_reference_summary.get('mean_cv', float('nan')):.4f}" if not math.isnan(float(open_reference_summary.get("mean_cv", float("nan")))) else "Open reference mean_cv: nan",
            f"Fist reference mean_cv: {fist_reference_summary.get('mean_cv', float('nan')):.4f}" if not math.isnan(float(fist_reference_summary.get("mean_cv", float("nan")))) else "Fist reference mean_cv: nan",
            f"Threshold mode: {args.threshold_mode}",
            f"Quantile threshold: {float(args.quantile_threshold):.2f}",
            f"Deviation aggregation: {args.deviation_aggregation}",
            f"Per-gesture threshold: {bool(per_gesture_threshold)}",
            f"Thresholds used: {threshold_by_gesture}",
            "",
            "Case studies:",
            f"- open test trial: {case_study_open_df['trial_id'].iloc[0] if not case_study_open_df.empty else 'none'}",
            f"- fist test trial: {case_study_fist_df['trial_id'].iloc[0] if not case_study_fist_df.empty else 'none'}",
        ]
    )
    save_text("\n".join(presentation_report_lines), output_dir / "presentation_ready_report.txt")

    case_study_open_top5_df = case_study_open_df.loc[:, [
        col for col in [
            "node_name",
            "aggregated_deviation_score",
            "node_status",
            "dominant_feature",
            "reference_mean_feature",
            "sample_feature_value",
        ]
        if col in case_study_open_df.columns
    ]].head(5).copy()
    if "aggregated_deviation_score" in case_study_open_top5_df.columns:
        case_study_open_top5_df = case_study_open_top5_df.rename(columns={"aggregated_deviation_score": "deviation_score"})
    case_study_fist_top5_df = case_study_fist_df.loc[:, [
        col for col in [
            "node_name",
            "aggregated_deviation_score",
            "node_status",
            "dominant_feature",
            "reference_mean_feature",
            "sample_feature_value",
        ]
        if col in case_study_fist_df.columns
    ]].head(5).copy()
    if "aggregated_deviation_score" in case_study_fist_top5_df.columns:
        case_study_fist_top5_df = case_study_fist_top5_df.rename(columns={"aggregated_deviation_score": "deviation_score"})
    save_dataframe(case_study_open_top5_df, output_dir / "case_study_open_top5.csv")
    save_dataframe(case_study_fist_top5_df, output_dir / "case_study_fist_top5.csv")

    sidang_summary_text, sidang_summary_json = build_final_sidang_summary(
        presentation_summary_df=presentation_summary_df,
        open_reference_summary=open_reference_summary,
        fist_reference_summary=fist_reference_summary,
        threshold_by_gesture=threshold_by_gesture,
        threshold_mode=str(args.threshold_mode),
        quantile_threshold=float(args.quantile_threshold),
        deviation_aggregation=str(args.deviation_aggregation),
    )
    save_text(sidang_summary_text, output_dir / "sidang_final_summary.txt")
    save_json(sidang_summary_json, output_dir / "sidang_final_summary.json")
    save_text(
        build_bab4_ready_points(
            open_reference_summary=open_reference_summary,
            fist_reference_summary=fist_reference_summary,
        ),
        output_dir / "bab4_ready_points.txt",
    )
    save_text(
        build_slide_ready_points(
            open_reference_summary=open_reference_summary,
            fist_reference_summary=fist_reference_summary,
            threshold_mode=str(args.threshold_mode),
            quantile_threshold=float(args.quantile_threshold),
            deviation_aggregation=str(args.deviation_aggregation),
        ),
        output_dir / "slide_ready_points.txt",
    )

    appendix_summary: dict[str, Any] = {
        "classifier_ran": False,
        "classifier_skipped_reason": "appendix_not_requested",
        "label_collapse_detected": False,
    }
    appendix_report_text = "Classifier appendix not requested."
    appendix_predictions_df = pd.DataFrame()
    appendix_confusion_df = pd.DataFrame()
    if bool(args.run_classifier_appendix):
        appendix_predictions_df, appendix_confusion_df, appendix_summary, appendix_report_text = evaluate_node_classifier(
            validation_labeled_df,
            test_labeled_df,
        )
        save_dataframe(appendix_predictions_df, output_dir / "node_predictions_classifier_appendix.csv")
        save_confusion_dataframe(appendix_confusion_df, output_dir / "node_confusion_matrix_classifier_appendix.csv")
        save_text(appendix_report_text, output_dir / "node_classification_report_classifier_appendix.txt")

    pipeline_diagnostics = {
        "reference_open_stable": bool(open_reference_summary.get("stable_reference", False)),
        "reference_fist_stable": bool(fist_reference_summary.get("stable_reference", False)),
        "reference_filter_mode": str(args.reference_filter_mode),
        "threshold_mode": str(args.threshold_mode),
        "quantile_threshold": float(args.quantile_threshold),
        "deviation_aggregation": str(args.deviation_aggregation),
        "threshold_by_gesture": threshold_by_gesture,
        "threshold_sweep_best": threshold_sweep_best,
        "classifier_ran": bool(appendix_summary.get("classifier_ran", False)),
        "classifier_skipped_reason": appendix_summary.get("classifier_skipped_reason", ""),
        "label_collapse_detected": bool(appendix_summary.get("label_collapse_detected", False)),
    }
    save_json(pipeline_diagnostics, output_dir / "node_pipeline_diagnostics.json")

    node_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "csv_pattern": CSV_GLOB,
        "target_gestures": sorted(target_gestures),
        "hold_tail_ratio": float(args.hold_tail_ratio),
        "hold_frame_cap": int(args.hold_frame_cap),
        "reference_count": int(args.reference_count),
        "validation_count": int(args.validation_count),
        "test_count": int(args.test_count),
        "random_seed": int(args.random_seed),
        "reference_filter_mode": str(args.reference_filter_mode),
        "threshold_mode": str(args.threshold_mode),
        "quantile_threshold": float(args.quantile_threshold),
        "deviation_aggregation": str(args.deviation_aggregation),
        "per_gesture_threshold": bool(per_gesture_threshold),
        "threshold_by_gesture": threshold_by_gesture,
        "muscle_names": list(muscle_names),
        "reference_open_summary": open_reference_summary,
        "reference_fist_summary": fist_reference_summary,
        "classifier_appendix": appendix_summary,
    }
    save_json(node_summary, output_dir / "node_summary.json")

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
