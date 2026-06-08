"""Reference-relative node deviation utilities.

The outputs describe whether a node is within, lower than, or higher than a
simulation-based reference profile for a target gesture. They are not clinical
diagnostic labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from node_features import NODE_TEMPORAL_FEATURE_NAMES


VALID_NODE_STATUS = ("within_reference", "lower_than_reference", "higher_than_reference")
VALID_THRESHOLD_MODES = ("fixed", "quantile", "adaptive")
VALID_DEVIATION_AGGREGATIONS = ("mean_abs", "max_abs", "weighted")


def _default_feature_weight(feature_name: str) -> float:
    name = str(feature_name).strip().lower()
    if "mean_hold" in name:
        return 1.50
    if "mean_transition" in name:
        return 1.25
    if "peak" in name:
        return 1.20
    if "auc" in name:
        return 1.20
    return 1.00


def compute_node_deviation(
    sample_features: pd.DataFrame,
    reference_profile: dict[str, object],
    *,
    z_threshold: float = 2.0,
    threshold_mode: str = "fixed",
    quantile_threshold: float = 0.9,
    deviation_aggregation: str = "mean_abs",
    threshold_value: float | None = None,
) -> pd.DataFrame:
    ref_mean = reference_profile.get("ref_mean")
    ref_std = reference_profile.get("ref_std")
    if not isinstance(ref_mean, pd.DataFrame) or ref_mean.empty or not isinstance(ref_std, pd.DataFrame) or ref_std.empty:
        return pd.DataFrame()

    merged_df = sample_features.merge(ref_mean, on="node_name", suffixes=("", "_ref_mean"))
    merged_df = merged_df.merge(ref_std, on="node_name", suffixes=("", "_ref_std"))

    base_feature_cols = [col for col in NODE_TEMPORAL_FEATURE_NAMES if col in sample_features.columns]
    z_columns: list[str] = []
    for feature_name in base_feature_cols:
        mean_col = f"{feature_name}_ref_mean"
        std_col = f"{feature_name}_ref_std"
        if mean_col not in merged_df.columns or std_col not in merged_df.columns:
            continue
        z_col = f"z_{feature_name}"
        merged_df[z_col] = (
            pd.to_numeric(merged_df[feature_name], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged_df[mean_col], errors="coerce").fillna(0.0)
        ) / np.maximum(pd.to_numeric(merged_df[std_col], errors="coerce").fillna(1e-6), 1e-6)
        z_columns.append(z_col)

    if not z_columns:
        return pd.DataFrame()

    z_matrix = merged_df.loc[:, z_columns].to_numpy(dtype=float, copy=True)
    abs_matrix = np.abs(z_matrix)
    feature_weights = np.asarray([_default_feature_weight(name.replace("z_", "", 1)) for name in z_columns], dtype=float)
    feature_weights = np.where(feature_weights > 0.0, feature_weights, 1.0)

    aggregation_mode = str(deviation_aggregation or "mean_abs").strip().lower()
    if aggregation_mode not in VALID_DEVIATION_AGGREGATIONS:
        aggregation_mode = "mean_abs"

    if aggregation_mode == "max_abs":
        dominant_indices = np.argmax(abs_matrix, axis=1)
        aggregated_score = abs_matrix[np.arange(abs_matrix.shape[0]), dominant_indices]
        signed_score = z_matrix[np.arange(z_matrix.shape[0]), dominant_indices]
    elif aggregation_mode == "weighted":
        aggregated_score = np.sum(abs_matrix * feature_weights.reshape(1, -1), axis=1) / float(np.sum(feature_weights))
        signed_score = np.sum(z_matrix * feature_weights.reshape(1, -1), axis=1) / float(np.sum(feature_weights))
        dominant_indices = np.argmax(abs_matrix * feature_weights.reshape(1, -1), axis=1)
    else:
        aggregated_score = np.mean(abs_matrix, axis=1)
        signed_score = np.mean(z_matrix, axis=1)
        dominant_indices = np.argmax(abs_matrix, axis=1)

    dominant_z = z_matrix[np.arange(z_matrix.shape[0]), dominant_indices]
    dominant_feature_names = [z_columns[int(idx)].replace("z_", "", 1) for idx in dominant_indices.tolist()]
    merged_df["aggregated_deviation_score"] = aggregated_score.astype(float)
    merged_df["deviation_score"] = merged_df["aggregated_deviation_score"]
    merged_df["signed_deviation_score"] = signed_score.astype(float)
    merged_df["dominant_feature"] = dominant_feature_names
    merged_df["dominant_feature_zscore"] = dominant_z.astype(float)
    merged_df["reference_mean_feature"] = [
        float(merged_df.iloc[row_idx].get(f"{feature_name}_ref_mean", 0.0))
        for row_idx, feature_name in enumerate(dominant_feature_names)
    ]
    merged_df["sample_feature_value"] = [
        float(merged_df.iloc[row_idx].get(feature_name, 0.0))
        for row_idx, feature_name in enumerate(dominant_feature_names)
    ]

    mode_name = str(threshold_mode or "fixed").strip().lower()
    if mode_name not in VALID_THRESHOLD_MODES:
        mode_name = "fixed"
    if threshold_value is None:
        if mode_name == "quantile":
            threshold = float(np.quantile(np.abs(signed_score), np.clip(float(quantile_threshold), 0.5, 0.99)))
        elif mode_name == "adaptive":
            threshold = float(np.mean(np.abs(signed_score)) + np.std(np.abs(signed_score)))
        else:
            threshold = float(z_threshold)
    else:
        threshold = float(threshold_value)

    merged_df["threshold_used"] = float(threshold)
    merged_df["node_status"] = np.where(
        signed_score > threshold,
        "higher_than_reference",
        np.where(signed_score < -threshold, "lower_than_reference", "within_reference"),
    )
    merged_df["target_gesture_label"] = str(reference_profile.get("gesture_label") or "")
    return merged_df


def check_relational_consistency(node_statuses, visible_edges, node_names):
    status_map = dict(node_statuses or {})
    consistency_rows: list[dict[str, Any]] = []
    for edge in list(visible_edges or []):
        source_name = str(edge.get("source_name", ""))
        target_name = str(edge.get("target_name", ""))
        source_status = str(status_map.get(source_name, "within_reference"))
        target_status = str(status_map.get(target_name, "within_reference"))
        consistency_rows.append(
            {
                "source_name": source_name,
                "target_name": target_name,
                "edge_type": str(edge.get("edge_type", "")),
                "source_status": source_status,
                "target_status": target_status,
                "relational_consistent": bool(
                    source_status != "within_reference" and target_status != "within_reference"
                ),
            }
        )
    return consistency_rows
