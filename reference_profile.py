"""Reference profile builders for gesture-conditioned node analysis.

Reference profiles summarize expected node behavior for a target gesture such
as open or fist. They are simulation-based templates, not clinical norms.
"""

from __future__ import annotations

import pandas as pd

from node_features import NODE_TEMPORAL_FEATURE_NAMES


REFERENCE_STD_EPS = 1e-3


def build_reference_profile(trials: pd.DataFrame, gesture_label: str) -> dict[str, object]:
    gesture_name = str(gesture_label).strip().lower()
    subset_df = trials.loc[trials["gesture_label"].astype(str).str.lower() == gesture_name].copy()
    if subset_df.empty:
        return {
            "gesture_label": gesture_name,
            "num_trials": 0,
            "ref_mean": pd.DataFrame(),
            "ref_std": pd.DataFrame(),
            "reference_table": pd.DataFrame(),
            "quality_stats": {},
        }

    feature_columns = [col for col in NODE_TEMPORAL_FEATURE_NAMES if col in subset_df.columns]
    grouped = subset_df.groupby("node_name", sort=False)
    ref_mean = grouped[feature_columns].mean().reset_index()
    ref_std = grouped[feature_columns].std(ddof=0).fillna(0.0).clip(lower=REFERENCE_STD_EPS).reset_index()
    reference_table = ref_mean.merge(ref_std, on="node_name", suffixes=("_mean", "_std"))
    quality_rows: list[dict[str, object]] = []
    for _, row in reference_table.iterrows():
        node_name = str(row["node_name"])
        for feature_name in feature_columns:
            mean_value = float(row.get(f"{feature_name}_mean", 0.0))
            std_value = float(row.get(f"{feature_name}_std", 0.0))
            denom = max(abs(mean_value), REFERENCE_STD_EPS)
            quality_rows.append(
                {
                    "gesture_label": gesture_name,
                    "node_name": node_name,
                    "feature_name": feature_name,
                    "ref_mean": mean_value,
                    "ref_std": std_value,
                    "cv": float(std_value / denom),
                    "num_trials": int(subset_df["trial_id"].nunique()) if "trial_id" in subset_df.columns else int(len(subset_df)),
                }
            )
    quality_table = pd.DataFrame(
        quality_rows,
        columns=["gesture_label", "node_name", "feature_name", "ref_mean", "ref_std", "cv", "num_trials"],
    )

    quality_stats = {}
    quality_cols = [col for col in subset_df.columns if col.startswith("hold_") or col in {"quality_score"}]
    if quality_cols:
        quality_stats = {
            str(col): float(pd.to_numeric(subset_df[col], errors="coerce").mean())
            for col in quality_cols
            if pd.to_numeric(subset_df[col], errors="coerce").notna().any()
        }

    return {
        "gesture_label": gesture_name,
        "num_trials": int(subset_df["trial_id"].nunique()) if "trial_id" in subset_df.columns else int(len(subset_df)),
        "ref_mean": ref_mean,
        "ref_std": ref_std,
        "reference_table": reference_table,
        "quality_table": quality_table,
        "quality_stats": quality_stats,
    }
