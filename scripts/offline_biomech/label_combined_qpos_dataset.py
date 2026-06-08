from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign cohort labels to a combined open_to_fist qpos dataset manifest.")
    parser.add_argument(
        "--dataset-dir",
        default=str(PROJECT_ROOT / "artifacts" / "open_to_fist_combined_qpos_dataset"),
        help="Dataset directory containing combined_qpos_manifest.csv.",
    )
    parser.add_argument("--cohort", default="normal", choices=("normal", "abnormal"))
    parser.add_argument("--label-source", default="user_declared_all_current_trials_normal")
    parser.add_argument("--motion-type", default="open_to_fist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    manifest_path = dataset_dir / "combined_qpos_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "used" not in df.columns:
        raise ValueError(f"Manifest missing used column: {manifest_path}")
    df["cohort"] = str(args.cohort)
    df["clinical_label"] = str(args.cohort)
    df["motion_type_final"] = str(args.motion_type)
    df["label_source"] = str(args.label_source)
    df["is_reference_candidate"] = (df["used"].astype(int) == 1) & (df["cohort"].astype(str) == "normal")
    df["is_abnormal_candidate"] = (df["used"].astype(int) == 1) & (df["cohort"].astype(str) == "abnormal")

    labeled_manifest = dataset_dir / "combined_qpos_labeled_manifest.csv"
    df.to_csv(labeled_manifest, index=False)

    used_df = df[df["used"].astype(int) == 1].copy()
    normal_df = used_df[used_df["cohort"].astype(str) == "normal"].copy()
    abnormal_df = used_df[used_df["cohort"].astype(str) == "abnormal"].copy()
    reference_pool = normal_df.copy()
    reference_pool_path = dataset_dir / "normal_reference_candidate_manifest.csv"
    reference_pool.to_csv(reference_pool_path, index=False)

    abnormal_path = dataset_dir / "abnormal_candidate_manifest.csv"
    abnormal_df.to_csv(abnormal_path, index=False)

    summary = {
        "dataset_dir": str(dataset_dir),
        "labeled_manifest": str(labeled_manifest),
        "normal_reference_candidate_manifest": str(reference_pool_path),
        "abnormal_candidate_manifest": str(abnormal_path),
        "motion_type": str(args.motion_type),
        "label_source": str(args.label_source),
        "num_total": int(len(df)),
        "num_used": int(len(used_df)),
        "num_normal": int(len(normal_df)),
        "num_abnormal": int(len(abnormal_df)),
        "num_reference_candidates": int(reference_pool["is_reference_candidate"].sum()) if not reference_pool.empty else 0,
    }
    summary_path = dataset_dir / "label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = PROJECT_ROOT / "reports" / "open_to_fist_combined_labeling.md"
    report_path.write_text(
        "\n".join(
            [
                "# Open-To-Fist Combined Dataset Labeling",
                "",
                "Current user decision: all currently combined trials are normal.",
                "",
                f"- Dataset: `{dataset_dir}`",
                f"- Labeled manifest: `{labeled_manifest}`",
                f"- Normal/reference candidate manifest: `{reference_pool_path}`",
                f"- Abnormal candidate manifest: `{abnormal_path}`",
                f"- Total rows: {summary['num_total']}",
                f"- Used trials: {summary['num_used']}",
                f"- Normal trials: {summary['num_normal']}",
                f"- Abnormal trials: {summary['num_abnormal']}",
                "",
                "Next abnormal-data step: record abnormal trials separately and stage them into a separate abnormal manifest. Do not add them to the normal reference pool.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Labeled manifest: {labeled_manifest}")
    print(f"[INFO] Normal/reference candidates: {summary['num_reference_candidates']}")
    print(f"[INFO] Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
