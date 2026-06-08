from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert audited usable legacy open/fist trials into a formal transition dataset."
    )
    parser.add_argument("--audit-csv", required=True, help="legacy_transition_trial_audit.csv from the audit script.")
    parser.add_argument("--output-dir", required=True, help="Formal transition dataset output directory.")
    parser.add_argument("--sequence-length", type=int, default=100, help="Progress-normalized sequence length.")
    parser.add_argument("--include-unusable", action="store_true", help="Convert unusable trials too, marked unusable in metadata.")
    return parser.parse_args()


def _qpos_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if str(col).startswith("qpos_")]
    return sorted(cols, key=lambda col: int(str(col).rsplit("_", 1)[-1]) if str(col).rsplit("_", 1)[-1].isdigit() else str(col))


def _resample(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    source = np.linspace(0.0, 1.0, values.shape[0], dtype=float)
    target = np.linspace(0.0, 1.0, int(sequence_length), dtype=float)
    out = np.empty((int(sequence_length), values.shape[1]), dtype=float)
    for idx in range(values.shape[1]):
        out[:, idx] = np.interp(target, source, values[:, idx])
    return out, target * 100.0


def _phase(progress: float, motion_type: str) -> str:
    if progress <= 10.0:
        return "start_fist" if motion_type == "fist_to_open" else "start_open"
    if progress < 85.0:
        return "transition"
    if progress < 95.0:
        return "near_open" if motion_type == "fist_to_open" else "near_fist"
    return "hold_open" if motion_type == "fist_to_open" else "hold_fist"


def _safe_trial_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def main() -> int:
    args = parse_args()
    audit = pd.read_csv(args.audit_csv)
    if not bool(args.include_unusable):
        audit = audit[audit["usable_as_transition"] == 1].copy()
    output_dir = Path(args.output_dir).resolve()
    trials_dir = output_dir / "transition_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    mapping_rows = []
    for _, row in audit.iterrows():
        source = Path(str(row["source_csv"]))
        motion_type = str(row["inferred_motion_type"])
        if motion_type == "unusable" and not bool(args.include_unusable):
            continue
        df = pd.read_csv(source)
        qpos_cols = _qpos_columns(df)
        if not qpos_cols:
            continue
        qpos = df[qpos_cols].apply(pd.to_numeric, errors="coerce")
        qpos = qpos.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
        qpos_out, progress = _resample(qpos, int(args.sequence_length))
        trial_id = _safe_trial_id(str(row["trial_id"]))
        new_trial_id = f"{motion_type}__{trial_id}"
        out_csv = trials_dir / f"{new_trial_id}.csv"
        out_npz = trials_dir / f"{new_trial_id}.npz"
        out_json = trials_dir / f"{new_trial_id}.json"

        out_df = pd.DataFrame(qpos_out, columns=[f"qpos_{idx}" for idx in range(qpos_out.shape[1])])
        out_df.insert(0, "phase", [_phase(float(value), motion_type) for value in progress])
        out_df.insert(0, "progress_percent", progress)
        out_df.insert(0, "frame_index", np.arange(qpos_out.shape[0], dtype=int))
        out_df.insert(0, "motion_type", motion_type)
        out_df.insert(0, "trial_id", new_trial_id)
        out_df.to_csv(out_csv, index=False)

        np.savez_compressed(out_npz, qpos=qpos_out, progress_percent=progress, trial_id=np.asarray(new_trial_id), motion_type=np.asarray(motion_type))
        metadata = {
            "trial_id": new_trial_id,
            "legacy_trial_id": str(row["trial_id"]),
            "source_csv": str(source),
            "original_label": str(row["original_label"]),
            "motion_type": motion_type,
            "num_frames_raw": int(row["num_frames"]),
            "num_frames_output": int(qpos_out.shape[0]),
            "qpos_dim": int(qpos_out.shape[1]),
            "usable_as_transition": int(row["usable_as_transition"]),
            "audit_reason_if_not_usable": str(row.get("reason_if_not_usable", "")),
        }
        out_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest_rows.append({**metadata, "csv_path": str(out_csv), "npz_path": str(out_npz), "json_path": str(out_json)})
        mapping_rows.append(
            {
                "source_csv": str(source),
                "original_label": str(row["original_label"]),
                "legacy_trial_id": str(row["trial_id"]),
                "new_trial_id": new_trial_id,
                "motion_type": motion_type,
                "usable_as_transition": int(row["usable_as_transition"]),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    mapping = pd.DataFrame(mapping_rows)
    manifest_path = output_dir / "transition_dataset_manifest.csv"
    mapping_path = output_dir / "mapping_legacy_to_transition.csv"
    summary_path = output_dir / "transition_dataset_summary.json"
    manifest.to_csv(manifest_path, index=False)
    mapping.to_csv(mapping_path, index=False)
    summary = {
        "num_trials": int(len(manifest)),
        "num_open_to_fist": int((manifest["motion_type"] == "open_to_fist").sum()) if not manifest.empty else 0,
        "num_fist_to_open": int((manifest["motion_type"] == "fist_to_open").sum()) if not manifest.empty else 0,
        "sequence_length": int(args.sequence_length),
        "manifest_path": str(manifest_path),
        "mapping_path": str(mapping_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "[INFO] legacy transition conversion done | "
        f"trials={summary['num_trials']} open_to_fist={summary['num_open_to_fist']} fist_to_open={summary['num_fist_to_open']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
