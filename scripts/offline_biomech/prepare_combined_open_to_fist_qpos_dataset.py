from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _qpos_columns(columns: Iterable[str]) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for name in columns:
        label = str(name)
        if not label.startswith("qpos_"):
            continue
        try:
            index = int(label[5:])
        except ValueError:
            continue
        indexed.append((index, label))
    indexed.sort(key=lambda item: item[0])
    return [label for _, label in indexed]


def _first_nonempty(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return ""
    values = [str(value).strip() for value in df[column].dropna().tolist() if str(value).strip()]
    return values[0] if values else ""


def _safe_slug(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "trial"


def _validate_csv(
    csv_path: Path,
    *,
    motion_type: str,
    qpos_dim: int,
    min_frames: int,
    min_qpos_change: float,
) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    qcols = _qpos_columns(df.columns)
    trial_id = _first_nonempty(df, "trial_id") or csv_path.stem
    detected_motion = _first_nonempty(df, "motion_type") or _first_nonempty(df, "exercise_name")
    qpos_change = 0.0
    qpos_path_length = 0.0
    if qcols and len(df) > 1:
        qpos = df[qcols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        qpos_change = float(np.linalg.norm(qpos[-1] - qpos[0]))
        qpos_path_length = float(np.sum(np.linalg.norm(np.diff(qpos, axis=0), axis=1)))

    reasons: list[str] = []
    if len(df) < int(min_frames):
        reasons.append(f"num_frames<{int(min_frames)}")
    if len(qcols) != int(qpos_dim):
        reasons.append(f"qpos_dim!={int(qpos_dim)}")
    if str(detected_motion) != str(motion_type):
        reasons.append(f"motion_type!={motion_type}")
    if qpos_change < float(min_qpos_change):
        reasons.append(f"qpos_change<{float(min_qpos_change):.4f}")

    return {
        "source_csv": str(csv_path),
        "trial_id": str(trial_id),
        "motion_type": str(detected_motion),
        "num_frames": int(len(df)),
        "qpos_dim": int(len(qcols)),
        "qpos_change": float(qpos_change),
        "qpos_path_length": float(qpos_path_length),
        "used": int(not reasons),
        "reason": ";".join(reasons),
    }


def _collect_csvs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(csv_path for csv_path in path.rglob("*.csv") if csv_path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine previous and latest GUI open_to_fist qpos datasets into one formal ctrl_est input folder.")
    parser.add_argument(
        "--previous-input",
        default=str(PROJECT_ROOT / "artifacts" / "open_to_fist_primary_source" / "ctrl_ref_input"),
        help="Previous formal qpos input folder.",
    )
    parser.add_argument(
        "--new-input",
        default=str(PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous" / "input_gui_records"),
        help="New GUI-recorded qpos input folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "open_to_fist_combined_qpos_dataset"),
        help="Combined dataset output directory.",
    )
    parser.add_argument("--motion-type", default="open_to_fist")
    parser.add_argument("--qpos-dim", type=int, default=23)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-qpos-change", type=float, default=0.05)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    combined_dir = output_dir / "ctrl_ref_input"
    if bool(args.clean) and output_dir.exists():
        if not str(output_dir).startswith(str(PROJECT_ROOT.resolve())):
            raise RuntimeError(f"Refusing to delete outside project: {output_dir}")
        shutil.rmtree(output_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    source_groups = [
        ("previous", Path(args.previous_input).resolve()),
        ("new_gui", Path(args.new_input).resolve()),
    ]
    rows: list[dict[str, object]] = []
    for group_name, source_path in source_groups:
        for csv_path in _collect_csvs(source_path):
            row = _validate_csv(
                csv_path,
                motion_type=str(args.motion_type),
                qpos_dim=int(args.qpos_dim),
                min_frames=int(args.min_frames),
                min_qpos_change=float(args.min_qpos_change),
            )
            row["source_group"] = group_name
            if int(row["used"]):
                source_name = _safe_slug(Path(str(row["source_csv"])).stem)
                trial_id = _safe_slug(str(row["trial_id"]))
                target = combined_dir / f"ctrl_est_{group_name}__{trial_id}__{source_name}.csv"
                shutil.copy2(csv_path, target)
                row["combined_csv"] = str(target)
            else:
                row["combined_csv"] = ""
            rows.append(row)

    manifest_df = pd.DataFrame(rows)
    manifest_path = output_dir / "combined_qpos_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    used_df = manifest_df[manifest_df["used"].astype(int) == 1].copy()
    summary = {
        "previous_input": str(Path(args.previous_input).resolve()),
        "new_input": str(Path(args.new_input).resolve()),
        "combined_input": str(combined_dir),
        "motion_type": str(args.motion_type),
        "num_source_csv": int(len(manifest_df)),
        "num_used": int(len(used_df)),
        "num_rejected": int(len(manifest_df) - len(used_df)),
        "used_by_source_group": used_df.groupby("source_group").size().to_dict() if not used_df.empty else {},
        "frame_summary": {
            "min": int(used_df["num_frames"].min()) if not used_df.empty else 0,
            "median": float(used_df["num_frames"].median()) if not used_df.empty else 0.0,
            "max": int(used_df["num_frames"].max()) if not used_df.empty else 0,
        },
        "qpos_change_summary": {
            "min": float(used_df["qpos_change"].min()) if not used_df.empty else 0.0,
            "median": float(used_df["qpos_change"].median()) if not used_df.empty else 0.0,
            "max": float(used_df["qpos_change"].max()) if not used_df.empty else 0.0,
        },
    }
    summary_path = output_dir / "combined_qpos_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = PROJECT_ROOT / "reports" / "open_to_fist_combined_qpos_dataset.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Combined Open-To-Fist QPOS Dataset",
                "",
                f"- Previous input: `{summary['previous_input']}`",
                f"- New GUI input: `{summary['new_input']}`",
                f"- Combined input: `{summary['combined_input']}`",
                f"- Source CSVs scanned: {summary['num_source_csv']}",
                f"- Used trials: {summary['num_used']}",
                f"- Rejected trials: {summary['num_rejected']}",
                f"- Used by source group: `{summary['used_by_source_group']}`",
                f"- Frame summary: `{summary['frame_summary']}`",
                f"- QPOS change summary: `{summary['qpos_change_summary']}`",
                "",
                "Use `combined_input` as the input path for the next ctrl_ref build.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Combined input: {combined_dir}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Summary: {summary_path}")
    print(f"[INFO] Report: {report_path}")
    print(f"[INFO] Used: {summary['num_used']} / {summary['num_source_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
