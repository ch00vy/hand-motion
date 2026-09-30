from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_biomech.run_retrain_with_gui_records import _build_ctrl_ref, _run, _safe_rmtree  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "retrain_labeled_records"
DEFAULT_NORMAL_RECORDS = PROJECT_ROOT / "records" / "normal"
DEFAULT_ABNORMAL_RECORDS = PROJECT_ROOT / "records" / "abnormal"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier_labeled_records"


@dataclass(frozen=True)
class SplitCounts:
    total: int
    reference_pool: int
    train: int
    val: int
    holdout: int


def _qpos_columns(columns: Iterable[str]) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for name in columns:
        label = str(name)
        if not label.startswith("qpos_"):
            continue
        try:
            idx = int(label[5:])
        except ValueError:
            continue
        indexed.append((idx, label))
    indexed.sort(key=lambda item: item[0])
    return [label for _, label in indexed]


def _first_nonempty(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return ""
    values = [str(item).strip() for item in df[column].dropna().tolist() if str(item).strip()]
    return values[0] if values else ""


def _stage_labeled_records(
    *,
    source_dir: Path,
    output_dir: Path,
    label: str,
    motion_type: str,
    min_frames: int,
    qpos_dim: int,
    min_qpos_change: float,
    max_trials: int,
    clean: bool,
) -> tuple[Path, Path, pd.DataFrame]:
    if clean:
        _safe_rmtree(output_dir, root=output_dir.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    used_count = 0
    for qpos_csv in sorted(source_dir.rglob("qpos.csv")):
        try:
            df = pd.read_csv(qpos_csv)
            qcols = _qpos_columns(df.columns)
            trial_id = _first_nonempty(df, "trial_id") or qpos_csv.parent.name
            detected_motion = _first_nonempty(df, "motion_type") or _first_nonempty(df, "exercise_name")
            subject_id = _first_nonempty(df, "subject_id") or qpos_csv.parent.parent.name
            qpos = (
                df[qcols]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float)
                if qcols
                else np.empty((0, 0), dtype=float)
            )
            qpos_change = float(np.linalg.norm(qpos[-1] - qpos[0])) if len(qpos) > 1 else 0.0
            qpos_path_length = float(np.sum(np.linalg.norm(np.diff(qpos, axis=0), axis=1))) if len(qpos) > 1 else 0.0
            valid = (
                len(df) >= int(min_frames)
                and len(qcols) == int(qpos_dim)
                and str(detected_motion) == str(motion_type)
                and qpos_change >= float(min_qpos_change)
                and (int(max_trials) <= 0 or used_count < int(max_trials))
            )
            staged_csv = ""
            if valid:
                staged = output_dir / f"ctrl_est_{label}_{trial_id}.csv"
                staged_df = df.copy()
                staged_df["trial_label"] = str(label)
                staged_df["label"] = str(label)
                staged_df["motion_type"] = str(motion_type)
                staged_df["exercise_name"] = str(motion_type)
                staged_df.to_csv(staged, index=False)
                staged_csv = str(staged)
                used_count += 1
            manifest_rows.append(
                {
                    "source_qpos_csv": str(qpos_csv),
                    "staged_csv": staged_csv,
                    "trial_id": trial_id,
                    "subject_id": subject_id,
                    "motion_type": detected_motion,
                    "label": label,
                    "num_frames": int(len(df)),
                    "qpos_dim": int(len(qcols)),
                    "qpos_change": qpos_change,
                    "qpos_path_length": qpos_path_length,
                    "used": int(valid),
                    "reason": "" if valid else "failed_filter_or_max_trials",
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "source_qpos_csv": str(qpos_csv),
                    "staged_csv": "",
                    "trial_id": qpos_csv.parent.name,
                    "subject_id": qpos_csv.parent.parent.name,
                    "motion_type": "",
                    "label": label,
                    "num_frames": 0,
                    "qpos_dim": 0,
                    "qpos_change": 0.0,
                    "qpos_path_length": 0.0,
                    "used": 0,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    manifest = output_dir.parent / f"{label}_record_input_manifest.csv"
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest, index=False)
    return output_dir, manifest, manifest_df


def prepare_records(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    normal_dir = output_root / "input_normal_records"
    abnormal_dir = output_root / "input_abnormal_records"
    normal_df: pd.DataFrame
    abnormal_df: pd.DataFrame
    _, normal_manifest, normal_df = _stage_labeled_records(
        source_dir=Path(args.normal_records).resolve(),
        output_dir=normal_dir,
        label="normal",
        motion_type=str(args.motion_type),
        min_frames=int(args.min_frames),
        qpos_dim=int(args.qpos_dim),
        min_qpos_change=float(args.min_qpos_change),
        max_trials=int(args.max_normal_trials),
        clean=bool(args.clean_prepare),
    )
    _, abnormal_manifest, abnormal_df = _stage_labeled_records(
        source_dir=Path(args.abnormal_records).resolve(),
        output_dir=abnormal_dir,
        label="abnormal",
        motion_type=str(args.motion_type),
        min_frames=int(args.min_frames),
        qpos_dim=int(args.qpos_dim),
        min_qpos_change=float(args.min_qpos_change),
        max_trials=int(args.max_abnormal_trials),
        clean=bool(args.clean_prepare),
    )
    summary = {
        "normal_records": str(Path(args.normal_records).resolve()),
        "abnormal_records": str(Path(args.abnormal_records).resolve()),
        "motion_type": str(args.motion_type),
        "normal_total": int(len(normal_df)),
        "normal_used": int(normal_df["used"].sum()) if "used" in normal_df.columns else 0,
        "abnormal_total": int(len(abnormal_df)),
        "abnormal_used": int(abnormal_df["used"].sum()) if "used" in abnormal_df.columns else 0,
        "normal_manifest": str(normal_manifest),
        "abnormal_manifest": str(abnormal_manifest),
        "normal_input_dir": str(normal_dir),
        "abnormal_input_dir": str(abnormal_dir),
    }
    (output_root / "labeled_records_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_prepare_report(args, summary, normal_df, abnormal_df)
    print(f"[INFO] Normal records staged: {summary['normal_used']}/{summary['normal_total']}")
    print(f"[INFO] Abnormal records staged: {summary['abnormal_used']}/{summary['abnormal_total']}")
    print(f"[INFO] Summary: {output_root / 'labeled_records_dataset_summary.json'}")


def _write_prepare_report(args: argparse.Namespace, summary: dict[str, object], normal_df: pd.DataFrame, abnormal_df: pd.DataFrame) -> None:
    report = PROJECT_ROOT / "reports" / "labeled_records_retrain_dataset.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    def stats(df: pd.DataFrame) -> str:
        used = df.loc[df["used"].astype(int) == 1].copy() if "used" in df.columns else df.iloc[0:0]
        if used.empty:
            return "No valid staged trials."
        return (
            f"{len(used)} valid trials; frames min/mean/max = "
            f"{int(used['num_frames'].min())}/"
            f"{float(used['num_frames'].mean()):.1f}/"
            f"{int(used['num_frames'].max())}; "
            f"qpos_change mean = {float(used['qpos_change'].mean()):.4f}"
        )

    report.write_text(
        "\n".join(
            [
                "# Labeled Records Retrain Dataset",
                "",
                "This dataset replaces the previous legacy-training source for the next retrain run.",
                "",
                "## Source",
                f"- Normal records: `{summary['normal_records']}`",
                f"- Abnormal records: `{summary['abnormal_records']}`",
                f"- Motion type: `{summary['motion_type']}`",
                f"- Minimum frames: `{int(args.min_frames)}`",
                f"- Minimum qpos change: `{float(args.min_qpos_change):.4f}`",
                "",
                "## Staged Trials",
                f"- Normal: {summary['normal_used']}/{summary['normal_total']} valid",
                f"- Abnormal: {summary['abnormal_used']}/{summary['abnormal_total']} valid",
                f"- Normal stats: {stats(normal_df)}",
                f"- Abnormal stats: {stats(abnormal_df)}",
                "",
                "## Manifests",
                f"- Normal manifest: `{summary['normal_manifest']}`",
                f"- Abnormal manifest: `{summary['abnormal_manifest']}`",
                "",
                "## Training Policy",
                "- Normal trials are used to build the biomechanical reference sequence and train/validate/test the normal-reference deviation model.",
                "- Abnormal trials are kept as abnormal validation/test data, not as normal reference trials.",
                "- The old `records/subject_gui` legacy folder is not used by this workflow.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_ctrl_refs(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    _build_ctrl_ref(
        args,
        input_dir=output_root / "input_normal_records",
        output_dir=output_root / "ctrl_ref_normal" / "reference_build",
        report_path=PROJECT_ROOT / "reports" / "labeled_records_normal_biomech_reference.md",
        clean=bool(args.clean_ctrl_ref),
        max_trials=0,
    )
    _build_ctrl_ref(
        args,
        input_dir=output_root / "input_abnormal_records",
        output_dir=output_root / "ctrl_ref_abnormal" / "reference_build",
        report_path=PROJECT_ROOT / "reports" / "labeled_records_abnormal_biomech_reference.md",
        clean=bool(args.clean_ctrl_ref),
        max_trials=0,
    )


def build_graph_datasets(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    normal_dir = output_root / "ctrl_ref_normal" / "reference_build" / "accepted_trials"
    abnormal_dir = output_root / "ctrl_ref_abnormal" / "reference_build" / "accepted_trials"
    graph_dir = output_root / "graph_sequence_dataset"
    abnormal_graph_dir = output_root / "abnormal_graph_sequence_dataset"
    if args.clean_graph:
        _safe_rmtree(graph_dir, root=output_root)
        _safe_rmtree(abnormal_graph_dir, root=output_root)
    base_cmd = [
        sys.executable,
        "scripts/offline_biomech/build_graph_sequence_dataset.py",
        "--sequence-length",
        str(int(args.sequence_length)),
        "--window-size",
        str(int(args.window_size)),
        "--window-stride",
        str(int(args.window_stride)),
        "--progress-mode",
        "linear",
        "--edge-mode",
        "dynamic_short_window",
        "--edge-threshold",
        str(float(args.edge_threshold)),
        "--include-edge-sign",
        "--transition-segmentation",
        "auto_motion",
        "--motion-threshold",
        str(float(args.motion_threshold)),
        "--transition-pad-frames",
        str(int(args.transition_pad_frames)),
    ]
    _run([*base_cmd, "--normal-input", str(normal_dir), "--output-dir", str(graph_dir)])
    _run([*base_cmd, "--normal-input", str(normal_dir), "--abnormal-input", str(abnormal_dir), "--output-dir", str(abnormal_graph_dir)])


def build_reference(args: argparse.Namespace) -> SplitCounts:
    output_root = Path(args.output_root).resolve()
    dataset_dir = output_root / "graph_sequence_dataset"
    reference_root = output_root / "reference_sequence"
    reference_pool_dataset = reference_root / "reference_pool_dataset"
    if args.clean_reference:
        _safe_rmtree(reference_root, root=output_root)
    reference_root.mkdir(parents=True, exist_ok=True)
    reference_pool_dataset.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(dataset_dir / "dataset_manifest.csv").sort_values("trial_id").reset_index(drop=True)
    total = int(len(manifest))
    if total < 5:
        raise RuntimeError(f"Need at least 5 normal graph trials for split, found {total}")
    holdout_count = max(1, int(round(total * float(args.holdout_fraction))))
    reference_pool_count = total - holdout_count
    val_count = max(1, int(round(reference_pool_count * float(args.val_fraction))))
    train_count = max(1, reference_pool_count - val_count)

    holdout_indices = set(np.linspace(0, total - 1, holdout_count, dtype=int).tolist())
    holdout = manifest.iloc[[idx for idx in range(total) if idx in holdout_indices]].copy()
    reference_pool = manifest.iloc[[idx for idx in range(total) if idx not in holdout_indices]].copy()
    reference_pool.to_csv(reference_root / "reference_pool_manifest.csv", index=False)
    holdout.to_csv(reference_root / "holdout_manifest.csv", index=False)
    reference_pool.to_csv(reference_pool_dataset / "dataset_manifest.csv", index=False)
    if (dataset_dir / "dataset_config.json").exists():
        shutil.copy2(dataset_dir / "dataset_config.json", reference_pool_dataset / "dataset_config.json")

    _run(
        [
            sys.executable,
            "scripts/offline_biomech/build_reference_sequence.py",
            "--dataset-dir",
            str(reference_pool_dataset),
            "--output-dir",
            str(reference_root / "reference"),
            "--cohort",
            "normal",
            "--onset-ratio",
            str(float(args.onset_ratio)),
        ]
    )
    counts = SplitCounts(total=total, reference_pool=reference_pool_count, train=train_count, val=val_count, holdout=holdout_count)
    (reference_root / "split_counts.json").write_text(json.dumps(counts.__dict__, indent=2), encoding="utf-8")
    print(f"[INFO] Split counts: {counts}")
    return counts


def train_dynamic_temporal_gcn(args: argparse.Namespace, split_counts: SplitCounts | None = None) -> None:
    output_root = Path(args.output_root).resolve()
    if split_counts is None:
        split_counts = SplitCounts(**json.loads((output_root / "reference_sequence" / "split_counts.json").read_text(encoding="utf-8")))
    _run(
        [
            sys.executable,
            "scripts/offline_biomech/train_dynamic_temporal_gcn.py",
            "--dataset-dir",
            str(output_root / "graph_sequence_dataset"),
            "--reference-dir",
            str(output_root / "reference_sequence" / "reference"),
            "--reference-root-dir",
            str(output_root / "reference_sequence"),
            "--abnormal-dataset-dir",
            str(output_root / "abnormal_graph_sequence_dataset"),
            "--output-dir",
            str(Path(args.model_output_dir).resolve()),
            "--reports-dir",
            "reports",
            "--train-reference-trials",
            str(int(split_counts.train)),
            "--binary-threshold-quantile",
            str(float(args.binary_threshold_quantile)),
            "--hidden-dim",
            str(int(args.hidden_dim)),
            "--temporal-dim",
            str(int(args.temporal_dim)),
            "--dropout",
            str(float(args.dropout)),
            "--epochs",
            str(int(args.epochs)),
            "--patience",
            str(int(args.patience)),
            "--lr",
            str(float(args.lr)),
            "--weight-decay",
            str(float(args.weight_decay)),
            "--seed",
            str(int(args.seed)),
            "--force-clean",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain Dynamic Temporal GCN from the new labeled records/normal and records/abnormal dataset only.")
    parser.add_argument("--stage", choices=("prepare", "ctrl_ref", "graph", "reference", "train", "all"), required=True)
    parser.add_argument("--normal-records", default=str(DEFAULT_NORMAL_RECORDS))
    parser.add_argument("--abnormal-records", default=str(DEFAULT_ABNORMAL_RECORDS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-output-dir", default=str(DEFAULT_MODEL_OUTPUT))
    parser.add_argument("--motion-type", default="open_to_fist")
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--qpos-dim", type=int, default=23)
    parser.add_argument("--min-qpos-change", type=float, default=0.05)
    parser.add_argument("--max-normal-trials", type=int, default=0)
    parser.add_argument("--max-abnormal-trials", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--motion-threshold", type=float, default=0.12)
    parser.add_argument("--transition-pad-frames", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--onset-ratio", type=float, default=0.20)
    parser.add_argument("--binary-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--temporal-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-prepare", action="store_true")
    parser.add_argument("--clean-ctrl-ref", action="store_true")
    parser.add_argument("--clean-graph", action="store_true")
    parser.add_argument("--clean-reference", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_counts: SplitCounts | None = None
    if args.stage in {"prepare", "all"}:
        prepare_records(args)
    if args.stage in {"ctrl_ref", "all"}:
        build_ctrl_refs(args)
    if args.stage in {"graph", "all"}:
        build_graph_datasets(args)
    if args.stage in {"reference", "all"}:
        split_counts = build_reference(args)
    if args.stage in {"train", "all"}:
        train_dynamic_temporal_gcn(args, split_counts=split_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
