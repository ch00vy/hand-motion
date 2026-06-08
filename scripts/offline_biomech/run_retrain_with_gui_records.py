from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous"
DEFAULT_PREVIOUS_ACCEPTED = PROJECT_ROOT / "artifacts" / "open_to_fist_biomech_reference_latest_short_horizon" / "reference_build" / "accepted_trials"
DEFAULT_PREVIOUS_QPOS_INPUT = PROJECT_ROOT / "artifacts" / "open_to_fist_primary_source" / "ctrl_ref_input"
DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "records"
DEFAULT_ABNORMAL_DATE_PREFIX = "20260604_,20260606_"


@dataclass(frozen=True)
class SplitCounts:
    total: int
    reference_pool: int
    train: int
    val: int
    holdout: int


def _run(cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _safe_rmtree(path: Path, *, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise RuntimeError(f"Refusing to delete outside output root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


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


def _split_prefixes(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def prepare_gui_records(args: argparse.Namespace) -> Path:
    records_root = Path(args.records_root).resolve()
    output_root = Path(args.output_root).resolve()
    input_dir = output_root / "input_gui_records"
    if args.clean_prepare:
        _safe_rmtree(input_dir, root=output_root)
    input_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    used = 0
    for qpos_csv in sorted(records_root.rglob("qpos.csv")):
        rel = str(qpos_csv)
        if args.date_prefix and str(args.date_prefix) not in rel:
            continue
        try:
            df = pd.read_csv(qpos_csv)
            qcols = _qpos_columns(df.columns)
            motion_type = _first_nonempty(df, "motion_type") or _first_nonempty(df, "exercise_name")
            trial_id = _first_nonempty(df, "trial_id") or qpos_csv.parent.name
            qpos = df[qcols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            qpos_change = float(np.linalg.norm(qpos[-1] - qpos[0])) if len(qpos) > 1 else 0.0
            path_length = float(np.sum(np.linalg.norm(np.diff(qpos, axis=0), axis=1))) if len(qpos) > 1 else 0.0
            valid = (
                len(df) >= int(args.min_frames)
                and len(qcols) == int(args.qpos_dim)
                and str(motion_type) == str(args.motion_type)
                and qpos_change >= float(args.min_qpos_change)
            )
            staged_path = ""
            if valid:
                staged = input_dir / f"ctrl_est_gui_{trial_id}.csv"
                shutil.copy2(qpos_csv, staged)
                staged_path = str(staged)
                used += 1
            manifest_rows.append(
                {
                    "source_qpos_csv": str(qpos_csv),
                    "staged_csv": staged_path,
                    "trial_id": trial_id,
                    "motion_type": motion_type,
                    "num_frames": int(len(df)),
                    "qpos_dim": int(len(qcols)),
                    "qpos_change": qpos_change,
                    "qpos_path_length": path_length,
                    "used": int(valid),
                    "reason": "" if valid else "failed_filter",
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "source_qpos_csv": str(qpos_csv),
                    "staged_csv": "",
                    "trial_id": qpos_csv.parent.name,
                    "motion_type": "",
                    "num_frames": 0,
                    "qpos_dim": 0,
                    "qpos_change": 0.0,
                    "qpos_path_length": 0.0,
                    "used": 0,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    manifest = output_root / "gui_record_input_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    print(f"[INFO] GUI records staged: {used}/{len(manifest_rows)}")
    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Manifest: {manifest}")
    return input_dir


def prepare_abnormal_records(args: argparse.Namespace) -> Path:
    records_root = Path(args.records_root).resolve()
    output_root = Path(args.output_root).resolve()
    input_dir = output_root / "input_abnormal_records"
    if args.clean_prepare:
        _safe_rmtree(input_dir, root=output_root)
    input_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    used = 0
    for qpos_csv in sorted(records_root.rglob("qpos.csv")):
        rel = str(qpos_csv)
        prefixes = _split_prefixes(str(args.abnormal_date_prefix))
        if prefixes and not any(prefix in rel for prefix in prefixes):
            continue
        try:
            df = pd.read_csv(qpos_csv)
            qcols = _qpos_columns(df.columns)
            motion_type = _first_nonempty(df, "motion_type") or _first_nonempty(df, "exercise_name")
            trial_id = _first_nonempty(df, "trial_id") or qpos_csv.parent.name
            qpos = df[qcols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            qpos_change = float(np.linalg.norm(qpos[-1] - qpos[0])) if len(qpos) > 1 else 0.0
            path_length = float(np.sum(np.linalg.norm(np.diff(qpos, axis=0), axis=1))) if len(qpos) > 1 else 0.0
            valid = (
                len(df) >= int(args.min_frames)
                and len(qcols) == int(args.qpos_dim)
                and str(motion_type) == str(args.motion_type)
                and qpos_change >= float(args.min_qpos_change)
            )
            staged_path = ""
            if valid:
                staged = input_dir / f"ctrl_est_abnormal_{trial_id}.csv"
                shutil.copy2(qpos_csv, staged)
                staged_path = str(staged)
                used += 1
            manifest_rows.append(
                {
                    "source_qpos_csv": str(qpos_csv),
                    "staged_csv": staged_path,
                    "trial_id": trial_id,
                    "motion_type": motion_type,
                    "clinical_label": "abnormal",
                    "num_frames": int(len(df)),
                    "qpos_dim": int(len(qcols)),
                    "qpos_change": qpos_change,
                    "qpos_path_length": path_length,
                    "used": int(valid),
                    "reason": "" if valid else "failed_filter",
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "source_qpos_csv": str(qpos_csv),
                    "staged_csv": "",
                    "trial_id": qpos_csv.parent.name,
                    "motion_type": "",
                    "clinical_label": "abnormal",
                    "num_frames": 0,
                    "qpos_dim": 0,
                    "qpos_change": 0.0,
                    "qpos_path_length": 0.0,
                    "used": 0,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    manifest = output_root / "abnormal_record_input_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    print(f"[INFO] Abnormal records staged: {used}/{len(manifest_rows)}")
    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Manifest: {manifest}")
    return input_dir


def build_gui_ctrl_ref(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    input_dir = output_root / "input_gui_records"
    output_dir = output_root / "ctrl_ref_gui_records" / "reference_build"
    report_path = PROJECT_ROOT / "reports" / "gui_records_biomech_reference_latest_short_horizon.md"
    _build_ctrl_ref(
        args,
        input_dir=input_dir,
        output_dir=output_dir,
        report_path=report_path,
        clean=bool(args.clean_ctrl_ref),
        max_trials=int(args.max_gui_trials),
    )


def build_abnormal_ctrl_ref(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    input_dir = output_root / "input_abnormal_records"
    output_dir = output_root / "ctrl_ref_abnormal" / "reference_build"
    report_path = PROJECT_ROOT / "reports" / "abnormal_validation_biomech_reference_latest_short_horizon.md"
    _build_ctrl_ref(
        args,
        input_dir=input_dir,
        output_dir=output_dir,
        report_path=report_path,
        clean=bool(args.clean_ctrl_ref),
        max_trials=int(args.max_abnormal_trials),
    )


def _build_ctrl_ref(
    args: argparse.Namespace,
    *,
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
    clean: bool,
    max_trials: int = 0,
) -> None:
    output_root = Path(args.output_root).resolve()
    if clean:
        _safe_rmtree(output_dir, root=output_root)
    cmd = [
        sys.executable,
        "scripts/offline_biomech/build_biomech_ctrl_reference.py",
        "--input-path",
        str(input_dir),
        "--output-path",
        str(output_dir),
        "--report-path",
        str(report_path),
        "--reference-mode",
        "short_horizon_rollout",
        "--reference-prior-mode",
        "static_qpos",
        "--ignore-recorded-timestamps",
        "--kinematic-filter",
        "savgol",
        "--qpos-smoothing-window",
        "9",
        "--qvel-smoothing-window",
        "7",
        "--qacc-smoothing-window",
        "7",
        "--qvel-clip",
        "25",
        "--qacc-clip",
        "250",
        "--tracking-tau-kp",
        "2.0",
        "--tracking-tau-kv",
        "0.08",
        "--mismatch-feedback-iters",
        "2",
        "--mismatch-feedback-kp",
        "0.25",
        "--mismatch-feedback-kv",
        "0.005",
        "--mismatch-feedback-tau-clip",
        "0.25",
        "--mismatch-feedback-qpos-weight",
        "1.0",
        "--mismatch-feedback-qvel-weight",
        "0.01",
        "--mismatch-feedback-final-weight",
        "4.0",
        "--mismatch-feedback-control-weight",
        "0.001",
        "--mismatch-feedback-smoothness-weight",
        "0.02",
        "--mismatch-feedback-alphas",
        "0.25,0.5,0.75",
        "--finger-flexion-feedback-weight",
        "1.25",
        "--distal-flexion-feedback-weight",
        "1.8",
        "--adaptive-finger-feedback",
        "--adaptive-feedback-min-weight",
        "0.75",
        "--adaptive-feedback-max-weight",
        "2.0",
        "--activation-preemphasis-gain",
        "0.35",
        "--activation-preemphasis-alphas",
        "0.25,0.5,1.0",
        "--horizon-size",
        "4",
        "--horizon-iters",
        "2",
        "--commit-frames",
        "1",
        "--rollout-qpos-weight",
        "1.0",
        "--rollout-qvel-weight",
        "0.05",
        "--rollout-final-qpos-weight",
        "4.0",
        "--rollout-u-weight",
        "0.005",
        "--rollout-delta-u-weight",
        "0.05",
        "--rollout-prior-weight",
        "0.05",
        "--prev-ctrl-weight",
        "0.02",
        "--static-prior-weight",
        "0.05",
        "--qp-reg",
        "0.0005",
        "--validation-window-size",
        "0",
        "--max-mean-joint-error",
        "10",
        "--max-rmse-forward-error",
        "10",
        "--max-upper-saturation-fraction",
        "1.0",
        "--max-ctrl-delta-mean",
        "10",
    ]
    if int(max_trials) > 0:
        cmd.extend(["--max-trials", str(int(max_trials))])
    _run(cmd)


def prepare_combined_qpos(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    prepare_gui_records(args)
    combined_dir = output_root / "combined_qpos_dataset"
    cmd = [
        sys.executable,
        "scripts/offline_biomech/prepare_combined_open_to_fist_qpos_dataset.py",
        "--previous-input",
        str(Path(args.previous_qpos_input).resolve()),
        "--new-input",
        str(output_root / "input_gui_records"),
        "--output-dir",
        str(combined_dir),
        "--motion-type",
        str(args.motion_type),
        "--qpos-dim",
        str(int(args.qpos_dim)),
        "--min-frames",
        str(int(args.min_frames)),
        "--min-qpos-change",
        str(float(args.min_qpos_change)),
    ]
    if bool(args.clean_prepare):
        cmd.append("--clean")
    _run(cmd)
    return combined_dir / "ctrl_ref_input"


def build_combined_ctrl_ref(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    input_dir = output_root / "combined_qpos_dataset" / "ctrl_ref_input"
    output_dir = output_root / "ctrl_ref_combined" / "reference_build"
    report_path = PROJECT_ROOT / "reports" / "combined_normal_biomech_reference_latest_short_horizon.md"
    _build_ctrl_ref(
        args,
        input_dir=input_dir,
        output_dir=output_dir,
        report_path=report_path,
        clean=bool(args.clean_ctrl_ref),
        max_trials=int(args.max_combined_trials),
    )


def merge_ctrl_ref(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    merged_dir = output_root / "merged_ctrl_ref_accepted_trials"
    if args.clean_merge:
        _safe_rmtree(merged_dir, root=output_root)
    merged_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("previous", Path(args.previous_accepted_dir).resolve()),
        ("gui", output_root / "ctrl_ref_gui_records" / "reference_build" / "accepted_trials"),
    ]
    rows: list[dict[str, object]] = []
    for source_name, source_dir in sources:
        if not source_dir.exists():
            print(f"[WARN] Missing ctrl_ref source dir: {source_dir}")
            continue
        for csv_path in sorted(source_dir.glob("*.csv")):
            target = merged_dir / f"{source_name}__{csv_path.name}"
            shutil.copy2(csv_path, target)
            rows.append({"source_group": source_name, "source_csv": str(csv_path), "merged_csv": str(target)})
    manifest = output_root / "merged_ctrl_ref_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"[INFO] Merged ctrl_ref trials: {len(rows)} -> {merged_dir}")
    return merged_dir


def build_graph_dataset(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    if str(args.graph_source).strip().lower() == "combined":
        merged_dir = output_root / "ctrl_ref_combined" / "reference_build" / "accepted_trials"
    else:
        merged_dir = output_root / "merged_ctrl_ref_accepted_trials"
    dataset_dir = output_root / "graph_sequence_dataset"
    if args.clean_graph:
        _safe_rmtree(dataset_dir, root=output_root)
    cmd = [
        sys.executable,
        "scripts/offline_biomech/build_graph_sequence_dataset.py",
        "--normal-input",
        str(merged_dir),
        "--output-dir",
        str(dataset_dir),
        "--sequence-length",
        "60",
        "--window-size",
        "5",
        "--window-stride",
        "1",
        "--progress-mode",
        "linear",
        "--edge-mode",
        "dynamic_short_window",
        "--edge-threshold",
        "0.02",
        "--include-edge-sign",
        "--transition-segmentation",
        "auto_motion",
        "--motion-threshold",
        "0.12",
        "--transition-pad-frames",
        "2",
    ]
    _run(cmd)
    return dataset_dir


def build_abnormal_graph_dataset(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    normal_dir = output_root / "ctrl_ref_combined" / "reference_build" / "accepted_trials"
    if not normal_dir.exists():
        normal_dir = Path(args.previous_accepted_dir).resolve()
    abnormal_dir = output_root / "ctrl_ref_abnormal" / "reference_build" / "accepted_trials"
    dataset_dir = output_root / "abnormal_graph_sequence_dataset"
    if args.clean_graph:
        _safe_rmtree(dataset_dir, root=output_root)
    cmd = [
        sys.executable,
        "scripts/offline_biomech/build_graph_sequence_dataset.py",
        "--normal-input",
        str(normal_dir),
        "--abnormal-input",
        str(abnormal_dir),
        "--output-dir",
        str(dataset_dir),
        "--sequence-length",
        "60",
        "--window-size",
        "5",
        "--window-stride",
        "1",
        "--progress-mode",
        "linear",
        "--edge-mode",
        "dynamic_short_window",
        "--edge-threshold",
        "0.02",
        "--include-edge-sign",
        "--transition-segmentation",
        "auto_motion",
        "--motion-threshold",
        "0.12",
        "--transition-pad-frames",
        "2",
    ]
    _run(cmd)
    return dataset_dir


def build_split_and_reference(args: argparse.Namespace) -> tuple[Path, SplitCounts]:
    output_root = Path(args.output_root).resolve()
    dataset_dir = output_root / "graph_sequence_dataset"
    reference_root = output_root / "reference_sequence"
    reference_pool_dataset = reference_root / "reference_pool_dataset"
    if args.clean_reference:
        _safe_rmtree(reference_root, root=output_root)
    reference_root.mkdir(parents=True, exist_ok=True)
    reference_pool_dataset.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(dataset_dir / "dataset_manifest.csv")
    manifest = manifest.sort_values("trial_id").reset_index(drop=True)
    total = int(len(manifest))
    if total < 5:
        raise RuntimeError(f"Need at least 5 graph trials for split, found {total}")
    holdout_count = max(1, int(round(total * float(args.holdout_fraction))))
    reference_pool_count = total - holdout_count
    val_count = max(1, int(round(reference_pool_count * float(args.val_fraction))))
    train_count = max(1, reference_pool_count - val_count)
    if train_count + val_count != reference_pool_count:
        raise RuntimeError("Invalid split counts")

    holdout_indices = set(np.linspace(0, total - 1, holdout_count, dtype=int).tolist())
    holdout = manifest.iloc[[idx for idx in range(total) if idx in holdout_indices]].copy()
    reference_pool = manifest.iloc[[idx for idx in range(total) if idx not in holdout_indices]].copy()
    reference_pool.to_csv(reference_root / "reference_pool_manifest.csv", index=False)
    holdout.to_csv(reference_root / "holdout_manifest.csv", index=False)
    reference_pool.to_csv(reference_pool_dataset / "dataset_manifest.csv", index=False)
    if (dataset_dir / "dataset_config.json").exists():
        shutil.copy2(dataset_dir / "dataset_config.json", reference_pool_dataset / "dataset_config.json")

    cmd = [
        sys.executable,
        "scripts/offline_biomech/build_reference_sequence.py",
        "--dataset-dir",
        str(reference_pool_dataset),
        "--output-dir",
        str(reference_root / "reference"),
        "--cohort",
        "normal",
        "--onset-ratio",
        "0.20",
    ]
    _run(cmd)
    counts = SplitCounts(total=total, reference_pool=reference_pool_count, train=train_count, val=val_count, holdout=holdout_count)
    (reference_root / "split_counts.json").write_text(json.dumps(counts.__dict__, indent=2), encoding="utf-8")
    print(f"[INFO] Split counts: {counts}")
    return reference_root, counts


def train_and_plot(args: argparse.Namespace, *, split_counts: SplitCounts | None = None) -> None:
    output_root = Path(args.output_root).resolve()
    dataset_dir = output_root / "graph_sequence_dataset"
    reference_root = output_root / "reference_sequence"
    run_dir = Path(args.model_output_dir).resolve()
    if split_counts is None:
        split_json = reference_root / "split_counts.json"
        if not split_json.exists():
            raise RuntimeError(f"Missing split counts: {split_json}")
        split_counts = SplitCounts(**json.loads(split_json.read_text(encoding="utf-8")))

    cmd = [
        sys.executable,
        "scripts/offline_biomech/train_final_binary_gcn.py",
        "--dataset-dir",
        str(dataset_dir),
        "--reference-dir",
        str(reference_root / "reference"),
        "--reference-root-dir",
        str(reference_root),
        "--output-dir",
        str(run_dir),
        "--reports-dir",
        "reports",
        "--train-reference-trials",
        str(int(split_counts.train)),
        "--binary-threshold-quantile",
        "0.75",
        "--hidden-dim",
        "32",
        "--dropout",
        "0.15",
        "--epochs",
        str(int(args.epochs)),
        "--patience",
        str(int(args.patience)),
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--seed",
        "42",
        "--class-weighting",
        "--force-clean",
    ]
    _run(cmd)
    fig_cmd = [
        sys.executable,
        "scripts/analysis/plot_final_binary_gcn_figures.py",
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(run_dir / "final_figures"),
        "--best-epoch-mode",
        "metrics_json",
        "--title-prefix",
        "Final Binary Vanilla GCN + GUI Records",
        "--audit-report",
        "reports/final_binary_gui_plus_previous_figure_input_audit.md",
        "--interpretation-report",
        "reports/final_binary_gui_plus_previous_figure_interpretation.md",
    ]
    _run(fig_cmd)


def plot_existing_training(args: argparse.Namespace) -> None:
    run_dir = Path(args.model_output_dir).resolve()
    fig_cmd = [
        sys.executable,
        "scripts/analysis/plot_final_binary_gcn_figures.py",
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(run_dir / "final_figures"),
        "--best-epoch-mode",
        "metrics_json",
        "--title-prefix",
        "Final Binary Vanilla GCN + GUI Records",
        "--audit-report",
        "reports/final_binary_gui_plus_previous_figure_input_audit.md",
        "--interpretation-report",
        "reports/final_binary_gui_plus_previous_figure_interpretation.md",
    ]
    _run(fig_cmd)


def evaluate_abnormal_validation(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    run_dir = Path(args.model_output_dir).resolve()
    model_path = run_dir / "checkpoints" / "best_final_binary_gcn.pt"
    cmd = [
        sys.executable,
        "scripts/offline_biomech/evaluate_binary_gcn_validation_dataset.py",
        "--dataset-dir",
        str(output_root / "abnormal_graph_sequence_dataset"),
        "--reference-dir",
        str(output_root / "reference_sequence" / "reference"),
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_root / "abnormal_validation_eval"),
        "--reports-dir",
        "reports",
        "--cohort",
        "abnormal",
    ]
    _run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare/retrain binary GCN with previous open_to_fist data plus new GUI records.")
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "prepare_combined",
            "prepare_abnormal",
            "ctrl_ref",
            "ctrl_ref_combined",
            "ctrl_ref_abnormal",
            "merge",
            "graph",
            "graph_abnormal",
            "reference",
            "train",
            "figures_train",
            "figures_only",
            "eval_abnormal",
            "all_after_ctrl_ref",
            "all_after_combined_ctrl_ref",
            "all_abnormal_after_ctrl_ref",
            "all",
            "all_combined",
            "all_combined_with_abnormal_validation",
        ),
        required=True,
        help="Use `prepare_combined`, then `ctrl_ref_combined`, then `all_after_combined_ctrl_ref` for normal-data retraining.",
    )
    parser.add_argument("--records-root", default=str(DEFAULT_RECORDS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--previous-accepted-dir", default=str(DEFAULT_PREVIOUS_ACCEPTED))
    parser.add_argument("--previous-qpos-input", default=str(DEFAULT_PREVIOUS_QPOS_INPUT))
    parser.add_argument("--model-output-dir", default=str(PROJECT_ROOT / "artifacts" / "final_binary_gcn_run_gui_plus_previous"))
    parser.add_argument("--motion-type", default="open_to_fist")
    parser.add_argument("--date-prefix", default="20260603_", help="Only stage GUI trials whose path contains this text. Empty string includes all records.")
    parser.add_argument(
        "--abnormal-date-prefix",
        default=DEFAULT_ABNORMAL_DATE_PREFIX,
        help="Comma-separated path/date prefixes for abnormal validation trials. Empty string includes all records.",
    )
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--qpos-dim", type=int, default=23)
    parser.add_argument("--min-qpos-change", type=float, default=0.05)
    parser.add_argument("--max-gui-trials", type=int, default=0, help="Optional cap for slow ctrl_ref debugging. 0 means all staged GUI records.")
    parser.add_argument("--max-combined-trials", type=int, default=0, help="Optional cap for combined ctrl_ref debugging. 0 means all combined normal records.")
    parser.add_argument("--max-abnormal-trials", type=int, default=0, help="Optional cap for abnormal validation ctrl_ref debugging. 0 means all staged abnormal records.")
    parser.add_argument("--graph-source", choices=("merged", "combined"), default="merged")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--clean-prepare", action="store_true")
    parser.add_argument("--clean-ctrl-ref", action="store_true")
    parser.add_argument("--clean-merge", action="store_true")
    parser.add_argument("--clean-graph", action="store_true")
    parser.add_argument("--clean-reference", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        prepare_gui_records(args)
    if args.stage in {"prepare_combined", "all_combined", "all_combined_with_abnormal_validation"}:
        prepare_combined_qpos(args)
    if args.stage in {"prepare_abnormal", "all_combined_with_abnormal_validation"}:
        prepare_abnormal_records(args)
    if args.stage in {"ctrl_ref", "all"}:
        build_gui_ctrl_ref(args)
    if args.stage in {"ctrl_ref_combined", "all_combined", "all_combined_with_abnormal_validation"}:
        build_combined_ctrl_ref(args)
    if args.stage in {"ctrl_ref_abnormal", "all_combined_with_abnormal_validation"}:
        build_abnormal_ctrl_ref(args)
    split_counts: SplitCounts | None = None
    if args.stage in {"merge", "all_after_ctrl_ref", "all"}:
        merge_ctrl_ref(args)
    if args.stage in {"graph", "all_after_ctrl_ref", "all", "all_after_combined_ctrl_ref", "all_combined", "all_combined_with_abnormal_validation"}:
        if args.stage in {"all_after_combined_ctrl_ref", "all_combined", "all_combined_with_abnormal_validation"}:
            args.graph_source = "combined"
        build_graph_dataset(args)
    if args.stage in {"graph_abnormal", "all_abnormal_after_ctrl_ref", "all_combined_with_abnormal_validation"}:
        build_abnormal_graph_dataset(args)
    if args.stage in {"reference", "all_after_ctrl_ref", "all", "all_after_combined_ctrl_ref", "all_combined", "all_combined_with_abnormal_validation"}:
        _, split_counts = build_split_and_reference(args)
    if args.stage in {"train", "figures_train", "all_after_ctrl_ref", "all", "all_after_combined_ctrl_ref", "all_combined", "all_combined_with_abnormal_validation"}:
        train_and_plot(args, split_counts=split_counts)
    if args.stage == "figures_only":
        plot_existing_training(args)
    if args.stage in {"eval_abnormal", "all_abnormal_after_ctrl_ref", "all_combined_with_abnormal_validation"}:
        evaluate_abnormal_validation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
