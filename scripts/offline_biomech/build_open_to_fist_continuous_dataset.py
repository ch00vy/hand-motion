from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FORMAL_MOTION_TOKENS = (
    "open_to_fist",
    "openfist",
    "open-fist",
    "open_to_close",
    "open-close",
)


@dataclass
class TrialBuildResult:
    trial_id: str
    subject_id: str
    motion_type: str
    source_file: str
    output_csv: str
    output_npz: str
    output_metadata: str
    num_frames_raw: int
    num_frames_output: int
    qpos_dim: int
    frame_rate_estimate: float | None
    duration_seconds: float | None
    qpos_path_norm: float
    start_end_qpos_distance: float
    missing_fraction: float
    is_proxy: bool
    proxy_reason: str
    status: str
    reject_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a formal progress-normalized open-to-fist continuous qpos dataset."
    )
    parser.add_argument("--input-path", required=True, help="CSV file or folder containing recorded qpos CSV files.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--sequence-length", type=int, default=100, help="Fixed output sequence length.")
    parser.add_argument(
        "--resample-mode",
        choices=("linear", "none"),
        default="linear",
        help="Progress normalization / resampling mode.",
    )
    parser.add_argument("--min-frames", type=int, default=20, help="Minimum raw frames required per trial.")
    parser.add_argument("--metadata-output", default=None, help="Optional path for dataset-level metadata JSON.")
    parser.add_argument("--report-path", default=None, help="Optional markdown audit report path.")
    parser.add_argument("--motion-type", default="open_to_fist", help="Motion type to write into metadata.")
    parser.add_argument("--subject-id", default=None, help="Fallback subject id if not present in source metadata.")
    parser.add_argument("--allow-proxy", action="store_true", help="Allow non-open-to-fist sources as proxy trials.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional maximum number of accepted trials.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input folders recursively. Non-recursive by default for predictable builds.",
    )
    return parser.parse_args()


def iter_csv_files(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".csv" else []
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in input_path.glob(pattern) if path.is_file())


def infer_trial_id(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    return stem or "trial"


def infer_subject_id(path: Path, fallback: str | None) -> str:
    if fallback:
        return str(fallback)
    for part in reversed(path.parts):
        match = re.search(r"(subject|subj|s)[_-]?([A-Za-z0-9]+)", part, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return "unknown"


def source_looks_formal_open_to_fist(path: Path, df: pd.DataFrame) -> bool:
    haystack = " ".join([str(path).lower(), " ".join(str(col).lower() for col in df.columns[:32])])
    if any(token in haystack for token in FORMAL_MOTION_TOKENS):
        return True
    for col in ("motion_type", "exercise_name", "gesture_label", "task_name"):
        if col in df.columns:
            values = " ".join(str(value).lower() for value in df[col].dropna().astype(str).unique()[:8])
            if any(token in values for token in FORMAL_MOTION_TOKENS):
                return True
    return False


def select_prefixed_columns(df: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    for prefix in prefixes:
        cols = [col for col in df.columns if str(col).startswith(prefix)]
        if cols:
            return sorted(cols, key=lambda name: int(str(name).rsplit("_", 1)[-1]) if str(name).rsplit("_", 1)[-1].isdigit() else str(name))
    return []


def estimate_frame_rate(df: pd.DataFrame) -> tuple[float | None, float | None]:
    time_cols = [col for col in ("timestamp", "time", "t", "seconds") if col in df.columns]
    if not time_cols:
        return None, None
    values = pd.to_numeric(df[time_cols[0]], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None, None
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None, None
    dt = float(np.median(diffs))
    frame_rate = 1.0 / dt if dt > 1e-9 else None
    duration = float(values[-1] - values[0]) if values[-1] >= values[0] else None
    return frame_rate, duration


def linear_resample(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected 2D frame x feature array.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if values.shape[0] == sequence_length:
        progress = np.linspace(0.0, 100.0, sequence_length, dtype=float)
        return values.copy(), progress
    source_x = np.linspace(0.0, 1.0, values.shape[0], dtype=float)
    target_x = np.linspace(0.0, 1.0, sequence_length, dtype=float)
    out = np.empty((sequence_length, values.shape[1]), dtype=float)
    for idx in range(values.shape[1]):
        out[:, idx] = np.interp(target_x, source_x, values[:, idx])
    progress = target_x * 100.0
    return out, progress


def phase_from_progress(progress: float) -> str:
    if progress <= 10.0:
        return "start_open"
    if progress < 85.0:
        return "transition"
    if progress < 95.0:
        return "near_fist"
    return "hold_fist"


def build_trial(
    source_file: Path,
    *,
    output_dir: Path,
    sequence_length: int,
    resample_mode: str,
    min_frames: int,
    motion_type: str,
    subject_id_fallback: str | None,
    allow_proxy: bool,
) -> TrialBuildResult:
    trial_id = infer_trial_id(source_file)
    subject_id = infer_subject_id(source_file, subject_id_fallback)
    try:
        df = pd.read_csv(source_file)
    except Exception as exc:
        return TrialBuildResult(
            trial_id,
            subject_id,
            motion_type,
            str(source_file),
            "",
            "",
            "",
            0,
            0,
            0,
            None,
            None,
            0.0,
            0.0,
            1.0,
            False,
            "",
            "rejected",
            f"read_failed:{type(exc).__name__}:{exc}",
        )

    formal = source_looks_formal_open_to_fist(source_file, df)
    is_proxy = not formal
    proxy_reason = "" if formal else "source_not_labeled_as_continuous_open_to_fist"
    if is_proxy and not allow_proxy:
        return TrialBuildResult(
            trial_id,
            subject_id,
            motion_type,
            str(source_file),
            "",
            "",
            "",
            int(len(df)),
            0,
            0,
            None,
            None,
            0.0,
            0.0,
            1.0,
            True,
            proxy_reason,
            "rejected",
            "proxy_source_requires_allow_proxy",
        )

    qpos_cols = select_prefixed_columns(df, ("qpos_", "qpos_target_", "target_qpos_", "qpos_raw_"))
    if not qpos_cols:
        return TrialBuildResult(
            trial_id,
            subject_id,
            motion_type,
            str(source_file),
            "",
            "",
            "",
            int(len(df)),
            0,
            0,
            None,
            None,
            0.0,
            0.0,
            1.0,
            is_proxy,
            proxy_reason,
            "rejected",
            "missing_qpos_columns",
        )

    qpos_raw = df[qpos_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing_fraction = float(np.count_nonzero(~np.isfinite(qpos_raw)) / max(1, qpos_raw.size))
    if qpos_raw.shape[0] < int(min_frames):
        return TrialBuildResult(
            trial_id,
            subject_id,
            motion_type,
            str(source_file),
            "",
            "",
            "",
            int(qpos_raw.shape[0]),
            0,
            int(qpos_raw.shape[1]),
            None,
            None,
            0.0,
            0.0,
            missing_fraction,
            is_proxy,
            proxy_reason,
            "rejected",
            f"too_few_frames:{qpos_raw.shape[0]}<{int(min_frames)}",
        )

    qpos_raw = np.where(np.isfinite(qpos_raw), qpos_raw, np.nan)
    qpos_raw = pd.DataFrame(qpos_raw).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
    frame_rate, duration = estimate_frame_rate(df)
    diffs = np.diff(qpos_raw, axis=0)
    qpos_path_norm = float(np.sum(np.linalg.norm(diffs, axis=1))) if diffs.size else 0.0
    start_end_distance = float(np.linalg.norm(qpos_raw[-1] - qpos_raw[0])) if qpos_raw.shape[0] else 0.0

    if resample_mode == "linear":
        qpos_out, progress = linear_resample(qpos_raw, int(sequence_length))
        source_frame_float = np.linspace(0.0, max(0.0, float(qpos_raw.shape[0] - 1)), int(sequence_length))
    else:
        qpos_out = qpos_raw
        progress = np.linspace(0.0, 100.0, qpos_raw.shape[0], dtype=float)
        source_frame_float = np.arange(qpos_raw.shape[0], dtype=float)

    trial_dir = output_dir / "continuous_trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    output_csv = trial_dir / f"{trial_id}.csv"
    output_npz = trial_dir / f"{trial_id}.npz"
    output_metadata = trial_dir / f"{trial_id}.json"

    frame_df = pd.DataFrame(qpos_out, columns=[f"qpos_{idx}" for idx in range(qpos_out.shape[1])])
    frame_df.insert(0, "phase", [phase_from_progress(float(value)) for value in progress])
    frame_df.insert(0, "progress_percent", progress)
    frame_df.insert(0, "source_frame_float", source_frame_float)
    frame_df.insert(0, "frame_index", np.arange(qpos_out.shape[0], dtype=int))
    frame_df.to_csv(output_csv, index=False)

    np.savez_compressed(
        output_npz,
        qpos=qpos_out,
        progress_percent=progress,
        source_frame_float=source_frame_float,
        trial_id=np.asarray(trial_id),
        source_file=np.asarray(str(source_file)),
    )

    metadata = {
        "trial_id": trial_id,
        "subject_id": subject_id,
        "motion_type": motion_type,
        "exercise_name": motion_type,
        "source_file": str(source_file),
        "num_frames_raw": int(qpos_raw.shape[0]),
        "num_frames_output": int(qpos_out.shape[0]),
        "qpos_dim": int(qpos_out.shape[1]),
        "frame_rate_estimate": frame_rate,
        "duration_seconds": duration,
        "qpos_path_norm": qpos_path_norm,
        "start_end_qpos_distance": start_end_distance,
        "missing_fraction": missing_fraction,
        "is_proxy": bool(is_proxy),
        "proxy_reason": proxy_reason,
        "progress_mode": resample_mode,
        "phase_rule": "progress_percent bins: 0-10 start_open, 10-85 transition, 85-95 near_fist, 95-100 hold_fist",
    }
    output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrialBuildResult(
        trial_id=trial_id,
        subject_id=subject_id,
        motion_type=motion_type,
        source_file=str(source_file),
        output_csv=str(output_csv),
        output_npz=str(output_npz),
        output_metadata=str(output_metadata),
        num_frames_raw=int(qpos_raw.shape[0]),
        num_frames_output=int(qpos_out.shape[0]),
        qpos_dim=int(qpos_out.shape[1]),
        frame_rate_estimate=frame_rate,
        duration_seconds=duration,
        qpos_path_norm=qpos_path_norm,
        start_end_qpos_distance=start_end_distance,
        missing_fraction=missing_fraction,
        is_proxy=bool(is_proxy),
        proxy_reason=proxy_reason,
        status="accepted",
        reject_reason="",
    )


def write_report(path: Path, *, input_path: Path, results: list[TrialBuildResult], summary: dict[str, object]) -> None:
    accepted = [item for item in results if item.status == "accepted"]
    rejected = [item for item in results if item.status != "accepted"]
    formal_count = sum(1 for item in accepted if not item.is_proxy)
    proxy_count = sum(1 for item in accepted if item.is_proxy)
    lines = [
        "# Open-to-Fist Source Data Audit",
        "",
        f"- Input path: `{input_path}`",
        f"- Source CSV files scanned: {len(results)}",
        f"- Accepted trials: {len(accepted)}",
        f"- Rejected trials: {len(rejected)}",
        f"- Formal continuous open-to-fist trials: {formal_count}",
        f"- Proxy trials: {proxy_count}",
        f"- Output sequence length: {summary.get('sequence_length')}",
        "",
        "## Interpretation",
        "",
    ]
    if formal_count == 0:
        lines.extend(
            [
                "No source trial was confidently identified as a formal continuous open-to-fist recording.",
                "Accepted trials should be treated as proxy data unless the metadata/source name is corrected.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "At least one source trial is labeled as continuous open-to-fist.",
                "Reference-quality eligibility still depends on biomechanical ctrl_ref validation.",
                "",
            ]
        )

    if accepted:
        accepted_df = pd.DataFrame([asdict(item) for item in accepted])
        lines.extend(
            [
                "## Accepted Trial Summary",
                "",
                f"- qpos dimensions: {sorted(accepted_df['qpos_dim'].unique().tolist())}",
                f"- raw frame min/median/max: {int(accepted_df['num_frames_raw'].min())} / {float(accepted_df['num_frames_raw'].median()):.1f} / {int(accepted_df['num_frames_raw'].max())}",
                f"- qpos path norm mean: {float(accepted_df['qpos_path_norm'].mean()):.4f}",
                f"- start-end qpos distance mean: {float(accepted_df['start_end_qpos_distance'].mean()):.4f}",
                "",
            ]
        )
    if rejected:
        reject_df = pd.DataFrame([asdict(item) for item in rejected])
        counts = reject_df["reject_reason"].value_counts().head(12)
        lines.extend(["## Rejection Reasons", ""])
        for reason, count in counts.items():
            lines.append(f"- `{reason}`: {int(count)}")
        lines.append("")

    lines.extend(
        [
            "## Output Files",
            "",
            f"- Manifest: `{summary.get('manifest_path')}`",
            f"- Source mapping: `{summary.get('source_mapping_path')}`",
            f"- Summary JSON: `{summary.get('summary_path')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = iter_csv_files(input_path, recursive=bool(args.recursive))
    results: list[TrialBuildResult] = []
    accepted_count = 0
    for source_file in source_files:
        if int(args.max_trials) > 0 and accepted_count >= int(args.max_trials):
            break
        result = build_trial(
            source_file,
            output_dir=output_dir,
            sequence_length=int(args.sequence_length),
            resample_mode=str(args.resample_mode),
            min_frames=int(args.min_frames),
            motion_type=str(args.motion_type),
            subject_id_fallback=args.subject_id,
            allow_proxy=bool(args.allow_proxy),
        )
        results.append(result)
        if result.status == "accepted":
            accepted_count += 1

    rows = [asdict(item) for item in results]
    manifest_df = pd.DataFrame([row for row in rows if row["status"] == "accepted"])
    rejected_df = pd.DataFrame([row for row in rows if row["status"] != "accepted"])
    manifest_path = output_dir / "dataset_manifest.csv"
    rejected_path = output_dir / "rejected_trials.csv"
    source_mapping_path = output_dir / "source_mapping.csv"
    summary_path = output_dir / "summary.json"

    manifest_df.to_csv(manifest_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)
    pd.DataFrame(rows).to_csv(source_mapping_path, index=False)

    accepted = [item for item in results if item.status == "accepted"]
    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "sequence_length": int(args.sequence_length) if str(args.resample_mode) == "linear" else None,
        "resample_mode": str(args.resample_mode),
        "min_frames": int(args.min_frames),
        "num_source_files": int(len(source_files)),
        "num_results": int(len(results)),
        "num_accepted": int(len(accepted)),
        "num_rejected": int(len(results) - len(accepted)),
        "num_formal_open_to_fist": int(sum(1 for item in accepted if not item.is_proxy)),
        "num_proxy": int(sum(1 for item in accepted if item.is_proxy)),
        "qpos_dims": sorted({int(item.qpos_dim) for item in accepted}),
        "manifest_path": str(manifest_path),
        "rejected_path": str(rejected_path),
        "source_mapping_path": str(source_mapping_path),
        "summary_path": str(summary_path),
    }
    if accepted:
        frame_counts = np.asarray([item.num_frames_raw for item in accepted], dtype=float)
        path_norms = np.asarray([item.qpos_path_norm for item in accepted], dtype=float)
        summary.update(
            {
                "raw_frames_min": int(np.min(frame_counts)),
                "raw_frames_median": float(np.median(frame_counts)),
                "raw_frames_max": int(np.max(frame_counts)),
                "qpos_path_norm_mean": float(np.mean(path_norms)),
                "qpos_path_norm_std": float(np.std(path_norms)),
            }
        )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.metadata_output:
        metadata_output = Path(args.metadata_output).resolve()
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = Path(args.report_path).resolve() if args.report_path else output_dir / "open_to_fist_source_data_audit.md"
    write_report(report_path, input_path=input_path, results=results, summary=summary)

    print(
        "[INFO] open-to-fist continuous dataset build done | "
        f"accepted={summary['num_accepted']} rejected={summary['num_rejected']} "
        f"formal={summary['num_formal_open_to_fist']} proxy={summary['num_proxy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
