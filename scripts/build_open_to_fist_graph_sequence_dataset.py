from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.graph_sequence_baseline import (  # noqa: E402
    build_dataset_manifest,
    build_trial_graph_sequence,
    detect_transition_segment,
    discover_recording_csvs,
    load_recording_trial,
    resolve_graph_template,
    save_trial_graph_sequence,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "open_to_fist_graph_sequence_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a progress-normalized graph-sequence dataset for open->fist style trials.")
    parser.add_argument("--normal-input", type=str, required=True, help="Folder or CSV for normal trials.")
    parser.add_argument("--abnormal-input", type=str, default="", help="Optional folder or CSV for patient/abnormal trials.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output dataset directory.")
    parser.add_argument("--sequence-length", type=int, default=40, help="Fixed number of progress-normalized steps.")
    parser.add_argument("--window-size", type=int, default=5, help="Window size over progress-normalized steps.")
    parser.add_argument("--window-stride", type=int, default=1, help="Window stride over progress-normalized steps.")
    parser.add_argument("--progress-mode", type=str, default="linear", choices=("linear", "resample_only"), help="Progress normalization mode.")
    parser.add_argument("--edge-mode", type=str, default="dynamic_short_window", choices=("static", "dynamic_short_window"), help="Edge construction mode per window.")
    parser.add_argument("--edge-threshold", type=float, default=0.02, help="Absolute threshold applied to window edge weights.")
    parser.add_argument("--include-edge-sign", action="store_true", help="Keep antagonistic relations negative.")
    parser.add_argument(
        "--node-feature-mode",
        type=str,
        default="legacy_ctrl",
        choices=("legacy_ctrl", "weakness_fusion"),
        help="Node feature contract. Use weakness_fusion for PD indicator + inverse-dynamics effort + qpos motion features.",
    )
    parser.add_argument("--transition-segmentation", type=str, default="auto_motion", choices=("none", "auto_motion"), help="Optional trim to isolate the transition region inside each trial.")
    parser.add_argument("--motion-threshold", type=float, default=0.12, help="Relative threshold used by auto_motion transition trimming.")
    parser.add_argument("--transition-pad-frames", type=int, default=2, help="Context frames kept before/after the detected motion region.")
    parser.add_argument("--max-trials-per-cohort", type=int, default=0, help="Optional cap per cohort.")
    return parser.parse_args()


def process_cohort(
    *,
    input_path: Path,
    cohort: str,
    output_dir: Path,
    graph_template: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    csv_paths = discover_recording_csvs(input_path)
    if int(args.max_trials_per_cohort) > 0:
        csv_paths = csv_paths[: int(args.max_trials_per_cohort)]
    if not csv_paths:
        return []

    records: list[dict[str, object]] = []
    for csv_path in csv_paths:
        trial = load_recording_trial(csv_path, source_root=input_path if input_path.is_dir() else input_path.parent)
        trial = detect_transition_segment(
            trial,
            segmentation_mode=str(args.transition_segmentation),
            motion_threshold=float(args.motion_threshold),
            pad_frames=int(args.transition_pad_frames),
        )
        sequence = build_trial_graph_sequence(
            trial,
            graph_template=graph_template,
            sequence_length=int(args.sequence_length),
            window_size=int(args.window_size),
            window_stride=int(args.window_stride),
            progress_mode=str(args.progress_mode),
            edge_mode=str(args.edge_mode),
            edge_threshold=float(args.edge_threshold),
            include_edge_sign=bool(args.include_edge_sign),
            cohort=str(cohort),
            node_feature_mode=str(args.node_feature_mode),
        )
        saved = save_trial_graph_sequence(output_dir, sequence)
        records.append({"metadata": sequence["metadata"], **saved})
    return records


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_path = Path(args.normal_input).resolve()
    abnormal_input = str(args.abnormal_input).strip()
    abnormal_path = Path(abnormal_input).resolve() if abnormal_input else None

    probe_paths: list[Path] = []
    probe_paths.extend(discover_recording_csvs(normal_path)[:1])
    if abnormal_path is not None:
        probe_paths.extend(discover_recording_csvs(abnormal_path)[:1])
    if not probe_paths:
        raise SystemExit("No recording CSVs found for dataset build.")
    probe_trial = load_recording_trial(probe_paths[0], source_root=normal_path if normal_path.is_dir() else normal_path.parent)
    graph_template = resolve_graph_template(expected_num_nodes=int(probe_trial["ctrl"].shape[1]))

    all_records: list[dict[str, object]] = []
    all_records.extend(
        process_cohort(
            input_path=normal_path,
            cohort="normal",
            output_dir=output_dir,
            graph_template=graph_template,
            args=args,
        )
    )

    if abnormal_input:
        all_records.extend(
            process_cohort(
                input_path=abnormal_path,
                cohort="abnormal",
                output_dir=output_dir,
                graph_template=graph_template,
                args=args,
            )
        )

    if not all_records:
        raise SystemExit("No recording CSVs found for dataset build.")

    manifest_df = build_dataset_manifest(all_records)
    manifest_path = output_dir / "dataset_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    config = {
        "normal_input": str(normal_path),
        "abnormal_input": abnormal_input,
        "sequence_length": int(args.sequence_length),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "progress_mode": str(args.progress_mode),
        "edge_mode": str(args.edge_mode),
        "edge_threshold": float(args.edge_threshold),
        "include_edge_sign": bool(args.include_edge_sign),
        "node_feature_mode": str(args.node_feature_mode),
        "transition_segmentation": str(args.transition_segmentation),
        "motion_threshold": float(args.motion_threshold),
        "transition_pad_frames": int(args.transition_pad_frames),
        "num_trials": int(len(manifest_df)),
        "num_normal_trials": int((manifest_df["cohort"] == "normal").sum()),
        "num_abnormal_trials": int((manifest_df["cohort"] == "abnormal").sum()),
    }
    (output_dir / "dataset_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[INFO] Dataset manifest: {manifest_path}")
    print(f"[INFO] Trials written: {len(manifest_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
