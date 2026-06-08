from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.graph_sequence_baseline import (  # noqa: E402
    compare_trial_to_reference,
    load_reference_sequence,
    load_trial_graph_sequence,
    save_trial_reference_comparison,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "open_to_fist_trial_vs_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one progress-normalized trial graph sequence against a reference sequence.")
    parser.add_argument("--trial-file", type=str, required=True, help="Trial .json or .npz produced by build_open_to_fist_graph_sequence_dataset.py.")
    parser.add_argument("--reference-dir", type=str, required=True, help="Reference directory produced by build_reference_graph_sequence.py.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Comparison output directory.")
    parser.add_argument("--onset-ratio", type=float, default=0.20, help="Relative threshold used to estimate activation onset.")
    parser.add_argument("--time-region-width", type=int, default=3, help="Window width used to summarize the worst progress region.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trial = load_trial_graph_sequence(Path(args.trial_file))
    reference = load_reference_sequence(Path(args.reference_dir))
    comparison = compare_trial_to_reference(
        trial,
        reference,
        onset_ratio=float(args.onset_ratio),
        time_region_width=int(args.time_region_width),
    )
    outputs = save_trial_reference_comparison(Path(args.output_dir), comparison)
    print(f"[INFO] Comparison summary: {outputs['summary_json_path']}")
    print(f"[INFO] Node deviation CSV: {outputs['node_csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
