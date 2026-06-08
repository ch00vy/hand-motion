from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.graph_sequence_baseline import (  # noqa: E402
    build_reference_sequence,
    load_trial_graph_sequence,
    save_reference_sequence,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "open_to_fist_reference_sequence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reference graph sequence from progress-normalized normal trials.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset directory produced by build_open_to_fist_graph_sequence_dataset.py.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Reference output directory.")
    parser.add_argument("--cohort", type=str, default="normal", choices=("normal", "abnormal"), help="Cohort to aggregate.")
    parser.add_argument("--onset-ratio", type=float, default=0.20, help="Relative threshold used to estimate activation onset.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional cap on number of trials used for the reference.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    manifest_path = dataset_dir / "dataset_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    manifest_df = manifest_df[manifest_df["cohort"].astype(str).str.lower() == str(args.cohort).strip().lower()].copy()
    if manifest_df.empty:
        raise SystemExit(f"No trials found for cohort={args.cohort!r} in {manifest_path}")
    if int(args.max_trials) > 0:
        manifest_df = manifest_df.head(int(args.max_trials)).copy()

    trials = [load_trial_graph_sequence(Path(path)) for path in manifest_df["json_path"].tolist()]
    reference = build_reference_sequence(trials, onset_ratio=float(args.onset_ratio))
    outputs = save_reference_sequence(Path(args.output_dir), reference)
    print(f"[INFO] Reference built from {len(trials)} trials")
    print(f"[INFO] Reference summary: {outputs['summary_json_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
