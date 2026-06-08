from __future__ import annotations

"""Archived analyzer for legacy fast-predictor live runtime logs."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "live_runtime_analysis"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "live_runtime_analysis.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze archived fast-predictor live runtime CSV logs.")
    parser.add_argument("--input-log", type=str, required=True, help="CSV produced by runtime live logging.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for summary outputs.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown report path.")
    return parser.parse_args()


def percentile(values: pd.Series, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def burst_summary(delta_series: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(delta_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if arr.size == 0:
        return {
            "burst_threshold": 0.0,
            "num_bursts": 0,
            "max_burst_length": 0,
        }
    threshold = max(0.15, float(np.percentile(arr, 95.0)))
    mask = arr >= threshold
    lengths: list[int] = []
    current = 0
    for flag in mask:
        if bool(flag):
            current += 1
        elif current > 0:
            lengths.append(int(current))
            current = 0
    if current > 0:
        lengths.append(int(current))
    return {
        "burst_threshold": float(threshold),
        "num_bursts": int(len(lengths)),
        "max_burst_length": int(max(lengths) if lengths else 0),
    }


def main() -> int:
    args = parse_args()
    input_log = Path(args.input_log).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_log)
    for col in (
        "used_fallback",
        "dropped_frame",
        "suspicious_runtime_frame",
        "predictor_inference_ms",
        "total_ctrl_pipeline_ms",
        "total_frame_loop_ms",
        "ctrl_pred_delta_norm",
        "ctrl_after_delta_norm",
        "validation_window_error",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    worst_loop = df.sort_values(by="total_frame_loop_ms", ascending=False).head(20).copy() if "total_frame_loop_ms" in df.columns else df.head(0).copy()
    worst_delta = df.sort_values(by="ctrl_after_delta_norm", ascending=False).head(20).copy() if "ctrl_after_delta_norm" in df.columns else df.head(0).copy()
    burst = burst_summary(df["ctrl_after_delta_norm"] if "ctrl_after_delta_norm" in df.columns else pd.Series(dtype=float))

    summary = {
        "input_log": str(input_log),
        "num_frames_total": int(len(df)),
        "num_fallback_frames": int(df["used_fallback"].sum()) if "used_fallback" in df.columns else 0,
        "fallback_ratio": float(df["used_fallback"].mean()) if "used_fallback" in df.columns and len(df) else 0.0,
        "num_dropped_frames": int(df["dropped_frame"].sum()) if "dropped_frame" in df.columns else 0,
        "dropped_ratio": float(df["dropped_frame"].mean()) if "dropped_frame" in df.columns and len(df) else 0.0,
        "num_suspicious_frames": int(df["suspicious_runtime_frame"].sum()) if "suspicious_runtime_frame" in df.columns else 0,
        "suspicious_ratio": float(df["suspicious_runtime_frame"].mean()) if "suspicious_runtime_frame" in df.columns and len(df) else 0.0,
        "mean_total_frame_loop_ms": float(pd.to_numeric(df.get("total_frame_loop_ms", pd.Series(dtype=float)), errors="coerce").mean() if len(df) else 0.0),
        "median_total_frame_loop_ms": percentile(df.get("total_frame_loop_ms", pd.Series(dtype=float)), 50.0),
        "p95_total_frame_loop_ms": percentile(df.get("total_frame_loop_ms", pd.Series(dtype=float)), 95.0),
        "mean_predictor_inference_ms": float(pd.to_numeric(df.get("predictor_inference_ms", pd.Series(dtype=float)), errors="coerce").mean() if len(df) else 0.0),
        "median_predictor_inference_ms": percentile(df.get("predictor_inference_ms", pd.Series(dtype=float)), 50.0),
        "p95_predictor_inference_ms": percentile(df.get("predictor_inference_ms", pd.Series(dtype=float)), 95.0),
        "mean_total_ctrl_pipeline_ms": float(pd.to_numeric(df.get("total_ctrl_pipeline_ms", pd.Series(dtype=float)), errors="coerce").mean() if len(df) else 0.0),
        "median_total_ctrl_pipeline_ms": percentile(df.get("total_ctrl_pipeline_ms", pd.Series(dtype=float)), 50.0),
        "p95_total_ctrl_pipeline_ms": percentile(df.get("total_ctrl_pipeline_ms", pd.Series(dtype=float)), 95.0),
        "mean_ctrl_delta_before_smoothing": float(pd.to_numeric(df.get("ctrl_pred_delta_norm", pd.Series(dtype=float)), errors="coerce").mean() if len(df) else 0.0),
        "mean_ctrl_delta_after_smoothing": float(pd.to_numeric(df.get("ctrl_after_delta_norm", pd.Series(dtype=float)), errors="coerce").mean() if len(df) else 0.0),
        "mean_validation_window_error": float(pd.to_numeric(df.get("validation_window_error", pd.Series(dtype=float)), errors="coerce").replace(0.0, np.nan).mean() if len(df) else 0.0),
        "median_validation_window_error": percentile(pd.to_numeric(df.get("validation_window_error", pd.Series(dtype=float)), errors="coerce").replace(0.0, np.nan), 50.0),
        "p95_validation_window_error": percentile(pd.to_numeric(df.get("validation_window_error", pd.Series(dtype=float)), errors="coerce").replace(0.0, np.nan), 95.0),
        **burst,
    }

    worst_loop_path = output_dir / "worst_loop_frames.csv"
    worst_delta_path = output_dir / "largest_ctrl_delta_frames.csv"
    summary_json_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    worst_loop.to_csv(worst_loop_path, index=False)
    worst_delta.to_csv(worst_delta_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Live Runtime Log Analysis",
        "",
        f"- Input log: `{input_log}`",
        f"- Total frames: `{summary['num_frames_total']}`",
        f"- Fallback frames: `{summary['num_fallback_frames']}` (`{summary['fallback_ratio']:.4f}`)",
        f"- Dropped frames: `{summary['num_dropped_frames']}` (`{summary['dropped_ratio']:.4f}`)",
        f"- Suspicious frames: `{summary['num_suspicious_frames']}` (`{summary['suspicious_ratio']:.4f}`)",
        "",
        "## Timing",
        "",
        f"- Mean total frame loop ms: `{summary['mean_total_frame_loop_ms']:.4f}`",
        f"- Median total frame loop ms: `{summary['median_total_frame_loop_ms']:.4f}`",
        f"- p95 total frame loop ms: `{summary['p95_total_frame_loop_ms']:.4f}`",
        f"- Mean predictor inference ms: `{summary['mean_predictor_inference_ms']:.4f}`",
        f"- p95 predictor inference ms: `{summary['p95_predictor_inference_ms']:.4f}`",
        f"- Mean total ctrl pipeline ms: `{summary['mean_total_ctrl_pipeline_ms']:.4f}`",
        f"- p95 total ctrl pipeline ms: `{summary['p95_total_ctrl_pipeline_ms']:.4f}`",
        "",
        "## Jitter",
        "",
        f"- Mean ctrl delta before smoothing: `{summary['mean_ctrl_delta_before_smoothing']:.4f}`",
        f"- Mean ctrl delta after smoothing: `{summary['mean_ctrl_delta_after_smoothing']:.4f}`",
        f"- Mean validation window error: `{summary['mean_validation_window_error']:.4f}`",
        f"- p95 validation window error: `{summary['p95_validation_window_error']:.4f}`",
        f"- Burst threshold: `{summary['burst_threshold']:.4f}`",
        f"- Number of bursts: `{summary['num_bursts']}`",
        f"- Max burst length: `{summary['max_burst_length']}`",
        "",
        "## Output Files",
        "",
        f"- Summary JSON: `{summary_json_path}`",
        f"- Summary CSV: `{summary_csv_path}`",
        f"- Worst loop frames: `{worst_loop_path}`",
        f"- Largest ctrl-delta frames: `{worst_delta_path}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
