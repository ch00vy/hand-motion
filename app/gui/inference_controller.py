from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.gcn_node_classifier import GCNNodeClassifier, build_adjacency_matrix, build_node_feature_matrix
from app.graph_sequence_baseline import compare_trial_to_reference, load_reference_sequence, load_trial_graph_sequence
from app.gui.trial_loader import TrialData, write_result_files
from app.temporal_gcn_node_classifier import is_temporal_gcn_checkpoint, predict_temporal_gcn_for_trial


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "HAND_TEMPORAL_GCN_MODEL_PATH",
        os.environ.get(
            "HAND_GCN_MODEL_PATH",
            str(PROJECT_ROOT / "artifacts" / "temporal_gcn_prototype_dynamic" / "checkpoints" / "best_temporal_gcn.pt"),
        ),
    )
)
DEFAULT_REFERENCE_DIR = Path(
    os.environ.get(
        "HAND_REFERENCE_DIR",
        str(PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous" / "reference_sequence" / "reference"),
    )
)
DEFAULT_CANONICAL_DATASET_DIR = Path(
    os.environ.get(
        "HAND_CANONICAL_DATASET_DIR",
        str(PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous" / "graph_sequence_dataset"),
    )
)


@dataclass
class AnalysisConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    reference_dir: Path = DEFAULT_REFERENCE_DIR
    canonical_dataset_dir: Path = DEFAULT_CANONICAL_DATASET_DIR
    sequence_length: int = 60
    window_size: int = 5
    window_stride: int = 1
    analysis_subdir: str = "dynamic_temporal_gcn"
    # <=0 means the offline biomech backend processes every recorded frame.
    # The graph sequence is still progress-normalized to sequence_length later.
    max_frames_per_trial: int = 0
    python_executable: str = sys.executable


class BinaryGCNInferenceController:
    """Run offline preprocessing and node-level binary GCN inference.

    The active default is dynamic Temporal GCN: each muscle node is classified
    from its feature trajectory plus the dynamic muscle-relation edge sequence.
    Vanilla GCN checkpoints are still supported as a compatibility fallback.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    def run_trial_analysis(self, trial: TrialData) -> dict[str, Any]:
        analysis_dir = trial.trial_dir / "analysis" / str(self.config.analysis_subdir)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        input_csv = self._write_biomech_input_csv(trial, analysis_dir / "input")
        ctrl_ref_dir = analysis_dir / "ctrl_ref"
        graph_dir = analysis_dir / "graph_sequence"

        self._build_ctrl_ref(input_csv=input_csv, output_dir=ctrl_ref_dir, report_path=analysis_dir / "ctrl_ref_report.md")
        accepted_dir = ctrl_ref_dir / "accepted_trials"
        if not any(accepted_dir.glob("*.csv")):
            raise RuntimeError(f"Biomech ctrl_ref produced no accepted trial under {accepted_dir}")
        self._build_graph_sequence(input_dir=accepted_dir, output_dir=graph_dir)

        manifest_path = graph_dir / "dataset_manifest.csv"
        if not manifest_path.exists():
            raise RuntimeError(f"Graph sequence manifest not found: {manifest_path}")
        manifest_df = pd.read_csv(manifest_path)
        if manifest_df.empty:
            raise RuntimeError(f"Graph sequence manifest is empty: {manifest_path}")
        graph_json = Path(str(manifest_df.iloc[0]["json_path"]))
        result = self._infer_graph(graph_json=graph_json, trial=trial)
        result["artifact_paths"] = {
            "analysis_dir": str(analysis_dir),
            "biomech_input_csv": str(input_csv),
            "ctrl_ref_dir": str(ctrl_ref_dir),
            "graph_sequence_dir": str(graph_dir),
            "graph_json": str(graph_json),
        }
        result_json, result_csv = write_result_files(trial, result)
        result["artifact_paths"]["result_json"] = str(result_json)
        result["artifact_paths"]["result_csv"] = str(result_csv)
        return result

    def find_control_reference_csv(self, trial: TrialData) -> Optional[Path]:
        result_path = trial.trial_dir / "result.json"
        candidate_dirs: list[Path] = []
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                ctrl_ref_dir = (result.get("artifact_paths") or {}).get("ctrl_ref_dir")
                if ctrl_ref_dir:
                    candidate_dirs.append(Path(str(ctrl_ref_dir)) / "accepted_trials")
            except Exception:
                pass
        candidate_dirs.append(trial.trial_dir / "analysis" / "final_binary_gcn" / "ctrl_ref" / "accepted_trials")
        candidate_dirs.append(trial.trial_dir / "analysis" / "dynamic_temporal_gcn" / "ctrl_ref" / "accepted_trials")
        for directory in candidate_dirs:
            if not directory.exists():
                continue
            files = sorted(directory.glob("*.csv"))
            if files:
                return files[0]
        return None

    def load_control_reference(self, trial: TrialData) -> dict[str, Any]:
        csv_path = self.find_control_reference_csv(trial)
        if csv_path is None:
            raise FileNotFoundError("Control replay requires analysis output. Please run analysis first.")
        df = pd.read_csv(csv_path)
        ctrl_columns = self._indexed_columns(df.columns, ("ctrl_ref_", "ctrl_teacher_", "ctrl_target_", "ctrl_"))
        if not ctrl_columns:
            raise ValueError(f"No ctrl_ref-compatible columns found in {csv_path}")
        ctrl = df[ctrl_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        if "frame_dt" in df.columns:
            dt_values = pd.to_numeric(df["frame_dt"], errors="coerce").to_numpy(dtype=float)
            dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0.0)]
            dt = float(np.median(dt_values)) if dt_values.size else float(trial.meta.get("dt") or (1.0 / 60.0))
        else:
            dt = float(trial.meta.get("dt") or (1.0 / 60.0))
        quality = self._load_control_reference_quality(csv_path=csv_path, trial=trial)
        return {
            "ctrl_sequence": ctrl,
            "dt": float(dt),
            "source_path": csv_path,
            "num_frames": int(ctrl.shape[0]),
            "num_actuators": int(ctrl.shape[1]),
            "quality": quality,
        }

    def control_reference_available(self, trial: TrialData) -> bool:
        return self.find_control_reference_csv(trial) is not None

    @staticmethod
    def _indexed_columns(columns: Any, prefixes: tuple[str, ...]) -> list[str]:
        for prefix in prefixes:
            indexed: list[tuple[int, str]] = []
            for name in columns:
                label = str(name)
                if not label.startswith(prefix):
                    continue
                suffix = label[len(prefix) :]
                if not suffix.isdigit():
                    continue
                indexed.append((int(suffix), label))
            if indexed:
                indexed.sort(key=lambda item: item[0])
                return [label for _, label in indexed]
        return []

    def _load_control_reference_quality(self, *, csv_path: Path, trial: TrialData) -> dict[str, Any]:
        summary_candidates = [
            csv_path.parent.parent / "biomech_reference_trial_summary.csv",
            csv_path.parent.parent / "teacher_trial_summary.csv",
        ]
        for summary_path in summary_candidates:
            if not summary_path.exists():
                continue
            try:
                df = pd.read_csv(summary_path)
            except Exception:
                continue
            if df.empty:
                continue
            rows = df
            if "trial_id" in rows.columns:
                trial_rows = rows.loc[rows["trial_id"].astype(str) == str(trial.trial_id)]
                if not trial_rows.empty:
                    rows = trial_rows
            row = rows.iloc[0].to_dict()
            out: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                    out[str(key)] = float(value)
                else:
                    out[str(key)] = value
            return out
        return {}

    def _write_biomech_input_csv(self, trial: TrialData, input_dir: Path) -> Path:
        input_dir.mkdir(parents=True, exist_ok=True)
        qpos = np.asarray(trial.qpos, dtype=float)
        frame_count = int(qpos.shape[0])
        dt = float(trial.meta.get("dt") or (1.0 / 60.0))
        trial_id = str(trial.trial_id)
        motion_type = str(trial.motion_type)
        subject_id = str(trial.meta.get("subject_id") or "")
        data: dict[str, Any] = {
            "frame_index": np.arange(frame_count, dtype=int),
            "frame": np.arange(frame_count, dtype=int),
            "t": np.arange(frame_count, dtype=float) * dt,
            "session_id": [subject_id] * frame_count,
            "subject_id": [subject_id] * frame_count,
            "trial_id": [trial_id] * frame_count,
            "exercise_name": [motion_type] * frame_count,
            "motion_type": [motion_type] * frame_count,
        }
        for idx in range(qpos.shape[1]):
            data[f"qpos_{idx}"] = qpos[:, idx]
        out = input_dir / f"ctrl_est_{trial_id}.csv"
        pd.DataFrame(data).to_csv(out, index=False)
        return out

    def _run_command(self, args: list[str], *, cwd: Path, log_path: Path) -> None:
        completed = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True)
        log_path.write_text(
            "COMMAND\n"
            + " ".join(args)
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\n\nSTDERR\n"
            + completed.stderr
            + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout)[-2000:]
            raise RuntimeError(f"Command failed ({completed.returncode}). See {log_path}\n{tail}")

    def _build_ctrl_ref(self, *, input_csv: Path, output_dir: Path, report_path: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            str(self.config.python_executable),
            "scripts/offline_biomech/build_biomech_ctrl_reference.py",
            "--input-path",
            str(input_csv),
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
            "--max-trials",
            "1",
            "--max-frames-per-trial",
            str(int(self.config.max_frames_per_trial)),
            "--max-mean-joint-error",
            "10",
            "--max-rmse-forward-error",
            "10",
            "--max-upper-saturation-fraction",
            "1.0",
            "--max-ctrl-delta-mean",
            "10",
        ]
        self._run_command(args, cwd=PROJECT_ROOT, log_path=output_dir / "build_ctrl_ref.log")

    def _build_graph_sequence(self, *, input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            str(self.config.python_executable),
            "scripts/offline_biomech/build_graph_sequence_dataset.py",
            "--normal-input",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--sequence-length",
            str(int(self.config.sequence_length)),
            "--window-size",
            str(int(self.config.window_size)),
            "--window-stride",
            str(int(self.config.window_stride)),
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
        self._run_command(args, cwd=PROJECT_ROOT, log_path=output_dir / "build_graph_sequence.log")

    def _infer_graph(self, *, graph_json: Path, trial: TrialData) -> dict[str, Any]:
        import torch

        checkpoint = torch.load(Path(self.config.model_path), map_location="cpu", weights_only=False)
        run_config = dict(checkpoint.get("run_config") or {})
        reference = load_reference_sequence(Path(self.config.reference_dir))
        graph_trial = self._align_to_canonical_template(load_trial_graph_sequence(Path(graph_json)))
        comparison = compare_trial_to_reference(
            graph_trial,
            reference,
            onset_ratio=float(run_config.get("onset_ratio", 0.2)),
            time_region_width=int(run_config.get("time_region_width", 3)),
        )
        if is_temporal_gcn_checkpoint(checkpoint):
            probabilities, class_names, model_variant = predict_temporal_gcn_for_trial(graph_trial, checkpoint)
            model_family = f"dynamic_temporal_gcn" if model_variant == "dynamic" else f"temporal_gcn_{model_variant}"
            summary_df = comparison["node_summary_df"].copy().sort_values("node_index").reset_index(drop=True)
        else:
            class_names = [str(item) for item in checkpoint.get("class_names") or ["within_reference", "deviating"]]
            x, feature_names, summary_df = build_node_feature_matrix(graph_trial, comparison)
            feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
            feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
            x_std = (np.asarray(x, dtype=np.float32) - feature_mean) / np.where(feature_std > 1e-8, feature_std, 1.0)
            adjacency = build_adjacency_matrix(graph_trial)
            model = GCNNodeClassifier(
                in_dim=int(x_std.shape[1]),
                hidden_dim=int(run_config.get("hidden_dim", 32)),
                num_classes=int(len(class_names)),
                dropout=float(run_config.get("dropout", 0.0)),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(x_std, dtype=torch.float32), torch.tensor(adjacency, dtype=torch.float32))
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            model_family = "vanilla_gcn"
        pred_idx = np.argmax(probabilities, axis=1)
        node_rows: list[dict[str, Any]] = []
        summary_rows = summary_df.sort_values("node_index").reset_index(drop=True).to_dict(orient="records")
        for idx, node_name in enumerate(list(graph_trial["node_names"])):
            row = dict(summary_rows[idx]) if idx < len(summary_rows) else {}
            label = class_names[int(pred_idx[idx])]
            deviating_idx = class_names.index("deviating") if "deviating" in class_names else int(pred_idx[idx])
            node_rows.append(
                {
                    "node_index": int(idx),
                    "node_name": str(node_name),
                    "predicted_label": label,
                    "probability_within_reference": float(probabilities[idx, 0]) if probabilities.shape[1] > 0 else 0.0,
                    "probability_deviating": float(probabilities[idx, deviating_idx]),
                    "confidence": float(np.max(probabilities[idx])),
                    "deviation_score": float(row.get("average_deviation_score", 0.0)),
                    "onset_status": str(row.get("onset_status", "")),
                    "magnitude_status": str(row.get("magnitude_status", "")),
                    "smoothness_status": str(row.get("smoothness_status", "")),
                }
            )
        node_rows.sort(key=lambda item: (item["predicted_label"] != "deviating", -float(item["probability_deviating"]), -float(item["deviation_score"])))
        deviating = [row for row in node_rows if row["predicted_label"] == "deviating"]
        node_deviation_df = comparison.get("node_deviation_df")
        node_progress_deviation: list[dict[str, Any]] = []
        if node_deviation_df is not None and not node_deviation_df.empty:
            for row in node_deviation_df.to_dict(orient="records"):
                node_progress_deviation.append(
                    {
                        "step_index": int(row.get("step_index", 0)),
                        "progress_percent": float(row.get("progress_percent", 0.0)),
                        "node_name": str(row.get("node_name", "")),
                        "deviation_score": float(row.get("aggregate_deviation_score", 0.0)),
                    }
                )
        worst_region = dict(comparison["summary"].get("worst_time_region") or {})
        return {
            "trial_id": str(trial.trial_id),
            "subject_id": str(trial.meta.get("subject_id") or ""),
            "motion_type": str(trial.motion_type),
            "model_path": str(self.config.model_path),
            "model_family": model_family,
            "reference_dir": str(self.config.reference_dir),
            "class_names": class_names,
            "summary": {
                "num_nodes": int(len(node_rows)),
                "num_deviating": int(len(deviating)),
                "num_within_reference": int(len(node_rows) - len(deviating)),
                "top_deviating_nodes": [row["node_name"] for row in deviating[:10]],
                "overall_node_deviation_score": float(comparison["summary"].get("overall_node_deviation_score", 0.0)),
                "overall_edge_deviation_score": float(comparison["summary"].get("overall_edge_deviation_score", 0.0)),
                "highest_deviation_region": worst_region,
            },
            "nodes": node_rows,
            "node_progress_deviation": node_progress_deviation,
        }

    def _align_to_canonical_template(self, trial: dict[str, Any]) -> dict[str, Any]:
        """Align GUI-built graph edges to the template used by final training.

        The anatomical graph builder can produce a few extra relation edges
        depending on the runtime MyoSuite context. The final reference sequence
        and final GCN were trained on one canonical edge template, so inference
        must use that same edge order/width.
        """
        manifest_path = Path(self.config.canonical_dataset_dir) / "dataset_manifest.csv"
        if not manifest_path.exists():
            return trial
        manifest_df = pd.read_csv(manifest_path)
        if manifest_df.empty or "json_path" not in manifest_df.columns:
            return trial
        canonical = load_trial_graph_sequence(Path(str(manifest_df.iloc[0]["json_path"])))
        canonical_edges = list(canonical.get("edges") or [])
        trial_edges = list(trial.get("edges") or [])
        if len(canonical_edges) == len(trial_edges):
            return trial

        def edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
            a = str(edge.get("source_name") or "")
            b = str(edge.get("target_name") or "")
            pair = tuple(sorted((a, b)))
            return (pair[0], pair[1], str(edge.get("edge_type") or ""))

        trial_index = {edge_key(edge): int(edge.get("edge_idx", idx)) for idx, edge in enumerate(trial_edges)}
        width = len(canonical_edges)
        aligned = dict(trial)
        aligned_edges: list[dict[str, Any]] = []
        for new_idx, edge in enumerate(canonical_edges):
            item = dict(edge)
            item["edge_idx"] = int(new_idx)
            aligned_edges.append(item)
        aligned["edges"] = aligned_edges
        aligned["edge_index"] = np.asarray(
            [[int(edge["source_idx"]) for edge in aligned_edges], [int(edge["target_idx"]) for edge in aligned_edges]],
            dtype=np.int64,
        )
        for key in ("edge_weight_raw_seq", "edge_active_seq", "edge_weight_seq"):
            arr = np.asarray(trial.get(key), dtype=np.float32 if key != "edge_active_seq" else np.int8)
            if arr.ndim != 2:
                continue
            out = np.zeros((arr.shape[0], width), dtype=arr.dtype)
            for new_idx, edge in enumerate(canonical_edges):
                old_idx = trial_index.get(edge_key(edge))
                if old_idx is not None and 0 <= old_idx < arr.shape[1]:
                    out[:, new_idx] = arr[:, old_idx]
            aligned[key] = out
        return aligned
