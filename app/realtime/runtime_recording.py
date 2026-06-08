from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from node_features import NODE_TEMPORAL_FEATURE_NAMES
from phase_segmentation import VALID_PHASE_SEGMENTATION_STRATEGIES, segment_motion_phases

try:
    import torch
except ModuleNotFoundError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
SEQUENCE_SCHEMA_VERSION = 2


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_local_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_artifact_root() -> Path:
    load_dotenv()
    raw_root = os.getenv("ARTIFACT_ROOT")
    if raw_root:
        return resolve_local_path(raw_root)
    return PROJECT_ROOT / "artifacts" / "graphs"


class ExerciseSequenceBuffer:
    def __init__(self) -> None:
        self.active = False
        self.session_id: str | None = None
        self.exercise_name: str | None = None
        self.gesture_label: str | None = None
        self.task_name: str | None = None
        self.trial_id: str | None = None
        self.source_csv: str | None = None
        self.static_graph: dict[str, Any] | None = None
        self.enable_node_pipeline: bool = True
        self.disable_graph_artifact: bool = False
        self.phase_segmentation_strategy: str = "fixed_ratio"
        self.frame_timestamps: list[float] = []
        self.frame_indices: list[int] = []
        self.x_seq: list[np.ndarray] = []
        self.joint_qpos_seq: list[np.ndarray] = []
        self.ctrl_raw_seq: list[np.ndarray] = []
        self.ctrl_smoothed_seq: list[np.ndarray] = []
        self.ctrl_analysis_seq: list[np.ndarray] = []
        self.ctrl_rolling_mean_seq: list[np.ndarray] = []
        self.ctrl_rolling_median_seq: list[np.ndarray] = []
        self.ctrl_summary_seq: list[dict[str, Any]] = []
        self.dynamic_edge_mask_seq: list[np.ndarray] = []
        self.dynamic_edge_weight_seq: list[np.ndarray] = []
        self.visible_edge_records_seq: list[list[dict[str, Any]]] = []
        self.dynamic_edge_summary_seq: list[dict[str, Any]] = []

    def start(
        self,
        *,
        session_id: str,
        exercise_name: str | None,
        gesture_label: str | None,
        task_name: str | None,
        trial_id: str | None,
        source_csv: str,
        static_graph: dict[str, Any],
        enable_node_pipeline: bool = True,
        disable_graph_artifact: bool = False,
        phase_segmentation_strategy: str = "fixed_ratio",
    ) -> None:
        self.active = True
        self.session_id = session_id
        self.exercise_name = None if exercise_name is None else str(exercise_name)
        self.gesture_label = None if gesture_label is None else str(gesture_label)
        self.task_name = None if task_name is None else str(task_name)
        self.trial_id = None if trial_id is None else str(trial_id)
        self.source_csv = source_csv
        self.static_graph = static_graph
        self.enable_node_pipeline = bool(enable_node_pipeline)
        self.disable_graph_artifact = bool(disable_graph_artifact)
        strategy_name = str(phase_segmentation_strategy or "fixed_ratio").strip().lower()
        self.phase_segmentation_strategy = strategy_name if strategy_name in VALID_PHASE_SEGMENTATION_STRATEGIES else "fixed_ratio"
        self.frame_timestamps = []
        self.frame_indices = []
        self.x_seq = []
        self.joint_qpos_seq = []
        self.ctrl_raw_seq = []
        self.ctrl_smoothed_seq = []
        self.ctrl_analysis_seq = []
        self.ctrl_rolling_mean_seq = []
        self.ctrl_rolling_median_seq = []
        self.ctrl_summary_seq = []
        self.dynamic_edge_mask_seq = []
        self.dynamic_edge_weight_seq = []
        self.visible_edge_records_seq = []
        self.dynamic_edge_summary_seq = []

    def append(self, graph_state: dict[str, Any]) -> None:
        if not self.active:
            return
        ctrl_raw = graph_state.get("ctrl_raw")
        ctrl_smoothed = graph_state.get("ctrl_smoothed")
        ctrl_analysis = graph_state.get("ctrl_analysis")
        self.x_seq.append(np.asarray(graph_state["x"], dtype=np.float32))
        self.joint_qpos_seq.append(np.asarray(graph_state["joint_qpos"], dtype=np.float32))
        self.ctrl_raw_seq.append(np.asarray(ctrl_raw if ctrl_raw is not None else [], dtype=np.float32))
        self.ctrl_smoothed_seq.append(np.asarray(ctrl_smoothed if ctrl_smoothed is not None else [], dtype=np.float32))
        self.ctrl_analysis_seq.append(np.asarray(ctrl_analysis if ctrl_analysis is not None else [], dtype=np.float32))
        ctrl_summary = dict(graph_state.get("ctrl_summary") or {})
        ctrl_rolling_mean = ctrl_summary.get("ctrl_rolling_mean")
        ctrl_rolling_median = ctrl_summary.get("ctrl_rolling_median")
        self.ctrl_rolling_mean_seq.append(np.asarray(ctrl_rolling_mean if ctrl_rolling_mean is not None else [], dtype=np.float32))
        self.ctrl_rolling_median_seq.append(np.asarray(ctrl_rolling_median if ctrl_rolling_median is not None else [], dtype=np.float32))
        self.ctrl_summary_seq.append(ctrl_summary)
        self.dynamic_edge_mask_seq.append(np.asarray(graph_state["dynamic_edge_mask"], dtype=bool))
        self.dynamic_edge_weight_seq.append(np.asarray(graph_state["dynamic_edge_score"], dtype=np.float32))
        self.visible_edge_records_seq.append(list(graph_state.get("visible_edges") or []))
        self.dynamic_edge_summary_seq.append(dict(graph_state.get("dynamic_edge_summary") or {}))
        self.frame_timestamps.append(float(graph_state.get("timestamp") or 0.0))
        self.frame_indices.append(int(graph_state.get("frame_index") or len(self.frame_indices)))

    def has_frames(self) -> bool:
        return bool(self.x_seq)

    def build_artifact(self) -> dict[str, Any]:
        if self.static_graph is None or self.session_id is None:
            raise RuntimeError("ExerciseSequenceBuffer has not been started.")

        x_seq = np.stack(self.x_seq, axis=0) if self.x_seq else np.empty((0, 0, 0), dtype=np.float32)
        joint_qpos_seq = np.stack(self.joint_qpos_seq, axis=0) if self.joint_qpos_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_raw_seq = np.stack(self.ctrl_raw_seq, axis=0) if self.ctrl_raw_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_smoothed_seq = np.stack(self.ctrl_smoothed_seq, axis=0) if self.ctrl_smoothed_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_analysis_seq = np.stack(self.ctrl_analysis_seq, axis=0) if self.ctrl_analysis_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_rolling_mean_seq = np.stack(self.ctrl_rolling_mean_seq, axis=0) if self.ctrl_rolling_mean_seq else np.empty((0, 0), dtype=np.float32)
        ctrl_rolling_median_seq = np.stack(self.ctrl_rolling_median_seq, axis=0) if self.ctrl_rolling_median_seq else np.empty((0, 0), dtype=np.float32)
        dynamic_edge_mask_seq = np.stack(self.dynamic_edge_mask_seq, axis=0) if self.dynamic_edge_mask_seq else np.empty((0, 0), dtype=bool)
        dynamic_edge_weight_seq = np.stack(self.dynamic_edge_weight_seq, axis=0) if self.dynamic_edge_weight_seq else np.empty((0, 0), dtype=np.float32)
        phase_input = joint_qpos_seq if joint_qpos_seq.size else ctrl_analysis_seq
        phase_seq = segment_motion_phases(
            phase_input,
            timestamps=list(self.frame_timestamps),
            strategy=self.phase_segmentation_strategy,
        ) if phase_input.size else []
        node_names = list(self.static_graph["node_names"])
        node_sequence_feature_names = ["ctrl_raw", "ctrl_smoothed", "ctrl_analysis"]
        graph_artifacts = {
            "edge_type_names": list(self.static_graph["edge_type_names"]),
            "prior_edge_index": torch.tensor(self.static_graph["edge_index"], dtype=torch.long) if torch is not None else self.static_graph["edge_index"].tolist(),
            "prior_edge_type": torch.tensor(self.static_graph["edge_type"], dtype=torch.long) if torch is not None else self.static_graph["edge_type"].tolist(),
            "prior_edge_weight": torch.tensor(self.static_graph["edge_weight"], dtype=torch.float32) if torch is not None else self.static_graph["edge_weight"].tolist(),
            "prior_edges": list(self.static_graph["edges"]),
            "dynamic_edge_mask_seq": torch.tensor(dynamic_edge_mask_seq.astype(np.int8), dtype=torch.int8) if torch is not None else dynamic_edge_mask_seq.astype(np.int8).tolist(),
            "dynamic_edge_weight_seq": torch.tensor(dynamic_edge_weight_seq, dtype=torch.float32) if torch is not None else dynamic_edge_weight_seq.tolist(),
            "visible_edge_records_seq": self.visible_edge_records_seq,
            "dynamic_edge_summary_seq": self.dynamic_edge_summary_seq,
            "dynamic_edge_config": dict(self.static_graph.get("dynamic_edge_config") or {}),
            "graph_node_feature_names": list(self.static_graph["node_feature_names"]),
        }
        artifact: dict[str, Any] = {
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            "graph_kind": "muscle_node_sequence",
            "session_id": self.session_id,
            "exercise_name": self.exercise_name,
            "gesture_label": self.gesture_label if self.gesture_label is not None else self.exercise_name,
            "task_name": self.task_name,
            "trial_id": self.trial_id,
            "source_csv": self.source_csv,
            "num_frames": int(x_seq.shape[0]),
            "sequence_role": "node_first",
            "node_names": node_names,
            "node_feature_names": list(node_sequence_feature_names),
            "node_temporal_feature_names": list(NODE_TEMPORAL_FEATURE_NAMES),
            "x_seq": torch.tensor(x_seq, dtype=torch.float32) if torch is not None else x_seq.tolist(),
            "joint_qpos_seq": torch.tensor(joint_qpos_seq, dtype=torch.float32) if torch is not None else joint_qpos_seq.tolist(),
            "ctrl_raw_seq": torch.tensor(ctrl_raw_seq, dtype=torch.float32) if torch is not None else ctrl_raw_seq.tolist(),
            "ctrl_smoothed_seq": torch.tensor(ctrl_smoothed_seq, dtype=torch.float32) if torch is not None else ctrl_smoothed_seq.tolist(),
            "ctrl_analysis_seq": torch.tensor(ctrl_analysis_seq, dtype=torch.float32) if torch is not None else ctrl_analysis_seq.tolist(),
            "ctrl_rolling_mean_seq": torch.tensor(ctrl_rolling_mean_seq, dtype=torch.float32) if torch is not None else ctrl_rolling_mean_seq.tolist(),
            "ctrl_rolling_median_seq": torch.tensor(ctrl_rolling_median_seq, dtype=torch.float32) if torch is not None else ctrl_rolling_median_seq.tolist(),
            "ctrl_summary_seq": self.ctrl_summary_seq,
            "phase_seq": list(phase_seq),
            "joint_qpos_names": list(self.static_graph["joint_qpos_names"]),
            "timestamps": list(self.frame_timestamps),
            "frame_indices": list(self.frame_indices),
            "graph_artifacts": {} if self.disable_graph_artifact else graph_artifacts,
            "phase_segmentation": {
                "strategy": self.phase_segmentation_strategy,
                "available_strategies": list(VALID_PHASE_SEGMENTATION_STRATEGIES),
            },
            "meta": {
                "env_id": self.static_graph["env_id"],
                "graph_kind": self.static_graph["graph_kind"],
                "session_id": self.session_id,
                "exercise_name": self.exercise_name,
                "gesture_label": self.gesture_label if self.gesture_label is not None else self.exercise_name,
                "task_name": self.task_name,
                "trial_id": self.trial_id,
                "enable_node_pipeline": bool(self.enable_node_pipeline),
                "disable_graph_artifact": bool(self.disable_graph_artifact),
            },
        }
        if not self.disable_graph_artifact:
            artifact["edge_type_names"] = graph_artifacts["edge_type_names"]
            artifact["prior_edge_index"] = graph_artifacts["prior_edge_index"]
            artifact["prior_edge_type"] = graph_artifacts["prior_edge_type"]
            artifact["prior_edge_weight"] = graph_artifacts["prior_edge_weight"]
            artifact["prior_edges"] = graph_artifacts["prior_edges"]
            artifact["dynamic_edge_mask_seq"] = graph_artifacts["dynamic_edge_mask_seq"]
            artifact["dynamic_edge_weight_seq"] = graph_artifacts["dynamic_edge_weight_seq"]
            artifact["visible_edge_records_seq"] = graph_artifacts["visible_edge_records_seq"]
            artifact["dynamic_edge_summary_seq"] = graph_artifacts["dynamic_edge_summary_seq"]
        return artifact

    def stop(self) -> dict[str, Path] | None:
        if not self.active:
            return None

        self.active = False
        if not self.has_frames():
            self.session_id = None
            self.exercise_name = None
            self.gesture_label = None
            self.task_name = None
            self.trial_id = None
            self.source_csv = None
            self.static_graph = None
            self.enable_node_pipeline = True
            self.disable_graph_artifact = False
            self.phase_segmentation_strategy = "fixed_ratio"
            return None

        artifact = self.build_artifact()
        artifact_root = get_artifact_root() / "sequences"
        artifact_root.mkdir(parents=True, exist_ok=True)
        session_id = str(self.session_id)
        pt_path = artifact_root / f"exercise_{session_id}.pt"
        summary_path = artifact_root / f"exercise_{session_id}.summary.json"

        if torch is not None:
            torch.save(artifact, pt_path)

        summary = {
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            "graph_kind": "muscle_node_sequence",
            "session_id": session_id,
            "exercise_name": self.exercise_name,
            "gesture_label": self.gesture_label if self.gesture_label is not None else self.exercise_name,
            "task_name": self.task_name,
            "trial_id": self.trial_id,
            "source_csv": self.source_csv,
            "num_frames": artifact["num_frames"],
            "num_nodes": len(artifact["node_names"]),
            "num_prior_edges": int(np.asarray(self.static_graph["edge_weight"]).shape[0]),
            "mean_visible_edges": float(
                np.mean([int(np.asarray(mask, dtype=np.int8).sum()) for mask in self.dynamic_edge_mask_seq])
            )
            if self.dynamic_edge_mask_seq
            else 0.0,
            "node_feature_names": artifact["node_feature_names"],
            "node_temporal_feature_names": artifact["node_temporal_feature_names"],
            "joint_qpos_names": artifact["joint_qpos_names"],
            "phase_segmentation_strategy": self.phase_segmentation_strategy,
            "graph_artifact_included": not bool(self.disable_graph_artifact),
            "pt_path": str(pt_path) if torch is not None else None,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        self.session_id = None
        self.exercise_name = None
        self.gesture_label = None
        self.task_name = None
        self.trial_id = None
        self.source_csv = None
        self.static_graph = None
        self.enable_node_pipeline = True
        self.disable_graph_artifact = False
        self.phase_segmentation_strategy = "fixed_ratio"
        self.frame_timestamps = []
        self.frame_indices = []
        self.x_seq = []
        self.joint_qpos_seq = []
        self.ctrl_raw_seq = []
        self.ctrl_smoothed_seq = []
        self.ctrl_analysis_seq = []
        self.ctrl_rolling_mean_seq = []
        self.ctrl_rolling_median_seq = []
        self.ctrl_summary_seq = []
        self.dynamic_edge_mask_seq = []
        self.dynamic_edge_weight_seq = []
        self.visible_edge_records_seq = []
        self.dynamic_edge_summary_seq = []
        return {"pt_path": pt_path, "summary_path": summary_path}


def _normalize_trial_prefix(prefix: str | None) -> str:
    return str(prefix or "").strip()


def _iter_existing_trial_ids(records_dir: str = "records") -> list[str]:
    root = resolve_local_path(records_dir)
    if not root.exists():
        return []

    trial_ids: list[str] = []
    for csv_path in sorted(root.glob("ctrl_est_*.csv")):
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    trial_id = str(row.get("trial_id") or "").strip()
                    if trial_id:
                        trial_ids.append(trial_id)
                        break
        except Exception:
            continue
    return trial_ids


def _build_next_trial_id(prefix: str, records_dir: str = "records") -> str:
    normalized_prefix = _normalize_trial_prefix(prefix)
    if not normalized_prefix:
        normalized_prefix = "trial"

    pattern = re.compile(rf"^{re.escape(normalized_prefix)}_t(\d+)$", re.IGNORECASE)
    max_index = 0
    for trial_id in _iter_existing_trial_ids(records_dir=records_dir):
        match = pattern.match(str(trial_id).strip())
        if match is None:
            continue
        try:
            max_index = max(max_index, int(match.group(1)))
        except Exception:
            continue

    next_index = max_index + 1
    width = max(2, len(str(next_index)))
    return f"{normalized_prefix}_t{next_index:0{width}d}"


class RecordingManager:
    def __init__(
        self,
        env,
        static_graph=None,
        *,
        exercise_name: str | None = None,
        gesture_label: str | None = None,
        task_name: str | None = None,
        trial_id: str | None = None,
        trial_prefix: str | None = None,
        enable_node_pipeline: bool = True,
        disable_graph_artifact: bool = False,
        phase_segmentation_strategy: str = "fixed_ratio",
    ):
        self.env = env
        self.static_graph = static_graph
        self.exercise_name = None if exercise_name is None else str(exercise_name)
        self.gesture_label = None if gesture_label is None else str(gesture_label)
        self.task_name = None if task_name is None else str(task_name)
        self.trial_id_template = None if trial_id is None else str(trial_id)
        self.trial_prefix = _normalize_trial_prefix(trial_prefix)
        self.trial_id = None
        self.enable_node_pipeline = bool(enable_node_pipeline)
        self.disable_graph_artifact = bool(disable_graph_artifact)
        strategy_name = str(phase_segmentation_strategy or "fixed_ratio").strip().lower()
        self.phase_segmentation_strategy = strategy_name if strategy_name in VALID_PHASE_SEGMENTATION_STRATEGIES else "fixed_ratio"
        self.recording = False
        self.record_fh = None
        self.record_writer = None
        self.record_frame = 0
        self.record_path = None
        self.record_session_id = None
        self.record_started_at = None
        self.exercise_buffer = ExerciseSequenceBuffer()

    def stop(self):
        if self.record_fh is None:
            self.recording = False
            self.record_writer = None
            return
        try:
            self.record_fh.flush()
        except Exception:
            pass
        try:
            self.record_fh.close()
        except Exception:
            pass
        self.recording = False
        self.record_fh = None
        self.record_writer = None
        saved_paths = self.exercise_buffer.stop()
        if saved_paths is not None:
            print(f"[INFO] Exercise sequence saved: {saved_paths.get('pt_path')}")
            print(f"[INFO] Exercise summary saved: {saved_paths.get('summary_path')}")
        self.record_path = None
        self.record_session_id = None
        self.record_started_at = None
        self.trial_id = None
        print("[INFO] Recording stopped.")

    def start(self):
        if self.env is None:
            print("[WARN] Cannot start recording: env not created.")
            return

        os.makedirs("records", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.record_session_id = ts
        path = os.path.join("records", f"ctrl_est_{ts}.csv")
        self.record_path = path
        effective_trial_id = None
        if self.trial_id_template is not None and str(self.trial_id_template).strip():
            if str(self.trial_id_template).strip().lower() == "auto":
                prefix = self.trial_prefix or self.exercise_name or "trial"
                effective_trial_id = _build_next_trial_id(prefix, records_dir="records")
            else:
                effective_trial_id = str(self.trial_id_template).strip()
        elif self.trial_prefix:
            effective_trial_id = _build_next_trial_id(self.trial_prefix, records_dir="records")
        self.trial_id = effective_trial_id

        try:
            self.record_fh = open(path, "w", newline="")
            self.record_writer = csv.writer(self.record_fh)

            q_n = int(self.env.sim.data.qpos.shape[0])
            nu = int(self.env.sim.data.ctrl.shape[0])
            header = (
                ["t", "frame", "frame_index", "session_id", "exercise_name", "trial_id"]
                + [f"qpos_{i}" for i in range(q_n)]
                + [f"ctrl_{j}" for j in range(nu)]
                + [f"ctrl_raw_{j}" for j in range(nu)]
                + [f"ctrl_smoothed_{j}" for j in range(nu)]
                + [f"ctrl_analysis_{j}" for j in range(nu)]
            )
            self.record_writer.writerow(header)
            self.record_fh.flush()
            self.record_frame = 0
            self.recording = True
            self.record_started_at = float(time.time())
            if self.static_graph is not None:
                self.exercise_buffer.start(
                    session_id=ts,
                    exercise_name=self.exercise_name,
                    gesture_label=self.gesture_label if self.gesture_label is not None else self.exercise_name,
                    task_name=self.task_name,
                    trial_id=self.trial_id,
                    source_csv=path,
                    static_graph=self.static_graph,
                    enable_node_pipeline=self.enable_node_pipeline,
                    disable_graph_artifact=self.disable_graph_artifact,
                    phase_segmentation_strategy=self.phase_segmentation_strategy,
                )
            if self.trial_id:
                print(f"[INFO] Recording started: {path} | trial_id={self.trial_id}")
            else:
                print(f"[INFO] Recording started: {path}")
        except Exception as e:
            print(f"[WARN] Failed to start recording: {e}")
            self.recording = False
            self.record_fh = None
            self.record_writer = None
            self.record_started_at = None
            self.trial_id = None

    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

    def recording_elapsed_seconds(self, now: float | None = None) -> float:
        if not self.recording or self.record_started_at is None:
            return 0.0
        ref_now = float(time.time()) if now is None else float(now)
        return max(0.0, ref_now - float(self.record_started_at))

    def write_frame(self, ctrl_vec, graph_state=None):
        if not self.recording or self.record_writer is None or ctrl_vec is None:
            return

        try:
            q_vec = np.asarray(self.env.sim.data.qpos, dtype=float).reshape(-1)
            now_t = time.time()
            q_n = int(self.env.sim.data.qpos.shape[0])
            nu = int(self.env.sim.data.ctrl.shape[0])

            q_out = np.zeros(q_n, dtype=float)
            c_out = np.zeros(nu, dtype=float)
            c_raw = np.zeros(nu, dtype=float)
            c_smoothed = np.zeros(nu, dtype=float)
            c_analysis = np.zeros(nu, dtype=float)
            q_out[: min(q_n, len(q_vec))] = q_vec[: min(q_n, len(q_vec))]
            if isinstance(ctrl_vec, dict):
                ctrl_raw = np.asarray(ctrl_vec.get("ctrl_raw") if ctrl_vec.get("ctrl_raw") is not None else [], dtype=float).reshape(-1)
                ctrl_smoothed = np.asarray(ctrl_vec.get("ctrl_smoothed") if ctrl_vec.get("ctrl_smoothed") is not None else ctrl_raw, dtype=float).reshape(-1)
                ctrl_analysis = np.asarray(ctrl_vec.get("ctrl_analysis") if ctrl_vec.get("ctrl_analysis") is not None else ctrl_smoothed, dtype=float).reshape(-1)
            else:
                ctrl_analysis = np.asarray(ctrl_vec, dtype=float).reshape(-1)
                ctrl_smoothed = ctrl_analysis.copy()
                ctrl_raw = ctrl_analysis.copy()

            c_out[: min(nu, len(ctrl_analysis))] = ctrl_analysis[: min(nu, len(ctrl_analysis))]
            c_raw[: min(nu, len(ctrl_raw))] = ctrl_raw[: min(nu, len(ctrl_raw))]
            c_smoothed[: min(nu, len(ctrl_smoothed))] = ctrl_smoothed[: min(nu, len(ctrl_smoothed))]
            c_analysis[: min(nu, len(ctrl_analysis))] = ctrl_analysis[: min(nu, len(ctrl_analysis))]

            row = [
                float(now_t),
                int(self.record_frame),
                int(self.record_frame),
                str(self.record_session_id or ""),
                str(self.exercise_name or ""),
                str(self.trial_id or ""),
            ] + q_out.tolist() + c_out.tolist() + c_raw.tolist() + c_smoothed.tolist() + c_analysis.tolist()
            self.record_writer.writerow(row)
            if graph_state is not None:
                self.exercise_buffer.append(graph_state)
            self.record_frame += 1
            if (self.record_frame % 30) == 0:
                self.record_fh.flush()
        except Exception:
            pass
