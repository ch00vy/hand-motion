from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.realtime.monolithic_runtime_backend import (  # noqa: E402
    DEFAULT_MYOSUITE_ENV_ID,
    FALLBACK_MUSCLE_NAMES,
    build_muscle_graph_spec,
    create_muscle_model_context,
)

EPS = 1e-8
DEFAULT_ONSET_RATIO = 0.20
DEFAULT_TIME_REGION_WIDTH = 3
NODE_FEATURE_NAMES = ("mean_ctrl", "std_ctrl", "max_ctrl", "delta_ctrl", "auc_ctrl")


def dataframe_to_markdown_simple(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    columns = [str(col) for col in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + rows)


def _indexed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols: list[tuple[int, str]] = []
    for name in df.columns:
        label = str(name)
        if not label.startswith(prefix):
            continue
        try:
            idx = int(label[len(prefix) :])
        except ValueError:
            continue
        cols.append((idx, label))
    cols.sort(key=lambda item: item[0])
    return [label for _, label in cols]


def discover_recording_csvs(root: Path) -> list[Path]:
    root = Path(root).resolve()
    if root.is_file():
        return [root] if root.suffix.lower() == ".csv" else []
    return sorted(path.resolve() for path in root.rglob("*.csv") if path.is_file())


def infer_subject_id(path: Path, root: Path, df: pd.DataFrame) -> str:
    if "subject_id" in df.columns:
        values = [str(value).strip() for value in df["subject_id"].dropna().tolist() if str(value).strip()]
        if values:
            return values[0]
    try:
        rel = path.resolve().relative_to(root.resolve())
    except Exception:
        rel = path
    if len(rel.parts) > 1:
        return str(rel.parts[0])
    return ""


def load_recording_trial(path: Path, *, source_root: Path | None = None) -> dict[str, Any]:
    csv_path = Path(path).resolve()
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty recording CSV: {csv_path}")

    ctrl_cols: list[str] = []
    for prefix in ("ctrl_ref_", "ctrl_teacher_", "ctrl_target_", "ctrl_analysis_", "ctrl_"):
        ctrl_cols = _indexed_columns(df, prefix)
        if ctrl_cols:
            break
    qpos_cols = _indexed_columns(df, "qpos_")
    if not ctrl_cols:
        raise ValueError(f"No ctrl columns found in {csv_path}")

    ctrl = df[ctrl_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    qpos = (
        df[qpos_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if qpos_cols
        else np.zeros((len(df), 0), dtype=np.float32)
    )
    timestamps = (
        pd.to_numeric(df["t"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
        if "t" in df.columns
        else np.arange(len(df), dtype=np.float64)
    )
    exercise_name = ""
    if "exercise_name" in df.columns:
        values = [str(value).strip() for value in df["exercise_name"].dropna().tolist() if str(value).strip()]
        if values:
            exercise_name = values[0]
    if not exercise_name:
        exercise_name = csv_path.parent.name if csv_path.parent != csv_path.parent.parent else csv_path.stem

    trial_id = ""
    if "trial_id" in df.columns:
        values = [str(value).strip() for value in df["trial_id"].dropna().tolist() if str(value).strip()]
        if values:
            trial_id = values[0]
    if not trial_id:
        trial_id = csv_path.stem

    return {
        "source_csv": str(csv_path),
        "trial_id": trial_id,
        "exercise_name": exercise_name,
        "subject_id": infer_subject_id(csv_path, source_root or csv_path.parent, df),
        "num_frames": int(len(df)),
        "ctrl": ctrl,
        "qpos": qpos,
        "timestamps": timestamps,
        "ctrl_columns": ctrl_cols,
        "qpos_columns": qpos_cols,
    }


def _resample_array(values: np.ndarray, target_length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    target_length = max(1, int(target_length))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for resampling, got shape={arr.shape}")
    if arr.shape[0] == target_length:
        return arr.copy()
    if arr.shape[0] <= 1:
        return np.repeat(arr[:1], target_length, axis=0)
    src_progress = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64)
    dst_progress = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    out = np.zeros((target_length, arr.shape[1]), dtype=np.float32)
    for col_idx in range(arr.shape[1]):
        out[:, col_idx] = np.interp(dst_progress, src_progress, arr[:, col_idx]).astype(np.float32)
    return out


def _smooth_1d(values: np.ndarray, width: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 1 or int(width) <= 1:
        return arr.copy()
    kernel_width = max(1, int(width))
    kernel = np.ones(kernel_width, dtype=np.float32) / float(kernel_width)
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


def detect_transition_segment(
    trial: dict[str, Any],
    *,
    segmentation_mode: str,
    motion_threshold: float,
    pad_frames: int,
) -> dict[str, Any]:
    ctrl = np.asarray(trial["ctrl"], dtype=np.float32)
    qpos = np.asarray(trial["qpos"], dtype=np.float32)
    num_frames = int(ctrl.shape[0])
    if num_frames <= 1:
        return {
            **trial,
            "segment_start_frame": 0,
            "segment_end_frame": max(0, num_frames - 1),
            "segment_num_frames": num_frames,
            "transition_segmentation": "none",
            "motion_threshold": float(motion_threshold),
        }

    mode = str(segmentation_mode or "none").strip().lower()
    if mode == "none":
        return {
            **trial,
            "segment_start_frame": 0,
            "segment_end_frame": int(num_frames - 1),
            "segment_num_frames": num_frames,
            "transition_segmentation": "none",
            "motion_threshold": float(motion_threshold),
        }

    qpos_delta = np.linalg.norm(np.diff(qpos, axis=0), axis=1) if qpos.size else np.zeros(num_frames - 1, dtype=np.float32)
    ctrl_delta = np.linalg.norm(np.diff(ctrl, axis=0), axis=1) if ctrl.size else np.zeros(num_frames - 1, dtype=np.float32)
    qpos_signal = _smooth_1d(qpos_delta, width=5)
    ctrl_signal = _smooth_1d(ctrl_delta, width=5)

    qpos_scale = float(np.max(qpos_signal)) if qpos_signal.size else 0.0
    ctrl_scale = float(np.max(ctrl_signal)) if ctrl_signal.size else 0.0
    qpos_norm = qpos_signal / max(EPS, qpos_scale) if qpos_scale > 0.0 else np.zeros_like(qpos_signal)
    ctrl_norm = ctrl_signal / max(EPS, ctrl_scale) if ctrl_scale > 0.0 else np.zeros_like(ctrl_signal)

    if qpos_scale > 0.0 and ctrl_scale > 0.0:
        motion_signal = 0.7 * qpos_norm + 0.3 * ctrl_norm
    elif qpos_scale > 0.0:
        motion_signal = qpos_norm
    else:
        motion_signal = ctrl_norm

    threshold = float(np.clip(motion_threshold, 0.01, 0.95))
    active = np.where(motion_signal >= threshold)[0]
    if active.size <= 0:
        return {
            **trial,
            "segment_start_frame": 0,
            "segment_end_frame": int(num_frames - 1),
            "segment_num_frames": num_frames,
            "transition_segmentation": "fallback_full",
            "motion_threshold": float(threshold),
        }

    pad = max(0, int(pad_frames))
    start_frame = max(0, int(active[0]) - pad)
    end_frame = min(int(num_frames - 1), int(active[-1] + 1) + pad)
    if end_frame <= start_frame:
        start_frame = 0
        end_frame = int(num_frames - 1)

    return {
        **trial,
        "ctrl": ctrl[start_frame : end_frame + 1].copy(),
        "qpos": qpos[start_frame : end_frame + 1].copy(),
        "timestamps": np.asarray(trial["timestamps"], dtype=np.float64)[start_frame : end_frame + 1].copy(),
        "num_frames": int(end_frame - start_frame + 1),
        "segment_start_frame": int(start_frame),
        "segment_end_frame": int(end_frame),
        "segment_num_frames": int(end_frame - start_frame + 1),
        "transition_segmentation": "auto_motion",
        "motion_threshold": float(threshold),
    }


def resolve_graph_template(*, expected_num_nodes: int, env_id: str = DEFAULT_MYOSUITE_ENV_ID) -> dict[str, Any]:
    graph_spec = None
    graph_source = "fallback"
    graph_warning = ""
    try:
        context = create_muscle_model_context(env_id)
        graph_spec = build_muscle_graph_spec(context=context, include_data_driven=False)
        graph_source = "myosuite_context"
    except Exception as exc:
        graph_warning = f"{type(exc).__name__}: {exc}"
        graph_spec = build_muscle_graph_spec(context=None, include_data_driven=False)
        graph_source = "fallback_static"

    node_names = [str(node["node_name"]) for node in graph_spec["nodes"]]
    if len(node_names) != int(expected_num_nodes):
        fallback_names = [str(name) for name in list(FALLBACK_MUSCLE_NAMES)[: int(expected_num_nodes)]]
        if len(fallback_names) == int(expected_num_nodes):
            node_names = fallback_names
            graph_source = f"{graph_source}_trimmed"
        else:
            node_names = [f"muscle_{idx}" for idx in range(int(expected_num_nodes))]
            graph_source = f"{graph_source}_generic"

    node_index = {name: idx for idx, name in enumerate(node_names)}
    edges: list[dict[str, Any]] = []
    for edge in graph_spec.get("edges", []):
        source_name = str(edge.get("source_name"))
        target_name = str(edge.get("target_name"))
        if source_name not in node_index or target_name not in node_index:
            continue
        edges.append(
            {
                "edge_idx": len(edges),
                "source_idx": int(node_index[source_name]),
                "target_idx": int(node_index[target_name]),
                "source_name": source_name,
                "target_name": target_name,
                "edge_type": str(edge.get("edge_type", "functional_group")),
                "prior_weight": float(edge.get("edge_weight", 1.0)),
                "metadata": dict(edge.get("metadata") or {}),
            }
        )
    return {
        "env_id": env_id,
        "node_names": node_names,
        "edges": edges,
        "graph_source": graph_source,
        "graph_warning": graph_warning,
    }


def relation_sign(edge_type: str) -> float:
    if str(edge_type).strip().lower() == "antagonist":
        return -1.0
    return 1.0


def build_window_node_features(ctrl_window: np.ndarray) -> np.ndarray:
    window = np.asarray(ctrl_window, dtype=np.float32)
    mean_ctrl = np.mean(window, axis=0)
    std_ctrl = np.std(window, axis=0)
    max_ctrl = np.max(window, axis=0)
    delta_ctrl = window[-1] - window[0]
    auc_ctrl = np.trapz(window, dx=1.0 / max(1, window.shape[0] - 1), axis=0)
    return np.stack((mean_ctrl, std_ctrl, max_ctrl, delta_ctrl, auc_ctrl), axis=1).astype(np.float32)


def build_window_edge_weights(
    ctrl_window: np.ndarray,
    edges: list[dict[str, Any]],
    *,
    edge_mode: str,
    edge_threshold: float,
    include_edge_sign: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.asarray(ctrl_window, dtype=np.float32)
    edge_threshold = max(0.0, float(edge_threshold))
    raw = np.zeros(len(edges), dtype=np.float32)
    active = np.zeros(len(edges), dtype=np.int8)
    signed = np.zeros(len(edges), dtype=np.float32)
    if not edges:
        return raw, active, signed

    for edge in edges:
        edge_idx = int(edge["edge_idx"])
        src_idx = int(edge["source_idx"])
        dst_idx = int(edge["target_idx"])
        base_sign = relation_sign(str(edge["edge_type"]))
        prior_weight = float(edge["prior_weight"])
        if str(edge_mode).strip().lower() == "static":
            value = prior_weight
        else:
            coactivation = float(np.mean(window[:, src_idx] * window[:, dst_idx]))
            value = prior_weight * coactivation
        if include_edge_sign:
            value *= base_sign
        raw[edge_idx] = float(value)
        if abs(float(value)) >= edge_threshold:
            active[edge_idx] = 1
            signed[edge_idx] = float(value)
    return raw, active, signed


def build_trial_graph_sequence(
    trial: dict[str, Any],
    *,
    graph_template: dict[str, Any],
    sequence_length: int,
    window_size: int,
    window_stride: int,
    progress_mode: str,
    edge_mode: str,
    edge_threshold: float,
    include_edge_sign: bool,
    cohort: str,
) -> dict[str, Any]:
    ctrl = np.asarray(trial["ctrl"], dtype=np.float32)
    qpos = np.asarray(trial["qpos"], dtype=np.float32)
    sequence_length = max(1, int(sequence_length))
    window_size = max(1, int(window_size))
    window_stride = max(1, int(window_stride))
    resolved_progress_mode = str(progress_mode).strip().lower()
    if resolved_progress_mode not in {"linear", "resample_only"}:
        resolved_progress_mode = "linear"

    ctrl_resampled = _resample_array(ctrl, sequence_length)
    qpos_resampled = _resample_array(qpos, sequence_length) if qpos.size else np.zeros((sequence_length, 0), dtype=np.float32)
    progress_steps = np.linspace(0.0, 100.0, sequence_length, dtype=np.float32)

    if window_size > sequence_length:
        window_size = sequence_length
    starts = list(range(0, max(1, sequence_length - window_size + 1), window_stride))
    if starts[-1] != sequence_length - window_size:
        starts.append(sequence_length - window_size)

    node_feature_seq = np.zeros((len(starts), ctrl_resampled.shape[1], len(NODE_FEATURE_NAMES)), dtype=np.float32)
    edge_weight_raw_seq = np.zeros((len(starts), len(graph_template["edges"])), dtype=np.float32)
    edge_active_seq = np.zeros((len(starts), len(graph_template["edges"])), dtype=np.int8)
    edge_weight_seq = np.zeros((len(starts), len(graph_template["edges"])), dtype=np.float32)
    window_progress = np.zeros(len(starts), dtype=np.float32)
    window_progress_start = np.zeros(len(starts), dtype=np.float32)
    window_progress_end = np.zeros(len(starts), dtype=np.float32)

    for window_idx, start_idx in enumerate(starts):
        stop_idx = start_idx + window_size
        ctrl_window = ctrl_resampled[start_idx:stop_idx]
        node_feature_seq[window_idx] = build_window_node_features(ctrl_window)
        raw_edge, active_edge, signed_edge = build_window_edge_weights(
            ctrl_window,
            graph_template["edges"],
            edge_mode=edge_mode,
            edge_threshold=edge_threshold,
            include_edge_sign=include_edge_sign,
        )
        edge_weight_raw_seq[window_idx] = raw_edge
        edge_active_seq[window_idx] = active_edge
        edge_weight_seq[window_idx] = signed_edge
        window_progress_start[window_idx] = float(progress_steps[start_idx])
        window_progress_end[window_idx] = float(progress_steps[stop_idx - 1])
        window_progress[window_idx] = float(np.mean(progress_steps[start_idx:stop_idx]))

    edge_index = (
        np.asarray([[edge["source_idx"] for edge in graph_template["edges"]], [edge["target_idx"] for edge in graph_template["edges"]]], dtype=np.int64)
        if graph_template["edges"]
        else np.zeros((2, 0), dtype=np.int64)
    )

    metadata = {
        "subject_id": str(trial.get("subject_id") or ""),
        "trial_id": str(trial["trial_id"]),
        "exercise_name": str(trial.get("exercise_name") or ""),
        "source_csv": str(trial["source_csv"]),
        "num_frames": int(trial["num_frames"]),
        "segment_start_frame": int(trial.get("segment_start_frame", 0)),
        "segment_end_frame": int(trial.get("segment_end_frame", max(0, int(trial["num_frames"]) - 1))),
        "segment_num_frames": int(trial.get("segment_num_frames", int(trial["num_frames"]))),
        "transition_segmentation": str(trial.get("transition_segmentation", "none")),
        "motion_threshold": float(trial.get("motion_threshold", 0.0)),
        "num_steps": int(sequence_length),
        "num_windows": int(len(starts)),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "progress_mode": resolved_progress_mode,
        "edge_mode": str(edge_mode),
        "edge_threshold": float(edge_threshold),
        "include_edge_sign": bool(include_edge_sign),
        "cohort": str(cohort),
        "graph_source": str(graph_template.get("graph_source") or ""),
        "graph_warning": str(graph_template.get("graph_warning") or ""),
    }

    return {
        "metadata": metadata,
        "progress_steps": progress_steps,
        "window_progress": window_progress,
        "window_progress_start": window_progress_start,
        "window_progress_end": window_progress_end,
        "qpos_resampled": qpos_resampled,
        "ctrl_resampled": ctrl_resampled,
        "node_feature_seq": node_feature_seq,
        "edge_weight_raw_seq": edge_weight_raw_seq,
        "edge_active_seq": edge_active_seq,
        "edge_weight_seq": edge_weight_seq,
        "edge_index": edge_index,
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "node_names": list(graph_template["node_names"]),
        "edges": list(graph_template["edges"]),
    }


def trial_file_stem(cohort: str, trial_id: str) -> str:
    safe_trial_id = str(trial_id).replace(" ", "_")
    return f"{str(cohort).strip().lower()}__{safe_trial_id}"


def save_trial_graph_sequence(output_dir: Path, trial_sequence: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(trial_sequence["metadata"])
    stem = trial_file_stem(str(metadata["cohort"]), str(metadata["trial_id"]))
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    np.savez_compressed(
        npz_path,
        progress_steps=np.asarray(trial_sequence["progress_steps"], dtype=np.float32),
        window_progress=np.asarray(trial_sequence["window_progress"], dtype=np.float32),
        window_progress_start=np.asarray(trial_sequence["window_progress_start"], dtype=np.float32),
        window_progress_end=np.asarray(trial_sequence["window_progress_end"], dtype=np.float32),
        qpos_resampled=np.asarray(trial_sequence["qpos_resampled"], dtype=np.float32),
        ctrl_resampled=np.asarray(trial_sequence["ctrl_resampled"], dtype=np.float32),
        node_feature_seq=np.asarray(trial_sequence["node_feature_seq"], dtype=np.float32),
        edge_weight_raw_seq=np.asarray(trial_sequence["edge_weight_raw_seq"], dtype=np.float32),
        edge_active_seq=np.asarray(trial_sequence["edge_active_seq"], dtype=np.int8),
        edge_weight_seq=np.asarray(trial_sequence["edge_weight_seq"], dtype=np.float32),
        edge_index=np.asarray(trial_sequence["edge_index"], dtype=np.int64),
    )

    json_payload = {
        **metadata,
        "node_feature_names": list(trial_sequence["node_feature_names"]),
        "node_names": list(trial_sequence["node_names"]),
        "num_nodes": int(len(trial_sequence["node_names"])),
        "num_edges": int(len(trial_sequence["edges"])),
        "edges": list(trial_sequence["edges"]),
        "npz_path": str(npz_path),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return {"npz_path": str(npz_path), "json_path": str(json_path)}


def load_trial_graph_sequence(path: Path) -> dict[str, Any]:
    candidate = Path(path).resolve()
    if candidate.suffix.lower() == ".json":
        json_path = candidate
    elif candidate.suffix.lower() == ".npz":
        json_path = candidate.with_suffix(".json")
    else:
        json_path = candidate.with_suffix(".json")
    npz_path = json_path.with_suffix(".npz")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    bundle = np.load(npz_path)
    return {
        "metadata": metadata,
        "progress_steps": np.asarray(bundle["progress_steps"], dtype=np.float32),
        "window_progress": np.asarray(bundle["window_progress"], dtype=np.float32),
        "window_progress_start": np.asarray(bundle["window_progress_start"], dtype=np.float32) if "window_progress_start" in bundle.files else np.asarray(bundle["window_progress"], dtype=np.float32),
        "window_progress_end": np.asarray(bundle["window_progress_end"], dtype=np.float32) if "window_progress_end" in bundle.files else np.asarray(bundle["window_progress"], dtype=np.float32),
        "qpos_resampled": np.asarray(bundle["qpos_resampled"], dtype=np.float32),
        "ctrl_resampled": np.asarray(bundle["ctrl_resampled"], dtype=np.float32),
        "node_feature_seq": np.asarray(bundle["node_feature_seq"], dtype=np.float32),
        "edge_weight_raw_seq": np.asarray(bundle["edge_weight_raw_seq"], dtype=np.float32),
        "edge_active_seq": np.asarray(bundle["edge_active_seq"], dtype=np.int8),
        "edge_weight_seq": np.asarray(bundle["edge_weight_seq"], dtype=np.float32),
        "edge_index": np.asarray(bundle["edge_index"], dtype=np.int64),
        "node_feature_names": list(metadata["node_feature_names"]),
        "node_names": list(metadata["node_names"]),
        "edges": list(metadata["edges"]),
    }


def build_dataset_manifest(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in records:
        rows.append(
            {
                "cohort": str(item["metadata"]["cohort"]),
                "subject_id": str(item["metadata"].get("subject_id") or ""),
                "trial_id": str(item["metadata"]["trial_id"]),
                "exercise_name": str(item["metadata"].get("exercise_name") or ""),
                "source_csv": str(item["metadata"]["source_csv"]),
                "num_frames": int(item["metadata"]["num_frames"]),
                "segment_num_frames": int(item["metadata"].get("segment_num_frames", item["metadata"]["num_frames"])),
                "transition_segmentation": str(item["metadata"].get("transition_segmentation", "none")),
                "num_steps": int(item["metadata"]["num_steps"]),
                "num_windows": int(item["metadata"]["num_windows"]),
                "edge_mode": str(item["metadata"]["edge_mode"]),
                "progress_mode": str(item["metadata"]["progress_mode"]),
                "npz_path": str(item["npz_path"]),
                "json_path": str(item["json_path"]),
            }
        )
    return pd.DataFrame(rows)


def activation_curve(node_feature_seq: np.ndarray, node_idx: int) -> np.ndarray:
    return np.asarray(node_feature_seq[:, node_idx, 0], dtype=np.float32)


def estimate_onset(curve: np.ndarray, progress: np.ndarray, *, threshold_value: float) -> dict[str, float]:
    values = np.asarray(curve, dtype=np.float32).reshape(-1)
    prog = np.asarray(progress, dtype=np.float32).reshape(-1)
    if values.size == 0 or prog.size == 0:
        return {"onset_index": -1.0, "onset_progress": math.nan}
    hits = np.where(values >= float(threshold_value))[0]
    if hits.size <= 0:
        return {"onset_index": -1.0, "onset_progress": math.nan}
    idx = int(hits[0])
    return {"onset_index": float(idx), "onset_progress": float(prog[idx])}


def baseline_value(curve: np.ndarray, fraction: float = 0.15) -> float:
    values = np.asarray(curve, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    count = max(1, int(round(values.size * float(np.clip(fraction, 0.05, 0.50)))))
    return float(np.median(values[:count]))


def smoothness_score(curve: np.ndarray) -> float:
    values = np.asarray(curve, dtype=np.float32).reshape(-1)
    if values.size <= 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values, n=2))))


def stable_deviation_scale(reference_mean: np.ndarray, reference_std: np.ndarray, *, minimum_scale: float) -> np.ndarray:
    mean_arr = np.asarray(reference_mean, dtype=np.float32)
    std_arr = np.asarray(reference_std, dtype=np.float32)
    baseline_scale = 0.10 * np.maximum(np.abs(mean_arr), np.float32(minimum_scale))
    return np.maximum.reduce(
        [
            std_arr,
            baseline_scale,
            np.full(std_arr.shape, float(minimum_scale), dtype=np.float32),
        ]
    )


def build_reference_sequence(trials: list[dict[str, Any]], *, onset_ratio: float = DEFAULT_ONSET_RATIO) -> dict[str, Any]:
    if not trials:
        raise ValueError("No trials provided for reference sequence.")

    first = trials[0]
    node_names = list(first["node_names"])
    edges = list(first["edges"])
    node_feature_names = list(first["node_feature_names"])
    window_progress = np.asarray(first["window_progress"], dtype=np.float32)
    window_progress_start = np.asarray(first.get("window_progress_start", first["window_progress"]), dtype=np.float32)
    window_progress_end = np.asarray(first.get("window_progress_end", first["window_progress"]), dtype=np.float32)
    node_stack = np.stack([np.asarray(item["node_feature_seq"], dtype=np.float32) for item in trials], axis=0)
    edge_stack = np.stack([np.asarray(item["edge_weight_seq"], dtype=np.float32) for item in trials], axis=0)
    edge_active_stack = np.stack([np.asarray(item["edge_active_seq"], dtype=np.int8) for item in trials], axis=0)

    node_mean = np.mean(node_stack, axis=0)
    node_std = np.std(node_stack, axis=0)
    edge_mean = np.mean(edge_stack, axis=0)
    edge_std = np.std(edge_stack, axis=0)
    edge_active_rate = np.mean(edge_active_stack, axis=0)

    node_summaries: list[dict[str, Any]] = []
    for node_idx, node_name in enumerate(node_names):
        curve = activation_curve(node_mean, node_idx)
        peak_idx = int(np.argmax(curve)) if curve.size else -1
        peak_value = float(curve[peak_idx]) if peak_idx >= 0 else 0.0
        baseline = baseline_value(curve)
        rise = max(0.0, peak_value - baseline)
        onset_threshold = float(max(EPS, baseline + onset_ratio * rise))
        onset = estimate_onset(curve, window_progress, threshold_value=onset_threshold)
        node_summaries.append(
            {
                "node_index": int(node_idx),
                "node_name": str(node_name),
                "baseline_magnitude": float(baseline),
                "activation_onset_progress": float(onset["onset_progress"]),
                "activation_onset_index": float(onset["onset_index"]),
                "activation_threshold": float(onset_threshold),
                "peak_progress": float(window_progress[peak_idx]) if peak_idx >= 0 else math.nan,
                "peak_index": int(peak_idx),
                "peak_magnitude": float(peak_value),
                "smoothness_score": float(smoothness_score(curve)),
            }
        )

    return {
        "trial_count": int(len(trials)),
        "node_names": node_names,
        "edges": edges,
        "node_feature_names": node_feature_names,
        "window_progress": window_progress,
        "window_progress_start": window_progress_start,
        "window_progress_end": window_progress_end,
        "node_mean": node_mean.astype(np.float32),
        "node_std": node_std.astype(np.float32),
        "edge_mean": edge_mean.astype(np.float32),
        "edge_std": edge_std.astype(np.float32),
        "edge_active_rate": edge_active_rate.astype(np.float32),
        "node_summaries": node_summaries,
        "metadata": {
            "edge_mode": str(first["metadata"]["edge_mode"]),
            "progress_mode": str(first["metadata"]["progress_mode"]),
            "window_size": int(first["metadata"]["window_size"]),
            "window_stride": int(first["metadata"]["window_stride"]),
            "num_windows": int(first["metadata"]["num_windows"]),
            "onset_ratio": float(onset_ratio),
        },
    }


def save_reference_sequence(output_dir: Path, reference: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "reference_sequence.npz"
    json_path = output_dir / "reference_summary.json"
    nodes_csv_path = output_dir / "reference_sequence_nodes.csv"
    edges_csv_path = output_dir / "reference_sequence_edges.csv"
    report_path = output_dir / "reference_report.md"

    np.savez_compressed(
        npz_path,
        window_progress=np.asarray(reference["window_progress"], dtype=np.float32),
        window_progress_start=np.asarray(reference["window_progress_start"], dtype=np.float32),
        window_progress_end=np.asarray(reference["window_progress_end"], dtype=np.float32),
        node_mean=np.asarray(reference["node_mean"], dtype=np.float32),
        node_std=np.asarray(reference["node_std"], dtype=np.float32),
        edge_mean=np.asarray(reference["edge_mean"], dtype=np.float32),
        edge_std=np.asarray(reference["edge_std"], dtype=np.float32),
        edge_active_rate=np.asarray(reference["edge_active_rate"], dtype=np.float32),
    )

    node_rows = []
    for step_idx, progress in enumerate(reference["window_progress"]):
        for node_idx, node_name in enumerate(reference["node_names"]):
            row = {
                "step_index": int(step_idx),
                "progress_percent": float(progress),
                "progress_start_percent": float(reference["window_progress_start"][step_idx]),
                "progress_end_percent": float(reference["window_progress_end"][step_idx]),
                "node_index": int(node_idx),
                "node_name": str(node_name),
            }
            for feature_idx, feature_name in enumerate(reference["node_feature_names"]):
                row[f"{feature_name}_mean"] = float(reference["node_mean"][step_idx, node_idx, feature_idx])
                row[f"{feature_name}_std"] = float(reference["node_std"][step_idx, node_idx, feature_idx])
            node_rows.append(row)
    nodes_df = pd.DataFrame(node_rows)
    nodes_df.to_csv(nodes_csv_path, index=False)

    edge_rows = []
    for step_idx, progress in enumerate(reference["window_progress"]):
        for edge in reference["edges"]:
            edge_idx = int(edge["edge_idx"])
            edge_rows.append(
                {
                    "step_index": int(step_idx),
                    "progress_percent": float(progress),
                    "progress_start_percent": float(reference["window_progress_start"][step_idx]),
                    "progress_end_percent": float(reference["window_progress_end"][step_idx]),
                    "edge_idx": edge_idx,
                    "source_name": str(edge["source_name"]),
                    "target_name": str(edge["target_name"]),
                    "edge_type": str(edge["edge_type"]),
                    "prior_weight": float(edge["prior_weight"]),
                    "edge_weight_mean": float(reference["edge_mean"][step_idx, edge_idx]),
                    "edge_weight_std": float(reference["edge_std"][step_idx, edge_idx]),
                    "edge_active_rate": float(reference["edge_active_rate"][step_idx, edge_idx]),
                }
            )
    edges_df = pd.DataFrame(edge_rows)
    edges_df.to_csv(edges_csv_path, index=False)

    summary_payload = {
        "trial_count": int(reference["trial_count"]),
        "num_trials": int(reference["trial_count"]),
        "sequence_length": int(reference["metadata"]["num_windows"]),
        "node_count": int(len(reference["node_names"])),
        "edge_count": int(len(reference["edges"])),
        "node_feature_names": list(reference["node_feature_names"]),
        "node_names": list(reference["node_names"]),
        "window_progress": [float(item) for item in np.asarray(reference["window_progress"], dtype=np.float32)],
        "window_progress_start": [float(item) for item in np.asarray(reference["window_progress_start"], dtype=np.float32)],
        "window_progress_end": [float(item) for item in np.asarray(reference["window_progress_end"], dtype=np.float32)],
        "node_summaries": list(reference["node_summaries"]),
        "metadata": dict(reference["metadata"]),
        "reference_sequence_npz": str(npz_path),
        "reference_sequence_nodes_csv": str(nodes_csv_path),
        "reference_sequence_edges_csv": str(edges_csv_path),
    }
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    top_nodes_df = (
        pd.DataFrame(reference["node_summaries"])
        .sort_values(["peak_magnitude", "smoothness_score"], ascending=[False, False])
        .head(10)
        .reset_index(drop=True)
    )
    report_lines = [
        "# Reference Graph Sequence",
        "",
        f"- Trial count: `{int(reference['trial_count'])}`",
        f"- Sequence length: `{int(reference['metadata']['num_windows'])}`",
        f"- Node count: `{int(len(reference['node_names']))}`",
        f"- Edge count: `{int(len(reference['edges']))}`",
        f"- Num windows: `{int(reference['metadata']['num_windows'])}`",
        f"- Edge mode: `{reference['metadata']['edge_mode']}`",
        f"- Progress mode: `{reference['metadata']['progress_mode']}`",
        "",
        "## Top Nodes By Peak Magnitude",
        "",
        dataframe_to_markdown_simple(top_nodes_df),
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "npz_path": str(npz_path),
        "summary_json_path": str(json_path),
        "nodes_csv_path": str(nodes_csv_path),
        "edges_csv_path": str(edges_csv_path),
        "report_path": str(report_path),
    }


def load_reference_sequence(reference_dir: Path) -> dict[str, Any]:
    root = Path(reference_dir).resolve()
    summary_path = root / "reference_summary.json"
    npz_path = root / "reference_sequence.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bundle = np.load(npz_path)
    edges: list[dict[str, Any]] = []
    edges_csv = Path(str(summary.get("reference_sequence_edges_csv") or root / "reference_sequence_edges.csv"))
    if edges_csv.exists():
        try:
            edge_df = pd.read_csv(edges_csv)
            if not edge_df.empty and "edge_idx" in edge_df.columns:
                node_index = {str(name): idx for idx, name in enumerate(summary.get("node_names") or [])}
                for edge_idx, group in edge_df.groupby("edge_idx", sort=True):
                    row = group.iloc[0].to_dict()
                    source_name = str(row.get("source_name") or "")
                    target_name = str(row.get("target_name") or "")
                    if source_name not in node_index or target_name not in node_index:
                        continue
                    edges.append(
                        {
                            "edge_idx": int(edge_idx),
                            "source_idx": int(node_index[source_name]),
                            "target_idx": int(node_index[target_name]),
                            "source_name": source_name,
                            "target_name": target_name,
                            "edge_type": str(row.get("edge_type") or "functional_group"),
                            "prior_weight": float(row.get("prior_weight", 1.0)),
                        }
                    )
        except Exception:
            edges = []
    return {
        "summary": summary,
        "edges": edges,
        "window_progress": np.asarray(bundle["window_progress"], dtype=np.float32),
        "window_progress_start": np.asarray(bundle["window_progress_start"], dtype=np.float32) if "window_progress_start" in bundle.files else np.asarray(bundle["window_progress"], dtype=np.float32),
        "window_progress_end": np.asarray(bundle["window_progress_end"], dtype=np.float32) if "window_progress_end" in bundle.files else np.asarray(bundle["window_progress"], dtype=np.float32),
        "node_mean": np.asarray(bundle["node_mean"], dtype=np.float32),
        "node_std": np.asarray(bundle["node_std"], dtype=np.float32),
        "edge_mean": np.asarray(bundle["edge_mean"], dtype=np.float32),
        "edge_std": np.asarray(bundle["edge_std"], dtype=np.float32),
        "edge_active_rate": np.asarray(bundle["edge_active_rate"], dtype=np.float32),
    }


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    source_name = str(edge.get("source_name") or "")
    target_name = str(edge.get("target_name") or "")
    a, b = sorted((source_name, target_name))
    return a, b, str(edge.get("edge_type") or "")


def align_trial_edges_to_reference(trial: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    reference_edges = list(reference.get("edges") or [])
    if not reference_edges:
        return trial
    edge_mean = np.asarray(reference["edge_mean"], dtype=np.float32)
    trial_edge = np.asarray(trial.get("edge_weight_seq"), dtype=np.float32)
    if trial_edge.ndim == 2 and trial_edge.shape == edge_mean.shape:
        return trial

    trial_edges = list(trial.get("edges") or [])
    trial_index = {_edge_key(edge): int(edge.get("edge_idx", idx)) for idx, edge in enumerate(trial_edges)}
    width = int(len(reference_edges))
    aligned = dict(trial)
    aligned_edges: list[dict[str, Any]] = []
    node_index = {str(name): idx for idx, name in enumerate(trial.get("node_names") or [])}
    for new_idx, edge in enumerate(reference_edges):
        source_name = str(edge.get("source_name") or "")
        target_name = str(edge.get("target_name") or "")
        item = dict(edge)
        item["edge_idx"] = int(new_idx)
        item["source_idx"] = int(node_index.get(source_name, edge.get("source_idx", 0)))
        item["target_idx"] = int(node_index.get(target_name, edge.get("target_idx", 0)))
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
        for new_idx, edge in enumerate(reference_edges):
            old_idx = trial_index.get(_edge_key(edge))
            if old_idx is not None and 0 <= old_idx < arr.shape[1]:
                out[:, new_idx] = arr[:, old_idx]
        aligned[key] = out
    return aligned


def _rolling_region(values: np.ndarray, width: int = DEFAULT_TIME_REGION_WIDTH) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"start_index": -1.0, "end_index": -1.0, "score": 0.0}
    width = max(1, min(int(width), int(arr.size)))
    best_score = -math.inf
    best_start = 0
    for start_idx in range(0, int(arr.size) - width + 1):
        score = float(np.mean(arr[start_idx : start_idx + width]))
        if score > best_score:
            best_score = score
            best_start = start_idx
    return {
        "start_index": float(best_start),
        "end_index": float(best_start + width - 1),
        "score": float(best_score),
    }


def rank_time_regions(
    values: np.ndarray,
    progress: np.ndarray,
    progress_start: np.ndarray | None = None,
    progress_end: np.ndarray | None = None,
    width: int = DEFAULT_TIME_REGION_WIDTH,
    top_k: int = 5,
) -> pd.DataFrame:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    prog = np.asarray(progress, dtype=np.float32).reshape(-1)
    prog_start = prog if progress_start is None else np.asarray(progress_start, dtype=np.float32).reshape(-1)
    prog_end = prog if progress_end is None else np.asarray(progress_end, dtype=np.float32).reshape(-1)
    if arr.size == 0 or prog.size == 0:
        return pd.DataFrame(
            columns=[
                "rank",
                "start_index",
                "end_index",
                "start_progress",
                "end_progress",
                "mean_score",
            ]
        )
    width = max(1, min(int(width), int(arr.size)))
    rows = []
    for start_idx in range(0, int(arr.size) - width + 1):
        end_idx = int(start_idx + width - 1)
        rows.append(
            {
                "start_index": int(start_idx),
                "end_index": int(end_idx),
                "start_progress": float(prog_start[start_idx]),
                "end_progress": float(prog_end[end_idx]),
                "mean_score": float(np.mean(arr[start_idx : end_idx + 1])),
            }
        )
    ranked = pd.DataFrame(rows).sort_values("mean_score", ascending=False).head(max(1, int(top_k))).reset_index(drop=True)
    if ranked.empty:
        return ranked
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked


def compare_trial_to_reference(
    trial: dict[str, Any],
    reference: dict[str, Any],
    *,
    onset_ratio: float = DEFAULT_ONSET_RATIO,
    time_region_width: int = DEFAULT_TIME_REGION_WIDTH,
) -> dict[str, Any]:
    node_mean = np.asarray(reference["node_mean"], dtype=np.float32)
    node_std = np.asarray(reference["node_std"], dtype=np.float32)
    edge_mean = np.asarray(reference["edge_mean"], dtype=np.float32)
    edge_std = np.asarray(reference["edge_std"], dtype=np.float32)
    progress = np.asarray(reference["window_progress"], dtype=np.float32)
    progress_start = np.asarray(reference.get("window_progress_start", reference["window_progress"]), dtype=np.float32)
    progress_end = np.asarray(reference.get("window_progress_end", reference["window_progress"]), dtype=np.float32)
    trial = align_trial_edges_to_reference(trial, reference)
    trial_node = np.asarray(trial["node_feature_seq"], dtype=np.float32)
    trial_edge = np.asarray(trial["edge_weight_seq"], dtype=np.float32)

    if list(trial["node_names"]) != list(reference["summary"]["node_names"]):
        raise ValueError("Trial node_names do not match reference node_names.")
    if trial_node.shape != node_mean.shape:
        raise ValueError(f"Trial node_feature_seq shape {trial_node.shape} does not match reference {node_mean.shape}.")
    if trial_edge.shape != edge_mean.shape:
        raise ValueError(f"Trial edge_weight_seq shape {trial_edge.shape} does not match reference {edge_mean.shape}.")

    node_scale = stable_deviation_scale(node_mean, node_std, minimum_scale=0.01)
    edge_scale = stable_deviation_scale(edge_mean, edge_std, minimum_scale=0.02)
    node_z = (trial_node - node_mean) / node_scale
    node_abs_z = np.abs(node_z)
    node_agg = np.mean(node_abs_z, axis=2)
    edge_z = (trial_edge - edge_mean) / edge_scale
    edge_abs_z = np.abs(edge_z)

    node_rows = []
    feature_names = list(reference["summary"]["node_feature_names"])
    for step_idx, progress_value in enumerate(progress.tolist()):
        for node_idx, node_name in enumerate(reference["summary"]["node_names"]):
            row = {
                "step_index": int(step_idx),
                "progress_percent": float(progress_value),
                "progress_start_percent": float(progress_start[step_idx]),
                "progress_end_percent": float(progress_end[step_idx]),
                "node_index": int(node_idx),
                "node_name": str(node_name),
                "aggregate_deviation_score": float(node_agg[step_idx, node_idx]),
            }
            for feature_idx, feature_name in enumerate(feature_names):
                row[f"{feature_name}_trial"] = float(trial_node[step_idx, node_idx, feature_idx])
                row[f"{feature_name}_reference_mean"] = float(node_mean[step_idx, node_idx, feature_idx])
                row[f"{feature_name}_reference_std"] = float(node_std[step_idx, node_idx, feature_idx])
                row[f"{feature_name}_zscore"] = float(node_z[step_idx, node_idx, feature_idx])
            node_rows.append(row)
    node_df = pd.DataFrame(node_rows)

    edge_rows = []
    for step_idx, progress_value in enumerate(progress.tolist()):
        for edge in trial["edges"]:
            edge_idx = int(edge["edge_idx"])
            edge_rows.append(
                {
                    "step_index": int(step_idx),
                    "progress_percent": float(progress_value),
                    "progress_start_percent": float(progress_start[step_idx]),
                    "progress_end_percent": float(progress_end[step_idx]),
                    "edge_idx": edge_idx,
                    "source_name": str(edge["source_name"]),
                    "target_name": str(edge["target_name"]),
                    "edge_type": str(edge["edge_type"]),
                    "prior_weight": float(edge["prior_weight"]),
                    "edge_weight_trial": float(trial_edge[step_idx, edge_idx]),
                    "edge_weight_reference_mean": float(edge_mean[step_idx, edge_idx]),
                    "edge_weight_reference_std": float(edge_std[step_idx, edge_idx]),
                    "edge_zscore": float(edge_z[step_idx, edge_idx]),
                    "edge_abs_zscore": float(edge_abs_z[step_idx, edge_idx]),
                }
            )
    edge_df = pd.DataFrame(edge_rows)

    node_summary_rows = []
    ref_node_summaries = {
        str(item["node_name"]): dict(item) for item in reference["summary"]["node_summaries"]
    }
    for node_idx, node_name in enumerate(reference["summary"]["node_names"]):
        node_name = str(node_name)
        ref_summary = ref_node_summaries[node_name]
        trial_curve = activation_curve(trial_node, node_idx)
        trial_peak_idx = int(np.argmax(trial_curve)) if trial_curve.size else -1
        trial_peak_value = float(trial_curve[trial_peak_idx]) if trial_peak_idx >= 0 else 0.0
        ref_baseline = float(ref_summary.get("baseline_magnitude", 0.0))
        ref_peak = float(ref_summary["peak_magnitude"])
        onset_threshold = float(max(EPS, ref_baseline + onset_ratio * max(0.0, ref_peak - ref_baseline)))
        trial_onset = estimate_onset(trial_curve, progress, threshold_value=onset_threshold)
        trial_smoothness = float(smoothness_score(trial_curve))
        ref_smoothness = float(ref_summary["smoothness_score"])
        peak_magnitude_deviation = float(trial_peak_value - float(ref_summary["peak_magnitude"]))
        peak_timing_deviation = float(progress[trial_peak_idx] - float(ref_summary["peak_progress"])) if trial_peak_idx >= 0 else math.nan
        onset_delay = float(trial_onset["onset_progress"] - float(ref_summary["activation_onset_progress"])) if np.isfinite(float(trial_onset["onset_progress"])) and np.isfinite(float(ref_summary["activation_onset_progress"])) else math.nan
        smoothness_deviation = float(trial_smoothness - ref_smoothness)
        average_score = float(np.mean(node_agg[:, node_idx]))
        trial_onset_detected = bool(np.isfinite(float(trial_onset["onset_progress"])))
        reference_onset_detected = bool(np.isfinite(float(ref_summary["activation_onset_progress"])))
        if not trial_onset_detected:
            onset_status = "missing"
        elif not reference_onset_detected:
            onset_status = "reference_missing"
        elif np.isfinite(onset_delay) and onset_delay > (100.0 / max(1, len(progress))):
            onset_status = "late"
        elif np.isfinite(onset_delay) and onset_delay < -(100.0 / max(1, len(progress))):
            onset_status = "early"
        else:
            onset_status = "on_time"
        magnitude_tolerance = max(0.05, 0.10 * max(EPS, float(ref_summary["peak_magnitude"])))
        if peak_magnitude_deviation < -magnitude_tolerance:
            magnitude_status = "too_weak"
        elif peak_magnitude_deviation > magnitude_tolerance:
            magnitude_status = "too_high"
        else:
            magnitude_status = "within_range"
        smoothness_tolerance = max(0.01, 0.15 * max(EPS, ref_smoothness))
        smoothness_status = "not_smooth" if smoothness_deviation > smoothness_tolerance else "within_range"
        node_summary_rows.append(
            {
                "node_index": int(node_idx),
                "node_name": node_name,
                "activation_threshold_reference": onset_threshold,
                "onset_progress_trial": float(trial_onset["onset_progress"]),
                "onset_progress_reference": float(ref_summary["activation_onset_progress"]),
                "onset_delay_progress": onset_delay,
                "peak_progress_trial": float(progress[trial_peak_idx]) if trial_peak_idx >= 0 else math.nan,
                "peak_progress_reference": float(ref_summary["peak_progress"]),
                "peak_timing_deviation_progress": peak_timing_deviation,
                "peak_magnitude_trial": float(trial_peak_value),
                "peak_magnitude_reference": float(ref_summary["peak_magnitude"]),
                "peak_magnitude_deviation": peak_magnitude_deviation,
                "smoothness_trial": trial_smoothness,
                "smoothness_reference": ref_smoothness,
                "smoothness_deviation": smoothness_deviation,
                "average_deviation_score": average_score,
                "onset_status": onset_status,
                "magnitude_status": magnitude_status,
                "smoothness_status": smoothness_status,
            }
        )
    node_summary_df = pd.DataFrame(node_summary_rows)

    per_step_node_score = np.mean(node_agg, axis=1)
    per_step_edge_score = np.mean(edge_abs_z, axis=1) if edge_abs_z.size else np.zeros_like(per_step_node_score)
    combined_step_score = 0.5 * per_step_node_score + 0.5 * per_step_edge_score
    worst_region = _rolling_region(combined_step_score, width=time_region_width)
    top_time_regions_df = rank_time_regions(
        combined_step_score,
        progress,
        progress_start=progress_start,
        progress_end=progress_end,
        width=time_region_width,
        top_k=5,
    )
    start_idx = int(worst_region["start_index"]) if worst_region["start_index"] >= 0 else -1
    end_idx = int(worst_region["end_index"]) if worst_region["end_index"] >= 0 else -1

    edge_group = edge_df.groupby(["edge_idx", "source_name", "target_name", "edge_type"], as_index=False)["edge_abs_zscore"].mean()
    worst_edges_df = edge_group.sort_values("edge_abs_zscore", ascending=False).head(10).reset_index(drop=True)
    worst_nodes_df = node_summary_df.sort_values("average_deviation_score", ascending=False).head(10).reset_index(drop=True)

    summary_payload = {
        "trial_id": str(trial["metadata"]["trial_id"]),
        "exercise_name": str(trial["metadata"].get("exercise_name") or ""),
        "source_csv": str(trial["metadata"]["source_csv"]),
        "overall_node_deviation_score": float(np.mean(node_agg)),
        "overall_edge_deviation_score": float(np.mean(edge_abs_z)) if edge_abs_z.size else 0.0,
        "worst_time_region": {
            "start_progress": float(progress_start[start_idx]) if start_idx >= 0 else math.nan,
            "end_progress": float(progress_end[end_idx]) if end_idx >= 0 else math.nan,
            "score": float(worst_region["score"]),
        },
        "top_time_regions": top_time_regions_df.to_dict(orient="records"),
        "worst_nodes": worst_nodes_df.to_dict(orient="records"),
        "worst_edges": worst_edges_df.to_dict(orient="records"),
        "node_summary": node_summary_df.to_dict(orient="records"),
    }

    return {
        "node_deviation_df": node_df,
        "edge_deviation_df": edge_df,
        "node_summary_df": node_summary_df,
        "worst_nodes_df": worst_nodes_df,
        "worst_edges_df": worst_edges_df,
        "top_time_regions_df": top_time_regions_df,
        "summary": summary_payload,
    }


def save_trial_reference_comparison(output_dir: Path, comparison: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    node_csv = output_dir / "trial_vs_reference_node_deviation.csv"
    edge_csv = output_dir / "trial_vs_reference_edge_deviation.csv"
    summary_json = output_dir / "trial_vs_reference_summary.json"
    report_md = output_dir / "trial_vs_reference_report.md"

    comparison["node_deviation_df"].to_csv(node_csv, index=False)
    comparison["edge_deviation_df"].to_csv(edge_csv, index=False)
    summary_json.write_text(json.dumps(comparison["summary"], indent=2), encoding="utf-8")

    top_time_regions_df = comparison.get("top_time_regions_df")
    if top_time_regions_df is None:
        top_time_regions_df = pd.DataFrame(comparison["summary"].get("top_time_regions") or [])

    report_lines = [
        "# Trial vs Reference Sequence",
        "",
        f"- Trial ID: `{comparison['summary']['trial_id']}`",
        f"- Exercise: `{comparison['summary']['exercise_name']}`",
        f"- Overall node deviation: `{comparison['summary']['overall_node_deviation_score']:.4f}`",
        f"- Overall edge deviation: `{comparison['summary']['overall_edge_deviation_score']:.4f}`",
        f"- Worst time region: `{comparison['summary']['worst_time_region']['start_progress']:.2f}% -> {comparison['summary']['worst_time_region']['end_progress']:.2f}%`",
        "",
        "## Worst Nodes",
        "",
        dataframe_to_markdown_simple(comparison["worst_nodes_df"]),
        "",
        "## Top Time Regions",
        "",
        dataframe_to_markdown_simple(top_time_regions_df),
        "",
        "## Worst Edges",
        "",
        dataframe_to_markdown_simple(comparison["worst_edges_df"]),
        "",
    ]
    report_md.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "node_csv_path": str(node_csv),
        "edge_csv_path": str(edge_csv),
        "summary_json_path": str(summary_json),
        "report_path": str(report_md),
    }
