"""
compare_trial_to_reference.py
==============================
Bandingkan 1 trial baru ke reference sequence.
Output: NodeDeviationResult per node -> label untuk GCN.

Deviasi yang dideteksi:
  - onset_shift : otot aktif terlalu awal / terlambat
  - too_weak    : puncak aktivasi di bawah referensi
  - too_high    : puncak aktivasi di atas referensi
  - not_smooth  : pola aktivasi tidak mulus
  - within_reference : normal
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Label constants (compatible with gcn_node_classifier.py)
# ---------------------------------------------------------------------------
LABEL_WITHIN_REFERENCE = "within_reference"
LABEL_ONSET_SHIFT      = "onset_shift"
LABEL_TOO_WEAK         = "too_weak"
LABEL_TOO_HIGH         = "too_high"
LABEL_NOT_SMOOTH       = "not_smooth"

LABEL_TO_INT = {
    LABEL_WITHIN_REFERENCE: 0,
    LABEL_ONSET_SHIFT:      1,
    LABEL_TOO_WEAK:         2,
    LABEL_TOO_HIGH:         3,
    LABEL_NOT_SMOOTH:       4,
}

_EPS = 1e-8
_DEFAULT_MAGNITUDE_Z = 2.0
_DEFAULT_SMOOTHNESS_Z = 2.0
_DEFAULT_ONSET_RATIO = 0.20
_DEFAULT_TIME_REGION_WIDTH = 3


# ---------------------------------------------------------------------------
# Dataclass hasil deviasi per node
# ---------------------------------------------------------------------------
@dataclass
class NodeDeviationResult:
    node_name: str
    node_index: int

    # Onset
    onset_status: str = "on_time"         # "on_time" | "early" | "late" | "missing"
    onset_delay_frames: float = 0.0       # positif = terlambat, negatif = terlalu awal
    onset_delay_progress: float = 0.0    # dalam satuan progress (0–100)

    # Magnitude
    magnitude_status: str = "within_range"  # "within_range" | "too_weak" | "too_high"
    magnitude_ratio: float = 1.0            # trial_peak / reference_peak
    magnitude_z_score: float = 0.0         # z-score terhadap referensi std

    # Smoothness
    smoothness_status: str = "within_range" # "within_range" | "not_smooth"
    smoothness_delta: float = 0.0           # trial_smoothness - reference_smoothness
    smoothness_z_score: float = 0.0

    # Label utama (prioritas: onset > magnitude > smooth > within)
    primary_label: str = LABEL_WITHIN_REFERENCE
    primary_label_int: int = 0

    # Metadata
    ref_peak_magnitude: float = 0.0
    ref_onset_progress: float = float("nan")
    trial_peak_magnitude: float = 0.0
    trial_onset_progress: float = float("nan")
    ref_smoothness: float = 0.0
    trial_smoothness: float = 0.0


# ---------------------------------------------------------------------------
# Helper: aktivasi, onset, smoothness
# ---------------------------------------------------------------------------

def _activation_curve(node_feature_seq: np.ndarray, node_idx: int, feature_idx: int = 0) -> np.ndarray:
    """Ambil kurva aktivasi (fitur ke-0 = mean_ctrl) untuk node_idx."""
    return np.asarray(node_feature_seq[:, node_idx, feature_idx], dtype=np.float32)


def _estimate_onset(curve: np.ndarray, progress: np.ndarray, *, threshold: float) -> float:
    """Kembalikan progress (0–100) saat aktivasi pertama kali melebihi threshold. NaN jika tidak ditemukan."""
    hits = np.where(np.asarray(curve, dtype=np.float32) >= float(threshold))[0]
    if hits.size == 0:
        return float("nan")
    return float(progress[hits[0]])


def _smoothness_score(curve: np.ndarray) -> float:
    """Mean absolute second difference (semakin kecil = semakin halus)."""
    vals = np.asarray(curve, dtype=np.float32).reshape(-1)
    if vals.size <= 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(vals, n=2))))


def _stable_scale(std: float, mean: float, min_scale: float = 0.01) -> float:
    return max(abs(std), 0.1 * max(abs(mean), min_scale), min_scale)


# ---------------------------------------------------------------------------
# Perbandingan per node
# ---------------------------------------------------------------------------

def compare_node(
    node_idx: int,
    node_name: str,
    ref_node_summary: dict[str, Any],
    trial_node_feature_seq: np.ndarray,
    trial_window_progress: np.ndarray,
    ref_node_feature_seq_mean: np.ndarray,
    ref_node_feature_seq_std: np.ndarray,
    ref_window_progress: np.ndarray,
    *,
    magnitude_z_threshold: float = _DEFAULT_MAGNITUDE_Z,
    smoothness_z_threshold: float = _DEFAULT_SMOOTHNESS_Z,
    onset_ratio: float = _DEFAULT_ONSET_RATIO,
) -> NodeDeviationResult:
    """
    Bandingkan 1 node dari trial vs referensi.

    Parameters
    ----------
    ref_node_summary : elemen dari reference["node_summaries"][node_idx]
    trial_node_feature_seq : (T_trial, num_nodes, num_features)
    ref_node_feature_seq_mean : (T_ref, num_nodes, num_features) — node_mean dari reference
    """
    result = NodeDeviationResult(node_name=node_name, node_index=node_idx)

    # Kurva aktivasi (fitur 0 = mean_ctrl)
    trial_curve = _activation_curve(trial_node_feature_seq, node_idx)
    ref_curve_mean = _activation_curve(ref_node_feature_seq_mean, node_idx)
    ref_curve_std  = _activation_curve(ref_node_feature_seq_std, node_idx)
    trial_prog = np.asarray(trial_window_progress, dtype=np.float32)
    ref_prog   = np.asarray(ref_window_progress, dtype=np.float32)

    # --- Magnitude ---
    trial_peak = float(np.max(trial_curve)) if trial_curve.size else 0.0
    ref_peak = float(ref_node_summary.get("peak_magnitude", float(np.max(ref_curve_mean)) if ref_curve_mean.size else 0.0))
    ref_peak_std_at_peak = float(ref_curve_std[int(np.argmax(ref_curve_mean))] if ref_curve_mean.size else 0.0)
    mag_scale = _stable_scale(ref_peak_std_at_peak, ref_peak)
    mag_z = (trial_peak - ref_peak) / mag_scale if mag_scale > 0 else 0.0

    result.ref_peak_magnitude = ref_peak
    result.trial_peak_magnitude = trial_peak
    result.magnitude_ratio = trial_peak / max(ref_peak, _EPS)
    result.magnitude_z_score = float(mag_z)

    if mag_z < -magnitude_z_threshold:
        result.magnitude_status = LABEL_TOO_WEAK
    elif mag_z > magnitude_z_threshold:
        result.magnitude_status = LABEL_TOO_HIGH
    else:
        result.magnitude_status = "within_range"

    # --- Onset ---
    ref_baseline_val = float(ref_node_summary.get("baseline_magnitude", 0.0))
    ref_rise = max(0.0, ref_peak - ref_baseline_val)
    onset_threshold = float(max(_EPS, ref_baseline_val + onset_ratio * ref_rise))
    ref_onset_prog = float(ref_node_summary.get("activation_onset_progress", float("nan")))
    trial_onset_prog = _estimate_onset(trial_curve, trial_prog, threshold=onset_threshold)

    result.ref_onset_progress = ref_onset_prog
    result.trial_onset_progress = trial_onset_prog

    if math.isnan(trial_onset_prog) and not math.isnan(ref_onset_prog):
        result.onset_status = "missing"
        result.onset_delay_progress = float("nan")
        result.onset_delay_frames = float("nan")
    elif not math.isnan(trial_onset_prog) and not math.isnan(ref_onset_prog):
        delay = trial_onset_prog - ref_onset_prog
        result.onset_delay_progress = float(delay)
        # Konversi ke frames (approx)
        n_windows = int(trial_node_feature_seq.shape[0])
        result.onset_delay_frames = delay / 100.0 * n_windows
        # Onset shift jika delay melebihi 1 region (default 3 frames → ~6% progress)
        shift_progress_threshold = 100.0 / max(1, n_windows) * _DEFAULT_TIME_REGION_WIDTH
        if abs(delay) > shift_progress_threshold:
            result.onset_status = "early" if delay < 0 else "late"
        else:
            result.onset_status = "on_time"
    else:
        result.onset_status = "on_time"
        result.onset_delay_progress = 0.0
        result.onset_delay_frames = 0.0

    # --- Smoothness ---
    trial_smooth = _smoothness_score(trial_curve)
    ref_smooth_vals = [_smoothness_score(_activation_curve(ref_node_feature_seq_mean + k * ref_node_feature_seq_std, node_idx)) for k in (-1.0, 0.0, 1.0)]
    ref_smooth_mean = float(ref_smooth_vals[1])
    ref_smooth_std = float(max(abs(ref_smooth_vals[2] - ref_smooth_vals[0]) / 2.0, 0.001))
    smooth_z = (trial_smooth - ref_smooth_mean) / ref_smooth_std if ref_smooth_std > 0 else 0.0

    result.trial_smoothness = float(trial_smooth)
    result.ref_smoothness = float(ref_smooth_mean)
    result.smoothness_delta = float(trial_smooth - ref_smooth_mean)
    result.smoothness_z_score = float(smooth_z)
    result.smoothness_status = LABEL_NOT_SMOOTH if smooth_z > smoothness_z_threshold else "within_range"

    # --- Primary label (prioritas: missing onset > onset_shift > too_weak > too_high > not_smooth > within) ---
    if result.onset_status == "missing":
        result.primary_label = LABEL_ONSET_SHIFT
    elif result.onset_status in ("early", "late"):
        result.primary_label = LABEL_ONSET_SHIFT
    elif result.magnitude_status == LABEL_TOO_WEAK:
        result.primary_label = LABEL_TOO_WEAK
    elif result.magnitude_status == LABEL_TOO_HIGH:
        result.primary_label = LABEL_TOO_HIGH
    elif result.smoothness_status == LABEL_NOT_SMOOTH:
        result.primary_label = LABEL_NOT_SMOOTH
    else:
        result.primary_label = LABEL_WITHIN_REFERENCE

    result.primary_label_int = LABEL_TO_INT.get(result.primary_label, 0)
    return result


# ---------------------------------------------------------------------------
# Bandingkan 1 trial vs reference
# ---------------------------------------------------------------------------

def compare_trial_to_reference(
    trial_graph_seq: dict[str, Any],
    reference: dict[str, Any],
    *,
    magnitude_z_threshold: float = _DEFAULT_MAGNITUDE_Z,
    smoothness_z_threshold: float = _DEFAULT_SMOOTHNESS_Z,
    onset_ratio: float = _DEFAULT_ONSET_RATIO,
) -> dict[str, Any]:
    """
    Bandingkan 1 trial graph sequence ke reference.

    Parameters
    ----------
    trial_graph_seq : output dari build_open_to_fist_sequence()
    reference       : output dari build_reference_from_trials()

    Returns
    -------
    {
        'node_deviation_results' : list[NodeDeviationResult],
        'node_label_vector'      : np.ndarray shape (num_nodes,) int  -> for GCN
        'trial_score'            : float  (0 = perfect, tinggi = banyak deviasi),
        'node_summary_rows'      : list[dict]  ← untuk DataFrame/CSV,
        'edge_deviation'         : dict,
        'metadata'               : dict,
    }
    """
    node_names: list[str] = list(trial_graph_seq.get("node_names", reference.get("node_names", [])))
    num_nodes = len(node_names)

    trial_nfs = np.asarray(trial_graph_seq["node_feature_seq"], dtype=np.float32)   # (T, N, F)
    trial_prog = np.asarray(trial_graph_seq["window_progress"], dtype=np.float32)

    ref_node_mean = np.asarray(reference["node_mean"], dtype=np.float32)             # (T, N, F)
    ref_node_std  = np.asarray(reference["node_std"],  dtype=np.float32)
    ref_prog = np.asarray(reference["window_progress"], dtype=np.float32)
    ref_node_summaries: list[dict[str, Any]] = list(reference.get("node_summaries", [{}] * num_nodes))

    # Resample trial ke panjang referensi jika berbeda
    T_ref = ref_node_mean.shape[0]
    T_trial = trial_nfs.shape[0]
    if T_trial != T_ref:
        from scipy.interpolate import interp1d
        src = np.linspace(0.0, 1.0, T_trial)
        dst = np.linspace(0.0, 1.0, T_ref)
        trial_nfs_r = np.zeros((T_ref, trial_nfs.shape[1], trial_nfs.shape[2]), dtype=np.float32)
        for ni in range(trial_nfs.shape[1]):
            for fi in range(trial_nfs.shape[2]):
                trial_nfs_r[:, ni, fi] = interp1d(src, trial_nfs[:, ni, fi], kind="linear")(dst)
        trial_nfs = trial_nfs_r
        trial_prog = np.linspace(float(trial_prog[0]), float(trial_prog[-1]), T_ref, dtype=np.float32)

    deviation_results: list[NodeDeviationResult] = []
    for ni, node_name in enumerate(node_names):
        summary = ref_node_summaries[ni] if ni < len(ref_node_summaries) else {}
        dr = compare_node(
            node_idx=ni,
            node_name=node_name,
            ref_node_summary=summary,
            trial_node_feature_seq=trial_nfs,
            trial_window_progress=trial_prog,
            ref_node_feature_seq_mean=ref_node_mean,
            ref_node_feature_seq_std=ref_node_std,
            ref_window_progress=ref_prog,
            magnitude_z_threshold=magnitude_z_threshold,
            smoothness_z_threshold=smoothness_z_threshold,
            onset_ratio=onset_ratio,
        )
        deviation_results.append(dr)

    # Node label vector for GCN
    node_label_vector = np.array([dr.primary_label_int for dr in deviation_results], dtype=np.int64)

    # Trial score: proporsi node yang abnormal
    n_abnormal = sum(1 for dr in deviation_results if dr.primary_label != LABEL_WITHIN_REFERENCE)
    trial_score = float(n_abnormal) / max(1, num_nodes)

    # Summary rows
    node_summary_rows = [
        {
            "node_index": dr.node_index,
            "node_name": dr.node_name,
            "primary_label": dr.primary_label,
            "primary_label_int": dr.primary_label_int,
            "onset_status": dr.onset_status,
            "onset_delay_progress": dr.onset_delay_progress,
            "magnitude_status": dr.magnitude_status,
            "magnitude_ratio": dr.magnitude_ratio,
            "magnitude_z_score": dr.magnitude_z_score,
            "smoothness_status": dr.smoothness_status,
            "smoothness_z_score": dr.smoothness_z_score,
            "ref_peak_magnitude": dr.ref_peak_magnitude,
            "trial_peak_magnitude": dr.trial_peak_magnitude,
        }
        for dr in deviation_results
    ]

    # Edge deviation (sederhana: cek edge_weight_seq vs referensi)
    edge_deviation: dict[str, Any] = {}
    if "edge_weight_seq" in trial_graph_seq and "edge_mean" in reference:
        trial_ew = np.asarray(trial_graph_seq["edge_weight_seq"], dtype=np.float32)
        ref_ew_mean = np.asarray(reference["edge_mean"], dtype=np.float32)
        ref_ew_std  = np.asarray(reference["edge_std"],  dtype=np.float32)
        if trial_ew.shape == ref_ew_mean.shape:
            edge_z = (trial_ew - ref_ew_mean) / np.maximum(ref_ew_std, 1e-4)
            edge_deviation = {
                "mean_abs_z": float(np.mean(np.abs(edge_z))),
                "max_abs_z": float(np.max(np.abs(edge_z))),
                "edge_z_matrix": edge_z,
            }

    return {
        "node_deviation_results": deviation_results,
        "node_label_vector": node_label_vector,
        "trial_score": trial_score,
        "node_summary_rows": node_summary_rows,
        "edge_deviation": edge_deviation,
        "metadata": {
            "trial_id": str(trial_graph_seq.get("metadata", {}).get("trial_id", "")),
            "exercise_name": str(trial_graph_seq.get("metadata", {}).get("exercise_name", "")),
            "num_nodes": num_nodes,
            "num_abnormal": n_abnormal,
            "magnitude_z_threshold": magnitude_z_threshold,
            "smoothness_z_threshold": smoothness_z_threshold,
            "onset_ratio": onset_ratio,
        },
    }


# ---------------------------------------------------------------------------
# Laporan teks
# ---------------------------------------------------------------------------

def generate_deviation_report(result: dict[str, Any], *, output_path: Path | None = None) -> str:
    """Buat laporan teks singkat dari hasil compare_trial_to_reference()."""
    meta = result.get("metadata", {})
    rows = result.get("node_summary_rows", [])
    lines = [
        f"# Deviation Report: {meta.get('trial_id', '?')} | {meta.get('exercise_name', '')}",
        "",
        f"- Nodes: {meta.get('num_nodes', '?')}",
        f"- Abnormal nodes: {meta.get('num_abnormal', '?')}",
        f"- Trial score: {float(result.get('trial_score', 0.0)):.3f}",
        "",
        "## Node Summary",
        "| # | Node | Label | Onset | Magnitude | Smooth |",
        "|---|------|-------|-------|-----------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['node_index']} | {r['node_name']} | **{r['primary_label']}** "
            f"| {r['onset_status']} (Δ{r['onset_delay_progress']:.1f}%) "
            f"| {r['magnitude_status']} (z={r['magnitude_z_score']:.2f}) "
            f"| {r['smoothness_status']} (z={r['smoothness_z_score']:.2f}) |"
        )

    text = "\n".join(lines)
    if output_path is not None:
        Path(output_path).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Build node label sequence for GCN (public API)
# ---------------------------------------------------------------------------

def build_node_label_sequence(deviation_result: dict[str, Any], node_names: list[str]) -> np.ndarray:
    """
    Map NodeDeviationResult -> integer class labels (nu,) for GCN input.

    Classes: within_reference=0, onset_shift=1, too_weak=2, too_high=3, not_smooth=4
    """
    label_map = {dr.node_name: dr.primary_label_int for dr in deviation_result["node_deviation_results"]}
    return np.array([label_map.get(name, 0) for name in node_names], dtype=np.int64)
