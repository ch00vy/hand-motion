"""
build_reference_sequence.py
============================
Bangun reference graph sequence dari biomech reference trials.
Wrap graph_sequence_baseline.build_reference_sequence() dengan ctrl_ref
dari inverse dynamics (bukan ctrl_analysis CSV langsung).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Import dari graph_sequence_baseline (sudah ada, keep as-is)
# ---------------------------------------------------------------------------
try:
    from app.graph_sequence.graph_sequence_baseline import (
        build_trial_graph_sequence,
        build_reference_sequence as _build_reference_sequence_base,
        save_reference_sequence,
        load_reference_sequence,
        resolve_graph_template,
        NODE_FEATURE_NAMES,
    )
    _GRAPH_SEQ_AVAILABLE = True
except ImportError:
    try:
        from app.graph_sequence_baseline import (  # type: ignore[no-redef]
            build_trial_graph_sequence,
            build_reference_sequence as _build_reference_sequence_base,
            save_reference_sequence,
            load_reference_sequence,
            resolve_graph_template,
            NODE_FEATURE_NAMES,
        )
        _GRAPH_SEQ_AVAILABLE = True
    except ImportError:
        _GRAPH_SEQ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEQUENCE_LENGTH = 50
DEFAULT_WINDOW_SIZE = 5
DEFAULT_WINDOW_STRIDE = 1
DEFAULT_ONSET_RATIO = 0.20
DEFAULT_EDGE_MODE = "dynamic_short_window"
DEFAULT_EDGE_THRESHOLD = 0.05
DEFAULT_INCLUDE_EDGE_SIGN = True
DEFAULT_COHORT = "reference"


# ---------------------------------------------------------------------------
# Konversi biomech trial → graph sequence format
# ---------------------------------------------------------------------------

def biomech_trial_to_graph_input(
    biomech_trial: dict[str, Any],
    *,
    use_ctrl_ref: bool = True,
) -> dict[str, Any]:
    """
    Konversi output build_biomech_reference_ctrl() ke format yang kompatibel
    dengan build_trial_graph_sequence().

    Parameters
    ----------
    biomech_trial : dict dari offline_biomech_reference.build_biomech_reference_ctrl()
                   Harus punya kunci: ctrl_ref, qpos, timestamps, trial_id, exercise_name
    use_ctrl_ref  : jika True pakai ctrl_ref (dari inverse dynamics),
                   jika False pakai kolom 'ctrl_analysis' jika tersedia
    """
    ctrl_key = "ctrl_ref" if use_ctrl_ref else "ctrl_analysis"
    if ctrl_key not in biomech_trial and use_ctrl_ref:
        ctrl_key = "ctrl_analysis"
    if ctrl_key not in biomech_trial:
        raise KeyError(f"Kunci '{ctrl_key}' tidak ditemukan di biomech_trial.")

    ctrl = np.asarray(biomech_trial[ctrl_key], dtype=np.float32)
    qpos = np.asarray(biomech_trial.get("qpos", np.zeros((ctrl.shape[0], 0), dtype=np.float32)), dtype=np.float32)
    timestamps = np.asarray(biomech_trial.get("timestamps", np.arange(ctrl.shape[0], dtype=np.float64)), dtype=np.float64)
    num_frames = int(ctrl.shape[0])

    return {
        "source_csv": str(biomech_trial.get("source_csv", "")),
        "trial_id": str(biomech_trial.get("trial_id", "unknown")),
        "exercise_name": str(biomech_trial.get("exercise_name", "")),
        "subject_id": str(biomech_trial.get("subject_id", "")),
        "num_frames": num_frames,
        "ctrl": ctrl,
        "qpos": qpos,
        "timestamps": timestamps,
        # Metadata segmentasi default (full sequence)
        "segment_start_frame": 0,
        "segment_end_frame": max(0, num_frames - 1),
        "segment_num_frames": num_frames,
        "transition_segmentation": "none",
        "motion_threshold": 0.0,
    }


# ---------------------------------------------------------------------------
# Build open → fist sequence dari 1 trial
# ---------------------------------------------------------------------------

def build_open_to_fist_sequence(
    biomech_trial: dict[str, Any],
    graph_template: dict[str, Any],
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    edge_mode: str = DEFAULT_EDGE_MODE,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    include_edge_sign: bool = DEFAULT_INCLUDE_EDGE_SIGN,
    cohort: str = DEFAULT_COHORT,
    use_ctrl_ref: bool = True,
) -> dict[str, Any]:
    """
    Ubah 1 trial open→fist menjadi sequence of graph snapshots.

    Parameters
    ----------
    biomech_trial  : output dari build_biomech_reference_ctrl()
    graph_template : output dari resolve_graph_template()

    Returns
    -------
    dict kompatibel dengan save_trial_graph_sequence() dan build_reference_from_trials()
    """
    if not _GRAPH_SEQ_AVAILABLE:
        raise ImportError("graph_sequence_baseline tidak tersedia. Pastikan app/graph_sequence/ ada.")

    trial_input = biomech_trial_to_graph_input(biomech_trial, use_ctrl_ref=use_ctrl_ref)
    return build_trial_graph_sequence(
        trial_input,
        graph_template=graph_template,
        sequence_length=sequence_length,
        window_size=window_size,
        window_stride=window_stride,
        progress_mode="linear",
        edge_mode=edge_mode,
        edge_threshold=edge_threshold,
        include_edge_sign=include_edge_sign,
        cohort=cohort,
    )


# ---------------------------------------------------------------------------
# Build reference dari multiple trials
# ---------------------------------------------------------------------------

def build_reference_from_trials(
    graph_sequences: list[dict[str, Any]],
    *,
    onset_ratio: float = DEFAULT_ONSET_RATIO,
) -> dict[str, Any]:
    """
    Aggregate multiple graph sequences → reference sequence (mean ± std per node/edge).

    Parameters
    ----------
    graph_sequences : list output dari build_open_to_fist_sequence() / build_trial_graph_sequence()

    Returns
    -------
    reference dict sesuai format build_reference_sequence() dari graph_sequence_baseline
    """
    if not _GRAPH_SEQ_AVAILABLE:
        raise ImportError("graph_sequence_baseline tidak tersedia.")
    if not graph_sequences:
        raise ValueError("Tidak ada graph sequences untuk dibangun referensinya.")
    return _build_reference_sequence_base(graph_sequences, onset_ratio=onset_ratio)


# ---------------------------------------------------------------------------
# Pipeline: folder biomech trials → reference
# ---------------------------------------------------------------------------

def build_reference_pipeline(
    biomech_trials: list[dict[str, Any]],
    *,
    env_id: str = "myoHandPoseRandom-v0",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    edge_mode: str = DEFAULT_EDGE_MODE,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    include_edge_sign: bool = DEFAULT_INCLUDE_EDGE_SIGN,
    onset_ratio: float = DEFAULT_ONSET_RATIO,
    cohort: str = DEFAULT_COHORT,
    output_dir: Path | None = None,
    use_ctrl_ref: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    End-to-end pipeline: list biomech_trials → reference dict → optional save.

    Parameters
    ----------
    biomech_trials : list output dari process_recording_dir() atau build_biomech_reference_ctrl()

    Returns
    -------
    {
        'reference'       : reference dict (node_mean, node_std, edge_mean, ...),
        'graph_sequences' : list graph sequences per trial,
        'graph_template'  : graph template yang dipakai,
        'save_paths'      : dict path file tersimpan (jika output_dir diberikan),
    }
    """
    if not _GRAPH_SEQ_AVAILABLE:
        raise ImportError("graph_sequence_baseline tidak tersedia.")

    num_nodes = int(np.asarray(biomech_trials[0].get("ctrl_ref", biomech_trials[0].get("ctrl_analysis")), dtype=np.float32).shape[1]
                   if np.asarray(biomech_trials[0].get("ctrl_ref", biomech_trials[0].get("ctrl_analysis")), dtype=np.float32).ndim == 2
                   else biomech_trials[0].get("nu", 39))

    if verbose:
        print(f"[build_ref] Resolving graph template | expected_num_nodes={num_nodes}")

    graph_template = resolve_graph_template(expected_num_nodes=num_nodes, env_id=env_id)

    if verbose and graph_template.get("graph_warning"):
        print(f"[build_ref][WARN] {graph_template['graph_warning']}")

    graph_sequences: list[dict[str, Any]] = []
    for i, trial in enumerate(biomech_trials):
        if verbose:
            print(f"[build_ref] Building graph sequence {i + 1}/{len(biomech_trials)} | {trial.get('trial_id', '?')}")
        try:
            seq = build_open_to_fist_sequence(
                trial,
                graph_template,
                sequence_length=sequence_length,
                window_size=window_size,
                window_stride=window_stride,
                edge_mode=edge_mode,
                edge_threshold=edge_threshold,
                include_edge_sign=include_edge_sign,
                cohort=cohort,
                use_ctrl_ref=use_ctrl_ref,
            )
            graph_sequences.append(seq)
        except Exception as exc:
            if verbose:
                print(f"[build_ref][WARN] Trial {trial.get('trial_id', '?')}: {exc}")
            continue

    if not graph_sequences:
        raise RuntimeError("Tidak ada graph sequence yang berhasil dibangun.")

    reference = build_reference_from_trials(graph_sequences, onset_ratio=onset_ratio)

    save_paths: dict[str, str] = {}
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        save_paths = save_reference_sequence(output_dir, reference)
        if verbose:
            for k, v in save_paths.items():
                print(f"[build_ref] Saved {k}: {v}")

    return {
        "reference": reference,
        "graph_sequences": graph_sequences,
        "graph_template": graph_template,
        "save_paths": save_paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from app.offline_biomech.offline_biomech_reference import process_recording_dir
    from app.realtime.monolithic_runtime_backend import DEFAULT_MYOSUITE_ENV_ID, create_probe_env  # type: ignore

    parser = argparse.ArgumentParser(description="Build reference graph sequence dari biomech trials.")
    parser.add_argument("--records-dir", default="records")
    parser.add_argument("--output-dir", default="artifacts/biomech_reference/reference_sequence")
    parser.add_argument("--exercise-names", default="fist", help="Comma-separated.")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--onset-ratio", type=float, default=DEFAULT_ONSET_RATIO)
    parser.add_argument("--env-id", default=DEFAULT_MYOSUITE_ENV_ID)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    args = parser.parse_args()

    exercise_filter = {n.strip() for n in args.exercise_names.split(",") if n.strip()}
    env = None
    try:
        env, model, data = create_probe_env(str(args.env_id))
        trials = process_recording_dir(
            Path(args.records_dir),
            model, data,
            exercise_name_filter=exercise_filter,
        )
        if not trials:
            raise SystemExit("[ERROR] Tidak ada trial ditemukan.")
        result = build_reference_pipeline(
            trials,
            env_id=str(args.env_id),
            sequence_length=int(args.sequence_length),
            window_size=int(args.window_size),
            onset_ratio=float(args.onset_ratio),
            cohort=str(args.cohort),
            output_dir=Path(args.output_dir),
        )
        ref = result["reference"]
        print(f"[INFO] Reference built: {ref['trial_count']} trials | {len(ref['node_names'])} nodes | {len(ref['edges'])} edges")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
