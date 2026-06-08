from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATASET_DIR = PROJECT_ROOT / "artifacts" / "open_to_fist_graph_sequence_rebuild" / "dataset"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "artifacts" / "open_to_fist_reference_sequence_rebuild"
DEFAULT_COMPARE_DIR = PROJECT_ROOT / "artifacts" / "open_to_fist_trial_vs_reference_reaudit"
DEFAULT_BINARY_PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "gcn_binary_smoke"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "neo4j_export_bundle"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "neo4j_export_smoke_test.md"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_binary_predictions(predictions_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        path = predictions_dir / f"gcn_binary_predictions_{split}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "trial_id",
                "node_name",
                "target_class",
                "predicted_class",
                "predicted_confidence",
                "split",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows_"
    view = df.head(int(max_rows)).copy()
    cols = [str(col) for col in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in view.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _trial_set_from_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "trial_id" not in df.columns:
        return set()
    return {str(item) for item in df["trial_id"].dropna().tolist()}


def _edge_relation_type(edge_type: str) -> str:
    normalized = str(edge_type).strip().lower()
    if normalized == "synergist":
        return "SYNERGIST_WITH"
    if normalized == "antagonist":
        return "ANTAGONIST_WITH"
    if normalized == "functional_group":
        return "FUNCTIONAL_GROUP_WITH"
    return "RELATED_TO"


def _select_trials(
    manifest_df: pd.DataFrame,
    *,
    include_reference: bool,
    include_holdout: bool,
    reference_trial_ids: set[str],
    holdout_trial_ids: set[str],
) -> pd.DataFrame:
    if not include_reference and not include_holdout:
        return manifest_df.copy()
    keep_ids: set[str] = set()
    if include_reference:
        keep_ids.update(reference_trial_ids)
    if include_holdout:
        keep_ids.update(holdout_trial_ids)
    return manifest_df.loc[manifest_df["trial_id"].astype(str).isin(keep_ids)].copy()


def build_export_bundle(
    *,
    dataset_dir: Path,
    reference_root_dir: Path,
    compare_dir: Path,
    binary_predictions_dir: Path,
    output_dir: Path,
    motion_type: str,
    include_reference: bool,
    include_holdout: bool,
    include_deviation: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = pd.read_csv(dataset_dir / "dataset_manifest.csv")
    reference_pool_path = reference_root_dir / "reference_pool_manifest.csv"
    holdout_path = reference_root_dir / "holdout_manifest.csv"
    reference_trial_ids = _trial_set_from_manifest(reference_pool_path)
    holdout_trial_ids = _trial_set_from_manifest(holdout_path)
    selected_manifest = _select_trials(
        dataset_manifest,
        include_reference=include_reference,
        include_holdout=include_holdout,
        reference_trial_ids=reference_trial_ids,
        holdout_trial_ids=holdout_trial_ids,
    )

    binary_df = _load_binary_predictions(binary_predictions_dir)
    binary_lookup = {
        (str(row["trial_id"]), str(row["node_name"])): dict(row)
        for row in binary_df.to_dict(orient="records")
    }

    trials: list[dict[str, Any]] = []
    muscles: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    muscle_states: list[dict[str, Any]] = []
    static_relations: dict[tuple[str, str, str], dict[str, Any]] = {}
    dynamic_relations: list[dict[str, Any]] = []
    classification_results: list[dict[str, Any]] = []

    for manifest_row in selected_manifest.to_dict(orient="records"):
        trial_id = str(manifest_row["trial_id"])
        json_path = Path(str(manifest_row["json_path"]))
        npz_path = Path(str(manifest_row["npz_path"]))
        trial_meta = json.loads(json_path.read_text(encoding="utf-8"))
        arrays = np.load(npz_path, allow_pickle=True)
        window_progress = np.asarray(arrays["window_progress"], dtype=float)
        window_progress_start = np.asarray(arrays["window_progress_start"], dtype=float)
        window_progress_end = np.asarray(arrays["window_progress_end"], dtype=float)
        node_feature_seq = np.asarray(arrays["node_feature_seq"], dtype=float)
        edge_weight_seq = np.asarray(arrays["edge_weight_seq"], dtype=float)
        edge_active_seq = np.asarray(arrays["edge_active_seq"], dtype=int)
        node_names = [str(name) for name in trial_meta["node_names"]]
        feature_names = [str(name) for name in trial_meta["node_feature_names"]]
        edge_rows = list(trial_meta["edges"])

        compare_node_df = pd.DataFrame()
        if include_deviation:
            compare_node_df = _read_csv_if_exists(compare_dir / trial_id / "trial_vs_reference_node_deviation.csv")
        compare_lookup = {
            (int(row["step_index"]), str(row["node_name"])): dict(row)
            for row in compare_node_df.to_dict(orient="records")
        } if not compare_node_df.empty else {}

        trials.append(
            {
                "trial_id": trial_id,
                "motion_type": str(trial_meta.get("exercise_name") or motion_type),
                "source_file": str(trial_meta.get("source_csv") or manifest_row.get("source_csv") or ""),
                "subject_id": str(trial_meta.get("subject_id") or ""),
                "cohort": str(trial_meta.get("cohort") or manifest_row.get("cohort") or ""),
                "is_reference": str(trial_id in reference_trial_ids).lower(),
                "is_holdout": str(trial_id in holdout_trial_ids).lower(),
                "num_frames": int(trial_meta.get("num_frames", 0)),
                "num_steps": int(trial_meta.get("num_steps", 0)),
                "num_windows": int(trial_meta.get("num_windows", 0)),
                "edge_mode": str(trial_meta.get("edge_mode") or ""),
            }
        )

        for node_idx, node_name in enumerate(node_names):
            muscles[node_name] = {"muscle_name": node_name, "node_index": int(node_idx)}

        for edge in edge_rows:
            source_name = str(edge["source_name"])
            target_name = str(edge["target_name"])
            edge_type = str(edge.get("edge_type") or "related")
            key = (source_name, target_name, edge_type)
            if key not in static_relations:
                static_relations[key] = {
                    "source_name": source_name,
                    "target_name": target_name,
                    "edge_type": edge_type,
                    "relationship_type": _edge_relation_type(edge_type),
                    "prior_weight": _safe_float(edge.get("prior_weight"), 0.0),
                }

        for step_idx, progress in enumerate(window_progress.tolist()):
            step_id = f"{trial_id}::step::{step_idx}"
            steps.append(
                {
                    "step_id": step_id,
                    "trial_id": trial_id,
                    "step_index": int(step_idx),
                    "progress": float(progress),
                    "progress_start": float(window_progress_start[step_idx]),
                    "progress_end": float(window_progress_end[step_idx]),
                    "time_normalized": float(progress) / 100.0,
                    "reference_sequence_id": f"{motion_type}::normal_reference",
                }
            )
            for node_idx, node_name in enumerate(node_names):
                feature_values = {
                    feature_names[feature_idx]: _safe_float(node_feature_seq[step_idx, node_idx, feature_idx])
                    for feature_idx in range(len(feature_names))
                }
                compare_row = compare_lookup.get((step_idx, node_name), {})
                prediction_row = binary_lookup.get((trial_id, node_name), {})
                state_id = f"{trial_id}::step::{step_idx}::muscle::{node_name}"
                result_id = f"{state_id}::classification"
                binary_label = str(prediction_row.get("target_class") or "")
                predicted_label = str(prediction_row.get("predicted_class") or "")
                confidence = _safe_float(prediction_row.get("predicted_confidence"), 0.0)
                muscle_states.append(
                    {
                        "state_id": state_id,
                        "step_id": step_id,
                        "trial_id": trial_id,
                        "step_index": int(step_idx),
                        "progress": float(progress),
                        "muscle_name": node_name,
                        "node_index": int(node_idx),
                        "mean_ctrl": feature_values.get("mean_ctrl", 0.0),
                        "std_ctrl": feature_values.get("std_ctrl", 0.0),
                        "max_ctrl": feature_values.get("max_ctrl", 0.0),
                        "delta_ctrl": feature_values.get("delta_ctrl", 0.0),
                        "auc_ctrl": feature_values.get("auc_ctrl", 0.0),
                        "deviation_score": _safe_float(compare_row.get("aggregate_deviation_score"), 0.0),
                        "binary_label": binary_label,
                        "predicted_label": predicted_label,
                        "prediction_confidence": confidence,
                    }
                )
                if predicted_label or binary_label:
                    classification_results.append(
                        {
                            "result_id": result_id,
                            "state_id": state_id,
                            "trial_id": trial_id,
                            "step_index": int(step_idx),
                            "muscle_name": node_name,
                            "baseline_label": binary_label,
                            "predicted_label": predicted_label,
                            "confidence": confidence,
                            "source": "gcn_binary_smoke",
                        }
                    )
            for edge in edge_rows:
                edge_idx = int(edge["edge_idx"])
                if edge_idx >= edge_weight_seq.shape[1]:
                    continue
                source_name = str(edge["source_name"])
                target_name = str(edge["target_name"])
                dynamic_relations.append(
                    {
                        "source_state_id": f"{trial_id}::step::{step_idx}::muscle::{source_name}",
                        "target_state_id": f"{trial_id}::step::{step_idx}::muscle::{target_name}",
                        "trial_id": trial_id,
                        "step_index": int(step_idx),
                        "edge_idx": int(edge_idx),
                        "edge_type": str(edge.get("edge_type") or ""),
                        "weight": _safe_float(edge_weight_seq[step_idx, edge_idx], 0.0),
                        "active": int(edge_active_seq[step_idx, edge_idx]),
                        "prior_weight": _safe_float(edge.get("prior_weight"), 0.0),
                    }
                )

    reference_dir = reference_root_dir / "reference"
    reference_nodes = _read_csv_if_exists(reference_dir / "reference_sequence_nodes.csv")
    reference_edges = _read_csv_if_exists(reference_dir / "reference_sequence_edges.csv")
    reference_summary_path = reference_dir / "reference_summary.json"
    reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8")) if reference_summary_path.exists() else {}

    reference_sequences = [
        {
            "reference_sequence_id": f"{motion_type}::normal_reference",
            "motion_type": motion_type,
            "source_dir": str(reference_dir),
            "num_reference_trials": int(reference_summary.get("num_trials", len(reference_trial_ids))),
            "num_steps": int(reference_summary.get("num_steps", 0)),
            "notes": "normal open_to_fist reference sequence",
        }
    ]
    reference_steps: list[dict[str, Any]] = []
    reference_muscle_states: list[dict[str, Any]] = []
    if not reference_nodes.empty:
        for step_idx, step_group in reference_nodes.groupby("step_index", sort=True):
            first = step_group.iloc[0]
            reference_step_id = f"{motion_type}::normal_reference::step::{int(step_idx)}"
            reference_steps.append(
                {
                    "reference_step_id": reference_step_id,
                    "reference_sequence_id": f"{motion_type}::normal_reference",
                    "step_index": int(step_idx),
                    "progress": _safe_float(first.get("progress_percent"), 0.0),
                    "progress_start": _safe_float(first.get("progress_start_percent"), 0.0),
                    "progress_end": _safe_float(first.get("progress_end_percent"), 0.0),
                    "time_normalized": _safe_float(first.get("progress_percent"), 0.0) / 100.0,
                }
            )
        for row in reference_nodes.to_dict(orient="records"):
            node_name = str(row["node_name"])
            step_idx = _safe_int(row["step_index"])
            reference_muscle_states.append(
                {
                    "reference_state_id": f"{motion_type}::normal_reference::step::{step_idx}::muscle::{node_name}",
                    "reference_step_id": f"{motion_type}::normal_reference::step::{step_idx}",
                    "reference_sequence_id": f"{motion_type}::normal_reference",
                    "step_index": step_idx,
                    "muscle_name": node_name,
                    "node_index": _safe_int(row.get("node_index"), 0),
                    "mean_ctrl_mean": _safe_float(row.get("mean_ctrl_mean"), 0.0),
                    "mean_ctrl_std": _safe_float(row.get("mean_ctrl_std"), 0.0),
                    "max_ctrl_mean": _safe_float(row.get("max_ctrl_mean"), 0.0),
                    "max_ctrl_std": _safe_float(row.get("max_ctrl_std"), 0.0),
                    "delta_ctrl_mean": _safe_float(row.get("delta_ctrl_mean"), 0.0),
                    "auc_ctrl_mean": _safe_float(row.get("auc_ctrl_mean"), 0.0),
                }
            )

    tables = {
        "trials.csv": pd.DataFrame(trials),
        "sequence_steps.csv": pd.DataFrame(steps),
        "muscles.csv": pd.DataFrame(sorted(muscles.values(), key=lambda item: item["node_index"])),
        "muscle_states.csv": pd.DataFrame(muscle_states),
        "static_relations.csv": pd.DataFrame(static_relations.values()),
        "dynamic_relations.csv": pd.DataFrame(dynamic_relations),
        "classification_results.csv": pd.DataFrame(classification_results),
        "reference_sequences.csv": pd.DataFrame(reference_sequences),
        "reference_steps.csv": pd.DataFrame(reference_steps),
        "reference_muscle_states.csv": pd.DataFrame(reference_muscle_states),
    }
    if not reference_edges.empty:
        tables["reference_edges.csv"] = reference_edges

    for name, df in tables.items():
        df.to_csv(output_dir / name, index=False)

    cypher_path = output_dir / "import_open_to_fist_bundle.cypher"
    cypher_path.write_text(build_import_cypher(), encoding="utf-8")

    summary = {
        "motion_type": motion_type,
        "output_dir": str(output_dir),
        "num_trials": int(len(trials)),
        "num_muscles": int(len(muscles)),
        "num_sequence_steps": int(len(steps)),
        "num_muscle_states": int(len(muscle_states)),
        "num_static_relations": int(len(static_relations)),
        "num_dynamic_relations": int(len(dynamic_relations)),
        "num_classification_results": int(len(classification_results)),
        "num_reference_steps": int(len(reference_steps)),
        "num_reference_muscle_states": int(len(reference_muscle_states)),
        "csv_files": sorted(tables.keys()),
        "cypher_file": str(cypher_path),
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_import_cypher() -> str:
    return """// Open-to-fist graph sequence import bundle.
// Copy these CSV files into Neo4j's import directory, then run this file in cypher-shell or Neo4j Browser.

CREATE CONSTRAINT trial_id IF NOT EXISTS FOR (t:Trial) REQUIRE t.trial_id IS UNIQUE;
CREATE CONSTRAINT muscle_name IF NOT EXISTS FOR (m:Muscle) REQUIRE m.muscle_name IS UNIQUE;
CREATE CONSTRAINT sequence_step_id IF NOT EXISTS FOR (s:SequenceStep) REQUIRE s.step_id IS UNIQUE;
CREATE CONSTRAINT muscle_state_id IF NOT EXISTS FOR (s:MuscleState) REQUIRE s.state_id IS UNIQUE;
CREATE CONSTRAINT reference_sequence_id IF NOT EXISTS FOR (r:ReferenceSequence) REQUIRE r.reference_sequence_id IS UNIQUE;
CREATE CONSTRAINT reference_step_id IF NOT EXISTS FOR (r:ReferenceStep) REQUIRE r.reference_step_id IS UNIQUE;
CREATE CONSTRAINT reference_state_id IF NOT EXISTS FOR (r:ReferenceMuscleState) REQUIRE r.reference_state_id IS UNIQUE;
CREATE CONSTRAINT classification_result_id IF NOT EXISTS FOR (r:ClassificationResult) REQUIRE r.result_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///muscles.csv' AS row
MERGE (m:Muscle {muscle_name: row.muscle_name})
SET m.node_index = toInteger(row.node_index);

LOAD CSV WITH HEADERS FROM 'file:///trials.csv' AS row
MERGE (t:Trial {trial_id: row.trial_id})
SET t.motion_type = row.motion_type,
    t.source_file = row.source_file,
    t.subject_id = row.subject_id,
    t.cohort = row.cohort,
    t.is_reference = row.is_reference = 'true',
    t.is_holdout = row.is_holdout = 'true',
    t.num_frames = toInteger(row.num_frames),
    t.num_steps = toInteger(row.num_steps),
    t.num_windows = toInteger(row.num_windows),
    t.edge_mode = row.edge_mode;

LOAD CSV WITH HEADERS FROM 'file:///sequence_steps.csv' AS row
MATCH (t:Trial {trial_id: row.trial_id})
MERGE (s:SequenceStep {step_id: row.step_id})
SET s.step_index = toInteger(row.step_index),
    s.progress = toFloat(row.progress),
    s.progress_start = toFloat(row.progress_start),
    s.progress_end = toFloat(row.progress_end),
    s.time_normalized = toFloat(row.time_normalized),
    s.reference_sequence_id = row.reference_sequence_id
MERGE (t)-[:HAS_STEP]->(s);

LOAD CSV WITH HEADERS FROM 'file:///reference_sequences.csv' AS row
MERGE (r:ReferenceSequence {reference_sequence_id: row.reference_sequence_id})
SET r.motion_type = row.motion_type,
    r.source_dir = row.source_dir,
    r.num_reference_trials = toInteger(row.num_reference_trials),
    r.num_steps = toInteger(row.num_steps),
    r.notes = row.notes;

LOAD CSV WITH HEADERS FROM 'file:///reference_steps.csv' AS row
MATCH (r:ReferenceSequence {reference_sequence_id: row.reference_sequence_id})
MERGE (rs:ReferenceStep {reference_step_id: row.reference_step_id})
SET rs.step_index = toInteger(row.step_index),
    rs.progress = toFloat(row.progress),
    rs.progress_start = toFloat(row.progress_start),
    rs.progress_end = toFloat(row.progress_end),
    rs.time_normalized = toFloat(row.time_normalized)
MERGE (r)-[:HAS_REFERENCE_STEP]->(rs);

LOAD CSV WITH HEADERS FROM 'file:///muscle_states.csv' AS row
MATCH (s:SequenceStep {step_id: row.step_id})
MATCH (m:Muscle {muscle_name: row.muscle_name})
MERGE (ms:MuscleState {state_id: row.state_id})
SET ms.trial_id = row.trial_id,
    ms.step_index = toInteger(row.step_index),
    ms.progress = toFloat(row.progress),
    ms.muscle_name = row.muscle_name,
    ms.node_index = toInteger(row.node_index),
    ms.mean_ctrl = toFloat(row.mean_ctrl),
    ms.std_ctrl = toFloat(row.std_ctrl),
    ms.max_ctrl = toFloat(row.max_ctrl),
    ms.delta_ctrl = toFloat(row.delta_ctrl),
    ms.auc_ctrl = toFloat(row.auc_ctrl),
    ms.deviation_score = toFloat(row.deviation_score),
    ms.binary_label = row.binary_label,
    ms.predicted_label = row.predicted_label,
    ms.prediction_confidence = toFloat(row.prediction_confidence)
MERGE (s)-[:HAS_MUSCLE_STATE]->(ms)
MERGE (ms)-[:STATE_OF]->(m);

LOAD CSV WITH HEADERS FROM 'file:///reference_muscle_states.csv' AS row
MATCH (rs:ReferenceStep {reference_step_id: row.reference_step_id})
MATCH (m:Muscle {muscle_name: row.muscle_name})
MERGE (rms:ReferenceMuscleState {reference_state_id: row.reference_state_id})
SET rms.reference_sequence_id = row.reference_sequence_id,
    rms.step_index = toInteger(row.step_index),
    rms.muscle_name = row.muscle_name,
    rms.node_index = toInteger(row.node_index),
    rms.mean_ctrl_mean = toFloat(row.mean_ctrl_mean),
    rms.mean_ctrl_std = toFloat(row.mean_ctrl_std),
    rms.max_ctrl_mean = toFloat(row.max_ctrl_mean),
    rms.max_ctrl_std = toFloat(row.max_ctrl_std),
    rms.delta_ctrl_mean = toFloat(row.delta_ctrl_mean),
    rms.auc_ctrl_mean = toFloat(row.auc_ctrl_mean)
MERGE (rs)-[:HAS_REFERENCE_MUSCLE_STATE]->(rms)
MERGE (rms)-[:REFERENCE_STATE_OF]->(m);

LOAD CSV WITH HEADERS FROM 'file:///classification_results.csv' AS row
MATCH (ms:MuscleState {state_id: row.state_id})
MERGE (cr:ClassificationResult {result_id: row.result_id})
SET cr.trial_id = row.trial_id,
    cr.step_index = toInteger(row.step_index),
    cr.muscle_name = row.muscle_name,
    cr.baseline_label = row.baseline_label,
    cr.predicted_label = row.predicted_label,
    cr.confidence = toFloat(row.confidence),
    cr.source = row.source
MERGE (ms)-[:HAS_RESULT]->(cr);

LOAD CSV WITH HEADERS FROM 'file:///static_relations.csv' AS row
MATCH (a:Muscle {muscle_name: row.source_name})
MATCH (b:Muscle {muscle_name: row.target_name})
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.relationship_type = 'SYNERGIST_WITH'
  MERGE (a)-[r:SYNERGIST_WITH]->(b)
  SET r.edge_type = row.edge_type, r.prior_weight = toFloat(row.prior_weight)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.relationship_type = 'ANTAGONIST_WITH'
  MERGE (a)-[r:ANTAGONIST_WITH]->(b)
  SET r.edge_type = row.edge_type, r.prior_weight = toFloat(row.prior_weight)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.relationship_type = 'FUNCTIONAL_GROUP_WITH'
  MERGE (a)-[r:FUNCTIONAL_GROUP_WITH]->(b)
  SET r.edge_type = row.edge_type, r.prior_weight = toFloat(row.prior_weight)
}
CALL {
  WITH a, b, row
  WITH a, b, row WHERE row.relationship_type = 'RELATED_TO'
  MERGE (a)-[r:RELATED_TO]->(b)
  SET r.edge_type = row.edge_type, r.prior_weight = toFloat(row.prior_weight)
};

LOAD CSV WITH HEADERS FROM 'file:///dynamic_relations.csv' AS row
MATCH (a:MuscleState {state_id: row.source_state_id})
MATCH (b:MuscleState {state_id: row.target_state_id})
MERGE (a)-[r:DYNAMIC_RELATION {trial_id: row.trial_id, step_index: toInteger(row.step_index), edge_idx: toInteger(row.edge_idx)}]->(b)
SET r.edge_type = row.edge_type,
    r.weight = toFloat(row.weight),
    r.active = toInteger(row.active),
    r.prior_weight = toFloat(row.prior_weight);

MATCH (s:SequenceStep), (r:ReferenceSequence {reference_sequence_id: s.reference_sequence_id})
MERGE (s)-[:COMPARED_TO]->(r);
"""


def maybe_run_neo4j_import(summary: dict[str, Any], *, uri: str, user: str, password: str) -> str:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as exc:
        return f"Neo4j Python driver unavailable; generated CSV/Cypher bundle only: {type(exc).__name__}: {exc}"

    if not uri:
        return "No Neo4j URI provided; generated CSV/Cypher bundle only."

    schema_statements = [
        "CREATE CONSTRAINT trial_id IF NOT EXISTS FOR (t:Trial) REQUIRE t.trial_id IS UNIQUE",
        "CREATE CONSTRAINT muscle_name IF NOT EXISTS FOR (m:Muscle) REQUIRE m.muscle_name IS UNIQUE",
        "CREATE CONSTRAINT sequence_step_id IF NOT EXISTS FOR (s:SequenceStep) REQUIRE s.step_id IS UNIQUE",
        "CREATE CONSTRAINT muscle_state_id IF NOT EXISTS FOR (s:MuscleState) REQUIRE s.state_id IS UNIQUE",
        "CREATE CONSTRAINT reference_sequence_id IF NOT EXISTS FOR (r:ReferenceSequence) REQUIRE r.reference_sequence_id IS UNIQUE",
        "CREATE CONSTRAINT classification_result_id IF NOT EXISTS FOR (r:ClassificationResult) REQUIRE r.result_id IS UNIQUE",
    ]
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for statement in schema_statements:
                session.run(statement)
        return "Connected to Neo4j and installed constraints. CSV bundle still needs LOAD CSV import from Neo4j import directory."
    finally:
        driver.close()


def write_report(report_path: Path, summary: dict[str, Any], import_status: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counts_df = pd.DataFrame(
        [
            {"item": "trials", "count": summary["num_trials"]},
            {"item": "muscles", "count": summary["num_muscles"]},
            {"item": "sequence_steps", "count": summary["num_sequence_steps"]},
            {"item": "muscle_states", "count": summary["num_muscle_states"]},
            {"item": "static_relations", "count": summary["num_static_relations"]},
            {"item": "dynamic_relations", "count": summary["num_dynamic_relations"]},
            {"item": "classification_results", "count": summary["num_classification_results"]},
            {"item": "reference_steps", "count": summary["num_reference_steps"]},
            {"item": "reference_muscle_states", "count": summary["num_reference_muscle_states"]},
        ]
    )
    report_path.write_text(
        "\n".join(
            [
                "# Neo4j Export Smoke Test",
                "",
                f"Motion type: `{summary['motion_type']}`",
                f"Bundle directory: `{summary['output_dir']}`",
                f"Cypher file: `{summary['cypher_file']}`",
                f"Import status: {import_status}",
                "",
                "## Export Counts",
                _markdown_table(counts_df),
                "",
                "## Generated CSV Files",
                "\n".join(f"- `{name}`" for name in summary["csv_files"]),
                "",
                "This export is an auxiliary graph database/visualization layer. It does not change the offline biomech pipeline, binary GCN baseline, or realtime path.",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export open_to_fist graph sequence artifacts to a Neo4j-ready CSV/Cypher bundle.")
    parser.add_argument("--neo4j-uri", type=str, default="", help="Neo4j Bolt URI, e.g. bolt://localhost:7687.")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j-password", type=str, default="", help="Neo4j password.")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_DATASET_DIR), help="Graph sequence dataset directory.")
    parser.add_argument("--reference-root-dir", type=str, default=str(DEFAULT_REFERENCE_DIR), help="Reference sequence rebuild root directory.")
    parser.add_argument("--compare-dir", type=str, default=str(DEFAULT_COMPARE_DIR), help="Trial-vs-reference comparison output directory.")
    parser.add_argument("--binary-predictions-dir", type=str, default=str(DEFAULT_BINARY_PREDICTIONS_DIR), help="Binary GCN predictions directory.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for CSV/Cypher bundle.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT), help="Markdown report path.")
    parser.add_argument("--motion-type", type=str, default="open_to_fist", help="Motion type to export.")
    parser.add_argument("--include-reference", action="store_true", help="Export reference-pool trials.")
    parser.add_argument("--include-holdout", action="store_true", help="Export holdout trials.")
    parser.add_argument("--include-deviation", action="store_true", help="Attach trial-vs-reference deviation scores where available.")
    parser.add_argument("--dry-run", action="store_true", help="Only generate bundle; do not connect to Neo4j.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_export_bundle(
        dataset_dir=Path(args.input_dir).resolve(),
        reference_root_dir=Path(args.reference_root_dir).resolve(),
        compare_dir=Path(args.compare_dir).resolve(),
        binary_predictions_dir=Path(args.binary_predictions_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        motion_type=str(args.motion_type),
        include_reference=bool(args.include_reference),
        include_holdout=bool(args.include_holdout),
        include_deviation=bool(args.include_deviation),
    )
    if bool(args.dry_run):
        import_status = "Dry-run: generated CSV/Cypher bundle only."
    else:
        import_status = maybe_run_neo4j_import(
            summary,
            uri=str(args.neo4j_uri),
            user=str(args.neo4j_user),
            password=str(args.neo4j_password),
        )
    write_report(Path(args.report_path).resolve(), summary, import_status)
    print(f"[INFO] Neo4j export bundle: {summary['output_dir']}")
    print(f"[INFO] Cypher import file: {summary['cypher_file']}")
    print(f"[INFO] Report: {Path(args.report_path).resolve()}")
    print(f"[INFO] {import_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
