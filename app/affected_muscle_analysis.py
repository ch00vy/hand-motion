from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.display_labels import SIMULATION_INTERPRETATION_NOTE, display_status, status_description


FINAL_STATUS_WITHIN = "sesuai_acuan"
FINAL_STATUS_DEVIATING = "terindikasi_menyimpang"
FINAL_STATUS_AFFECTED = "berpotensi_terdampak"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _node_name(row: dict[str, Any]) -> str:
    return str(row.get("muscle_name") or row.get("node_name") or "").strip()


def _edge_strength_matrix(
    graph_trial: dict[str, Any],
    *,
    aggregation_method: str = "mean",
    use_absolute_weight: bool = True,
) -> np.ndarray:
    node_names = list(graph_trial.get("node_names") or [])
    node_count = int(len(node_names))
    edge_weight_seq = np.asarray(graph_trial.get("edge_weight_seq"), dtype=float)
    edges = list(graph_trial.get("edges") or [])
    matrix = np.zeros((node_count, node_count), dtype=float)
    if node_count <= 0 or edge_weight_seq.ndim != 2 or not edges:
        return matrix

    method = str(aggregation_method or "mean").strip().lower()
    for edge in edges:
        src = int(edge.get("source_idx", -1))
        dst = int(edge.get("target_idx", -1))
        edge_idx = int(edge.get("edge_idx", -1))
        if not (0 <= src < node_count and 0 <= dst < node_count and 0 <= edge_idx < edge_weight_seq.shape[1]):
            continue
        values = edge_weight_seq[:, edge_idx].astype(float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            weight = 0.0
        else:
            if bool(use_absolute_weight):
                values = np.abs(values)
            if method == "max":
                weight = float(np.max(values))
            else:
                weight = float(np.mean(values))
        if not np.isfinite(weight):
            weight = 0.0
        matrix[src, dst] = max(matrix[src, dst], weight)
        matrix[dst, src] = max(matrix[dst, src], weight)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def analyze_affected_muscles(
    *,
    graph_trial: dict[str, Any],
    node_results: list[dict[str, Any]],
    threshold_deviating: float = 0.5,
    threshold_affected: float = 0.25,
    edge_aggregation_method: str = "mean",
    top_k_affected: int = 0,
    use_absolute_edge_weight: bool = True,
) -> dict[str, Any]:
    """Infer relationally connected muscles from model probabilities and graph edges.

    This is post-processing only. It does not change the primary GCN/Temporal
    GCN classification result. The relational status is not a medical assessment;
    it is an interpretation based on graph connectivity to muscles whose
    activity pattern differs from the reference.
    """

    node_names = [str(name) for name in list(graph_trial.get("node_names") or [])]
    result_by_name = {_node_name(row): dict(row) for row in node_results if _node_name(row)}
    if not node_names:
        node_names = list(result_by_name.keys())
    node_count = int(len(node_names))
    threshold_deviating = float(threshold_deviating)
    threshold_affected = float(threshold_affected)
    top_k_affected = max(0, int(top_k_affected))
    edge_matrix = _edge_strength_matrix(
        graph_trial,
        aggregation_method=edge_aggregation_method,
        use_absolute_weight=bool(use_absolute_edge_weight),
    )
    if edge_matrix.shape != (node_count, node_count):
        edge_matrix = np.zeros((node_count, node_count), dtype=float)

    p_dev = np.zeros(node_count, dtype=float)
    deviation_scores = np.zeros(node_count, dtype=float)
    for idx, name in enumerate(node_names):
        row = result_by_name.get(name, {})
        p_dev[idx] = np.clip(_safe_float(row.get("probability_deviating", row.get("p_deviating", 0.0))), 0.0, 1.0)
        deviation_scores[idx] = max(0.0, _safe_float(row.get("deviation_score", 0.0)))

    deviating_mask = p_dev >= threshold_deviating
    risk = np.zeros(node_count, dtype=float)
    strongest_index = np.full(node_count, -1, dtype=int)
    strongest_edge = np.zeros(node_count, dtype=float)
    deviating_indices = np.where(deviating_mask)[0].tolist()
    for dst in range(node_count):
        best_contribution = 0.0
        for src in deviating_indices:
            if src == dst:
                continue
            edge_weight = float(edge_matrix[src, dst])
            contribution = edge_weight * float(p_dev[src])
            risk[dst] += contribution
            if contribution > best_contribution:
                best_contribution = contribution
                strongest_index[dst] = int(src)
                strongest_edge[dst] = edge_weight

    affected_by_threshold = (~deviating_mask) & (risk >= threshold_affected)
    affected_by_topk = np.zeros(node_count, dtype=bool)
    if top_k_affected > 0:
        candidates = [idx for idx in range(node_count) if not bool(deviating_mask[idx]) and float(risk[idx]) > 0.0]
        candidates.sort(key=lambda idx: float(risk[idx]), reverse=True)
        for idx in candidates[:top_k_affected]:
            affected_by_topk[idx] = True

    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(node_names):
        is_deviating = bool(deviating_mask[idx])
        is_affected = bool(affected_by_threshold[idx] or affected_by_topk[idx])
        related_idx = int(strongest_index[idx])
        related_name = node_names[related_idx] if 0 <= related_idx < node_count else ""
        node_status = FINAL_STATUS_DEVIATING if is_deviating else FINAL_STATUS_WITHIN
        if is_deviating:
            final_status = FINAL_STATUS_DEVIATING
            explanation = (
                f"{name} memiliki pola aktivitas berbeda terhadap acuan karena "
                f"Probabilitas Pola Berbeda={p_dev[idx]:.3f} >= threshold {threshold_deviating:.3f}."
            )
        elif is_affected:
            final_status = FINAL_STATUS_AFFECTED
            explanation = (
                f"{name} terkait secara relasional dengan "
                f"{related_name or 'otot berpola aktivitas berbeda'} yang memiliki pola aktivitas berbeda "
                f"(Skor Relasi Antarotot={risk[idx]:.3f}, edge={strongest_edge[idx]:.3f})."
            )
        else:
            final_status = FINAL_STATUS_WITHIN
            explanation = (
                f"{name} sesuai pola acuan karena Probabilitas Pola Berbeda={p_dev[idx]:.3f} dan "
                f"Skor Relasi Antarotot={risk[idx]:.3f} berada di bawah threshold."
            )
        rows.append(
            {
                "muscle_name": name,
                "p_deviating": float(p_dev[idx]),
                "deviation_score": float(deviation_scores[idx]),
                "node_classification_status": node_status,
                "relational_risk": float(risk[idx]),
                "strongest_related_deviating_muscle": related_name,
                "strongest_edge_weight": float(strongest_edge[idx]),
                "final_status": final_status,
                "display_status": display_status(final_status),
                "display_label": display_status(final_status),
                "status_description": status_description(final_status),
                "explanation": explanation,
            }
        )

    top_affected = [
        {
            "muscle_name": row["muscle_name"],
            "relational_risk": row["relational_risk"],
            "strongest_related_deviating_muscle": row["strongest_related_deviating_muscle"],
            "strongest_edge_weight": row["strongest_edge_weight"],
        }
        for row in sorted(
            [row for row in rows if row["final_status"] == FINAL_STATUS_AFFECTED],
            key=lambda item: float(item["relational_risk"]),
            reverse=True,
        )[:10]
    ]
    summary = {
        "total_muscles": int(len(rows)),
        "deviating_muscles": int(sum(1 for row in rows if row["final_status"] == FINAL_STATUS_DEVIATING)),
        "affected_muscles": int(sum(1 for row in rows if row["final_status"] == FINAL_STATUS_AFFECTED)),
        "within_reference_muscles": int(sum(1 for row in rows if row["final_status"] == FINAL_STATUS_WITHIN)),
        "top_affected_muscles": top_affected,
        "threshold_deviating": float(threshold_deviating),
        "threshold_affected": float(threshold_affected),
        "edge_aggregation_method": str(edge_aggregation_method),
        "top_k_affected": int(top_k_affected),
        "use_absolute_edge_weight": bool(use_absolute_edge_weight),
        "display_status_labels": {
            FINAL_STATUS_WITHIN: display_status(FINAL_STATUS_WITHIN),
            FINAL_STATUS_DEVIATING: display_status(FINAL_STATUS_DEVIATING),
            FINAL_STATUS_AFFECTED: display_status(FINAL_STATUS_AFFECTED),
        },
        "note": (
            f"{SIMULATION_INTERPRETATION_NOTE} Otot terkait secara relasional adalah hasil analisis "
            "relasi antarotot terhadap otot dengan pola aktivitas berbeda, bukan penilaian medis."
        ),
    }
    return {"rows": rows, "summary": summary}


def write_affected_muscle_analysis_files(output_dir: Path, analysis: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "affected_muscle_analysis.csv"
    json_path = output_dir / "affected_muscle_analysis_summary.json"
    pd.DataFrame(list(analysis.get("rows") or [])).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(dict(analysis.get("summary") or {}), indent=2), encoding="utf-8")
    return csv_path, json_path
