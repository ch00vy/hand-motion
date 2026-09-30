from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.graph_sequence_baseline import load_trial_graph_sequence  # noqa: E402
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "dynamic_temporal_gcn_node_classifier"
    / "relation_score_visualization"
)

STATUS_WITHIN = "sesuai_acuan"
STATUS_DIFFERENT = "terindikasi_menyimpang"
STATUS_RELATED = "berpotensi_terdampak"

DISPLAY_STATUS = {
    STATUS_WITHIN: "Sesuai Pola Acuan",
    STATUS_DIFFERENT: "Pola Aktivitas Berbeda",
    STATUS_RELATED: "Otot Terkait Secara Relasional",
}

COLORS = {
    STATUS_WITHIN: "#22C55E",
    STATUS_DIFFERENT: "#EF4444",
    STATUS_RELATED: "#F59E0B",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _infer_final_status(row: dict[str, Any], *, threshold_prob: float, threshold_relation: float) -> str:
    explicit = str(row.get("affected_final_status") or row.get("final_status") or "").strip()
    if explicit in DISPLAY_STATUS:
        return explicit
    prob = _safe_float(row.get("probability_deviating", row.get("prob_different_pattern", 0.0)))
    relation_score = _safe_float(row.get("relational_risk", row.get("relation_score", 0.0)))
    if prob >= threshold_prob:
        return STATUS_DIFFERENT
    if relation_score >= threshold_relation:
        return STATUS_RELATED
    return STATUS_WITHIN


def _rows_from_analysis(
    analysis_result: dict[str, Any],
    *,
    threshold_prob: float,
    threshold_relation: float,
) -> pd.DataFrame:
    trial_id = str(analysis_result.get("trial_id") or "current_trial")
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(list(analysis_result.get("nodes") or [])):
        row = dict(raw)
        status = _infer_final_status(row, threshold_prob=threshold_prob, threshold_relation=threshold_relation)
        prob = _safe_float(row.get("probability_deviating", row.get("prob_different_pattern", 0.0)))
        relation_score = _safe_float(row.get("relational_risk", row.get("relation_score", 0.0)))
        rows.append(
            {
                "trial_id": trial_id,
                "node_id": int(row.get("node_index", idx)),
                "muscle_name": str(row.get("node_name") or row.get("muscle_name") or f"node_{idx}"),
                "prob_different_pattern": float(np.clip(prob, 0.0, 1.0)),
                "relation_score": float(max(0.0, relation_score)),
                "final_status": status,
                "final_status_display": DISPLAY_STATUS[status],
                "predicted_label": str(row.get("predicted_label") or ""),
                "deviation_score": _safe_float(row.get("deviation_score", 0.0)),
                "strongest_related_muscle": str(row.get("strongest_related_deviating_muscle") or ""),
            }
        )
    return pd.DataFrame(rows)


def _plot_relation_score(
    df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
    *,
    threshold_prob: float,
    threshold_relation: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#E5E7EB", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)

    for status in (STATUS_WITHIN, STATUS_DIFFERENT, STATUS_RELATED):
        part = df.loc[df["final_status"] == status]
        # Keep legend stable even when a category has no points.
        ax.scatter(
            part["prob_different_pattern"] if not part.empty else [],
            part["relation_score"] if not part.empty else [],
            s=92,
            color=COLORS[status],
            edgecolor="#111827",
            linewidth=0.7,
            alpha=0.92,
            label=DISPLAY_STATUS[status],
        )

    ax.axvline(threshold_prob, color="#334155", linestyle="--", linewidth=1.2)
    ax.axhline(threshold_relation, color="#334155", linestyle="--", linewidth=1.2)
    ax.text(
        threshold_prob + 0.012,
        0.98,
        "Ambang Prob. 0,5",
        transform=ax.get_xaxis_transform(),
        fontsize=8.5,
        color="#334155",
        va="top",
    )
    ax.text(
        0.012,
        threshold_relation + 0.01,
        "Ambang Relasi 0,25",
        transform=ax.get_yaxis_transform(),
        fontsize=8.5,
        color="#334155",
        va="bottom",
    )

    for idx, row in enumerate(df.to_dict(orient="records")):
        dx = 8 if idx % 2 == 0 else -8
        dy = 8 if idx % 3 else -8
        ax.annotate(
            str(row["muscle_name"]),
            (float(row["prob_different_pattern"]), float(row["relation_score"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color="#1F2937",
            ha="left" if dx > 0 else "right",
            va="center",
            arrowprops=dict(arrowstyle="-", color="#CBD5E1", lw=0.45, alpha=0.75),
        )

    y_max = max(float(df["relation_score"].max()) if not df.empty else 0.0, threshold_relation, 1.0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, y_max * 1.12)
    ax.set_title("Visualisasi Skor Relasi Antarotot pada Trial Aktif", fontsize=14, fontweight="bold", pad=18)
    ax.text(
        0.5,
        1.01,
        "Probabilitas Pola Berbeda vs Skor Relasi Antarotot",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#64748B",
    )
    ax.set_xlabel("Probabilitas Pola Berbeda")
    ax.set_ylabel("Skor Relasi Antarotot")
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#E5E7EB")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.18, dpi=350)
    fig.savefig(output_svg, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _top_relation_rows(
    *,
    analysis_result: dict[str, Any],
    node_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    artifact_paths = dict(analysis_result.get("artifact_paths") or {})
    graph_json = artifact_paths.get("graph_json")
    if not graph_json:
        return pd.DataFrame()
    graph_path = Path(str(graph_json))
    if not graph_path.exists():
        return pd.DataFrame()
    trial = load_trial_graph_sequence(graph_path)
    node_names = [str(name) for name in list(trial.get("node_names") or [])]
    edge_weight_seq = np.asarray(trial.get("edge_weight_seq"), dtype=float)
    edges = list(trial.get("edges") or [])
    if edge_weight_seq.ndim != 2 or not node_names or not edges:
        return pd.DataFrame()

    by_name = {str(row["muscle_name"]): dict(row) for row in node_df.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for edge in edges:
        src_idx = int(edge.get("source_idx", -1))
        dst_idx = int(edge.get("target_idx", -1))
        edge_idx = int(edge.get("edge_idx", -1))
        if not (0 <= src_idx < len(node_names) and 0 <= dst_idx < len(node_names) and 0 <= edge_idx < edge_weight_seq.shape[1]):
            continue
        src = str(edge.get("source_name") or node_names[src_idx])
        dst = str(edge.get("target_name") or node_names[dst_idx])
        values = edge_weight_seq[:, edge_idx].astype(float)
        values = values[np.isfinite(values)]
        edge_weight = float(np.mean(np.abs(values))) if values.size else 0.0
        src_row = by_name.get(src, {})
        dst_row = by_name.get(dst, {})
        src_prob = _safe_float(src_row.get("prob_different_pattern", 0.0))
        dst_prob = _safe_float(dst_row.get("prob_different_pattern", 0.0))
        src_contrib = edge_weight * src_prob
        dst_contrib = edge_weight * dst_prob
        if dst_contrib > src_contrib:
            source_name, target_name = dst, src
            source_row, target_row = dst_row, src_row
            source_prob, target_prob = dst_prob, src_prob
            contribution = dst_contrib
        else:
            source_name, target_name = src, dst
            source_row, target_row = src_row, dst_row
            source_prob, target_prob = src_prob, dst_prob
            contribution = src_contrib
        rows.append(
            {
                "Source Muscle / Node": source_name,
                "Target Muscle / Node": target_name,
                "Source Status": str(source_row.get("final_status_display") or "-"),
                "Target Status": str(target_row.get("final_status_display") or "-"),
                "Source Prob. Pola Berbeda": float(source_prob),
                "Target Prob. Pola Berbeda": float(target_prob),
                "Jenis Relasi": str(edge.get("edge_type") or "functional_group"),
                "Bobot Edge": float(edge_weight),
                "Kontribusi Skor Relasi": float(contribution),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Kontribusi Skor Relasi", "Bobot Edge"], ascending=False)
    out["_pair_key"] = out.apply(
        lambda row: "::".join(
            sorted([str(row["Source Muscle / Node"]), str(row["Target Muscle / Node"])])
            + [str(row["Jenis Relasi"])]
        ),
        axis=1,
    )
    out = out.drop_duplicates("_pair_key", keep="first").drop(columns=["_pair_key"])
    out = out.sort_values(["Kontribusi Skor Relasi", "Bobot Edge"], ascending=False).head(max(1, int(top_k)))
    return out.reset_index(drop=True)


def _plot_top_relation_bar(
    df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
    *,
    top_k: int = 15,
) -> None:
    import matplotlib.pyplot as plt

    plot_df = df.sort_values("relation_score", ascending=False).head(max(1, int(top_k))).copy()
    colors = [COLORS.get(str(status), "#64748B") for status in plot_df["final_status"].tolist()]
    fig, ax = plt.subplots(figsize=(11.0, 6.8), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.bar(plot_df["muscle_name"], plot_df["relation_score"], color=colors, edgecolor="#111827", linewidth=0.6)
    ax.set_title("Top Skor Relasi Antarotot", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Node / Otot")
    ax.set_ylabel("Skor Relasi Antarotot")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.18, dpi=350)
    fig.savefig(output_svg, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _write_notes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Catatan Visualisasi Skor Relasi Antarotot",
                "",
                "Scatter plot menampilkan hubungan antara probabilitas pola berbeda dan skor relasi antarotot.",
                "Garis vertikal 0,5 menunjukkan ambang interpretasi probabilitas pola berbeda.",
                "Garis horizontal 0,25 menunjukkan ambang interpretasi skor relasi antarotot.",
                "Titik pada kanan garis vertikal dikategorikan sebagai Pola Aktivitas Berbeda.",
                "Titik pada kiri garis vertikal tetapi di atas garis horizontal dikategorikan sebagai Otot Terkait Secara Relasional.",
                "Titik pada kiri garis vertikal dan di bawah garis horizontal dikategorikan sebagai Sesuai Pola Acuan.",
                "Relasi antarotot tetap terbentuk melalui edge graph dan dynamic edge weight sequence.",
                "Kategori Otot Terkait Secara Relasional hanya muncul jika node memiliki Prob. Pola Berbeda < 0,5 dan Skor Relasi Antarotot >= 0,25.",
                "Hasil ini merupakan interpretasi berbasis simulasi musculoskeletal, bukan diagnosis klinis.",
            ]
        ),
        encoding="utf-8",
    )


def export_relation_score_visualization(
    *,
    analysis_result: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    threshold_prob: float = 0.5,
    threshold_relation: float = 0.25,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _rows_from_analysis(
        analysis_result,
        threshold_prob=float(threshold_prob),
        threshold_relation=float(threshold_relation),
    )
    if df.empty:
        raise RuntimeError("No node rows found in analysis result.")

    output_png = output_dir / "relation_score_scatter_current_trial.png"
    output_svg = output_dir / "relation_score_scatter_current_trial.svg"
    output_csv = output_dir / "relation_score_scatter_current_trial.csv"
    output_summary = output_dir / "relation_score_scatter_current_trial_summary.json"
    output_notes = output_dir / "relation_score_visualization_notes.md"
    top_relations_csv = output_dir / "top_muscle_relations_current_trial.csv"
    top_relations_json = output_dir / "top_muscle_relations_current_trial.json"
    top_bar_png = output_dir / "top_relation_score_bar_current_trial.png"
    top_bar_svg = output_dir / "top_relation_score_bar_current_trial.svg"

    df.to_csv(output_csv, index=False)
    top_relations = _top_relation_rows(analysis_result=analysis_result, node_df=df, top_k=15)
    top_relations.to_csv(top_relations_csv, index=False)
    top_relations_json.write_text(
        json.dumps(top_relations.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    _plot_relation_score(
        df,
        output_png,
        output_svg,
        threshold_prob=float(threshold_prob),
        threshold_relation=float(threshold_relation),
    )
    _plot_top_relation_bar(df, top_bar_png, top_bar_svg, top_k=50)
    _write_notes(output_notes)
    summary = {
        "trial_id": str(analysis_result.get("trial_id") or ""),
        "num_nodes": int(len(df)),
        "threshold_prob": float(threshold_prob),
        "threshold_relation": float(threshold_relation),
        "counts_by_status": df["final_status_display"].value_counts().to_dict(),
        "note": (
            "Tidak ditemukan otot terkait secara relasional pada ambang 0,25."
            if int((df["final_status"] == STATUS_RELATED).sum()) == 0
            else ""
        ),
        "png": str(output_png),
        "svg": str(output_svg),
        "csv": str(output_csv),
        "top_relations_csv": str(top_relations_csv),
        "top_relations_json": str(top_relations_json),
        "top_relation_bar_png": str(top_bar_png),
        "top_relation_bar_svg": str(top_bar_svg),
        "num_top_relations": int(len(top_relations)),
        "notes": str(output_notes),
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(output_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export relation score scatter plot for a GUI analysis result.")
    parser.add_argument("--analysis-result-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-prob", type=float, default=0.5)
    parser.add_argument("--threshold-relation", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_relation_score_visualization(
        analysis_result=_read_json(Path(args.analysis_result_json)),
        output_dir=Path(args.output_dir),
        threshold_prob=float(args.threshold_prob),
        threshold_relation=float(args.threshold_relation),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
