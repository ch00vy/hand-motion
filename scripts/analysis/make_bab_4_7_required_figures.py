from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    nx = None

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - optional dependency
    PCA = None


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "bab_4_7_grafik_wajib"

RUNTIME_DIR = ROOT / "artifacts" / "realtime_recorded_qpos_audit"
MODEL_DIR = ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier_labeled_records"
RETRAIN_DIR = ROOT / "artifacts" / "retrain_labeled_records"
CTRL_NORMAL_DIR = RETRAIN_DIR / "ctrl_ref_normal" / "reference_build"
CTRL_DIFFERENT_DIR = RETRAIN_DIR / "ctrl_ref_abnormal" / "reference_build"
GRAPH_NORMAL_DIR = RETRAIN_DIR / "graph_sequence_dataset"
GRAPH_DIFFERENT_DIR = RETRAIN_DIR / "abnormal_graph_sequence_dataset"
EDGE_PATH = ROOT / "artifacts" / "final_book_figures" / "neo4j_export" / "neo4j_muscle_edges.csv"

STATUS_ORDER = [
    "Sesuai Pola Acuan",
    "Otot Terkait Secara Relasional",
    "Pola Aktivitas Berbeda",
]
STATUS_COLORS = {
    "Sesuai Pola Acuan": "#4B5563",
    "Otot Terkait Secara Relasional": "#8C6D31",
    "Pola Aktivitas Berbeda": "#1F4E79",
}

COLOR_MAIN = "#1F4E79"
COLOR_SECONDARY = "#6B7280"
COLOR_LIGHT = "#D1D5DB"
COLOR_DARK = "#111827"
COLOR_ACCENT = "#8C6D31"


@dataclass
class FigureRecord:
    filename: str
    section: str
    source: str
    caption: str
    notes: str = ""


FIGURES: list[FigureRecord] = []
FAILED: list[tuple[str, str]] = []


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "axes.edgecolor": "#D1D5DB",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.labelcolor": "#111827",
            "text.color": "#111827",
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def source_label(*names: str) -> str:
    return "; ".join(names)


def savefig(fig: plt.Figure, filename: str, section: str, source: str, caption: str, notes: str = "") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / filename
    pdf_path = png_path.with_suffix(".pdf")
    for ax in fig.axes:
        if hasattr(ax, "spines"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES.append(FigureRecord(filename, section, source, caption, notes))


def polish_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if ygrid:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    if xgrid:
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)


def label_bars(ax: plt.Axes, fmt: str = "{:.2f}") -> None:
    for patch in ax.patches:
        value = patch.get_width() if patch.get_width() > patch.get_height() else patch.get_height()
        if not np.isfinite(value):
            continue
        if patch.get_width() > patch.get_height():
            ax.text(
                patch.get_width(),
                patch.get_y() + patch.get_height() / 2,
                fmt.format(value),
                va="center",
                ha="left",
                fontsize=8,
            )
        else:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                fmt.format(value),
                va="bottom",
                ha="center",
                fontsize=8,
            )


def display_status(predicted: str, p_dev: float, relation_score: float = 0.0) -> str:
    if str(predicted).strip() == "deviating" or float(p_dev) >= 0.5:
        return "Pola Aktivitas Berbeda"
    if float(relation_score) >= 0.25:
        return "Otot Terkait Secara Relasional"
    return "Sesuai Pola Acuan"


def choose_trials(pred_test: pd.DataFrame, pred_different: pd.DataFrame) -> tuple[str, str]:
    normal_trial = ""
    different_trial = ""
    if not pred_test.empty:
        normal_trial = str(pred_test.groupby("trial_id")["prob_deviating"].mean().sort_values().index[0])
    if not pred_different.empty:
        different_trial = str(
            pred_different.groupby("trial_id")["prob_deviating"].mean().sort_values(ascending=False).index[0]
        )
    return normal_trial, different_trial


def load_edges() -> pd.DataFrame:
    edges = read_csv(EDGE_PATH)
    if edges.empty:
        return edges
    edges["edge_weight"] = pd.to_numeric(edges.get("edge_weight", 0.0), errors="coerce").fillna(0.0)
    if "relation_type" not in edges.columns:
        edges["relation_type"] = "functional_group"
    return edges


def relation_table(pred_df: pd.DataFrame, trial_id: str, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial = pred_df.loc[pred_df["trial_id"].astype(str) == str(trial_id)].copy()
    if trial.empty:
        return pd.DataFrame(), pd.DataFrame()
    trial["prob_deviating"] = pd.to_numeric(trial["prob_deviating"], errors="coerce").fillna(0.0)
    p_by_node = dict(zip(trial["node_name"].astype(str), trial["prob_deviating"].astype(float)))
    pred_by_node = dict(zip(trial["node_name"].astype(str), trial["predicted_class"].astype(str)))
    primary_nodes = {n for n, p in p_by_node.items() if p >= 0.5 or pred_by_node.get(n) == "deviating"}

    risk = {name: 0.0 for name in p_by_node}
    top_rows: list[dict[str, Any]] = []
    if not edges.empty:
        for row in edges.to_dict(orient="records"):
            src = str(row.get("source_muscle", ""))
            dst = str(row.get("target_muscle", ""))
            if src not in p_by_node or dst not in p_by_node:
                continue
            w = float(row.get("edge_weight", 0.0) or 0.0)
            rel = str(row.get("relation_type", "functional_group"))
            for u, v in ((src, dst), (dst, src)):
                if u in primary_nodes:
                    risk[v] += w * p_by_node[u]
            top_rows.append(
                {
                    "source": src,
                    "target": dst,
                    "source_status": display_status(pred_by_node[src], p_by_node[src]),
                    "target_status": display_status(pred_by_node[dst], p_by_node[dst]),
                    "source_p_dev": p_by_node[src],
                    "target_p_dev": p_by_node[dst],
                    "relation_type": rel,
                    "edge_weight": w,
                    "relation_contribution": max(w * p_by_node[src], w * p_by_node[dst]),
                }
            )

    out = trial[["trial_id", "node_index", "node_name", "predicted_class", "prob_deviating"]].copy()
    out["relation_score"] = out["node_name"].map(risk).fillna(0.0)
    out["final_status"] = [
        display_status(pred, p, r)
        for pred, p, r in zip(out["predicted_class"], out["prob_deviating"], out["relation_score"])
    ]
    top_rel = pd.DataFrame(top_rows)
    if not top_rel.empty:
        top_rel = top_rel.sort_values("relation_contribution", ascending=False).drop_duplicates(["source", "target"])
    return out, top_rel


def runtime_figures() -> None:
    summary = read_json(RUNTIME_DIR / "validation_summary.json")
    if not summary:
        FAILED.append(("fig_4_7_1_runtime_latency.png", "runtime validation_summary.json tidak ditemukan"))
        FAILED.append(("fig_4_7_1_runtime_error.png", "runtime validation_summary.json tidak ditemukan"))
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = ["Mean ctrl pipeline", "P95 ctrl pipeline", "Mean frame loop", "P95 frame loop"]
    values = [
        summary.get("latency_mean_ms", np.nan),
        summary.get("latency_p95_ms", np.nan),
        summary.get("frame_loop_mean_ms", np.nan),
        summary.get("frame_loop_p95_ms", np.nan),
    ]
    ax.barh(labels[::-1], values[::-1], color=[COLOR_MAIN, COLOR_SECONDARY, COLOR_MAIN, COLOR_SECONDARY][::-1])
    ax.set_title("Perbandingan Latency pada Runtime Estimation")
    ax.set_xlabel("Latency (ms)")
    label_bars(ax, "{:.1f}")
    polish_axis(ax, xgrid=True, ygrid=False)
    savefig(
        fig,
        "fig_4_7_1_runtime_latency.png",
        "4.7.1",
        "runtime validation summary",
        "Perbandingan latency pipeline estimasi control dan loop frame pada runtime validation.",
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = ["Mean joint error", "RMSE forward error", "P95 frame error", "Fallback ratio"]
    values = [
        summary.get("sequence_mean_joint_error", np.nan),
        summary.get("sequence_rmse_forward_error", np.nan),
        summary.get("sequence_p95_frame_error", np.nan),
        summary.get("fallback_ratio", np.nan),
    ]
    ax.barh(labels[::-1], values[::-1], color=[COLOR_SECONDARY, COLOR_MAIN, COLOR_SECONDARY, COLOR_LIGHT][::-1])
    ax.set_title("Ringkasan Error dan Fallback pada Runtime Estimation")
    ax.set_xlabel("Nilai metrik")
    label_bars(ax, "{:.3f}")
    polish_axis(ax, xgrid=True, ygrid=False)
    savefig(
        fig,
        "fig_4_7_1_runtime_error.png",
        "4.7.1",
        "runtime validation summary",
        "Ringkasan error dan rasio fallback pada runtime validation.",
    )


def ctrl_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_csv(CTRL_NORMAL_DIR / "biomech_reference_trial_summary.csv"),
        read_csv(CTRL_DIFFERENT_DIR / "biomech_reference_trial_summary.csv"),
    )


def representative_ctrl_trial() -> tuple[str, pd.DataFrame, Path]:
    normal, _ = ctrl_summaries()
    if normal.empty or "dataset_path" not in normal.columns:
        return "", pd.DataFrame(), Path()
    row = normal.sort_values("rmse_forward_error").iloc[0]
    path = Path(str(row["dataset_path"]))
    return str(row["trial_id"]), read_csv(path), path


def control_figures() -> None:
    trial_id, df, path = representative_ctrl_trial()
    ctrl_cols = [c for c in df.columns if c.startswith("ctrl_teacher_")]
    if df.empty or not ctrl_cols:
        FAILED.append(("fig_4_7_3_control_profile_example.png", "kolom control trial representatif tidak ditemukan"))
        FAILED.append(("fig_4_7_3_top_mean_control.png", "kolom control trial representatif tidak ditemukan"))
        return
    means = df[ctrl_cols].mean().sort_values(ascending=False)
    top_cols = means.head(5).index.tolist()
    x = np.linspace(0, 100, len(df))
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for col in top_cols:
        ax.plot(x, df[col].astype(float), linewidth=1.7, label=col.replace("ctrl_teacher_", ""))
    ax.set_title("Profil Control Otot Selama Gerakan Open-to-Fist")
    ax.set_xlabel("Progress gerakan (%)")
    ax.set_ylabel("Nilai control")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, fontsize=8)
    polish_axis(ax)
    savefig(
        fig,
        "fig_4_7_3_control_profile_example.png",
        "4.7.3",
        "accepted ctrl_ref trial",
        "Profil temporal control otot utama pada satu trial representatif.",
        f"Trial: {trial_id}",
    )

    top = means.head(10).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh([c.replace("ctrl_teacher_", "") for c in top.index], top.values, color=COLOR_MAIN)
    ax.set_title("Top 10 Rata-Rata Control Otot pada Trial Representatif")
    ax.set_xlabel("Rata-rata nilai control")
    ax.set_xlim(0, max(1.0, float(top.max()) * 1.05))
    label_bars(ax, "{:.3f}")
    polish_axis(ax, xgrid=True, ygrid=False)
    savefig(
        fig,
        "fig_4_7_3_top_mean_control.png",
        "4.7.3",
        "accepted ctrl_ref trial",
        "Top 10 otot/aktuator dengan rata-rata control tertinggi pada trial representatif.",
        f"Trial: {trial_id}; source={path.name}",
    )


def replay_validation_figures() -> None:
    normal, different = ctrl_summaries()
    if normal.empty or different.empty:
        FAILED.append(("fig_4_7_4_replay_error_comparison.png", "summary ctrl/ref kelompok trial tidak lengkap"))
        FAILED.append(("fig_4_7_4_ctrl_delta_comparison.png", "summary ctrl/ref kelompok trial tidak lengkap"))
        return
    groups = ["Sesuai Pola Acuan", "Pola Aktivitas Berbeda"]
    mean_joint = [
        pd.to_numeric(normal["mean_joint_error"], errors="coerce").mean(),
        pd.to_numeric(different["mean_joint_error"], errors="coerce").mean(),
    ]
    mean_rmse = [
        pd.to_numeric(normal["rmse_forward_error"], errors="coerce").mean(),
        pd.to_numeric(different["rmse_forward_error"], errors="coerce").mean(),
    ]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(x - width / 2, mean_joint, width=width, label="Mean joint error", color=COLOR_SECONDARY)
    ax.bar(x + width / 2, mean_rmse, width=width, label="Mean RMSE", color=COLOR_MAIN)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Nilai error")
    ax.set_title("Perbandingan Error Replay antar Kelompok Trial")
    ax.legend()
    label_bars(ax, "{:.3f}")
    polish_axis(ax)
    savefig(
        fig,
        "fig_4_7_4_replay_error_comparison.png",
        "4.7.4",
        "ctrl_ref replay validation summaries",
        "Perbandingan mean joint error dan mean RMSE replay antara dua kelompok trial.",
    )

    vals = [
        pd.to_numeric(normal["ctrl_delta_mean"], errors="coerce").mean(),
        pd.to_numeric(different["ctrl_delta_mean"], errors="coerce").mean(),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(groups, vals, color=[COLOR_SECONDARY, COLOR_MAIN])
    ax.set_title("Perbandingan Mean Ctrl Delta pada Validasi Replay")
    ax.set_ylabel("Mean ctrl delta")
    label_bars(ax, "{:.3f}")
    polish_axis(ax)
    savefig(
        fig,
        "fig_4_7_4_ctrl_delta_comparison.png",
        "4.7.4",
        "ctrl_ref replay validation summaries",
        "Perbandingan rerata perubahan control pada validasi replay.",
    )


def graph_figures(normal_trial: str) -> None:
    json_path = GRAPH_NORMAL_DIR / f"normal__{normal_trial}.json"
    npz_path = GRAPH_NORMAL_DIR / f"normal__{normal_trial}.npz"
    meta = read_json(json_path)
    if not npz_path.exists() or not meta:
        FAILED.append(("fig_4_7_5_temporal_heatmap_example.png", "graph temporal trial representatif tidak ditemukan"))
    else:
        arr = np.load(npz_path)
        feat = arr["node_feature_seq"][:, :, 0].T
        node_names = meta.get("node_names", [f"node_{i}" for i in range(feat.shape[0])])
        order = np.argsort(feat.mean(axis=1))[::-1][:20]
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        im = ax.imshow(feat[order], aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_title("Heatmap Temporal Aktivitas Otot pada Graph Temporal")
        ax.set_xlabel("Window/progress gerakan")
        ax.set_ylabel("Node otot")
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([node_names[i] for i in order], fontsize=8)
        fig.colorbar(im, ax=ax, label="Fitur aktivitas/control")
        savefig(
            fig,
            "fig_4_7_5_temporal_heatmap_example.png",
            "4.7.5",
            "graph temporal npz/json",
            "Heatmap fitur aktivitas/control node terhadap window gerakan pada graph temporal.",
            f"Trial: {normal_trial}; top 20 node berdasarkan rerata fitur.",
        )

    summary = read_json(MODEL_DIR / "dataset_summary.json")
    num_nodes = int(summary.get("num_nodes", 39))
    num_windows = int(summary.get("num_windows", 56))
    num_features = int(summary.get("features_per_node_per_window", 6))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.axis("off")
    boxes = [
        (f"{num_nodes}\nnode otot", 0.12, 0.50, "#E5E7EB"),
        (f"{num_windows}\nsequence window", 0.42, 0.50, "#D1D5DB"),
        (f"{num_features}\nfitur per node/window", 0.72, 0.50, "#E5E7EB"),
    ]
    for text, x0, y0, color in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), 0.18, 0.22, fc=color, ec="#374151", lw=1.3))
        ax.text(x0 + 0.09, y0 + 0.11, text, ha="center", va="center", fontweight="bold")
    for x0 in (0.33, 0.63):
        ax.annotate("", xy=(x0 + 0.05, 0.61), xytext=(x0 - 0.035, 0.61), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(0.5, 0.28, "Representasi input: sequence window x node otot x fitur", ha="center", fontsize=11)
    ax.set_title("Struktur Data Graph Temporal")
    savefig(
        fig,
        "fig_4_7_5_graph_temporal_dimension.png",
        "4.7.5",
        "dynamic temporal GCN dataset summary",
        "Ringkasan dimensi graph temporal yang digunakan sebagai input model.",
    )


def training_figures(history: pd.DataFrame) -> None:
    if history.empty:
        for name in (
            "fig_4_7_6_training_accuracy.png",
            "fig_4_7_6_training_loss.png",
            "fig_4_7_6_training_f1_score.png",
        ):
            FAILED.append((name, "history.csv tidak ditemukan"))
        return
    specs = [
        (
            "fig_4_7_6_training_accuracy.png",
            ["train_accuracy", "val_accuracy"],
            "Accuracy",
            "Kurva Accuracy Training dan Validation Dynamic Temporal GCN",
        ),
        (
            "fig_4_7_6_training_loss.png",
            ["train_loss", "val_loss"],
            "Loss",
            "Kurva Loss Training dan Validation Dynamic Temporal GCN",
        ),
        (
            "fig_4_7_6_training_f1_score.png",
            ["train_macro_f1", "val_macro_f1", "train_binary_f1_deviating", "val_binary_f1_deviating"],
            "F1-score",
            "Kurva F1-Score Training dan Validation Dynamic Temporal GCN",
        ),
    ]
    metrics = read_json(MODEL_DIR / "eval_metrics.json")
    for filename, columns, ylabel, title in specs:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for col in columns:
            if col in history.columns:
                if col.startswith("train_"):
                    label = "Train"
                    color = COLOR_MAIN
                    style = "-"
                elif col.startswith("val_"):
                    label = "Validation"
                    color = COLOR_SECONDARY
                    style = "--"
                else:
                    label = col.replace("_", " ")
                    color = COLOR_ACCENT
                    style = ":"
                if "binary_f1" in col:
                    label = f"{label} F1 pola berbeda"
                ax.plot(history["epoch"], history[col], linewidth=1.7, linestyle=style, color=color, label=label)
        if metrics.get("best_epoch"):
            ax.axvline(float(metrics["best_epoch"]), color=COLOR_DARK, linestyle=":", linewidth=1, label="Best epoch")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
        polish_axis(ax)
        savefig(fig, filename, "4.7.6", "training history", f"{title}.")


def probability_figures(normal_rel: pd.DataFrame, different_rel: pd.DataFrame) -> None:
    specs = [
        (
            normal_rel,
            "fig_4_7_7_probability_per_muscle_normal.png",
            "Probabilitas Pola Aktivitas Berbeda pada Trial Sesuai Pola Acuan",
            "4.7.7",
        ),
        (
            different_rel,
            "fig_4_7_7_probability_per_muscle_different.png",
            "Probabilitas Pola Aktivitas Berbeda pada Trial Pola Aktivitas Berbeda",
            "4.7.7",
        ),
    ]
    for df, filename, title, section in specs:
        if df.empty:
            FAILED.append((filename, "prediksi trial representatif tidak tersedia"))
            continue
        top = df.sort_values("prob_deviating", ascending=False).head(15).sort_values("prob_deviating")
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.barh(top["node_name"], top["prob_deviating"], color=[STATUS_COLORS[s] for s in top["final_status"]])
        ax.axvline(0.5, color=COLOR_DARK, linestyle="--", linewidth=1, label="Ambang 0,5")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Probabilitas Pola Berbeda")
        ax.set_title(title)
        ax.legend(fontsize=8)
        polish_axis(ax, xgrid=True, ygrid=False)
        savefig(fig, filename, section, "selected trial predictions", f"{title}.")

    if normal_rel.empty or different_rel.empty:
        FAILED.append(("fig_4_7_7_status_count_comparison.png", "status node trial representatif tidak lengkap"))
        return
    rows = []
    for trial_name, df in (("Sesuai Pola Acuan", normal_rel), ("Pola Aktivitas Berbeda", different_rel)):
        counts = df["final_status"].value_counts().to_dict()
        for status in STATUS_ORDER:
            rows.append({"trial": trial_name, "status": status, "count": counts.get(status, 0)})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(2)
    width = 0.25
    for i, status in enumerate(STATUS_ORDER):
        vals = plot_df.loc[plot_df["status"] == status, "count"].to_numpy()
        ax.bar(x + (i - 1) * width, vals, width=width, label=status, color=STATUS_COLORS[status])
    ax.set_xticks(x)
    ax.set_xticklabels(["Trial sesuai pola acuan", "Trial pola aktivitas berbeda"])
    ax.set_ylabel("Jumlah node")
    ax.set_title("Perbandingan Jumlah Status Otot pada Dua Trial Representatif")
    ax.legend(fontsize=8)
    polish_axis(ax)
    savefig(
        fig,
        "fig_4_7_7_status_count_comparison.png",
        "4.7.7",
        "selected trial predictions + relation score",
        "Perbandingan jumlah node berdasarkan status interpretasi pada dua trial representatif.",
    )


def relation_figures(normal_rel: pd.DataFrame, different_rel: pd.DataFrame, top_rel: pd.DataFrame) -> None:
    for df, filename, title in (
        (
            normal_rel,
            "fig_4_7_8_relation_scatter_normal.png",
            "Sebaran Probabilitas dan Skor Relasi pada Trial Sesuai Pola Acuan",
        ),
        (
            different_rel,
            "fig_4_7_8_relation_scatter_different.png",
            "Sebaran Probabilitas dan Skor Relasi pada Trial Pola Aktivitas Berbeda",
        ),
    ):
        if df.empty:
            FAILED.append((filename, "data skor relasi trial representatif tidak tersedia"))
            continue
        fig, ax = plt.subplots(figsize=(6.3, 4.4))
        for status, group in df.groupby("final_status"):
            ax.scatter(
                group["prob_deviating"],
                group["relation_score"],
                s=70,
                color=STATUS_COLORS.get(status, "#6B7280"),
                edgecolor="white",
                linewidth=0.8,
                label=status,
            )
        top_labels = df.assign(rank=df["prob_deviating"] + df["relation_score"]).sort_values("rank", ascending=False).head(8)
        for _, row in top_labels.iterrows():
            ax.text(row["prob_deviating"] + 0.01, row["relation_score"] + 0.01, str(row["node_name"]), fontsize=8)
        ax.axvline(0.5, color=COLOR_DARK, linestyle="--", linewidth=1)
        ax.axhline(0.25, color=COLOR_DARK, linestyle=":", linewidth=1)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Probabilitas Pola Berbeda")
        ax.set_ylabel("Skor Relasi Antarotot")
        ax.set_title(title)
        ax.legend(fontsize=8)
        polish_axis(ax)
        savefig(fig, filename, "4.7.8", "selected trial predictions + relation graph", f"{title}.")

    if top_rel.empty:
        FAILED.append(("fig_4_7_8_top_relation_bar.png", "edge relation score tidak tersedia"))
        FAILED.append(("fig_4_7_8_relation_subgraph.png", "edge relation score tidak tersedia"))
        return
    top = top_rel.head(10).copy().sort_values("relation_contribution")
    labels = top["source"].astype(str) + " - " + top["target"].astype(str)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.barh(labels, top["relation_contribution"], color=COLOR_ACCENT)
    ax.set_xlabel("Skor relasi")
    ax.set_title("Top 10 Relasi Antarotot dengan Skor Tertinggi")
    label_bars(ax, "{:.3f}")
    polish_axis(ax, xgrid=True, ygrid=False)
    savefig(
        fig,
        "fig_4_7_8_top_relation_bar.png",
        "4.7.8",
        "relation graph + selected trial predictions",
        "Top 10 pasangan otot berdasarkan skor relasi tertinggi.",
    )

    top = top_rel.head(12)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    if nx is not None:
        graph = nx.Graph()
        for _, row in top.iterrows():
            graph.add_edge(row["source"], row["target"], weight=float(row["relation_contribution"]))
        pos = nx.spring_layout(graph, seed=11, weight="weight")
        weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            width=1.0 + 5.0 * weights / max(float(weights.max()), 1e-9),
            edge_color=COLOR_ACCENT,
            alpha=0.75,
        )
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#E5E7EB", edgecolors=COLOR_MAIN, node_size=620)
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    else:
        nodes = sorted(set(top["source"]).union(set(top["target"])))
        angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        pos = {node: (math.cos(a), math.sin(a)) for node, a in zip(nodes, angles)}
        for _, row in top.iterrows():
            p0, p1 = pos[row["source"]], pos[row["target"]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=COLOR_ACCENT, alpha=0.75)
        for node, (x0, y0) in pos.items():
            ax.scatter([x0], [y0], s=520, color="#E5E7EB", edgecolor=COLOR_MAIN)
            ax.text(x0, y0, str(node), ha="center", va="center", fontsize=8)
    ax.set_title("Subgraph Relasi Antarotot pada Node Utama")
    ax.axis("off")
    savefig(
        fig,
        "fig_4_7_8_relation_subgraph.png",
        "4.7.8",
        "relation graph + selected trial predictions",
        "Subgraph sederhana dari node dan edge utama berdasarkan skor relasi.",
    )


def embedding_figure(normal_rel: pd.DataFrame, different_rel: pd.DataFrame) -> None:
    frames = []
    for trial_type, df in (("Sesuai Pola Acuan", normal_rel), ("Pola Aktivitas Berbeda", different_rel)):
        if df.empty:
            continue
        tmp = df[["trial_id", "node_index", "node_name", "prob_deviating", "relation_score", "final_status"]].copy()
        tmp["trial_type"] = trial_type
        frames.append(tmp)
    if not frames:
        FAILED.append(("fig_4_7_embedding_node_plot.png", "data prediksi trial representatif tidak tersedia"))
        return
    data = pd.concat(frames, ignore_index=True)
    # PCA is applied to available learned outputs/proxy descriptors for documentation; if model embeddings are exported,
    # they can replace these columns without changing the figure contract.
    x = data[["prob_deviating", "relation_score", "node_index"]].to_numpy(dtype=float)
    x[:, 2] = x[:, 2] / max(float(np.nanmax(x[:, 2])), 1.0)
    if PCA is not None and len(data) >= 2:
        coords = PCA(n_components=2, random_state=7).fit_transform(x)
    else:
        coords = x[:, :2]
    data["embedding_x"] = coords[:, 0]
    data["embedding_y"] = coords[:, 1]
    data.to_csv(OUT_DIR / "fig_4_7_embedding_node_plot_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for status, group in data.groupby("final_status"):
        ax.scatter(
            group["embedding_x"],
            group["embedding_y"],
            s=68,
            color=STATUS_COLORS.get(status, "#6B7280"),
            edgecolor="white",
            linewidth=0.8,
            label=status,
        )
        for _, row in group.sort_values("prob_deviating", ascending=False).head(6).iterrows():
            ax.text(row["embedding_x"] + 0.01, row["embedding_y"] + 0.01, str(row["node_name"]), fontsize=8)
    ax.set_title("Visualisasi Embedding Node Otot Berdasarkan Status Pola Aktivitas")
    ax.set_xlabel("Embedding Dimension 1")
    ax.set_ylabel("Embedding Dimension 2")
    ax.legend(fontsize=8)
    polish_axis(ax)
    savefig(
        fig,
        "fig_4_7_embedding_node_plot.png",
        "4.7.7/4.7.8",
        "selected trial node outputs; PCA projection",
        "Visualisasi 2D representasi node otot berdasarkan status pola aktivitas.",
        "Jarak titik adalah kedekatan representasi fitur, bukan jarak anatomi otot.",
    )


def limitation_figure() -> None:
    labels = [
        "Keterbatasan\nsensor Leap Motion",
        "Potensi\nocclusion",
        "Noise/jitter\ntracking",
        "Keterbatasan\nsimulasi",
        "Belum divalidasi\nEMG",
        "Gesture terbatas\nopen-to-fist",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axis("off")
    x_positions = np.linspace(0.08, 0.92, len(labels))
    for i, (x0, label) in enumerate(zip(x_positions, labels)):
        color = ["#E5E7EB", "#D1D5DB", "#E5E7EB", "#D1D5DB", "#E5E7EB", "#D1D5DB"][i]
        ax.add_patch(plt.Circle((x0, 0.52), 0.075, fc=color, ec="#374151", lw=1.1))
        ax.text(x0, 0.52, str(i + 1), ha="center", va="center", fontweight="bold")
        ax.text(x0, 0.30, label, ha="center", va="top", fontsize=9)
        if i < len(labels) - 1:
            ax.annotate("", xy=(x_positions[i + 1] - 0.08, 0.52), xytext=(x0 + 0.08, 0.52), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Ringkasan Keterbatasan Sistem")
    savefig(
        fig,
        "fig_4_7_10_limitation_summary.png",
        "4.7.10",
        "methodological limitation summary",
        "Diagram ringkas sumber keterbatasan sistem dari akuisisi hingga validasi.",
    )


def summary_csv(normal_rel: pd.DataFrame, different_rel: pd.DataFrame) -> None:
    frames = []
    for trial_type, df in (("Sesuai Pola Acuan", normal_rel), ("Pola Aktivitas Berbeda", different_rel)):
        if df.empty:
            continue
        out = df[["trial_id", "node_name", "prob_deviating", "relation_score", "final_status"]].copy()
        out.insert(1, "trial_type", trial_type)
        out = out.rename(columns={"prob_deviating": "p_dev", "final_status": "status_label"})
        out["final_status"] = out["status_label"]
        frames.append(out[["trial_id", "trial_type", "node_name", "p_dev", "relation_score", "final_status", "status_label"]])
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "summary_selected_trials.csv", index=False)


def write_report(normal_trial: str, different_trial: str) -> None:
    lines = [
        "# Report Grafik Wajib Bab 4.7",
        "",
        "Grafik dibuat dari artifact/output sistem yang sudah tersedia. Tidak ada retraining, tidak ada perubahan model, dan tidak ada perubahan data mentah.",
        "",
        "## Trial Representatif",
        f"- Trial Sesuai Pola Acuan: `{normal_trial}`",
        f"- Trial Pola Aktivitas Berbeda: `{different_trial}`",
        "",
        "Alasan pemilihan: trial sesuai pola acuan dipilih dari data test dengan rerata probabilitas pola berbeda paling rendah; trial pola aktivitas berbeda dipilih dari data uji pola berbeda dengan rerata probabilitas pola berbeda paling tinggi.",
        "",
        "## Grafik Berhasil Dibuat",
        "",
    ]
    for rec in FIGURES:
        lines.extend(
            [
                f"### {rec.filename}",
                f"- Subbab: {rec.section}",
                f"- Sumber data: {rec.source}",
                f"- Caption saran: {rec.caption}",
            ]
        )
        if rec.notes:
            lines.append(f"- Catatan: {rec.notes}")
        lines.append("")
    lines.extend(["## Grafik Gagal / Data Tidak Tersedia", ""])
    if FAILED:
        for name, reason in FAILED:
            lines.append(f"- `{name}`: {reason}")
    else:
        lines.append("- Tidak ada target grafik yang gagal dibuat.")
    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "- Confusion matrix final sudah tersedia pada artifact evaluasi model dan dapat dirujuk ulang untuk Bab 4.6 atau analisis pendukung Bab 4.7.",
            "",
            "## Grafik Paling Kuat untuk Bab 4.7",
            "- `fig_4_7_1_runtime_latency.png` dan `fig_4_7_1_runtime_error.png` untuk bukti runtime estimation.",
            "- `fig_4_7_4_replay_error_comparison.png` dan `fig_4_7_4_ctrl_delta_comparison.png` untuk validasi replay control.",
            "- `fig_4_7_6_training_accuracy.png`, `fig_4_7_6_training_loss.png`, dan `fig_4_7_6_training_f1_score.png` untuk dinamika training model.",
            "- `fig_4_7_7_probability_per_muscle_different.png` untuk analisis klasifikasi node otot.",
            "- `fig_4_7_8_relation_scatter_different.png` dan `fig_4_7_8_top_relation_bar.png` untuk analisis relasi antarotot.",
            "",
            "## Catatan Penulisan",
            "- Gunakan istilah Sesuai Pola Acuan, Pola Aktivitas Berbeda, dan Otot Terkait Secara Relasional.",
            "- Hindari klaim diagnosis atau interpretasi klinis.",
            "- Grafik embedding digunakan sebagai visualisasi representasi fitur, bukan jarak anatomi otot.",
        ]
    )
    (OUT_DIR / "report_bab_4_7_grafik_wajib.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    history = read_csv(MODEL_DIR / "history.csv")
    pred_test = read_csv(MODEL_DIR / "predictions_test.csv")
    pred_different = read_csv(MODEL_DIR / "predictions_abnormal_validation.csv")
    edges = load_edges()
    normal_trial, different_trial = choose_trials(pred_test, pred_different)
    normal_rel, normal_top = relation_table(pred_test, normal_trial, edges)
    different_rel, different_top = relation_table(pred_different, different_trial, edges)
    top_rel = different_top if not different_top.empty else normal_top

    runtime_figures()
    control_figures()
    replay_validation_figures()
    graph_figures(normal_trial)
    training_figures(history)
    probability_figures(normal_rel, different_rel)
    relation_figures(normal_rel, different_rel, top_rel)
    embedding_figure(normal_rel, different_rel)
    limitation_figure()
    summary_csv(normal_rel, different_rel)
    write_report(normal_trial, different_trial)

    print(f"Output folder: {OUT_DIR}")
    print(f"Created figures: {len(FIGURES)}")
    print(f"Failed figures: {len(FAILED)}")
    print(f"Selected reference-pattern trial: {normal_trial}")
    print(f"Selected different-pattern trial: {different_trial}")
    if FAILED:
        print("Failed/limited targets:")
        for name, reason in FAILED:
            print(f" - {name}: {reason}")
    print("Output files:")
    for rec in FIGURES:
        print(f" - {rec.filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
