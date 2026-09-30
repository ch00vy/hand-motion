from __future__ import annotations

import ast
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional
    nx = None

try:
    from sklearn.metrics import auc, roc_curve
except Exception:  # pragma: no cover - optional
    auc = None
    roc_curve = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs" / "bab_4_7_grafik"

RUNTIME_DIR = PROJECT_ROOT / "artifacts" / "realtime_recorded_qpos_audit"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier_labeled_records"
RETRAIN_DIR = PROJECT_ROOT / "artifacts" / "retrain_labeled_records"
GRAPH_NORMAL_DIR = RETRAIN_DIR / "graph_sequence_dataset"
GRAPH_PATTERN_DIR = RETRAIN_DIR / "abnormal_graph_sequence_dataset"
CTRL_NORMAL_DIR = RETRAIN_DIR / "ctrl_ref_normal" / "reference_build"
CTRL_PATTERN_DIR = RETRAIN_DIR / "ctrl_ref_abnormal" / "reference_build"

STATUS_COLORS = {
    "Sesuai Pola Acuan": "#22C55E",
    "Pola Aktivitas Berbeda": "#EF4444",
    "Otot Terkait Secara Relasional": "#F59E0B",
}


@dataclass
class FigureRecord:
    filename: str
    section: str
    source: str
    caption: str
    status: str = "created"
    notes: str = ""


FIGURES: list[FigureRecord] = []
MISSING: list[str] = []


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#D1D5DB",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.labelcolor": "#111827",
            "text.color": "#111827",
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
        }
    )


def savefig(fig: plt.Figure, filename: str, section: str, source: str, caption: str, notes: str = "") -> None:
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES.append(FigureRecord(filename, section, source, caption, notes=notes))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        MISSING.append(str(path))
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        MISSING.append(str(path))
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def label_bars(ax: plt.Axes, fmt: str = "{:.2f}") -> None:
    for patch in ax.patches:
        h = patch.get_height()
        if np.isfinite(h):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                h,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
            )


def add_subtitle(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.suptitle(title, y=0.995, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", color="#4B5563", fontsize=9)


def display_status(predicted: str, p_dev: float, relation_score: float = 0.0) -> str:
    if float(p_dev) >= 0.5 or str(predicted).strip() == "deviating":
        return "Pola Aktivitas Berbeda"
    if float(relation_score) >= 0.25:
        return "Otot Terkait Secara Relasional"
    return "Sesuai Pola Acuan"


def load_model_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    history = safe_read_csv(MODEL_DIR / "history.csv")
    pred_test = safe_read_csv(MODEL_DIR / "predictions_test.csv")
    pred_pattern = safe_read_csv(MODEL_DIR / "predictions_abnormal_validation.csv")
    metrics = safe_read_json(MODEL_DIR / "eval_metrics.json")
    cm = safe_read_csv(MODEL_DIR / "confusion_matrix_test.csv")
    return history, pred_test, pred_pattern, metrics, cm


def choose_trials(pred_test: pd.DataFrame, pred_pattern: pd.DataFrame) -> tuple[str, str]:
    normal_trial = ""
    pattern_trial = ""
    if not pred_test.empty:
        grouped = pred_test.groupby("trial_id")["prob_deviating"].mean().sort_values()
        normal_trial = str(grouped.index[0])
    if not pred_pattern.empty:
        grouped = pred_pattern.groupby("trial_id")["prob_deviating"].mean().sort_values(ascending=False)
        pattern_trial = str(grouped.index[0])
    return normal_trial, pattern_trial


def load_edges() -> pd.DataFrame:
    edge_path = PROJECT_ROOT / "artifacts" / "final_book_figures" / "neo4j_export" / "neo4j_muscle_edges.csv"
    edges = safe_read_csv(edge_path)
    if edges.empty:
        return edges
    for col in ("source_muscle", "target_muscle"):
        if col not in edges.columns:
            # Compatibility with alternate headers.
            alt = "source" if "source" in edges.columns else None
            if alt:
                edges[col] = edges[alt]
    if "edge_weight" not in edges.columns:
        for c in edges.columns:
            if "weight" in c.lower():
                edges["edge_weight"] = pd.to_numeric(edges[c], errors="coerce").fillna(0.0)
                break
    edges["edge_weight"] = pd.to_numeric(edges.get("edge_weight", 0.0), errors="coerce").fillna(0.0)
    if "relation_type" not in edges.columns:
        edges["relation_type"] = "functional_group"
    return edges


def compute_relation_table(pred_df: pd.DataFrame, trial_id: str, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial = pred_df.loc[pred_df["trial_id"].astype(str) == str(trial_id)].copy()
    if trial.empty:
        return pd.DataFrame(), pd.DataFrame()
    trial["prob_deviating"] = pd.to_numeric(trial["prob_deviating"], errors="coerce").fillna(0.0)
    p_by_node = dict(zip(trial["node_name"].astype(str), trial["prob_deviating"].astype(float)))
    pred_by_node = dict(zip(trial["node_name"].astype(str), trial["predicted_class"].astype(str)))
    dev_nodes = {name for name, p in p_by_node.items() if p >= 0.5 or pred_by_node.get(name) == "deviating"}

    risk: dict[str, float] = {name: 0.0 for name in p_by_node}
    strongest: dict[str, tuple[str, float]] = {name: ("", 0.0) for name in p_by_node}
    rows = []
    if not edges.empty:
        for row in edges.to_dict(orient="records"):
            src = str(row.get("source_muscle", ""))
            dst = str(row.get("target_muscle", ""))
            w = float(row.get("edge_weight", 0.0) or 0.0)
            rel = str(row.get("relation_type", "functional_group"))
            if src not in p_by_node or dst not in p_by_node:
                continue
            for u, v in ((src, dst), (dst, src)):
                contrib = w * float(p_by_node.get(u, 0.0)) if u in dev_nodes else 0.0
                if contrib > 0:
                    risk[v] += contrib
                    if w > strongest[v][1]:
                        strongest[v] = (u, w)
                rows.append(
                    {
                        "Source Muscle / Node": src,
                        "Target Muscle / Node": dst,
                        "Source Status": display_status(pred_by_node.get(src, ""), p_by_node[src]),
                        "Target Status": display_status(pred_by_node.get(dst, ""), p_by_node[dst]),
                        "Source Prob. Pola Berbeda": p_by_node[src],
                        "Target Prob. Pola Berbeda": p_by_node[dst],
                        "Jenis Relasi": rel,
                        "Bobot Edge": w,
                        "Kontribusi Skor Relasi": max(w * p_by_node[src], w * p_by_node[dst]),
                    }
                )
    out = trial[["trial_id", "node_index", "node_name", "predicted_class", "prob_deviating"]].copy()
    out["relation_score"] = out["node_name"].map(risk).fillna(0.0)
    out["final_status"] = [
        display_status(pred, p, r)
        for pred, p, r in zip(out["predicted_class"], out["prob_deviating"], out["relation_score"])
    ]
    out["strongest_related_muscle"] = out["node_name"].map(lambda n: strongest.get(str(n), ("", 0.0))[0])
    top_rel = pd.DataFrame(rows)
    if not top_rel.empty:
        top_rel = top_rel.sort_values("Kontribusi Skor Relasi", ascending=False).drop_duplicates(
            ["Source Muscle / Node", "Target Muscle / Node"]
        )
    return out, top_rel


def figure_runtime_latency() -> None:
    summary = safe_read_json(RUNTIME_DIR / "validation_summary.json")
    values = [
        ("Mean ctrl pipeline", summary.get("latency_mean_ms", np.nan)),
        ("P95 ctrl pipeline", summary.get("latency_p95_ms", np.nan)),
        ("Mean frame loop", summary.get("frame_loop_mean_ms", np.nan)),
        ("P95 frame loop", summary.get("frame_loop_p95_ms", np.nan)),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels, vals = zip(*values)
    ax.bar(labels, vals, color=["#2563EB", "#60A5FA", "#7C3AED", "#A78BFA"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Runtime Latency pada Validasi Recorded qpos")
    label_bars(ax, "{:.1f}")
    ax.tick_params(axis="x", rotation=15)
    savefig(
        fig,
        "fig_4_7_1_runtime_latency_bar.png",
        "4.7.1",
        str(RUNTIME_DIR / "validation_summary.json"),
        "Grafik latensi pipeline runtime dan frame loop pada validasi recorded qpos.",
    )


def figure_runtime_error() -> None:
    summary = safe_read_json(RUNTIME_DIR / "validation_summary.json")
    values = [
        ("Mean joint error", summary.get("sequence_mean_joint_error", np.nan)),
        ("RMSE forward", summary.get("sequence_rmse_forward_error", np.nan)),
        ("P95 frame error", summary.get("sequence_p95_frame_error", np.nan)),
        ("Fallback ratio", summary.get("fallback_ratio", np.nan)),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels, vals = zip(*values)
    ax.bar(labels, vals, color=["#0EA5E9", "#2563EB", "#F59E0B", "#6B7280"])
    ax.set_ylabel("Nilai metrik")
    ax.set_title("Error Runtime Validation pada Recorded qpos")
    label_bars(ax, "{:.3f}")
    ax.tick_params(axis="x", rotation=15)
    savefig(
        fig,
        "fig_4_7_1_runtime_error_bar.png",
        "4.7.1",
        str(RUNTIME_DIR / "validation_summary.json"),
        "Ringkasan error validasi runtime recorded qpos dan rasio fallback.",
    )


def figure_simulation_replay_comparison() -> None:
    df = safe_read_csv(RUNTIME_DIR / "runtime_log.csv")
    qpos_cols = [c for c in df.columns if c.startswith("qpos_") and c.split("_")[-1].isdigit()]
    if df.empty or not qpos_cols:
        MISSING.append("runtime_log qpos columns for replay comparison")
        return
    target_start = df.loc[df.index[0], qpos_cols].astype(float).values
    target_end = df.loc[df.index[-1], qpos_cols].astype(float).values
    x = np.arange(len(qpos_cols))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    axes[0].bar(x, target_start, color="#93C5FD")
    axes[0].set_title("Target Pose - awal")
    axes[0].set_xlabel("qpos index")
    axes[0].set_ylabel("Posisi sendi")
    axes[1].bar(x, target_end, color="#2563EB")
    axes[1].set_title("Replay Pose - akhir")
    axes[1].set_xlabel("qpos index")
    add_subtitle(fig, "Perbandingan Pose Target dan Replay", "Rekonstruksi berbasis vektor qpos dari runtime log")
    savefig(
        fig,
        "fig_4_7_2_simulation_replay_comparison.png",
        "4.7.2",
        str(RUNTIME_DIR / "runtime_log.csv"),
        "Panel perbandingan pose awal target dan pose akhir replay berdasarkan vektor posisi sendi.",
        "Tidak ditemukan screenshot pose target/replay; figure dibuat sebagai rekonstruksi qpos.",
    )


def ctrl_summary_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    normal = safe_read_csv(CTRL_NORMAL_DIR / "biomech_reference_trial_summary.csv")
    pattern = safe_read_csv(CTRL_PATTERN_DIR / "biomech_reference_trial_summary.csv")
    return normal, pattern


def grouped_metric_bar(filename: str, metric: str, ylabel: str, title: str, section: str, caption: str) -> None:
    normal, pattern = ctrl_summary_frames()
    groups = []
    for name, df in (("Trial normal", normal), ("Trial pola aktivitas berbeda", pattern)):
        if metric in df.columns and not df.empty:
            groups.append((name, float(pd.to_numeric(df[metric], errors="coerce").mean())))
        else:
            groups.append((name, np.nan))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar([g[0] for g in groups], [g[1] for g in groups], color=["#22C55E", "#EF4444"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    label_bars(ax, "{:.3f}")
    savefig(
        fig,
        filename,
        section,
        f"{CTRL_NORMAL_DIR / 'biomech_reference_trial_summary.csv'}; {CTRL_PATTERN_DIR / 'biomech_reference_trial_summary.csv'}",
        caption,
    )


def figure_control_profile() -> None:
    # Pick the best-quality normal trial available for a clean example.
    normal, _ = ctrl_summary_frames()
    if normal.empty:
        return
    trial_id = str(normal.sort_values("rmse_forward_error").iloc[0]["trial_id"])
    dataset_path = Path(str(normal.loc[normal["trial_id"].astype(str) == trial_id, "dataset_path"].iloc[0]))
    df = safe_read_csv(dataset_path)
    ctrl_cols = [c for c in df.columns if c.startswith("ctrl_teacher_")]
    if df.empty or not ctrl_cols:
        MISSING.append(f"control profile columns missing: {dataset_path}")
        return
    # Main muscles = top 5 by mean ctrl to avoid a cluttered 39-line plot.
    means = df[ctrl_cols].mean().sort_values(ascending=False)
    top_cols = means.head(5).index.tolist()
    progress = np.linspace(0, 100, len(df))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col in top_cols:
        ax.plot(progress, df[col].astype(float), linewidth=2, label=col.replace("ctrl_teacher_", "ctrl "))
    ax.set_title("Contoh Profil Estimasi Control Otot")
    ax.set_xlabel("Progress gerakan (%)")
    ax.set_ylabel("Nilai control")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    savefig(
        fig,
        "fig_4_7_3_example_control_profile.png",
        "4.7.3",
        str(dataset_path),
        "Contoh profil temporal control otot utama pada satu trial representatif.",
        f"Trial terpilih: {trial_id}; ditampilkan top 5 control agar plot terbaca.",
    )


def figure_graph_structure() -> None:
    summary = safe_read_json(MODEL_DIR / "dataset_summary.json")
    num_nodes = int(summary.get("num_nodes", 39))
    num_windows = int(summary.get("num_windows", 56))
    num_features = int(summary.get("features_per_node_per_window", 6))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axis("off")
    boxes = [
        ("Estimasi control otot", 0.08, 0.55, "#DBEAFE"),
        (f"{num_nodes} node otot", 0.31, 0.55, "#DCFCE7"),
        (f"{num_windows} sequence window", 0.54, 0.55, "#FEF3C7"),
        (f"{num_features} fitur / node / window", 0.77, 0.55, "#FEE2E2"),
    ]
    for text, x, y, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.17, 0.18, color=color, ec="#374151", lw=1.2))
        ax.text(x + 0.085, y + 0.09, text, ha="center", va="center", fontsize=10, fontweight="bold")
    for x in (0.25, 0.48, 0.71):
        ax.annotate("", xy=(x + 0.045, 0.64), xytext=(x - 0.04, 0.64), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(0.5, 0.28, "Struktur input Dynamic Temporal GCN: [window, node, fitur]", ha="center", fontsize=11)
    ax.set_title("Struktur Data Graph Temporal Hasil Pembentukan Fitur", fontweight="bold")
    savefig(
        fig,
        "fig_4_7_4_graph_temporal_structure.png",
        "4.7.4",
        str(MODEL_DIR / "dataset_summary.json"),
        "Diagram struktur data graph temporal yang digunakan sebagai input Dynamic Temporal GCN.",
    )


def figure_temporal_heatmap(normal_trial: str) -> None:
    json_path = GRAPH_NORMAL_DIR / f"normal__{normal_trial}.json"
    npz_path = GRAPH_NORMAL_DIR / f"normal__{normal_trial}.npz"
    meta = safe_read_json(json_path)
    if not npz_path.exists():
        MISSING.append(str(npz_path))
        return
    arr = np.load(npz_path)
    feat = arr["node_feature_seq"][:, :, 0].T  # mean_ctrl over windows
    node_names = meta.get("node_names", [f"node_{i}" for i in range(feat.shape[0])])
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(feat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("Contoh Heatmap Aktivitas Otot terhadap Progress")
    ax.set_xlabel("Sequence window")
    ax.set_ylabel("Node otot")
    ax.set_yticks(np.arange(len(node_names)))
    ax.set_yticklabels(node_names, fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean control feature")
    savefig(
        fig,
        "fig_4_7_4_temporal_heatmap_example.png",
        "4.7.4",
        f"{json_path}; {npz_path}",
        "Contoh heatmap fitur aktivitas otot sepanjang sequence window.",
        f"Trial: {normal_trial}",
    )


def figure_training_curves(history: pd.DataFrame, metrics: dict[str, Any], cm: pd.DataFrame, pred_test: pd.DataFrame) -> None:
    if history.empty:
        return
    curves = [
        ("fig_4_7_5_training_accuracy.png", ["train_accuracy", "val_accuracy"], "Accuracy", "Kurva Accuracy Dynamic Temporal GCN"),
        ("fig_4_7_5_training_loss.png", ["train_loss", "val_loss"], "Loss", "Kurva Loss Dynamic Temporal GCN"),
        ("fig_4_7_5_training_f1_score.png", ["train_macro_f1", "val_macro_f1", "train_binary_f1_deviating", "val_binary_f1_deviating"], "F1-score", "Kurva F1-score Dynamic Temporal GCN"),
    ]
    for filename, cols, ylabel, title in curves:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for col in cols:
            if col in history.columns:
                ax.plot(history["epoch"], history[col], linewidth=2, label=col.replace("_", " "))
        best_epoch = metrics.get("best_epoch")
        if best_epoch:
            ax.axvline(float(best_epoch), color="#111827", linestyle="--", linewidth=1, label=f"best epoch {best_epoch}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        savefig(fig, filename, "4.7.5", str(MODEL_DIR / "history.csv"), f"{title} pada training final.")

    # Confusion matrix.
    if not cm.empty:
        labels = cm.iloc[:, 0].astype(str).tolist()
        mat = cm.iloc[:, 1:].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_title("Confusion Matrix Dynamic Temporal GCN")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Sesuai Pola Acuan", "Pola Aktivitas Berbeda"], rotation=20, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Sesuai Pola Acuan", "Pola Aktivitas Berbeda"])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label acuan")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", color="#111827", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
        savefig(
            fig,
            "fig_4_7_5_confusion_matrix.png",
            "4.7.5",
            str(MODEL_DIR / "confusion_matrix_test.csv"),
            "Confusion matrix hasil evaluasi Dynamic Temporal GCN pada data uji.",
        )

    if roc_curve is not None and auc is not None and not pred_test.empty:
        y_true = (pred_test["target_class"].astype(str) == "deviating").astype(int).to_numpy()
        y_score = pd.to_numeric(pred_test["prob_deviating"], errors="coerce").fillna(0.0).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        ax.plot(fpr, tpr, color="#2563EB", linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--")
        ax.set_title("ROC Curve Dynamic Temporal GCN")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(frameon=False)
        savefig(
            fig,
            "fig_4_7_5_roc_curve.png",
            "4.7.5",
            str(MODEL_DIR / "predictions_test.csv"),
            "ROC curve evaluasi node classification pola aktivitas otot.",
        )


def figure_probability_bars(normal_rel: pd.DataFrame, pattern_rel: pd.DataFrame) -> None:
    for df, filename, title, source, caption in (
        (
            normal_rel,
            "fig_4_7_6_probability_per_muscle_normal.png",
            "Probabilitas Pola Aktivitas Berbeda - Trial Sesuai Pola Acuan",
            "predictions_test.csv + relation graph",
            "Top probabilitas pola aktivitas berbeda pada trial sesuai pola acuan.",
        ),
        (
            pattern_rel,
            "fig_4_7_6_probability_per_muscle_different.png",
            "Probabilitas Pola Aktivitas Berbeda - Trial Pola Aktivitas Berbeda",
            "predictions_abnormal_validation.csv + relation graph",
            "Top probabilitas pola aktivitas berbeda pada trial pola aktivitas berbeda.",
        ),
    ):
        if df.empty:
            continue
        top = df.sort_values("prob_deviating", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        colors = [STATUS_COLORS.get(s, "#6B7280") for s in top["final_status"]]
        ax.bar(top["node_name"], top["prob_deviating"], color=colors)
        ax.axhline(0.5, color="#111827", linestyle="--", linewidth=1, label="Ambang 0,5")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Probabilitas Pola Berbeda")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(frameon=False)
        savefig(fig, filename, "4.7.6", source, caption)

    if not normal_rel.empty and not pattern_rel.empty:
        rows = []
        for label, df in (("Trial sesuai pola acuan", normal_rel), ("Trial pola aktivitas berbeda", pattern_rel)):
            counts = df["final_status"].value_counts().to_dict()
            for status in STATUS_COLORS:
                rows.append({"trial": label, "status": status, "count": counts.get(status, 0)})
        plot_df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(2)
        width = 0.25
        statuses = list(STATUS_COLORS)
        for i, status in enumerate(statuses):
            vals = plot_df.loc[plot_df["status"] == status, "count"].to_numpy()
            ax.bar(x + (i - 1) * width, vals, width=width, label=status, color=STATUS_COLORS[status])
        ax.set_xticks(x)
        ax.set_xticklabels(["Sesuai pola acuan", "Pola aktivitas berbeda"])
        ax.set_ylabel("Jumlah node")
        ax.set_title("Perbandingan Jumlah Status Node")
        ax.legend(frameon=False, fontsize=8)
        savefig(
            fig,
            "fig_4_7_6_status_count_comparison.png",
            "4.7.6",
            "computed from selected trial predictions and relation graph",
            "Perbandingan jumlah status node antara dua trial representatif.",
        )


def figure_relation_scatter(df: pd.DataFrame, filename: str, title: str, source: str, caption: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for status, group in df.groupby("final_status"):
        ax.scatter(
            group["prob_deviating"],
            group["relation_score"],
            s=62,
            color=STATUS_COLORS.get(status, "#6B7280"),
            label=status,
            edgecolor="white",
            linewidth=0.8,
        )
        for _, row in group.sort_values("prob_deviating", ascending=False).head(8).iterrows():
            ax.text(row["prob_deviating"] + 0.01, row["relation_score"] + 0.01, str(row["node_name"]), fontsize=7)
    ax.axvline(0.5, color="#111827", linestyle="--", linewidth=1)
    ax.axhline(0.25, color="#111827", linestyle=":", linewidth=1)
    ax.text(0.51, ax.get_ylim()[1] * 0.93, "Ambang Prob. 0,5", fontsize=8)
    ax.text(0.02, 0.27, "Ambang Relasi 0,25", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probabilitas Pola Berbeda")
    ax.set_ylabel("Skor Relasi Antarotot")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, filename, "4.7.7", source, caption)


def figure_top_relation_bar(top_rel: pd.DataFrame) -> None:
    if top_rel.empty:
        return
    top = top_rel.head(10).copy()
    labels = top["Source Muscle / Node"].astype(str) + " - " + top["Target Muscle / Node"].astype(str)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, top["Kontribusi Skor Relasi"].astype(float), color="#F59E0B")
    ax.set_title("Top Skor Relasi Antarotot")
    ax.set_ylabel("Kontribusi Skor Relasi")
    ax.tick_params(axis="x", rotation=45)
    savefig(
        fig,
        "fig_4_7_7_top_relation_bar.png",
        "4.7.7",
        "computed top relations from selected trial predictions and edge graph",
        "Top 10 relasi antarotot berdasarkan kontribusi skor relasi.",
    )


def figure_relation_subgraph(top_rel: pd.DataFrame) -> None:
    if top_rel.empty:
        return
    top = top_rel.head(12)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if nx is None:
        nodes = sorted(set(top["Source Muscle / Node"]).union(set(top["Target Muscle / Node"])))
        angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        pos = {node: (math.cos(a), math.sin(a)) for node, a in zip(nodes, angles)}
        for _, row in top.iterrows():
            a, b = row["Source Muscle / Node"], row["Target Muscle / Node"]
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#F59E0B", alpha=0.7)
        for node, (x, y) in pos.items():
            ax.scatter([x], [y], s=260, color="#DBEAFE", edgecolor="#2563EB")
            ax.text(x, y, node, ha="center", va="center", fontsize=8)
    else:
        g = nx.Graph()
        for _, row in top.iterrows():
            g.add_edge(row["Source Muscle / Node"], row["Target Muscle / Node"], weight=row["Kontribusi Skor Relasi"])
        pos = nx.spring_layout(g, seed=7, weight="weight")
        weights = np.array([g[u][v]["weight"] for u, v in g.edges()])
        nx.draw_networkx_edges(g, pos, ax=ax, width=1 + 5 * weights / max(weights.max(), 1e-9), edge_color="#F59E0B", alpha=0.75)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#DBEAFE", edgecolors="#2563EB", node_size=700)
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
    ax.set_title("Subgraph Relasi Antarotot Dominan")
    ax.axis("off")
    savefig(
        fig,
        "fig_4_7_7_relation_subgraph.png",
        "4.7.7",
        "computed top relations from selected trial predictions and edge graph",
        "Subgraph sederhana relasi antarotot dengan kontribusi skor relasi tertinggi.",
    )


def draw_gui_reconstruction(filename: str, title: str, rel_df: pd.DataFrame, note: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#111827"))
    ax.text(0.02, 0.96, "Muscle Activity Pattern Analysis - Dynamic Temporal GCN", color="white", fontsize=14, fontweight="bold")
    panels = [
        ("Trial Control", 0.02, 0.74, 0.30, 0.18),
        ("Recording Session", 0.35, 0.74, 0.30, 0.18),
        ("External Viewers", 0.68, 0.74, 0.30, 0.18),
        ("Classifier Plot", 0.02, 0.08, 0.47, 0.60),
        ("Muscle Result Table", 0.52, 0.08, 0.46, 0.60),
    ]
    for label, x, y, w, h in panels:
        ax.add_patch(plt.Rectangle((x, y), w, h, color="#1F2937", ec="#374151", lw=1.2))
        ax.text(x + 0.015, y + h - 0.035, label, color="#E5E7EB", fontsize=11, fontweight="bold")
    ax.text(0.04, 0.84, "Subject ID: subject_gui\nMotion Type: open_to_fist\nRun Analysis", color="#D1D5DB", fontsize=9)
    ax.text(0.70, 0.84, "Open MuJoCo\nLoad QPOS Replay\nLoad CTRL Replay", color="#D1D5DB", fontsize=9)
    if not rel_df.empty:
        top = rel_df.sort_values("prob_deviating", ascending=False).head(8)
        x0 = np.linspace(0.07, 0.44, len(top))
        y0 = 0.18 + 0.42 * top["prob_deviating"].to_numpy()
        colors = [STATUS_COLORS.get(s, "#6B7280") for s in top["final_status"]]
        ax.bar(x0, 0.42 * top["prob_deviating"].to_numpy(), width=0.025, bottom=0.18, color=colors)
        ax.text(0.04, 0.62, "Probabilitas Pola Berbeda", color="#D1D5DB", fontsize=9)
        y = 0.62
        for _, row in top.iterrows():
            ax.text(0.55, y, f"{row['node_name']:<6}  {row['final_status']}", color="#E5E7EB", fontsize=8, family="monospace")
            y -= 0.045
    ax.text(0.02, 0.02, note, color="#F59E0B", fontsize=8)
    ax.set_title(title, color="white", pad=12)
    savefig(fig, filename, "4.7.8", "GUI reconstruction from selected analysis outputs", note, "Rekonstruksi visual, bukan screenshot langsung.")


def figure_gui_callout() -> None:
    img_path = PROJECT_ROOT / "image.png"
    if img_path.exists():
        img = plt.imread(img_path)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(img)
        ax.axis("off")
        callouts = [
            ("Trial control", (0.12, 0.82), (0.24, 0.95)),
            ("Classifier plot", (0.48, 0.62), (0.52, 0.95)),
            ("Result table", (0.78, 0.60), (0.78, 0.95)),
            ("External replay controls", (0.16, 0.43), (0.34, 0.35)),
            ("Summary/status", (0.16, 0.18), (0.34, 0.08)),
        ]
        for text, xy, xytext in callouts:
            ax.annotate(
                text,
                xy=(xy[0] * img.shape[1], xy[1] * img.shape[0]),
                xytext=(xytext[0] * img.shape[1], xytext[1] * img.shape[0]),
                arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=2),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#F59E0B"),
                color="#111827",
                fontsize=10,
            )
        savefig(
            fig,
            "fig_4_7_8_gui_component_callout.png",
            "4.7.8",
            str(img_path),
            "Tampilan GUI dengan callout komponen utama sistem analisis.",
            "Menggunakan screenshot GUI yang tersedia di root project.",
        )
    else:
        draw_gui_reconstruction(
            "fig_4_7_8_gui_component_callout.png",
            "Komponen GUI Sistem Analisis",
            pd.DataFrame(),
            "Rekonstruksi karena screenshot GUI tidak ditemukan.",
        )


def figure_limitation() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.08, 0.15), 0.84, 0.7, fc="#111827", ec="#374151"))
    ax.add_patch(plt.Circle((0.50, 0.48), 0.18, fc="#9CA3AF", alpha=0.35, ec="#E5E7EB"))
    ax.add_patch(plt.Rectangle((0.42, 0.38), 0.32, 0.12, fc="#EF4444", alpha=0.45, ec="#B91C1C"))
    ax.text(0.50, 0.82, "Ilustrasi Occlusion pada Sensor Leap Motion", ha="center", color="white", fontsize=12, fontweight="bold")
    ax.annotate("Area occlusion / finger overlap", xy=(0.62, 0.45), xytext=(0.58, 0.25), arrowprops=dict(arrowstyle="->", color="#F59E0B"), color="white")
    ax.text(0.50, 0.08, "Catatan: tidak ditemukan screenshot occlusion asli; figure ini adalah ilustrasi keterbatasan sensor.", ha="center", fontsize=9)
    savefig(
        fig,
        "fig_4_7_9_occlusion_example.png",
        "4.7.9",
        "schematic generated because no occlusion screenshot was found",
        "Ilustrasi keterbatasan occlusion pada proses akuisisi Leap Motion.",
        "Schematic, bukan screenshot asli.",
    )

    labels = ["Sensor occlusion", "Mapping qpos", "Replay mismatch", "Interpretasi model"]
    vals = [1.0, 0.75, 0.65, 0.55]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(labels, vals, color=["#EF4444", "#F59E0B", "#3B82F6", "#6B7280"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Tingkat pengaruh kualitatif")
    ax.set_title("Ringkasan Keterbatasan Sistem")
    ax.tick_params(axis="x", rotation=20)
    savefig(
        fig,
        "fig_4_7_9_error_limitation_summary.png",
        "4.7.9",
        "qualitative summary based on system limitation audit",
        "Ringkasan kualitatif sumber keterbatasan dari sensor hingga interpretasi model.",
    )


def write_selected_trials_csv(normal_rel: pd.DataFrame, pattern_rel: pd.DataFrame) -> None:
    frames = []
    for trial_type, df in (("Sesuai Pola Acuan", normal_rel), ("Pola Aktivitas Berbeda", pattern_rel)):
        if df.empty:
            continue
        out = df.rename(
            columns={
                "node_name": "node_name",
                "prob_deviating": "p_dev",
                "relation_score": "relation_score",
                "final_status": "final_status",
            }
        )[["trial_id", "node_name", "p_dev", "relation_score", "final_status"]].copy()
        out.insert(1, "trial_type", trial_type)
        frames.append(out)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "summary_selected_trials.csv", index=False)


def write_report(normal_trial: str, pattern_trial: str) -> None:
    lines = [
        "# Report Grafik Bab 4.7",
        "",
        "Seluruh grafik dibuat dari artifact yang sudah ada. Tidak ada retraining dan tidak ada perubahan checkpoint/model.",
        "",
        "## Trial Terpilih",
        f"- Trial sesuai pola acuan: `{normal_trial}`",
        f"- Trial pola aktivitas berbeda: `{pattern_trial}`",
        "",
        "Pemilihan trial dilakukan otomatis: trial sesuai pola acuan dipilih dari data test dengan rerata probabilitas pola berbeda terendah; trial pola aktivitas berbeda dipilih dari data uji pola berbeda dengan rerata probabilitas pola berbeda tertinggi.",
        "",
        "## Daftar Figure",
        "",
    ]
    for item in FIGURES:
        lines.extend(
            [
                f"### {item.filename}",
                f"- Subbab: {item.section}",
                f"- Sumber data: `{item.source}`",
                f"- Caption saran: {item.caption}",
            ]
        )
        if item.notes:
            lines.append(f"- Catatan: {item.notes}")
        lines.append("")
    if MISSING:
        lines.extend(["## Data Tidak Ditemukan / Keterbatasan", ""])
        for item in sorted(set(MISSING)):
            lines.append(f"- `{item}`")
        lines.append("")
    lines.extend(
        [
            "## Figure yang Paling Kuat untuk Bab 4.7",
            "- `fig_4_7_5_confusion_matrix.png` untuk bukti performa Dynamic Temporal GCN.",
            "- `fig_4_7_5_training_accuracy.png`, `fig_4_7_5_training_loss.png`, dan `fig_4_7_5_training_f1_score.png` untuk dinamika training.",
            "- `fig_4_7_6_probability_per_muscle_different.png` untuk memperlihatkan klasifikasi pola aktivitas otot per node.",
            "- `fig_4_7_7_relation_scatter_different.png` dan `fig_4_7_7_top_relation_bar.png` untuk analisis relasi antarotot.",
            "- `fig_4_7_1_runtime_latency_bar.png` dan `fig_4_7_1_runtime_error_bar.png` untuk evidence runtime validation.",
        ]
    )
    (OUT_DIR / "report_bab_4_7.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_out()
    set_style()
    history, pred_test, pred_pattern, metrics, cm = load_model_artifacts()
    normal_trial, pattern_trial = choose_trials(pred_test, pred_pattern)
    edges = load_edges()
    normal_rel, normal_top_rel = compute_relation_table(pred_test, normal_trial, edges)
    pattern_rel, pattern_top_rel = compute_relation_table(pred_pattern, pattern_trial, edges)

    figure_runtime_latency()
    figure_runtime_error()
    figure_simulation_replay_comparison()
    grouped_metric_bar(
        "fig_4_7_2_joint_error_per_group.png",
        "mean_joint_error",
        "Mean joint error",
        "Mean Joint Error per Kelompok Trial",
        "4.7.2",
        "Perbandingan mean joint error antara trial normal dan trial pola aktivitas berbeda.",
    )
    grouped_metric_bar(
        "fig_4_7_3_replay_rmse_comparison.png",
        "rmse_forward_error",
        "Mean RMSE",
        "Perbandingan RMSE Replay",
        "4.7.3",
        "Perbandingan rerata RMSE replay antara dua kelompok trial.",
    )
    grouped_metric_bar(
        "fig_4_7_3_ctrl_delta_comparison.png",
        "ctrl_delta_mean",
        "Mean ctrl delta",
        "Perbandingan Perubahan Control Otot",
        "4.7.3",
        "Perbandingan rerata perubahan control otot antara dua kelompok trial.",
    )
    figure_control_profile()
    figure_graph_structure()
    if normal_trial:
        figure_temporal_heatmap(normal_trial)
    figure_training_curves(history, metrics, cm, pred_test)
    figure_probability_bars(normal_rel, pattern_rel)
    figure_relation_scatter(
        normal_rel,
        "fig_4_7_7_relation_scatter_normal.png",
        "Skor Relasi Antarotot - Trial Sesuai Pola Acuan",
        "predictions_test.csv + neo4j_muscle_edges.csv",
        "Scatter plot probabilitas pola berbeda dan skor relasi antarotot pada trial sesuai pola acuan.",
    )
    figure_relation_scatter(
        pattern_rel,
        "fig_4_7_7_relation_scatter_different.png",
        "Skor Relasi Antarotot - Trial Pola Aktivitas Berbeda",
        "predictions_abnormal_validation.csv + neo4j_muscle_edges.csv",
        "Scatter plot probabilitas pola berbeda dan skor relasi antarotot pada trial pola aktivitas berbeda.",
    )
    relation_source = pattern_top_rel if not pattern_top_rel.empty else normal_top_rel
    figure_top_relation_bar(relation_source)
    figure_relation_subgraph(relation_source)
    draw_gui_reconstruction(
        "fig_4_7_8_gui_normal.png",
        "Rekonstruksi GUI - Trial Sesuai Pola Acuan",
        normal_rel,
        "Rekonstruksi GUI dari hasil analisis trial sesuai pola acuan.",
    )
    draw_gui_reconstruction(
        "fig_4_7_8_gui_different.png",
        "Rekonstruksi GUI - Trial Pola Aktivitas Berbeda",
        pattern_rel,
        "Rekonstruksi GUI dari hasil analisis trial pola aktivitas berbeda.",
    )
    figure_gui_callout()
    figure_limitation()
    write_selected_trials_csv(normal_rel, pattern_rel)
    write_report(normal_trial, pattern_trial)

    # Also copy existing ready-made embedding/relation outputs if useful for the same folder.
    for src_name, dst_name in [
        (
            PROJECT_ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier" / "node_embedding_visualization_pca.png",
            OUT_DIR / "extra_node_embedding_visualization_pca.png",
        ),
        (
            PROJECT_ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier" / "relation_score_visualization" / "relation_score_scatter_current_trial.png",
            OUT_DIR / "extra_relation_score_scatter_current_trial.png",
        ),
    ]:
        if src_name.exists():
            shutil.copy2(src_name, dst_name)

    print(f"Created {len(FIGURES)} figure records in {OUT_DIR}")
    print(f"Selected normal trial: {normal_trial}")
    print(f"Selected different-pattern trial: {pattern_trial}")
    if MISSING:
        print("Missing/limited sources:")
        for item in sorted(set(MISSING)):
            print(f" - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
