from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "bab_4_7_figures_final"

RUNTIME_DIR = ROOT / "artifacts" / "realtime_recorded_qpos_audit"
MODEL_DIR = ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier_labeled_records"
EMBED_DIR = ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier"
REL_DIR = EMBED_DIR / "relation_score_visualization"
RETRAIN_DIR = ROOT / "artifacts" / "retrain_labeled_records"
CTRL_NORMAL_DIR = RETRAIN_DIR / "ctrl_ref_normal" / "reference_build"
CTRL_DIFFERENT_DIR = RETRAIN_DIR / "ctrl_ref_abnormal" / "reference_build"
GRAPH_NORMAL_DIR = RETRAIN_DIR / "graph_sequence_dataset"
EDGE_PATH = ROOT / "artifacts" / "final_book_figures" / "neo4j_export" / "neo4j_muscle_edges.csv"

BLUE = "#1F4E79"
GRAY = "#6B7280"
LIGHT = "#D1D5DB"
DARK = "#111827"
BROWN = "#8C6D31"
STATUS_COLORS = {
    "Sesuai Pola Acuan": GRAY,
    "Otot Terkait Secara Relasional": BROWN,
    "Pola Aktivitas Berbeda": BLUE,
}


@dataclass
class InventoryRow:
    subbab: str
    figure_file: str
    figure_type: str
    source_file: str
    status: str
    caption: str
    note: str = ""


INVENTORY: list[InventoryRow] = []


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": LIGHT,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def ensure_dirs() -> None:
    for sub in ["4_7_1", "4_7_2", "4_7_3", "4_7_4", "4_7_5", "4_7_6", "4_7_7", "4_7_8"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def polish(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if ygrid:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    if xgrid:
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)


def save(fig: plt.Figure, sub: str, filename: str, figure_type: str, source: str, caption: str, note: str = "") -> None:
    path = OUT / sub / filename
    for ax in fig.axes:
        if hasattr(ax, "spines"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    INVENTORY.append(InventoryRow(sub.replace("_", "."), str(path.relative_to(OUT)), figure_type, source, "CREATED", caption, note))


def skip(sub: str, filename: str, figure_type: str, source: str, reason: str, caption: str = "") -> None:
    INVENTORY.append(InventoryRow(sub.replace("_", "."), str(Path(sub) / filename), figure_type, source, "NOT_CREATED", caption, reason))


def annotate_barh(ax: plt.Axes, fmt: str = "{:.2f}") -> None:
    for patch in ax.patches:
        w = patch.get_width()
        if np.isfinite(w):
            ax.text(w, patch.get_y() + patch.get_height() / 2, fmt.format(w), va="center", ha="left", fontsize=8)


def display_status(predicted: str, p_dev: float, relation_score: float = 0.0) -> str:
    if str(predicted).strip() == "deviating" or float(p_dev) >= 0.5:
        return "Pola Aktivitas Berbeda"
    if float(relation_score) >= 0.25:
        return "Otot Terkait Secara Relasional"
    return "Sesuai Pola Acuan"


def choose_trials(pred_test: pd.DataFrame, pred_diff: pd.DataFrame) -> tuple[str, str]:
    normal = ""
    diff = ""
    if not pred_test.empty:
        normal = str(pred_test.groupby("trial_id")["prob_deviating"].mean().sort_values().index[0])
    if not pred_diff.empty:
        diff = str(pred_diff.groupby("trial_id")["prob_deviating"].mean().sort_values(ascending=False).index[0])
    return normal, diff


def load_edges() -> pd.DataFrame:
    edges = read_csv(EDGE_PATH)
    if edges.empty:
        return edges
    edges["edge_weight"] = pd.to_numeric(edges.get("edge_weight", 0.0), errors="coerce").fillna(0.0)
    if "relation_type" not in edges.columns:
        edges["relation_type"] = "functional_group"
    return edges


def build_relation_table(pred_df: pd.DataFrame, trial_id: str, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial = pred_df.loc[pred_df["trial_id"].astype(str) == str(trial_id)].copy()
    if trial.empty:
        return pd.DataFrame(), pd.DataFrame()
    trial["prob_deviating"] = pd.to_numeric(trial["prob_deviating"], errors="coerce").fillna(0.0)
    p_by_node = dict(zip(trial["node_name"].astype(str), trial["prob_deviating"].astype(float)))
    pred_by_node = dict(zip(trial["node_name"].astype(str), trial["predicted_class"].astype(str)))
    primary = {n for n, p in p_by_node.items() if p >= 0.5 or pred_by_node.get(n) == "deviating"}
    risk = {n: 0.0 for n in p_by_node}
    relation_rows: list[dict[str, Any]] = []

    if not edges.empty:
        for row in edges.to_dict(orient="records"):
            src = str(row.get("source_muscle", ""))
            dst = str(row.get("target_muscle", ""))
            if src not in p_by_node or dst not in p_by_node:
                continue
            w = float(row.get("edge_weight", 0.0) or 0.0)
            for u, v in ((src, dst), (dst, src)):
                if u in primary:
                    risk[v] += w * p_by_node[u]
            relation_rows.append(
                {
                    "source": src,
                    "target": dst,
                    "relation_type": str(row.get("relation_type", "functional_group")),
                    "edge_weight": w,
                    "relation_score": max(w * p_by_node[src], w * p_by_node[dst]),
                }
            )

    out = trial[["trial_id", "node_index", "node_name", "predicted_class", "prob_deviating"]].copy()
    out["relation_score"] = out["node_name"].map(risk).fillna(0.0)
    out["final_status"] = [
        display_status(pred, p, r)
        for pred, p, r in zip(out["predicted_class"], out["prob_deviating"], out["relation_score"])
    ]
    rel = pd.DataFrame(relation_rows)
    if not rel.empty:
        rel = rel.sort_values("relation_score", ascending=False).drop_duplicates(["source", "target"])
    return out, rel


def fig_4_7_1() -> None:
    summary = read_json(RUNTIME_DIR / "validation_summary.json")
    if not summary:
        skip("4_7_1", "fig_4_7_1_runtime_summary.png", "runtime summary", str(RUNTIME_DIR / "validation_summary.json"), "runtime summary tidak ditemukan")
        return
    latency_labels = ["Mean ctrl pipeline", "P95 ctrl pipeline", "Mean frame loop", "P95 frame loop"]
    latency_vals = [
        summary.get("latency_mean_ms", np.nan),
        summary.get("latency_p95_ms", np.nan),
        summary.get("frame_loop_mean_ms", np.nan),
        summary.get("frame_loop_p95_ms", np.nan),
    ]
    error_labels = ["Mean joint error", "RMSE forward error", "P95 frame error", "Fallback ratio"]
    error_vals = [
        summary.get("sequence_mean_joint_error", np.nan),
        summary.get("sequence_rmse_forward_error", np.nan),
        summary.get("sequence_p95_frame_error", np.nan),
        summary.get("fallback_ratio", np.nan),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    axes[0].barh(latency_labels[::-1], latency_vals[::-1], color=[BLUE, GRAY, BLUE, GRAY][::-1])
    axes[0].set_xlabel("Latency (ms)")
    axes[0].set_title("A. Runtime latency")
    annotate_barh(axes[0], "{:.1f}")
    polish(axes[0], xgrid=True, ygrid=False)

    axes[1].barh(error_labels[::-1], error_vals[::-1], color=[GRAY, BLUE, GRAY, LIGHT][::-1])
    axes[1].set_xlabel("Nilai metrik")
    axes[1].set_title("B. Runtime error")
    annotate_barh(axes[1], "{:.3f}")
    polish(axes[1], xgrid=True, ygrid=False)
    save(
        fig,
        "4_7_1",
        "fig_4_7_1_runtime_summary.png",
        "data numerik",
        str(RUNTIME_DIR / "validation_summary.json"),
        "Ringkasan latency dan error pada runtime estimation.",
    )


def fig_4_7_2() -> None:
    # No reliable paired Leap/MuJoCo screenshot is available in the current artifact set.
    skip(
        "4_7_2",
        "fig_4_7_2_leap_mujoco_comparison.png",
        "screenshot comparison",
        "not available",
        "NEED_MANUAL_SCREENSHOT: tidak ditemukan pasangan screenshot Leap Viewer dan MuJoCo Viewer.",
        "Perbandingan representasi gerakan open-to-fist pada Leap Viewer dan model simulasi MuJoCo/MyoSuite.",
    )


def fig_4_7_3() -> None:
    normal = read_csv(CTRL_NORMAL_DIR / "biomech_reference_trial_summary.csv")
    diff = read_csv(CTRL_DIFFERENT_DIR / "biomech_reference_trial_summary.csv")
    if normal.empty or diff.empty:
        skip("4_7_3", "fig_4_7_3_replay_validation_comparison.png", "grouped metrics", "ctrl_ref summaries", "summary validasi replay tidak lengkap")
    else:
        labels = ["Mean Joint Error", "Mean RMSE", "Mean Ctrl Delta"]
        normal_vals = [
            pd.to_numeric(normal["mean_joint_error"], errors="coerce").mean(),
            pd.to_numeric(normal["rmse_forward_error"], errors="coerce").mean(),
            pd.to_numeric(normal["ctrl_delta_mean"], errors="coerce").mean(),
        ]
        diff_vals = [
            pd.to_numeric(diff["mean_joint_error"], errors="coerce").mean(),
            pd.to_numeric(diff["rmse_forward_error"], errors="coerce").mean(),
            pd.to_numeric(diff["ctrl_delta_mean"], errors="coerce").mean(),
        ]
        x = np.arange(len(labels))
        width = 0.34
        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        ax.bar(x - width / 2, normal_vals, width=width, label="Sesuai Pola Acuan", color=GRAY)
        ax.bar(x + width / 2, diff_vals, width=width, label="Pola Aktivitas Berbeda", color=BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_ylabel("Nilai metrik")
        ax.legend(fontsize=8)
        polish(ax)
        save(
            fig,
            "4_7_3",
            "fig_4_7_3_replay_validation_comparison.png",
            "data numerik",
            f"{CTRL_NORMAL_DIR / 'biomech_reference_trial_summary.csv'}; {CTRL_DIFFERENT_DIR / 'biomech_reference_trial_summary.csv'}",
            "Perbandingan error replay dan perubahan control otot pada trial sesuai pola acuan dan trial pola aktivitas berbeda.",
        )

    if normal.empty or "dataset_path" not in normal.columns:
        skip("4_7_3", "fig_4_7_3_control_profile_example.png", "line chart", "ctrl_ref accepted trial", "data control per frame/window tidak ditemukan")
        return
    row = normal.sort_values("rmse_forward_error").iloc[0]
    path = Path(str(row["dataset_path"]))
    ctrl_df = read_csv(path)
    ctrl_cols = [c for c in ctrl_df.columns if c.startswith("ctrl_teacher_")]
    if ctrl_df.empty or not ctrl_cols:
        skip("4_7_3", "fig_4_7_3_control_profile_example.png", "line chart", str(path), "kolom control otot tidak ditemukan")
        return
    top = ctrl_df[ctrl_cols].mean().sort_values(ascending=False).head(5).index.tolist()
    x = np.linspace(0, 100, len(ctrl_df))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for col in top:
        ax.plot(x, ctrl_df[col].astype(float), linewidth=1.6, label=col.replace("ctrl_teacher_", ""))
    ax.set_xlabel("Progress gerakan (%)")
    ax.set_ylabel("Nilai control")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, fontsize=8)
    polish(ax)
    save(
        fig,
        "4_7_3",
        "fig_4_7_3_control_profile_example.png",
        "data numerik",
        str(path),
        "Contoh profil control otot selama gerakan open-to-fist.",
        f"Trial representatif: {row['trial_id']}",
    )


def fig_4_7_4(normal_trial: str) -> None:
    ds = read_json(MODEL_DIR / "dataset_summary.json")
    num_nodes = int(ds.get("num_nodes", 39))
    num_windows = int(ds.get("num_windows", 56))
    num_features = int(ds.get("features_per_node_per_window", 6))
    edge_note = "dynamic adjacency / edge weight sequence tersedia pada graph sequence"
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.axis("off")
    boxes = [
        (f"{num_nodes}\nnode otot", 0.08, 0.48),
        (f"{num_windows}\nsequence window", 0.36, 0.48),
        (f"{num_features}\nfitur per node/window", 0.64, 0.48),
    ]
    for text, x0, y0 in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), 0.21, 0.24, fc="#E5E7EB", ec=DARK, lw=0.9))
        ax.text(x0 + 0.105, y0 + 0.12, text, ha="center", va="center", fontsize=9)
    for x0 in [0.31, 0.59]:
        ax.annotate("", xy=(x0 + 0.04, 0.60), xytext=(x0 - 0.04, 0.60), arrowprops=dict(arrowstyle="->", lw=1))
    ax.text(0.5, 0.25, edge_note, ha="center", fontsize=8, color="#374151")
    save(
        fig,
        "4_7_4",
        "fig_4_7_4_graph_temporal_structure.png",
        "diagram teknis",
        str(MODEL_DIR / "dataset_summary.json"),
        "Struktur data graph temporal yang digunakan sebagai input Dynamic Temporal GCN.",
    )

    json_path = GRAPH_NORMAL_DIR / f"normal__{normal_trial}.json"
    npz_path = GRAPH_NORMAL_DIR / f"normal__{normal_trial}.npz"
    meta = read_json(json_path)
    if not npz_path.exists() or not meta:
        skip("4_7_4", "fig_4_7_4_temporal_heatmap.png", "heatmap", f"{json_path}; {npz_path}", "file graph temporal trial representatif tidak ditemukan")
        return
    arr = np.load(npz_path)
    feat = arr["node_feature_seq"][:, :, 0].T
    names = meta.get("node_names", [f"node_{i}" for i in range(feat.shape[0])])
    order = np.argsort(feat.mean(axis=1))[::-1][:20]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    im = ax.imshow(feat[order], aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Sequence window")
    ax.set_ylabel("Node otot")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    fig.colorbar(im, ax=ax, label="Fitur control")
    save(
        fig,
        "4_7_4",
        "fig_4_7_4_temporal_heatmap.png",
        "heatmap",
        f"{json_path}; {npz_path}",
        "Heatmap temporal aktivitas otot pada graph temporal.",
        "Top 20 node berdasarkan rerata fitur control.",
    )


def fig_4_7_5() -> None:
    hist = read_csv(MODEL_DIR / "history.csv")
    metrics = read_json(MODEL_DIR / "eval_metrics.json")
    if hist.empty:
        skip("4_7_5", "fig_4_7_5_training_curves.png", "training curves", str(MODEL_DIR / "history.csv"), "history per epoch tidak ditemukan")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    panels = [
        (axes[0], ["train_accuracy", "val_accuracy"], "Accuracy", "A. Accuracy"),
        (axes[1], ["train_loss", "val_loss"], "Loss", "B. Loss"),
        (axes[2], ["train_macro_f1", "val_macro_f1", "val_binary_f1_deviating"], "F1-score", "C. F1-score"),
    ]
    for ax, cols, ylabel, title in panels:
        for col in cols:
            if col not in hist.columns:
                continue
            label = "Train" if col.startswith("train_") else "Validation"
            if "binary" in col:
                label = "Validation F1 pola berbeda"
            color = BLUE if label.startswith("Train") else GRAY
            if "pola berbeda" in label:
                color = BROWN
            style = "-" if label.startswith("Train") else "--"
            ax.plot(hist["epoch"], hist[col], linewidth=1.6, color=color, linestyle=style, label=label)
        if metrics.get("best_epoch"):
            best = float(metrics["best_epoch"])
            ax.axvline(best, color=DARK, linestyle=":", linewidth=1)
            ax.text(best, ax.get_ylim()[1], f"Best epoch: {int(best)}", rotation=90, va="top", ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7)
        polish(ax)
    save(
        fig,
        "4_7_5",
        "fig_4_7_5_training_curves.png",
        "training log",
        str(MODEL_DIR / "history.csv"),
        "Kurva accuracy, loss, dan F1-score Dynamic Temporal GCN selama proses training.",
    )


def fig_4_7_6() -> None:
    cm = np.array([[418, 51], [59, 330]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Greys")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sesuai\nPola Acuan", "Pola Aktivitas\nBerbeda"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sesuai\nPola Acuan", "Pola Aktivitas\nBerbeda"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label acuan")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10, color="white" if cm[i, j] > cm.max() * 0.55 else DARK)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(
        fig,
        "4_7_6",
        "fig_4_7_6_confusion_matrix_clean.png",
        "output model",
        str(MODEL_DIR / "confusion_matrix_test.csv"),
        "Confusion matrix hasil klasifikasi pola aktivitas otot pada data test/holdout.",
    )

    emb = read_csv(EMBED_DIR / "node_embedding_visualization_pca.csv")
    if emb.empty:
        skip("4_7_6", "fig_4_7_6_embedding_node.png", "embedding plot", str(EMBED_DIR / "node_embedding_visualization_pca.csv"), "file embedding tidak ditemukan")
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    status_col = "predicted_status_display"
    for status, group in emb.groupby(status_col):
        ax.scatter(group["embedding_x"], group["embedding_y"], s=56, color=STATUS_COLORS.get(status, GRAY), label=status, edgecolor="white", linewidth=0.7)
        for _, row in group.sort_values("prob_different_pattern", ascending=False).head(5).iterrows():
            ax.text(row["embedding_x"] + 0.03, row["embedding_y"] + 0.03, str(row["muscle_name"]), fontsize=8)
    ax.set_xlabel("Embedding Dimension 1")
    ax.set_ylabel("Embedding Dimension 2")
    ax.legend(fontsize=8)
    polish(ax)
    save(
        fig,
        "4_7_6",
        "fig_4_7_6_embedding_node.png",
        "output model",
        str(EMBED_DIR / "node_embedding_visualization_pca.csv"),
        "Visualisasi embedding node otot berdasarkan status pola aktivitas.",
        "Embedding menunjukkan kedekatan representasi fitur, bukan posisi anatomi otot.",
    )


def fig_4_7_7(diff_rel: pd.DataFrame, top_rel: pd.DataFrame) -> None:
    if diff_rel.empty:
        skip("4_7_7", "fig_4_7_7_relation_scatter.png", "scatter", "selected trial relation table", "data skor relasi trial representatif tidak tersedia")
    else:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for status, group in diff_rel.groupby("final_status"):
            ax.scatter(group["prob_deviating"], group["relation_score"], s=58, color=STATUS_COLORS.get(status, GRAY), label=status, edgecolor="white", linewidth=0.7)
        top = diff_rel.assign(rank=diff_rel["prob_deviating"] + diff_rel["relation_score"]).sort_values("rank", ascending=False).head(8)
        for _, row in top.iterrows():
            ax.text(row["prob_deviating"] + 0.01, row["relation_score"] + 0.01, str(row["node_name"]), fontsize=8)
        ax.axvline(0.5, color=DARK, linestyle="--", linewidth=1)
        ax.axhline(0.25, color=DARK, linestyle=":", linewidth=1)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Probabilitas Pola Berbeda")
        ax.set_ylabel("Skor Relasi Antarotot")
        ax.legend(fontsize=8)
        polish(ax)
        save(
            fig,
            "4_7_7",
            "fig_4_7_7_relation_scatter.png",
            "output model + edge graph",
            "selected predictions + neo4j_muscle_edges.csv",
            "Sebaran probabilitas pola aktivitas berbeda dan skor relasi antarotot.",
        )

    if top_rel.empty:
        skip("4_7_7", "fig_4_7_7_top_relation_bar.png", "bar chart", "relation table", "top relation antarotot tidak tersedia")
        return
    top = top_rel.head(10).copy().sort_values("relation_score")
    labels = top["source"].astype(str) + "-" + top["target"].astype(str)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.barh(labels, top["relation_score"], color=BROWN)
    ax.set_xlabel("Skor relasi")
    polish(ax, xgrid=True, ygrid=False)
    save(
        fig,
        "4_7_7",
        "fig_4_7_7_top_relation_bar.png",
        "output model + edge graph",
        "selected predictions + neo4j_muscle_edges.csv",
        "Top 10 relasi antarotot dengan skor relasi tertinggi.",
    )


def fig_4_7_8() -> None:
    screenshot = ROOT / "image.png"
    if not screenshot.exists():
        skip("4_7_8", "fig_4_7_8_gui_component_callout.png", "screenshot asli", str(screenshot), "NEED_MANUAL_SCREENSHOT: screenshot GUI tidak ditemukan")
        return
    img = plt.imread(screenshot)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.imshow(img)
    ax.axis("off")
    callouts = [
        ("Trial Control Panel", (0.12, 0.13), (0.19, 0.03)),
        ("Recording Session", (0.11, 0.34), (0.24, 0.25)),
        ("External Viewers\nLeap/MuJoCo controls", (0.12, 0.55), (0.29, 0.47)),
        ("Classifier Plots / Heatmap", (0.47, 0.34), (0.52, 0.05)),
        ("Muscle Result Table", (0.78, 0.18), (0.82, 0.04)),
        ("Trial Summary", (0.13, 0.82), (0.30, 0.76)),
    ]
    height, width = img.shape[0], img.shape[1]
    for text, xy, xytext in callouts:
        ax.annotate(
            text,
            xy=(xy[0] * width, xy[1] * height),
            xytext=(xytext[0] * width, xytext[1] * height),
            arrowprops=dict(arrowstyle="->", color=BROWN, lw=1.4),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=BROWN, lw=0.8),
            fontsize=8,
            color=DARK,
        )
    save(
        fig,
        "4_7_8",
        "fig_4_7_8_gui_component_callout.png",
        "screenshot asli",
        str(screenshot),
        "Komponen GUI untuk visualisasi hasil runtime monitoring dan post-recording analysis.",
        "Callout dibuat dari screenshot GUI yang tersedia.",
    )


def write_outputs() -> None:
    inv = pd.DataFrame([row.__dict__ for row in INVENTORY])
    inv.to_csv(OUT / "figure_inventory.csv", index=False)
    created = inv.loc[inv["status"] == "CREATED"]
    missing = inv.loc[inv["status"] != "CREATED"]
    lines = [
        "# Report Final Figure Bab 4.7",
        "",
        "Semua figure dibuat dari artifact/output yang tersedia. Tidak ada retraining, tidak ada perubahan model, dan tidak ada perubahan file buku utama.",
        "",
        "## Figure Berhasil Dibuat",
        "",
    ]
    for _, row in created.iterrows():
        lines.extend(
            [
                f"### {row['figure_file']}",
                f"- Subbab: {row['subbab']}",
                f"- Jenis: {row['figure_type']}",
                f"- Sumber: `{row['source_file']}`",
                f"- Caption: {row['caption']}",
            ]
        )
        if row["note"]:
            lines.append(f"- Catatan: {row['note']}")
        lines.append("")
    lines.extend(["## Figure Tidak Dibuat", ""])
    if missing.empty:
        lines.append("- Tidak ada figure gagal selain target yang memang membutuhkan screenshot manual.")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- `{row['figure_file']}`: {row['note']}")
    lines.extend(
        [
            "",
            "## Catatan Penulisan Analisis",
            "- Realtime estimation ditulis sebagai preview/monitoring, bukan hasil final estimasi aktivitas otot.",
            "- Analisis final tetap berbasis trial lengkap, estimasi control, validasi replay, graph temporal, dan Dynamic Temporal GCN.",
            "- Hindari klaim diagnosis atau interpretasi klinis.",
            "- Embedding node adalah representasi fitur model, bukan posisi anatomi otot.",
        ]
    )
    (OUT / "report_bab_4_7_figures_final.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    setup_style()
    pred_test = read_csv(MODEL_DIR / "predictions_test.csv")
    pred_diff = read_csv(MODEL_DIR / "predictions_abnormal_validation.csv")
    edges = load_edges()
    normal_trial, diff_trial = choose_trials(pred_test, pred_diff)
    normal_rel, normal_top = build_relation_table(pred_test, normal_trial, edges)
    diff_rel, diff_top = build_relation_table(pred_diff, diff_trial, edges)
    top_rel = diff_top if not diff_top.empty else normal_top

    fig_4_7_1()
    fig_4_7_2()
    fig_4_7_3()
    fig_4_7_4(normal_trial)
    fig_4_7_5()
    fig_4_7_6()
    fig_4_7_7(diff_rel, top_rel)
    fig_4_7_8()
    write_outputs()

    inv = pd.DataFrame([row.__dict__ for row in INVENTORY])
    created = inv.loc[inv["status"] == "CREATED"]
    not_created = inv.loc[inv["status"] != "CREATED"]
    print(f"Output folder: {OUT}")
    print(f"Created figures: {len(created)}")
    print(f"Not created: {len(not_created)}")
    print(f"Selected reference-pattern trial: {normal_trial}")
    print(f"Selected different-pattern trial: {diff_trial}")
    if not not_created.empty:
        print("Needs manual screenshot / unavailable data:")
        for _, row in not_created.iterrows():
            print(f" - {row['figure_file']}: {row['note']}")
    print("Strong figures for Bab 4.7:")
    for name in [
        "4_7_1/fig_4_7_1_runtime_summary.png",
        "4_7_3/fig_4_7_3_replay_validation_comparison.png",
        "4_7_5/fig_4_7_5_training_curves.png",
        "4_7_6/fig_4_7_6_confusion_matrix_clean.png",
        "4_7_7/fig_4_7_7_relation_scatter.png",
        "4_7_8/fig_4_7_8_gui_component_callout.png",
    ]:
        print(f" - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
