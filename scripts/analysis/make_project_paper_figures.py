from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.affected_muscle_analysis import analyze_affected_muscles  # noqa: E402
from app.graph_sequence_baseline import compare_trial_to_reference, load_reference_sequence, load_trial_graph_sequence  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
RUN_DIR = PROJECT_ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier_labeled_records"
REFERENCE_DIR = PROJECT_ROOT / "artifacts" / "retrain_labeled_records" / "reference_sequence" / "reference"
SPLIT_COUNTS = PROJECT_ROOT / "artifacts" / "retrain_labeled_records" / "reference_sequence" / "split_counts.json"
ABNORMAL_DATASET = PROJECT_ROOT / "artifacts" / "retrain_labeled_records" / "abnormal_graph_sequence_dataset"
NORMAL_DATASET = PROJECT_ROOT / "artifacts" / "retrain_labeled_records" / "graph_sequence_dataset"

COLORS = {
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#16A34A",
    "orange": "#D97706",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "light": "#F8FAFC",
    "dark": "#111827",
    "line": "#CBD5E1",
    "ink": "#0F172A",
    "muted": "#475569",
}


def _setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.16)
    plt.close(fig)
    return path


def _box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, *, fc: str, ec: str = "#334155") -> patches.FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.035",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=COLORS["ink"], weight="bold", fontsize=8.5, wrap=True)
    return patch


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = "#475569", lw: float = 1.8) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=6, shrinkB=6))


def _blank_ax(figsize: tuple[float, float] = (12, 6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def fig_3_1_system_workflow() -> Path:
    fig, ax = _blank_ax((14, 5.0))
    labels = [
        ("Hand motion\nrecording", "Data acquisition"),
        ("qpos.csv\ntrajectory", "Trial file"),
        ("Smoothing\nqvel / qacc", "Preprocessing"),
        ("Inverse\ndynamics", "Biomechanics"),
        ("OSQP\nctrl_ref", "Muscle control"),
        ("Graph sequence", "Temporal graph"),
        ("Dynamic\nTemporal GCN", "Node model"),
        ("Pattern difference\n+ muscle relation", "Interpretation"),
    ]
    xs = np.linspace(0.035, 0.86, len(labels))
    box_w, box_h = 0.105, 0.22
    colors = ["#EAF2FF", "#EAF2FF", "#F2F7F2", "#F2F7F2", "#F2F7F2", "#FFF7ED", "#FFF7ED", "#FEE2E2"]
    for idx, ((main, sub), x) in enumerate(zip(labels, xs)):
        _box(ax, (x, 0.48), (box_w, box_h), main, fc=colors[idx])
        ax.text(x + box_w / 2, 0.425, sub, ha="center", va="center", color=COLORS["muted"], fontsize=7.8)
        if idx < len(labels) - 1:
            _arrow(ax, (x + box_w, 0.59), (xs[idx + 1], 0.59), color="#64748B", lw=1.6)
    ax.text(0.5, 0.89, "System Workflow for Open-to-Fist Muscle Activity-Pattern Analysis", ha="center", va="center", fontsize=15, weight="bold", color=COLORS["ink"])
    ax.text(0.5, 0.24, "Main output: Sesuai Pola Acuan, Pola Aktivitas Berbeda, and relational muscle interpretation", ha="center", fontsize=9.5, color=COLORS["muted"])
    return _save(fig, "fig_3_1_system_workflow.png")


def fig_3_2_system_architecture() -> Path:
    fig, ax = _blank_ax((13, 7.2))
    lanes = [
        ("User Interface Layer", 0.78, "#EAF2FF", ["Tkinter GUI", "Record / Load", "Run Analysis", "Result Table + Plots"]),
        ("Viewer Layer", 0.58, "#F8FAFC", ["Leap Viewer", "MuJoCo Kinematic Replay", "MuJoCo Control Replay"]),
        ("Biomechanics Layer", 0.38, "#F2F7F2", ["qpos preprocessing", "inverse dynamics", "OSQP ctrl_ref", "mismatch feedback"]),
        ("Graph + Learning Layer", 0.18, "#FFF5E6", ["temporal graph", "dynamic edges", "Dynamic Temporal GCN", "relation analysis"]),
    ]
    for title, y, fc, items in lanes:
        _box(ax, (0.04, y), (0.18, 0.11), title, fc=fc)
        x0 = 0.28
        for i, item in enumerate(items):
            _box(ax, (x0 + i * 0.17, y), (0.135, 0.11), item, fc=fc)
            if i > 0:
                _arrow(ax, (x0 + (i - 1) * 0.17 + 0.135, y + 0.055), (x0 + i * 0.17, y + 0.055), color="#94A3B8", lw=1.25)
    for y1, y2 in [(0.78, 0.58), (0.58, 0.38), (0.38, 0.18)]:
        _arrow(ax, (0.5, y1), (0.5, y2 + 0.11), color="#64748B")
    ax.text(0.5, 0.95, "Final System Architecture", ha="center", va="center", fontsize=15, weight="bold", color=COLORS["ink"])
    ax.text(0.5, 0.06, "Neo4j is optional for graph database and visual analytics; it is not required for training or GUI execution.", ha="center", fontsize=9, color=COLORS["muted"])
    return _save(fig, "fig_3_2_system_architecture.png")


def _sample_graph() -> dict[str, Any]:
    manifest_path = NORMAL_DATASET / "dataset_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    return load_trial_graph_sequence(Path(str(manifest.iloc[0]["json_path"])))


def _node_positions(node_names: list[str]) -> dict[str, tuple[float, float]]:
    groups = {
        "Extensors": [n for n in node_names if n.startswith(("E", "LU", "UI", "RI"))],
        "Flexors": [n for n in node_names if n.startswith(("F", "P"))],
        "Thumb / Intrinsic": [n for n in node_names if n not in []],
    }
    used = set(groups["Extensors"] + groups["Flexors"])
    groups["Thumb / Intrinsic"] = [n for n in node_names if n not in used]
    centers = {"Extensors": (0.22, 0.58), "Flexors": (0.58, 0.58), "Thumb / Intrinsic": (0.42, 0.25)}
    positions: dict[str, tuple[float, float]] = {}
    for group, names in groups.items():
        cx, cy = centers[group]
        radius = 0.14 if len(names) > 8 else 0.10
        for i, name in enumerate(names):
            angle = 2 * math.pi * i / max(1, len(names))
            positions[name] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    return positions


def fig_3_6_muscle_relation_graph() -> Path:
    graph = _sample_graph()
    node_names = [str(n) for n in graph["node_names"]]
    edges = list(graph.get("edges") or [])
    edge_palette = {"synergist": COLORS["green"], "antagonist": COLORS["red"], "functional_group": COLORS["blue"]}
    degree: dict[str, int] = {name: 0 for name in node_names}
    filtered_edges: list[tuple[str, str, str]] = []
    for edge in edges:
        src = str(edge.get("source_name", ""))
        dst = str(edge.get("target_name", ""))
        et = str(edge.get("edge_type", "functional_group"))
        if et in edge_palette and src in degree and dst in degree:
            degree[src] += 1
            degree[dst] += 1
            filtered_edges.append((src, dst, et))
    preferred_nodes = [
        "FDS2", "FDS3", "FDS4", "FDP2", "FDP3", "FDP4",
        "EDC2", "EDC3", "EDC4", "EPL", "FPL", "APL",
        "PL", "FCR", "FCU", "PT", "PQ", "RI5",
    ]
    selected_nodes = [name for name in preferred_nodes if name in degree]
    if len(selected_nodes) < 12:
        selected_nodes = [name for name, _ in sorted(degree.items(), key=lambda item: item[1], reverse=True)[:18]]
    positions = _node_positions(selected_nodes)

    fig, ax = _blank_ax((10.5, 7.5))
    ax.text(0.5, 0.96, "Representative Muscle Relation Graph", ha="center", va="center", fontsize=15, weight="bold", color=COLORS["ink"])
    ax.text(0.5, 0.915, "Representative muscles and capped relation edges for readable anatomical interpretation.", ha="center", fontsize=8.8, color=COLORS["muted"])
    seen: set[tuple[str, str, str]] = set()
    type_counts = {key: 0 for key in edge_palette}
    max_edges_per_type = {"synergist": 14, "antagonist": 16, "functional_group": 8}
    for src, dst, et in filtered_edges:
        key = tuple(sorted((src, dst)) + [et])
        if key in seen or src not in positions or dst not in positions:
            continue
        if type_counts[et] >= max_edges_per_type[et]:
            continue
        seen.add(key)
        type_counts[et] += 1
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        ax.plot([x0, x1], [y0, y1], color=edge_palette[et], lw=1.35, alpha=0.58, zorder=1)
    for name, (x, y) in positions.items():
        fill = "#EAF2FF" if name.startswith(("E", "LU", "UI", "RI")) else "#F2F7F2" if name.startswith(("F", "P")) else "#FFF5E6"
        ax.scatter([x], [y], s=300, c=fill, edgecolors="#334155", linewidths=0.9, zorder=3)
        ax.text(x, y, name, ha="center", va="center", fontsize=7.0, weight="bold", zorder=4)
    for i, (label, color) in enumerate(edge_palette.items()):
        ax.plot([0.73, 0.79], [0.82 - i * 0.045, 0.82 - i * 0.045], color=color, lw=2.6)
        ax.text(0.805, 0.82 - i * 0.045, label, va="center", fontsize=9)
    return _save(fig, "fig_3_6_muscle_relation_graph.png")


def fig_3_8_dynamic_temporal_gcn_scheme() -> Path:
    fig, ax = _blank_ax((12.5, 5.6))
    ax.text(0.5, 0.92, "Dynamic Temporal GCN Scheme", ha="center", fontsize=15, weight="bold", color=COLORS["ink"])
    _box(ax, (0.04, 0.58), (0.17, 0.16), "Node feature\nsequence X[t]", fc="#EAF2FF")
    _box(ax, (0.04, 0.28), (0.17, 0.16), "Dynamic edge\nsequence A[t]", fc="#EAF2FF")
    _box(ax, (0.29, 0.43), (0.16, 0.18), "GCN layer\nper progress step", fc="#F2F7F2")
    _box(ax, (0.53, 0.43), (0.16, 0.18), "GRU temporal\nencoder", fc="#FFF7ED")
    _box(ax, (0.75, 0.43), (0.16, 0.18), "Temporal\npooling", fc="#FFF7ED")
    _box(ax, (0.40, 0.12), (0.21, 0.14), "Node classifier\nProb. Pola Berbeda", fc="#FEE2E2")
    _box(ax, (0.69, 0.12), (0.24, 0.14), "Output per muscle:\nSesuai Acuan / Pola Berbeda", fc="#FEE2E2")
    _arrow(ax, (0.20, 0.66), (0.28, 0.52))
    _arrow(ax, (0.20, 0.36), (0.28, 0.52))
    _arrow(ax, (0.44, 0.52), (0.52, 0.52))
    _arrow(ax, (0.68, 0.52), (0.74, 0.52))
    _arrow(ax, (0.82, 0.43), (0.50, 0.26))
    _arrow(ax, (0.60, 0.19), (0.68, 0.19))
    return _save(fig, "fig_3_8_dynamic_temporal_gcn_scheme.png")


def fig_3_9_affected_muscle_analysis_flow() -> Path:
    fig, ax = _blank_ax((12.5, 5.3))
    ax.text(0.5, 0.91, "Inter-Muscle Relation Analysis Flow", ha="center", fontsize=15, weight="bold", color=COLORS["ink"])
    steps = [
        ("GCN output\nProb. Pola Berbeda", "#EAF2FF"),
        ("Threshold\nP >= 0.5", "#F2F7F2"),
        ("Different-pattern\nmuscles", "#FEE2E2"),
        ("Aggregate edge\nweights over time", "#FFF7ED"),
        ("Risk(v)=sum\nA(u,v) * P(u)", "#FFF7ED"),
        ("Final status:\nacuan / pola berbeda /\nterkait relasional", "#EAF2FF"),
    ]
    xs = np.linspace(0.05, 0.78, len(steps))
    for i, (text, fc) in enumerate(steps):
        _box(ax, (xs[i], 0.48), (0.14, 0.18), text, fc=fc)
        if i < len(steps) - 1:
            _arrow(ax, (xs[i] + 0.14, 0.57), (xs[i + 1], 0.57))
    ax.text(0.5, 0.27, "Relational muscle status is a simulation-based interpretation, not a medical assessment.", ha="center", color=COLORS["muted"], fontsize=9.2)
    return _save(fig, "fig_3_9_affected_muscle_analysis_flow.png")


def _load_history() -> pd.DataFrame:
    return pd.read_csv(RUN_DIR / "history.csv")


def _load_metrics() -> dict[str, Any]:
    return json.loads((RUN_DIR / "eval_metrics.json").read_text(encoding="utf-8"))


def fig_4_1_dataset_split() -> Path:
    counts = json.loads(SPLIT_COUNTS.read_text(encoding="utf-8"))
    abnormal_count = len(pd.read_csv(RUN_DIR / "abnormal_split_manifest.csv")["trial_id"].unique()) if (RUN_DIR / "abnormal_split_manifest.csv").exists() else 0
    labels = ["Train", "Validation", "Reference-pattern holdout", "Pattern-different validation"]
    values = [counts["train"], counts["val"], counts["holdout"], abnormal_count]
    colors = [COLORS["blue"], COLORS["cyan"], COLORS["orange"], COLORS["red"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, edgecolor="#334155", linewidth=0.8)
    ax.set_title("Dataset Split")
    ax.set_ylabel("Number of trials")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(value), ha="center", va="bottom", weight="bold")
    return _save(fig, "fig_4_1_dataset_split.png")


def _curve_figure(cols: list[str], labels: list[str], title: str, ylabel: str, name: str, *, ylim: tuple[float, float] | None = None) -> Path:
    history = _load_history()
    metrics = _load_metrics()
    best_epoch = int(metrics.get("best_epoch", 0))
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]
    for col, label, color in zip(cols, labels, palette):
        if col in history.columns:
            ax.plot(history["epoch"], history[col], label=label, lw=2.35, color=color)
    if best_epoch > 0:
        ax.axvline(best_epoch, color=COLORS["red"], ls="--", lw=1.5, label=f"best epoch {best_epoch}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    return _save(fig, name)


def fig_4_3_training_accuracy() -> Path:
    return _curve_figure(["train_accuracy", "val_accuracy"], ["Training accuracy", "Validation accuracy"], "Training and Validation Accuracy", "Accuracy", "fig_4_3_training_accuracy.png", ylim=(0.0, 1.02))


def fig_4_4_training_loss() -> Path:
    return _curve_figure(["train_loss", "val_loss"], ["Training loss", "Validation loss"], "Training and Validation Loss", "Loss", "fig_4_4_training_loss.png")


def fig_4_5_training_f1_score() -> Path:
    return _curve_figure(["train_macro_f1", "val_macro_f1", "train_binary_f1_deviating", "val_binary_f1_deviating"], ["Train macro F1", "Val macro F1", "Train different-pattern F1", "Val different-pattern F1"], "Training and Validation F1 Score", "F1 score", "fig_4_5_training_f1_score.png", ylim=(0.0, 1.02))


def fig_4_6_confusion_matrix() -> Path:
    cm = np.asarray(_load_metrics()["test"]["confusion_matrix"], dtype=float)
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    norm = cm / row_sum
    labels = ["Sesuai Pola Acuan", "Pola Aktivitas Berbeda"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xticks(range(2), labels=labels, rotation=20, ha="right")
    ax.set_yticks(range(2), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(2):
        for j in range(2):
            txt_color = "white" if norm[i, j] > 0.55 else COLORS["ink"]
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]*100:.1f}%", ha="center", va="center", color=txt_color, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")
    return _save(fig, "fig_4_6_confusion_matrix.png")


def _representative_trial() -> tuple[str, dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary = pd.read_csv(RUN_DIR / "abnormal_node_summary.csv")
    pred = pd.read_csv(RUN_DIR / "predictions_abnormal_validation.csv")
    trial_scores = summary.groupby("trial_id")["average_deviation_score"].mean().sort_values(ascending=False)
    trial_id = str(trial_scores.index[0])
    manifest = pd.read_csv(ABNORMAL_DATASET / "dataset_manifest.csv")
    row = manifest.loc[(manifest["trial_id"].astype(str) == trial_id) & (manifest["cohort"].astype(str) == "abnormal")]
    if row.empty:
        row = manifest.loc[manifest["cohort"].astype(str) == "abnormal"].head(1)
        trial_id = str(row.iloc[0]["trial_id"])
    graph = load_trial_graph_sequence(Path(str(row.iloc[0]["json_path"])))
    summary_trial = summary.loc[summary["trial_id"].astype(str) == trial_id].copy()
    pred_trial = pred.loc[pred["trial_id"].astype(str) == trial_id].copy()
    pred_by_node = pred_trial.set_index("node_name").to_dict(orient="index")
    node_results = []
    for item in summary_trial.to_dict(orient="records"):
        node = str(item["node_name"])
        p = pred_by_node.get(node, {})
        node_results.append(
            {
                "node_name": node,
                "probability_deviating": float(p.get("prob_deviating", 0.0)),
                "predicted_label": str(p.get("predicted_class", "within_reference")),
                "deviation_score": float(item.get("average_deviation_score", 0.0)),
            }
        )
    affected = analyze_affected_muscles(graph_trial=graph, node_results=node_results, threshold_deviating=0.5, threshold_affected=0.25, top_k_affected=8)
    affected_df = pd.DataFrame(affected["rows"])
    return trial_id, graph, summary_trial, pred_trial, affected


def fig_4_8_muscle_deviation_barplot() -> Path:
    trial_id, _graph, summary, _pred, affected = _representative_trial()
    status = pd.DataFrame(affected["rows"])[["muscle_name", "final_status"]]
    df = summary.merge(status, left_on="node_name", right_on="muscle_name", how="left")
    df = df.sort_values("average_deviation_score", ascending=False).head(15).iloc[::-1]
    color_map = {"terindikasi_menyimpang": COLORS["red"], "berpotensi_terdampak": COLORS["orange"], "sesuai_acuan": COLORS["green"]}
    colors = [color_map.get(str(s), COLORS["gray"]) for s in df["final_status"]]
    fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)
    ax.barh(df["node_name"], df["average_deviation_score"], color=colors, edgecolor="#334155", linewidth=0.6)
    ax.set_title(f"Muscle Activity-Pattern Difference Score ({trial_id})")
    ax.set_xlabel("Average activity-pattern difference score")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    return _save(fig, "fig_4_8_muscle_deviation_barplot.png")


def fig_4_9_muscle_deviation_heatmap() -> Path:
    trial_id, graph, _summary, _pred, _affected = _representative_trial()
    reference = load_reference_sequence(REFERENCE_DIR)
    comparison = compare_trial_to_reference(graph, reference)
    heat = comparison["node_deviation_df"].copy()
    top_nodes = (
        heat.groupby("node_name")["aggregate_deviation_score"].max().sort_values(ascending=False).head(20).index.tolist()
    )
    heat = heat.loc[heat["node_name"].isin(top_nodes)]
    pivot = heat.pivot_table(index="node_name", columns="step_index", values="aggregate_deviation_score", aggfunc="mean").reindex(top_nodes)
    fig, ax = plt.subplots(figsize=(11.2, 6.2), constrained_layout=True)
    im = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_title(f"Muscle-vs-Progress Activity-Pattern Heatmap ({trial_id})")
    ax.set_xlabel("Progress-normalized step")
    ax.set_ylabel("Muscle node")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    xticks = np.linspace(0, max(0, pivot.shape[1] - 1), 5, dtype=int)
    ax.set_xticks(xticks, labels=[f"{int(x / max(1, pivot.shape[1]-1) * 100)}%" for x in xticks])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Activity-pattern difference score")
    return _save(fig, "fig_4_9_muscle_deviation_heatmap.png")


def fig_4_11_relational_risk_barplot() -> Path:
    trial_id, _graph, _summary, _pred, affected = _representative_trial()
    df = pd.DataFrame(affected["rows"]).sort_values("relational_risk", ascending=False).head(15).iloc[::-1]
    colors = [COLORS["red"] if s == "terindikasi_menyimpang" else COLORS["orange"] if s == "berpotensi_terdampak" else COLORS["green"] for s in df["final_status"]]
    fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)
    ax.barh(df["muscle_name"], df["relational_risk"], color=colors, edgecolor="#334155", linewidth=0.6)
    ax.set_title(f"Relational Risk Score ({trial_id})")
    ax.set_xlabel("Relational risk")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    return _save(fig, "fig_4_11_relational_risk_barplot.png")


def fig_4_12_affected_muscle_table() -> Path:
    _trial_id, _graph, _summary, _pred, affected = _representative_trial()
    df = pd.DataFrame(affected["rows"])
    rank = {"terindikasi_menyimpang": 0, "berpotensi_terdampak": 1, "sesuai_acuan": 2}
    df["rank"] = df["final_status"].map(rank).fillna(3)
    df = df.sort_values(["rank", "relational_risk", "p_deviating"], ascending=[True, False, False]).head(12)
    table_df = df[["muscle_name", "p_deviating", "relational_risk", "strongest_related_deviating_muscle", "final_status"]].copy()
    table_df["p_deviating"] = table_df["p_deviating"].map(lambda x: f"{float(x):.3f}")
    table_df["relational_risk"] = table_df["relational_risk"].map(lambda x: f"{float(x):.3f}")
    fig, ax = plt.subplots(figsize=(12.6, 5.8), constrained_layout=True)
    ax.axis("off")
    ax.set_title("Inter-Muscle Relation Analysis Table", pad=18, fontsize=14, weight="bold")
    table = ax.table(cellText=table_df.values, colLabels=["Muscle", "Prob. pola berbeda", "Risk", "Related different-pattern muscle", "Final status"], loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(weight="bold")
        elif table_df.iloc[row - 1]["final_status"] == "terindikasi_menyimpang":
            cell.set_facecolor("#FEE2E2")
        elif table_df.iloc[row - 1]["final_status"] == "berpotensi_terdampak":
            cell.set_facecolor("#FEF3C7")
        else:
            cell.set_facecolor("#DCFCE7")
    return _save(fig, "fig_4_12_affected_muscle_table.png")


def _edge_strength_matrix(graph: dict[str, Any]) -> np.ndarray:
    node_count = len(graph["node_names"])
    edge_weight_seq = np.asarray(graph.get("edge_weight_seq"), dtype=float)
    mat = np.zeros((node_count, node_count), dtype=float)
    for edge in list(graph.get("edges") or []):
        src = int(edge.get("source_idx", -1))
        dst = int(edge.get("target_idx", -1))
        edge_idx = int(edge.get("edge_idx", -1))
        if 0 <= src < node_count and 0 <= dst < node_count and 0 <= edge_idx < edge_weight_seq.shape[1]:
            weight = float(np.mean(np.abs(edge_weight_seq[:, edge_idx])))
            mat[src, dst] = max(mat[src, dst], weight)
            mat[dst, src] = max(mat[dst, src], weight)
    return mat


def fig_4_13_affected_muscle_subgraph() -> Path:
    trial_id, graph, _summary, _pred, affected = _representative_trial()
    affected_df = pd.DataFrame(affected["rows"])
    selected = affected_df.loc[affected_df["final_status"].isin(["terindikasi_menyimpang", "berpotensi_terdampak"])].copy()
    if selected.empty:
        selected = affected_df.sort_values("relational_risk", ascending=False).head(10)
    selected = selected.sort_values(["final_status", "relational_risk"], ascending=[True, False]).head(14)
    selected_names = selected["muscle_name"].tolist()
    node_names = list(graph["node_names"])
    index = {name: i for i, name in enumerate(node_names)}
    mat = _edge_strength_matrix(graph)
    positions = {}
    for i, name in enumerate(selected_names):
        angle = 2 * math.pi * i / max(1, len(selected_names))
        positions[name] = (0.5 + 0.34 * math.cos(angle), 0.5 + 0.34 * math.sin(angle))
    fig, ax = _blank_ax((8, 7))
    ax.text(0.5, 0.95, f"Relational Muscle Subgraph ({trial_id})", ha="center", fontsize=14, weight="bold")
    for i, src in enumerate(selected_names):
        for dst in selected_names[i + 1 :]:
            w = mat[index[src], index[dst]] if src in index and dst in index else 0.0
            if w <= 0:
                continue
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            ax.plot([x0, x1], [y0, y1], color="#94A3B8", lw=0.8 + 5 * min(0.2, w), alpha=0.55)
    status_map = selected.set_index("muscle_name")["final_status"].to_dict()
    fill_map = {"terindikasi_menyimpang": "#FEE2E2", "berpotensi_terdampak": "#FEF3C7", "sesuai_acuan": "#DCFCE7"}
    edge_map = {"terindikasi_menyimpang": COLORS["red"], "berpotensi_terdampak": COLORS["orange"], "sesuai_acuan": COLORS["green"]}
    for name, (x, y) in positions.items():
        status = status_map.get(name, "sesuai_acuan")
        ax.scatter([x], [y], s=520, c=fill_map.get(status, "#F8FAFC"), edgecolors=edge_map.get(status, "#334155"), linewidths=1.6, zorder=3)
        ax.text(x, y, name, ha="center", va="center", fontsize=8, weight="bold", zorder=4)
    ax.text(0.08, 0.07, "red: terindikasi_menyimpang   yellow: berpotensi_terdampak", fontsize=9, color="#475569")
    return _save(fig, "fig_4_13_affected_muscle_subgraph.png")


def main() -> int:
    _setup()
    builders = [
        fig_3_1_system_workflow,
        fig_3_2_system_architecture,
        fig_3_6_muscle_relation_graph,
        fig_3_8_dynamic_temporal_gcn_scheme,
        fig_3_9_affected_muscle_analysis_flow,
        fig_4_1_dataset_split,
        fig_4_3_training_accuracy,
        fig_4_4_training_loss,
        fig_4_5_training_f1_score,
        fig_4_6_confusion_matrix,
        fig_4_8_muscle_deviation_barplot,
        fig_4_9_muscle_deviation_heatmap,
        fig_4_11_relational_risk_barplot,
        fig_4_12_affected_muscle_table,
        fig_4_13_affected_muscle_subgraph,
    ]
    paths = []
    for builder in builders:
        path = builder()
        paths.append(path)
        print(f"[FIG] {path}")
    (OUT_DIR / "figure_manifest.txt").write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {len(paths)} figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
