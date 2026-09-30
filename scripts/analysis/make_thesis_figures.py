from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier"
OUT = ROOT / "outputs" / "thesis_figures"

INK = "#172033"
MUTED = "#64748B"
LINE = "#CBD5E1"
PANEL = "#F8FAFC"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#D97706"
RED = "#DC2626"
GREEN = "#16A34A"
VIOLET = "#7C3AED"

STATUS_COLORS = {
    "deviating": RED,
    "affected": ORANGE,
    "within": BLUE,
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=320, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    fc: str = "white",
    ec: str = LINE,
    color: str = INK,
    fs: float = 8.5,
    lw: float = 1.0,
    weight: str = "normal",
    radius: float = 0.06,
):
    x, y = xy
    w, h = wh
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        fontweight=weight,
        wrap=True,
    )
    return patch


def arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = MUTED, lw: float = 1.2, rad: float = 0.0) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": lw,
            "shrinkA": 4,
            "shrinkB": 4,
            "connectionstyle": f"arc3,rad={rad}",
        },
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def history() -> pd.DataFrame:
    return pd.read_csv(RUN / "history.csv")


def eval_metrics() -> dict:
    return read_json(RUN / "eval_metrics.json")


def split_manifest() -> pd.DataFrame:
    return pd.read_csv(RUN / "split_manifest.csv")


def select_case_trial() -> str:
    pred = pd.read_csv(RUN / "predictions_test.csv")
    grouped = pred.groupby("trial_id", as_index=True)["prob_deviating"].mean().sort_values(ascending=False)
    return str(grouped.index[0])


def muscle_group(name: str) -> str:
    if name in {"ECRL", "ECRB", "ECU"}:
        return "wrist_extensor"
    if name in {"FCR", "FCU", "PL", "PT", "PQ", "FPL"} or name.startswith("FDS") or name.startswith("FDP"):
        return "flexor_pronator"
    if name.startswith("EDC") or name in {"EDM", "EIP", "EPL", "EPB", "APL"}:
        return "extensor_thumb"
    return "intrinsic"


def group_color(name: str) -> str:
    return {
        "wrist_extensor": "#38BDF8",
        "flexor_pronator": "#34D399",
        "extensor_thumb": "#F87171",
        "intrinsic": "#A78BFA",
    }[muscle_group(name)]


def load_case() -> dict:
    trial_id = select_case_trial()
    sm = split_manifest()
    row = sm.loc[sm["trial_id"].astype(str) == trial_id].iloc[0]
    graph = read_json(Path(row["json_path"]))
    npz = np.load(Path(row["npz_path"]))
    edge_weight_seq = np.asarray(npz["edge_weight_seq"], dtype=float)
    node_names = list(graph["node_names"])
    node_count = len(node_names)

    edge_matrix = np.zeros((node_count, node_count), dtype=float)
    edge_type_by_pair: dict[tuple[int, int], str] = {}
    edge_weight_by_pair: dict[tuple[int, int], float] = {}
    for edge in graph["edges"]:
        src = int(edge["source_idx"])
        dst = int(edge["target_idx"])
        idx = int(edge["edge_idx"])
        if idx >= edge_weight_seq.shape[1]:
            continue
        weight = float(np.nanmean(np.abs(edge_weight_seq[:, idx])))
        if not np.isfinite(weight):
            weight = 0.0
        edge_matrix[src, dst] = max(edge_matrix[src, dst], weight)
        edge_matrix[dst, src] = max(edge_matrix[dst, src], weight)
        pair = tuple(sorted((src, dst)))
        if weight >= edge_weight_by_pair.get(pair, -1.0):
            edge_weight_by_pair[pair] = weight
            edge_type_by_pair[pair] = str(edge.get("edge_type", "relation"))

    predictions = pd.read_csv(RUN / "predictions_test.csv")
    node_summary = pd.read_csv(RUN / "normal_node_summary.csv")
    trial_pred = predictions.loc[predictions["trial_id"].astype(str) == trial_id].copy()
    trial_dev = node_summary.loc[node_summary["trial_id"].astype(str) == trial_id].copy()
    trial = trial_pred.merge(
        trial_dev[
            [
                "node_name",
                "average_deviation_score",
                "onset_delay_progress",
                "peak_timing_deviation_progress",
                "peak_magnitude_deviation",
                "smoothness_deviation",
            ]
        ],
        on="node_name",
        how="left",
    )
    trial = trial.set_index("node_name").reindex(node_names).reset_index()
    trial["node_index"] = np.arange(node_count)
    trial["prob_deviating"] = trial["prob_deviating"].fillna(0.0)
    trial["average_deviation_score"] = trial["average_deviation_score"].fillna(0.0)

    p_dev = trial["prob_deviating"].to_numpy(dtype=float)
    deviating = p_dev >= 0.5
    risk = np.zeros(node_count, dtype=float)
    strongest = np.full(node_count, -1, dtype=int)
    strongest_weight = np.zeros(node_count, dtype=float)
    for dst in range(node_count):
        best = 0.0
        for src in np.where(deviating)[0]:
            if src == dst:
                continue
            contribution = float(edge_matrix[src, dst]) * float(p_dev[src])
            risk[dst] += contribution
            if contribution > best:
                best = contribution
                strongest[dst] = int(src)
                strongest_weight[dst] = float(edge_matrix[src, dst])

    affected_threshold = 0.001
    status = np.full(node_count, "within", dtype=object)
    status[deviating] = "deviating"
    status[(~deviating) & (risk >= affected_threshold)] = "affected"
    trial["relational_risk"] = risk
    trial["strongest_related_node"] = [node_names[i] if i >= 0 else "" for i in strongest]
    trial["strongest_edge_weight"] = strongest_weight
    trial["final_status"] = status

    return {
        "trial_id": trial_id,
        "node_names": node_names,
        "trial": trial,
        "edge_matrix": edge_matrix,
        "edge_weight_by_pair": edge_weight_by_pair,
        "edge_type_by_pair": edge_type_by_pair,
        "affected_threshold": affected_threshold,
        "num_windows": int(edge_weight_seq.shape[0]),
        "num_edges": int(edge_weight_seq.shape[1]),
    }


def title(ax, text: str, subtitle: str | None = None) -> None:
    ax.text(0.0, 1.075, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        ax.text(0.0, 1.03, subtitle, transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=8.5)


def fig_3_1_system_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    clean_axis(ax)
    title(ax, "Fig. 3.1 System Workflow", "End-to-end processing from hand motion capture to muscle activity-pattern interpretation")

    steps = [
        ("Hand motion\ncapture", "Ultraleap / Leap\nrecords qpos", "#EFF6FF", BLUE),
        ("Preprocess\ntrajectory", "smoothing\nqvel / qacc", "#F8FAFC", MUTED),
        ("Reference\ncontrol", "inverse dynamics\nOSQP ctrl_ref", "#FFF7ED", ORANGE),
        ("Graph sequence\nbuilder", "node features\nedge_weight[t]", "#ECFDF5", TEAL),
        ("Dynamic\nTemporal GCN", "GCN + GRU\nattention pooling", "#EEF2FF", VIOLET),
        ("Activity-pattern\nclassification", "probability of\ndifferent pattern", "#FEF2F2", RED),
        ("Muscle relation\nanalysis", "relation score\nstatus table", "#F0FDF4", GREEN),
    ]
    x_positions = np.linspace(0.35, 11.25, len(steps))
    for i, (name, detail, fc, color) in enumerate(steps):
        x = float(x_positions[i])
        box(ax, (x, 3.25), (1.35, 1.0), name, fc=fc, ec=color, fs=8.6, weight="bold")
        box(ax, (x, 1.75), (1.35, 0.72), detail, fc="white", ec=LINE, fs=7.7)
        arrow(ax, (x + 0.67, 3.23), (x + 0.67, 2.5), color=color, lw=1.0)
        if i < len(steps) - 1:
            arrow(ax, (x + 1.35, 3.75), (float(x_positions[i + 1]), 3.75), color=MUTED, lw=1.0)

    box(ax, (0.55, 0.55), (2.35, 0.5), "records / qpos CSV", fc=PANEL, ec=LINE, fs=8)
    box(ax, (3.45, 0.55), (2.75, 0.5), "reference sequence + graph dataset", fc=PANEL, ec=LINE, fs=8)
    box(ax, (6.8, 0.55), (2.2, 0.5), "trained model artifact", fc=PANEL, ec=LINE, fs=8)
    box(ax, (9.65, 0.55), (2.55, 0.5), "GUI plots / CSV / report", fc=PANEL, ec=LINE, fs=8)
    save(fig, "fig_3_1_system_workflow.png")


def fig_3_2_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    clean_axis(ax)
    title(ax, "Fig. 3.2 System Architecture", "Runtime GUI, offline model artifacts, and analysis modules")

    lanes = [
        ("User Interface", 5.55, "#EFF6FF"),
        ("Application Controllers", 4.25, "#F8FAFC"),
        ("Analysis and Model Layer", 2.95, "#ECFDF5"),
        ("Artifacts and Storage", 1.35, "#FFF7ED"),
    ]
    for label, y, fc in lanes:
        ax.add_patch(patches.Rectangle((0.3, y - 0.45), 11.4, 0.95, facecolor=fc, edgecolor=LINE, linewidth=0.8))
        ax.text(0.55, y, label, ha="left", va="center", fontsize=8.5, color=INK, fontweight="bold")

    box(ax, (2.1, 5.22), (1.6, 0.58), "MainWindow", fc="white", ec=BLUE, fs=8.2, weight="bold")
    box(ax, (4.25, 5.22), (1.8, 0.58), "Trial replay view", fc="white", ec=BLUE, fs=8.2)
    box(ax, (6.6, 5.22), (1.8, 0.58), "Inference panel", fc="white", ec=BLUE, fs=8.2)
    box(ax, (8.95, 5.22), (1.75, 0.58), "Result plots", fc="white", ec=BLUE, fs=8.2)

    controllers = [
        ("trial_loader", 1.35),
        ("recorder_controller", 3.0),
        ("mujoco_replay_controller", 5.0),
        ("inference_controller", 7.35),
        ("plot_controller", 9.45),
    ]
    for name, x in controllers:
        box(ax, (x, 3.92), (1.45, 0.58), name, fc="white", ec=MUTED, fs=7.4)

    modules = [
        ("MyoSuite / MuJoCo\nreplay", 0.95, TEAL),
        ("reference sequence\ncomparison", 3.05, TEAL),
        ("graph sequence\nbuilder", 5.1, TEAL),
        ("Dynamic Temporal\nGCN model", 7.15, VIOLET),
        ("muscle relation\npost-processing", 9.25, GREEN),
    ]
    for name, x, color in modules:
        box(ax, (x, 2.52), (1.55, 0.72), name, fc="white", ec=color, fs=7.5)

    stores = [
        ("qpos records", 1.2),
        ("reference graphs", 3.25),
        ("history + metrics", 5.35),
        ("model checkpoint", 7.45),
        ("figures + CSV", 9.55),
    ]
    for name, x in stores:
        box(ax, (x, 1.07), (1.35, 0.52), name, fc="white", ec=ORANGE, fs=7.4)

    for start_x, end_x in [(2.9, 2.1), (5.15, 5.85), (7.5, 7.9), (9.85, 10.15)]:
        arrow(ax, (start_x, 5.2), (end_x, 4.52), color=BLUE, lw=0.9)
    for x in [1.95, 3.75, 5.85, 7.95, 10.0]:
        arrow(ax, (x, 3.9), (x, 3.26), color=MUTED, lw=0.9)
    for x in [1.9, 3.95, 5.95, 8.05, 10.1]:
        arrow(ax, (x, 2.5), (x, 1.62), color=ORANGE, lw=0.9)
    save(fig, "fig_3_2_system_architecture.png")


def graph_edges_for_plot(case: dict, top_k: int = 55) -> list[tuple[str, str, float, str]]:
    node_names = case["node_names"]
    rows = []
    for pair, weight in case["edge_weight_by_pair"].items():
        if weight <= 0:
            continue
        src, dst = pair
        rows.append((node_names[src], node_names[dst], float(weight), case["edge_type_by_pair"].get(pair, "relation")))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows[:top_k]


def fig_3_6_muscle_relation_graph(case: dict) -> None:
    trial = case["trial"].set_index("node_name")
    edges = graph_edges_for_plot(case, top_k=58)
    graph = nx.Graph()
    for name in case["node_names"]:
        graph.add_node(name)
    for src, dst, weight, edge_type in edges:
        graph.add_edge(src, dst, weight=weight, edge_type=edge_type)

    pos = nx.spring_layout(graph, seed=16, k=0.65, weight="weight", iterations=250)
    fig, ax = plt.subplots(figsize=(10.8, 8.2))
    clean_axis(ax)
    title(ax, "Fig. 3.6 Muscle Relation Graph", f"Top temporal relations from trial {case['trial_id']}")

    edge_colors = {"synergist": TEAL, "antagonist": RED, "functional_group": MUTED}
    widths = [0.8 + 5.0 * float(graph[u][v]["weight"]) for u, v in graph.edges()]
    colors = [edge_colors.get(graph[u][v]["edge_type"], LINE) for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=colors, width=widths, alpha=0.38)

    node_colors = []
    node_sizes = []
    for name in graph.nodes():
        status = str(trial.loc[name, "final_status"])
        node_colors.append(STATUS_COLORS.get(status, BLUE))
        node_sizes.append(190 + 360 * float(trial.loc[name, "prob_deviating"]))
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=1.2)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7.2, font_color=INK)

    legend_items = [
        patches.Patch(color=RED, label="pola aktivitas berbeda"),
        patches.Patch(color=ORANGE, label="terkait secara relasional"),
        patches.Patch(color=BLUE, label="sesuai pola acuan"),
        patches.Patch(color=TEAL, label="synergist edge"),
        patches.Patch(color=MUTED, label="functional group edge"),
    ]
    ax.legend(handles=legend_items, frameon=False, loc="lower left", ncol=2)
    save(fig, "fig_3_6_muscle_relation_graph.png")


def fig_3_8_dynamic_temporal_gcn_scheme(case: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    clean_axis(ax)
    title(ax, "Fig. 3.8 Dynamic Temporal GCN Scheme", "Temporal graph input, graph convolution, recurrent aggregation, and node classification")

    for i, x in enumerate([0.65, 1.05, 1.45, 1.85]):
        box(ax, (x, 2.7 + i * 0.08), (1.1, 1.0), f"G{i + 1}", fc="#EFF6FF", ec="#93C5FD", fs=8.5, weight="bold")
    ax.text(1.55, 1.92, f"T={case['num_windows']} windows\nN=39 muscle nodes\nF=6 node features", ha="center", fontsize=8.2, color=MUTED)

    blocks = [
        ("Dynamic adjacency\nedge_weight_seq[t]", 3.25, "#F0F9FF", TEAL),
        ("Graph convolution\nper time step", 5.05, "#ECFDF5", TEAL),
        ("GRU over movement\nprogress per node", 6.95, "#EEF2FF", VIOLET),
        ("Attention pooling\nnode representation", 8.85, "#F5F3FF", VIOLET),
        ("Classifier head\nProb. acuan / pola berbeda", 10.55, "#FEF2F2", RED),
    ]
    for text, x, fc, ec in blocks:
        box(ax, (x, 2.63), (1.35, 1.12), text, fc=fc, ec=ec, fs=7.8)
    arrow(ax, (2.75, 3.2), (3.25, 3.2), color=BLUE)
    for x0, x1 in [(4.6, 5.05), (6.4, 6.95), (8.3, 8.85), (10.2, 10.55)]:
        arrow(ax, (x0, 3.2), (x1, 3.2), color=MUTED)

    box(ax, (4.35, 0.8), (5.3, 0.62), "Training target: binary node activity-pattern label from trial-vs-reference score", fc="white", ec=LINE, fs=8)
    arrow(ax, (6.85, 1.44), (7.65, 2.62), color=ORANGE, rad=-0.12)
    ax.text(6.0, 4.75, "Output is node-level, not trial-level: each muscle/actuator receives an independent different-pattern probability.", ha="center", color=INK, fontsize=8.4)
    save(fig, "fig_3_8_dynamic_temporal_gcn_scheme.png")


def fig_3_9_affected_muscle_analysis_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    clean_axis(ax)
    title(ax, "Fig. 3.9 Inter-Muscle Relation Analysis Flow", "Post-processing layer after Dynamic Temporal GCN inference")

    box(ax, (0.6, 3.8), (2.0, 0.75), "D-TGCN output\nProb. pola berbeda", fc="#FEF2F2", ec=RED, fs=8.2)
    box(ax, (0.6, 2.65), (2.0, 0.75), "Pattern difference\nfrom reference", fc="#FFF7ED", ec=ORANGE, fs=8.2)
    box(ax, (0.6, 1.5), (2.0, 0.75), "Muscle relation graph\nedge weights", fc="#ECFDF5", ec=TEAL, fs=8.2)
    box(ax, (3.35, 3.2), (1.9, 0.82), "Rule 1\nProb. pola berbeda >= 0.5", fc="white", ec=RED, fs=8.2, weight="bold")
    box(ax, (5.95, 3.2), (2.15, 0.82), "Mark muscle as\nactivity pattern different", fc="#FEF2F2", ec=RED, fs=8.2)
    box(ax, (3.35, 1.85), (2.1, 0.92), "Compute relational risk\nRisk(v)=sum w(u,v)P(u)", fc="white", ec=TEAL, fs=8.0, weight="bold")
    box(ax, (6.15, 1.85), (2.0, 0.92), "Rule 2\nrisk >= threshold", fc="white", ec=ORANGE, fs=8.2, weight="bold")
    box(ax, (9.0, 2.55), (2.0, 0.82), "Final status\nper muscle", fc="#F0FDF4", ec=GREEN, fs=8.2, weight="bold")
    box(ax, (9.0, 1.35), (2.0, 0.82), "CSV table,\nbar plot, subgraph", fc=PANEL, ec=LINE, fs=8.1)

    arrow(ax, (2.62, 4.17), (3.35, 3.62), color=RED)
    arrow(ax, (2.62, 3.02), (3.35, 3.48), color=ORANGE)
    arrow(ax, (2.62, 1.87), (3.35, 2.25), color=TEAL)
    arrow(ax, (5.25, 3.6), (5.95, 3.6), color=RED)
    arrow(ax, (5.45, 2.3), (6.15, 2.3), color=TEAL)
    arrow(ax, (8.15, 2.3), (9.0, 2.88), color=ORANGE)
    arrow(ax, (10.0, 2.53), (10.0, 2.18), color=MUTED)
    ax.text(
        6.05,
        0.62,
        "Interpretation note: relational status means graph-related to different-pattern muscles; it is not a medical assessment.",
        ha="center",
        color=MUTED,
        fontsize=8.2,
    )
    save(fig, "fig_3_9_affected_muscle_analysis_flow.png")


def fig_4_1_dataset_split() -> None:
    sm = split_manifest()
    summary = read_json(RUN / "dataset_summary.json")
    counts = sm["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int)
    node_counts = counts * int(summary["num_nodes"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]})
    colors = [BLUE, ORANGE, GREEN]
    x = np.arange(len(counts))
    axes[0].bar(x - 0.18, counts.values, width=0.36, color=colors, alpha=0.92, label="trials")
    axes[0].bar(x + 0.18, node_counts.values, width=0.36, color=colors, alpha=0.35, label="node samples")
    axes[0].set_xticks(x, ["train", "validation", "test"])
    axes[0].set_ylabel("Count")
    axes[0].set_title("Dataset split")
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[0].legend(frameon=False)
    for i, (trial_count, node_count) in enumerate(zip(counts.values, node_counts.values)):
        axes[0].text(i - 0.18, trial_count + 1, str(trial_count), ha="center", va="bottom", fontsize=8)
        axes[0].text(i + 0.18, node_count + 20, str(node_count), ha="center", va="bottom", fontsize=8)

    axes[1].axis("off")
    title(axes[1], "Fig. 4.1 Dataset Split", "Final Dynamic Temporal GCN dataset")
    metrics = [
        ("Trials", int(counts.sum())),
        ("Muscle nodes", int(summary["num_nodes"])),
        ("Windows / trial", int(summary["num_windows"])),
        ("Features / node / window", int(summary["features_per_node_per_window"])),
        ("Binary threshold", f"{float(summary['binary_threshold']):.3f}"),
    ]
    for i, (label, value) in enumerate(metrics):
        y = 0.78 - i * 0.14
        axes[1].add_patch(patches.Rectangle((0.05, y), 0.82, 0.1, transform=axes[1].transAxes, facecolor=PANEL, edgecolor=LINE, linewidth=0.8))
        axes[1].text(0.1, y + 0.05, label, transform=axes[1].transAxes, ha="left", va="center", fontsize=8.5, color=INK)
        axes[1].text(0.8, y + 0.05, str(value), transform=axes[1].transAxes, ha="right", va="center", fontsize=8.5, color=INK, fontweight="bold")
    save(fig, "fig_4_1_dataset_split.png")


def plot_training_curve(filename: str, title_text: str, columns: Iterable[tuple[str, str, str, str]], ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    hist = history()
    metrics = eval_metrics()
    best_epoch = int(metrics.get("best_epoch", 0))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for column, label, color, linestyle in columns:
        ax.plot(hist["epoch"], hist[column], color=color, linestyle=linestyle, linewidth=2.0, label=label)
    if best_epoch:
        ax.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1.1)
        ax.text(best_epoch + 0.4, ax.get_ylim()[1], f"best epoch {best_epoch}", va="top", ha="left", fontsize=8, color=MUTED)
    ax.set_title(title_text, fontweight="bold", loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False)
    save(fig, filename)


def fig_4_3_training_accuracy() -> None:
    plot_training_curve(
        "fig_4_3_training_accuracy.png",
        "Fig. 4.3 Training Accuracy",
        [("train_accuracy", "train", BLUE, "-"), ("val_accuracy", "validation", ORANGE, "-")],
        "Accuracy",
        (0.80, 0.93),
    )


def fig_4_4_training_loss() -> None:
    plot_training_curve(
        "fig_4_4_training_loss.png",
        "Fig. 4.4 Training Loss",
        [("train_loss", "train", BLUE, "-"), ("val_loss", "validation", ORANGE, "-")],
        "Loss",
        None,
    )


def fig_4_5_training_f1_score() -> None:
    plot_training_curve(
        "fig_4_5_training_f1_score.png",
        "Fig. 4.5 Training F1 Score",
        [
            ("train_macro_f1", "train macro F1", BLUE, "-"),
            ("val_macro_f1", "validation macro F1", ORANGE, "-"),
            ("val_binary_f1_deviating", "validation different-pattern F1", RED, "--"),
        ],
        "F1 score",
        (0.74, 0.93),
    )


def fig_4_6_confusion_matrix() -> None:
    cm = np.asarray(eval_metrics()["test"]["confusion_matrix"], dtype=float)
    labels = ["sesuai\npola acuan", "pola\nberbeda"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if norm[i, j] > 0.55 else INK
            ax.text(j, i, f"{int(cm[i, j])}\n{100 * norm[i, j]:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Target label")
    ax.set_title("Fig. 4.6 Confusion Matrix", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized rate")
    save(fig, "fig_4_6_confusion_matrix.png")


def top_case_deviation(case: dict, n: int = 15) -> pd.DataFrame:
    return case["trial"].sort_values("average_deviation_score", ascending=False).head(n).copy()


def fig_4_8_muscle_deviation_barplot(case: dict) -> None:
    top = top_case_deviation(case, 15).sort_values("average_deviation_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = [STATUS_COLORS[str(status)] for status in top["final_status"]]
    ax.barh(top["node_name"], top["average_deviation_score"], color=colors, alpha=0.88)
    ax.set_xlabel("Average activity-pattern difference score")
    ax.set_title("Fig. 4.8 Muscle Activity-Pattern Difference Barplot", loc="left", fontweight="bold")
    ax.text(0.0, 1.01, f"Representative test trial: {case['trial_id']}", transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=8)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    handles = [patches.Patch(color=color, label=label) for label, color in [("pola berbeda", RED), ("terkait relasional", ORANGE), ("sesuai acuan", BLUE)]]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    save(fig, "fig_4_8_muscle_deviation_barplot.png")


def normalize_series(values: pd.Series) -> pd.Series:
    values = values.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - lo) / (hi - lo)


def fig_4_9_muscle_deviation_heatmap(case: dict) -> None:
    trial = case["trial"].copy()
    trial["abs_onset_delay"] = trial["onset_delay_progress"].abs()
    trial["abs_peak_timing"] = trial["peak_timing_deviation_progress"].abs()
    selected = trial.sort_values("average_deviation_score", ascending=False).head(12).copy()
    columns = [
        ("prob_deviating", "Prob. pola berbeda"),
        ("average_deviation_score", "pattern score"),
        ("abs_onset_delay", "onset"),
        ("abs_peak_timing", "peak time"),
        ("peak_magnitude_deviation", "magnitude"),
        ("smoothness_deviation", "smoothness"),
    ]
    heat = pd.DataFrame({label: normalize_series(selected[col]) for col, label in columns}, index=selected["node_name"])

    cmap = LinearSegmentedColormap.from_list("thesis_heat", ["#F8FAFC", "#FBBF24", "#DC2626"])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(heat.columns)), heat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    ax.set_title("Fig. 4.9 Muscle Activity-Pattern Difference Heatmap", loc="left", fontweight="bold")
    ax.text(0.0, 1.01, "Values are min-max normalized within the selected trial.", transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=8)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.1, color="white" if val > 0.62 else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized intensity")
    save(fig, "fig_4_9_muscle_deviation_heatmap.png")


def fig_4_11_relational_risk_barplot(case: dict) -> None:
    trial = case["trial"].copy()
    candidates = trial.loc[trial["final_status"] != "deviating"].sort_values("relational_risk", ascending=False).head(12)
    candidates = candidates.sort_values("relational_risk", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    colors = [STATUS_COLORS[str(status)] for status in candidates["final_status"]]
    ax.barh(candidates["node_name"], candidates["relational_risk"], color=colors, alpha=0.88)
    ax.axvline(case["affected_threshold"], color=ORANGE, linewidth=1.3, linestyle="--", label=f"relation threshold={case['affected_threshold']:.3f}")
    for _, row in candidates.iterrows():
        if row["strongest_related_node"]:
            ax.text(row["relational_risk"] + 0.00012, row["node_name"], str(row["strongest_related_node"]), va="center", ha="left", fontsize=7.4, color=MUTED)
    ax.set_xlabel("Relational risk")
    ax.set_title("Fig. 4.11 Relational Risk Barplot", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_4_11_relational_risk_barplot.png")


def status_label(status: str) -> str:
    return {
        "deviating": "pola\nberbeda",
        "affected": "terkait\nrelasional",
        "within": "sesuai\nacuan",
    }.get(status, status)


def fig_4_12_affected_muscle_table(case: dict) -> None:
    trial = case["trial"].copy()
    affected = trial.loc[trial["final_status"] == "affected"].sort_values("relational_risk", ascending=False)
    deviating = trial.loc[trial["final_status"] == "deviating"].sort_values("prob_deviating", ascending=False).head(5)
    table_df = pd.concat([affected, deviating], ignore_index=True)
    table_df = table_df[
        [
            "node_name",
            "prob_deviating",
            "average_deviation_score",
            "relational_risk",
            "strongest_related_node",
            "final_status",
        ]
    ].copy()
    rows = []
    for row in table_df.itertuples(index=False):
        rows.append(
            [
                row.node_name,
                f"{row.prob_deviating:.3f}",
                f"{row.average_deviation_score:.3f}",
                f"{row.relational_risk:.4f}",
                row.strongest_related_node or "-",
                status_label(str(row.final_status)),
            ]
        )

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.axis("off")
    title(ax, "Fig. 4.12 Muscle Relation Analysis Table", f"Representative test trial: {case['trial_id']}")
    table = ax.table(
        cellText=rows,
        colLabels=["Muscle", "Prob. pola berbeda", "Pattern score", "Relation score", "Related node", "Final status"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.1, 0.13, 0.13, 0.15, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(color=INK, fontweight="bold")
        elif c == 5:
            status_text = rows[r - 1][5]
            if "menyimpang" in status_text:
                cell.set_facecolor("#FEE2E2")
            elif "terdampak" in status_text:
                cell.set_facecolor("#FEF3C7")
            else:
                cell.set_facecolor("#DBEAFE")
        else:
            cell.set_facecolor("white")
    save(fig, "fig_4_12_affected_muscle_table.png")


def fig_4_13_affected_muscle_subgraph(case: dict) -> None:
    trial = case["trial"].set_index("node_name")
    affected_nodes = trial.loc[trial["final_status"] == "affected"].index.tolist()
    related_nodes = [str(trial.loc[name, "strongest_related_node"]) for name in affected_nodes if str(trial.loc[name, "strongest_related_node"])]
    top_deviating = trial.loc[trial["final_status"] == "deviating"].sort_values("relational_risk", ascending=False).head(6).index.tolist()
    selected = list(dict.fromkeys(affected_nodes + related_nodes + top_deviating))

    graph = nx.Graph()
    for name in selected:
        graph.add_node(name)
    for pair, weight in case["edge_weight_by_pair"].items():
        src_idx, dst_idx = pair
        src = case["node_names"][src_idx]
        dst = case["node_names"][dst_idx]
        if src in selected and dst in selected and weight > 0:
            graph.add_edge(src, dst, weight=float(weight), edge_type=case["edge_type_by_pair"].get(pair, "relation"))

    pos = nx.spring_layout(graph, seed=9, k=0.9, weight="weight", iterations=250)
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    clean_axis(ax)
    title(ax, "Fig. 4.13 Relational Muscle Subgraph", "Relational muscles and their strongest related different-pattern muscles")

    edge_colors = []
    edge_widths = []
    for u, v in graph.edges():
        edge_type = graph[u][v].get("edge_type", "relation")
        edge_colors.append({"synergist": TEAL, "antagonist": RED, "functional_group": MUTED}.get(edge_type, LINE))
        edge_widths.append(1.0 + 7.0 * float(graph[u][v].get("weight", 0.0)))
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.42)

    node_colors = [STATUS_COLORS[str(trial.loc[name, "final_status"])] for name in graph.nodes()]
    node_sizes = [360 + 520 * float(trial.loc[name, "prob_deviating"]) for name in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=1.4)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color=INK, font_weight="bold")

    for name in affected_nodes:
        related = str(trial.loc[name, "strongest_related_node"])
        if related in graph.nodes and name in graph.nodes:
            x0, y0 = pos[related]
            x1, y1 = pos[name]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.7, "linestyle": "--", "shrinkA": 12, "shrinkB": 12})

    handles = [patches.Patch(color=color, label=label) for label, color in [("pola berbeda", RED), ("terkait relasional", ORANGE), ("sesuai acuan", BLUE)]]
    ax.legend(handles=handles, frameon=False, loc="lower left")
    save(fig, "fig_4_13_affected_muscle_subgraph.png")


def write_notes(case: dict) -> None:
    notes = OUT / "figure_source_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# Thesis Figure Source Notes",
                "",
                f"- Output directory: `{OUT}`",
                f"- Dynamic Temporal GCN run: `{RUN}`",
                f"- Representative muscle-analysis trial: `{case['trial_id']}`",
                "- Figures 4.3-4.6 use final `history.csv` and `eval_metrics.json`.",
                "- Figures 4.8-4.13 use one representative test trial plus its graph-sequence edge weights.",
                "- Relational muscle status is a post-processing result from inter-muscle relation score, not a medical assessment.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    setup_style()
    case = load_case()
    fig_3_1_system_workflow()
    fig_3_2_system_architecture()
    fig_3_6_muscle_relation_graph(case)
    fig_3_8_dynamic_temporal_gcn_scheme(case)
    fig_3_9_affected_muscle_analysis_flow()
    fig_4_1_dataset_split()
    fig_4_3_training_accuracy()
    fig_4_4_training_loss()
    fig_4_5_training_f1_score()
    fig_4_6_confusion_matrix()
    fig_4_8_muscle_deviation_barplot(case)
    fig_4_9_muscle_deviation_heatmap(case)
    fig_4_11_relational_risk_barplot(case)
    fig_4_12_affected_muscle_table(case)
    fig_4_13_affected_muscle_subgraph(case)
    write_notes(case)
    print(f"Generated thesis figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
