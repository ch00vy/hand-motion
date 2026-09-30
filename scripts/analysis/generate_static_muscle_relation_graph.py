from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "artifacts" / "final_book_figures" / "neo4j_export"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "final_book_figures"
DEFAULT_PAPER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"

RELATION_COLORS = {
    "synergist": "#16A34A",
    "antagonist": "#D97706",
    "functional_group": "#8B7355",
}
LEGACY_DARK_BG = "#120C08"
LEGACY_DARK_GRID = "#2B241D"
LEGACY_DARK_TEXT = "#8B949E"
LEGACY_NODE_FACE = "#C98A45"
LEGACY_NODE_EDGE = "#D7C2A0"
LEGACY_NODE_DIM_EDGE = "#725E49"
CORE_EDGE_MUSCLES = {
    "EIP", "EDC3", "EDC4", "EDC5", "EDM",
    "FDS4", "FDS5",
    "FDP2", "FDP3", "FDP4", "FDP5",
    "RI2", "RI3", "RI4", "RI5",
    "LU_RB2", "LU_RB3", "LU_RB4", "LU_RB5",
    "UI_UB2", "UI_UB3", "UI_UB4", "UI_UB5",
    "ECU", "ECRB",
}

NODE_GROUP_COLORS = {
    "thumb": "#FFF7ED",
    "digit2": "#EAF2FF",
    "digit3": "#ECFDF5",
    "digit4": "#FEF3C7",
    "digit5": "#FCE7F3",
    "wrist": "#F1F5F9",
    "forearm": "#F8FAFC",
    "other": "#FFFFFF",
}


def _split_csv_arg(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _digit_family(muscle_name: str, muscle_group: str) -> str:
    text = f"{muscle_name} {muscle_group}".lower()
    if "thumb" in text or muscle_name in {"EPL", "EPB", "FPL", "APL", "OP"}:
        return "thumb"
    if "digit2" in text or muscle_name.endswith("2"):
        return "digit2"
    if "digit3" in text or muscle_name.endswith("3"):
        return "digit3"
    if "digit4" in text or muscle_name.endswith("4"):
        return "digit4"
    if "digit5" in text or muscle_name.endswith("5") or muscle_name == "EDM":
        return "digit5"
    if "wrist" in text or muscle_name in {"ECRL", "ECRB", "ECU", "FCR", "FCU", "PL"}:
        return "wrist"
    if muscle_name in {"PT", "PQ"}:
        return "forearm"
    return "other"


def _node_color(muscle_name: str, muscle_group: str) -> str:
    return NODE_GROUP_COLORS.get(_digit_family(muscle_name, muscle_group), NODE_GROUP_COLORS["other"])


def _manual_positions(nodes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    # Node merepresentasikan otot/aktuator dan bersifat statis. Aktivitas otot tidak
    # digunakan untuk menghilangkan atau mengubah ukuran node pada visualisasi graph
    # hubungan antarotot.
    base_columns = {
        "thumb": (0.12, ["APL", "EPB", "EPL", "OP", "FPL"]),
        "digit2": (0.31, ["EIP", "EDC2", "RI2", "LU_RB2", "UI_UB2", "FDS2", "FDP2"]),
        "digit3": (0.47, ["EDC3", "RI3", "LU_RB3", "UI_UB3", "FDS3", "FDP3"]),
        "digit4": (0.63, ["EDC4", "RI4", "LU_RB4", "UI_UB4", "FDS4", "FDP4"]),
        "digit5": (0.79, ["EDM", "EDC5", "RI5", "LU_RB5", "UI_UB5", "FDS5", "FDP5"]),
        "wrist": (0.47, ["ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"]),
    }
    y_by_role = {
        "EIP": 0.76,
        "EDM": 0.72,
        "EDC2": 0.58,
        "EDC3": 0.58,
        "EDC4": 0.58,
        "EDC5": 0.54,
        "RI2": 0.45,
        "RI3": 0.45,
        "RI4": 0.45,
        "RI5": 0.43,
        "LU_RB2": 0.35,
        "LU_RB3": 0.35,
        "LU_RB4": 0.35,
        "LU_RB5": 0.34,
        "UI_UB2": 0.27,
        "UI_UB3": 0.27,
        "UI_UB4": 0.27,
        "UI_UB5": 0.27,
        "FDS2": 0.27,
        "FDS3": 0.27,
        "FDS4": 0.43,
        "FDS5": 0.39,
        "FDP2": 0.14,
        "FDP3": 0.17,
        "FDP4": 0.19,
        "FDP5": 0.23,
        "APL": 0.64,
        "EPB": 0.53,
        "EPL": 0.42,
        "OP": 0.31,
        "FPL": 0.20,
    }
    wrist_positions = {
        "ECRL": (0.86, 0.72),
        "ECRB": (0.50, 0.10),
        "ECU": (0.40, 0.08),
        "FCR": (0.74, 0.10),
        "FCU": (0.58, 0.07),
        "PL": (0.66, 0.07),
        "PT": (0.50, 0.055),
        "PQ": (0.58, 0.055),
    }

    positions: dict[str, tuple[float, float]] = {}
    known = set(nodes["muscle_name"].astype(str))
    for family, (x, names) in base_columns.items():
        if family == "wrist":
            for name, pos in wrist_positions.items():
                if name in known:
                    positions[name] = pos
            continue
        fallback_y = np.linspace(0.72, 0.18, max(1, len(names)))
        for idx, name in enumerate(names):
            if name not in known:
                continue
            positions[name] = (x, y_by_role.get(name, float(fallback_y[idx])))

    missing = [name for name in nodes["muscle_name"].astype(str) if name not in positions]
    for idx, name in enumerate(missing):
        positions[name] = (0.52 + 0.08 * ((idx % 3) - 1), 0.06 + 0.06 * (idx // 3))
    return positions


def _paper_positions(nodes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Book-friendly muscle layout.

    Node tetap merepresentasikan otot/aktuator otot. Layout disusun bertingkat:
    otot finger-specific di area atas, otot intrinsik/flexor-extensor jari di
    area tengah, dan otot wrist/forearm di area bawah.
    """
    x_by_digit = {
        "thumb": 0.10,
        "digit2": 0.30,
        "digit3": 0.50,
        "digit4": 0.70,
        "digit5": 0.90,
    }
    y_by_name = {
        "APL": 0.80,
        "EPB": 0.68,
        "EPL": 0.56,
        "OP": 0.44,
        "FPL": 0.32,
        "EIP": 0.82,
        "EDC2": 0.70,
        "RI2": 0.58,
        "LU_RB2": 0.46,
        "UI_UB2": 0.34,
        "FDS2": 0.24,
        "FDP2": 0.14,
        "EDC3": 0.72,
        "RI3": 0.60,
        "LU_RB3": 0.48,
        "UI_UB3": 0.36,
        "FDS3": 0.25,
        "FDP3": 0.15,
        "EDC4": 0.72,
        "RI4": 0.60,
        "LU_RB4": 0.48,
        "UI_UB4": 0.36,
        "FDS4": 0.25,
        "FDP4": 0.15,
        "EDM": 0.84,
        "EDC5": 0.72,
        "RI5": 0.60,
        "LU_RB5": 0.48,
        "UI_UB5": 0.36,
        "FDS5": 0.25,
        "FDP5": 0.15,
    }
    wrist_positions = {
        "ECRL": (0.18, 0.03),
        "ECRB": (0.28, 0.03),
        "ECU": (0.38, 0.03),
        "FCR": (0.52, 0.03),
        "FCU": (0.62, 0.03),
        "PL": (0.72, 0.03),
        "PT": (0.82, 0.03),
        "PQ": (0.92, 0.03),
    }
    positions: dict[str, tuple[float, float]] = {}
    for row in nodes.to_dict(orient="records"):
        name = str(row["muscle_name"])
        if name in wrist_positions:
            positions[name] = wrist_positions[name]
            continue
        family = _digit_family(name, str(row.get("muscle_group", "")))
        x = x_by_digit.get(family, 0.50)
        y = y_by_name.get(name, 0.48)
        positions[name] = (float(x), float(y))
    return positions


def _paper_label_offset(name: str) -> tuple[float, float, str]:
    if name in {"ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"}:
        return 0.000, -0.045, "center"
    if name in {"FDS2", "FDP2", "FDS3", "FDP3", "FDS4", "FDP4", "FDS5", "FDP5"}:
        return 0.020, -0.012, "left"
    if name in {"APL", "EPB", "EPL", "OP", "FPL"}:
        return 0.020, 0.010, "left"
    if name in {"EDM", "EDC5", "RI5", "LU_RB5", "UI_UB5"}:
        return -0.020, 0.010, "right"
    return 0.018, 0.010, "left"


def _edge_curve_radius(src: str, dst: str, relation: str) -> float:
    code = sum(ord(char) for char in f"{src}-{dst}-{relation}")
    return float(((code % 7) - 3) * 0.045)


def _draw_curved_edge(
    ax: plt.Axes,
    *,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str,
    width: float,
    alpha: float,
    radius: float,
    zorder: int = 1,
) -> None:
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={radius:.3f}",
        linewidth=width,
        color=color,
        alpha=alpha,
        mutation_scale=1.0,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _filter_edges(
    edges: pd.DataFrame,
    *,
    top_k_edges: int,
    relation_types: Iterable[str],
    focus_muscles: Iterable[str],
    selected_nodes: set[str],
    show_all_edges: bool = False,
) -> pd.DataFrame:
    filtered = edges.copy()
    if not filtered.empty:
        filtered["_pair"] = filtered.apply(
            lambda row: "::".join(sorted([str(row["source_muscle"]), str(row["target_muscle"])])),
            axis=1,
        )
        filtered = (
            filtered.sort_values("edge_weight", ascending=False)
            .drop_duplicates(subset=["_pair", "relation_type"], keep="first")
            .drop(columns=["_pair"])
        )
    relation_types = set(relation_types)
    focus_muscles = set(focus_muscles)
    if relation_types:
        filtered = filtered.loc[filtered["relation_type"].isin(relation_types)]
    filtered = filtered.loc[
        filtered["source_muscle"].isin(selected_nodes) & filtered["target_muscle"].isin(selected_nodes)
    ]
    if focus_muscles:
        filtered = filtered.loc[
            filtered["source_muscle"].isin(focus_muscles) | filtered["target_muscle"].isin(focus_muscles)
        ]
    filtered = filtered.loc[filtered["edge_weight"].astype(float) > 0.0].copy()
    if filtered.empty:
        return filtered
    filtered["edge_weight"] = filtered["edge_weight"].astype(float)
    filtered = filtered.sort_values("edge_weight", ascending=False)
    if bool(show_all_edges):
        return filtered.reset_index(drop=True)
    preferred = filtered.loc[
        filtered["source_muscle"].isin(CORE_EDGE_MUSCLES) & filtered["target_muscle"].isin(CORE_EDGE_MUSCLES)
    ].copy()
    if len(preferred) >= max(8, min(int(top_k_edges), 18)):
        filtered = preferred.sort_values("edge_weight", ascending=False)
    top_k_edges = max(1, int(top_k_edges))
    if not relation_types and "relation_type" in filtered.columns:
        buckets = []
        relation_order = ["synergist", "antagonist", "functional_group"]
        per_type_min = max(1, min(5, top_k_edges // max(1, len(relation_order))))
        taken_indices: set[int] = set()
        for relation in relation_order:
            part = filtered.loc[filtered["relation_type"] == relation].head(per_type_min)
            buckets.append(part)
            taken_indices.update(int(idx) for idx in part.index)
        remaining = filtered.loc[~filtered.index.isin(taken_indices)].head(max(0, top_k_edges - sum(len(item) for item in buckets)))
        filtered = pd.concat([*buckets, remaining], axis=0).drop_duplicates()
        return filtered.sort_values("edge_weight", ascending=False).head(top_k_edges)
    return filtered.head(top_k_edges)


def generate_static_graph(
    *,
    nodes_csv: Path,
    edges_csv: Path,
    output_png: Path,
    output_svg: Path,
    top_k_edges: int,
    relation_type_filter: list[str],
    focus_muscles: list[str],
    selected_nodes: list[str],
    visual_style: str = "legacy_dark",
    show_all_edges: bool = False,
) -> dict[str, object]:
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    nodes["muscle_name"] = nodes["muscle_name"].astype(str)
    edges["source_muscle"] = edges["source_muscle"].astype(str)
    edges["target_muscle"] = edges["target_muscle"].astype(str)
    edges["relation_type"] = edges["relation_type"].astype(str)

    if selected_nodes:
        node_set = set(selected_nodes)
        nodes = nodes.loc[nodes["muscle_name"].isin(node_set)].copy()
    else:
        node_set = set(nodes["muscle_name"])

    filtered_edges = _filter_edges(
        edges,
        top_k_edges=top_k_edges,
        relation_types=relation_type_filter,
        focus_muscles=focus_muscles,
        selected_nodes=node_set,
        show_all_edges=show_all_edges,
    )

    graph = nx.Graph()
    for item in nodes.to_dict(orient="records"):
        graph.add_node(item["muscle_name"], **item)
    for item in filtered_edges.to_dict(orient="records"):
        graph.add_edge(
            item["source_muscle"],
            item["target_muscle"],
            relation_type=item["relation_type"],
            edge_weight=float(item["edge_weight"]),
        )

    style = str(visual_style).strip().lower()
    positions = _paper_positions(nodes) if style == "paper" else _manual_positions(nodes)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "font.family": "DejaVu Sans",
            "font.size": 9,
        }
    )
    use_legacy = style == "legacy_dark"
    use_paper = style == "paper"
    if use_legacy:
        fig, ax = plt.subplots(figsize=(9.0, 9.0))
        fig.patch.set_facecolor(LEGACY_DARK_BG)
        ax.set_facecolor(LEGACY_DARK_BG)
        ax.axis("off")
        ax.set_xlim(0.02, 1.05)
        ax.set_ylim(0.05, 0.88)
        for x in [0.215, 0.39, 0.55, 0.71]:
            ax.axvline(x, ymin=0.03, ymax=0.94, color=LEGACY_DARK_GRID, linewidth=1.0, zorder=0)
        for y in [0.43, 0.10]:
            ax.axhline(y, xmin=0.0, xmax=1.0, color=LEGACY_DARK_GRID, linewidth=1.0, zorder=0)
    elif use_paper:
        fig, ax = plt.subplots(figsize=(13.5, 8.2))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.set_xlim(0.02, 0.98)
        ax.set_ylim(-0.08, 0.94)
        ax.text(
            0.5,
            1.055,
            "Graph Relasi Antarotot pada Gesture Open-to-Fist",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="#111827",
            clip_on=False,
        )
        ax.text(
            0.5,
            1.020,
            "Node = otot/aktuator otot | Edge = relasi antarotot",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#475569",
            clip_on=False,
        )
        for x in [0.20, 0.40, 0.60, 0.80]:
            ax.axvline(x, ymin=0.08, ymax=0.82, color="#E5E7EB", linewidth=0.9, zorder=0)
        for y, label in [(0.76, "Finger-specific"), (0.43, "Shared finger muscles"), (0.10, "Wrist / Forearm")]:
            ax.axhline(y, xmin=0.04, xmax=0.96, color="#F1F5F9", linewidth=1.0, zorder=0)
            ax.text(-0.012, y, label, ha="right", va="center", fontsize=8, color="#94A3B8", clip_on=False)
    else:
        fig, ax = plt.subplots(figsize=(12.5, 7.5))
        ax.set_facecolor("white")
        ax.axis("off")

        ax.text(
            0.5,
            1.035,
            "Muscle Relation Graph for Open-to-Fist Motion",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color="#0F172A",
        )
        ax.text(
            0.5,
            1.000,
            "Static muscle/actuator nodes with strongest relation edges",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#475569",
        )

    if not filtered_edges.empty:
        weights = filtered_edges["edge_weight"].astype(float)
        min_w = float(weights.min())
        max_w = float(weights.max())
        denom = max(max_w - min_w, 1e-12)
        for item in filtered_edges.to_dict(orient="records"):
            src = item["source_muscle"]
            dst = item["target_muscle"]
            if src not in positions or dst not in positions:
                continue
            relation = item["relation_type"]
            raw_weight = float(item["edge_weight"])
            width = 0.9 + 4.2 * ((raw_weight - min_w) / denom)
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            color = RELATION_COLORS.get(relation, "#8B7355")
            if use_paper:
                edge_alpha = 0.22 if show_all_edges else (0.56 if relation != "functional_group" else 0.34)
                edge_width = 0.35 + 1.8 * ((raw_weight - min_w) / denom) if show_all_edges else 0.65 + 3.1 * ((raw_weight - min_w) / denom)
                _draw_curved_edge(
                    ax,
                    p0=(x0, y0),
                    p1=(x1, y1),
                    color=color,
                    width=edge_width,
                    alpha=edge_alpha,
                    radius=_edge_curve_radius(src, dst, relation),
                    zorder=1,
                )
            else:
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=color,
                    linewidth=width,
                    alpha=0.34 if (show_all_edges and use_legacy) else (0.74 if use_legacy else 0.58),
                    solid_capstyle="round",
                    zorder=1,
                )

    edge_nodes = set()
    if not filtered_edges.empty:
        edge_nodes.update(filtered_edges["source_muscle"].astype(str).tolist())
        edge_nodes.update(filtered_edges["target_muscle"].astype(str).tolist())
    for item in nodes.to_dict(orient="records"):
        name = item["muscle_name"]
        x, y = positions[name]
        group = str(item.get("muscle_group", ""))
        family = _digit_family(name, group)
        if use_legacy:
            # Node merepresentasikan otot/aktuator dan bersifat statis. Aktivitas otot
            # tidak digunakan untuk menghilangkan atau mengubah ukuran node pada
            # visualisasi graph hubungan antarotot.
            face = LEGACY_NODE_FACE
            edge_color = LEGACY_NODE_EDGE
            size = 520
            ax.scatter(
                [x],
                [y],
                s=size,
                facecolor=face,
                edgecolor=edge_color,
                linewidth=2.0,
                zorder=3,
            )
            ax.scatter(
                [x],
                [y],
                s=760,
                facecolor="none",
                edgecolor="#5D4C3B",
                linewidth=1.0,
                alpha=0.75,
                zorder=2,
            )
            ax.text(
                x + 0.012,
                y + 0.018,
                name,
                ha="left",
                va="bottom",
                fontsize=9.0,
                fontweight="bold",
                color="#C7D0DD",
                zorder=4,
            )
            continue
        if use_paper:
            family = _digit_family(name, group)
            node_face = {
                "thumb": "#FFF7ED",
                "digit2": "#EFF6FF",
                "digit3": "#ECFDF5",
                "digit4": "#FFFBEB",
                "digit5": "#FDF2F8",
                "wrist": "#F8FAFC",
                "forearm": "#F8FAFC",
            }.get(family, "#FFFFFF")
            ax.scatter(
                [x],
                [y],
                s=430 if family not in {"wrist", "forearm"} else 390,
                facecolor=node_face,
                edgecolor="#334155",
                linewidth=1.05,
                zorder=3,
            )
            dx, dy, align = _paper_label_offset(name)
            label_x = x + dx
            label_y = y + dy
            if abs(dx) > 0.015 or abs(dy) > 0.015:
                ax.plot([x, label_x], [y, label_y], color="#CBD5E1", linewidth=0.6, zorder=2)
            ax.text(
                label_x,
                label_y,
                name,
                ha=align,
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#1F2937",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.72),
            )
            continue
        node_size = 700 if family == "wrist" else 760 if family == "thumb" else 720
        ax.scatter(
            [x],
            [y],
            s=node_size,
            facecolor=_node_color(name, group),
            edgecolor="#334155",
            linewidth=1.25,
            zorder=3,
        )
        ax.text(x, y, name, ha="center", va="center", fontsize=8.3, fontweight="bold", color="#0F172A", zorder=4)

    finger_labels = [
        ("Thumb", 0.12),
        ("Index", 0.31),
        ("Middle", 0.47),
        ("Ring", 0.63),
        ("Little", 0.79),
    ]
    if use_paper:
        finger_labels = [
            ("Thumb", 0.10),
            ("Index", 0.30),
            ("Middle", 0.50),
            ("Ring", 0.70),
            ("Little", 0.90),
        ]
    elif not use_legacy:
        finger_labels.append(("Wrist /\nForearm", 0.94))
    for label, x in finger_labels:
        ax.text(
            x,
            0.865 if use_paper else (0.84 if use_legacy else 0.855),
            label,
            ha="center",
            va="center",
            fontsize=10 if use_paper else (12 if use_legacy else 9),
            color="#475569" if use_paper else (LEGACY_DARK_TEXT if use_legacy else "#475569"),
            fontweight="bold",
        )

    if use_legacy:
        pass
    elif use_paper:
        legend_handles = [
            mlines.Line2D([], [], color="#334155", marker="o", markersize=8, linestyle="None", markerfacecolor="#F8FAFC", label="Node = otot/aktuator otot"),
            mlines.Line2D([], [], color=RELATION_COLORS["synergist"], linewidth=2.6, label="Sinergis"),
            mlines.Line2D([], [], color=RELATION_COLORS["antagonist"], linewidth=2.6, label="Antagonis"),
            mlines.Line2D([], [], color=RELATION_COLORS["functional_group"], linewidth=2.6, label="Kelompok fungsi"),
            mlines.Line2D([], [], color="#64748B", linewidth=4.0, label="Ketebalan = kekuatan relasi"),
        ]
        ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False, fontsize=8.2)
    else:
        legend_handles = [
            mlines.Line2D([], [], color="#334155", marker="o", markersize=9, linestyle="None", markerfacecolor="#F8FAFC", label="Node = muscle actuator"),
            mlines.Line2D([], [], color=RELATION_COLORS["synergist"], linewidth=3.0, label="Synergist relation"),
            mlines.Line2D([], [], color=RELATION_COLORS["antagonist"], linewidth=3.0, label="Antagonist relation"),
            mlines.Line2D([], [], color=RELATION_COLORS["functional_group"], linewidth=3.0, label="Functional-group relation"),
            mlines.Line2D([], [], color="#64748B", linewidth=5.0, label="Line width = relation strength"),
        ]
        ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
        ax.set_xlim(0.02, 1.05)
        ax.set_ylim(0.00, 0.90)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    fig.savefig(output_svg, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "nodes_csv": str(nodes_csv),
        "edges_csv": str(edges_csv),
        "output_png": str(output_png),
        "output_svg": str(output_svg),
        "nodes_displayed": int(len(nodes)),
        "edges_displayed": int(len(filtered_edges)),
        "top_k_edges": int(top_k_edges),
        "show_all_edges": bool(show_all_edges),
        "relation_type_filter": relation_type_filter or "all",
        "focus_muscles": focus_muscles or "none",
        "relation_types_used": sorted(filtered_edges["relation_type"].unique().tolist()) if not filtered_edges.empty else [],
        "visual_style": visual_style,
    }
    (output_png.parent / f"{output_png.stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static muscle relation graph figure for Chapter 3.")
    parser.add_argument("--nodes-csv", type=Path, default=DEFAULT_INPUT_DIR / "neo4j_muscle_nodes.csv")
    parser.add_argument("--edges-csv", type=Path, default=DEFAULT_INPUT_DIR / "neo4j_muscle_edges.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k-edges", type=int, default=30)
    parser.add_argument("--show-all-edges", action="store_true", help="Render every filtered edge instead of top-k. Useful for audit figures.")
    parser.add_argument("--relation-type", type=str, default="", help="Comma-separated relation filter, e.g. synergist,antagonist.")
    parser.add_argument("--focus-muscles", type=str, default="", help="Comma-separated muscles. If set, keeps edges touching these muscles.")
    parser.add_argument("--selected-nodes", type=str, default="", help="Comma-separated node subset. Default: all nodes.")
    parser.add_argument("--visual-style", choices=["legacy_dark", "paper"], default="legacy_dark")
    parser.add_argument("--layout", choices=["legacy_dark", "paper"], default=None, help="Alias for --visual-style.")
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--output-svg", type=Path, default=None)
    args = parser.parse_args()

    visual_style = str(args.layout or args.visual_style)
    if args.output_png is not None:
        output_png = args.output_png
    elif visual_style == "paper":
        output_png = DEFAULT_PAPER_OUTPUT_DIR / "fig_graph_muscle_relation_paper.png"
    else:
        output_png = args.output_dir / "fig_3_6_muscle_relation_graph.png"
    if args.output_svg is not None:
        output_svg = args.output_svg
    elif visual_style == "paper":
        output_svg = DEFAULT_PAPER_OUTPUT_DIR / "fig_graph_muscle_relation_paper.svg"
    else:
        output_svg = args.output_dir / "fig_3_6_muscle_relation_graph.svg"
    summary = generate_static_graph(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        output_png=output_png,
        output_svg=output_svg,
        top_k_edges=args.top_k_edges,
        relation_type_filter=[item.strip() for item in args.relation_type.split(",") if item.strip()],
        focus_muscles=[item.strip() for item in args.focus_muscles.split(",") if item.strip()],
        selected_nodes=[item.strip() for item in args.selected_nodes.split(",") if item.strip()],
        visual_style=visual_style,
        show_all_edges=bool(args.show_all_edges),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
