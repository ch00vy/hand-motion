from __future__ import annotations

"""Lightweight realtime graph utilities driven by ctrl history.

This module keeps graph updates cheap enough for realtime usage by avoiding
inverse-dynamics or optimizer-heavy processing inside the runtime loop.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


STATIC_EDGE_SIGN = {
    "synergist": 1.0,
    "functional_group": 1.0,
    "antagonist": -1.0,
}


def build_node_features(
    ctrl: np.ndarray,
    ctrl_history: list[np.ndarray] | deque[np.ndarray],
    *,
    hold_window: int = 12,
) -> dict[str, np.ndarray]:
    ctrl_arr = np.asarray(ctrl, dtype=float).reshape(-1)
    if ctrl_history:
        hist = np.stack([np.asarray(item, dtype=float).reshape(-1) for item in ctrl_history], axis=0)
        moving_mean = np.mean(hist, axis=0)
        delta_ctrl = ctrl_arr - hist[-1]
        hold_hist = hist[-max(1, int(hold_window)) :]
        hold_mean = np.mean(hold_hist, axis=0)
    else:
        moving_mean = ctrl_arr.copy()
        delta_ctrl = np.zeros_like(ctrl_arr)
        hold_mean = ctrl_arr.copy()
    return {
        "activation": ctrl_arr.astype(np.float32),
        "ctrl": ctrl_arr.astype(np.float32),
        "moving_mean": moving_mean.astype(np.float32),
        "delta_ctrl": delta_ctrl.astype(np.float32),
        "hold_mean": hold_mean.astype(np.float32),
    }


def build_edge_matrix_static(
    node_names: list[str],
    static_edges: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    node_index = {str(name): idx for idx, name in enumerate(node_names)}
    matrix = np.zeros((len(node_names), len(node_names)), dtype=np.float32)
    visible_edges: list[dict[str, Any]] = []
    for edge in static_edges:
        source = str(edge.get("source_name"))
        target = str(edge.get("target_name"))
        relation = str(edge.get("edge_type", "functional_group"))
        if source not in node_index or target not in node_index:
            continue
        i = node_index[source]
        j = node_index[target]
        sign = float(STATIC_EDGE_SIGN.get(relation, 0.0))
        matrix[i, j] = sign
        matrix[j, i] = sign
        visible_edges.append(
            {
                "source_name": source,
                "target_name": target,
                "edge_type": relation,
                "dynamic_score": abs(sign),
                "pair_strength_score": abs(sign),
                "coactivation_score": 0.0,
            }
        )
    return matrix, visible_edges


def build_edge_matrix_dynamic_short_window(
    node_names: list[str],
    static_edges: list[dict[str, Any]],
    ctrl_history: list[np.ndarray] | deque[np.ndarray],
    *,
    node_activation_threshold: float,
    edge_score_threshold: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    node_index = {str(name): idx for idx, name in enumerate(node_names)}
    matrix = np.zeros((len(node_names), len(node_names)), dtype=np.float32)
    visible_edges: list[dict[str, Any]] = []
    if not ctrl_history:
        return matrix, visible_edges
    hist = np.stack([np.asarray(item, dtype=float).reshape(-1) for item in ctrl_history], axis=0)
    latest = hist[-1]
    for edge in static_edges:
        source = str(edge.get("source_name"))
        target = str(edge.get("target_name"))
        relation = str(edge.get("edge_type", "functional_group"))
        if source not in node_index or target not in node_index:
            continue
        i = node_index[source]
        j = node_index[target]
        if latest[i] < float(node_activation_threshold) or latest[j] < float(node_activation_threshold):
            continue
        pair_hist = hist[:, [i, j]]
        if pair_hist.shape[0] >= 2 and np.std(pair_hist[:, 0]) > 1e-6 and np.std(pair_hist[:, 1]) > 1e-6:
            corr = float(np.corrcoef(pair_hist[:, 0], pair_hist[:, 1])[0, 1])
        else:
            corr = 0.0
        coact = float(np.mean(pair_hist[:, 0] * pair_hist[:, 1]))
        relation_sign = float(STATIC_EDGE_SIGN.get(relation, 0.0))
        if relation_sign < 0.0:
            score = 0.5 * max(0.0, -corr) + 0.5 * coact
        else:
            score = 0.5 * max(0.0, corr) + 0.5 * coact
        if score < float(edge_score_threshold):
            continue
        signed_score = relation_sign * score
        matrix[i, j] = signed_score
        matrix[j, i] = signed_score
        visible_edges.append(
            {
                "source_name": source,
                "target_name": target,
                "edge_type": relation,
                "dynamic_score": float(score),
                "pair_strength_score": float(abs(corr)),
                "coactivation_score": float(coact),
            }
        )
    visible_edges.sort(key=lambda item: float(item.get("dynamic_score", 0.0)), reverse=True)
    return matrix, visible_edges


def export_graph_snapshot(
    *,
    node_names: list[str],
    node_features: dict[str, np.ndarray],
    edge_matrix: np.ndarray,
    visible_edges: list[dict[str, Any]],
    edge_mode_used: str,
    timestamp: float | None,
    frame_index: int | None,
    edge_relation_filter: str,
) -> dict[str, Any]:
    x = np.stack(
        [
            np.asarray(node_features["activation"], dtype=np.float32),
            np.asarray(node_features["ctrl"], dtype=np.float32),
            np.asarray(node_features["moving_mean"], dtype=np.float32),
            np.asarray(node_features["delta_ctrl"], dtype=np.float32),
            np.asarray(node_features["hold_mean"], dtype=np.float32),
        ],
        axis=1,
    )
    top_indices = np.argsort(x[:, 0])[-6:][::-1] if x.shape[0] else np.asarray([], dtype=int)
    top_active = [
        {
            "node_name": str(node_names[idx]),
            "activation": float(x[idx, 0]),
            "ctrl_analysis": float(x[idx, 0]),
            "zone": "shared",
        }
        for idx in top_indices
    ]
    return {
        "timestamp": float(timestamp) if timestamp is not None else None,
        "frame_index": int(frame_index) if frame_index is not None else None,
        "node_names": list(node_names),
        "node_feature_names": ["activation", "ctrl", "moving_mean", "delta_ctrl", "hold_mean"],
        "x": x,
        "visible_edges": list(visible_edges),
        "dominant_visible_edges": list(visible_edges[:10]),
        "ctrl_analysis": np.asarray(node_features["activation"], dtype=np.float32),
        "ctrl_summary": {
            "ctrl_analysis": np.asarray(node_features["activation"], dtype=np.float32),
            "ctrl_rolling_mean": np.asarray(node_features["moving_mean"], dtype=np.float32),
            "ctrl_delta": np.asarray(node_features["delta_ctrl"], dtype=np.float32),
        },
        "analysis": {
            "top_active": top_active,
            "top_dynamic_edges": list(visible_edges[:8]),
            "dominant_visible_node_names": [str(item["node_name"]) for item in top_active],
            "dominant_nodes": top_active,
            "active_zones": [{"zone_display": "Shared", "mean_activation": float(np.mean(x[:, 0])) if len(x) else 0.0}],
        },
        "meta": {
            "graph_kind": "realtime_light_graph",
            "dynamic_edge_config": {
                "edge_relation_filter": str(edge_relation_filter),
                "edge_mode_used": str(edge_mode_used),
            },
        },
        "edge_matrix": np.asarray(edge_matrix, dtype=np.float32),
        "edge_mode_used": str(edge_mode_used),
    }


@dataclass
class RealtimeGraphRuntime:
    node_names: list[str]
    static_edges: list[dict[str, Any]]
    edge_mode: str = "static"
    history_size: int = 10
    node_activation_threshold: float = 0.05
    edge_score_threshold: float = 0.02
    edge_relation_filter: str = "all_active"
    hold_window: int = 12

    def __post_init__(self) -> None:
        self.ctrl_history: deque[np.ndarray] = deque(maxlen=max(2, int(self.history_size)))

    @classmethod
    def from_static_payload(
        cls,
        static_payload: dict[str, Any],
        *,
        edge_mode: str,
        history_size: int,
        node_activation_threshold: float,
        edge_score_threshold: float,
        edge_relation_filter: str,
    ) -> "RealtimeGraphRuntime":
        return cls(
            node_names=[str(name) for name in list(static_payload.get("node_names") or [])],
            static_edges=[dict(edge) for edge in list(static_payload.get("edges") or [])],
            edge_mode=str(edge_mode),
            history_size=int(history_size),
            node_activation_threshold=float(node_activation_threshold),
            edge_score_threshold=float(edge_score_threshold),
            edge_relation_filter=str(edge_relation_filter),
        )

    def update(
        self,
        *,
        ctrl_values: np.ndarray | list[float],
        timestamp: float | None = None,
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        ctrl_arr = np.asarray(ctrl_values, dtype=float).reshape(-1)
        if ctrl_arr.shape[0] < len(self.node_names):
            padded = np.zeros(len(self.node_names), dtype=float)
            padded[: ctrl_arr.shape[0]] = ctrl_arr
            ctrl_arr = padded
        elif ctrl_arr.shape[0] > len(self.node_names):
            ctrl_arr = ctrl_arr[: len(self.node_names)]
        self.ctrl_history.append(ctrl_arr.copy())
        node_features = build_node_features(ctrl_arr, self.ctrl_history, hold_window=int(self.hold_window))
        if str(self.edge_mode) == "dynamic_short_window":
            edge_matrix, visible_edges = build_edge_matrix_dynamic_short_window(
                self.node_names,
                self.static_edges,
                self.ctrl_history,
                node_activation_threshold=float(self.node_activation_threshold),
                edge_score_threshold=float(self.edge_score_threshold),
            )
        else:
            edge_matrix, visible_edges = build_edge_matrix_static(self.node_names, self.static_edges)
        return export_graph_snapshot(
            node_names=self.node_names,
            node_features=node_features,
            edge_matrix=edge_matrix,
            visible_edges=visible_edges,
            edge_mode_used=self.edge_mode,
            timestamp=timestamp,
            frame_index=frame_index,
            edge_relation_filter=self.edge_relation_filter,
        )
