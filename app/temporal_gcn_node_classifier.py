from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from app.gcn_node_classifier import build_adjacency_matrix, normalized_adjacency


class DenseTemporalGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x_seq: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = self.linear(x_seq)
        if a_norm.ndim == 2:
            return torch.einsum("ij,tjf->tif", a_norm, h)
        return torch.einsum("tij,tjf->tif", a_norm, h)


def normalized_dynamic_adjacency(adjacency_seq: torch.Tensor) -> torch.Tensor:
    a = adjacency_seq.float()
    if a.ndim == 2:
        return normalized_adjacency(a)
    degree = torch.sum(torch.abs(a), dim=2)
    inv_sqrt = torch.pow(torch.clamp(degree, min=1e-6), -0.5)
    return inv_sqrt[:, :, None] * a * inv_sqrt[:, None, :]


class EdgeTypeTemporalGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_channels: int = 3):
        super().__init__()
        self.channel_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(int(num_channels))])
        self.self_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x_seq: torch.Tensor, type_adj_seq: torch.Tensor) -> torch.Tensor:
        outputs = [self.self_linear(x_seq)]
        for channel_idx, linear in enumerate(self.channel_linears):
            a_norm = normalized_dynamic_adjacency(type_adj_seq[channel_idx])
            outputs.append(torch.einsum("tij,tjf->tif", a_norm, linear(x_seq)))
        return torch.stack(outputs, dim=0).mean(dim=0)


class TemporalGCNNodeClassifier(nn.Module):
    """Node-level Temporal GCN for muscle deviation classification.

    Input is one trial graph sequence:
    - x_seq: [sequence_step, muscle_node, node_feature]
    - adjacency_seq: dynamic edge weight sequence, one graph per step

    Output is one binary prediction per muscle node.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        temporal_dim: int,
        num_classes: int,
        dropout: float,
        variant: str = "dynamic",
        temporal_pooling: str = "last",
    ):
        super().__init__()
        self.variant = str(variant)
        self.temporal_pooling = str(temporal_pooling)
        if self.variant == "edge_type":
            self.gcn1 = EdgeTypeTemporalGraphConv(in_dim, hidden_dim)
            self.gcn2 = EdgeTypeTemporalGraphConv(hidden_dim, hidden_dim)
        else:
            self.gcn1 = DenseTemporalGraphConv(in_dim, hidden_dim)
            self.gcn2 = DenseTemporalGraphConv(hidden_dim, hidden_dim)
        self.temporal = nn.GRU(input_size=hidden_dim, hidden_size=temporal_dim, batch_first=True)
        self.attention = nn.Linear(temporal_dim, 1) if self.temporal_pooling == "attention" else None
        self.head = nn.Linear(temporal_dim, num_classes)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x_seq: torch.Tensor, adjacency: torch.Tensor, adjacency_seq: torch.Tensor, type_adjacency_seq: torch.Tensor) -> torch.Tensor:
        if self.variant == "dynamic":
            graph_input = normalized_dynamic_adjacency(adjacency_seq)
        elif self.variant == "edge_type":
            graph_input = type_adjacency_seq
        else:
            graph_input = normalized_adjacency(adjacency)
        h = torch.relu(self.gcn1(x_seq, graph_input))
        h = self.dropout(h)
        h = torch.relu(self.gcn2(h, graph_input))
        h = self.dropout(h)
        h_node_time = h.permute(1, 0, 2).contiguous()
        temporal_out, _ = self.temporal(h_node_time)
        if self.attention is not None:
            attention_logits = self.attention(temporal_out).squeeze(-1)
            attention_weights = torch.softmax(attention_logits, dim=1)
            node_repr = torch.sum(temporal_out * attention_weights.unsqueeze(-1), dim=1)
        else:
            node_repr = temporal_out[:, -1, :]
        return self.head(self.dropout(node_repr))


def dynamic_adjacency_seq(trial: dict[str, Any]) -> np.ndarray:
    node_count = int(len(trial["node_names"]))
    edge_weight_seq = np.asarray(trial["edge_weight_seq"], dtype=np.float32)
    if edge_weight_seq.ndim != 2:
        raise ValueError(f"Expected edge_weight_seq rank-2, got {edge_weight_seq.shape}")
    out = np.zeros((edge_weight_seq.shape[0], node_count, node_count), dtype=np.float32)
    for edge in list(trial["edges"]):
        src = int(edge["source_idx"])
        dst = int(edge["target_idx"])
        edge_idx = int(edge["edge_idx"])
        if 0 <= edge_idx < edge_weight_seq.shape[1]:
            values = edge_weight_seq[:, edge_idx].astype(np.float32)
            out[:, src, dst] = values
            out[:, dst, src] = values
    diag = np.arange(node_count)
    out[:, diag, diag] = 1.0
    return out


def type_adjacency_seq(trial: dict[str, Any]) -> np.ndarray:
    node_count = int(len(trial["node_names"]))
    edge_weight_seq = np.asarray(trial["edge_weight_seq"], dtype=np.float32)
    if edge_weight_seq.ndim != 2:
        raise ValueError(f"Expected edge_weight_seq rank-2, got {edge_weight_seq.shape}")
    out = np.zeros((3, edge_weight_seq.shape[0], node_count, node_count), dtype=np.float32)
    for edge in list(trial["edges"]):
        src = int(edge["source_idx"])
        dst = int(edge["target_idx"])
        edge_idx = int(edge["edge_idx"])
        edge_type = str(edge.get("edge_type") or "").strip().lower()
        if edge_type == "synergist":
            channel = 0
        elif edge_type == "antagonist":
            channel = 1
        else:
            channel = 2
        if 0 <= edge_idx < edge_weight_seq.shape[1]:
            values = edge_weight_seq[:, edge_idx].astype(np.float32)
            out[channel, :, src, dst] = values
            out[channel, :, dst, src] = values
    diag = np.arange(node_count)
    out[:, :, diag, diag] = 1.0
    return out


def is_temporal_gcn_checkpoint(checkpoint: dict[str, Any]) -> bool:
    run_config = dict(checkpoint.get("run_config") or {})
    model_name = str(run_config.get("model") or "").lower()
    return "temporal_gcn" in model_name or "temporal_dim" in run_config or "best_temporal_gcn" in str(checkpoint.get("checkpoint_name", "")).lower()


def predict_temporal_gcn_for_trial(trial: dict[str, Any], checkpoint: dict[str, Any]) -> tuple[np.ndarray, list[str], str]:
    class_names = [str(item) for item in checkpoint.get("class_names") or ["within_reference", "deviating"]]
    run_config = dict(checkpoint.get("run_config") or {})
    variant = str(run_config.get("model_variant") or "dynamic")
    x_seq = np.asarray(trial["node_feature_seq"], dtype=np.float32)
    if x_seq.ndim != 3:
        raise ValueError(f"Expected node_feature_seq rank-3, got {x_seq.shape}")
    if bool(run_config.get("add_progress_feature", False)):
        step_count, node_count, _ = x_seq.shape
        progress = np.linspace(0.0, 1.0, max(1, int(step_count)), dtype=np.float32).reshape(step_count, 1, 1)
        progress = np.repeat(progress, int(node_count), axis=1)
        x_seq = np.concatenate([x_seq, progress], axis=2).astype(np.float32)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(1, 1, -1)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(1, 1, -1)
    if int(x_seq.shape[-1]) != int(feature_mean.shape[-1]):
        feature_mode = str((trial.get("metadata") or {}).get("node_feature_mode") or "unknown")
        raise ValueError(
            f"Graph feature dimension {int(x_seq.shape[-1])} does not match checkpoint dimension "
            f"{int(feature_mean.shape[-1])}. node_feature_mode={feature_mode}. "
            "Use a checkpoint trained with the same node feature contract."
        )
    x_std = (x_seq - feature_mean) / np.where(feature_std > 1e-8, feature_std, 1.0)
    adjacency = build_adjacency_matrix(trial)
    adjacency_seq = dynamic_adjacency_seq(trial)
    type_adj_seq = type_adjacency_seq(trial)
    model = TemporalGCNNodeClassifier(
        in_dim=int(x_std.shape[-1]),
        hidden_dim=int(run_config.get("hidden_dim", 32)),
        temporal_dim=int(run_config.get("temporal_dim", 32)),
        num_classes=int(len(class_names)),
        dropout=float(run_config.get("dropout", 0.0)),
        variant=variant,
        temporal_pooling=str(run_config.get("temporal_pooling") or "last"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(x_std, dtype=torch.float32),
            torch.tensor(adjacency, dtype=torch.float32),
            torch.tensor(adjacency_seq, dtype=torch.float32),
            torch.tensor(type_adj_seq, dtype=torch.float32),
        )
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities, class_names, variant
