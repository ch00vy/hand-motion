from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gcn_node_classifier import build_adjacency_matrix, normalized_adjacency, set_seed  # noqa: E402
from app.graph_sequence_baseline import load_trial_graph_sequence  # noqa: E402
from scripts.offline_biomech.run_binary_node_deviation_smoke import (  # noqa: E402
    BINARY_CLASSES,
    _build_trial_audits,
    _label_distribution,
    _markdown_table,
    binary_label,
)
from scripts.offline_biomech.train_final_binary_gcn import _collect_node_rows, _safe_float  # noqa: E402


DEFAULT_DATASET_DIR = PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous" / "graph_sequence_dataset"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous" / "reference_sequence" / "reference"
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous" / "reference_sequence"
DEFAULT_ABNORMAL_DATASET_DIR = PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous" / "abnormal_graph_sequence_dataset"
DEFAULT_VANILLA_METRICS = PROJECT_ROOT / "artifacts" / "final_binary_gcn_run_gui_plus_previous" / "eval_metrics.json"
DEFAULT_VANILLA_ABNORMAL_METRICS = PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous" / "abnormal_validation_eval" / "validation_metrics.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "temporal_gcn_prototype"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass
class TemporalGraphSample:
    trial_id: str
    cohort: str
    exercise_name: str
    node_names: list[str]
    feature_names: list[str]
    x_seq: np.ndarray
    adjacency: np.ndarray
    adjacency_seq: np.ndarray
    type_adjacency_seq: np.ndarray
    y: np.ndarray
    split: str


def _append_progress_feature(samples: list[TemporalGraphSample]) -> list[TemporalGraphSample]:
    out: list[TemporalGraphSample] = []
    for sample in samples:
        x_seq = np.asarray(sample.x_seq, dtype=np.float32)
        step_count, node_count, _ = x_seq.shape
        progress = np.linspace(0.0, 1.0, max(1, int(step_count)), dtype=np.float32).reshape(step_count, 1, 1)
        progress = np.repeat(progress, int(node_count), axis=1)
        x_aug = np.concatenate([x_seq, progress], axis=2)
        out.append(
            TemporalGraphSample(
                trial_id=sample.trial_id,
                cohort=sample.cohort,
                exercise_name=sample.exercise_name,
                node_names=list(sample.node_names),
                feature_names=[*sample.feature_names, "progress_norm"],
                x_seq=x_aug.astype(np.float32),
                adjacency=np.asarray(sample.adjacency, dtype=np.float32),
                adjacency_seq=np.asarray(sample.adjacency_seq, dtype=np.float32),
                type_adjacency_seq=np.asarray(sample.type_adjacency_seq, dtype=np.float32),
                y=np.asarray(sample.y, dtype=np.int64),
                split=str(sample.split),
            )
        )
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_split_manifest(reference_root_dir: Path, train_reference_trials: int) -> pd.DataFrame:
    reference_pool = pd.read_csv(reference_root_dir / "reference_pool_manifest.csv")
    holdout = pd.read_csv(reference_root_dir / "holdout_manifest.csv")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(reference_pool.to_dict(orient="records")):
        item = dict(row)
        item["split"] = "train" if idx < int(train_reference_trials) else "val"
        rows.append(item)
    for row in holdout.to_dict(orient="records"):
        item = dict(row)
        item["split"] = "test"
        rows.append(item)
    return pd.DataFrame(rows)


def _json_by_trial(dataset_dir: Path) -> dict[str, Path]:
    manifest = pd.read_csv(dataset_dir / "dataset_manifest.csv")
    return {str(row["trial_id"]): Path(str(row["json_path"])) for row in manifest.to_dict(orient="records")}


def _build_temporal_samples(
    *,
    dataset_dir: Path,
    reference_dir: Path,
    split_manifest_path: Path,
    binary_threshold: float,
    onset_ratio: float,
    time_region_width: int,
) -> tuple[dict[str, list[TemporalGraphSample]], pd.DataFrame]:
    audits = _build_trial_audits(
        dataset_dir=dataset_dir,
        reference_dir=reference_dir,
        split_manifest_path=split_manifest_path,
        onset_ratio=float(onset_ratio),
        time_region_width=int(time_region_width),
    )
    json_paths = _json_by_trial(dataset_dir)
    split_samples: dict[str, list[TemporalGraphSample]] = {"train": [], "val": [], "test": []}
    node_rows = _collect_node_rows(audits)
    label_by_trial: dict[str, np.ndarray] = {}
    for audit in audits:
        labels = [
            BINARY_CLASSES.index(binary_label(dict(row), deviation_threshold=float(binary_threshold)))
            for row in audit.node_summary_df.to_dict(orient="records")
        ]
        label_by_trial[str(audit.trial_id)] = np.asarray(labels, dtype=np.int64)

    for audit in audits:
        trial_id = str(audit.trial_id)
        trial = load_trial_graph_sequence(json_paths[trial_id])
        x_seq = np.asarray(trial["node_feature_seq"], dtype=np.float32)
        if x_seq.ndim != 3:
            raise ValueError(f"Expected node_feature_seq rank-3 for {trial_id}, got {x_seq.shape}")
        sample = TemporalGraphSample(
            trial_id=trial_id,
            cohort=str(trial["metadata"].get("cohort") or ""),
            exercise_name=str(trial["metadata"].get("exercise_name") or ""),
            node_names=list(trial["node_names"]),
            feature_names=list(trial["node_feature_names"]),
            x_seq=x_seq,
            adjacency=build_adjacency_matrix(trial),
            adjacency_seq=_dynamic_adjacency_seq(trial),
            type_adjacency_seq=_type_adjacency_seq(trial),
            y=label_by_trial[trial_id],
            split=str(audit.split),
        )
        split_samples.setdefault(sample.split, []).append(sample)
    return split_samples, pd.DataFrame(node_rows)


def _dynamic_adjacency_seq(trial: dict[str, Any]) -> np.ndarray:
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


def _type_adjacency_seq(trial: dict[str, Any]) -> np.ndarray:
    node_count = int(len(trial["node_names"]))
    edge_weight_seq = np.asarray(trial["edge_weight_seq"], dtype=np.float32)
    if edge_weight_seq.ndim != 2:
        raise ValueError(f"Expected edge_weight_seq rank-2, got {edge_weight_seq.shape}")
    # Channels: synergist, antagonist, functional/other. Keep signed dynamic
    # weights so antagonist relations can carry inhibitory/negative evidence.
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


def _fit_temporal_standardizer(samples: list[TemporalGraphSample]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("Cannot fit temporal standardizer on empty samples.")
    stacked = np.concatenate([sample.x_seq.reshape(-1, sample.x_seq.shape[-1]) for sample in samples], axis=0)
    mean = np.mean(stacked, axis=0).astype(np.float32)
    std = np.std(stacked, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _standardize_temporal_samples(samples: list[TemporalGraphSample], mean: np.ndarray, std: np.ndarray) -> list[TemporalGraphSample]:
    out: list[TemporalGraphSample] = []
    for sample in samples:
        x_std = (np.asarray(sample.x_seq, dtype=np.float32) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        out.append(
            TemporalGraphSample(
                trial_id=sample.trial_id,
                cohort=sample.cohort,
                exercise_name=sample.exercise_name,
                node_names=list(sample.node_names),
                feature_names=list(sample.feature_names),
                x_seq=x_std.astype(np.float32),
                adjacency=np.asarray(sample.adjacency, dtype=np.float32),
                adjacency_seq=np.asarray(sample.adjacency_seq, dtype=np.float32),
                type_adjacency_seq=np.asarray(sample.type_adjacency_seq, dtype=np.float32),
                y=np.asarray(sample.y, dtype=np.int64),
                split=str(sample.split),
            )
        )
    return out


class DenseTemporalGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x_seq: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        # x_seq: [T, N, F]
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
        # type_adj_seq: [C, T, N, N]
        outputs = [self.self_linear(x_seq)]
        for channel_idx, linear in enumerate(self.channel_linears):
            a_norm = normalized_dynamic_adjacency(type_adj_seq[channel_idx])
            outputs.append(torch.einsum("tij,tjf->tif", a_norm, linear(x_seq)))
        return torch.stack(outputs, dim=0).mean(dim=0)


class TemporalGCNNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, temporal_dim: int, num_classes: int, dropout: float, variant: str, temporal_pooling: str = "last"):
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
        # [T, N, H] -> [N, T, H], one temporal sequence per muscle node.
        h_node_time = h.permute(1, 0, 2).contiguous()
        temporal_out, _ = self.temporal(h_node_time)
        if self.attention is not None:
            attention_logits = self.attention(temporal_out).squeeze(-1)
            attention_weights = torch.softmax(attention_logits, dim=1)
            node_repr = torch.sum(temporal_out * attention_weights.unsqueeze(-1), dim=1)
        else:
            node_repr = temporal_out[:, -1, :]
        node_repr = self.dropout(node_repr)
        return self.head(node_repr)


def _sample_to_tensors(sample: TemporalGraphSample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(sample.x_seq, dtype=torch.float32, device=device),
        torch.tensor(sample.adjacency, dtype=torch.float32, device=device),
        torch.tensor(sample.adjacency_seq, dtype=torch.float32, device=device),
        torch.tensor(sample.type_adjacency_seq, dtype=torch.float32, device=device),
        torch.tensor(sample.y, dtype=torch.long, device=device),
    )


def _run_epoch(
    model: nn.Module,
    samples: list[TemporalGraphSample],
    *,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_nodes = 0
    total_correct = 0
    for sample in samples:
        x_seq, adjacency, adjacency_seq, type_adjacency_seq, y = _sample_to_tensors(sample, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x_seq, adjacency, adjacency_seq, type_adjacency_seq)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            optimizer.step()
        pred = torch.argmax(logits, dim=1)
        total_loss += float(loss.detach().cpu()) * int(y.shape[0])
        total_correct += int((pred == y).sum().item())
        total_nodes += int(y.shape[0])
    return {
        "loss": float(total_loss / max(1, total_nodes)),
        "accuracy": float(total_correct / max(1, total_nodes)),
        "num_nodes": int(total_nodes),
    }


def _predict(model: nn.Module, samples: list[TemporalGraphSample], device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            x_seq, adjacency, adjacency_seq, type_adjacency_seq, y = _sample_to_tensors(sample, device)
            logits = model(x_seq, adjacency, adjacency_seq, type_adjacency_seq)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            for node_idx, node_name in enumerate(sample.node_names):
                target_idx = int(y[node_idx].item())
                pred_idx = int(pred[node_idx].item())
                rows.append(
                    {
                        "split": str(sample.split),
                        "trial_id": str(sample.trial_id),
                        "cohort": str(sample.cohort),
                        "exercise_name": str(sample.exercise_name),
                        "node_index": int(node_idx),
                        "node_name": str(node_name),
                        "target_class": BINARY_CLASSES[target_idx],
                        "predicted_class": BINARY_CLASSES[pred_idx],
                        "prob_within_reference": float(probs[node_idx, 0].detach().cpu()),
                        "prob_deviating": float(probs[node_idx, 1].detach().cpu()),
                        "predicted_confidence": float(probs[node_idx, pred_idx].detach().cpu()),
                    }
                )
    return pd.DataFrame(rows)


def _evaluate(pred_df: pd.DataFrame) -> dict[str, Any]:
    if pred_df.empty:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "binary_f1_deviating": 0.0,
            "precision_deviating": 0.0,
            "recall_deviating": 0.0,
            "roc_auc": None,
            "average_precision": None,
            "majority_baseline_accuracy": 0.0,
            "beats_majority_accuracy": False,
            "confusion_matrix": [],
            "classification_report": {},
        }
    y_true = np.asarray([BINARY_CLASSES.index(str(item)) for item in pred_df["target_class"].tolist()], dtype=int)
    y_pred = np.asarray([BINARY_CLASSES.index(str(item)) for item in pred_df["predicted_class"].tolist()], dtype=int)
    prob_deviating = pd.to_numeric(pred_df["prob_deviating"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=list(BINARY_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    majority = float(pd.Series(pred_df["target_class"].astype(str)).value_counts(normalize=True).max())
    try:
        roc_auc = float(roc_auc_score(y_true, prob_deviating)) if len(set(y_true.tolist())) == 2 else None
    except Exception:
        roc_auc = None
    try:
        avg_precision = float(average_precision_score(y_true, prob_deviating)) if len(set(y_true.tolist())) == 2 else None
    except Exception:
        avg_precision = None
    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "binary_f1_deviating": float(f1),
        "precision_deviating": float(precision),
        "recall_deviating": float(recall),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "majority_baseline_accuracy": majority,
        "beats_majority_accuracy": bool(float(report.get("accuracy", 0.0)) > majority),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": report,
    }


def _compute_class_weights(samples: list[TemporalGraphSample]) -> torch.Tensor:
    labels = np.concatenate([sample.y for sample in samples], axis=0)
    counts = np.bincount(labels, minlength=len(BINARY_CLASSES)).astype(np.float32)
    counts = np.where(counts > 0.0, counts, 1.0)
    return torch.tensor(counts.sum() / (counts * float(len(BINARY_CLASSES))), dtype=torch.float32)


def _write_confusion_csv(path: Path, conf: list[list[int]]) -> None:
    rows = []
    for idx, label in enumerate(BINARY_CLASSES):
        row = {"target_class": label}
        for pred_idx, pred_label in enumerate(BINARY_CLASSES):
            row[pred_label] = int(conf[idx][pred_idx]) if idx < len(conf) and pred_idx < len(conf[idx]) else 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _load_vanilla_metric(path: Path, split: str) -> dict[str, Any]:
    payload = _read_json(path)
    if "metrics_against_deviation_rule" in payload:
        return dict(payload["metrics_against_deviation_rule"])
    return dict(payload.get(split, {}))


def _comparison_rows(temporal_metrics: dict[str, Any], vanilla_metrics_path: Path, vanilla_abnormal_metrics_path: Path, *, temporal_model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, vanilla_path, vanilla_split in (
        ("test", vanilla_metrics_path, "test"),
        ("abnormal", vanilla_abnormal_metrics_path, "test"),
    ):
        temporal = temporal_metrics.get(split, {})
        vanilla = _load_vanilla_metric(vanilla_path, vanilla_split)
        for model_name, metrics in (("vanilla_gcn", vanilla), (temporal_model_name, temporal)):
            rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "accuracy": _safe_float(metrics.get("accuracy"), 0.0),
                    "macro_f1": _safe_float(metrics.get("macro_f1"), 0.0),
                    "weighted_f1": _safe_float(metrics.get("weighted_f1"), 0.0),
                    "deviating_f1": _safe_float(metrics.get("binary_f1_deviating"), 0.0),
                    "deviating_precision": _safe_float(metrics.get("precision_deviating"), 0.0),
                    "deviating_recall": _safe_float(metrics.get("recall_deviating"), 0.0),
                    "majority_baseline": _safe_float(metrics.get("majority_baseline_accuracy"), 0.0),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    *,
    report_path: Path,
    metrics: dict[str, Any],
    comparison_df: pd.DataFrame,
    dataset_summary: dict[str, Any],
    best_epoch: int,
) -> None:
    test = metrics.get("test", {})
    abnormal = metrics.get("abnormal", {})
    if comparison_df.empty:
        verdict = "Temporal GCN prototype completed, but comparison data is missing."
    else:
        test_rows = comparison_df.loc[comparison_df["split"] == "test"].set_index("model")
        abnormal_rows = comparison_df.loc[comparison_df["split"] == "abnormal"].set_index("model")
        test_delta = 0.0
        abnormal_delta = 0.0
        temporal_names = [str(item) for item in test_rows.index if str(item).startswith("temporal_gcn")]
        temporal_name = temporal_names[0] if temporal_names else "temporal_gcn_gru"
        if {temporal_name, "vanilla_gcn"}.issubset(test_rows.index):
            test_delta = float(test_rows.loc[temporal_name, "macro_f1"] - test_rows.loc["vanilla_gcn", "macro_f1"])
        if {temporal_name, "vanilla_gcn"}.issubset(abnormal_rows.index):
            abnormal_delta = float(abnormal_rows.loc[temporal_name, "macro_f1"] - abnormal_rows.loc["vanilla_gcn", "macro_f1"])
        if test_delta > 0.02 and abnormal_delta >= -0.02:
            verdict = "Temporal GCN is promising as a next experiment, but still prototype-level."
        elif test_delta <= 0.0 and abnormal_delta <= 0.0:
            verdict = "Temporal GCN does not currently beat vanilla GCN; keep vanilla as main baseline."
        else:
            verdict = "Temporal GCN is mixed; continue only after more abnormal trials and repeated splits."

    report_path.write_text(
        "\n".join(
            [
                "# Temporal GCN Prototype Report",
                "",
                "Model: shared dense GCN per graph step + GRU over time per muscle node.",
                "",
                f"Best epoch: `{best_epoch}`",
                "",
                "## Dataset Summary",
                "```json",
                json.dumps(dataset_summary, indent=2),
                "```",
                "",
                "## Holdout Test Metrics",
                f"- Accuracy: {test.get('accuracy', 0.0):.4f}",
                f"- Macro F1: {test.get('macro_f1', 0.0):.4f}",
                f"- Deviating F1: {test.get('binary_f1_deviating', 0.0):.4f}",
                f"- Majority baseline: {test.get('majority_baseline_accuracy', 0.0):.4f}",
                "",
                "## Abnormal Validation Metrics",
                f"- Accuracy: {abnormal.get('accuracy', 0.0):.4f}",
                f"- Macro F1: {abnormal.get('macro_f1', 0.0):.4f}",
                f"- Deviating F1: {abnormal.get('binary_f1_deviating', 0.0):.4f}",
                f"- Majority baseline: {abnormal.get('majority_baseline_accuracy', 0.0):.4f}",
                "",
                "## Vanilla GCN vs Temporal GCN",
                _markdown_table(comparison_df),
                "",
                "## Verdict",
                verdict,
                "",
                "Do not present this as final. This is a prototype check to decide whether temporal modeling is worth continuing.",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a small temporal GCN prototype for binary node deviation classification.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--reference-root-dir", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--abnormal-dataset-dir", default=str(DEFAULT_ABNORMAL_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-variant", default="static", choices=("static", "dynamic", "edge_type"), help="static uses averaged adjacency; dynamic uses edge_weight_seq[t]; edge_type uses dynamic synergist/antagonist/functional channels.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--vanilla-metrics-json", default=str(DEFAULT_VANILLA_METRICS))
    parser.add_argument("--vanilla-abnormal-metrics-json", default=str(DEFAULT_VANILLA_ABNORMAL_METRICS))
    parser.add_argument("--train-reference-trials", type=int, default=66)
    parser.add_argument("--binary-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--onset-ratio", type=float, default=0.20)
    parser.add_argument("--time-region-width", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--temporal-dim", type=int, default=32)
    parser.add_argument("--temporal-pooling", default="last", choices=("last", "attention"), help="last uses final GRU state; attention learns which progress steps matter per muscle node.")
    parser.add_argument("--add-progress-feature", action="store_true", help="Append normalized progress 0..1 as an explicit per-step node feature.")
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weighting", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    random.seed(int(args.seed))
    output_dir = Path(args.output_dir).resolve()
    if output_dir == DEFAULT_OUTPUT_DIR.resolve() and str(args.model_variant) != "static":
        output_dir = DEFAULT_OUTPUT_DIR.resolve().parent / f"temporal_gcn_prototype_{args.model_variant}"
    reports_dir = Path(args.reports_dir).resolve()
    if output_dir.exists() and bool(args.force_clean):
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = _load_split_manifest(Path(args.reference_root_dir).resolve(), int(args.train_reference_trials))
    split_manifest_path = output_dir / "split_manifest.csv"
    split_manifest.to_csv(split_manifest_path, index=False)

    tmp_audits = _build_trial_audits(
        dataset_dir=Path(args.dataset_dir).resolve(),
        reference_dir=Path(args.reference_dir).resolve(),
        split_manifest_path=split_manifest_path,
        onset_ratio=float(args.onset_ratio),
        time_region_width=int(args.time_region_width),
    )
    node_rows = _collect_node_rows(tmp_audits)
    train_scores = [_safe_float(row["average_deviation_score"]) for row in node_rows if str(row["split"]) == "train"]
    binary_threshold = float(np.quantile(train_scores, float(args.binary_threshold_quantile))) if train_scores else 0.0
    for row in node_rows:
        row["binary_label"] = binary_label(row, deviation_threshold=binary_threshold)
    label_dist = _label_distribution(node_rows, "binary_label", BINARY_CLASSES)
    label_dist.to_csv(output_dir / "label_distribution.csv", index=False)

    split_samples, node_summary = _build_temporal_samples(
        dataset_dir=Path(args.dataset_dir).resolve(),
        reference_dir=Path(args.reference_dir).resolve(),
        split_manifest_path=split_manifest_path,
        binary_threshold=binary_threshold,
        onset_ratio=float(args.onset_ratio),
        time_region_width=int(args.time_region_width),
    )
    if bool(args.add_progress_feature):
        split_samples["train"] = _append_progress_feature(split_samples["train"])
        split_samples["val"] = _append_progress_feature(split_samples["val"])
        split_samples["test"] = _append_progress_feature(split_samples["test"])
    node_summary.to_csv(output_dir / "normal_node_summary.csv", index=False)
    feature_mean, feature_std = _fit_temporal_standardizer(split_samples["train"])
    train_samples = _standardize_temporal_samples(split_samples["train"], feature_mean, feature_std)
    val_samples = _standardize_temporal_samples(split_samples["val"], feature_mean, feature_std)
    test_samples = _standardize_temporal_samples(split_samples["test"], feature_mean, feature_std)

    dataset_summary = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "reference_dir": str(Path(args.reference_dir).resolve()),
        "abnormal_dataset_dir": str(Path(args.abnormal_dataset_dir).resolve()),
        "num_train_trials": int(len(train_samples)),
        "num_val_trials": int(len(val_samples)),
        "num_test_trials": int(len(test_samples)),
        "num_nodes": int(len(train_samples[0].node_names)) if train_samples else 0,
        "num_windows": int(train_samples[0].x_seq.shape[0]) if train_samples else 0,
        "features_per_node_per_window": int(train_samples[0].x_seq.shape[-1]) if train_samples else 0,
        "binary_threshold": float(binary_threshold),
        "binary_threshold_quantile": float(args.binary_threshold_quantile),
        "model": f"temporal_gcn_gru_{args.model_variant}",
        "model_variant": str(args.model_variant),
        "temporal_pooling": str(args.temporal_pooling),
        "add_progress_feature": bool(args.add_progress_feature),
    }
    _write_json(output_dir / "dataset_summary.json", dataset_summary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGCNNodeClassifier(
        in_dim=int(train_samples[0].x_seq.shape[-1]),
        hidden_dim=int(args.hidden_dim),
        temporal_dim=int(args.temporal_dim),
        num_classes=len(BINARY_CLASSES),
        dropout=float(args.dropout),
        variant=str(args.model_variant),
        temporal_pooling=str(args.temporal_pooling),
    ).to(device)
    class_weights = _compute_class_weights(train_samples).to(device) if bool(args.class_weighting) else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    run_config = {
        "model": f"temporal_gcn_gru_{args.model_variant}",
        "model_variant": str(args.model_variant),
        "classification_task": "binary_node_deviation",
        "classes": list(BINARY_CLASSES),
        "hidden_dim": int(args.hidden_dim),
        "temporal_dim": int(args.temporal_dim),
        "temporal_pooling": str(args.temporal_pooling),
        "add_progress_feature": bool(args.add_progress_feature),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "class_weighting": bool(args.class_weighting),
        "binary_threshold": float(binary_threshold),
    }
    _write_json(output_dir / "run_config.json", run_config)

    history_rows: list[dict[str, Any]] = []
    best_state = None
    best_epoch = -1
    best_val_macro = -1.0
    stale_epochs = 0
    for epoch_idx in range(int(args.epochs)):
        train_stats = _run_epoch(model, train_samples, optimizer=optimizer, criterion=criterion, device=device)
        val_stats = _run_epoch(model, val_samples, optimizer=None, criterion=criterion, device=device)
        train_metrics = _evaluate(_predict(model, train_samples, device))
        val_metrics = _evaluate(_predict(model, val_samples, device))
        history_rows.append(
            {
                "epoch": int(epoch_idx + 1),
                "train_loss": float(train_stats["loss"]),
                "val_loss": float(val_stats["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "train_binary_f1_deviating": float(train_metrics["binary_f1_deviating"]),
                "val_binary_f1_deviating": float(val_metrics["binary_f1_deviating"]),
            }
        )
        val_macro = float(val_metrics["macro_f1"])
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_epoch = int(epoch_idx + 1)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "feature_names": list(train_samples[0].feature_names),
                    "class_names": list(BINARY_CLASSES),
                    "run_config": run_config,
                    "best_epoch": int(best_epoch),
                },
                output_dir / "checkpoints" / "best_temporal_gcn.pt",
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(args.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(output_dir / "history.csv", index=False)

    pred_train = _predict(model, train_samples, device)
    pred_val = _predict(model, val_samples, device)
    pred_test = _predict(model, test_samples, device)
    pred_train.to_csv(output_dir / "predictions_train.csv", index=False)
    pred_val.to_csv(output_dir / "predictions_val.csv", index=False)
    pred_test.to_csv(output_dir / "predictions_test.csv", index=False)

    metrics: dict[str, Any] = {
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(history_df)),
        "device": str(device),
        "binary_threshold": float(binary_threshold),
        "train": _evaluate(pred_train),
        "val": _evaluate(pred_val),
        "test": _evaluate(pred_test),
    }
    _write_confusion_csv(output_dir / "confusion_matrix_test.csv", metrics["test"]["confusion_matrix"])

    abnormal_dataset_dir = Path(args.abnormal_dataset_dir).resolve()
    if abnormal_dataset_dir.exists() and (abnormal_dataset_dir / "dataset_manifest.csv").exists():
        abnormal_manifest = pd.read_csv(abnormal_dataset_dir / "dataset_manifest.csv")
        abnormal_split = abnormal_manifest.loc[abnormal_manifest["cohort"].astype(str) == "abnormal"].copy()
        if not abnormal_split.empty:
            abnormal_split["split"] = "test"
            abnormal_split_path = output_dir / "abnormal_split_manifest.csv"
            abnormal_split.to_csv(abnormal_split_path, index=False)
            abnormal_samples_raw, abnormal_node_summary = _build_temporal_samples(
                dataset_dir=abnormal_dataset_dir,
                reference_dir=Path(args.reference_dir).resolve(),
                split_manifest_path=abnormal_split_path,
                binary_threshold=binary_threshold,
                onset_ratio=float(args.onset_ratio),
                time_region_width=int(args.time_region_width),
            )
            if bool(args.add_progress_feature):
                abnormal_samples_raw["test"] = _append_progress_feature(abnormal_samples_raw["test"])
            abnormal_samples = _standardize_temporal_samples(abnormal_samples_raw["test"], feature_mean, feature_std)
            pred_abnormal = _predict(model, abnormal_samples, device)
            pred_abnormal.to_csv(output_dir / "predictions_abnormal_validation.csv", index=False)
            abnormal_node_summary.to_csv(output_dir / "abnormal_node_summary.csv", index=False)
            metrics["abnormal"] = _evaluate(pred_abnormal)
            _write_confusion_csv(output_dir / "confusion_matrix_abnormal.csv", metrics["abnormal"]["confusion_matrix"])
    _write_json(output_dir / "eval_metrics.json", metrics)

    comparison_df = _comparison_rows(
        metrics,
        Path(args.vanilla_metrics_json).resolve(),
        Path(args.vanilla_abnormal_metrics_json).resolve(),
        temporal_model_name=f"temporal_gcn_gru_{args.model_variant}",
    )
    comparison_df.to_csv(output_dir / "temporal_vs_vanilla_comparison.csv", index=False)
    _write_report(
        report_path=reports_dir / "temporal_gcn_prototype_report.md",
        metrics=metrics,
        comparison_df=comparison_df,
        dataset_summary=dataset_summary,
        best_epoch=int(best_epoch),
    )

    print(f"[INFO] Temporal GCN prototype output: {output_dir}")
    print(f"[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Test macro F1: {metrics['test']['macro_f1']:.4f}")
    if "abnormal" in metrics:
        print(f"[INFO] Abnormal macro F1: {metrics['abnormal']['macro_f1']:.4f}")
    print(f"[INFO] Report: {reports_dir / 'temporal_gcn_prototype_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
