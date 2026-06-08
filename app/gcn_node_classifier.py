from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from app.graph_sequence_baseline import compare_trial_to_reference, load_reference_sequence, load_trial_graph_sequence


PRIMARY_NODE_CLASSES = (
    "within_reference",
    "onset_shift",
    "too_weak",
    "too_high",
    "not_smooth",
)


@dataclass
class GraphNodeSample:
    trial_id: str
    cohort: str
    exercise_name: str
    node_names: list[str]
    feature_names: list[str]
    x: np.ndarray
    adjacency: np.ndarray
    y: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def primary_node_label(row: dict[str, Any]) -> str:
    onset_status = str(row.get("onset_status") or "on_time")
    magnitude_status = str(row.get("magnitude_status") or "within_range")
    smoothness_status = str(row.get("smoothness_status") or "within_range")
    if onset_status in {"late", "early", "missing"}:
        return "onset_shift"
    if magnitude_status == "too_weak":
        return "too_weak"
    if magnitude_status == "too_high":
        return "too_high"
    if smoothness_status == "not_smooth":
        return "not_smooth"
    return "within_reference"


def build_adjacency_matrix(trial: dict[str, Any]) -> np.ndarray:
    node_names = list(trial["node_names"])
    num_nodes = int(len(node_names))
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edge_weight_seq = np.asarray(trial["edge_weight_seq"], dtype=np.float32)
    mean_edge = np.mean(np.abs(edge_weight_seq), axis=0) if edge_weight_seq.size else np.zeros((0,), dtype=np.float32)
    for edge in list(trial["edges"]):
        src_idx = int(edge["source_idx"])
        dst_idx = int(edge["target_idx"])
        edge_idx = int(edge["edge_idx"])
        if 0 <= edge_idx < len(mean_edge):
            weight = float(mean_edge[edge_idx])
        else:
            weight = 0.0
        if weight <= 1e-8:
            weight = abs(_safe_float(edge.get("prior_weight"), 0.0))
        adjacency[src_idx, dst_idx] = max(adjacency[src_idx, dst_idx], weight)
        adjacency[dst_idx, src_idx] = max(adjacency[dst_idx, src_idx], weight)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def _incident_edge_stats(trial: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = int(len(trial["node_names"]))
    edge_weight_seq = np.asarray(trial["edge_weight_seq"], dtype=np.float32)
    if edge_weight_seq.ndim != 2 or edge_weight_seq.shape[0] == 0:
        return np.zeros((num_nodes,), dtype=np.float32), np.zeros((num_nodes,), dtype=np.float32)
    mean_signed = np.mean(edge_weight_seq, axis=0)
    mean_abs = np.mean(np.abs(edge_weight_seq), axis=0)
    abs_acc = np.zeros((num_nodes,), dtype=np.float32)
    signed_acc = np.zeros((num_nodes,), dtype=np.float32)
    counts = np.zeros((num_nodes,), dtype=np.float32)
    for edge in list(trial["edges"]):
        src_idx = int(edge["source_idx"])
        dst_idx = int(edge["target_idx"])
        edge_idx = int(edge["edge_idx"])
        if edge_idx >= len(mean_signed):
            continue
        abs_value = float(mean_abs[edge_idx])
        signed_value = float(mean_signed[edge_idx])
        abs_acc[src_idx] += abs_value
        abs_acc[dst_idx] += abs_value
        signed_acc[src_idx] += signed_value
        signed_acc[dst_idx] += signed_value
        counts[src_idx] += 1.0
        counts[dst_idx] += 1.0
    counts = np.where(counts > 0.0, counts, 1.0)
    return abs_acc / counts, signed_acc / counts


def build_node_feature_matrix(
    trial: dict[str, Any],
    comparison: dict[str, Any],
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    node_seq = np.asarray(trial["node_feature_seq"], dtype=np.float32)
    base_feature_names = list(trial["node_feature_names"])
    num_nodes = int(node_seq.shape[1])
    if node_seq.ndim != 3:
        raise ValueError(f"Expected node_feature_seq rank-3, got {node_seq.shape}")

    trial_mean = np.mean(node_seq, axis=0)
    trial_std = np.std(node_seq, axis=0)
    trial_max = np.max(node_seq, axis=0)
    edge_abs_incident, edge_signed_incident = _incident_edge_stats(trial)

    summary_df = comparison["node_summary_df"].copy()
    summary_df = summary_df.sort_values("node_index").reset_index(drop=True)
    rows: list[list[float]] = []
    feature_names: list[str] = []
    for prefix in ("mean", "std", "max"):
        for feature_name in base_feature_names:
            feature_names.append(f"{prefix}_{feature_name}")
    feature_names.extend(
        [
            "trial_onset_progress",
            "trial_peak_progress",
            "trial_peak_magnitude",
            "trial_smoothness",
            "reference_onset_progress",
            "reference_peak_progress",
            "reference_peak_magnitude",
            "reference_smoothness",
            "delta_onset_progress",
            "delta_peak_progress",
            "delta_peak_magnitude",
            "delta_smoothness",
            "incident_abs_edge_weight",
            "incident_signed_edge_weight",
        ]
    )
    for node_idx in range(num_nodes):
        summary_row = dict(summary_df.iloc[node_idx].to_dict()) if node_idx < len(summary_df) else {}
        row = []
        row.extend(trial_mean[node_idx].tolist())
        row.extend(trial_std[node_idx].tolist())
        row.extend(trial_max[node_idx].tolist())
        onset_trial = _safe_float(summary_row.get("onset_progress_trial"), 0.0)
        peak_trial = _safe_float(summary_row.get("peak_progress_trial"), 0.0)
        peak_mag_trial = _safe_float(summary_row.get("peak_magnitude_trial"), 0.0)
        smooth_trial = _safe_float(summary_row.get("smoothness_trial"), 0.0)
        onset_ref = _safe_float(summary_row.get("onset_progress_reference"), 0.0)
        peak_ref = _safe_float(summary_row.get("peak_progress_reference"), 0.0)
        peak_mag_ref = _safe_float(summary_row.get("peak_magnitude_reference"), 0.0)
        smooth_ref = _safe_float(summary_row.get("smoothness_reference"), 0.0)
        row.extend(
            [
                onset_trial,
                peak_trial,
                peak_mag_trial,
                smooth_trial,
                onset_ref,
                peak_ref,
                peak_mag_ref,
                smooth_ref,
                onset_trial - onset_ref,
                peak_trial - peak_ref,
                peak_mag_trial - peak_mag_ref,
                smooth_trial - smooth_ref,
                float(edge_abs_incident[node_idx]),
                float(edge_signed_incident[node_idx]),
            ]
        )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32), feature_names, summary_df


def build_graph_node_sample(
    trial: dict[str, Any],
    reference: dict[str, Any],
    *,
    onset_ratio: float,
    time_region_width: int,
) -> GraphNodeSample:
    comparison = compare_trial_to_reference(
        trial,
        reference,
        onset_ratio=float(onset_ratio),
        time_region_width=int(time_region_width),
    )
    x, feature_names, summary_df = build_node_feature_matrix(trial, comparison)
    labels = [
        PRIMARY_NODE_CLASSES.index(primary_node_label(dict(row)))
        for row in summary_df.to_dict(orient="records")
    ]
    return GraphNodeSample(
        trial_id=str(trial["metadata"]["trial_id"]),
        cohort=str(trial["metadata"].get("cohort") or ""),
        exercise_name=str(trial["metadata"].get("exercise_name") or ""),
        node_names=list(trial["node_names"]),
        feature_names=list(feature_names),
        x=x,
        adjacency=build_adjacency_matrix(trial),
        y=np.asarray(labels, dtype=np.int64),
    )


def build_samples_from_dataset(
    dataset_dir: Path,
    reference_dir: Path,
    *,
    cohorts: set[str] | None,
    onset_ratio: float,
    time_region_width: int,
) -> tuple[list[GraphNodeSample], pd.DataFrame]:
    dataset_dir = Path(dataset_dir).resolve()
    manifest_df = pd.read_csv(dataset_dir / "dataset_manifest.csv")
    if cohorts:
        manifest_df = manifest_df.loc[manifest_df["cohort"].astype(str).str.lower().isin({str(item).lower() for item in cohorts})].copy()
    if manifest_df.empty:
        raise ValueError("No trials available after cohort filtering.")
    reference = load_reference_sequence(Path(reference_dir))
    samples: list[GraphNodeSample] = []
    manifest_rows: list[dict[str, Any]] = []
    for row in manifest_df.to_dict(orient="records"):
        trial = load_trial_graph_sequence(Path(row["json_path"]))
        sample = build_graph_node_sample(
            trial,
            reference,
            onset_ratio=float(onset_ratio),
            time_region_width=int(time_region_width),
        )
        samples.append(sample)
        labels = [PRIMARY_NODE_CLASSES[int(label)] for label in sample.y.tolist()]
        counts = pd.Series(labels).value_counts().to_dict()
        manifest_rows.append(
            {
                "trial_id": str(sample.trial_id),
                "cohort": str(sample.cohort),
                "exercise_name": str(sample.exercise_name),
                "num_nodes": int(len(sample.node_names)),
                "dominant_label": str(pd.Series(labels).value_counts().index[0]),
                **{f"class_count_{name}": int(counts.get(name, 0)) for name in PRIMARY_NODE_CLASSES},
            }
        )
    return samples, pd.DataFrame(manifest_rows)


def split_trial_samples(samples: list[GraphNodeSample], *, seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[GraphNodeSample]]:
    if not samples:
        return {"train": [], "val": [], "test": []}
    rng = random.Random(int(seed))
    ordered = list(samples)
    rng.shuffle(ordered)
    n_total = len(ordered)
    n_train = max(1, int(round(n_total * float(train_ratio)))) if n_total >= 3 else max(1, n_total - 1)
    n_val = max(1, int(round(n_total * float(val_ratio)))) if n_total >= 4 else (1 if n_total >= 3 else 0)
    if n_train + n_val >= n_total:
        n_val = max(0, n_total - n_train - 1)
    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    if not test and val:
        test = [val.pop()]
    elif not test and train:
        test = [train.pop()]
    if not val and len(train) > 1:
        val = [train.pop()]
    return {"train": train, "val": val, "test": test}


def fit_standardizer(samples: list[GraphNodeSample]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("Cannot fit feature standardizer on empty samples.")
    stacked = np.concatenate([sample.x for sample in samples], axis=0)
    mean = np.mean(stacked, axis=0).astype(np.float32)
    std = np.std(stacked, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def standardize_samples(samples: list[GraphNodeSample], mean: np.ndarray, std: np.ndarray) -> list[GraphNodeSample]:
    out: list[GraphNodeSample] = []
    for sample in samples:
        x_std = (np.asarray(sample.x, dtype=np.float32) - mean) / std
        out.append(
            GraphNodeSample(
                trial_id=sample.trial_id,
                cohort=sample.cohort,
                exercise_name=sample.exercise_name,
                node_names=list(sample.node_names),
                feature_names=list(sample.feature_names),
                x=x_std.astype(np.float32),
                adjacency=np.asarray(sample.adjacency, dtype=np.float32),
                y=np.asarray(sample.y, dtype=np.int64),
            )
        )
    return out


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    a = adjacency.float()
    degree = torch.sum(a, dim=1)
    inv_sqrt = torch.pow(torch.clamp(degree, min=1e-6), -0.5)
    d = torch.diag(inv_sqrt)
    return d @ a @ d


class DenseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ self.linear(x)


class GCNNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.conv1 = DenseGraphConv(in_dim, hidden_dim)
        self.conv2 = DenseGraphConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        a_norm = normalized_adjacency(adjacency)
        h = torch.relu(self.conv1(x, a_norm))
        h = self.dropout(h)
        h = torch.relu(self.conv2(h, a_norm))
        h = self.dropout(h)
        return self.head(h)


def compute_class_weights(samples: list[GraphNodeSample], num_classes: int) -> torch.Tensor:
    labels = np.concatenate([sample.y for sample in samples], axis=0)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float32)
    counts = np.where(counts > 0.0, counts, 1.0)
    weights = counts.sum() / (counts * float(num_classes))
    return torch.tensor(weights, dtype=torch.float32)


def _sample_to_tensors(sample: GraphNodeSample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(sample.x, dtype=torch.float32, device=device)
    adjacency = torch.tensor(sample.adjacency, dtype=torch.float32, device=device)
    y = torch.tensor(sample.y, dtype=torch.long, device=device)
    return x, adjacency, y


def run_epoch(
    model: nn.Module,
    samples: list[GraphNodeSample],
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
        x, adjacency, y = _sample_to_tensors(sample, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x, adjacency)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(y.shape[0])
        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred == y).sum().item())
        total_nodes += int(y.shape[0])
    return {
        "loss": float(total_loss / max(1, total_nodes)),
        "accuracy": float(total_correct / max(1, total_nodes)),
        "num_nodes": int(total_nodes),
    }


def predict_samples(
    model: nn.Module,
    samples: list[GraphNodeSample],
    *,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            x, adjacency, y = _sample_to_tensors(sample, device)
            logits = model(x, adjacency)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            for node_idx, node_name in enumerate(sample.node_names):
                rows.append(
                    {
                        "trial_id": str(sample.trial_id),
                        "cohort": str(sample.cohort),
                        "exercise_name": str(sample.exercise_name),
                        "node_index": int(node_idx),
                        "node_name": str(node_name),
                        "target_class": str(PRIMARY_NODE_CLASSES[int(y[node_idx].item())]),
                        "predicted_class": str(PRIMARY_NODE_CLASSES[int(pred[node_idx].item())]),
                        "predicted_confidence": float(probs[node_idx, int(pred[node_idx].item())].cpu()),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_prediction_table(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "labels": list(PRIMARY_NODE_CLASSES),
            "confusion_matrix": [],
            "classification_report": {},
        }
    y_true = [PRIMARY_NODE_CLASSES.index(str(item)) for item in df["target_class"].tolist()]
    y_pred = [PRIMARY_NODE_CLASSES.index(str(item)) for item in df["predicted_class"].tolist()]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(PRIMARY_NODE_CLASSES))),
        target_names=list(PRIMARY_NODE_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    conf = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(PRIMARY_NODE_CLASSES))),
    )
    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "labels": list(PRIMARY_NODE_CLASSES),
        "confusion_matrix": conf.tolist(),
        "classification_report": report,
    }


def save_gcn_outputs(
    output_dir: Path,
    *,
    model: nn.Module,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_names: list[str],
    sample_manifest_df: pd.DataFrame,
    split_tables: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
) -> dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gcn_node_classifier.pt"
    summary_json = output_dir / "gcn_node_classifier_summary.json"
    report_md = output_dir / "gcn_node_classifier_report.md"
    manifest_csv = output_dir / "gcn_sample_manifest.csv"
    train_csv = output_dir / "gcn_node_predictions_train.csv"
    val_csv = output_dir / "gcn_node_predictions_val.csv"
    test_csv = output_dir / "gcn_node_predictions_test.csv"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": np.asarray(feature_mean, dtype=np.float32),
            "feature_std": np.asarray(feature_std, dtype=np.float32),
            "feature_names": list(feature_names),
            "class_names": list(PRIMARY_NODE_CLASSES),
        },
        model_path,
    )
    sample_manifest_df.to_csv(manifest_csv, index=False)
    split_tables["train"].to_csv(train_csv, index=False)
    split_tables["val"].to_csv(val_csv, index=False)
    split_tables["test"].to_csv(test_csv, index=False)

    summary_payload = {
        "class_names": list(PRIMARY_NODE_CLASSES),
        "feature_names": list(feature_names),
        "num_features": int(len(feature_names)),
        "metrics": metrics,
        "model_path": str(model_path),
        "manifest_csv": str(manifest_csv),
        "train_predictions_csv": str(train_csv),
        "val_predictions_csv": str(val_csv),
        "test_predictions_csv": str(test_csv),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if sample_manifest_df.empty:
        manifest_preview = "_No samples_"
    else:
        preview_df = sample_manifest_df.head(20).copy()
        header = "| " + " | ".join(preview_df.columns.tolist()) + " |"
        divider = "| " + " | ".join(["---"] * len(preview_df.columns)) + " |"
        rows = [
            "| " + " | ".join(str(item) for item in row) + " |"
            for row in preview_df.itertuples(index=False, name=None)
        ]
        manifest_preview = "\n".join([header, divider, *rows])

    lines = [
        "# GCN Node Classifier",
        "",
        f"- Num features: `{len(feature_names)}`",
        f"- Classes: `{', '.join(PRIMARY_NODE_CLASSES)}`",
        f"- Train accuracy: `{metrics['train']['accuracy']:.4f}`",
        f"- Val accuracy: `{metrics['val']['accuracy']:.4f}`",
        f"- Test accuracy: `{metrics['test']['accuracy']:.4f}`",
        f"- Test macro-F1: `{metrics['test']['macro_f1']:.4f}`",
        "",
        "## Sample Manifest",
        "",
        manifest_preview,
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "model_path": str(model_path),
        "summary_json_path": str(summary_json),
        "report_path": str(report_md),
        "manifest_csv_path": str(manifest_csv),
        "train_predictions_csv_path": str(train_csv),
        "val_predictions_csv_path": str(val_csv),
        "test_predictions_csv_path": str(test_csv),
    }
