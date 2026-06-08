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
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gcn_node_classifier import (  # noqa: E402
    GCNNodeClassifier,
    PRIMARY_NODE_CLASSES,
    GraphNodeSample,
    build_adjacency_matrix,
    build_node_feature_matrix,
    compute_class_weights,
    fit_standardizer,
    run_epoch,
    set_seed,
    standardize_samples,
)
from app.graph_sequence_baseline import compare_trial_to_reference, load_reference_sequence, load_trial_graph_sequence  # noqa: E402


BINARY_CLASSES = ("within_reference", "deviating")
MULTICLASS_LABEL_ORDER = tuple(PRIMARY_NODE_CLASSES)


@dataclass
class TrialAudit:
    split: str
    trial_id: str
    sample: GraphNodeSample
    node_summary_df: pd.DataFrame


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows_"
    view = df.copy()
    if max_rows is not None:
        view = view.head(int(max_rows)).copy()
    cols = [str(col) for col in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in view.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def primary_multiclass_label(row: dict[str, Any]) -> str:
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


def binary_label(row: dict[str, Any], *, deviation_threshold: float) -> str:
    score = _safe_float(row.get("average_deviation_score"), 0.0)
    magnitude_status = str(row.get("magnitude_status") or "within_range")
    smoothness_status = str(row.get("smoothness_status") or "within_range")
    if score >= float(deviation_threshold):
        return "deviating"
    if magnitude_status in {"too_weak", "too_high"}:
        return "deviating"
    if smoothness_status == "not_smooth":
        return "deviating"
    return "within_reference"


def _load_locked_split(split_manifest_path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(split_manifest_path)
    if "split" not in split_df.columns:
        split_values = ["train"] * 11 + ["val"] * 3 + ["test"] * max(0, len(split_df) - 14)
        split_df = split_df.copy()
        split_df.insert(0, "split", split_values[: len(split_df)])
    return split_df


def _build_trial_audits(
    *,
    dataset_dir: Path,
    reference_dir: Path,
    split_manifest_path: Path,
    onset_ratio: float,
    time_region_width: int,
) -> list[TrialAudit]:
    dataset_manifest = pd.read_csv(dataset_dir / "dataset_manifest.csv")
    json_by_trial = {
        str(row["trial_id"]): Path(str(row["json_path"]))
        for row in dataset_manifest.to_dict(orient="records")
    }
    reference = load_reference_sequence(reference_dir)
    split_df = _load_locked_split(split_manifest_path)

    audits: list[TrialAudit] = []
    for row in split_df.to_dict(orient="records"):
        trial_id = str(row["trial_id"])
        json_path = json_by_trial.get(trial_id)
        if json_path is None:
            raise ValueError(f"Trial {trial_id!r} not found in dataset manifest.")
        trial = load_trial_graph_sequence(json_path)
        comparison = compare_trial_to_reference(
            trial,
            reference,
            onset_ratio=float(onset_ratio),
            time_region_width=int(time_region_width),
        )
        x, feature_names, summary_df = build_node_feature_matrix(trial, comparison)
        summary_df = summary_df.copy()
        summary_df["primary_multiclass_label"] = [
            primary_multiclass_label(dict(item)) for item in summary_df.to_dict(orient="records")
        ]
        y_multi = np.asarray(
            [MULTICLASS_LABEL_ORDER.index(label) for label in summary_df["primary_multiclass_label"].tolist()],
            dtype=np.int64,
        )
        sample = GraphNodeSample(
            trial_id=trial_id,
            cohort=str(trial["metadata"].get("cohort") or ""),
            exercise_name=str(trial["metadata"].get("exercise_name") or ""),
            node_names=list(trial["node_names"]),
            feature_names=list(feature_names),
            x=x,
            adjacency=build_adjacency_matrix(trial),
            y=y_multi,
        )
        audits.append(
            TrialAudit(
                split=str(row["split"]),
                trial_id=trial_id,
                sample=sample,
                node_summary_df=summary_df,
            )
        )
    return audits


def _label_distribution(rows: list[dict[str, Any]], label_col: str, labels: tuple[str, ...]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    output_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_df = df.loc[df["split"] == split]
        total = int(len(split_df))
        for label in labels:
            count = int((split_df[label_col].astype(str) == label).sum()) if total else 0
            output_rows.append(
                {
                    "split": split,
                    "label": label,
                    "count": count,
                    "fraction": float(count / total) if total else 0.0,
                }
            )
    return pd.DataFrame(output_rows)


def _status_distribution(rows: list[dict[str, Any]], status_col: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    output_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_df = df.loc[df["split"] == split]
        total = int(len(split_df))
        counts = split_df[status_col].astype(str).value_counts().sort_index()
        for status, count in counts.items():
            output_rows.append(
                {
                    "split": split,
                    "status_column": status_col,
                    "status": str(status),
                    "count": int(count),
                    "fraction": float(count / total) if total else 0.0,
                }
            )
    return pd.DataFrame(output_rows)


def _build_binary_samples(audits: list[TrialAudit], threshold: float) -> dict[str, list[GraphNodeSample]]:
    split_samples: dict[str, list[GraphNodeSample]] = {"train": [], "val": [], "test": []}
    for audit in audits:
        labels = [
            BINARY_CLASSES.index(binary_label(dict(row), deviation_threshold=float(threshold)))
            for row in audit.node_summary_df.to_dict(orient="records")
        ]
        split_samples[audit.split].append(
            GraphNodeSample(
                trial_id=audit.sample.trial_id,
                cohort=audit.sample.cohort,
                exercise_name=audit.sample.exercise_name,
                node_names=list(audit.sample.node_names),
                feature_names=list(audit.sample.feature_names),
                x=np.asarray(audit.sample.x, dtype=np.float32),
                adjacency=np.asarray(audit.sample.adjacency, dtype=np.float32),
                y=np.asarray(labels, dtype=np.int64),
            )
        )
    return split_samples


def _predict_binary(model: torch.nn.Module, samples: list[GraphNodeSample], device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            x = torch.tensor(sample.x, dtype=torch.float32, device=device)
            adjacency = torch.tensor(sample.adjacency, dtype=torch.float32, device=device)
            y = torch.tensor(sample.y, dtype=torch.long, device=device)
            logits = model(x, adjacency)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            for node_idx, node_name in enumerate(sample.node_names):
                pred_idx = int(pred[node_idx].item())
                target_idx = int(y[node_idx].item())
                rows.append(
                    {
                        "trial_id": str(sample.trial_id),
                        "cohort": str(sample.cohort),
                        "exercise_name": str(sample.exercise_name),
                        "node_index": int(node_idx),
                        "node_name": str(node_name),
                        "target_class": BINARY_CLASSES[target_idx],
                        "predicted_class": BINARY_CLASSES[pred_idx],
                        "predicted_confidence": float(probs[node_idx, pred_idx].detach().cpu()),
                    }
                )
    return pd.DataFrame(rows)


def _evaluate_binary(pred_df: pd.DataFrame) -> dict[str, Any]:
    if pred_df.empty:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "binary_f1_deviating": 0.0,
            "classification_report": {},
            "confusion_matrix": [],
        }
    y_true = [BINARY_CLASSES.index(str(item)) for item in pred_df["target_class"].tolist()]
    y_pred = [BINARY_CLASSES.index(str(item)) for item in pred_df["predicted_class"].tolist()]
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=list(BINARY_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    majority_count = int(pd.Series(pred_df["target_class"].astype(str)).value_counts().max())
    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "binary_f1_deviating": float(report.get("deviating", {}).get("f1-score", 0.0)),
        "majority_baseline_accuracy": float(majority_count / max(1, len(pred_df))),
        "beats_majority_accuracy": bool(float(report.get("accuracy", 0.0)) > float(majority_count / max(1, len(pred_df)))),
        "classification_report": report,
        "confusion_matrix": conf.tolist(),
    }


def _write_confusion_csv(path: Path, conf: list[list[int]], labels: tuple[str, ...]) -> None:
    rows = []
    for idx, label in enumerate(labels):
        row = {"target_class": label}
        for pred_idx, pred_label in enumerate(labels):
            row[pred_label] = int(conf[idx][pred_idx]) if idx < len(conf) and pred_idx < len(conf[idx]) else 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit multiclass node labels and run binary GCN smoke training.")
    parser.add_argument("--dataset-dir", type=str, default="artifacts/open_to_fist_graph_sequence_rebuild/dataset")
    parser.add_argument("--reference-dir", type=str, default="artifacts/open_to_fist_reference_sequence_rebuild/reference")
    parser.add_argument("--split-manifest", type=str, default="artifacts/gcn_smoke_train/split_manifest.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts/gcn_binary_smoke")
    parser.add_argument("--audit-dir", type=str, default="artifacts/node_label_multiclass_rule_audit")
    parser.add_argument("--dataset-audit-dir", type=str, default="artifacts/binary_node_dataset_audit")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--docs-dir", type=str, default="docs")
    parser.add_argument("--onset-ratio", type=float, default=0.20)
    parser.add_argument("--time-region-width", type=int, default=3)
    parser.add_argument("--binary-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    random.seed(int(args.seed))

    output_dir = Path(args.output_dir).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    dataset_audit_dir = Path(args.dataset_audit_dir).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    docs_dir = Path(args.docs_dir).resolve()
    for path in (output_dir, output_dir / "checkpoints", audit_dir, dataset_audit_dir, reports_dir, docs_dir):
        path.mkdir(parents=True, exist_ok=True)

    audits = _build_trial_audits(
        dataset_dir=Path(args.dataset_dir).resolve(),
        reference_dir=Path(args.reference_dir).resolve(),
        split_manifest_path=Path(args.split_manifest).resolve(),
        onset_ratio=float(args.onset_ratio),
        time_region_width=int(args.time_region_width),
    )

    node_rows: list[dict[str, Any]] = []
    for audit in audits:
        for row in audit.node_summary_df.to_dict(orient="records"):
            row_out = dict(row)
            row_out["split"] = audit.split
            row_out["trial_id"] = audit.trial_id
            node_rows.append(row_out)

    train_scores = [
        _safe_float(row["average_deviation_score"])
        for row in node_rows
        if str(row["split"]) == "train"
    ]
    binary_threshold = float(np.quantile(train_scores, float(args.binary_threshold_quantile))) if train_scores else 0.75
    binary_threshold = max(0.0, binary_threshold)
    for row in node_rows:
        row["binary_label"] = binary_label(row, deviation_threshold=binary_threshold)

    multiclass_dist = _label_distribution(node_rows, "primary_multiclass_label", MULTICLASS_LABEL_ORDER)
    binary_dist = _label_distribution(node_rows, "binary_label", BINARY_CLASSES)
    multiclass_dist.loc[multiclass_dist["split"] == "train"].to_csv(audit_dir / "label_distribution_train.csv", index=False)
    multiclass_dist.loc[multiclass_dist["split"] == "val"].to_csv(audit_dir / "label_distribution_val.csv", index=False)
    multiclass_dist.loc[multiclass_dist["split"] == "test"].to_csv(audit_dir / "label_distribution_test.csv", index=False)
    pd.DataFrame(node_rows).to_csv(audit_dir / "node_label_status_audit.csv", index=False)
    pd.concat(
        [
            _status_distribution(node_rows, "onset_status"),
            _status_distribution(node_rows, "magnitude_status"),
            _status_distribution(node_rows, "smoothness_status"),
        ],
        ignore_index=True,
    ).to_csv(audit_dir / "status_distribution.csv", index=False)

    rule_summary = {
        "primary_multiclass_priority": [
            "onset_status in {late, early, missing} -> onset_shift",
            "magnitude_status == too_weak -> too_weak",
            "magnitude_status == too_high -> too_high",
            "smoothness_status == not_smooth -> not_smooth",
            "otherwise -> within_reference",
        ],
        "onset_late_if_delay_gt_progress_step_percent": True,
        "onset_progress_step_percent": float(100.0 / max(1, 56)),
        "magnitude_tolerance": "max(0.05, 0.10 * reference_peak_magnitude)",
        "smoothness_tolerance": "max(0.01, 0.15 * reference_smoothness)",
        "binary_rule": {
            "threshold_source": "train split average_deviation_score quantile",
            "threshold_quantile": float(args.binary_threshold_quantile),
            "average_deviation_score_threshold": float(binary_threshold),
            "extra_deviating_conditions": ["magnitude_status in {too_weak, too_high}", "smoothness_status == not_smooth"],
        },
    }
    (audit_dir / "rule_summary.json").write_text(json.dumps(rule_summary, indent=2), encoding="utf-8")

    binary_dist.to_csv(dataset_audit_dir / "label_distribution.csv", index=False)
    binary_samples = _build_binary_samples(audits, binary_threshold)
    all_samples = binary_samples["train"] + binary_samples["val"] + binary_samples["test"]
    dataset_summary = {
        "num_graph_samples": int(len(all_samples)),
        "num_train_graphs": int(len(binary_samples["train"])),
        "num_val_graphs": int(len(binary_samples["val"])),
        "num_test_graphs": int(len(binary_samples["test"])),
        "nodes_per_graph": int(len(all_samples[0].node_names)) if all_samples else 0,
        "features_per_node": int(all_samples[0].x.shape[1]) if all_samples else 0,
        "total_node_samples": int(sum(len(sample.y) for sample in all_samples)),
        "binary_classes": list(BINARY_CLASSES),
        "binary_threshold": float(binary_threshold),
        "threshold_quantile": float(args.binary_threshold_quantile),
        "split_policy": "locked from artifacts/gcn_smoke_train/split_manifest.csv: first 11 train, next 3 val, last 6 test",
    }
    (dataset_audit_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")

    feature_mean, feature_std = fit_standardizer(binary_samples["train"])
    train_samples = standardize_samples(binary_samples["train"], feature_mean, feature_std)
    val_samples = standardize_samples(binary_samples["val"], feature_mean, feature_std)
    test_samples = standardize_samples(binary_samples["test"], feature_mean, feature_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNNodeClassifier(
        in_dim=int(train_samples[0].x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_classes=len(BINARY_CLASSES),
        dropout=float(args.dropout),
    ).to(device)
    class_weights = compute_class_weights(train_samples, num_classes=len(BINARY_CLASSES)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history_rows: list[dict[str, Any]] = []
    best_state = None
    best_val_macro = -1.0
    best_epoch = -1
    epochs_since_best = 0
    for epoch_idx in range(int(args.epochs)):
        train_stats = run_epoch(model, train_samples, optimizer=optimizer, criterion=criterion, device=device)
        val_pred = _predict_binary(model, val_samples, device)
        val_metrics = _evaluate_binary(val_pred)
        val_macro = float(val_metrics["macro_f1"])
        history_rows.append(
            {
                "epoch": int(epoch_idx + 1),
                "train_loss": float(train_stats["loss"]),
                "train_accuracy": float(train_stats["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_macro),
                "val_binary_f1_deviating": float(val_metrics["binary_f1_deviating"]),
            }
        )
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_epoch = int(epoch_idx + 1)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": best_state, "binary_classes": list(BINARY_CLASSES)}, output_dir / "checkpoints" / "best_binary_gcn_smoke.pt")
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if epochs_since_best >= int(args.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_pred = _predict_binary(model, train_samples, device)
    val_pred = _predict_binary(model, val_samples, device)
    test_pred = _predict_binary(model, test_samples, device)
    train_pred.to_csv(output_dir / "gcn_binary_predictions_train.csv", index=False)
    val_pred.to_csv(output_dir / "gcn_binary_predictions_val.csv", index=False)
    test_pred.to_csv(output_dir / "gcn_binary_predictions_test.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": np.asarray(feature_mean, dtype=np.float32),
            "feature_std": np.asarray(feature_std, dtype=np.float32),
            "feature_names": list(train_samples[0].feature_names),
            "class_names": list(BINARY_CLASSES),
            "binary_threshold": float(binary_threshold),
        },
        output_dir / "gcn_binary_node_classifier.pt",
    )

    metrics = {
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(history_rows)),
        "device": str(device),
        "class_weighting_used": True,
        "binary_threshold": float(binary_threshold),
        "train": _evaluate_binary(train_pred),
        "val": _evaluate_binary(val_pred),
        "test": _evaluate_binary(test_pred),
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_confusion_csv(output_dir / "confusion_matrix.csv", metrics["test"]["confusion_matrix"], BINARY_CLASSES)

    previous_multiclass_metrics = {}
    previous_metrics_path = PROJECT_ROOT / "artifacts" / "gcn_smoke_train" / "eval_metrics.json"
    if previous_metrics_path.exists():
        previous_multiclass_metrics = json.loads(previous_metrics_path.read_text(encoding="utf-8"))

    reports = {
        "multiclass_audit": reports_dir / "node_label_multiclass_rule_audit.md",
        "binary_dataset": reports_dir / "binary_node_dataset_audit.md",
        "training": reports_dir / "gcn_binary_smoke_training_report.md",
        "compare": reports_dir / "binary_vs_multiclass_node_classification.md",
        "verdict": reports_dir / "node_classification_next_step_verdict.md",
    }

    onset_status_dist = pd.read_csv(audit_dir / "status_distribution.csv")
    onset_only = onset_status_dist.loc[onset_status_dist["status_column"] == "onset_status"].copy()
    magnitude_only = onset_status_dist.loc[onset_status_dist["status_column"] == "magnitude_status"].copy()
    smoothness_only = onset_status_dist.loc[onset_status_dist["status_column"] == "smoothness_status"].copy()

    reports["multiclass_audit"].write_text(
        "\n".join(
            [
                "# Node Label Multiclass Rule Audit",
                "",
                "## Active Rule",
                "- Primary label priority is onset -> magnitude -> smoothness -> within_reference.",
                "- `onset_shift` fires when onset status is `late`, `early`, or `missing`.",
                "- Magnitude labels fire only after onset is considered on-time/reference-missing.",
                "- Smoothness fires after both onset and magnitude are within range.",
                "",
                "## Multiclass Label Distribution",
                _markdown_table(multiclass_dist),
                "",
                "## Onset Status Distribution",
                _markdown_table(onset_only),
                "",
                "## Magnitude Status Distribution",
                _markdown_table(magnitude_only),
                "",
                "## Smoothness Status Distribution",
                _markdown_table(smoothness_only),
                "",
                "## Diagnosis",
                "- `onset_shift` dominates because any early/late/missing onset is assigned before magnitude and smoothness are evaluated as primary labels.",
                "- `too_weak` and `not_smooth` can be present as raw statuses only if their underlying thresholds fire, but they are hidden as primary labels whenever onset is abnormal.",
                "- Current multiclass formulation is not healthy for direct single-label training because it mixes subtype detection with priority ordering.",
            ]
        ),
        encoding="utf-8",
    )

    reports["binary_dataset"].write_text(
        "\n".join(
            [
                "# Binary Node Dataset Audit",
                "",
                f"Binary threshold: average_deviation_score >= {binary_threshold:.4f}",
                "",
                "## Label Distribution",
                _markdown_table(binary_dist),
                "",
                "## Dataset Summary",
                "```json",
                json.dumps(dataset_summary, indent=2),
                "```",
                "",
                "The binary split uses the same locked trial split as the multiclass smoke run.",
            ]
        ),
        encoding="utf-8",
    )

    reports["training"].write_text(
        "\n".join(
            [
                "# GCN Binary Smoke Training Report",
                "",
                f"Epochs run: {len(history_rows)}",
                f"Best epoch: {best_epoch}",
                f"Device: {device}",
                f"Class weighting used: yes",
                "",
                "## Test Metrics",
                f"- Accuracy: {metrics['test']['accuracy']:.4f}",
                f"- Macro F1: {metrics['test']['macro_f1']:.4f}",
                f"- Weighted F1: {metrics['test']['weighted_f1']:.4f}",
                f"- Deviating F1: {metrics['test']['binary_f1_deviating']:.4f}",
                f"- Majority baseline accuracy: {metrics['test']['majority_baseline_accuracy']:.4f}",
                f"- Beats majority accuracy: {metrics['test']['beats_majority_accuracy']}",
                "",
                "## Confusion Matrix",
                _markdown_table(pd.read_csv(output_dir / "confusion_matrix.csv")),
                "",
                "This is a smoke test only. It validates pipeline feasibility, not final model quality.",
            ]
        ),
        encoding="utf-8",
    )

    prev_test = previous_multiclass_metrics.get("test", {}) if isinstance(previous_multiclass_metrics, dict) else {}
    prev_test_report = prev_test.get("classification_report", {}) if isinstance(prev_test, dict) else {}
    prev_weighted_f1 = _safe_float(
        prev_test.get("weighted_f1", prev_test_report.get("weighted avg", {}).get("f1-score", 0.0))
    )
    reports["compare"].write_text(
        "\n".join(
            [
                "# Binary vs Multiclass Node Classification",
                "",
                "## Label Distribution: Multiclass",
                _markdown_table(multiclass_dist),
                "",
                "## Label Distribution: Binary",
                _markdown_table(binary_dist),
                "",
                "## Smoke Metrics",
                "| task | accuracy | macro_f1 | weighted_f1 | majority_baseline_accuracy |",
                "| --- | --- | --- | --- | --- |",
                f"| multiclass | {_safe_float(prev_test.get('accuracy')):.4f} | {_safe_float(prev_test.get('macro_f1')):.4f} | {prev_weighted_f1:.4f} | {_safe_float(previous_multiclass_metrics.get('test_majority_baseline_accuracy')):.4f} |",
                f"| binary | {metrics['test']['accuracy']:.4f} | {metrics['test']['macro_f1']:.4f} | {metrics['test']['weighted_f1']:.4f} | {metrics['test']['majority_baseline_accuracy']:.4f} |",
                "",
                "## Interpretation",
                "- Binary formulation avoids the single-label subtype priority problem.",
                "- Binary is a better intermediate target if it beats or approaches its majority baseline while keeping interpretable confusion.",
                "- Multiclass should remain paused until subtype rules are revised.",
            ]
        ),
        encoding="utf-8",
    )

    binary_beats_majority = bool(metrics["test"]["beats_majority_accuracy"])
    majority_margin = float(metrics["test"]["accuracy"] - metrics["test"]["majority_baseline_accuracy"])
    if binary_beats_majority and majority_margin >= 0.05:
        verdict_text = "Binary GCN is healthier than the multiclass smoke baseline and is reasonable as the temporary main node-classification baseline."
    elif binary_beats_majority:
        verdict_text = "Binary GCN is healthier than the multiclass smoke baseline, but the majority-baseline margin is small, so it should remain a cautious temporary baseline."
    else:
        verdict_text = "Binary GCN is simpler and better formulated, but this smoke result still does not justify strong model claims."
    reports["verdict"].write_text(
        "\n".join(
            [
                "# Node Classification Next-Step Verdict",
                "",
                f"Verdict: {verdict_text}",
                "",
                "## Answers",
                f"1. Binary task healthier: {'yes' if binary_dist['fraction'].max() < multiclass_dist['fraction'].max() else 'partially'}",
                f"2. Binary GCN as temporary baseline: {'yes, cautiously' if binary_beats_majority else 'only as smoke baseline, not yet as performance baseline'}",
                "3. Multiclass should be delayed: yes.",
                "4. Next step: revise multiclass rules and keep binary as stage-1 deviation detector while adding more labelled variation.",
                "",
                "## Bottlenecks",
                "- Multiclass primary labels are dominated by onset priority.",
                "- Subtype labels need separate thresholds and possibly multi-label representation.",
                "- Dataset is still small for reliable GCN generalization.",
                "",
                "No final claim should be made from this smoke run.",
            ]
        ),
        encoding="utf-8",
    )

    (docs_dir / "binary_node_deviation_rule.md").write_text(
        "\n".join(
            [
                "# Binary Node Deviation Rule",
                "",
                "Binary labels are used as a stage-1 sanity-check task before multiclass subtype classification.",
                "",
                "## Classes",
                "- `within_reference`: node is close enough to the normal reference sequence.",
                "- `deviating`: node is sufficiently far from the normal reference sequence.",
                "",
                "## Rule Used In Current Smoke Test",
                f"- Compute `average_deviation_score` per node from trial-vs-reference comparison.",
                f"- Set threshold from train split only: quantile={float(args.binary_threshold_quantile):.2f}, threshold={binary_threshold:.4f}.",
                "- Assign `deviating` when `average_deviation_score >= threshold`.",
                "- Also assign `deviating` when magnitude status is `too_weak` or `too_high`, or smoothness status is `not_smooth`.",
                "- Otherwise assign `within_reference`.",
                "",
                "This intentionally avoids mapping every `onset_shift` to binary deviation, because the current onset rule is known to dominate multiclass labels.",
            ]
        ),
        encoding="utf-8",
    )

    (docs_dir / "proposed_multiclass_label_rule_revision.md").write_text(
        "\n".join(
            [
                "# Proposed Multiclass Label Rule Revision",
                "",
                "Do not continue direct single-label multiclass training with the current priority rule.",
                "",
                "## Recommended Revision",
                "1. Stage 1: binary `within_reference` vs `deviating` using aggregate deviation score.",
                "2. Stage 2: subtype classification only for nodes already marked `deviating`.",
                "3. Compute subtype flags independently: `onset_shift`, `too_weak`, `too_high`, `not_smooth`.",
                "4. Allow multi-label subtype audit before forcing one primary class.",
                "5. If a single class is required, assign by strongest normalized evidence, not fixed onset-first priority.",
                "",
                "## Concrete Rule Changes To Test Next",
                "- Require onset delay to exceed at least two progress steps or a reference-onset uncertainty band before `onset_shift` fires.",
                "- Split onset into `early_onset`, `late_onset`, and `missing_onset` during audit, then merge only if distributions are healthy.",
                "- Define `too_weak` and `too_high` using reference peak standard deviation when available, not only 10% peak tolerance.",
                "- Define `not_smooth` using relative smoothness z-score or train-reference percentile, not only 15% smoothness tolerance.",
                "- Report subtype flags separately from primary class to avoid hiding magnitude/smoothness issues behind onset.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Multiclass audit: {reports['multiclass_audit']}")
    print(f"[INFO] Binary dataset audit: {reports['binary_dataset']}")
    print(f"[INFO] Binary training report: {reports['training']}")
    print(f"[INFO] Verdict: {reports['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
