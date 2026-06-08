from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gcn_node_classifier import (  # noqa: E402
    GCNNodeClassifier,
    compute_class_weights,
    fit_standardizer,
    run_epoch,
    set_seed,
    standardize_samples,
)
from scripts.offline_biomech.run_binary_node_deviation_smoke import (  # noqa: E402
    BINARY_CLASSES,
    _build_binary_samples,
    _build_trial_audits,
    _label_distribution,
    _markdown_table,
    binary_label,
)


DEFAULT_DATASET_DIR = PROJECT_ROOT / "artifacts" / "open_to_fist_graph_sequence_rebuild" / "dataset"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "artifacts" / "open_to_fist_reference_sequence_rebuild" / "reference"
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / "artifacts" / "open_to_fist_reference_sequence_rebuild"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "final_binary_gcn_run"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _reset_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)


def _build_final_split_manifest(
    *,
    reference_root_dir: Path,
    output_path: Path,
    train_reference_trials: int,
) -> pd.DataFrame:
    reference_pool = pd.read_csv(reference_root_dir / "reference_pool_manifest.csv")
    holdout = pd.read_csv(reference_root_dir / "holdout_manifest.csv")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(reference_pool.to_dict(orient="records")):
        split = "train" if idx < int(train_reference_trials) else "val"
        row_out = dict(row)
        row_out["split"] = split
        row_out["is_reference_pool"] = True
        row_out["is_holdout"] = False
        rows.append(row_out)
    for row in holdout.to_dict(orient="records"):
        row_out = dict(row)
        row_out["split"] = "test"
        row_out["is_reference_pool"] = False
        row_out["is_holdout"] = True
        rows.append(row_out)
    split_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    return split_df


def _collect_node_rows(audits) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit in audits:
        for item in audit.node_summary_df.to_dict(orient="records"):
            row = dict(item)
            row["split"] = audit.split
            row["trial_id"] = audit.trial_id
            rows.append(row)
    return rows


def _write_split_summary(report_path: Path, split_df: pd.DataFrame, label_dist: pd.DataFrame) -> None:
    split_counts = split_df.groupby("split", as_index=False)["trial_id"].count().rename(columns={"trial_id": "num_trials"})
    report_path.write_text(
        "\n".join(
            [
                "# Final Binary GCN Split Summary",
                "",
                "Source: formal `open_to_fist` transition graph-sequence dataset.",
                "",
                "## Trial Split",
                _markdown_table(split_counts),
                "",
                "## Split Manifest Preview",
                _markdown_table(split_df.loc[:, [col for col in ("split", "trial_id", "exercise_name", "source_csv") if col in split_df.columns]], max_rows=30),
                "",
                "## Binary Label Distribution",
                _markdown_table(label_dist),
            ]
        ),
        encoding="utf-8",
    )


def _write_dataset_audit(report_path: Path, dataset_summary: dict[str, Any], label_dist: pd.DataFrame) -> None:
    rows = []
    for split in ("train", "val", "test"):
        split_df = label_dist.loc[label_dist["split"] == split]
        total = int(split_df["count"].sum())
        max_count = int(split_df["count"].max()) if not split_df.empty else 0
        rows.append(
            {
                "split": split,
                "total_nodes": total,
                "majority_count": max_count,
                "majority_baseline": float(max_count / total) if total else 0.0,
            }
        )
    report_path.write_text(
        "\n".join(
            [
                "# Final Binary GCN Dataset Audit",
                "",
                "## Dataset Summary",
                "```json",
                json.dumps(dataset_summary, indent=2),
                "```",
                "",
                "## Label Distribution",
                _markdown_table(label_dist),
                "",
                "## Majority Baseline By Split",
                _markdown_table(pd.DataFrame(rows)),
                "",
                "## Notes",
                "- Binary classes: `within_reference`, `deviating`.",
                "- Labels are generated from trial-vs-reference deviation scores using a train-only threshold.",
                "- The run is a clean final baseline run, not a multiclass experiment.",
            ]
        ),
        encoding="utf-8",
    )


def _predict_binary(model: torch.nn.Module, samples, device: torch.device) -> pd.DataFrame:
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
                target_idx = int(y[node_idx].item())
                pred_idx = int(pred[node_idx].item())
                rows.append(
                    {
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


def _evaluate_binary(pred_df: pd.DataFrame) -> dict[str, Any]:
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
            "classification_report": {},
            "confusion_matrix": [],
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
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    majority = float(pd.Series(pred_df["target_class"].astype(str)).value_counts(normalize=True).max())
    try:
        roc_auc = float(roc_auc_score(y_true, prob_deviating)) if len(set(y_true.tolist())) == 2 else None
    except Exception:
        roc_auc = None
    try:
        average_precision = float(average_precision_score(y_true, prob_deviating)) if len(set(y_true.tolist())) == 2 else None
    except Exception:
        average_precision = None
    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "binary_f1_deviating": float(f1),
        "precision_deviating": float(precision),
        "recall_deviating": float(recall),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "majority_baseline_accuracy": majority,
        "beats_majority_accuracy": bool(float(report.get("accuracy", 0.0)) > majority),
        "classification_report": report,
        "confusion_matrix": conf.tolist(),
    }


def _write_confusion_csv(path: Path, conf: list[list[int]]) -> None:
    rows = []
    for idx, label in enumerate(BINARY_CLASSES):
        row = {"target_class": label}
        for pred_idx, pred_label in enumerate(BINARY_CLASSES):
            row[pred_label] = int(conf[idx][pred_idx]) if idx < len(conf) and pred_idx < len(conf[idx]) else 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_training_report(report_path: Path, history_df: pd.DataFrame, run_config: dict[str, Any], best_epoch: int) -> None:
    last = history_df.tail(1).to_dict(orient="records")[0] if not history_df.empty else {}
    report_path.write_text(
        "\n".join(
            [
                "# Final Binary Vanilla GCN Training Report",
                "",
                f"Best epoch: `{best_epoch}`",
                f"Epochs run: `{len(history_df)}`",
                "",
                "## Final Logged Epoch",
                _markdown_table(pd.DataFrame([last])),
                "",
                "## Run Config",
                "```json",
                json.dumps(run_config, indent=2),
                "```",
            ]
        ),
        encoding="utf-8",
    )


def _write_evaluation_report(report_path: Path, metrics: dict[str, Any], confusion_csv: Path) -> None:
    test = metrics["test"]
    report_path.write_text(
        "\n".join(
            [
                "# Final Binary Vanilla GCN Evaluation Report",
                "",
                "## Holdout/Test Metrics",
                f"- Accuracy: {test['accuracy']:.4f}",
                f"- Macro F1: {test['macro_f1']:.4f}",
                f"- Weighted F1: {test['weighted_f1']:.4f}",
                f"- Deviating precision: {test['precision_deviating']:.4f}",
                f"- Deviating recall: {test['recall_deviating']:.4f}",
                f"- Deviating F1: {test['binary_f1_deviating']:.4f}",
                f"- ROC AUC: {test['roc_auc'] if test['roc_auc'] is not None else 'n/a'}",
                f"- Average precision: {test['average_precision'] if test['average_precision'] is not None else 'n/a'}",
                f"- Majority baseline accuracy: {test['majority_baseline_accuracy']:.4f}",
                f"- Beats majority baseline: {test['beats_majority_accuracy']}",
                "",
                "## Confusion Matrix",
                _markdown_table(pd.read_csv(confusion_csv)),
            ]
        ),
        encoding="utf-8",
    )


def _write_smoke_comparison(report_path: Path, final_metrics: dict[str, Any], smoke_metrics_path: Path) -> None:
    smoke = json.loads(smoke_metrics_path.read_text(encoding="utf-8")) if smoke_metrics_path.exists() else {}
    rows = []
    for name, payload in (("previous_binary_smoke", smoke.get("test", {})), ("final_binary_run", final_metrics.get("test", {}))):
        rows.append(
            {
                "run": name,
                "accuracy": _safe_float(payload.get("accuracy"), 0.0),
                "macro_f1": _safe_float(payload.get("macro_f1"), 0.0),
                "weighted_f1": _safe_float(payload.get("weighted_f1"), 0.0),
                "majority_baseline": _safe_float(payload.get("majority_baseline_accuracy"), 0.0),
                "beats_majority": bool(payload.get("beats_majority_accuracy", False)),
            }
        )
    report_path.write_text(
        "\n".join(
            [
                "# Final Binary GCN vs Previous Smoke",
                "",
                _markdown_table(pd.DataFrame(rows)),
                "",
                "The final run is a fresh retrain from scratch using the same final binary formulation, clean output directory, and complete train/validation loss logging.",
            ]
        ),
        encoding="utf-8",
    )


def _write_verdict(report_path: Path, metrics: dict[str, Any]) -> None:
    test = metrics["test"]
    if bool(test["beats_majority_accuracy"]):
        verdict = "Final binary vanilla GCN retrain succeeded and is acceptable as a clean baseline for reporting, with cautious wording."
    else:
        verdict = "Final binary vanilla GCN retrain succeeded technically, but does not beat the majority baseline and should not be used as a strong reporting baseline."
    report_path.write_text(
        "\n".join(
            [
                "# Final Binary GCN Verdict",
                "",
                f"Verdict: {verdict}",
                "",
                "## Answers",
                "1. Retrain final binary vanilla GCN berhasil: yes.",
                f"2. Layak sebagai baseline utama laporan: {'yes, cautiously' if bool(test['beats_majority_accuracy']) else 'not yet'}",
                f"3. Lebih baik dari majority baseline: {bool(test['beats_majority_accuracy'])}.",
                "4. Next step: pertahankan binary baseline ini, tambah data/reference variation, dan revisi label rule sebelum kembali ke multiclass.",
                "",
                "## Test Metrics",
                f"- Accuracy: {test['accuracy']:.4f}",
                f"- Macro F1: {test['macro_f1']:.4f}",
                f"- Weighted F1: {test['weighted_f1']:.4f}",
                f"- Majority baseline: {test['majority_baseline_accuracy']:.4f}",
                "",
                "Do not present this as a final clinical/biomechanical classifier. It is a clean vanilla GCN baseline.",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain a clean final binary vanilla GCN baseline for open_to_fist node deviation.")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--reference-dir", type=str, default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--reference-root-dir", type=str, default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reports-dir", type=str, default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--train-reference-trials", type=int, default=11)
    parser.add_argument("--binary-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--onset-ratio", type=float, default=0.20)
    parser.add_argument("--time-region-width", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weighting", action="store_true", help="Use inverse-frequency class weighting.")
    parser.add_argument("--force-clean", action="store_true", help="Remove existing output directory before running.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    random.seed(int(args.seed))
    output_dir = Path(args.output_dir).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    _reset_output_dir(output_dir, force=bool(args.force_clean))
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = _build_final_split_manifest(
        reference_root_dir=Path(args.reference_root_dir).resolve(),
        output_path=output_dir / "split_manifest.csv",
        train_reference_trials=int(args.train_reference_trials),
    )
    audits = _build_trial_audits(
        dataset_dir=Path(args.dataset_dir).resolve(),
        reference_dir=Path(args.reference_dir).resolve(),
        split_manifest_path=output_dir / "split_manifest.csv",
        onset_ratio=float(args.onset_ratio),
        time_region_width=int(args.time_region_width),
    )
    node_rows = _collect_node_rows(audits)
    train_scores = [_safe_float(row["average_deviation_score"]) for row in node_rows if str(row["split"]) == "train"]
    binary_threshold = float(np.quantile(train_scores, float(args.binary_threshold_quantile))) if train_scores else 0.0
    for row in node_rows:
        row["binary_label"] = binary_label(row, deviation_threshold=binary_threshold)
    label_dist = _label_distribution(node_rows, "binary_label", BINARY_CLASSES)
    for split in ("train", "val", "test"):
        label_dist.loc[label_dist["split"] == split].to_csv(output_dir / f"label_distribution_{split}.csv", index=False)

    split_samples = _build_binary_samples(audits, binary_threshold)
    all_samples = split_samples["train"] + split_samples["val"] + split_samples["test"]
    dataset_summary = {
        "source_dataset_dir": str(Path(args.dataset_dir).resolve()),
        "reference_dir": str(Path(args.reference_dir).resolve()),
        "num_graph_samples": int(len(all_samples)),
        "num_train_graphs": int(len(split_samples["train"])),
        "num_val_graphs": int(len(split_samples["val"])),
        "num_test_graphs": int(len(split_samples["test"])),
        "nodes_per_graph": int(len(all_samples[0].node_names)) if all_samples else 0,
        "features_per_node": int(all_samples[0].x.shape[1]) if all_samples else 0,
        "total_node_samples": int(sum(len(sample.y) for sample in all_samples)),
        "binary_classes": list(BINARY_CLASSES),
        "binary_threshold": float(binary_threshold),
        "binary_threshold_quantile": float(args.binary_threshold_quantile),
        "split_policy": f"first {int(args.train_reference_trials)} reference-pool trials train; remaining reference-pool trials val; holdout manifest test",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    _write_split_summary(reports_dir / "final_binary_gcn_split_summary.md", split_manifest, label_dist)
    _write_dataset_audit(reports_dir / "final_binary_gcn_dataset_audit.md", dataset_summary, label_dist)

    feature_mean, feature_std = fit_standardizer(split_samples["train"])
    train_samples = standardize_samples(split_samples["train"], feature_mean, feature_std)
    val_samples = standardize_samples(split_samples["val"], feature_mean, feature_std)
    test_samples = standardize_samples(split_samples["test"], feature_mean, feature_std)

    run_config = {
        "model": "vanilla_dense_gcn",
        "classification_task": "binary_node_deviation",
        "classes": list(BINARY_CLASSES),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "class_weighting": bool(args.class_weighting),
        "binary_threshold": float(binary_threshold),
        "binary_threshold_quantile": float(args.binary_threshold_quantile),
        "onset_ratio": float(args.onset_ratio),
        "time_region_width": int(args.time_region_width),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNNodeClassifier(
        in_dim=int(train_samples[0].x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_classes=len(BINARY_CLASSES),
        dropout=float(args.dropout),
    ).to(device)
    if bool(args.class_weighting):
        class_weights = compute_class_weights(train_samples, num_classes=len(BINARY_CLASSES)).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        class_weights = None
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history_rows: list[dict[str, Any]] = []
    best_state = None
    best_epoch = -1
    best_val_macro = -1.0
    epochs_since_best = 0
    for epoch_idx in range(int(args.epochs)):
        train_stats = run_epoch(model, train_samples, optimizer=optimizer, criterion=criterion, device=device)
        val_stats = run_epoch(model, val_samples, optimizer=None, criterion=criterion, device=device)
        train_pred = _predict_binary(model, train_samples, device)
        val_pred = _predict_binary(model, val_samples, device)
        train_metrics = _evaluate_binary(train_pred)
        val_metrics = _evaluate_binary(val_pred)
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
                    "feature_mean": np.asarray(feature_mean, dtype=np.float32),
                    "feature_std": np.asarray(feature_std, dtype=np.float32),
                    "feature_names": list(train_samples[0].feature_names),
                    "class_names": list(BINARY_CLASSES),
                    "run_config": run_config,
                    "best_epoch": int(best_epoch),
                },
                output_dir / "checkpoints" / "best_final_binary_gcn.pt",
            )
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if epochs_since_best >= int(args.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(output_dir / "history.csv", index=False)

    train_pred = _predict_binary(model, train_samples, device)
    val_pred = _predict_binary(model, val_samples, device)
    test_pred = _predict_binary(model, test_samples, device)
    train_pred.to_csv(output_dir / "predictions_train.csv", index=False)
    val_pred.to_csv(output_dir / "predictions_val.csv", index=False)
    test_pred.to_csv(output_dir / "predictions_test.csv", index=False)

    metrics = {
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(history_df)),
        "device": str(device),
        "class_weighting_used": bool(args.class_weighting),
        "class_weights": None if class_weights is None else [float(item) for item in class_weights.detach().cpu().tolist()],
        "binary_threshold": float(binary_threshold),
        "train": _evaluate_binary(train_pred),
        "val": _evaluate_binary(val_pred),
        "test": _evaluate_binary(test_pred),
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_confusion_csv(output_dir / "confusion_matrix.csv", metrics["test"]["confusion_matrix"])
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": np.asarray(feature_mean, dtype=np.float32),
            "feature_std": np.asarray(feature_std, dtype=np.float32),
            "feature_names": list(train_samples[0].feature_names),
            "class_names": list(BINARY_CLASSES),
            "run_config": run_config,
            "best_epoch": int(best_epoch),
        },
        output_dir / "final_binary_gcn_model.pt",
    )

    _write_training_report(reports_dir / "final_binary_gcn_training_report.md", history_df, run_config, best_epoch)
    _write_evaluation_report(reports_dir / "final_binary_gcn_evaluation_report.md", metrics, output_dir / "confusion_matrix.csv")
    _write_smoke_comparison(
        reports_dir / "final_binary_gcn_vs_previous_smoke.md",
        metrics,
        PROJECT_ROOT / "artifacts" / "gcn_binary_smoke" / "eval_metrics.json",
    )
    _write_verdict(reports_dir / "final_binary_gcn_verdict.md", metrics)

    print(f"[INFO] Final binary GCN run: {output_dir}")
    print(f"[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"[INFO] Test macro F1: {metrics['test']['macro_f1']:.4f}")
    print(f"[INFO] Beats majority: {metrics['test']['beats_majority_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
