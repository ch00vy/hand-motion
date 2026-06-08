from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gcn_node_classifier import GCNNodeClassifier, standardize_samples  # noqa: E402
from scripts.offline_biomech.run_binary_node_deviation_smoke import (  # noqa: E402
    BINARY_CLASSES,
    _build_binary_samples,
    _build_trial_audits,
    _label_distribution,
    _markdown_table,
    binary_label,
)
from scripts.offline_biomech.train_final_binary_gcn import (  # noqa: E402
    _collect_node_rows,
    _evaluate_binary,
    _predict_binary,
    _safe_float,
    _write_confusion_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained binary GCN on a validation cohort such as abnormal open_to_fist records.")
    parser.add_argument("--dataset-dir", required=True, help="Graph sequence dataset containing abnormal trials.")
    parser.add_argument("--reference-dir", required=True, help="Normal reference sequence directory.")
    parser.add_argument("--model-path", required=True, help="Trained binary GCN checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Output directory for validation predictions/metrics.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--cohort", default="abnormal", help="Cohort to evaluate from dataset_manifest.csv.")
    parser.add_argument("--binary-threshold", type=float, default=None, help="Override binary deviation threshold. Defaults to checkpoint run_config.")
    parser.add_argument("--onset-ratio", type=float, default=0.20)
    parser.add_argument("--time-region-width", type=int, default=3)
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_report(
    *,
    report_path: Path,
    metrics: dict[str, Any],
    dataset_summary: dict[str, Any],
    label_dist: pd.DataFrame,
    predictions: pd.DataFrame,
    node_summary: pd.DataFrame,
) -> None:
    pred_counts = predictions["predicted_class"].astype(str).value_counts().rename_axis("predicted_class").reset_index(name="count")
    total_nodes = int(len(predictions))
    deviating_pred = int((predictions["predicted_class"].astype(str) == "deviating").sum()) if total_nodes else 0
    top_nodes = (
        node_summary.sort_values("average_deviation_score", ascending=False)
        .loc[:, [col for col in ("trial_id", "node_name", "average_deviation_score", "binary_label") if col in node_summary.columns]]
        .head(20)
        if not node_summary.empty
        else pd.DataFrame()
    )
    report_path.write_text(
        "\n".join(
            [
                "# Binary GCN Abnormal Validation Report",
                "",
                "This report evaluates the trained binary GCN on user-declared abnormal validation trials.",
                "These trials are not used to build the normal reference pool.",
                "",
                "## Dataset Summary",
                "```json",
                json.dumps(dataset_summary, indent=2),
                "```",
                "",
                "## Deviation-Rule Label Distribution",
                _markdown_table(label_dist),
                "",
                "## Model Prediction Distribution",
                _markdown_table(pred_counts),
                "",
                "## Metrics Against Deviation-Rule Labels",
                f"- Accuracy: {metrics['accuracy']:.4f}",
                f"- Macro F1: {metrics['macro_f1']:.4f}",
                f"- Weighted F1: {metrics['weighted_f1']:.4f}",
                f"- Deviating precision: {metrics['precision_deviating']:.4f}",
                f"- Deviating recall: {metrics['recall_deviating']:.4f}",
                f"- Deviating F1: {metrics['binary_f1_deviating']:.4f}",
                f"- Majority baseline accuracy: {metrics['majority_baseline_accuracy']:.4f}",
                "",
                "## Clinical Validation Reading",
                f"- Predicted deviating nodes: {deviating_pred}/{total_nodes}",
                f"- Predicted deviating fraction: {(deviating_pred / total_nodes) if total_nodes else 0.0:.4f}",
                "",
                "## Top Deviation Nodes",
                _markdown_table(top_nodes, max_rows=20),
                "",
                "Interpretation note: abnormal validation here is a research validation set. Metrics are measured against the normal-reference deviation rule, not a manual per-node clinical ground truth.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(dataset_dir / "dataset_manifest.csv")
    selected = manifest.loc[manifest["cohort"].astype(str) == str(args.cohort)].copy()
    if selected.empty:
        raise RuntimeError(f"No trials with cohort={args.cohort!r} in {dataset_dir / 'dataset_manifest.csv'}")
    selected = selected.sort_values("trial_id").reset_index(drop=True)
    split_manifest = output_dir / "validation_split_manifest.csv"
    selected["split"] = "test"
    selected.to_csv(split_manifest, index=False)

    checkpoint = _load_checkpoint(model_path)
    run_config = dict(checkpoint.get("run_config") or {})
    binary_threshold = args.binary_threshold
    if binary_threshold is None:
        binary_threshold = _safe_float(run_config.get("binary_threshold"), 0.0)

    audits = _build_trial_audits(
        dataset_dir=dataset_dir,
        reference_dir=reference_dir,
        split_manifest_path=split_manifest,
        onset_ratio=float(args.onset_ratio),
        time_region_width=int(args.time_region_width),
    )
    node_rows = _collect_node_rows(audits)
    for row in node_rows:
        row["binary_label"] = binary_label(row, deviation_threshold=float(binary_threshold))
    node_summary = pd.DataFrame(node_rows)
    node_summary.to_csv(output_dir / "validation_node_summary.csv", index=False)
    label_dist = _label_distribution(node_rows, "binary_label", BINARY_CLASSES)
    label_dist.to_csv(output_dir / "validation_label_distribution.csv", index=False)

    split_samples = _build_binary_samples(audits, float(binary_threshold))
    samples = split_samples["test"]
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    samples = standardize_samples(samples, feature_mean, feature_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = int(run_config.get("hidden_dim", 32))
    dropout = float(run_config.get("dropout", 0.0))
    model = GCNNodeClassifier(
        in_dim=int(samples[0].x.shape[1]),
        hidden_dim=hidden_dim,
        num_classes=len(BINARY_CLASSES),
        dropout=dropout,
    ).to(device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)

    pred_df = _predict_binary(model, samples, device)
    pred_df.to_csv(output_dir / "validation_predictions.csv", index=False)
    metrics = _evaluate_binary(pred_df)
    metrics_out = {
        "cohort": str(args.cohort),
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "reference_dir": str(reference_dir),
        "binary_threshold": float(binary_threshold),
        "num_trials": int(len(selected)),
        "num_node_samples": int(len(pred_df)),
        "metrics_against_deviation_rule": metrics,
    }
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    _write_confusion_csv(output_dir / "validation_confusion_matrix.csv", metrics["confusion_matrix"])

    dataset_summary = {
        "dataset_dir": str(dataset_dir),
        "reference_dir": str(reference_dir),
        "cohort": str(args.cohort),
        "num_trials": int(len(selected)),
        "num_nodes_per_trial": int(len(samples[0].node_names)) if samples else 0,
        "binary_threshold": float(binary_threshold),
        "trial_ids": selected["trial_id"].astype(str).tolist(),
    }
    (output_dir / "validation_dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    _write_report(
        report_path=reports_dir / "binary_gcn_abnormal_validation_report.md",
        metrics=metrics,
        dataset_summary=dataset_summary,
        label_dist=label_dist,
        predictions=pred_df,
        node_summary=node_summary,
    )

    print(f"[INFO] Validation cohort: {args.cohort}")
    print(f"[INFO] Trials: {len(selected)}")
    print(f"[INFO] Predictions: {output_dir / 'validation_predictions.csv'}")
    print(f"[INFO] Metrics: {output_dir / 'validation_metrics.json'}")
    print(f"[INFO] Report: {reports_dir / 'binary_gcn_abnormal_validation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
