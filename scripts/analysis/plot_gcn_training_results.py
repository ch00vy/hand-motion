from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_confusion_matrix(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    df = pd.read_csv(path)
    if df.empty:
        return np.zeros((0, 0), dtype=float), [], []
    first_col = str(df.columns[0])
    target_labels = [str(item) for item in df[first_col].tolist()]
    pred_labels = [str(col) for col in df.columns[1:]]
    matrix = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return matrix, target_labels, pred_labels


def _plot_line_curves(
    history_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    ylabel: str,
    columns: list[tuple[str, str]],
    warnings: list[str],
) -> bool:
    available = [(col, label) for col, label in columns if col in history_df.columns]
    missing = [col for col, _ in columns if col not in history_df.columns]
    for col in missing:
        warnings.append(f"Column `{col}` not found in history; omitted from {output_path.name}.")
    if not available:
        warnings.append(f"No usable columns found for {output_path.name}; plot not generated.")
        return False

    epoch_col = "epoch" if "epoch" in history_df.columns else None
    x_values = pd.to_numeric(history_df[epoch_col], errors="coerce").to_numpy(dtype=float) if epoch_col else np.arange(1, len(history_df) + 1)

    plt.figure(figsize=(10.5, 6.0), dpi=150)
    for col, label in available:
        y_values = pd.to_numeric(history_df[col], errors="coerce").to_numpy(dtype=float)
        plt.plot(x_values, y_values, marker="o", linewidth=2.0, markersize=3.8, label=label)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return True


def _plot_confusion_heatmap(
    matrix: np.ndarray,
    target_labels: list[str],
    pred_labels: list[str],
    *,
    output_path: Path,
    title: str,
    normalize: bool,
) -> bool:
    if matrix.size == 0:
        return False
    plot_matrix = matrix.astype(float)
    annotation_matrix = matrix.astype(int)
    if normalize:
        row_sum = np.maximum(plot_matrix.sum(axis=1, keepdims=True), 1.0)
        plot_matrix = plot_matrix / row_sum

    plt.figure(figsize=(8.2, 7.0), dpi=150)
    image = plt.imshow(plot_matrix, cmap="Blues", interpolation="nearest")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted label")
    plt.ylabel("Target label")
    plt.xticks(np.arange(len(pred_labels)), pred_labels, rotation=35, ha="right")
    plt.yticks(np.arange(len(target_labels)), target_labels)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Row-normalized value" if normalize else "Count")

    max_value = float(plot_matrix.max()) if plot_matrix.size else 0.0
    threshold = max_value * 0.55
    for row_idx in range(plot_matrix.shape[0]):
        for col_idx in range(plot_matrix.shape[1]):
            count = int(annotation_matrix[row_idx, col_idx])
            if normalize:
                text = f"{count}\n{plot_matrix[row_idx, col_idx]:.2f}"
            else:
                text = str(count)
            colour = "white" if plot_matrix[row_idx, col_idx] > threshold else "black"
            plt.text(col_idx, row_idx, text, ha="center", va="center", color=colour, fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return True


def _extract_metric(metrics: dict[str, Any], split: str, metric_name: str) -> float | None:
    split_payload = metrics.get(split, {})
    if not isinstance(split_payload, dict):
        return None
    if metric_name in split_payload:
        return _safe_float(split_payload[metric_name])
    report = split_payload.get("classification_report", {})
    if metric_name == "weighted_f1" and isinstance(report, dict):
        return _safe_float(report.get("weighted avg", {}).get("f1-score"), default=0.0)
    return None


def _write_summary(
    *,
    output_path: Path,
    task_name: str,
    title_prefix: str,
    metrics: dict[str, Any],
    generated_files: list[str],
    warnings: list[str],
) -> None:
    lines = [
        f"# {title_prefix} Metrics Summary",
        "",
        f"Task: `{task_name}`",
        "",
        "## Metrics",
        "| split | accuracy | macro_f1 | weighted_f1 | majority_baseline_accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split in ("train", "val", "test"):
        accuracy = _extract_metric(metrics, split, "accuracy")
        macro_f1 = _extract_metric(metrics, split, "macro_f1")
        weighted_f1 = _extract_metric(metrics, split, "weighted_f1")
        majority = _extract_metric(metrics, split, "majority_baseline_accuracy")
        lines.append(
            f"| {split} | "
            f"{accuracy:.4f}" if accuracy is not None else f"| {split} | n/a"
        )
        if accuracy is not None:
            lines[-1] = (
                f"| {split} | {accuracy:.4f} | "
                f"{macro_f1:.4f}" if macro_f1 is not None else f"| {split} | {accuracy:.4f} | n/a"
            )
            if macro_f1 is not None:
                lines[-1] += f" | {weighted_f1:.4f}" if weighted_f1 is not None else " | n/a"
                lines[-1] += f" | {majority:.4f}" if majority is not None else " | n/a"
                lines[-1] += " |"

    if "best_epoch" in metrics:
        lines.extend(["", f"Best epoch: `{metrics['best_epoch']}`"])
    if "epochs_run" in metrics:
        lines.append(f"Epochs run: `{metrics['epochs_run']}`")

    lines.extend(["", "## Generated Files"])
    lines.extend(f"- `{item}`" for item in generated_files)
    lines.extend(["", "## Warnings"])
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- None")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reusable GCN training curves and confusion matrix.")
    parser.add_argument("--history-csv", type=str, required=True, help="Path to training history CSV.")
    parser.add_argument("--confusion-csv", type=str, required=True, help="Path to confusion matrix CSV.")
    parser.add_argument("--metrics-json", type=str, required=True, help="Path to eval metrics JSON.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for PNG/summary outputs.")
    parser.add_argument("--title-prefix", type=str, default="GCN", help="Plot title prefix.")
    parser.add_argument("--task-name", type=str, default="gcn", help="Task name used in output filenames.")
    parser.add_argument("--normalize-confusion", action="store_true", help="Show row-normalized confusion heatmap colors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    history_path = Path(args.history_csv).resolve()
    confusion_path = Path(args.confusion_csv).resolve()
    metrics_path = Path(args.metrics_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    if not history_path.exists():
        raise FileNotFoundError(f"History CSV not found: {history_path}")
    if not confusion_path.exists():
        raise FileNotFoundError(f"Confusion CSV not found: {confusion_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {metrics_path}")

    history_df = pd.read_csv(history_path)
    metrics = _read_metrics(metrics_path)
    matrix, target_labels, pred_labels = _normalise_confusion_matrix(confusion_path)

    task_name = str(args.task_name)
    title_prefix = str(args.title_prefix)
    generated: list[str] = []

    accuracy_path = output_dir / f"{task_name}_accuracy_curve.png"
    if _plot_line_curves(
        history_df,
        output_path=accuracy_path,
        title=f"{title_prefix}: Accuracy Curve",
        ylabel="Accuracy / F1",
        columns=[
            ("train_accuracy", "Train accuracy"),
            ("val_accuracy", "Validation accuracy"),
            ("val_macro_f1", "Validation macro F1"),
            ("val_binary_f1_deviating", "Validation deviating F1"),
        ],
        warnings=warnings,
    ):
        generated.append(str(accuracy_path))

    loss_path = output_dir / f"{task_name}_loss_curve.png"
    if _plot_line_curves(
        history_df,
        output_path=loss_path,
        title=f"{title_prefix}: Loss Curve",
        ylabel="Loss",
        columns=[
            ("train_loss", "Train loss"),
            ("val_loss", "Validation loss"),
        ],
        warnings=warnings,
    ):
        generated.append(str(loss_path))

    confusion_png = output_dir / f"{task_name}_confusion_matrix.png"
    if _plot_confusion_heatmap(
        matrix,
        target_labels,
        pred_labels,
        output_path=confusion_png,
        title=f"{title_prefix}: Confusion Matrix",
        normalize=bool(args.normalize_confusion),
    ):
        generated.append(str(confusion_png))
    else:
        warnings.append("Confusion matrix was empty; heatmap not generated.")

    summary_path = output_dir / f"{task_name}_metrics_summary.md"
    _write_summary(
        output_path=summary_path,
        task_name=task_name,
        title_prefix=title_prefix,
        metrics=metrics,
        generated_files=generated,
        warnings=warnings,
    )
    generated.append(str(summary_path))

    print("[INFO] Generated visualization outputs:")
    for item in generated:
        print(f"  {item}")
    if warnings:
        print("[WARN] Plot warnings:")
        for item in warnings:
            print(f"  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
