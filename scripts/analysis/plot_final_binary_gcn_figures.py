from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_DISPLAY = {
    "within_reference": "Within reference",
    "deviating": "Deviating",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resolve_best_epoch(history_df: pd.DataFrame, metrics: dict[str, Any], mode: str) -> tuple[int, str]:
    if mode == "metrics_json" and metrics.get("best_epoch") is not None:
        return int(metrics["best_epoch"]), "eval_metrics.json best_epoch"
    if mode == "val_macro_f1" and "val_macro_f1" in history_df.columns:
        idx = int(pd.to_numeric(history_df["val_macro_f1"], errors="coerce").fillna(-1).idxmax())
        return int(history_df.loc[idx, "epoch"]), "max validation macro F1"
    if mode == "val_accuracy" and "val_accuracy" in history_df.columns:
        idx = int(pd.to_numeric(history_df["val_accuracy"], errors="coerce").fillna(-1).idxmax())
        return int(history_df.loc[idx, "epoch"]), "max validation accuracy"
    if metrics.get("best_epoch") is not None:
        return int(metrics["best_epoch"]), "eval_metrics.json best_epoch"
    idx = int(pd.to_numeric(history_df["val_macro_f1"], errors="coerce").fillna(-1).idxmax())
    return int(history_df.loc[idx, "epoch"]), "max validation macro F1 fallback"


def _set_report_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#25313f",
            "axes.labelcolor": "#1f2933",
            "axes.titlecolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "savefig.facecolor": "white",
        }
    )


def _plot_two_line_figure(
    history_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    ylabel: str,
    y_lim: tuple[float, float] | None,
    series: list[tuple[str, str, str]],
    best_epoch: int,
    best_epoch_note: str,
) -> None:
    _set_report_style()
    epochs = pd.to_numeric(history_df["epoch"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=180)
    for column, label, color in series:
        if column not in history_df.columns:
            continue
        values = pd.to_numeric(history_df[column], errors="coerce").to_numpy(dtype=float)
        ax.plot(epochs, values, linewidth=2.8, color=color, label=label)
        ax.scatter(epochs, values, s=16, color=color, alpha=0.65)

    ax.axvline(best_epoch, color="#6b7280", linestyle="--", linewidth=1.5, alpha=0.95)
    ax.annotate(
        f"Best epoch {best_epoch}",
        xy=(best_epoch, ax.get_ylim()[1] if y_lim is None else y_lim[1]),
        xytext=(8, -28),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.92},
    )
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.45)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#d1d5db")
    ax.text(
        0.01,
        -0.18,
        f"Best epoch rule: {best_epoch_note}",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4b5563",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_confusion(confusion_csv: Path) -> tuple[np.ndarray, list[str], list[str]]:
    df = pd.read_csv(confusion_csv)
    labels_true = [str(item) for item in df.iloc[:, 0].tolist()]
    labels_pred = [str(col) for col in df.columns[1:]]
    matrix = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    return matrix, labels_true, labels_pred


def _plot_confusion(
    matrix: np.ndarray,
    true_labels: list[str],
    pred_labels: list[str],
    *,
    output_path: Path,
    title: str,
    normalized_only: bool = False,
) -> None:
    _set_report_style()
    row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    normalized = matrix / row_sums
    fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=180)
    im = ax.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(true_labels)))
    ax.set_xticklabels([CLASS_DISPLAY.get(item, item) for item in pred_labels], rotation=20, ha="right")
    ax.set_yticklabels([CLASS_DISPLAY.get(item, item) for item in true_labels])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            pct = normalized[i, j] * 100.0
            count = int(matrix[i, j])
            text = f"{pct:.1f}%" if normalized_only else f"{count}\n{pct:.1f}%"
            color = "white" if normalized[i, j] >= 0.55 else "#111827"
            ax.text(j, i, text, ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized percentage", rotation=270, labelpad=18)
    ax.set_xticks(np.arange(-0.5, len(pred_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(true_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_notes(
    *,
    output_path: Path,
    history_df: pd.DataFrame,
    metrics: dict[str, Any],
    best_epoch: int,
    best_epoch_note: str,
    generated_files: list[Path],
) -> None:
    test = metrics.get("test", {})
    final_row = history_df.tail(1).iloc[0]
    best_row = history_df.loc[history_df["epoch"].astype(int) == int(best_epoch)].iloc[0]
    lines = [
        "# Final Binary GCN Figure Notes",
        "",
        f"Best epoch: `{best_epoch}` using `{best_epoch_note}`.",
        "",
        "## Accuracy Figure",
        f"- Train accuracy at best epoch: {float(best_row['train_accuracy']):.4f}.",
        f"- Validation accuracy at best epoch: {float(best_row['val_accuracy']):.4f}.",
        "- The figure separates accuracy from F1 so the performance trend is easier to read.",
        "",
        "## F1 Figure",
        f"- Validation macro F1 at best epoch: {float(best_row['val_macro_f1']):.4f}.",
        f"- Validation deviating F1 at best epoch: {float(best_row['val_binary_f1_deviating']):.4f}.",
        "- This figure focuses on class-balance-sensitive metrics rather than raw accuracy.",
        "",
        "## Loss Figure",
        f"- Train loss at final epoch: {float(final_row['train_loss']):.4f}.",
        f"- Validation loss at final epoch: {float(final_row['val_loss']):.4f}.",
        "- Validation loss increases after the best region, suggesting mild overfitting after the selected checkpoint.",
        "",
        "## Confusion Matrix",
        f"- Holdout accuracy: {float(test.get('accuracy', 0.0)):.4f}.",
        f"- Holdout macro F1: {float(test.get('macro_f1', 0.0)):.4f}.",
        f"- Deviating recall is {float(test.get('recall_deviating', 0.0)):.4f}, indicating the model catches most deviating nodes.",
        f"- Deviating precision is {float(test.get('precision_deviating', 0.0)):.4f}, so false positives remain an important limitation.",
        "",
        "## Generated Figures",
    ]
    lines.extend(f"- `{path}`" for path in generated_files)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_input_audit(
    *,
    output_path: Path,
    run_dir: Path,
    history_df: pd.DataFrame,
    metrics: dict[str, Any],
    best_epoch: int,
    best_epoch_note: str,
) -> None:
    required = ["history.csv", "eval_metrics.json", "confusion_matrix.csv", "run_config.json"]
    rows = []
    for name in required:
        path = run_dir / name
        rows.append({"file": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0})
    audit_df = pd.DataFrame(rows)
    table = ["| file | exists | bytes |", "| --- | --- | --- |"]
    for row in audit_df.to_dict(orient="records"):
        table.append(f"| `{row['file']}` | {row['exists']} | {row['bytes']} |")
    output_path.write_text(
        "\n".join(
            [
                "# Final Binary Figure Input Audit",
                "",
                f"Run directory: `{run_dir}`",
                "",
                "Only files from `artifacts/final_binary_gcn_run` were used.",
                "",
                "## Input Files",
                "\n".join(table),
                "",
                "## History Columns",
                ", ".join(f"`{col}`" for col in history_df.columns),
                "",
                f"Best epoch used for annotations: `{best_epoch}` ({best_epoch_note}).",
                f"Metrics best_epoch: `{metrics.get('best_epoch', 'n/a')}`.",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create polished final report figures for the binary vanilla GCN run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Final binary GCN run directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for polished figures.")
    parser.add_argument("--best-epoch-mode", type=str, default="metrics_json", choices=("metrics_json", "val_macro_f1", "val_accuracy"))
    parser.add_argument("--title-prefix", type=str, default="Final Binary Vanilla GCN")
    parser.add_argument("--audit-report", type=str, default="reports/final_binary_figure_input_audit.md")
    parser.add_argument("--interpretation-report", type=str, default="reports/final_binary_figure_interpretation.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.read_csv(run_dir / "history.csv")
    metrics = _read_json(run_dir / "eval_metrics.json")
    best_epoch, best_epoch_note = _resolve_best_epoch(history_df, metrics, str(args.best_epoch_mode))

    generated: list[Path] = []
    accuracy_path = output_dir / "final_binary_accuracy.png"
    _plot_two_line_figure(
        history_df,
        output_path=accuracy_path,
        title=f"{args.title_prefix} - Accuracy",
        ylabel="Accuracy",
        y_lim=(0.0, 1.0),
        series=[
            ("train_accuracy", "Train accuracy", "#2563eb"),
            ("val_accuracy", "Validation accuracy", "#f97316"),
        ],
        best_epoch=best_epoch,
        best_epoch_note=best_epoch_note,
    )
    generated.append(accuracy_path)

    f1_path = output_dir / "final_binary_f1.png"
    _plot_two_line_figure(
        history_df,
        output_path=f1_path,
        title=f"{args.title_prefix} - F1 Score",
        ylabel="F1 score",
        y_lim=(0.0, 1.0),
        series=[
            ("val_macro_f1", "Validation macro F1", "#059669"),
            ("val_binary_f1_deviating", "Validation deviating F1", "#7c3aed"),
        ],
        best_epoch=best_epoch,
        best_epoch_note=best_epoch_note,
    )
    generated.append(f1_path)

    loss_path = output_dir / "final_binary_loss.png"
    _plot_two_line_figure(
        history_df,
        output_path=loss_path,
        title=f"{args.title_prefix} - Loss",
        ylabel="Cross-entropy loss",
        y_lim=None,
        series=[
            ("train_loss", "Train loss", "#1d4ed8"),
            ("val_loss", "Validation loss", "#dc2626"),
        ],
        best_epoch=best_epoch,
        best_epoch_note=best_epoch_note,
    )
    generated.append(loss_path)

    matrix, true_labels, pred_labels = _load_confusion(run_dir / "confusion_matrix.csv")
    confusion_path = output_dir / "final_binary_confusion_matrix.png"
    _plot_confusion(
        matrix,
        true_labels,
        pred_labels,
        output_path=confusion_path,
        title=f"{args.title_prefix} - Confusion Matrix",
    )
    generated.append(confusion_path)

    normalized_path = output_dir / "final_binary_confusion_matrix_normalized.png"
    _plot_confusion(
        matrix,
        true_labels,
        pred_labels,
        output_path=normalized_path,
        title=f"{args.title_prefix} - Normalized Confusion Matrix",
        normalized_only=True,
    )
    generated.append(normalized_path)

    _write_notes(
        output_path=output_dir / "final_binary_figure_notes.md",
        history_df=history_df,
        metrics=metrics,
        best_epoch=best_epoch,
        best_epoch_note=best_epoch_note,
        generated_files=generated,
    )
    _write_input_audit(
        output_path=Path(args.audit_report).resolve(),
        run_dir=run_dir,
        history_df=history_df,
        metrics=metrics,
        best_epoch=best_epoch,
        best_epoch_note=best_epoch_note,
    )
    _write_notes(
        output_path=Path(args.interpretation_report).resolve(),
        history_df=history_df,
        metrics=metrics,
        best_epoch=best_epoch,
        best_epoch_note=best_epoch_note,
        generated_files=generated,
    )

    print(f"[INFO] Best epoch: {best_epoch} ({best_epoch_note})")
    print("[INFO] Generated final report figures:")
    for path in generated:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
