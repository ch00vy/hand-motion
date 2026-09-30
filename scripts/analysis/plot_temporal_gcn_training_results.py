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
    "within_reference": "Sesuai Pola Acuan",
    "deviating": "Pola Aktivitas Berbeda",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#263241",
            "axes.labelcolor": "#1f2937",
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


def _best_epoch(history: pd.DataFrame, metrics: dict[str, Any], mode: str) -> tuple[int, str]:
    if mode == "metrics_json" and metrics.get("best_epoch") is not None:
        return int(metrics["best_epoch"]), "eval_metrics.json best_epoch"
    column = "val_macro_f1" if mode == "val_macro_f1" else "val_accuracy"
    if column in history.columns:
        idx = int(pd.to_numeric(history[column], errors="coerce").fillna(-1.0).idxmax())
        return int(history.loc[idx, "epoch"]), f"max {column}"
    if metrics.get("best_epoch") is not None:
        return int(metrics["best_epoch"]), "eval_metrics.json best_epoch fallback"
    return int(history["epoch"].iloc[-1]), "last epoch fallback"


def _plot_lines(
    history: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    ylabel: str,
    series: list[tuple[str, str, str]],
    best_epoch: int,
    best_note: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    _set_style()
    epochs = pd.to_numeric(history["epoch"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=180)
    plotted = 0
    for column, label, color in series:
        if column not in history.columns:
            continue
        values = pd.to_numeric(history[column], errors="coerce").to_numpy(dtype=float)
        ax.plot(epochs, values, linewidth=2.8, color=color, label=label)
        ax.scatter(epochs, values, s=18, color=color, alpha=0.62)
        plotted += 1
    if plotted == 0:
        raise ValueError(f"No requested columns available for {title}")
    ax.axvline(best_epoch, color="#6b7280", linestyle="--", linewidth=1.5)
    ax.annotate(
        f"Best epoch {best_epoch}",
        xy=(best_epoch, 1.0 if ylim else ax.get_ylim()[1]),
        xytext=(8, -28),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.94},
    )
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.45)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#d1d5db")
    ax.text(0.01, -0.18, f"Best epoch rule: {best_note}", transform=ax.transAxes, fontsize=9.5, color="#4b5563")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_confusion(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    df = pd.read_csv(path)
    true_labels = [str(item) for item in df.iloc[:, 0].tolist()]
    pred_labels = [str(col) for col in df.columns[1:]]
    matrix = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    return matrix, true_labels, pred_labels


def _plot_confusion(path: Path, *, output_path: Path, title: str) -> None:
    matrix, true_labels, pred_labels = _load_confusion(path)
    row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    normalized = matrix / row_sums
    _set_style()
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
            count = int(matrix[i, j])
            pct = normalized[i, j] * 100.0
            color = "white" if normalized[i, j] >= 0.55 else "#111827"
            ax.text(j, i, f"{count}\n{pct:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
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


def _write_summary(
    *,
    output_path: Path,
    run_dir: Path,
    metrics: dict[str, Any],
    best_epoch: int,
    best_note: str,
    figures: list[Path],
) -> None:
    test = metrics.get("test", {})
    abnormal = metrics.get("abnormal", {})
    lines = [
        "# Dynamic Temporal GCN Training Figure Summary",
        "",
        f"Run directory: `{run_dir}`",
        f"Best epoch: `{best_epoch}` using `{best_note}`.",
        "",
        "## Test Metrics",
        f"- Accuracy: {float(test.get('accuracy', 0.0)):.4f}",
        f"- Macro F1: {float(test.get('macro_f1', 0.0)):.4f}",
        f"- Pola Aktivitas Berbeda F1: {float(test.get('binary_f1_deviating', 0.0)):.4f}",
        "",
        "## Pattern-Different Validation Metrics",
        f"- Accuracy: {float(abnormal.get('accuracy', 0.0)):.4f}",
        f"- Macro F1: {float(abnormal.get('macro_f1', 0.0)):.4f}",
        f"- Pola Aktivitas Berbeda F1: {float(abnormal.get('binary_f1_deviating', 0.0)):.4f}",
        "",
        "## Generated Figures",
    ]
    lines.extend(f"- `{path}`" for path in figures)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training figures for a Temporal GCN node classifier run.")
    parser.add_argument("--run-dir", default="artifacts/temporal_gcn_prototype_dynamic")
    parser.add_argument("--output-dir", default="artifacts/temporal_gcn_prototype_dynamic/figures")
    parser.add_argument("--title-prefix", default="Dynamic Temporal GCN")
    parser.add_argument("--best-epoch-mode", default="metrics_json", choices=("metrics_json", "val_macro_f1", "val_accuracy"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "eval_metrics.json"
    confusion_path = run_dir / "confusion_matrix_test.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    history = pd.read_csv(history_path)
    metrics = _read_json(metrics_path)
    best_epoch, best_note = _best_epoch(history, metrics, str(args.best_epoch_mode))

    figures = [
        output_dir / "dynamic_temporal_gcn_accuracy.png",
        output_dir / "dynamic_temporal_gcn_loss.png",
        output_dir / "dynamic_temporal_gcn_f1.png",
    ]
    _plot_lines(
        history,
        output_path=figures[0],
        title=f"{args.title_prefix} - Accuracy",
        ylabel="Accuracy",
        series=[
            ("train_accuracy", "Train accuracy", "#2563eb"),
            ("val_accuracy", "Validation accuracy", "#f97316"),
        ],
        best_epoch=best_epoch,
        best_note=best_note,
        ylim=(0.0, 1.0),
    )
    _plot_lines(
        history,
        output_path=figures[1],
        title=f"{args.title_prefix} - Loss",
        ylabel="Cross-entropy loss",
        series=[
            ("train_loss", "Train loss", "#2563eb"),
            ("val_loss", "Validation loss", "#f97316"),
        ],
        best_epoch=best_epoch,
        best_note=best_note,
    )
    _plot_lines(
        history,
        output_path=figures[2],
        title=f"{args.title_prefix} - F1 Score",
        ylabel="F1 score",
        series=[
            ("val_macro_f1", "Validation macro F1", "#16a34a"),
            ("val_binary_f1_deviating", "Validation different-pattern F1", "#dc2626"),
        ],
        best_epoch=best_epoch,
        best_note=best_note,
        ylim=(0.0, 1.0),
    )
    if confusion_path.exists():
        figures.append(output_dir / "dynamic_temporal_gcn_confusion_matrix.png")
        _plot_confusion(confusion_path, output_path=figures[-1], title=f"{args.title_prefix} - Confusion Matrix")
    _write_summary(
        output_path=output_dir / "dynamic_temporal_gcn_figure_summary.md",
        run_dir=run_dir,
        metrics=metrics,
        best_epoch=best_epoch,
        best_note=best_note,
        figures=figures,
    )
    print(f"[OK] Wrote {len(figures)} figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
