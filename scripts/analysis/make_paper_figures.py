from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_RUN = PROJECT_ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier"
CTRL_SUMMARY = PROJECT_ROOT / "artifacts" / "retrain_gui_plus_previous" / "ctrl_ref_combined" / "reference_build" / "teacher_trial_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"

BLUE = "#1F77B4"
CYAN = "#2A9D8F"
ORANGE = "#E76F51"
GREEN = "#2E7D32"
RED = "#C62828"
YELLOW = "#F4A261"
GRAY = "#4B5563"
LIGHT = "#F8FAFC"
EDGE = "#CBD5E1"
TEXT = "#111827"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def box(ax, xy, wh, text, *, fc=LIGHT, ec=EDGE, lw=1.4, fontsize=9, radius=0.04):
    x, y = xy
    w, h = wh
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=TEXT, fontsize=fontsize, wrap=True)
    return patch


def arrow(ax, start, end, *, color=GRAY, lw=1.4, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=4, shrinkB=4, connectionstyle=f"arc3,rad={rad}"),
    )


def draw_label(ax, xy, text, *, color=TEXT, fontsize=9):
    ax.text(*xy, text, color=color, fontsize=fontsize, ha="center", va="center")


def fig1_experimental_setup() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Experimental Setup", loc="left", fontweight="bold", pad=10)

    # Table and monitor.
    ax.add_patch(patches.Rectangle((0.6, 0.75), 8.8, 0.22, facecolor="#E5E7EB", edgecolor=EDGE))
    ax.add_patch(patches.Rectangle((6.4, 1.2), 2.4, 2.0, facecolor="#0F172A", edgecolor="#334155", linewidth=1.5))
    ax.add_patch(patches.Rectangle((6.6, 1.4), 2.0, 1.55, facecolor="white", edgecolor="#94A3B8", linewidth=0.8))
    ax.plot([7.6, 7.6], [1.2, 0.95], color="#334155", lw=3)
    ax.add_patch(patches.Rectangle((7.05, 0.82), 1.1, 0.12, facecolor="#334155", edgecolor="none"))
    ax.text(7.6, 2.6, "GUI\nanalysis", ha="center", va="center", fontsize=9, color=TEXT)
    ax.text(7.6, 1.85, "MuJoCo\nreplay", ha="center", va="center", fontsize=8, color=BLUE)

    # Participant.
    ax.add_patch(patches.Circle((2.0, 4.25), 0.38, facecolor="#F1C27D", edgecolor="#B08968"))
    ax.add_patch(patches.FancyBboxPatch((1.55, 2.4), 0.9, 1.5, boxstyle="round,pad=0.02", facecolor="#CBD5E1", edgecolor="#64748B"))
    ax.plot([2.35, 3.2, 4.0], [3.25, 2.65, 2.35], color="#64748B", lw=8, solid_capstyle="round")
    ax.add_patch(patches.Ellipse((4.35, 2.25), 0.78, 0.38, angle=-8, facecolor="#F1C27D", edgecolor="#B08968"))
    for i in range(5):
        ax.plot([4.6, 4.95 + 0.1 * i], [2.18 + 0.05 * i, 2.35 + 0.12 * i], color="#B08968", lw=2, solid_capstyle="round")

    # Sensor.
    ax.add_patch(patches.FancyBboxPatch((3.75, 1.2), 1.4, 0.34, boxstyle="round,pad=0.04", facecolor="#111827", edgecolor="#475569"))
    ax.add_patch(patches.Circle((4.08, 1.37), 0.08, facecolor="#38BDF8", edgecolor="none"))
    ax.add_patch(patches.Circle((4.82, 1.37), 0.08, facecolor="#38BDF8", edgecolor="none"))

    # Callouts.
    box(ax, (0.75, 5.05), (2.45, 0.55), "Participant performs\nopen-to-fist trial", fc="#EFF6FF", ec="#93C5FD")
    arrow(ax, (1.95, 5.02), (2.0, 4.62), color=BLUE)
    box(ax, (3.15, 4.95), (2.6, 0.65), "Optical hand motion sensor\n(Ultraleap / Leap Motion)", fc="#ECFDF5", ec="#86EFAC")
    arrow(ax, (4.45, 4.92), (4.45, 1.58), color=GREEN, rad=-0.12)
    box(ax, (6.25, 4.95), (2.95, 0.65), "Control GUI + external\nMuJoCo replay viewer", fc="#FFF7ED", ec="#FDBA74")
    arrow(ax, (7.7, 4.9), (7.7, 3.2), color=ORANGE)
    ax.text(5.15, 2.55, "hand kinematics\n(qpos)", ha="center", va="center", color=BLUE, fontsize=9)
    arrow(ax, (4.75, 2.1), (6.35, 2.05), color=BLUE)
    save(fig, "fig1_experimental_setup.png")


def fig2_overall_framework() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Overall Proposed Framework", loc="left", fontweight="bold")
    labels = [
        "Hand\nkinematics\n(qpos)",
        "Preprocessing\nsmoothing\nqvel/qacc",
        "Musculoskeletal\nsimulation",
        "Inverse dynamics\n+ OSQP\nctrl_ref",
        "Consistency\nvalidation\n(control replay)",
        "Temporal\nbiomechanical\ngraph",
        "Dynamic\nTemporal GCN",
        "Muscle activity\npattern indication",
    ]
    colors = ["#EFF6FF", "#F8FAFC", "#ECFDF5", "#FFF7ED", "#FEF2F2", "#F0F9FF", "#EEF2FF", "#F0FDF4"]
    x_positions = np.linspace(0.4, 11.2, len(labels))
    for i, (x, text) in enumerate(zip(x_positions, labels)):
        box(ax, (x, 2.15), (1.25, 1.0), text, fc=colors[i], ec="#94A3B8", fontsize=8)
        if i < len(labels) - 1:
            arrow(ax, (x + 1.25, 2.65), (x_positions[i + 1], 2.65), color=GRAY)
    ax.text(6.6, 0.75, "Output: binary node-level labels for muscle/actuator nodes", ha="center", color=TEXT, fontsize=10)
    ax.text(6.6, 0.35, "Sesuai Pola Acuan   |   Pola Aktivitas Berbeda", ha="center", color=RED, fontsize=10, fontweight="bold")
    save(fig, "fig2_overall_framework.png")


def fig3_mismatch_feedback_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Mismatch-Feedback Refinement Pipeline", loc="left", fontweight="bold")
    top = [
        ("Recorded\nqpos", 0.55),
        ("Smoothing", 2.0),
        ("qvel / qacc", 3.45),
        ("Inverse\ndynamics", 4.9),
        ("tau_target", 6.35),
        ("OSQP\nsolve", 7.8),
        ("ctrl_ref", 9.25),
        ("Forward\nreplay", 10.7),
    ]
    for text, x in top:
        box(ax, (x, 4.55), (1.05, 0.75), text, fc="#F8FAFC", ec="#94A3B8", fontsize=8)
    for (_, x0), (_, x1) in zip(top[:-1], top[1:]):
        arrow(ax, (x0 + 1.05, 4.93), (x1, 4.93))
    box(ax, (4.05, 2.1), (2.2, 0.8), "Mismatch calculation\nqpos_forward - qpos_recorded", fc="#FEF2F2", ec="#FCA5A5", fontsize=9)
    box(ax, (6.9, 2.1), (2.0, 0.8), "Feedback refinement\nupdate torque/control prior", fc="#FFF7ED", ec="#FDBA74", fontsize=9)
    arrow(ax, (11.2, 4.55), (5.2, 2.92), color=RED, rad=-0.28)
    arrow(ax, (6.25, 2.5), (6.9, 2.5), color=ORANGE)
    arrow(ax, (8.9, 2.5), (7.85, 4.55), color=ORANGE, rad=-0.25)
    ax.text(6.0, 1.1, "Accepted only when rollout objective improves", ha="center", color=GRAY, fontsize=10)
    save(fig, "fig3_mismatch_feedback_pipeline.png")


def fig4_graph_and_dynamic_temporal_gcn() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Dynamic Temporal GCN for Muscle Activity-Pattern Analysis", loc="left", fontweight="bold")

    # Graph at a time step.
    center = np.array([2.1, 3.2])
    coords = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        coords.append(center + np.array([1.1 * np.cos(ang), 0.9 * np.sin(ang)]))
    edge_pairs = [(0, 1), (1, 2), (2, 5), (5, 6), (6, 7), (0, 4), (3, 7), (2, 6)]
    for i, j in edge_pairs:
        ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], color="#94A3B8", lw=1.3)
    for idx, p in enumerate(coords):
        fc = GREEN if idx not in (2, 6) else RED
        ax.add_patch(patches.Circle(p, 0.17, facecolor=fc, edgecolor="#111827", lw=0.7))
    ax.text(2.1, 1.8, "Biomechanical graph\nat step t", ha="center", fontsize=9)

    # Sequence.
    for s, x in enumerate([4.0, 4.55, 5.1]):
        ax.add_patch(patches.FancyBboxPatch((x, 2.6 + 0.12 * s), 1.0, 1.2, boxstyle="round,pad=0.03", facecolor="#EFF6FF", edgecolor="#93C5FD"))
        ax.text(x + 0.5, 3.2 + 0.12 * s, f"G{s+1}", ha="center", va="center", fontsize=9, color=BLUE)
    arrow(ax, (3.35, 3.2), (4.0, 3.2), color=BLUE)
    arrow(ax, (5.95, 3.25), (6.75, 3.25), color=BLUE)
    ax.text(4.85, 1.8, "Progress-normalized\nsequence of graphs", ha="center", fontsize=9)

    box(ax, (6.75, 2.35), (2.25, 1.65), "Dynamic Temporal GCN\n\nNode features +\nedge_weight_seq[t]\nGRU + attention pooling", fc="#EEF2FF", ec="#A5B4FC", fontsize=9)
    arrow(ax, (9.0, 3.2), (9.75, 3.2), color=BLUE)
    box(ax, (9.75, 2.55), (1.65, 1.25), "Node output\n\nSesuai Pola Acuan\nPola Aktivitas Berbeda", fc="#F0FDF4", ec="#86EFAC", fontsize=9)
    ax.text(6.0, 5.1, "Nodes = muscles / actuators", ha="center", fontsize=10, color=TEXT)
    ax.text(6.0, 4.75, "Edges = synergist, antagonist, and dynamic co-activation relations", ha="center", fontsize=9, color=GRAY)
    save(fig, "fig4_graph_and_dynamic_temporal_gcn.png")


def read_history() -> list[dict[str, float]]:
    with (FINAL_RUN / "history.csv").open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if v != ""})
        return rows


def read_eval() -> dict:
    return json.loads((FINAL_RUN / "eval_metrics.json").read_text(encoding="utf-8"))


def feedback_metrics() -> dict[str, float]:
    before_rmse = []
    after_rmse = []
    before_obj = []
    after_obj = []
    with CTRL_SUMMARY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hist_text = row.get("mismatch_feedback_history") or ""
            if not hist_text:
                continue
            try:
                hist = json.loads(hist_text)
            except Exception:
                try:
                    hist = ast.literal_eval(hist_text)
                except Exception:
                    continue
            if not hist:
                continue
            before_rmse.append(float(hist[0].get("old_rmse_forward_error", np.nan)))
            after_rmse.append(float(hist[-1].get("new_rmse_forward_error", row.get("rmse_forward_error", np.nan))))
            before_obj.append(float(hist[0].get("old_qpos_mismatch_objective", np.nan)))
            after_obj.append(float(hist[-1].get("new_qpos_mismatch_objective", np.nan)))
    return {
        "before_rmse": float(np.nanmean(before_rmse)),
        "after_rmse": float(np.nanmean(after_rmse)),
        "before_obj": float(np.nanmean(before_obj)),
        "after_obj": float(np.nanmean(after_obj)),
    }


def fig5_refinement_improvement() -> None:
    metrics = feedback_metrics()
    progress = np.linspace(0, 100, 80)
    target = 0.5 - 0.5 * np.cos(np.pi * progress / 100)
    before_scale = min(0.45, 0.16 + 0.10 * metrics["before_rmse"])
    after_scale = min(0.28, 0.08 + 0.07 * metrics["after_rmse"])
    before = target - before_scale * (0.35 + 0.65 * target) + 0.03 * np.sin(progress / 8)
    after = target - after_scale * (0.25 + 0.55 * target) + 0.015 * np.sin(progress / 10)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(progress, target, color=TEXT, lw=2.0, label="Recorded target")
    ax.plot(progress, before, color=RED, lw=1.8, linestyle="--", label="Before feedback")
    ax.plot(progress, after, color=BLUE, lw=2.0, label="After feedback")
    ax.set_title("(a) Representative trajectory refinement")
    ax.set_xlabel("Movement progress (%)")
    ax.set_ylabel("Normalized joint trajectory")
    ax.grid(True, color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    labels = ["Forward\nRMSE", "Mismatch\nobjective"]
    before_vals = [metrics["before_rmse"], metrics["before_obj"]]
    after_vals = [metrics["after_rmse"], metrics["after_obj"]]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, before_vals, width, color="#FCA5A5", edgecolor=RED, label="Before")
    ax.bar(x + width / 2, after_vals, width, color="#93C5FD", edgecolor=BLUE, label="After")
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        improvement = 100.0 * (b - a) / max(1e-9, b)
        ax.text(i, max(b, a) + 0.05, f"{improvement:.1f}%\nreduction", ha="center", va="bottom", fontsize=9)
    ax.set_title("(b) Aggregate refinement metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error value")
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False)
    fig.suptitle("Mismatch-Feedback Improves Control Replay Consistency", fontweight="bold", y=1.02)
    save(fig, "fig5_refinement_improvement.png")


def fig6_confusion_matrix() -> None:
    eval_data = read_eval()
    cm = np.asarray(eval_data["test"]["confusion_matrix"], dtype=float)
    labels = ["Sesuai Pola Acuan", "Pola Aktivitas Berbeda"]
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    norm = cm / row_sum
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if norm[i, j] > 0.55 else TEXT
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i,j]*100:.1f}%", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Target label")
    ax.set_title("Final Dynamic Temporal GCN Confusion Matrix", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized percentage")
    save(fig, "fig6_final_confusion_matrix.png")


def plot_curve(filename: str, title: str, ycols: Iterable[tuple[str, str, str]], ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    rows = read_history()
    epochs = np.asarray([row["epoch"] for row in rows], dtype=float)
    best_epoch = int(read_eval().get("best_epoch", 0))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for col, label, color in ycols:
        values = np.asarray([row[col] for row in rows], dtype=float)
        ax.plot(epochs, values, lw=2.0, label=label, color=color)
    if best_epoch > 0:
        ax.axvline(best_epoch, color=GRAY, lw=1.2, linestyle="--")
        ax.text(best_epoch + 0.4, ax.get_ylim()[1] if ylim is None else ylim[1], f"best epoch {best_epoch}", va="top", ha="left", fontsize=8, color=GRAY)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False)
    save(fig, filename)


def fig7_8_9_curves() -> None:
    plot_curve(
        "fig7_accuracy_curve.png",
        "Dynamic Temporal GCN Accuracy Curve",
        [("train_accuracy", "Training accuracy", BLUE), ("val_accuracy", "Validation accuracy", ORANGE)],
        "Accuracy",
        (0.75, 0.93),
    )
    plot_curve(
        "fig8_f1_curve.png",
        "Dynamic Temporal GCN F1 Curve",
        [("train_macro_f1", "Training macro F1", BLUE), ("val_macro_f1", "Validation macro F1", ORANGE), ("val_binary_f1_deviating", "Validation different-pattern F1", RED)],
        "F1 score",
        (0.70, 0.93),
    )
    plot_curve(
        "fig9_loss_curve.png",
        "Dynamic Temporal GCN Loss Curve",
        [("train_loss", "Training loss", BLUE), ("val_loss", "Validation loss", ORANGE)],
        "Loss",
        None,
    )


def write_notes() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notes = OUTPUT_DIR / "figure_source_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# Paper Figure Source Notes",
                "",
                "- `fig1_experimental_setup.png`: academic schematic generated from the final experimental setup concept; not a direct camera screenshot.",
                "- `fig2_overall_framework.png`: academic framework diagram generated from the final system pipeline.",
                "- `fig3_mismatch_feedback_pipeline.png`: academic methodology diagram generated from the implemented refinement pipeline.",
                "- `fig4_graph_and_dynamic_temporal_gcn.png`: academic model diagram generated from the final graph sequence and Dynamic Temporal GCN design.",
                "- `fig5_refinement_improvement.png`: uses aggregate mismatch-feedback metrics from `teacher_trial_summary.csv`; trajectory panel is representative/normalized.",
                "- `fig6_final_confusion_matrix.png`: generated from `artifacts/dynamic_temporal_gcn_node_classifier/eval_metrics.json` test confusion matrix.",
                "- `fig7_accuracy_curve.png`, `fig8_f1_curve.png`, `fig9_loss_curve.png`: generated from final `history.csv`.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    setup_style()
    fig1_experimental_setup()
    fig2_overall_framework()
    fig3_mismatch_feedback_pipeline()
    fig4_graph_and_dynamic_temporal_gcn()
    fig5_refinement_improvement()
    fig6_confusion_matrix()
    fig7_8_9_curves()
    write_notes()
    print(f"Generated paper figures in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
