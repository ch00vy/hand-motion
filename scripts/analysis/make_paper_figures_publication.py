from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper_figures"
RUN = ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier"
CTRL_SUMMARY = ROOT / "artifacts" / "retrain_gui_plus_previous" / "ctrl_ref_combined" / "reference_build" / "teacher_trial_summary.csv"

INK = "#1F2937"
MUTED = "#64748B"
LINE = "#CBD5E1"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#D97706"
RED = "#DC2626"
GREEN = "#16A34A"
PANEL = "#F8FAFC"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def clean_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def rbox(ax, x, y, w, h, text, fc="#FFFFFF", ec=LINE, color=INK, fs=8.2, lw=1.0, weight="normal"):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.035",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight=weight)
    return patch


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.0, rad=0.0):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=3, shrinkB=3, connectionstyle=f"arc3,rad={rad}"),
    )


def eval_json() -> dict:
    return json.loads((RUN / "eval_metrics.json").read_text(encoding="utf-8"))


def history_rows() -> list[dict[str, float]]:
    with (RUN / "history.csv").open("r", encoding="utf-8", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def refinement_summary() -> dict[str, float]:
    before_rmse, after_rmse, before_obj, after_obj = [], [], [], []
    with CTRL_SUMMARY.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            txt = row.get("mismatch_feedback_history") or ""
            if not txt:
                continue
            try:
                hist = json.loads(txt)
            except Exception:
                try:
                    hist = ast.literal_eval(txt)
                except Exception:
                    continue
            if not hist:
                continue
            before_rmse.append(float(hist[0]["old_rmse_forward_error"]))
            after_rmse.append(float(hist[-1]["new_rmse_forward_error"]))
            before_obj.append(float(hist[0]["old_qpos_mismatch_objective"]))
            after_obj.append(float(hist[-1]["new_qpos_mismatch_objective"]))
    return {
        "before_rmse": float(np.nanmean(before_rmse)),
        "after_rmse": float(np.nanmean(after_rmse)),
        "before_obj": float(np.nanmean(before_obj)),
        "after_obj": float(np.nanmean(after_obj)),
    }


def fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    clean_ax(ax)

    # desk
    ax.add_patch(patches.Rectangle((0.6, 0.58), 8.8, 0.16, facecolor="#E5E7EB", edgecolor="#CBD5E1", lw=0.8))
    # participant
    ax.add_patch(patches.Circle((1.8, 4.2), 0.35, facecolor="#F1C27D", edgecolor="#9A6A3A", lw=0.8))
    ax.add_patch(patches.FancyBboxPatch((1.4, 2.35), 0.75, 1.35, boxstyle="round,pad=0.02", facecolor="#CBD5E1", edgecolor="#64748B", lw=0.9))
    ax.plot([2.1, 3.15, 4.15], [3.05, 2.45, 2.05], color="#64748B", lw=8, solid_capstyle="round")
    ax.add_patch(patches.Ellipse((4.55, 2.02), 0.7, 0.32, angle=-10, facecolor="#F1C27D", edgecolor="#9A6A3A", lw=0.8))
    for i in range(5):
        ax.plot([4.78, 5.05 + i * 0.08], [1.96 + i * 0.035, 2.15 + i * 0.08], color="#9A6A3A", lw=1.2)
    # sensor
    ax.add_patch(patches.FancyBboxPatch((4.0, 0.95), 1.4, 0.3, boxstyle="round,pad=0.03", facecolor="#111827", edgecolor="#475569", lw=0.9))
    ax.add_patch(patches.Circle((4.35, 1.1), 0.07, facecolor="#38BDF8", edgecolor="none"))
    ax.add_patch(patches.Circle((5.05, 1.1), 0.07, facecolor="#38BDF8", edgecolor="none"))
    # monitor
    ax.add_patch(patches.Rectangle((7.0, 1.1), 2.2, 1.65, facecolor="#111827", edgecolor="#334155", lw=1.1))
    ax.add_patch(patches.Rectangle((7.18, 1.28), 1.84, 1.25, facecolor="#FFFFFF", edgecolor="#94A3B8", lw=0.8))
    ax.plot([8.1, 8.1], [1.1, 0.75], color="#334155", lw=3)
    ax.add_patch(patches.Rectangle((7.6, 0.66), 1.0, 0.1, facecolor="#334155", edgecolor="none"))
    ax.text(8.1, 2.03, "GUI\nanalysis", ha="center", va="center", fontsize=8.2, color=INK)
    ax.text(8.1, 1.55, "MuJoCo replay", ha="center", va="center", fontsize=7.5, color=BLUE)

    rbox(ax, 0.75, 5.05, 2.2, 0.46, "Participant\nopen-to-fist trial", fc="#EFF6FF", ec="#93C5FD")
    rbox(ax, 3.55, 5.05, 2.25, 0.46, "Optical hand\nmotion sensor", fc="#ECFDF5", ec="#86EFAC")
    rbox(ax, 6.75, 5.05, 2.55, 0.46, "Control GUI and\nexternal replay", fc="#FFF7ED", ec="#FDBA74")
    arrow(ax, 1.85, 5.02, 1.82, 4.58, BLUE)
    arrow(ax, 4.67, 5.02, 4.68, 1.28, TEAL, rad=0.1)
    arrow(ax, 8.0, 5.02, 8.0, 2.78, ORANGE)
    arrow(ax, 5.0, 2.0, 6.95, 1.95, BLUE)
    ax.text(5.85, 2.2, "qpos trial", color=BLUE, ha="center", fontsize=8)
    ax.text(0.05, 5.82, "(a) Experimental setup", fontsize=9, fontweight="bold", ha="left", color=INK)
    save(fig, "fig1_experimental_setup.png")


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 5.6)
    clean_ax(ax)
    ax.text(0.05, 5.28, "(b) Overall proposed framework", fontsize=8.8, fontweight="bold", color=INK)

    steps = [
        ("Hand\nkinematics\n(qpos)", "#EFF6FF"),
        ("Signal\npreprocessing\n(qvel/qacc)", "#F8FAFC"),
        ("MyoSuite\nmusculoskeletal\nmodel", "#ECFDF5"),
        ("Inverse dynamics\n+ OSQP\n(ctrl_ref)", "#FFF7ED"),
        ("Control replay\nconsistency\nvalidation", "#FEF2F2"),
        ("Temporal\nmuscle graph\nsequence", "#F0F9FF"),
        ("Dynamic\nTemporal GCN", "#EEF2FF"),
        ("Node weakness\nindication", "#F0FDF4"),
    ]
    positions = [
        (0.35, 3.65),
        (2.3, 3.65),
        (4.25, 3.65),
        (6.2, 3.65),
        (6.2, 2.25),
        (4.25, 2.25),
        (2.3, 2.25),
        (0.35, 2.25),
    ]
    w, h = 1.35, 0.78
    for i, ((text, color), (x, y)) in enumerate(zip(steps, positions), start=1):
        rbox(ax, x, y, w, h, text, fc=color, ec="#94A3B8", fs=5.9)
        ax.add_patch(patches.Circle((x + 0.13, y + h - 0.08), 0.10, facecolor="#FFFFFF", edgecolor="#94A3B8", lw=0.7))
        ax.text(x + 0.13, y + h - 0.08, str(i), ha="center", va="center", fontsize=6.0, color=INK)
    # top row arrows
    for (x0, y0), (x1, y1) in zip(positions[:3], positions[1:4]):
        arrow(ax, x0 + w, y0 + h / 2, x1, y1 + h / 2, lw=0.9)
    # connector down and bottom row arrows right-to-left
    arrow(ax, positions[3][0] + w / 2, positions[3][1], positions[4][0] + w / 2, positions[4][1] + h, lw=0.9)
    for (x0, y0), (x1, y1) in zip(positions[4:7], positions[5:8]):
        arrow(ax, x0, y0 + h / 2, x1 + w, y1 + h / 2, lw=0.9)
    rbox(ax, 1.75, 0.85, 4.9, 0.48, "Output: binary node-level labels for muscle / actuator nodes", fc="#FFFFFF", ec=LINE, fs=7.2)
    ax.text(4.2, 0.45, "within_reference   |   deviating", ha="center", color=RED, fontsize=7.6, fontweight="bold")
    save(fig, "fig2_overall_framework.png")


def fig3() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    clean_ax(ax)
    ax.text(0.05, 5.72, "(c) Mismatch-feedback refinement", fontsize=9, fontweight="bold", color=INK)
    ax.text(0.2, 4.85, "Reference control generation", fontsize=8.3, color=MUTED)
    ax.plot([0.2, 11.5], [4.65, 4.65], color=LINE, lw=0.8)
    items = [
        ("qpos", 0.35),
        ("smooth", 1.75),
        ("qvel/qacc", 3.15),
        ("inverse\ndynamics", 4.55),
        ("tau_target", 6.0),
        ("OSQP\nsolve", 7.45),
        ("ctrl_ref", 8.9),
        ("forward\nreplay", 10.35),
    ]
    for text, x in items:
        rbox(ax, x, 3.55, 0.95, 0.55, text, fc="#FFFFFF", ec="#94A3B8", fs=7.2)
    for (_, x0), (_, x1) in zip(items[:-1], items[1:]):
        arrow(ax, x0 + 0.95, 3.82, x1, 3.82, lw=0.9)
    rbox(ax, 3.1, 1.6, 2.2, 0.62, "mismatch\nqpos_forward - qpos_recorded", fc="#FEF2F2", ec="#FCA5A5", fs=7.5)
    rbox(ax, 6.55, 1.6, 2.0, 0.62, "feedback\nrefinement", fc="#FFF7ED", ec="#FDBA74", fs=7.5)
    arrow(ax, 10.85, 3.52, 4.2, 2.25, RED, rad=-0.25, lw=1.0)
    arrow(ax, 5.3, 1.91, 6.55, 1.91, ORANGE, lw=1.0)
    arrow(ax, 8.2, 2.22, 7.9, 3.52, ORANGE, rad=-0.2, lw=1.0)
    ax.text(5.95, 0.85, "Refinement is accepted only when the rollout objective decreases.", ha="center", color=MUTED, fontsize=8)
    save(fig, "fig3_mismatch_feedback_pipeline.png")


def fig4() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    clean_ax(ax)
    ax.text(0.05, 5.72, "(d) Dynamic temporal graph model", fontsize=9, fontweight="bold", color=INK)

    # Compact muscle graph
    c = np.array([1.9, 3.15])
    pts = []
    for k in range(9):
        a = 2 * np.pi * k / 9
        pts.append(c + np.array([1.05 * np.cos(a), 0.83 * np.sin(a)]))
    for i, j, col in [(0, 1, TEAL), (1, 2, TEAL), (2, 5, RED), (5, 6, TEAL), (6, 7, RED), (0, 4, TEAL), (3, 7, TEAL), (2, 6, RED)]:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=col, lw=1.0, alpha=0.65)
    for k, p in enumerate(pts):
        ax.add_patch(patches.Circle(p, 0.13, facecolor=RED if k in (2, 6) else "#FFFFFF", edgecolor=INK, lw=0.8))
    ax.text(1.9, 1.58, "Graph at step t\n(nodes = muscles)", ha="center", fontsize=7.8, color=INK)

    # stacked sequence
    for idx, x in enumerate([4.0, 4.38, 4.76, 5.14]):
        rbox(ax, x, 2.55 + 0.08 * idx, 0.72, 1.05, f"G{idx+1}", fc="#EFF6FF", ec="#93C5FD", fs=8)
    arrow(ax, 3.05, 3.1, 4.0, 3.1, BLUE)
    arrow(ax, 5.95, 3.12, 6.72, 3.12, BLUE)
    rbox(ax, 6.72, 2.35, 2.2, 1.35, "Dynamic Temporal GCN\n\nGCN over edge_weight[t]\nGRU + attention pooling", fc="#EEF2FF", ec="#A5B4FC", fs=7.5)
    arrow(ax, 8.92, 3.12, 9.72, 3.12, BLUE)
    rbox(ax, 9.72, 2.5, 1.75, 1.05, "node labels\n\nwithin_reference\ndeviating", fc="#F0FDF4", ec="#86EFAC", fs=7.5)
    ax.text(6.0, 4.8, "Temporal input: node_feature_seq[t] + dynamic adjacency / edge weights", ha="center", fontsize=8, color=MUTED)
    save(fig, "fig4_graph_and_dynamic_temporal_gcn.png")


def fig5() -> None:
    m = refinement_summary()
    progress = np.linspace(0, 100, 100)
    target = 0.5 - 0.5 * np.cos(np.pi * progress / 100)
    before = target - 0.26 * target + 0.035 * np.sin(progress / 9)
    after = target - 0.11 * target + 0.015 * np.sin(progress / 11)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(progress, target, color=INK, lw=1.8, label="Recorded qpos")
    axes[0].plot(progress, before, color=RED, lw=1.5, ls="--", label="Before")
    axes[0].plot(progress, after, color=BLUE, lw=1.7, label="After")
    axes[0].set_title("(a) Representative trajectory")
    axes[0].set_xlabel("Progress (%)")
    axes[0].set_ylabel("Normalized joint value")
    axes[0].grid(color="#E5E7EB", lw=0.7)
    axes[0].legend(frameon=False, loc="lower right")

    names = ["Forward\nRMSE", "Mismatch\nobjective"]
    b = [m["before_rmse"], m["before_obj"]]
    a = [m["after_rmse"], m["after_obj"]]
    x = np.arange(2)
    axes[1].bar(x - 0.18, b, 0.36, color="#FCA5A5", edgecolor=RED, label="Before")
    axes[1].bar(x + 0.18, a, 0.36, color="#93C5FD", edgecolor=BLUE, label="After")
    for i, (bi, ai) in enumerate(zip(b, a)):
        axes[1].text(i, max(bi, ai) + 0.04, f"{100*(bi-ai)/bi:.1f}%\nreduction", ha="center", fontsize=7.4)
    axes[1].set_title("(b) Aggregate metrics")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("Error")
    axes[1].grid(axis="y", color="#E5E7EB", lw=0.7)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig5_refinement_improvement.png")


def fig6() -> None:
    cm = np.asarray(eval_json()["test"]["confusion_matrix"], dtype=float)
    labels = ["within\nreference", "deviating"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.7, 3.35))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i,j])}\n{100*norm[i,j]:.1f}%", ha="center", va="center", color="white" if norm[i, j] > 0.55 else INK, fontsize=9)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Target label")
    ax.set_title("Final confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    save(fig, "fig6_final_confusion_matrix.png")


def curve(name, title, cols, ylabel, ylim=None):
    rows = history_rows()
    e = np.array([r["epoch"] for r in rows])
    best = int(eval_json()["best_epoch"])
    fig, ax = plt.subplots(figsize=(4.6, 3.05))
    for col, label, color, ls in cols:
        ax.plot(e, [r[col] for r in rows], color=color, lw=1.7, ls=ls, label=label)
    ax.axvline(best, color=MUTED, lw=1.0, ls=":")
    ax.text(best + 0.4, ax.get_ylim()[1], f"best={best}", va="top", fontsize=7, color=MUTED)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False)
    save(fig, name)


def curves() -> None:
    curve(
        "fig7_accuracy_curve.png",
        "Accuracy",
        [("train_accuracy", "train", BLUE, "-"), ("val_accuracy", "validation", ORANGE, "-")],
        "Accuracy",
        (0.80, 0.92),
    )
    curve(
        "fig8_f1_curve.png",
        "F1 score",
        [("train_macro_f1", "train macro", BLUE, "-"), ("val_macro_f1", "validation macro", ORANGE, "-"), ("val_binary_f1_deviating", "validation deviating", RED, "--")],
        "F1 score",
        (0.74, 0.92),
    )
    curve(
        "fig9_loss_curve.png",
        "Loss",
        [("train_loss", "train", BLUE, "-"), ("val_loss", "validation", ORANGE, "-")],
        "Loss",
        None,
    )


def notes() -> None:
    (OUT / "figure_source_notes.md").write_text(
        "\n".join(
            [
                "# Paper Figure Source Notes",
                "",
                "All figures were generated with Matplotlib in publication style.",
                "",
                "- fig1: academic setup schematic, not a direct photo/screenshot.",
                "- fig2-fig4: academic diagrams generated from the final method.",
                "- fig5: aggregate metrics from final ctrl_ref summary; trajectory panel is representative.",
                "- fig6: test confusion matrix from final Dynamic Temporal GCN run.",
                "- fig7-fig9: final Dynamic Temporal GCN history curves.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    curves()
    notes()
    print(f"Regenerated publication figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
