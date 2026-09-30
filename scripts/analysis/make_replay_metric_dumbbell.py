from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "bab_4_7_3_replay_evidence"
NORMAL_SUMMARY = ROOT / "artifacts" / "retrain_labeled_records" / "ctrl_ref_normal" / "reference_build" / "biomech_reference_trial_summary.csv"
DIFF_SUMMARY = ROOT / "artifacts" / "retrain_labeled_records" / "ctrl_ref_abnormal" / "reference_build" / "biomech_reference_trial_summary.csv"

BLUE = "#1F4E79"
BROWN = "#8C6D31"
GRAY = "#9CA3AF"
DARK = "#111827"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#D1D5DB",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def polish(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()

    normal = pd.read_csv(NORMAL_SUMMARY)
    different = pd.read_csv(DIFF_SUMMARY)

    rows = [
        ("Mean joint error", "mean_joint_error"),
        ("Mean RMSE", "rmse_forward_error"),
        ("Mean ctrl delta", "ctrl_delta_mean"),
    ]
    labels = [r[0] for r in rows]
    normal_values = np.array([pd.to_numeric(normal[col], errors="coerce").mean() for _, col in rows], dtype=float)
    different_values = np.array([pd.to_numeric(different[col], errors="coerce").mean() for _, col in rows], dtype=float)
    increase_pct = (different_values - normal_values) / normal_values * 100.0

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))

    for yi, nval, dval, pct in zip(y, normal_values, different_values, increase_pct):
        ax.plot([nval, dval], [yi, yi], color=GRAY, linewidth=1.2, zorder=1)
        ax.scatter(nval, yi, s=58, color=BLUE, edgecolor="white", linewidth=0.8, label="Normal" if yi == y[0] else "", zorder=3)
        ax.scatter(dval, yi, s=58, color=BROWN, edgecolor="white", linewidth=0.8, label="Pola berbeda" if yi == y[0] else "", zorder=3)
        ax.text(
            max(nval, dval) + 0.045,
            yi,
            f"+{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
            color=DARK,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Nilai metrik")
    ax.set_xlim(0, max(different_values.max(), normal_values.max()) * 1.22)
    ax.legend(loc="lower right", fontsize=8)
    polish(ax)

    png = OUT / "fig_4_7_3_replay_metric_dumbbell.png"
    pdf = OUT / "fig_4_7_3_replay_metric_dumbbell.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = [
        "# Perbaikan Figure Bab 4.7.3",
        "",
        "## Data yang Digunakan",
        f"- Trial normal: `{NORMAL_SUMMARY}`",
        f"- Trial pola gerakan berbeda: `{DIFF_SUMMARY}`",
        "",
        "## Jenis Data",
        "Data berasal dari agregat per-trial pada summary validasi replay post-recording. Nilai pada figure adalah rerata dari kolom metrik pada setiap kelompok trial.",
        "",
        "## Nilai Kenaikan",
        "",
        "| Metrik | Trial normal | Trial pola gerakan berbeda | Kenaikan |",
        "|---|---:|---:|---:|",
    ]
    for label, nval, dval, pct in zip(labels, normal_values, different_values, increase_pct):
        report.append(f"| {label} | {nval:.4f} | {dval:.4f} | {pct:.1f}% |")
    report.extend(
        [
            "",
            "## Alasan Memilih Dumbbell Plot",
            "Dumbbell plot dipilih karena tujuan figure adalah menunjukkan perbandingan langsung antar dua kelompok pada tiga metrik yang berbeda. Format ini lebih ringkas daripada tiga panel bar chart, mengurangi pengulangan label panjang, dan membuat arah kenaikan dari trial normal ke trial pola gerakan berbeda lebih jelas.",
        ]
    )
    (OUT / "report_figure_fix.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Output PNG: {png}")
    print(f"Output PDF: {pdf}")
    for label, nval, dval, pct in zip(labels, normal_values, different_values, increase_pct):
        print(f"{label}: normal={nval:.4f}, pola_berbeda={dval:.4f}, increase={pct:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
