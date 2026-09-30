from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "bab_4_7_3_replay_evidence"

NORMAL_SUMMARY = ROOT / "artifacts" / "retrain_labeled_records" / "ctrl_ref_normal" / "reference_build" / "biomech_reference_trial_summary.csv"
DIFF_SUMMARY = ROOT / "artifacts" / "retrain_labeled_records" / "ctrl_ref_abnormal" / "reference_build" / "biomech_reference_trial_summary.csv"
REFINE_DIR = ROOT / "artifacts" / "qpos_refinement_comparison"
CHAINED_SWEEP = REFINE_DIR / "chained_refinement_sweep.csv"

BLUE = "#1F4E79"
GRAY = "#6B7280"
LIGHT = "#D1D5DB"
DARK = "#111827"
BROWN = "#8C6D31"

TARGETS = {
    ("trial_normal", "mean_joint_error"): 0.1583,
    ("trial_normal", "mean_rmse"): 0.8727,
    ("trial_normal", "mean_ctrl_delta"): 0.1417,
    ("trial_pola_gerakan_berbeda", "mean_joint_error"): 0.2220,
    ("trial_pola_gerakan_berbeda", "mean_rmse"): 1.3154,
    ("trial_pola_gerakan_berbeda", "mean_ctrl_delta"): 0.2183,
    ("refinement_terbaik", "rmse_forward_error"): 0.1325,
    ("refinement_terbaik", "mean_pose_error_norm"): 0.1153,
}


@dataclass
class MetricRow:
    metric_name: str
    dataset_group: str
    computed_value: float
    target_value: float | str
    absolute_difference: float | str
    source_file: str
    source_column: str
    unit: str
    note: str


@dataclass
class FigureRow:
    filename: str
    status: str
    source_file: str
    caption: str
    note: str = ""


METRICS: list[MetricRow] = []
FIGURES: list[FigureRow] = []
NOT_CREATED: list[str] = []


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
            "axes.edgecolor": LIGHT,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def polish(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if ygrid:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    if xgrid:
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)


def save(fig: plt.Figure, filename: str, source: str, caption: str, note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    for ax in fig.axes:
        if hasattr(ax, "spines"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES.append(FigureRow(filename, "CREATED", source, caption, note))


def add_metric(
    dataset_group: str,
    metric_name: str,
    value: float,
    source_file: Path,
    source_column: str,
    unit: str,
    note: str,
) -> None:
    target = TARGETS.get((dataset_group, metric_name))
    diff: float | str = "" if target is None else abs(float(value) - float(target))
    METRICS.append(
        MetricRow(
            metric_name=metric_name,
            dataset_group=dataset_group,
            computed_value=float(value),
            target_value="" if target is None else float(target),
            absolute_difference=diff,
            source_file=str(source_file),
            source_column=source_column,
            unit=unit,
            note=note,
        )
    )


def annotate_barh(ax: plt.Axes, fmt: str = "{:.4f}") -> None:
    for patch in ax.patches:
        val = patch.get_width()
        if np.isfinite(val):
            ax.text(val, patch.get_y() + patch.get_height() / 2, fmt.format(val), va="center", ha="left", fontsize=8)


def compute_metrics(normal: pd.DataFrame, diff: pd.DataFrame, sweep: pd.DataFrame) -> pd.Series:
    for group, df, source in [
        ("trial_normal", normal, NORMAL_SUMMARY),
        ("trial_pola_gerakan_berbeda", diff, DIFF_SUMMARY),
    ]:
        if df.empty:
            continue
        add_metric(group, "mean_joint_error", pd.to_numeric(df["mean_joint_error"], errors="coerce").mean(), source, "mean_joint_error", "model unit", "Rata-rata dari summary per-trial post-recording validation.")
        add_metric(group, "mean_rmse", pd.to_numeric(df["rmse_forward_error"], errors="coerce").mean(), source, "rmse_forward_error", "model unit", "Rata-rata dari summary per-trial post-recording validation.")
        add_metric(group, "mean_ctrl_delta", pd.to_numeric(df["ctrl_delta_mean"], errors="coerce").mean(), source, "ctrl_delta_mean", "model unit", "Rata-rata dari summary per-trial post-recording validation.")
        add_metric(group, "num_trials", len(df), source, "rows", "count", "Jumlah trial pada summary validasi.")

    best = pd.Series(dtype=object)
    if not sweep.empty and "rmse" in sweep.columns:
        best = sweep.sort_values("rmse").iloc[0]
        add_metric("refinement_terbaik", "rmse_forward_error", float(best["rmse"]), CHAINED_SWEEP, "rmse", "model unit", "Baris terbaik berdasarkan RMSE terendah pada chained refinement sweep.")
        add_metric("refinement_terbaik", "mean_pose_error_norm", float(best["mean"]), CHAINED_SWEEP, "mean", "model unit", "Mean pose error norm dari baris terbaik pada chained refinement sweep.")
        for metric, col in [
            ("median_frame_error", "med"),
            ("p95_frame_error", "p95"),
            ("max_frame_error", "mx"),
            ("mean_joint_error", "mj"),
            ("mean_ctrl_delta", "delta"),
            ("saturation_fraction", "sat"),
        ]:
            if col in best.index:
                add_metric("refinement_terbaik", metric, float(best[col]), CHAINED_SWEEP, col, "model unit", "Metrik tambahan dari baris refinement terbaik.")
    return best


def figure_normal_vs_different(normal: pd.DataFrame, diff: pd.DataFrame) -> None:
    if normal.empty or diff.empty:
        NOT_CREATED.append("fig_4_7_3_normal_vs_different_pattern.png: summary trial normal atau trial pola gerakan berbeda tidak ditemukan")
        return
    metrics = [
        ("Mean joint error", "mean_joint_error"),
        ("Mean RMSE", "rmse_forward_error"),
        ("Mean ctrl delta", "ctrl_delta_mean"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4))
    for ax, (label, col) in zip(axes, metrics):
        vals = [
            pd.to_numeric(normal[col], errors="coerce").mean(),
            pd.to_numeric(diff[col], errors="coerce").mean(),
        ]
        ax.barh(["Trial normal", "Trial pola\ngerakan berbeda"], vals, color=[BLUE, BROWN])
        ax.set_xlabel("Nilai metrik")
        ax.set_title(label)
        annotate_barh(ax)
        polish(ax, xgrid=True, ygrid=False)
    save(
        fig,
        "fig_4_7_3_normal_vs_different_pattern.png",
        f"{NORMAL_SUMMARY}; {DIFF_SUMMARY}",
        "Perbandingan mean joint error, mean RMSE, dan mean ctrl delta antara trial normal dan trial pola gerakan berbeda pada validasi replay post-recording.",
    )


def figure_refinement(best: pd.Series) -> None:
    if best.empty:
        NOT_CREATED.append("fig_4_7_3_refinement_best_result.png: chained refinement sweep tidak ditemukan")
        return
    labels = ["RMSE forward error", "Mean pose error norm"]
    vals = [float(best["rmse"]), float(best["mean"])]
    if "med" in best.index:
        labels.append("Median frame error")
        vals.append(float(best["med"]))
    if "p95" in best.index:
        labels.append("P95 frame error")
        vals.append(float(best["p95"]))
    if "mx" in best.index:
        labels.append("Max frame error")
        vals.append(float(best["mx"]))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(labels[::-1], vals[::-1], color=[GRAY, GRAY, GRAY, BLUE, BLUE][: len(vals)][::-1])
    ax.set_xlabel("Nilai metrik")
    annotate_barh(ax)
    polish(ax, xgrid=True, ygrid=False)
    save(
        fig,
        "fig_4_7_3_refinement_best_result.png",
        str(CHAINED_SWEEP),
        "Ringkasan metrik pada hasil refinement terbaik untuk validasi replay berbasis simulasi.",
        f"Konfigurasi terbaik: {best.get('name', '-')}",
    )


def figure_distribution(normal: pd.DataFrame, diff: pd.DataFrame) -> None:
    if normal.empty or diff.empty:
        NOT_CREATED.append("fig_4_7_3_replay_distribution.png: data per-trial tidak lengkap")
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, col, title, ylabel in [
        (axes[0], "mean_joint_error", "A. Mean joint error per trial", "Mean joint error"),
        (axes[1], "ctrl_delta_mean", "B. Ctrl delta per trial", "Mean ctrl delta"),
    ]:
        data = [
            pd.to_numeric(normal[col], errors="coerce").dropna().to_numpy(),
            pd.to_numeric(diff[col], errors="coerce").dropna().to_numpy(),
        ]
        bp = ax.boxplot(data, tick_labels=["Normal", "Pola\nberbeda"], patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], [BLUE, BROWN]):
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_edgecolor(DARK)
        for median in bp["medians"]:
            median.set_color(DARK)
            median.set_linewidth(1.2)
        rng = np.random.default_rng(7)
        for idx, vals in enumerate(data, start=1):
            if len(vals):
                jitter = rng.normal(0, 0.025, size=len(vals))
                ax.scatter(np.full(len(vals), idx) + jitter, vals, s=18, color=[BLUE, BROWN][idx - 1], alpha=0.55, edgecolors="none")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        polish(ax)
    save(
        fig,
        "fig_4_7_3_replay_distribution.png",
        f"{NORMAL_SUMMARY}; {DIFF_SUMMARY}",
        "Distribusi joint error dan ctrl delta pada validasi replay post-recording.",
        "Data distribusi berasal dari summary per-trial, bukan per-frame.",
    )


def write_outputs(normal: pd.DataFrame, diff: pd.DataFrame, sweep: pd.DataFrame, best: pd.Series) -> None:
    pd.DataFrame([m.__dict__ for m in METRICS]).to_csv(OUT / "replay_metric_summary.csv", index=False)

    lines = [
        "# Replay Evidence Bab 4.7.3",
        "",
        "Audit ini menggunakan file output asli. Tidak ada retraining, tidak ada perubahan model, dan tidak ada data dummy.",
        "",
        "## File Sumber Data yang Ditemukan",
        f"- `{NORMAL_SUMMARY}`: summary per-trial post-recording validation untuk trial normal.",
        f"- `{DIFF_SUMMARY}`: summary per-trial post-recording validation untuk trial pola gerakan berbeda. Nama folder sumber masih memakai istilah legacy internal.",
        f"- `{CHAINED_SWEEP}`: sweep refinement untuk mencari hasil refinement terbaik.",
        "",
        "## Kolom/Key yang Digunakan",
        "- `mean_joint_error` untuk mean joint error per trial.",
        "- `rmse_forward_error` untuk RMSE replay per trial.",
        "- `ctrl_delta_mean` untuk mean ctrl delta per trial.",
        "- `rmse`, `mean`, `med`, `p95`, `mx`, `mj`, dan `delta` pada chained refinement sweep.",
        "",
        "## Jenis Data",
        f"- Trial normal: per-trial aggregate, jumlah baris {len(normal)}.",
        f"- Trial pola gerakan berbeda: per-trial aggregate, jumlah baris {len(diff)}.",
        f"- Refinement: sweep aggregate, jumlah baris {len(sweep)}.",
        "",
        "## Rumus/Metode Hitung Ulang",
        "- Mean joint error kelompok = rata-rata kolom `mean_joint_error` pada summary per-trial.",
        "- Mean RMSE kelompok = rata-rata kolom `rmse_forward_error` pada summary per-trial.",
        "- Mean ctrl delta kelompok = rata-rata kolom `ctrl_delta_mean` pada summary per-trial.",
        "- Refinement terbaik = baris dengan nilai `rmse` terendah pada `chained_refinement_sweep.csv`.",
        "",
        "## Perbandingan Target",
        "",
        "| Group | Metric | Computed | Target | Selisih | Sumber |",
        "|---|---|---:|---:|---:|---|",
    ]
    for m in METRICS:
        if m.target_value == "":
            continue
        lines.append(
            f"| {m.dataset_group} | {m.metric_name} | {m.computed_value:.6f} | {m.target_value} | {m.absolute_difference} | `{Path(m.source_file).name}:{m.source_column}` |"
        )
    lines.extend(["", "## Figure Berhasil Dibuat"])
    for fig in FIGURES:
        lines.append(f"- `{fig.filename}`: {fig.caption}")
        if fig.note:
            lines.append(f"  Catatan: {fig.note}")
    lines.append("")
    lines.append("## Figure Tidak Dibuat")
    if NOT_CREATED:
        for item in NOT_CREATED:
            lines.append(f"- {item}")
    else:
        lines.append("- Tidak ada.")
    lines.extend(
        [
            "",
            "## Kalimat Rujukan Aman untuk Bab 4.7.3",
            "Berdasarkan summary post-recording validation, trial pola gerakan berbeda menunjukkan mean joint error, mean RMSE, dan mean ctrl delta yang lebih tinggi dibandingkan trial normal. Hal ini menunjukkan bahwa replay berbasis control otot lebih sulit mereproduksi pola gerakan yang berbeda dari pola acuan. Pada sweep refinement, konfigurasi terbaik menghasilkan RMSE forward error sekitar 0,1325 dan mean pose error norm sekitar 0,1153, sehingga refinement dapat memperbaiki kedekatan replay terhadap target pada kondisi tertentu.",
        ]
    )
    (OUT / "report_4_7_3_replay_evidence.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    normal = read_csv(NORMAL_SUMMARY)
    diff = read_csv(DIFF_SUMMARY)
    sweep = read_csv(CHAINED_SWEEP)
    best = compute_metrics(normal, diff, sweep)
    figure_normal_vs_different(normal, diff)
    figure_refinement(best)
    figure_distribution(normal, diff)
    write_outputs(normal, diff, sweep, best)

    key = {(m.dataset_group, m.metric_name): m for m in METRICS}
    print(f"Output folder: {OUT}")
    print(f"Source files: {NORMAL_SUMMARY}; {DIFF_SUMMARY}; {CHAINED_SWEEP}")
    print("Recomputed / audited metrics:")
    for group, metric in TARGETS:
        m = key[(group, metric)]
        ok = float(m.absolute_difference) <= 1e-3
        print(f" - {group} {metric}: {m.computed_value:.6f} target={m.target_value} diff={m.absolute_difference} match={ok}")
    print("Created figures:")
    for fig in FIGURES:
        print(f" - {fig.filename}")
    if NOT_CREATED:
        print("Not created:")
        for item in NOT_CREATED:
            print(f" - {item}")
    print("Best figure for Bab 4.7.3:")
    print(" - fig_4_7_3_normal_vs_different_pattern.png")
    print(" - fig_4_7_3_refinement_best_result.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
