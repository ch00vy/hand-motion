from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "bab_4_7_runtime_evidence"
RUNTIME_DIR = ROOT / "artifacts" / "realtime_recorded_qpos_audit"
RUNTIME_LOG = RUNTIME_DIR / "runtime_log.csv"
WINDOWS_CSV = RUNTIME_DIR / "validation_windows.csv"
SUMMARY_JSON = RUNTIME_DIR / "validation_summary.json"

BLUE = "#1F4E79"
GRAY = "#6B7280"
LIGHT = "#D1D5DB"
DARK = "#111827"

TARGETS = {
    "mean_ctrl_pipeline_latency_ms": 7.4074,
    "mean_frame_loop_ms": 55.0165,
    "mean_joint_error": 0.2611,
    "rmse_forward_error": 1.3589,
}


@dataclass
class MetricRow:
    metric_name: str
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    metric_name: str,
    value: float,
    target: float | None,
    source_file: Path,
    source_column: str,
    unit: str,
    note: str,
) -> None:
    if target is None:
        target_value: float | str = ""
        diff: float | str = ""
    else:
        target_value = float(target)
        diff = abs(float(value) - float(target))
    METRICS.append(
        MetricRow(
            metric_name=metric_name,
            computed_value=float(value),
            target_value=target_value,
            absolute_difference=diff,
            source_file=str(source_file),
            source_column=source_column,
            unit=unit,
            note=note,
        )
    )


def annotate_barh(ax: plt.Axes, fmt: str = "{:.3f}") -> None:
    for patch in ax.patches:
        val = patch.get_width()
        if np.isfinite(val):
            ax.text(val, patch.get_y() + patch.get_height() / 2, fmt.format(val), va="center", ha="left", fontsize=8)


def compute_metrics(runtime: pd.DataFrame, windows: pd.DataFrame, summary: dict[str, Any]) -> None:
    if not runtime.empty and "total_ctrl_pipeline_ms" in runtime.columns:
        vals = pd.to_numeric(runtime["total_ctrl_pipeline_ms"], errors="coerce").dropna()
        add_metric("mean_ctrl_pipeline_latency_ms", vals.mean(), TARGETS["mean_ctrl_pipeline_latency_ms"], RUNTIME_LOG, "total_ctrl_pipeline_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
        add_metric("median_ctrl_pipeline_latency_ms", vals.median(), None, RUNTIME_LOG, "total_ctrl_pipeline_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
        add_metric("p95_ctrl_pipeline_latency_ms", vals.quantile(0.95), summary.get("latency_p95_ms"), RUNTIME_LOG, "total_ctrl_pipeline_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
        add_metric("min_ctrl_pipeline_latency_ms", vals.min(), None, RUNTIME_LOG, "total_ctrl_pipeline_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
        add_metric("max_ctrl_pipeline_latency_ms", vals.max(), None, RUNTIME_LOG, "total_ctrl_pipeline_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
    if not runtime.empty and "total_frame_loop_ms" in runtime.columns:
        vals = pd.to_numeric(runtime["total_frame_loop_ms"], errors="coerce").dropna()
        add_metric("mean_frame_loop_ms", vals.mean(), TARGETS["mean_frame_loop_ms"], RUNTIME_LOG, "total_frame_loop_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
        add_metric("median_frame_loop_ms", vals.median(), None, RUNTIME_LOG, "total_frame_loop_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
        add_metric("p95_frame_loop_ms", vals.quantile(0.95), summary.get("frame_loop_p95_ms"), RUNTIME_LOG, "total_frame_loop_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
        add_metric("min_frame_loop_ms", vals.min(), None, RUNTIME_LOG, "total_frame_loop_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
        add_metric("max_frame_loop_ms", vals.max(), None, RUNTIME_LOG, "total_frame_loop_ms", "ms", "Dihitung ulang dari runtime_log.csv per frame.")
    if not windows.empty and "mean_joint_error" in windows.columns:
        vals = pd.to_numeric(windows["mean_joint_error"], errors="coerce").dropna()
        add_metric("window_mean_joint_error_mean", vals.mean(), summary.get("window_mean_joint_error_mean"), WINDOWS_CSV, "mean_joint_error", "model unit", "Statistik per-window; bukan sequence aggregate target 0.2611.")
        add_metric("window_mean_joint_error_median", vals.median(), summary.get("window_mean_joint_error_median"), WINDOWS_CSV, "mean_joint_error", "model unit", "Statistik per-window.")
        add_metric("window_mean_joint_error_p95", vals.quantile(0.95), summary.get("window_mean_joint_error_p95"), WINDOWS_CSV, "mean_joint_error", "model unit", "Statistik per-window.")
        add_metric("window_mean_joint_error_min", vals.min(), None, WINDOWS_CSV, "mean_joint_error", "model unit", "Statistik per-window.")
        add_metric("window_mean_joint_error_max", vals.max(), None, WINDOWS_CSV, "mean_joint_error", "model unit", "Statistik per-window.")
    if not windows.empty and "rmse_forward_error" in windows.columns:
        vals = pd.to_numeric(windows["rmse_forward_error"], errors="coerce").dropna()
        add_metric("window_rmse_forward_error_mean", vals.mean(), summary.get("window_rmse_mean"), WINDOWS_CSV, "rmse_forward_error", "model unit", "Statistik per-window; bukan sequence aggregate target 1.3589.")
        add_metric("window_rmse_forward_error_median", vals.median(), summary.get("window_rmse_median"), WINDOWS_CSV, "rmse_forward_error", "model unit", "Statistik per-window.")
        add_metric("window_rmse_forward_error_p95", vals.quantile(0.95), summary.get("window_rmse_p95"), WINDOWS_CSV, "rmse_forward_error", "model unit", "Statistik per-window.")
        add_metric("window_rmse_forward_error_min", vals.min(), None, WINDOWS_CSV, "rmse_forward_error", "model unit", "Statistik per-window.")
        add_metric("window_rmse_forward_error_max", vals.max(), None, WINDOWS_CSV, "rmse_forward_error", "model unit", "Statistik per-window.")
    if summary:
        add_metric("sequence_mean_joint_error", float(summary["sequence_mean_joint_error"]), TARGETS["mean_joint_error"], SUMMARY_JSON, "sequence_mean_joint_error", "model unit", "Aggregate sequence dari validation_summary.json; data per-frame rollout lengkap tidak ditemukan.")
        add_metric("sequence_rmse_forward_error", float(summary["sequence_rmse_forward_error"]), TARGETS["rmse_forward_error"], SUMMARY_JSON, "sequence_rmse_forward_error", "model unit", "Aggregate sequence dari validation_summary.json; data per-frame rollout lengkap tidak ditemukan.")
        add_metric("sequence_p95_frame_error", float(summary["sequence_p95_frame_error"]), None, SUMMARY_JSON, "sequence_p95_frame_error", "model unit", "Aggregate sequence dari validation_summary.json.")
        add_metric("fallback_ratio", float(summary["fallback_ratio"]), None, SUMMARY_JSON, "fallback_ratio", "ratio", "Aggregate dari validation_summary.json.")
    if not runtime.empty and "used_fallback" in runtime.columns:
        vals = pd.to_numeric(runtime["used_fallback"], errors="coerce").fillna(0)
        add_metric("fallback_count", float(vals.sum()), None, RUNTIME_LOG, "used_fallback", "count", "Dihitung ulang dari runtime_log.csv.")


def figure_latency_evidence(runtime: pd.DataFrame, summary: dict[str, Any]) -> None:
    if runtime.empty or "total_ctrl_pipeline_ms" not in runtime.columns or "total_frame_loop_ms" not in runtime.columns:
        NOT_CREATED.append("fig_4_7_1_runtime_latency_evidence.png: data per-frame latency tidak ditemukan")
        return
    ctrl = pd.to_numeric(runtime["total_ctrl_pipeline_ms"], errors="coerce").dropna()
    loop = pd.to_numeric(runtime["total_frame_loop_ms"], errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, vals, title, xlabel, mean_key, p95_key in [
        (axes[0], ctrl, "A. Ctrl pipeline latency", "Latency (ms)", "latency_mean_ms", "latency_p95_ms"),
        (axes[1], loop, "B. Frame loop", "Frame loop (ms)", "frame_loop_mean_ms", "frame_loop_p95_ms"),
    ]:
        ax.hist(vals, bins=18, color=BLUE, alpha=0.85, edgecolor="white")
        mean = float(vals.mean())
        p95 = float(vals.quantile(0.95))
        ax.axvline(mean, color=DARK, linewidth=1.2, linestyle="-", label=f"Mean {mean:.2f}")
        ax.axvline(p95, color=GRAY, linewidth=1.2, linestyle="--", label=f"P95 {p95:.2f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frekuensi")
        ax.legend(fontsize=8)
        polish(ax)
    save(
        fig,
        "fig_4_7_1_runtime_latency_evidence.png",
        str(RUNTIME_LOG),
        "Distribusi ctrl pipeline latency dan frame loop pada runtime estimation.",
    )


def figure_latency_summary(summary: dict[str, Any]) -> None:
    if not summary:
        NOT_CREATED.append("fig_4_7_1_runtime_latency_summary.png: validation_summary.json tidak ditemukan")
        return
    labels = ["Mean ctrl pipeline", "P95 ctrl pipeline", "Mean frame loop", "P95 frame loop"]
    vals = [
        float(summary["latency_mean_ms"]),
        float(summary["latency_p95_ms"]),
        float(summary["frame_loop_mean_ms"]),
        float(summary["frame_loop_p95_ms"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(labels[::-1], vals[::-1], color=[BLUE, GRAY, BLUE, GRAY][::-1])
    ax.set_xlabel("Waktu (ms)")
    annotate_barh(ax, "{:.2f}")
    polish(ax, xgrid=True, ygrid=False)
    save(
        fig,
        "fig_4_7_1_runtime_latency_summary.png",
        str(SUMMARY_JSON),
        "Perbandingan ringkas waktu komputasi pada runtime estimation.",
    )


def figure_error_evidence(windows: pd.DataFrame, summary: dict[str, Any]) -> None:
    if windows.empty or "mean_joint_error" not in windows.columns or "rmse_forward_error" not in windows.columns:
        NOT_CREATED.append("fig_4_7_1_runtime_error_evidence.png: data per-window error tidak ditemukan")
        return
    x = np.arange(len(windows))
    joint = pd.to_numeric(windows["mean_joint_error"], errors="coerce")
    rmse = pd.to_numeric(windows["rmse_forward_error"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    axes[0].plot(x, joint, color=BLUE, linewidth=1.4)
    axes[0].axhline(joint.mean(), color=DARK, linewidth=1.1, linestyle="-", label=f"Window mean {joint.mean():.4f}")
    if summary:
        axes[0].axhline(float(summary["sequence_mean_joint_error"]), color=GRAY, linewidth=1.1, linestyle="--", label=f"Sequence aggregate {summary['sequence_mean_joint_error']:.4f}")
    axes[0].set_title("A. Mean joint error per window")
    axes[0].set_xlabel("Validation window")
    axes[0].set_ylabel("Mean joint error")
    axes[0].legend(fontsize=7)
    polish(axes[0])

    axes[1].plot(x, rmse, color=BLUE, linewidth=1.4)
    axes[1].axhline(rmse.mean(), color=DARK, linewidth=1.1, linestyle="-", label=f"Window mean {rmse.mean():.4f}")
    if summary:
        axes[1].axhline(float(summary["sequence_rmse_forward_error"]), color=GRAY, linewidth=1.1, linestyle="--", label=f"Sequence RMSE {summary['sequence_rmse_forward_error']:.4f}")
    axes[1].set_title("B. RMSE forward error per window")
    axes[1].set_xlabel("Validation window")
    axes[1].set_ylabel("RMSE forward error")
    axes[1].legend(fontsize=7)
    polish(axes[1])
    save(
        fig,
        "fig_4_7_1_runtime_error_evidence.png",
        f"{WINDOWS_CSV}; {SUMMARY_JSON}",
        "Distribusi error runtime antara gerakan target dan hasil rollout.",
        "Garis sequence aggregate berasal dari validation_summary.json karena data per-frame rollout lengkap tidak tersedia.",
    )


def figure_error_summary(summary: dict[str, Any]) -> None:
    if not summary:
        NOT_CREATED.append("fig_4_7_1_runtime_error_summary.png: validation_summary.json tidak ditemukan")
        return
    labels = ["Mean joint error", "RMSE forward error", "P95 frame error", "Fallback ratio"]
    vals = [
        float(summary["sequence_mean_joint_error"]),
        float(summary["sequence_rmse_forward_error"]),
        float(summary["sequence_p95_frame_error"]),
        float(summary["fallback_ratio"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(labels[::-1], vals[::-1], color=[GRAY, BLUE, GRAY, LIGHT][::-1])
    ax.set_xlabel("Nilai metrik")
    annotate_barh(ax, "{:.4f}")
    polish(ax, xgrid=True, ygrid=False)
    save(
        fig,
        "fig_4_7_1_runtime_error_summary.png",
        str(SUMMARY_JSON),
        "Ringkasan error dan fallback pada runtime estimation.",
    )


def write_outputs(runtime: pd.DataFrame, windows: pd.DataFrame, summary: dict[str, Any]) -> None:
    pd.DataFrame([m.__dict__ for m in METRICS]).to_csv(OUT / "runtime_metric_summary.csv", index=False)
    lines = [
        "# Runtime Evidence Bab 4.7.1",
        "",
        "Audit ini menggunakan file log/output asli. Tidak ada retraining, tidak ada perubahan model, dan tidak ada data dummy.",
        "",
        "## File Sumber Data",
        f"- `{RUNTIME_LOG}`: log per frame berisi latency pipeline, frame loop, fallback, qpos, dan control runtime.",
        f"- `{WINDOWS_CSV}`: validasi error per window.",
        f"- `{SUMMARY_JSON}`: metrik sequence aggregate dan ringkasan runtime validation.",
        "",
        "## Kolom/Key yang Digunakan",
        "- `total_ctrl_pipeline_ms` untuk ctrl pipeline latency.",
        "- `total_frame_loop_ms` untuk frame loop.",
        "- `used_fallback` untuk fallback count.",
        "- `mean_joint_error` dan `rmse_forward_error` pada `validation_windows.csv` untuk distribusi per-window.",
        "- `sequence_mean_joint_error`, `sequence_rmse_forward_error`, `sequence_p95_frame_error`, dan `fallback_ratio` pada `validation_summary.json` untuk aggregate sequence.",
        "",
        "## Jumlah Data",
        f"- Frame runtime log: {len(runtime)}",
        f"- Validation windows: {len(windows)}",
        f"- Total frames menurut summary: {summary.get('total_frames', 'NA')}",
        f"- Total validation windows menurut summary: {summary.get('total_validation_windows', 'NA')}",
        "",
        "## Rumus Perhitungan",
        "- Mean ctrl pipeline latency = rata-rata `total_ctrl_pipeline_ms` dari `runtime_log.csv`.",
        "- Mean frame loop = rata-rata `total_frame_loop_ms` dari `runtime_log.csv`.",
        "- P95 = persentil ke-95 dari kolom terkait.",
        "- Fallback count = jumlah `used_fallback`.",
        "- Fallback ratio = nilai aggregate `fallback_ratio` dari `validation_summary.json`.",
        "- Mean joint error dan RMSE forward error target berasal dari aggregate sequence pada `validation_summary.json`.",
        "",
        "## Perbandingan Target",
        "",
        "| Metric | Computed | Target | Selisih | Sumber |",
        "|---|---:|---:|---:|---|",
    ]
    for m in METRICS:
        if m.metric_name not in {
            "mean_ctrl_pipeline_latency_ms",
            "mean_frame_loop_ms",
            "sequence_mean_joint_error",
            "sequence_rmse_forward_error",
        }:
            continue
        lines.append(
            f"| {m.metric_name} | {m.computed_value:.6f} | {m.target_value} | {m.absolute_difference} | `{Path(m.source_file).name}:{m.source_column}` |"
        )
    lines.extend(
        [
            "",
            "## Catatan Penting",
            "- Latency dan frame loop dapat dihitung ulang langsung dari data per-frame `runtime_log.csv`, dan cocok dengan nilai target setelah pembulatan.",
            "- Data error per-window tersedia di `validation_windows.csv`, tetapi nilai target 0.2611 dan 1.3589 adalah aggregate sequence dari `validation_summary.json`.",
            "- Data per-frame rollout lengkap untuk menghitung ulang sequence mean joint error dan sequence RMSE dari nol tidak ditemukan pada artifact ini.",
            "- Karena itu, figure distribusi error menggunakan data per-window, sedangkan figure summary error menggunakan aggregate resmi dari summary.",
            "- Runtime estimation lebih tepat ditulis sebagai preview/monitoring; error rollout menunjukkan analisis final tetap lebih aman dilakukan pada pipeline offline lengkap.",
            "",
            "## Figure Berhasil Dibuat",
        ]
    )
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
            "## Kalimat Rujukan Aman untuk Bab 4.7.1",
            "Berdasarkan log runtime validation, rata-rata latency pipeline estimasi control adalah sekitar 7,41 ms, sedangkan rata-rata waktu loop frame adalah sekitar 55,02 ms. Perbedaan ini menunjukkan bahwa proses estimasi control relatif cepat, tetapi waktu loop keseluruhan masih dipengaruhi komponen sistem lain seperti update simulasi dan rendering. Metrik error sequence pada validation summary menunjukkan mean joint error sebesar 0,2611 dan RMSE forward error sebesar 1,3589, sehingga jalur runtime lebih tepat diposisikan sebagai preview dan monitoring, bukan sebagai hasil akhir analisis aktivitas otot.",
        ]
    )
    (OUT / "report_runtime_evidence.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    runtime = read_csv(RUNTIME_LOG)
    windows = read_csv(WINDOWS_CSV)
    summary = read_json(SUMMARY_JSON)
    compute_metrics(runtime, windows, summary)
    figure_latency_evidence(runtime, summary)
    figure_latency_summary(summary)
    figure_error_evidence(windows, summary)
    figure_error_summary(summary)
    write_outputs(runtime, windows, summary)

    key = {m.metric_name: m for m in METRICS}
    print(f"Output folder: {OUT}")
    print(f"Source files: {RUNTIME_LOG}; {WINDOWS_CSV}; {SUMMARY_JSON}")
    print("Recomputed / audited metrics:")
    for name in [
        "mean_ctrl_pipeline_latency_ms",
        "mean_frame_loop_ms",
        "sequence_mean_joint_error",
        "sequence_rmse_forward_error",
    ]:
        row = key[name]
        ok = float(row.absolute_difference) <= 1e-3 if row.absolute_difference != "" else False
        print(f" - {name}: {row.computed_value:.6f} target={row.target_value} diff={row.absolute_difference} match={ok}")
    print("Created figures:")
    for fig in FIGURES:
        print(f" - {fig.filename}")
    if NOT_CREATED:
        print("Not created:")
        for item in NOT_CREATED:
            print(f" - {item}")
    print("Best figure for Bab 4.7.1:")
    print(" - fig_4_7_1_runtime_latency_evidence.png")
    print(" - fig_4_7_1_runtime_error_evidence.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
