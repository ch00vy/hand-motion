from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gcn_node_classifier import build_adjacency_matrix, normalized_adjacency  # noqa: E402
from app.graph_sequence_baseline import load_trial_graph_sequence  # noqa: E402
from scripts.offline_biomech.train_temporal_gcn_prototype import (  # noqa: E402
    BINARY_CLASSES,
    TemporalGCNNodeClassifier,
    TemporalGraphSample,
    _append_progress_feature,
    _build_temporal_samples,
    _dynamic_adjacency_seq,
    _standardize_temporal_samples,
    _type_adjacency_seq,
    normalized_dynamic_adjacency,
)


DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier_labeled_records"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "dynamic_temporal_gcn_node_classifier"
DEFAULT_CURRENT_TRIAL_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "current_trial_embedding"
DEFAULT_PCA_BASIS_PATH = DEFAULT_OUTPUT_DIR / "node_embedding_visualization_pca_basis.npz"

DISPLAY_LABELS = {
    "within_reference": "Sesuai Pola Acuan",
    "deviating": "Pola Aktivitas Berbeda",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _pca_2d(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Small PCA implementation so the exporter works even without scikit-learn."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), {"explained_variance_ratio_1": 0.0, "explained_variance_ratio_2": 0.0}
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:2].T if vt.shape[0] >= 2 else np.pad(vt[:1].T, ((0, 0), (0, 1)))
    coords = x_centered @ components
    variances = (singular_values**2) / max(1, x.shape[0] - 1)
    total = float(np.sum(variances)) if variances.size else 0.0
    ratio = variances[:2] / total if total > 0.0 else np.zeros((2,), dtype=float)
    if ratio.size < 2:
        ratio = np.pad(ratio, (0, 2 - ratio.size))
    return coords.astype(float), {
        "explained_variance_ratio_1": float(ratio[0]),
        "explained_variance_ratio_2": float(ratio[1]),
    }


def _fit_pca_basis(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Fit a 2D PCA basis and return coords, mean, components and metadata."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        mean = np.zeros((x.shape[1] if x.ndim == 2 else 0,), dtype=float)
        components = np.zeros((mean.shape[0], 2), dtype=float)
        coords = np.zeros((0, 2), dtype=float)
        ratio = np.zeros((2,), dtype=float)
    else:
        mean = np.mean(x, axis=0)
        x_centered = x - mean.reshape(1, -1)
        _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
        components = vt[:2].T if vt.shape[0] >= 2 else np.pad(vt[:1].T, ((0, 0), (0, 1)))
        coords = x_centered @ components
        variances = (singular_values**2) / max(1, x.shape[0] - 1)
        total = float(np.sum(variances)) if variances.size else 0.0
        ratio = variances[:2] / total if total > 0.0 else np.zeros((2,), dtype=float)
        if ratio.size < 2:
            ratio = np.pad(ratio, (0, 2 - ratio.size))
    info = {
        "explained_variance_ratio_1": float(ratio[0]),
        "explained_variance_ratio_2": float(ratio[1]),
    }
    return coords.astype(float), mean.astype(float), components.astype(float), info


def _transform_with_pca_basis(values: np.ndarray, basis_path: Path) -> tuple[np.ndarray, dict[str, float], str]:
    data = np.load(basis_path)
    mean = np.asarray(data["mean"], dtype=np.float64)
    components = np.asarray(data["components"], dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != mean.shape[0]:
        raise ValueError(f"PCA basis dimension {mean.shape[0]} does not match embedding shape {x.shape}.")
    coords = (x - mean.reshape(1, -1)) @ components
    info = {
        "explained_variance_ratio_1": float(np.asarray(data["explained_variance_ratio"])[0]),
        "explained_variance_ratio_2": float(np.asarray(data["explained_variance_ratio"])[1]),
    }
    return coords.astype(float), info, "dataset_pca_basis"


def _sample_to_tensors(sample, device: torch.device):
    return (
        torch.tensor(sample.x_seq, dtype=torch.float32, device=device),
        torch.tensor(sample.adjacency, dtype=torch.float32, device=device),
        torch.tensor(sample.adjacency_seq, dtype=torch.float32, device=device),
        torch.tensor(sample.type_adjacency_seq, dtype=torch.float32, device=device),
        torch.tensor(sample.y, dtype=torch.long, device=device),
    )


def _forward_with_node_embeddings(
    model: TemporalGCNNodeClassifier,
    x_seq: torch.Tensor,
    adjacency: torch.Tensor,
    adjacency_seq: torch.Tensor,
    type_adjacency_seq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits and the node representation before the classifier head.

    This is intentionally external to the model class: training/evaluation
    behavior remains unchanged and no checkpoint fields are modified.
    """

    if model.variant == "dynamic":
        graph_input = normalized_dynamic_adjacency(adjacency_seq)
    elif model.variant == "edge_type":
        graph_input = type_adjacency_seq
    else:
        graph_input = normalized_adjacency(adjacency)
    h = torch.relu(model.gcn1(x_seq, graph_input))
    h = model.dropout(h)
    h = torch.relu(model.gcn2(h, graph_input))
    h = model.dropout(h)
    h_node_time = h.permute(1, 0, 2).contiguous()
    temporal_out, _ = model.temporal(h_node_time)
    if model.attention is not None:
        attention_logits = model.attention(temporal_out).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        node_repr = torch.sum(temporal_out * attention_weights.unsqueeze(-1), dim=1)
    else:
        node_repr = temporal_out[:, -1, :]
    node_repr = model.dropout(node_repr)
    logits = model.head(node_repr)
    return logits, node_repr


def _build_test_samples(model_dir: Path, checkpoint: dict[str, Any]):
    dataset_summary = _read_json(model_dir / "dataset_summary.json")
    run_config = dict(checkpoint.get("run_config") or _read_json(model_dir / "run_config.json"))
    split_manifest_path = model_dir / "split_manifest.csv"
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest_path}")
    threshold = float(dataset_summary.get("binary_threshold", run_config.get("binary_threshold", 0.0)))
    split_samples, _ = _build_temporal_samples(
        dataset_dir=Path(dataset_summary["dataset_dir"]).resolve(),
        reference_dir=Path(dataset_summary["reference_dir"]).resolve(),
        split_manifest_path=split_manifest_path,
        binary_threshold=threshold,
        onset_ratio=0.20,
        time_region_width=3,
    )
    test_samples = list(split_samples.get("test") or [])
    if bool(run_config.get("add_progress_feature", dataset_summary.get("add_progress_feature", False))):
        test_samples = _append_progress_feature(test_samples)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return _standardize_temporal_samples(test_samples, feature_mean, feature_std), dataset_summary, run_config


def _load_model(model_dir: Path, device: torch.device) -> tuple[TemporalGCNNodeClassifier, dict[str, Any]]:
    checkpoint_path = model_dir / "checkpoints" / "best_temporal_gcn.pt"
    return _load_model_from_checkpoint_path(checkpoint_path, device)


def _load_model_from_checkpoint_path(checkpoint_path: Path, device: torch.device) -> tuple[TemporalGCNNodeClassifier, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    # Local project checkpoint includes numpy arrays for feature_mean/std.
    # weights_only=False is required by PyTorch >=2.6 for this trusted artifact.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_config = dict(checkpoint.get("run_config") or _read_json(model_dir / "run_config.json"))
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    model = TemporalGCNNodeClassifier(
        in_dim=int(feature_mean.shape[0]),
        hidden_dim=int(run_config.get("hidden_dim", 32)),
        temporal_dim=int(run_config.get("temporal_dim", 32)),
        num_classes=len(BINARY_CLASSES),
        dropout=float(run_config.get("dropout", 0.0)),
        variant=str(run_config.get("model_variant", "dynamic")),
        temporal_pooling=str(run_config.get("temporal_pooling", "last")),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _current_trial_sample_from_graph(trial_graph: dict[str, Any], checkpoint: dict[str, Any]) -> TemporalGraphSample:
    run_config = dict(checkpoint.get("run_config") or {})
    x_seq = np.asarray(trial_graph["node_feature_seq"], dtype=np.float32)
    sample = TemporalGraphSample(
        trial_id=str((trial_graph.get("metadata") or {}).get("trial_id") or "current_trial"),
        cohort=str((trial_graph.get("metadata") or {}).get("cohort") or "current_gui_trial"),
        exercise_name=str((trial_graph.get("metadata") or {}).get("exercise_name") or "open_to_fist"),
        node_names=list(trial_graph["node_names"]),
        feature_names=list(trial_graph.get("node_feature_names") or []),
        x_seq=x_seq,
        adjacency=build_adjacency_matrix(trial_graph),
        adjacency_seq=_dynamic_adjacency_seq(trial_graph),
        type_adjacency_seq=_type_adjacency_seq(trial_graph),
        y=np.zeros((len(trial_graph["node_names"]),), dtype=np.int64),
        split="current_trial",
    )
    samples = [sample]
    if bool(run_config.get("add_progress_feature", False)):
        samples = _append_progress_feature(samples)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return _standardize_temporal_samples(samples, feature_mean, feature_std)[0]


def _analysis_nodes_by_name(analysis_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("node_name") or ""): dict(row) for row in list(analysis_result.get("nodes") or [])}


def _collect_current_trial_embedding_rows(
    *,
    graph_json: Path,
    analysis_result: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    model, checkpoint = _load_model_from_checkpoint_path(checkpoint_path, device)
    trial_graph = load_trial_graph_sequence(Path(graph_json))
    sample = _current_trial_sample_from_graph(trial_graph, checkpoint)
    analysis_by_name = _analysis_nodes_by_name(analysis_result)
    with torch.no_grad():
        x_seq, adjacency, adjacency_seq, type_adjacency_seq, _ = _sample_to_tensors(sample, device)
        logits, node_repr = _forward_with_node_embeddings(model, x_seq, adjacency, adjacency_seq, type_adjacency_seq)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        embeddings = node_repr.detach().cpu().numpy()
    class_names = [str(item) for item in checkpoint.get("class_names") or list(BINARY_CLASSES)]
    deviating_idx = class_names.index("deviating") if "deviating" in class_names else min(1, len(class_names) - 1)
    trial_id = str(analysis_result.get("trial_id") or sample.trial_id)
    rows: list[dict[str, Any]] = []
    for node_idx, muscle_name in enumerate(sample.node_names):
        from_result = analysis_by_name.get(str(muscle_name), {})
        pred_label = str(from_result.get("predicted_label") or class_names[int(pred[node_idx].item())])
        prob = float(from_result.get("probability_deviating", probs[node_idx, deviating_idx].detach().cpu().item()))
        row = {
            "trial_id": trial_id,
            "sample_id": f"{trial_id}__node_{node_idx}",
            "node_id": int(node_idx),
            "muscle_name": str(muscle_name),
            "predicted_label": pred_label,
            "predicted_status_display": DISPLAY_LABELS.get(pred_label, pred_label),
            "prob_different_pattern": prob,
            "true_label": str(from_result.get("true_label") or ""),
            "true_status_display": DISPLAY_LABELS.get(str(from_result.get("true_label") or ""), ""),
        }
        for dim_idx, value in enumerate(embeddings[node_idx].tolist()):
            row[f"embedding_{dim_idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _collect_embedding_rows(model_dir: Path, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any]]:
    model, checkpoint = _load_model(model_dir, device)
    test_samples, dataset_summary, run_config = _build_test_samples(model_dir, checkpoint)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in test_samples:
            x_seq, adjacency, adjacency_seq, type_adjacency_seq, y = _sample_to_tensors(sample, device)
            logits, node_repr = _forward_with_node_embeddings(model, x_seq, adjacency, adjacency_seq, type_adjacency_seq)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            embeddings = node_repr.detach().cpu().numpy()
            for node_idx, muscle_name in enumerate(sample.node_names):
                pred_idx = int(pred[node_idx].item())
                true_idx = int(y[node_idx].item())
                row = {
                    "trial_id": str(sample.trial_id),
                    "sample_id": f"{sample.trial_id}__node_{node_idx}",
                    "node_id": int(node_idx),
                    "muscle_name": str(muscle_name),
                    "predicted_label": BINARY_CLASSES[pred_idx],
                    "predicted_status_display": DISPLAY_LABELS[BINARY_CLASSES[pred_idx]],
                    "prob_different_pattern": float(probs[node_idx, 1].detach().cpu()),
                    "true_label": BINARY_CLASSES[true_idx],
                    "true_status_display": DISPLAY_LABELS[BINARY_CLASSES[true_idx]],
                }
                for dim_idx, value in enumerate(embeddings[node_idx].tolist()):
                    row[f"embedding_{dim_idx}"] = float(value)
                rows.append(row)
    meta = {"dataset_summary": dataset_summary, "run_config": run_config, "num_test_samples": len(test_samples)}
    return pd.DataFrame(rows), meta


def _aggregate_by_muscle(rows: pd.DataFrame) -> pd.DataFrame:
    embedding_cols = [col for col in rows.columns if col.startswith("embedding_")]
    grouped = rows.groupby(["node_id", "muscle_name"], as_index=False)
    embedding_mean = grouped[embedding_cols].mean()
    prob_mean = grouped["prob_different_pattern"].mean()
    pred_label = np.where(prob_mean["prob_different_pattern"].to_numpy(dtype=float) >= 0.5, "deviating", "within_reference")
    true_label = (
        rows.groupby(["node_id", "muscle_name"])["true_label"]
        .agg(lambda values: values.value_counts().idxmax())
        .reset_index(drop=True)
    )
    out = embedding_mean.copy()
    out["trial_id"] = "test_holdout_mean"
    out["sample_id"] = out["muscle_name"].map(lambda name: f"test_holdout_mean__{name}")
    out["predicted_label"] = pred_label
    out["predicted_status_display"] = [DISPLAY_LABELS[label] for label in pred_label]
    out["prob_different_pattern"] = prob_mean["prob_different_pattern"].to_numpy(dtype=float)
    out["true_label"] = true_label.to_numpy()
    out["true_status_display"] = out["true_label"].map(DISPLAY_LABELS)
    return out


def _plot_embedding(
    df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
    *,
    pca_info: dict[str, float],
    title: str = "Visualisasi Embedding Node Otot Hasil Dynamic Temporal GCN",
    subtitle: str = "PCA 2D projection of node embeddings on test/holdout data",
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "within_reference": "#22C55E",
        "deviating": "#EF4444",
    }
    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#E5E7EB", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    for label, part in df.groupby("predicted_label"):
        ax.scatter(
            part["embedding_x"],
            part["embedding_y"],
            s=92,
            color=colors.get(str(label), "#64748B"),
            edgecolor="#111827",
            linewidth=0.7,
            alpha=0.92,
            label=DISPLAY_LABELS.get(str(label), str(label)),
        )

    for idx, row in enumerate(df.to_dict(orient="records")):
        dx = 0.018 if idx % 2 == 0 else -0.018
        dy = 0.018 if idx % 3 else -0.018
        ax.annotate(
            str(row["muscle_name"]),
            (float(row["embedding_x"]), float(row["embedding_y"])),
            xytext=(dx, dy),
            textcoords="offset fontsize",
            fontsize=8.0,
            color="#1F2937",
            ha="left" if dx > 0 else "right",
            va="center",
            arrowprops=dict(arrowstyle="-", color="#CBD5E1", lw=0.45, alpha=0.75),
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    ax.text(
        0.5,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#64748B",
    )
    ax.set_xlabel(f"Node Embedding Dimension 1 (PCA {pca_info['explained_variance_ratio_1'] * 100:.1f}%)")
    ax.set_ylabel(f"Node Embedding Dimension 2 (PCA {pca_info['explained_variance_ratio_2'] * 100:.1f}%)")
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#E5E7EB")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.18, dpi=350)
    fig.savefig(output_svg, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _write_notes(path: Path, *, source_model_dir: Path, csv_path: Path, png_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Catatan Visualisasi Embedding Node Otot",
                "",
                "Embedding node adalah representasi fitur setiap node otot setelah diproses oleh Dynamic Temporal GCN.",
                "PCA digunakan hanya untuk memproyeksikan embedding berdimensi tinggi menjadi ruang 2D agar dapat divisualisasikan.",
                "Jarak antar titik menunjukkan kedekatan representasi embedding dalam ruang fitur model, bukan jarak anatomi antarotot.",
                "Warna titik menunjukkan status klasifikasi pola aktivitas otot: Sesuai Pola Acuan atau Pola Aktivitas Berbeda.",
                "Visualisasi ini merupakan interpretasi berbasis simulasi musculoskeletal dan bukan diagnosis klinis.",
                "",
                f"Sumber model: `{source_model_dir}`",
                f"CSV koordinat: `{csv_path}`",
                f"Gambar PNG: `{png_path}`",
            ]
        ),
        encoding="utf-8",
    )


def _write_current_trial_notes(path: Path, *, graph_json: Path, csv_path: Path, png_path: Path, pca_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_note = (
        "Koordinat menggunakan PCA basis dataset-level sehingga lebih konsisten dengan visualisasi test/holdout."
        if pca_mode == "dataset_pca_basis"
        else "Koordinat menggunakan PCA lokal pada 39 node trial aktif, sehingga sumbu PCA hanya berlaku untuk trial ini."
    )
    path.write_text(
        "\n".join(
            [
                "# Catatan Visualisasi Embedding Node Otot Trial Aktif",
                "",
                "Plot current trial berasal dari trial aktif yang sudah dianalisis di GUI.",
                "Setiap titik merepresentasikan satu node otot pada trial tersebut.",
                "Warna titik mengikuti hasil klasifikasi model pada trial aktif: Sesuai Pola Acuan atau Pola Aktivitas Berbeda.",
                "Embedding node adalah representasi fitur node setelah diproses Dynamic Temporal GCN sebelum classifier akhir.",
                "PCA digunakan hanya untuk visualisasi 2D.",
                "Jarak antar titik menunjukkan kedekatan representasi embedding model, bukan jarak anatomi otot.",
                "Hasil ini merupakan interpretasi berbasis simulasi musculoskeletal, bukan diagnosis klinis.",
                comparison_note,
                "",
                f"Graph trial: `{graph_json}`",
                f"CSV koordinat: `{csv_path}`",
                f"Gambar PNG: `{png_path}`",
            ]
        ),
        encoding="utf-8",
    )


def export_current_trial_embedding_visualization(
    *,
    graph_json: Path,
    analysis_result: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path = DEFAULT_CURRENT_TRIAL_OUTPUT_DIR,
    pca_basis_path: Path = DEFAULT_PCA_BASIS_PATH,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Export node embedding visualization for the current GUI trial.

    This reads the already-built graph sequence and the GUI analysis result.
    It does not retrain and does not modify the checkpoint.
    """

    if device_name == "cuda":
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_json = Path(graph_json).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_current_trial_embedding_rows(
        graph_json=graph_json,
        analysis_result=analysis_result,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    if rows.empty:
        raise RuntimeError("No current-trial embedding rows were produced.")
    embedding_cols = [col for col in rows.columns if col.startswith("embedding_")]
    embedding_values = rows[embedding_cols].to_numpy(dtype=float)
    pca_mode = "local_current_trial_pca"
    try:
        if Path(pca_basis_path).exists():
            coords, pca_info, pca_mode = _transform_with_pca_basis(embedding_values, Path(pca_basis_path))
        else:
            coords, pca_info = _pca_2d(embedding_values)
    except Exception:
        coords, pca_info = _pca_2d(embedding_values)
        pca_mode = "local_current_trial_pca"
    rows["embedding_x"] = coords[:, 0]
    rows["embedding_y"] = coords[:, 1]

    output_csv = output_dir / "current_trial_node_embedding_pca.csv"
    output_png = output_dir / "current_trial_node_embedding_pca.png"
    output_svg = output_dir / "current_trial_node_embedding_pca.svg"
    output_notes = output_dir / "current_trial_embedding_notes.md"
    output_summary = output_dir / "current_trial_node_embedding_summary.json"
    export_cols = [
        "trial_id",
        "sample_id",
        "node_id",
        "muscle_name",
        "embedding_x",
        "embedding_y",
        "predicted_label",
        "predicted_status_display",
        "prob_different_pattern",
        "true_label",
        "true_status_display",
    ]
    rows[export_cols].to_csv(output_csv, index=False)
    _plot_embedding(
        rows,
        output_png,
        output_svg,
        pca_info=pca_info,
        title="Visualisasi Embedding Node Otot pada Trial Aktif",
        subtitle="PCA 2D projection of node embeddings from current GUI analysis",
    )
    _write_current_trial_notes(output_notes, graph_json=graph_json, csv_path=output_csv, png_path=output_png, pca_mode=pca_mode)
    summary = {
        "trial_id": str(analysis_result.get("trial_id") or ""),
        "graph_json": str(graph_json),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "num_nodes": int(len(rows)),
        "pca_mode": pca_mode,
        "pca": pca_info,
        "png": str(output_png),
        "svg": str(output_svg),
        "csv": str(output_csv),
        "notes": str(output_notes),
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(output_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Dynamic Temporal GCN node embedding visualization without retraining.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--graph-json", type=Path, default=None, help="Current-trial graph JSON. Enables current-trial export mode.")
    parser.add_argument("--analysis-result-json", type=Path, default=None, help="GUI result.json for the current trial.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Checkpoint path for current-trial export mode.")
    parser.add_argument("--current-output-dir", type=Path, default=DEFAULT_CURRENT_TRIAL_OUTPUT_DIR)
    parser.add_argument("--pca-basis-path", type=Path, default=DEFAULT_PCA_BASIS_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.graph_json is not None:
        if args.analysis_result_json is None:
            raise ValueError("--analysis-result-json is required when --graph-json is used.")
        analysis_result = _read_json(Path(args.analysis_result_json).resolve())
        checkpoint_path = Path(args.checkpoint_path or analysis_result.get("model_path") or (model_dir / "checkpoints" / "best_temporal_gcn.pt"))
        summary = export_current_trial_embedding_visualization(
            graph_json=Path(args.graph_json),
            analysis_result=analysis_result,
            checkpoint_path=checkpoint_path,
            output_dir=Path(args.current_output_dir),
            pca_basis_path=Path(args.pca_basis_path),
            device_name=str(args.device),
        )
        print(json.dumps(summary, indent=2))
        return 0

    rows, meta = _collect_embedding_rows(model_dir, device)
    if rows.empty:
        raise RuntimeError("No embedding rows were produced.")
    aggregate = _aggregate_by_muscle(rows)
    embedding_cols = [col for col in aggregate.columns if col.startswith("embedding_")]
    coords, pca_mean, pca_components, pca_info = _fit_pca_basis(aggregate[embedding_cols].to_numpy(dtype=float))
    aggregate["embedding_x"] = coords[:, 0]
    aggregate["embedding_y"] = coords[:, 1]
    output_csv = output_dir / "node_embedding_visualization_pca.csv"
    output_png = output_dir / "node_embedding_visualization_pca.png"
    output_svg = output_dir / "node_embedding_visualization_pca.svg"
    output_notes = output_dir / "node_embedding_visualization_notes.md"
    output_summary = output_dir / "node_embedding_visualization_summary.json"
    output_pca_basis = output_dir / "node_embedding_visualization_pca_basis.npz"

    export_cols = [
        "trial_id",
        "sample_id",
        "node_id",
        "muscle_name",
        "embedding_x",
        "embedding_y",
        "predicted_label",
        "predicted_status_display",
        "prob_different_pattern",
        "true_label",
        "true_status_display",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate[export_cols].to_csv(output_csv, index=False)
    np.savez(
        output_pca_basis,
        mean=pca_mean,
        components=pca_components,
        explained_variance_ratio=np.asarray(
            [pca_info["explained_variance_ratio_1"], pca_info["explained_variance_ratio_2"]],
            dtype=float,
        ),
    )
    _plot_embedding(aggregate, output_png, output_svg, pca_info=pca_info)
    _write_notes(output_notes, source_model_dir=model_dir, csv_path=output_csv, png_path=output_png)
    output_summary.write_text(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "output_dir": str(output_dir),
                "num_source_test_node_rows": int(len(rows)),
                "num_plotted_muscle_nodes": int(len(aggregate)),
                "pca": pca_info,
                "pca_basis": str(output_pca_basis),
                **meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "png": str(output_png),
                "svg": str(output_svg),
                "csv": str(output_csv),
                "notes": str(output_notes),
                "summary": str(output_summary),
                "pca_basis": str(output_pca_basis),
                "num_nodes": int(len(aggregate)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
