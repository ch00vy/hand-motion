from __future__ import annotations

"""Train lightweight ctrl predictors against the offline teacher dataset."""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.fast_ctrl_predictor import FastCtrlGRU, FastCtrlMLP, Standardizer  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "fast_ctrl_predictor"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "fast_ctrl_predictor_training_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight realtime ctrl predictor from the offline teacher dataset.")
    parser.add_argument("--dataset-dir", type=str, default=str(PROJECT_ROOT / "artifacts" / "realtime_ctrl_teacher_dataset"), help="Directory containing train.csv / val.csv / test.csv.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for checkpoints and summaries.")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Markdown report path.")
    parser.add_argument("--model-type", type=str, default="mlp", choices=("mlp", "gru"), help="Predictor architecture.")
    parser.add_argument("--hidden-dims", type=str, default="256,256", help="Comma-separated hidden dims for the MLP baseline.")
    parser.add_argument("--gru-hidden-size", type=int, default=128, help="Hidden size for the GRU baseline.")
    parser.add_argument("--gru-num-layers", type=int, default=1, help="Number of GRU layers.")
    parser.add_argument("--history-length", type=int, default=-1, help="Override dataset history length. Use -1 to infer from CSV columns.")
    parser.add_argument("--include-qvel", action=argparse.BooleanOptionalAction, default=True, help="Use qvel features.")
    parser.add_argument("--include-prev-ctrl", action=argparse.BooleanOptionalAction, default=True, help="Use prev_ctrl and ctrl history features.")
    parser.add_argument("--include-gesture-label", action=argparse.BooleanOptionalAction, default=False, help="Append gesture one-hot features.")
    parser.add_argument("--loss-type", type=str, default="smoothl1", choices=("mse", "smoothl1"), help="Primary supervision loss.")
    parser.add_argument("--smoothness-weight", type=float, default=0.05, help="Penalty toward prev_ctrl.")
    parser.add_argument("--bound-penalty-weight", type=float, default=0.02, help="Penalty for predictions outside [0,1].")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--benchmark-iters", type=int, default=200, help="Number of val samples used for latency benchmark.")
    return parser.parse_args()


def ordered_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols: list[tuple[int, str]] = []
    for name in df.columns:
        label = str(name)
        if not label.startswith(prefix):
            continue
        suffix = label[len(prefix) :]
        try:
            idx = int(suffix)
        except ValueError:
            continue
        cols.append((idx, label))
    cols.sort(key=lambda item: item[0])
    return [label for _, label in cols]


def ordered_history_columns(df: pd.DataFrame, family: str, history_length: int, dim: int) -> list[list[str]]:
    groups: list[list[str]] = []
    for h in range(history_length, 0, -1):
        prefix = f"{family}_hist{h}_"
        cols = []
        for idx in range(dim):
            label = f"{prefix}{idx}"
            if label in df.columns:
                cols.append(label)
        groups.append(cols)
    return groups


def infer_history_length(df: pd.DataFrame) -> int:
    best = 0
    for name in df.columns:
        label = str(name)
        if "_hist" not in label:
            continue
        try:
            head = label.split("_hist", 1)[1]
            hist_idx = int(head.split("_", 1)[0])
        except Exception:
            continue
        best = max(best, hist_idx)
    return best


def gesture_vocab_from_frames(*dfs: pd.DataFrame) -> list[str]:
    vocab = sorted({str(item).strip().lower() for df in dfs for item in df["exercise_name"].fillna("").tolist() if str(item).strip()})
    return vocab


def build_gesture_one_hot(series: pd.Series, vocab: list[str]) -> np.ndarray:
    if not vocab:
        return np.zeros((len(series), 0), dtype=np.float32)
    out = np.zeros((len(series), len(vocab)), dtype=np.float32)
    index = {name: idx for idx, name in enumerate(vocab)}
    for row_idx, item in enumerate(series.fillna("").tolist()):
        label = str(item).strip().lower()
        if label in index:
            out[row_idx, index[label]] = 1.0
    return out


@dataclass
class PreparedDataset:
    inputs: np.ndarray
    targets: np.ndarray
    prev_ctrl: np.ndarray
    model_inputs: np.ndarray
    gesture_vocab: list[str]
    feature_spec: dict[str, Any]


def prepare_dataset(
    df: pd.DataFrame,
    *,
    history_length: int,
    include_qvel: bool,
    include_prev_ctrl: bool,
    include_gesture_label: bool,
    model_type: str,
    gesture_vocab: list[str],
) -> PreparedDataset:
    qpos_cols = ordered_columns(df, "qpos_")
    qvel_cols = ordered_columns(df, "qvel_")
    prev_ctrl_cols = ordered_columns(df, "prev_ctrl_")
    target_cols = ordered_columns(df, "ctrl_target_")
    if not qpos_cols or not target_cols:
        raise ValueError("Dataset must contain qpos_* and ctrl_target_* columns.")
    num_qpos = len(qpos_cols)
    num_ctrl = len(target_cols)
    qpos = df.loc[:, qpos_cols].to_numpy(dtype=np.float32)
    qvel = df.loc[:, qvel_cols].to_numpy(dtype=np.float32) if include_qvel and qvel_cols else np.zeros((len(df), num_qpos), dtype=np.float32)
    prev_ctrl = df.loc[:, prev_ctrl_cols].to_numpy(dtype=np.float32) if include_prev_ctrl and prev_ctrl_cols else np.zeros((len(df), num_ctrl), dtype=np.float32)
    targets = df.loc[:, target_cols].to_numpy(dtype=np.float32)
    gesture_feats = build_gesture_one_hot(df["exercise_name"], gesture_vocab) if include_gesture_label else np.zeros((len(df), 0), dtype=np.float32)

    if str(model_type).strip().lower() == "gru":
        steps: list[np.ndarray] = []
        qpos_hist_groups = ordered_history_columns(df, "qpos", history_length, num_qpos)
        qvel_hist_groups = ordered_history_columns(df, "qvel", history_length, num_qpos)
        ctrl_hist_groups = ordered_history_columns(df, "ctrl", history_length, num_ctrl)
        for step_idx in range(history_length):
            parts = []
            hist_qpos = df.loc[:, qpos_hist_groups[step_idx]].to_numpy(dtype=np.float32) if qpos_hist_groups[step_idx] else np.zeros((len(df), num_qpos), dtype=np.float32)
            parts.append(hist_qpos)
            if include_qvel:
                hist_qvel = df.loc[:, qvel_hist_groups[step_idx]].to_numpy(dtype=np.float32) if qvel_hist_groups[step_idx] else np.zeros((len(df), num_qpos), dtype=np.float32)
                parts.append(hist_qvel)
            if include_prev_ctrl:
                hist_ctrl = df.loc[:, ctrl_hist_groups[step_idx]].to_numpy(dtype=np.float32) if ctrl_hist_groups[step_idx] else np.zeros((len(df), num_ctrl), dtype=np.float32)
                parts.append(hist_ctrl)
            if include_gesture_label and gesture_feats.shape[1] > 0:
                parts.append(gesture_feats)
            steps.append(np.concatenate(parts, axis=1))
        current_parts = [qpos]
        if include_qvel:
            current_parts.append(qvel)
        if include_prev_ctrl:
            current_parts.append(prev_ctrl)
        if include_gesture_label and gesture_feats.shape[1] > 0:
            current_parts.append(gesture_feats)
        steps.append(np.concatenate(current_parts, axis=1))
        model_inputs = np.stack(steps, axis=1).astype(np.float32)
        flat_inputs = model_inputs.reshape(len(df), -1)
        feature_spec = {
            "num_qpos": int(num_qpos),
            "num_ctrl": int(num_ctrl),
            "history_length": int(history_length),
            "include_qvel": bool(include_qvel),
            "include_prev_ctrl": bool(include_prev_ctrl),
            "include_gesture": bool(include_gesture_label),
            "gesture_vocab": list(gesture_vocab),
            "sequence_feature_dim": int(model_inputs.shape[-1]),
            "model_type": "gru",
        }
        return PreparedDataset(inputs=flat_inputs, targets=targets, prev_ctrl=prev_ctrl, model_inputs=model_inputs, gesture_vocab=gesture_vocab, feature_spec=feature_spec)

    parts = [qpos]
    if include_qvel:
        parts.append(qvel)
    if include_prev_ctrl:
        parts.append(prev_ctrl)
    for h in range(1, history_length + 1):
        hist_qpos_cols = [f"qpos_hist{h}_{idx}" for idx in range(num_qpos) if f"qpos_hist{h}_{idx}" in df.columns]
        parts.append(df.loc[:, hist_qpos_cols].to_numpy(dtype=np.float32) if hist_qpos_cols else np.zeros((len(df), num_qpos), dtype=np.float32))
        if include_qvel:
            hist_qvel_cols = [f"qvel_hist{h}_{idx}" for idx in range(num_qpos) if f"qvel_hist{h}_{idx}" in df.columns]
            parts.append(df.loc[:, hist_qvel_cols].to_numpy(dtype=np.float32) if hist_qvel_cols else np.zeros((len(df), num_qpos), dtype=np.float32))
        if include_prev_ctrl:
            hist_ctrl_cols = [f"ctrl_hist{h}_{idx}" for idx in range(num_ctrl) if f"ctrl_hist{h}_{idx}" in df.columns]
            parts.append(df.loc[:, hist_ctrl_cols].to_numpy(dtype=np.float32) if hist_ctrl_cols else np.zeros((len(df), num_ctrl), dtype=np.float32))
    if include_gesture_label and gesture_feats.shape[1] > 0:
        parts.append(gesture_feats)
    model_inputs = np.concatenate(parts, axis=1).astype(np.float32)
    feature_spec = {
        "num_qpos": int(num_qpos),
        "num_ctrl": int(num_ctrl),
        "history_length": int(history_length),
        "include_qvel": bool(include_qvel),
        "include_prev_ctrl": bool(include_prev_ctrl),
        "include_gesture": bool(include_gesture_label),
        "gesture_vocab": list(gesture_vocab),
        "model_type": "mlp",
    }
    return PreparedDataset(inputs=model_inputs, targets=targets, prev_ctrl=prev_ctrl, model_inputs=model_inputs, gesture_vocab=gesture_vocab, feature_spec=feature_spec)


class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, prev_ctrl: np.ndarray):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.prev_ctrl = torch.from_numpy(np.asarray(prev_ctrl, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.prev_ctrl[idx]


def build_model(args: argparse.Namespace, input_dim: int, output_dim: int) -> nn.Module:
    if str(args.model_type).strip().lower() == "gru":
        return FastCtrlGRU(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            hidden_size=int(args.gru_hidden_size),
            num_layers=int(args.gru_num_layers),
        )
    hidden_dims = [int(item) for item in str(args.hidden_dims).split(",") if str(item).strip()]
    return FastCtrlMLP(input_dim=int(input_dim), output_dim=int(output_dim), hidden_dims=hidden_dims or [256, 256])


def boundedness_penalty(pred: torch.Tensor) -> torch.Tensor:
    lower = torch.relu(-pred)
    upper = torch.relu(pred - 1.0)
    return torch.mean(lower ** 2 + upper ** 2)


def compute_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev_ctrl: torch.Tensor,
    *,
    primary_loss: str,
    smoothness_weight: float,
    bound_penalty_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if primary_loss == "mse":
        base = torch.mean((pred - target) ** 2)
    else:
        base = nn.functional.smooth_l1_loss(pred, target)
    smoothness = torch.mean((pred - prev_ctrl) ** 2)
    bound_pen = boundedness_penalty(pred)
    total = base + float(smoothness_weight) * smoothness + float(bound_penalty_weight) * bound_pen
    return total, {
        "base_loss": float(base.detach().cpu().item()),
        "smoothness_loss": float(smoothness.detach().cpu().item()),
        "bound_penalty": float(bound_pen.detach().cpu().item()),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    primary_loss: str,
    smoothness_weight: float,
    bound_penalty_weight: float,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    mae_items: list[float] = []
    with torch.no_grad():
        for x, y, prev_ctrl in loader:
            x = x.to(device)
            y = y.to(device)
            prev_ctrl = prev_ctrl.to(device)
            pred = model(x)
            total, _parts = compute_losses(
                pred,
                y,
                prev_ctrl,
                primary_loss=primary_loss,
                smoothness_weight=smoothness_weight,
                bound_penalty_weight=bound_penalty_weight,
            )
            losses.append(float(total.detach().cpu().item()))
            mae_items.append(float(torch.mean(torch.abs(pred - y)).detach().cpu().item()))
    return {
        "loss": float(np.mean(losses)) if losses else math.inf,
        "mae": float(np.mean(mae_items)) if mae_items else math.inf,
    }


def benchmark_latency(
    model: nn.Module,
    sample_inputs: np.ndarray,
    *,
    device: torch.device,
    benchmark_iters: int,
) -> dict[str, float]:
    if sample_inputs.shape[0] == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    model.eval()
    timings: list[float] = []
    with torch.no_grad():
        for idx in range(min(int(benchmark_iters), int(sample_inputs.shape[0]))):
            x = torch.from_numpy(np.asarray(sample_inputs[idx : idx + 1], dtype=np.float32)).to(device)
            t0 = time.perf_counter()
            _ = model(x)
            timings.append((time.perf_counter() - t0) * 1000.0)
    if not timings:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(np.mean(timings)),
        "p95_ms": float(np.percentile(timings, 95.0)),
    }


def write_report(
    report_path: Path,
    *,
    args: argparse.Namespace,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fast Ctrl Predictor Training Summary",
        "",
        f"- Dataset dir: `{Path(args.dataset_dir).resolve()}`",
        f"- Model type: `{args.model_type}`",
        f"- History length: `{summary['metadata']['history_length']}`",
        f"- Include qvel: `{summary['metadata']['include_qvel']}`",
        f"- Include prev_ctrl: `{summary['metadata']['include_prev_ctrl']}`",
        f"- Include gesture: `{summary['metadata']['include_gesture']}`",
        "",
        "## Validation Metrics",
        "",
        "```json",
        json.dumps(summary["validation_metrics"], indent=2),
        "```",
        "",
        "## Latency Benchmark",
        "",
        "```json",
        json.dumps(summary["latency_benchmark_ms"], indent=2),
        "```",
        "",
        "## Output Files",
        "",
        f"- Checkpoint: `{output_dir / 'model_checkpoint.pt'}`",
        f"- Summary JSON: `{output_dir / 'training_summary.json'}`",
        f"- Metrics CSV: `{output_dir / 'validation_metrics.csv'}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(dataset_dir / "train.csv")
    val_df = pd.read_csv(dataset_dir / "val.csv")
    test_df = pd.read_csv(dataset_dir / "test.csv")
    history_length = infer_history_length(train_df) if int(args.history_length) < 0 else int(args.history_length)
    gesture_vocab = gesture_vocab_from_frames(train_df, val_df, test_df) if bool(args.include_gesture_label) else []
    prepared_train = prepare_dataset(
        train_df,
        history_length=history_length,
        include_qvel=bool(args.include_qvel),
        include_prev_ctrl=bool(args.include_prev_ctrl),
        include_gesture_label=bool(args.include_gesture_label),
        model_type=str(args.model_type),
        gesture_vocab=gesture_vocab,
    )
    prepared_val = prepare_dataset(
        val_df,
        history_length=history_length,
        include_qvel=bool(args.include_qvel),
        include_prev_ctrl=bool(args.include_prev_ctrl),
        include_gesture_label=bool(args.include_gesture_label),
        model_type=str(args.model_type),
        gesture_vocab=gesture_vocab,
    )
    prepared_test = prepare_dataset(
        test_df,
        history_length=history_length,
        include_qvel=bool(args.include_qvel),
        include_prev_ctrl=bool(args.include_prev_ctrl),
        include_gesture_label=bool(args.include_gesture_label),
        model_type=str(args.model_type),
        gesture_vocab=gesture_vocab,
    )
    if str(args.model_type).strip().lower() == "gru":
        train_inputs = prepared_train.model_inputs
        val_inputs = prepared_val.model_inputs
        test_inputs = prepared_test.model_inputs
    else:
        train_inputs = prepared_train.inputs
        val_inputs = prepared_val.inputs
        test_inputs = prepared_test.inputs
    if str(args.model_type).strip().lower() == "gru":
        scaler = Standardizer.from_arrays(np.mean(train_inputs, axis=(0, 1)), np.std(train_inputs, axis=(0, 1)))
        train_inputs_std = scaler.transform(train_inputs.reshape(-1, train_inputs.shape[-1])).reshape(train_inputs.shape)
        val_inputs_std = scaler.transform(val_inputs.reshape(-1, val_inputs.shape[-1])).reshape(val_inputs.shape)
        test_inputs_std = scaler.transform(test_inputs.reshape(-1, test_inputs.shape[-1])).reshape(test_inputs.shape)
    else:
        scaler = Standardizer.from_arrays(np.mean(train_inputs, axis=0), np.std(train_inputs, axis=0))
        train_inputs_std = scaler.transform(train_inputs)
        val_inputs_std = scaler.transform(val_inputs)
        test_inputs_std = scaler.transform(test_inputs)
    train_ds = NumpyDataset(train_inputs_std, prepared_train.targets, prepared_train.prev_ctrl)
    val_ds = NumpyDataset(val_inputs_std, prepared_val.targets, prepared_val.prev_ctrl)
    test_ds = NumpyDataset(test_inputs_std, prepared_test.targets, prepared_test.prev_ctrl)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False)

    device = torch.device(str(args.device))
    model = build_model(args, input_dim=int(train_inputs_std.shape[-1]), output_dim=int(prepared_train.targets.shape[-1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best_state = None
    best_val = math.inf
    epoch_rows: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        model.train()
        train_losses: list[float] = []
        for x, y, prev_ctrl in train_loader:
            x = x.to(device)
            y = y.to(device)
            prev_ctrl = prev_ctrl.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            total, _parts = compute_losses(
                pred,
                y,
                prev_ctrl,
                primary_loss=str(args.loss_type),
                smoothness_weight=float(args.smoothness_weight),
                bound_penalty_weight=float(args.bound_penalty_weight),
            )
            total.backward()
            optimizer.step()
            train_losses.append(float(total.detach().cpu().item()))
        reference_loader = val_loader if len(val_ds) > 0 else test_loader
        val_metrics = evaluate_model(
            model,
            reference_loader,
            device=device,
            primary_loss=str(args.loss_type),
            smoothness_weight=float(args.smoothness_weight),
            bound_penalty_weight=float(args.bound_penalty_weight),
        )
        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(train_losses)) if train_losses else math.inf,
            "val_loss": float(val_metrics["loss"]),
            "val_mae": float(val_metrics["mae"]),
        }
        epoch_rows.append(row)
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    reference_loader = val_loader if len(val_ds) > 0 else test_loader
    reference_inputs = val_inputs_std if len(val_df) > 0 else test_inputs_std
    val_metrics = evaluate_model(
        model,
        reference_loader,
        device=device,
        primary_loss=str(args.loss_type),
        smoothness_weight=float(args.smoothness_weight),
        bound_penalty_weight=float(args.bound_penalty_weight),
    )
    test_metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        primary_loss=str(args.loss_type),
        smoothness_weight=float(args.smoothness_weight),
        bound_penalty_weight=float(args.bound_penalty_weight),
    )
    latency = benchmark_latency(model, reference_inputs, device=device, benchmark_iters=int(args.benchmark_iters))
    metadata = dict(prepared_train.feature_spec)
    metadata.update(
        {
            "model_type": str(args.model_type),
            "input_dim": int(train_inputs_std.shape[-1]),
            "output_dim": int(prepared_train.targets.shape[-1]),
            "loss_type": str(args.loss_type),
            "smoothness_weight": float(args.smoothness_weight),
            "bound_penalty_weight": float(args.bound_penalty_weight),
            "hidden_dims": [int(item) for item in str(args.hidden_dims).split(",") if str(item).strip()],
            "hidden_size": int(args.gru_hidden_size),
            "num_layers": int(args.gru_num_layers),
        }
    )
    checkpoint = {
        "model_state": model.state_dict(),
        "metadata": metadata,
        "scaler": scaler.to_dict(),
    }
    torch.save(checkpoint, output_dir / "model_checkpoint.pt")
    pd.DataFrame(epoch_rows).to_csv(output_dir / "validation_metrics.csv", index=False)
    summary = {
        "metadata": metadata,
        "validation_metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "latency_benchmark_ms": latency,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(Path(args.report_path).resolve(), args=args, summary=summary, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
