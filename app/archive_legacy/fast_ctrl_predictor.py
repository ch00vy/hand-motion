from __future__ import annotations

"""Lightweight ctrl predictor models and runtime helpers.

These models are intended to imitate offline teacher controls in realtime.
They are model-derived surrogates for fast inference, not biological truth.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - runtime fallback
    torch = None
    nn = None


def _as_float32(values: np.ndarray | list[float] | tuple[float, ...], size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] == size:
        return arr
    out = np.zeros(int(size), dtype=np.float32)
    out[: min(int(size), arr.shape[0])] = arr[: min(int(size), arr.shape[0])]
    return out


class FastCtrlMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] | tuple[int, ...]):
        super().__init__()
        dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(output_dim)]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastCtrlGRU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.head = nn.Linear(int(hidden_size), int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.gru(x)
        return self.head(seq_out[:, -1, :])


def build_model_from_metadata(metadata: dict[str, Any]) -> nn.Module:
    model_type = str(metadata.get("model_type", "mlp")).strip().lower()
    input_dim = int(metadata["input_dim"])
    output_dim = int(metadata["output_dim"])
    if model_type == "gru":
        return FastCtrlGRU(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=int(metadata.get("hidden_size", 128)),
            num_layers=int(metadata.get("num_layers", 1)),
        )
    hidden_dims = [int(dim) for dim in list(metadata.get("hidden_dims") or [256, 256])]
    return FastCtrlMLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_arrays(cls, mean: np.ndarray, std: np.ndarray) -> "Standardizer":
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1)
        std_arr = np.asarray(std, dtype=np.float32).reshape(-1)
        std_arr = np.where(np.abs(std_arr) < 1e-6, 1.0, std_arr)
        return cls(mean=mean_arr, std=std_arr)

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return (arr - self.mean) / self.std

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }


class FastCtrlPredictorRuntime:
    """Stateful runtime wrapper for lightweight ctrl prediction."""

    def __init__(self, checkpoint: dict[str, Any], *, device: str = "cpu"):
        if torch is None:
            raise RuntimeError("torch is required for fast predictor runtime")
        self.device = torch.device(device)
        self.metadata = dict(checkpoint.get("metadata") or {})
        self.model = build_model_from_metadata(self.metadata)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        scaler_dict = dict(checkpoint.get("scaler") or {})
        self.scaler = Standardizer.from_arrays(
            np.asarray(scaler_dict.get("mean") or [], dtype=np.float32),
            np.asarray(scaler_dict.get("std") or [], dtype=np.float32),
        )
        self.history_length = int(self.metadata.get("history_length", 0))
        self.sequence_feature_dim = int(self.metadata.get("sequence_feature_dim", 0))
        self.num_qpos = int(self.metadata["num_qpos"])
        self.num_ctrl = int(self.metadata["num_ctrl"])
        self.include_qvel = bool(self.metadata.get("include_qvel", True))
        self.include_prev_ctrl = bool(self.metadata.get("include_prev_ctrl", True))
        self.include_gesture = bool(self.metadata.get("include_gesture", False))
        self.gesture_vocab = list(self.metadata.get("gesture_vocab") or [])
        self.model_type = str(self.metadata.get("model_type", "mlp")).strip().lower()
        self.qpos_history: deque[np.ndarray] = deque(maxlen=max(1, self.history_length))
        self.qvel_history: deque[np.ndarray] = deque(maxlen=max(1, self.history_length))
        self.ctrl_history: deque[np.ndarray] = deque(maxlen=max(1, self.history_length))

    def reset(self) -> None:
        self.qpos_history.clear()
        self.qvel_history.clear()
        self.ctrl_history.clear()

    def _gesture_one_hot(self, gesture_label: str | None) -> np.ndarray:
        if not self.include_gesture or not self.gesture_vocab:
            return np.zeros(0, dtype=np.float32)
        one_hot = np.zeros(len(self.gesture_vocab), dtype=np.float32)
        if gesture_label is None:
            return one_hot
        label = str(gesture_label).strip().lower()
        for idx, item in enumerate(self.gesture_vocab):
            if label == str(item).strip().lower():
                one_hot[idx] = 1.0
                break
        return one_hot

    def _sequence_steps(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
        prev_ctrl: np.ndarray,
        gesture_vec: np.ndarray,
    ) -> np.ndarray:
        steps: list[np.ndarray] = []
        hist_count = int(self.history_length)
        zero_qpos = np.zeros(self.num_qpos, dtype=np.float32)
        zero_qvel = np.zeros(self.num_qpos, dtype=np.float32)
        zero_ctrl = np.zeros(self.num_ctrl, dtype=np.float32)
        for idx in range(hist_count, 0, -1):
            qpos_hist = self.qpos_history[-idx] if len(self.qpos_history) >= idx else zero_qpos
            qvel_hist = self.qvel_history[-idx] if len(self.qvel_history) >= idx else zero_qvel
            ctrl_hist = self.ctrl_history[-idx] if len(self.ctrl_history) >= idx else zero_ctrl
            parts = [qpos_hist]
            if self.include_qvel:
                parts.append(qvel_hist)
            if self.include_prev_ctrl:
                parts.append(ctrl_hist)
            if gesture_vec.size:
                parts.append(gesture_vec)
            steps.append(np.concatenate(parts, axis=0).astype(np.float32, copy=False))
        parts_now = [qpos]
        if self.include_qvel:
            parts_now.append(qvel)
        if self.include_prev_ctrl:
            parts_now.append(prev_ctrl)
        if gesture_vec.size:
            parts_now.append(gesture_vec)
        steps.append(np.concatenate(parts_now, axis=0).astype(np.float32, copy=False))
        return np.stack(steps, axis=0)

    def _flat_features(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
        prev_ctrl: np.ndarray,
        gesture_vec: np.ndarray,
    ) -> np.ndarray:
        parts = [qpos]
        if self.include_qvel:
            parts.append(qvel)
        if self.include_prev_ctrl:
            parts.append(prev_ctrl)
        for idx in range(1, self.history_length + 1):
            qpos_hist = self.qpos_history[-idx] if len(self.qpos_history) >= idx else np.zeros(self.num_qpos, dtype=np.float32)
            parts.append(qpos_hist)
            if self.include_qvel:
                qvel_hist = self.qvel_history[-idx] if len(self.qvel_history) >= idx else np.zeros(self.num_qpos, dtype=np.float32)
                parts.append(qvel_hist)
            if self.include_prev_ctrl:
                ctrl_hist = self.ctrl_history[-idx] if len(self.ctrl_history) >= idx else np.zeros(self.num_ctrl, dtype=np.float32)
                parts.append(ctrl_hist)
        if gesture_vec.size:
            parts.append(gesture_vec)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def predict(
        self,
        *,
        qpos: np.ndarray | list[float],
        qvel: np.ndarray | list[float] | None,
        prev_ctrl: np.ndarray | list[float] | None,
        gesture_label: str | None = None,
    ) -> np.ndarray:
        qpos_arr = _as_float32(qpos, self.num_qpos)
        qvel_source = np.zeros(self.num_qpos, dtype=np.float32) if qvel is None else qvel
        prev_ctrl_source = np.zeros(self.num_ctrl, dtype=np.float32) if prev_ctrl is None else prev_ctrl
        qvel_arr = _as_float32(qvel_source, self.num_qpos)
        prev_ctrl_arr = _as_float32(prev_ctrl_source, self.num_ctrl)
        gesture_vec = self._gesture_one_hot(gesture_label)
        if self.model_type == "gru":
            seq = self._sequence_steps(qpos=qpos_arr, qvel=qvel_arr, prev_ctrl=prev_ctrl_arr, gesture_vec=gesture_vec)
            seq_std = self.scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(seq.shape)
            x = torch.from_numpy(seq_std[None, ...]).to(self.device)
        else:
            flat = self._flat_features(qpos=qpos_arr, qvel=qvel_arr, prev_ctrl=prev_ctrl_arr, gesture_vec=gesture_vec)
            flat_std = self.scaler.transform(flat)
            x = torch.from_numpy(flat_std[None, ...]).to(self.device)
        with torch.no_grad():
            pred = self.model(x).detach().cpu().numpy().reshape(-1)
        pred = np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)
        self.qpos_history.append(qpos_arr.copy())
        self.qvel_history.append(qvel_arr.copy())
        self.ctrl_history.append(pred.copy())
        return pred


def load_fast_ctrl_predictor_runtime(checkpoint_path: str | Path, *, device: str = "cpu") -> FastCtrlPredictorRuntime:
    if torch is None:
        raise RuntimeError("torch is required to load a fast ctrl predictor checkpoint")
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    return FastCtrlPredictorRuntime(checkpoint, device=device)
