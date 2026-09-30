from __future__ import annotations

"""Normalization helpers for indicator signals.

These helpers convert torque/effort-like signals into bounded 0..1 indicators.
The normalized values are for visualization and node features, not direct
actuator commands.
"""

from dataclasses import dataclass

import numpy as np


def finite_vector(values: np.ndarray | list[float] | tuple[float, ...], *, width: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if width is not None:
        out = np.zeros((max(0, int(width)),), dtype=float)
        out[: min(out.size, arr.size)] = arr[: min(out.size, arr.size)]
        arr = out
    return np.where(np.isfinite(arr), arr, 0.0)


def normalize_abs_indicator(
    values: np.ndarray | list[float] | tuple[float, ...],
    *,
    scale: float | None = None,
    clip: float | None = None,
    percentile: float = 95.0,
) -> np.ndarray:
    """Return a stable 0..1 indicator from absolute signal magnitude."""

    arr = np.abs(finite_vector(values))
    if arr.size == 0:
        return arr.astype(float)
    if clip is not None and float(clip) > 0.0:
        arr = np.minimum(arr, float(clip))
    if scale is None or float(scale) <= 1e-12:
        positive = arr[arr > 1e-12]
        if positive.size:
            pct = float(np.clip(percentile, 50.0, 100.0))
            scale_value = float(np.percentile(positive, pct))
        else:
            scale_value = 1.0
    else:
        scale_value = float(scale)
    scale_value = max(scale_value, 1e-12)
    return np.clip(arr / scale_value, 0.0, 1.0)


@dataclass
class RunningIndicatorNormalizer:
    """Online normalizer for realtime indicators."""

    initial_scale: float = 1.0
    ema_alpha: float = 0.08
    min_scale: float = 1e-6

    def __post_init__(self) -> None:
        self.scale = max(float(self.initial_scale), float(self.min_scale))

    def reset(self) -> None:
        self.scale = max(float(self.initial_scale), float(self.min_scale))

    def normalize(self, values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        arr = np.abs(finite_vector(values))
        if arr.size == 0:
            return arr.astype(float)
        observed = float(np.max(arr))
        if observed > float(self.min_scale):
            alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
            self.scale = max(float(self.min_scale), (1.0 - alpha) * self.scale + alpha * observed)
        return np.clip(arr / max(self.scale, float(self.min_scale)), 0.0, 1.0)
