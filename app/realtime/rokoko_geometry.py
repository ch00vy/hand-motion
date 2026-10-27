"""Hand geometry in a world-coordinate-independent anatomical frame."""
from __future__ import annotations

from typing import Mapping
import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or length < 1e-9:
        raise ValueError("Invalid or degenerate hand geometry")
    return vector / length


def hand_basis(points: Mapping[str, np.ndarray]) -> np.ndarray:
    """Columns: radial (little to index), longitudinal, right-hand normal."""
    longitudinal = unit(np.mean([
        points[f"{name}_proximal"] - points["wrist"]
        for name in ("index", "middle", "ring", "little")
    ], axis=0))
    radial = points["index_proximal"] - points["little_proximal"]
    radial = unit(radial - longitudinal * np.dot(radial, longitudinal))
    return np.column_stack((radial, longitudinal, unit(np.cross(radial, longitudinal))))
