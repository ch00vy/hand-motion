from __future__ import annotations

"""Lightweight CSV logger for live realtime runtime inspection.

The logger is intentionally simple: append-only CSV with periodic flushes so
long live sessions can be inspected even if the process is interrupted.
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np


def flatten_array(prefix: str, values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {f"{prefix}{idx}": float(arr[idx]) for idx in range(arr.shape[0])}


class RuntimeCSVLogger:
    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = False,
        max_frames: int = 0,
        flush_every: int = 10,
    ) -> None:
        self.path = Path(path).resolve()
        self.enabled = bool(enabled)
        self.max_frames = max(0, int(max_frames))
        self.flush_every = max(1, int(flush_every))
        self._file = None
        self._writer = None
        self._fieldnames: list[str] | None = None
        self._written = 0

    def write_row(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self.max_frames > 0 and self._written >= self.max_frames:
            return
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
            self._writer.writeheader()
        payload = {key: row.get(key, "") for key in (self._fieldnames or row.keys())}
        self._writer.writerow(payload)
        self._written += 1
        if self._written % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._file is not None:
                self._file.close()
            self._file = None
            self._writer = None
