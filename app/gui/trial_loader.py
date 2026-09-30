from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "records"


@dataclass
class TrialData:
    trial_dir: Path
    qpos_csv: Path
    meta_path: Optional[Path]
    qpos: np.ndarray
    meta: dict[str, Any]
    qpos_columns: list[str]

    @property
    def trial_id(self) -> str:
        return str(self.meta.get("trial_id") or self.trial_dir.name)

    @property
    def motion_type(self) -> str:
        return str(self.meta.get("motion_type") or self.meta.get("exercise_name") or "open_to_fist")

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])


def _safe_slug(value: str, default: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return text or str(default)


def _qpos_columns(df: pd.DataFrame) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for name in df.columns:
        label = str(name)
        if not label.startswith("qpos_"):
            continue
        try:
            idx = int(label[5:])
        except ValueError:
            continue
        indexed.append((idx, label))
    indexed.sort(key=lambda item: item[0])
    return [label for _, label in indexed]


def save_trial(
    *,
    qpos: np.ndarray,
    subject_id: str,
    motion_type: str = "open_to_fist",
    trial_label: str = "normal",
    dt: float = 1.0 / 60.0,
    records_root: Path = DEFAULT_RECORDS_ROOT,
    source_mode: str = "gui",
) -> TrialData:
    qpos_arr = np.asarray(qpos, dtype=float)
    if qpos_arr.ndim != 2 or qpos_arr.shape[0] <= 0:
        raise ValueError("Cannot save an empty qpos trial.")

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subject_slug = _safe_slug(subject_id, "subject_unknown")
    motion_slug = _safe_slug(motion_type, "open_to_fist")
    label_slug = _safe_slug(trial_label, "normal").lower()
    if label_slug not in {"normal", "abnormal", "testing"}:
        label_slug = "normal"
    trial_id = f"{stamp}_{motion_slug}"
    trial_dir = Path(records_root).resolve() / label_slug / subject_slug / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    frame_count = int(qpos_arr.shape[0])
    columns: dict[str, Any] = {
        "frame_index": np.arange(frame_count, dtype=int),
        "t": np.arange(frame_count, dtype=float) * float(dt),
        "progress_percent": np.linspace(0.0, 100.0, frame_count, dtype=float),
        "subject_id": [subject_slug] * frame_count,
        "trial_label": [label_slug] * frame_count,
        "label": [label_slug] * frame_count,
        "trial_id": [trial_id] * frame_count,
        "exercise_name": [motion_slug] * frame_count,
        "motion_type": [motion_slug] * frame_count,
    }
    for idx in range(qpos_arr.shape[1]):
        columns[f"qpos_{idx}"] = qpos_arr[:, idx]
    qpos_csv = trial_dir / "qpos.csv"
    pd.DataFrame(columns).to_csv(qpos_csv, index=False)

    meta = {
        "trial_id": trial_id,
        "subject_id": subject_slug,
        "trial_label": label_slug,
        "label": label_slug,
        "motion_type": motion_slug,
        "exercise_name": motion_slug,
        "num_frames": frame_count,
        "qpos_dim": int(qpos_arr.shape[1]),
        "dt": float(dt),
        "created_at": created_at,
        "source_mode": str(source_mode),
        "qpos_csv": str(qpos_csv),
    }
    meta_path = trial_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return load_trial(qpos_csv)


def load_trial(path: Union[str, Path]) -> TrialData:
    raw_path = Path(path).resolve()
    if raw_path.is_dir():
        csv_path = raw_path / "qpos.csv"
    else:
        csv_path = raw_path
    if not csv_path.exists():
        raise FileNotFoundError(f"qpos.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = _qpos_columns(df)
    if not cols:
        raise ValueError(f"No qpos_* columns found in {csv_path}")
    qpos = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    trial_dir = csv_path.parent
    meta_path = trial_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    def first_string(column: str) -> str:
        if column not in df.columns:
            return ""
        values = [str(item).strip() for item in df[column].dropna().tolist() if str(item).strip()]
        return values[0] if values else ""

    inferred_label = "normal"
    inferred_subject = trial_dir.parent.name
    if trial_dir.parent.parent.name in {"normal", "abnormal", "testing"}:
        inferred_label = trial_dir.parent.parent.name
        inferred_subject = trial_dir.parent.name
    meta.setdefault("trial_id", first_string("trial_id") or trial_dir.name)
    meta.setdefault("subject_id", first_string("subject_id") or inferred_subject)
    meta.setdefault("trial_label", first_string("trial_label") or first_string("label") or inferred_label)
    meta.setdefault("label", meta["trial_label"])
    meta.setdefault("motion_type", first_string("motion_type") or first_string("exercise_name") or "open_to_fist")
    meta.setdefault("exercise_name", meta["motion_type"])
    meta.setdefault("num_frames", int(qpos.shape[0]))
    meta.setdefault("qpos_dim", int(qpos.shape[1]))
    if "t" in df.columns and len(df) > 1:
        t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
        diffs = np.diff(t[np.isfinite(t)])
        if diffs.size:
            meta.setdefault("dt", float(np.median(diffs)))
    meta.setdefault("dt", 1.0 / 60.0)
    meta["qpos_csv"] = str(csv_path)
    return TrialData(trial_dir=trial_dir, qpos_csv=csv_path, meta_path=meta_path if meta_path.exists() else None, qpos=qpos, meta=meta, qpos_columns=cols)


def write_result_files(trial: TrialData, result: dict[str, Any]) -> tuple[Path, Path]:
    result_json = trial.trial_dir / "result.json"
    result_csv = trial.trial_dir / "result.csv"
    result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = list(result.get("nodes") or [])
    pd.DataFrame(rows).to_csv(result_csv, index=False)
    return result_json, result_csv
