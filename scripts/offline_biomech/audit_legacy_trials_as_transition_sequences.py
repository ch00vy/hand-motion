from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TrialAudit:
    source_csv: str
    original_label: str
    trial_id: str
    inferred_motion_type: str
    num_frames: int
    qpos_dim: int
    total_qpos_change: float
    qpos_path_norm: float
    path_efficiency: float
    start_stability_score: float
    end_stability_score: float
    qpos_noise_score: float
    has_end_hold: int
    start_distance_to_open: float
    start_distance_to_fist: float
    end_distance_to_open: float
    end_distance_to_fist: float
    start_posture_margin: float
    end_posture_margin: float
    usable_as_transition: int
    reason_if_not_usable: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit legacy open/fist records as possible transition sequences."
    )
    parser.add_argument("--records-root", default="records", help="Root containing legacy open/ and fist/ folders.")
    parser.add_argument("--open-dir", default=None, help="Optional explicit legacy open directory.")
    parser.add_argument("--fist-dir", default=None, help="Optional explicit legacy fist directory.")
    parser.add_argument("--output-dir", default="artifacts/legacy_transition_sequence_audit", help="Audit output directory.")
    parser.add_argument("--report-path", default="reports/legacy_transition_sequence_audit.md", help="Markdown report path.")
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-total-qpos-change", type=float, default=2.5)
    parser.add_argument("--min-path-efficiency", type=float, default=0.55)
    parser.add_argument("--max-start-stability", type=float, default=0.04)
    parser.add_argument("--max-end-stability", type=float, default=0.04)
    parser.add_argument("--max-noise-score", type=float, default=0.08)
    parser.add_argument("--min-posture-margin-ratio", type=float, default=0.25)
    parser.add_argument("--max-expected-distance-ratio", type=float, default=0.50)
    parser.add_argument("--hold-window-fraction", type=float, default=0.15)
    parser.add_argument("--max-end-hold-path-fraction", type=float, default=0.12)
    return parser.parse_args()


def _qpos_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if str(col).startswith("qpos_")]
    return sorted(cols, key=lambda col: int(str(col).rsplit("_", 1)[-1]) if str(col).rsplit("_", 1)[-1].isdigit() else str(col))


def _load_qpos(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    cols = _qpos_columns(df)
    if not cols:
        raise ValueError("missing_qpos_columns")
    qpos = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    qpos = pd.DataFrame(qpos).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
    return qpos


def _trial_id(path: Path, fallback_index: int) -> str:
    try:
        df = pd.read_csv(path, nrows=1)
        if "trial_id" in df.columns:
            value = str(df["trial_id"].iloc[0]).strip()
            if value:
                return value
    except Exception:
        pass
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    return stem or f"trial_{fallback_index:03d}"


def _window_size(num_frames: int, fraction: float) -> int:
    return max(3, min(12, int(round(max(1, num_frames) * float(fraction)))))


def _mean_step_norm(qpos: np.ndarray) -> float:
    if qpos.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(qpos, axis=0), axis=1)))


def _path_norm(qpos: np.ndarray) -> float:
    if qpos.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(qpos, axis=0), axis=1)))


def _noise_score(qpos: np.ndarray, total_change: float) -> float:
    if qpos.shape[0] < 3:
        return 0.0
    second_diff = np.diff(qpos, n=2, axis=0)
    score = float(np.median(np.linalg.norm(second_diff, axis=1)))
    return score / max(1e-8, float(total_change))


def _collect_trials(open_dir: Path, fist_dir: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for label, directory in (("open", open_dir), ("fist", fist_dir)):
        if not directory.exists():
            continue
        rows.extend((label, path) for path in sorted(directory.glob("*.csv")) if path.is_file())
    return rows


def _build_posture_prototypes(trials: list[tuple[str, Path]], hold_window_fraction: float) -> tuple[np.ndarray, np.ndarray, int]:
    open_like: list[np.ndarray] = []
    fist_like: list[np.ndarray] = []
    qpos_dim = 0
    for label, path in trials:
        try:
            qpos = _load_qpos(path)
        except Exception:
            continue
        if qpos.shape[0] < 3:
            continue
        qpos_dim = int(qpos.shape[1])
        w = _window_size(qpos.shape[0], hold_window_fraction)
        start = np.mean(qpos[:w], axis=0)
        end = np.mean(qpos[-w:], axis=0)
        if label == "fist":
            open_like.append(start)
            fist_like.append(end)
        elif label == "open":
            fist_like.append(start)
            open_like.append(end)
    if not open_like or not fist_like:
        raise RuntimeError("Unable to build open/fist prototypes from legacy records.")
    open_proto = np.median(np.stack(open_like, axis=0), axis=0)
    fist_proto = np.median(np.stack(fist_like, axis=0), axis=0)
    return open_proto, fist_proto, qpos_dim


def _audit_trial(
    *,
    label: str,
    path: Path,
    index: int,
    open_proto: np.ndarray,
    fist_proto: np.ndarray,
    args: argparse.Namespace,
) -> TrialAudit:
    inferred = "open_to_fist" if label == "fist" else "fist_to_open" if label == "open" else "unusable"
    try:
        qpos = _load_qpos(path)
        reject_reasons: list[str] = []
    except Exception as exc:
        return TrialAudit(
            source_csv=str(path),
            original_label=label,
            trial_id=_trial_id(path, index),
            inferred_motion_type="unusable",
            num_frames=0,
            qpos_dim=0,
            total_qpos_change=0.0,
            qpos_path_norm=0.0,
            path_efficiency=0.0,
            start_stability_score=math.inf,
            end_stability_score=math.inf,
            qpos_noise_score=math.inf,
            has_end_hold=0,
            start_distance_to_open=math.inf,
            start_distance_to_fist=math.inf,
            end_distance_to_open=math.inf,
            end_distance_to_fist=math.inf,
            start_posture_margin=-math.inf,
            end_posture_margin=-math.inf,
            usable_as_transition=0,
            reason_if_not_usable=f"load_failed:{type(exc).__name__}:{exc}",
        )

    w = _window_size(qpos.shape[0], float(args.hold_window_fraction))
    start_window = qpos[:w]
    end_window = qpos[-w:]
    start = np.mean(start_window, axis=0)
    end = np.mean(end_window, axis=0)
    total_change = float(np.linalg.norm(end - start))
    path_norm = _path_norm(qpos)
    path_efficiency = total_change / max(1e-8, path_norm)
    start_stability = _mean_step_norm(start_window)
    end_stability = _mean_step_norm(end_window)
    noise_score = _noise_score(qpos, total_change)
    end_hold_path = _path_norm(end_window)
    has_end_hold = int(end_stability <= float(args.max_end_stability) and end_hold_path <= float(args.max_end_hold_path_fraction) * max(1e-8, total_change))

    start_open = float(np.linalg.norm(start - open_proto))
    start_fist = float(np.linalg.norm(start - fist_proto))
    end_open = float(np.linalg.norm(end - open_proto))
    end_fist = float(np.linalg.norm(end - fist_proto))
    proto_distance = float(np.linalg.norm(open_proto - fist_proto))
    margin_denom = max(1e-8, proto_distance)

    if inferred == "open_to_fist":
        start_expected = start_open
        start_opposite = start_fist
        end_expected = end_fist
        end_opposite = end_open
    elif inferred == "fist_to_open":
        start_expected = start_fist
        start_opposite = start_open
        end_expected = end_open
        end_opposite = end_fist
    else:
        start_expected = math.inf
        start_opposite = 0.0
        end_expected = math.inf
        end_opposite = 0.0

    start_margin = (start_opposite - start_expected) / margin_denom
    end_margin = (end_opposite - end_expected) / margin_denom
    max_expected_distance = float(args.max_expected_distance_ratio) * margin_denom

    reject_reasons: list[str] = []
    if qpos.shape[0] < int(args.min_frames):
        reject_reasons.append(f"too_few_frames:{qpos.shape[0]}<{int(args.min_frames)}")
    if total_change < float(args.min_total_qpos_change):
        reject_reasons.append(f"low_total_qpos_change:{total_change:.4f}<{float(args.min_total_qpos_change):.4f}")
    if path_efficiency < float(args.min_path_efficiency):
        reject_reasons.append(f"low_path_efficiency:{path_efficiency:.4f}<{float(args.min_path_efficiency):.4f}")
    if start_stability > float(args.max_start_stability):
        reject_reasons.append(f"unstable_start:{start_stability:.4f}>{float(args.max_start_stability):.4f}")
    if end_stability > float(args.max_end_stability):
        reject_reasons.append(f"unstable_end:{end_stability:.4f}>{float(args.max_end_stability):.4f}")
    if noise_score > float(args.max_noise_score):
        reject_reasons.append(f"noisy_qpos:{noise_score:.4f}>{float(args.max_noise_score):.4f}")
    if start_expected > start_opposite:
        reject_reasons.append("start_closer_to_wrong_posture")
    if end_expected > end_opposite:
        reject_reasons.append("end_closer_to_wrong_posture")
    if start_margin < float(args.min_posture_margin_ratio):
        reject_reasons.append(f"weak_start_posture_margin:{start_margin:.4f}<{float(args.min_posture_margin_ratio):.4f}")
    if end_margin < float(args.min_posture_margin_ratio):
        reject_reasons.append(f"weak_end_posture_margin:{end_margin:.4f}<{float(args.min_posture_margin_ratio):.4f}")
    if start_expected > max_expected_distance:
        reject_reasons.append(f"start_too_far_from_expected:{start_expected:.4f}>{max_expected_distance:.4f}")
    if end_expected > max_expected_distance:
        reject_reasons.append(f"end_too_far_from_expected:{end_expected:.4f}>{max_expected_distance:.4f}")
    if not bool(has_end_hold):
        reject_reasons.append("missing_clear_end_hold")

    usable = int(len(reject_reasons) == 0)
    return TrialAudit(
        source_csv=str(path),
        original_label=label,
        trial_id=_trial_id(path, index),
        inferred_motion_type=inferred if usable else "unusable",
        num_frames=int(qpos.shape[0]),
        qpos_dim=int(qpos.shape[1]),
        total_qpos_change=total_change,
        qpos_path_norm=path_norm,
        path_efficiency=path_efficiency,
        start_stability_score=start_stability,
        end_stability_score=end_stability,
        qpos_noise_score=noise_score,
        has_end_hold=has_end_hold,
        start_distance_to_open=start_open,
        start_distance_to_fist=start_fist,
        end_distance_to_open=end_open,
        end_distance_to_fist=end_fist,
        start_posture_margin=start_margin,
        end_posture_margin=end_margin,
        usable_as_transition=usable,
        reason_if_not_usable=";".join(reject_reasons),
    )


def _write_report(report_path: Path, output_dir: Path, rows: pd.DataFrame, summary: dict[str, object]) -> None:
    usable = rows[rows["usable_as_transition"] == 1]
    unusable = rows[rows["usable_as_transition"] == 0]
    lines = [
        "# Legacy Transition Sequence Audit",
        "",
        "## Summary",
        "",
        f"- Total audited trials: {len(rows)}",
        f"- Usable as open_to_fist: {int(((usable['inferred_motion_type'] == 'open_to_fist')).sum()) if not usable.empty else 0}",
        f"- Usable as fist_to_open: {int(((usable['inferred_motion_type'] == 'fist_to_open')).sum()) if not usable.empty else 0}",
        f"- Unusable: {len(unusable)}",
        f"- Open/fist prototype distance: {float(summary.get('prototype_distance', 0.0)):.4f}",
        "",
        "## Criteria",
        "",
        f"- min_frames: {summary['criteria']['min_frames']}",
        f"- min_total_qpos_change: {summary['criteria']['min_total_qpos_change']}",
        f"- min_path_efficiency: {summary['criteria']['min_path_efficiency']}",
        f"- max_start_stability: {summary['criteria']['max_start_stability']}",
        f"- max_end_stability: {summary['criteria']['max_end_stability']}",
        f"- max_noise_score: {summary['criteria']['max_noise_score']}",
        f"- min_posture_margin_ratio: {summary['criteria']['min_posture_margin_ratio']}",
        "",
        "## Findings",
        "",
    ]
    if not usable.empty:
        lines.extend(
            [
                f"- Median total qpos change for usable trials: {float(usable['total_qpos_change'].median()):.4f}",
                f"- Median path efficiency for usable trials: {float(usable['path_efficiency'].median()):.4f}",
                f"- Median end stability for usable trials: {float(usable['end_stability_score'].median()):.4f}",
                "",
            ]
        )
    if not unusable.empty:
        lines.append("Top rejection reasons:")
        reason_counts: dict[str, int] = {}
        for reason_text in unusable["reason_if_not_usable"].fillna(""):
            for reason in str(reason_text).split(";"):
                if not reason:
                    continue
                key = reason.split(":", 1)[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- `{reason}`: {count}")
        lines.append("")
    lines.extend(
        [
            "## Verdict",
            "",
            str(summary.get("verdict", "")),
            "",
            "## Output Files",
            "",
            f"- `{output_dir / 'legacy_transition_trial_audit.csv'}`",
            f"- `{output_dir / 'legacy_transition_summary.json'}`",
            f"- `{output_dir / 'usable_transition_trials.csv'}`",
            f"- `{output_dir / 'unusable_transition_trials.csv'}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    records_root = Path(args.records_root).resolve()
    open_dir = Path(args.open_dir).resolve() if args.open_dir else records_root / "open"
    fist_dir = Path(args.fist_dir).resolve() if args.fist_dir else records_root / "fist"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = _collect_trials(open_dir, fist_dir)
    open_proto, fist_proto, qpos_dim = _build_posture_prototypes(trials, float(args.hold_window_fraction))
    audits = [
        _audit_trial(label=label, path=path, index=idx, open_proto=open_proto, fist_proto=fist_proto, args=args)
        for idx, (label, path) in enumerate(trials, start=1)
    ]
    rows = pd.DataFrame([asdict(item) for item in audits])
    rows.to_csv(output_dir / "legacy_transition_trial_audit.csv", index=False)
    rows[rows["usable_as_transition"] == 1].to_csv(output_dir / "usable_transition_trials.csv", index=False)
    rows[rows["usable_as_transition"] == 0].to_csv(output_dir / "unusable_transition_trials.csv", index=False)

    usable = rows[rows["usable_as_transition"] == 1]
    summary = {
        "records_root": str(records_root),
        "open_dir": str(open_dir),
        "fist_dir": str(fist_dir),
        "num_trials": int(len(rows)),
        "num_usable": int(len(usable)),
        "num_unusable": int(len(rows) - len(usable)),
        "usable_open_to_fist": int((usable["inferred_motion_type"] == "open_to_fist").sum()) if not usable.empty else 0,
        "usable_fist_to_open": int((usable["inferred_motion_type"] == "fist_to_open").sum()) if not usable.empty else 0,
        "qpos_dim": int(qpos_dim),
        "prototype_distance": float(np.linalg.norm(open_proto - fist_proto)),
        "criteria": {
            "min_frames": int(args.min_frames),
            "min_total_qpos_change": float(args.min_total_qpos_change),
            "min_path_efficiency": float(args.min_path_efficiency),
            "max_start_stability": float(args.max_start_stability),
            "max_end_stability": float(args.max_end_stability),
            "max_noise_score": float(args.max_noise_score),
            "min_posture_margin_ratio": float(args.min_posture_margin_ratio),
            "max_expected_distance_ratio": float(args.max_expected_distance_ratio),
            "max_end_hold_path_fraction": float(args.max_end_hold_path_fraction),
        },
    }
    if int(summary["num_usable"]) >= max(1, int(0.75 * len(rows))):
        summary["verdict"] = (
            "Legacy open/fist records are usable as a formal transition-sequence dataset, "
            "provided they are relabeled with explicit motion_type metadata and still pass downstream ctrl_ref validation."
        )
    else:
        summary["verdict"] = (
            "Legacy open/fist records are not yet sufficient as a formal transition-sequence dataset; "
            "recording cleanup or re-recording is recommended."
        )
    (output_dir / "legacy_transition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "posture_prototypes.npz").write_bytes(b"")
    np.savez_compressed(output_dir / "posture_prototypes.npz", open_proto=open_proto, fist_proto=fist_proto)

    _write_report(Path(args.report_path).resolve(), output_dir, rows, summary)
    print(
        "[INFO] legacy transition audit done | "
        f"usable={summary['num_usable']} unusable={summary['num_unusable']} "
        f"open_to_fist={summary['usable_open_to_fist']} fist_to_open={summary['usable_fist_to_open']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
