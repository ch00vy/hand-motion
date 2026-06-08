from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import DEFAULT_MYOSUITE_ENV_ID, _get_raw_model_data, create_probe_env, estimate_ctrl_from_qpos, set_pose
from app.offline_u_analysis import (
    DEFAULT_EXERCISE_NAMES,
    build_exercise_profiles,
    build_trial_table,
    run_leave_one_trial_out_classifier,
    write_confusion_matrix_csv,
    write_trial_feature_csv,
)


def _indexed_columns(field_names: list[str], prefix: str) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for name in field_names:
        if not str(name).startswith(prefix):
            continue
        try:
            idx = int(str(name)[len(prefix) :])
        except ValueError:
            continue
        indexed.append((idx, str(name)))
    indexed.sort(key=lambda item: item[0])
    return [name for _, name in indexed]


def _unique_nonempty(values: list[str]) -> str | None:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    unique = sorted(set(cleaned))
    if not unique:
        return None
    if len(unique) > 1:
        raise ValueError(f"Inconsistent metadata values: {unique}")
    return unique[0]


def load_labelled_qpos_trial_csv(path: Path, allowed_exercises: set[str]) -> dict[str, Any] | None:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        field_names = list(reader.fieldnames or [])
        if not field_names:
            return None

        qpos_columns = _indexed_columns(field_names, "qpos_")
        if not qpos_columns:
            return None

        exercise_values: list[str] = []
        trial_values: list[str] = []
        session_values: list[str] = []
        frame_indices: list[int] = []
        qpos_rows: list[list[float]] = []

        for row_idx, row in enumerate(reader):
            exercise_values.append(str(row.get("exercise_name") or ""))
            trial_values.append(str(row.get("trial_id") or ""))
            session_values.append(str(row.get("session_id") or ""))
            frame_indices.append(int(float(row.get("frame_index") or row.get("frame") or row_idx)))
            qpos_rows.append([float(row.get(name) or 0.0) for name in qpos_columns])

    exercise_name = _unique_nonempty(exercise_values)
    trial_id = _unique_nonempty(trial_values)
    session_id = _unique_nonempty(session_values)
    if exercise_name is None or trial_id is None or session_id is None:
        return None
    if exercise_name not in allowed_exercises:
        return None

    qpos = np.asarray(qpos_rows, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        return None

    return {
        "source_csv": str(path),
        "exercise_name": exercise_name,
        "trial_id": trial_id,
        "session_id": session_id,
        "frame_indices": frame_indices,
        "qpos": qpos,
    }


def discover_labelled_qpos_trials(records_dir: Path, allowed_exercises: set[str]) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("*.csv")):
        try:
            item = load_labelled_qpos_trial_csv(path, allowed_exercises)
        except Exception as exc:
            print(f"[WARN] Skipping {path.name}: {type(exc).__name__}: {exc}")
            continue
        if item is not None:
            trials.append(item)
    return trials


def _select_frame_indices(num_frames: int, *, frame_step: int, max_frames_per_trial: int) -> list[int]:
    indices = list(range(0, int(num_frames), max(1, int(frame_step))))
    if max_frames_per_trial > 0:
        indices = indices[: int(max_frames_per_trial)]
    return indices


def evaluate_forward_pose_validity(
    model: Any,
    data: Any,
    qpos_target: np.ndarray,
    u_vec: np.ndarray,
    *,
    hold_steps: int,
) -> dict[str, float]:
    qpos_target = np.asarray(qpos_target, dtype=np.float64).reshape(-1)
    u_vec = np.asarray(u_vec, dtype=np.float64).reshape(-1)
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()
    qacc_orig = data.qacc.copy()
    ctrl_orig = data.ctrl.copy()
    act_orig = data.act.copy() if int(getattr(model, "na", 0)) > 0 else None
    m_raw, d_raw = _get_raw_model_data(model, data)

    pose_errors: list[float] = []
    try:
        set_pose(model, data, qpos_target)
        if act_orig is not None:
            data.act[:] = 0.0
            mujoco.mj_forward(m_raw, d_raw)

        q_len = min(int(model.nq), int(qpos_target.shape[0]))
        for _ in range(max(1, int(hold_steps))):
            data.ctrl[:] = 0.0
            data.ctrl[: min(int(model.nu), int(u_vec.shape[0]))] = u_vec[: min(int(model.nu), int(u_vec.shape[0]))]
            mujoco.mj_step(m_raw, d_raw)
            err = data.qpos[:q_len] - qpos_target[:q_len]
            pose_errors.append(float(np.linalg.norm(err)))

        final_pose_error = pose_errors[-1] if pose_errors else 0.0
        mean_pose_error = float(np.mean(pose_errors)) if pose_errors else 0.0
        return {
            "final_pose_error": float(final_pose_error),
            "mean_pose_error": float(mean_pose_error),
        }
    finally:
        data.qpos[:] = qpos_orig
        data.qvel[:] = qvel_orig
        data.qacc[:] = qacc_orig
        data.ctrl[:] = ctrl_orig
        if act_orig is not None:
            data.act[:] = act_orig
        mujoco.mj_forward(m_raw, d_raw)


def build_static_opt_trials(
    trials: list[dict[str, Any]],
    *,
    model: Any,
    data: Any,
    frame_step: int,
    max_frames_per_trial: int,
    hold_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_trials: list[dict[str, Any]] = []
    frame_metric_rows: list[dict[str, Any]] = []
    num_muscles = int(getattr(model, "nu", 0))
    muscle_names = [f"muscle_{idx}" for idx in range(num_muscles)]

    for trial in trials:
        qpos = np.asarray(trial["qpos"], dtype=np.float64)
        selected_indices = _select_frame_indices(qpos.shape[0], frame_step=frame_step, max_frames_per_trial=max_frames_per_trial)
        u_rows: list[np.ndarray] = []
        residuals: list[float] = []
        relative_errors: list[float] = []
        final_pose_errors: list[float] = []
        mean_pose_errors: list[float] = []
        used_frame_indices: list[int] = []

        print(
            "[INFO] Static optimization "
            f"{trial['exercise_name']} / {trial['trial_id']} | "
            f"frames={len(selected_indices)}/{qpos.shape[0]}"
        )

        for frame_idx in selected_indices:
            try:
                u_vec, info = estimate_ctrl_from_qpos(model, data, qpos[frame_idx])
            except Exception as exc:
                print(f"[WARN] Failed static solve {trial['trial_id']} frame {frame_idx}: {type(exc).__name__}: {exc}")
                continue
            u_vec = np.asarray(u_vec, dtype=np.float64).reshape(-1)
            tau_req = np.asarray(info.get("tau_required"), dtype=np.float64).reshape(-1)
            tau_hat = np.asarray(info.get("tau_reconstructed"), dtype=np.float64).reshape(-1)
            residual_norm = float(info.get("residual_norm", 0.0))
            tau_req_norm = float(np.linalg.norm(tau_req))
            relative_error = float(residual_norm / max(tau_req_norm, 1e-8))
            pose_validity = evaluate_forward_pose_validity(
                model,
                data,
                qpos[frame_idx],
                u_vec,
                hold_steps=hold_steps,
            )

            u_rows.append(u_vec)
            residuals.append(residual_norm)
            relative_errors.append(relative_error)
            final_pose_errors.append(float(pose_validity["final_pose_error"]))
            mean_pose_errors.append(float(pose_validity["mean_pose_error"]))
            used_frame_indices.append(int(frame_idx))
            frame_metric_rows.append(
                {
                    "source_csv": str(trial["source_csv"]),
                    "exercise_name": str(trial["exercise_name"]),
                    "trial_id": str(trial["trial_id"]),
                    "session_id": str(trial["session_id"]),
                    "frame_index": int(frame_idx),
                    "residual_torque_norm": float(residual_norm),
                    "relative_torque_error": float(relative_error),
                    "final_pose_error": float(pose_validity["final_pose_error"]),
                    "mean_pose_error": float(pose_validity["mean_pose_error"]),
                }
            )

        if not u_rows:
            print(f"[WARN] No valid static optimization frames for {trial['trial_id']}")
            continue

        output_trials.append(
            {
                "source_csv": str(trial["source_csv"]),
                "exercise_name": str(trial["exercise_name"]),
                "trial_id": str(trial["trial_id"]),
                "session_id": str(trial["session_id"]),
                "frame_indices": list(used_frame_indices),
                "u": np.stack(u_rows, axis=0),
                "muscle_names": list(muscle_names),
                "mean_residual_norm": float(np.mean(residuals)) if residuals else 0.0,
                "mean_relative_torque_error": float(np.mean(relative_errors)) if relative_errors else 0.0,
                "mean_final_pose_error": float(np.mean(final_pose_errors)) if final_pose_errors else 0.0,
                "mean_pose_error": float(np.mean(mean_pose_errors)) if mean_pose_errors else 0.0,
                "num_frames_used": int(len(u_rows)),
            }
        )
    return output_trials, frame_metric_rows


def attach_static_metadata_rows(
    rows: list[dict[str, Any]],
    static_trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (str(item["exercise_name"]), str(item["trial_id"]), str(item["session_id"])): item
        for item in static_trials
    }
    enriched: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["exercise_name"]), str(row["trial_id"]), str(row["session_id"]))
        info = by_key.get(key, {})
        out = dict(row)
        out["u_source"] = "static_optimization"
        out["num_frames_used"] = int(info.get("num_frames_used", 0))
        out["mean_residual_norm"] = float(info.get("mean_residual_norm", 0.0))
        out["mean_relative_torque_error"] = float(info.get("mean_relative_torque_error", 0.0))
        out["mean_final_pose_error"] = float(info.get("mean_final_pose_error", 0.0))
        out["mean_pose_error"] = float(info.get("mean_pose_error", 0.0))
        enriched.append(out)
    return enriched


def build_validation_summary(static_trials: list[dict[str, Any]], frame_metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_trials": int(len(static_trials)),
        "num_frames": int(len(frame_metric_rows)),
    }
    if frame_metric_rows:
        summary["mechanical_validity"] = {
            "mean_residual_torque_norm": float(np.mean([row["residual_torque_norm"] for row in frame_metric_rows])),
            "mean_relative_torque_error": float(np.mean([row["relative_torque_error"] for row in frame_metric_rows])),
        }
        summary["forward_pose_validity"] = {
            "mean_final_pose_error": float(np.mean([row["final_pose_error"] for row in frame_metric_rows])),
            "mean_pose_error": float(np.mean([row["mean_pose_error"] for row in frame_metric_rows])),
        }

    per_exercise: dict[str, dict[str, list[float]]] = {}
    for row in frame_metric_rows:
        bucket = per_exercise.setdefault(
            str(row["exercise_name"]),
            {
                "residual_torque_norm": [],
                "relative_torque_error": [],
                "final_pose_error": [],
                "mean_pose_error": [],
            },
        )
        bucket["residual_torque_norm"].append(float(row["residual_torque_norm"]))
        bucket["relative_torque_error"].append(float(row["relative_torque_error"]))
        bucket["final_pose_error"].append(float(row["final_pose_error"]))
        bucket["mean_pose_error"].append(float(row["mean_pose_error"]))

    summary["per_exercise"] = {
        exercise_name: {
            "mean_residual_torque_norm": float(np.mean(values["residual_torque_norm"])) if values["residual_torque_norm"] else 0.0,
            "mean_relative_torque_error": float(np.mean(values["relative_torque_error"])) if values["relative_torque_error"] else 0.0,
            "mean_final_pose_error": float(np.mean(values["final_pose_error"])) if values["final_pose_error"] else 0.0,
            "mean_pose_error": float(np.mean(values["mean_pose_error"])) if values["mean_pose_error"] else 0.0,
            "num_frames": int(len(values["residual_torque_norm"])),
        }
        for exercise_name, values in sorted(per_exercise.items())
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Static-optimization baseline from labelled recording CSV files.")
    parser.add_argument("--records-dir", type=str, default="records", help="Directory containing labelled recording CSV files.")
    parser.add_argument("--output-dir", type=str, default="artifacts/research_u_static", help="Directory to save static baseline outputs.")
    parser.add_argument(
        "--exercise-names",
        type=str,
        default=",".join(DEFAULT_EXERCISE_NAMES),
        help="Comma-separated exercise labels to include.",
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_MYOSUITE_ENV_ID, help="MyoSuite env id used for static optimization.")
    parser.add_argument("--frame-step", type=int, default=5, help="Use every Nth frame from each trial to keep the baseline practical.")
    parser.add_argument("--max-frames-per-trial", type=int, default=40, help="Maximum number of frames to optimize per trial. Use <=0 for no cap.")
    parser.add_argument("--hold-steps", type=int, default=15, help="Simulation steps used to test hold-pose forward validity for each static solution.")
    parser.add_argument("--top-k-muscles", type=int, default=5, help="Number of dominant muscles to report per exercise.")
    args = parser.parse_args()

    records_dir = Path(args.records_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_exercises = {item.strip() for item in str(args.exercise_names).split(",") if item.strip()}
    trials = discover_labelled_qpos_trials(records_dir, allowed_exercises)
    if not trials:
        raise SystemExit("No labelled trials found. Record labelled CSVs first.")

    env = None
    try:
        env, model, data = create_probe_env(str(args.env_id))
        static_trials, frame_metric_rows = build_static_opt_trials(
            trials,
            model=model,
            data=data,
            frame_step=int(args.frame_step),
            max_frames_per_trial=int(args.max_frames_per_trial),
            hold_steps=int(args.hold_steps),
        )
        if not static_trials:
            raise SystemExit("Static optimization failed for all trials.")

        trial_rows, x, y, _, label_names = build_trial_table(static_trials)
        trial_rows = attach_static_metadata_rows(trial_rows, static_trials)
        conf, prediction_rows, metrics = run_leave_one_trial_out_classifier(x, y, trial_rows, label_names)
        profile_rows, profile_json, distance_rows = build_exercise_profiles(static_trials, top_k=int(args.top_k_muscles))
        validation_summary = build_validation_summary(static_trials, frame_metric_rows)

        trial_features_csv = output_dir / "static_trial_features.csv"
        predictions_csv = output_dir / "static_trial_predictions.csv"
        confusion_csv = output_dir / "static_confusion_matrix.csv"
        metrics_json = output_dir / "static_metrics.json"
        frame_validity_csv = output_dir / "static_frame_validity.csv"
        trial_validity_csv = output_dir / "static_trial_validity.csv"
        validation_json = output_dir / "static_validation_summary.json"
        exercise_profiles_csv = output_dir / "static_exercise_profiles.csv"
        exercise_profiles_json = output_dir / "static_exercise_profiles.json"
        exercise_distances_csv = output_dir / "static_exercise_distances.csv"

        write_trial_feature_csv(trial_rows, trial_features_csv)
        write_trial_feature_csv(prediction_rows, predictions_csv)
        write_confusion_matrix_csv(conf, label_names, confusion_csv)
        write_trial_feature_csv(frame_metric_rows, frame_validity_csv)
        write_trial_feature_csv(
            [
                {
                    "source_csv": str(item["source_csv"]),
                    "exercise_name": str(item["exercise_name"]),
                    "trial_id": str(item["trial_id"]),
                    "session_id": str(item["session_id"]),
                    "num_frames_used": int(item["num_frames_used"]),
                    "mean_residual_torque_norm": float(item["mean_residual_norm"]),
                    "mean_relative_torque_error": float(item["mean_relative_torque_error"]),
                    "mean_final_pose_error": float(item["mean_final_pose_error"]),
                    "mean_pose_error": float(item["mean_pose_error"]),
                }
                for item in static_trials
            ],
            trial_validity_csv,
        )
        write_trial_feature_csv(profile_rows, exercise_profiles_csv)
        write_trial_feature_csv(distance_rows, exercise_distances_csv)
        exercise_profiles_json.write_text(json.dumps(profile_json, indent=2, ensure_ascii=False), encoding="utf-8")
        metrics_json.write_text(
            json.dumps(
                {
                    "u_source": "static_optimization",
                    "env_id": str(args.env_id),
                    "frame_step": int(args.frame_step),
                    "max_frames_per_trial": int(args.max_frames_per_trial),
                    "hold_steps": int(args.hold_steps),
                    "labels": label_names,
                    "confusion_matrix": conf.astype(int).tolist(),
                    **metrics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        validation_json.write_text(
            json.dumps(
                {
                    "u_source": "static_optimization",
                    "env_id": str(args.env_id),
                    "frame_step": int(args.frame_step),
                    "max_frames_per_trial": int(args.max_frames_per_trial),
                    "hold_steps": int(args.hold_steps),
                    **validation_summary,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"[INFO] Trials loaded: {len(static_trials)}")
        print(f"[INFO] Labels: {label_names}")
        print(f"[INFO] Accuracy: {metrics['accuracy']:.4f}")
        if frame_metric_rows:
            print(f"[INFO] Mean residual torque norm: {validation_summary['mechanical_validity']['mean_residual_torque_norm']:.6f}")
            print(f"[INFO] Mean relative torque error: {validation_summary['mechanical_validity']['mean_relative_torque_error']:.6f}")
            print(f"[INFO] Mean final pose error: {validation_summary['forward_pose_validity']['mean_final_pose_error']:.6f}")
            print(f"[INFO] Mean pose error: {validation_summary['forward_pose_validity']['mean_pose_error']:.6f}")
        print(f"[INFO] Wrote: {trial_features_csv}")
        print(f"[INFO] Wrote: {predictions_csv}")
        print(f"[INFO] Wrote: {confusion_csv}")
        print(f"[INFO] Wrote: {metrics_json}")
        print(f"[INFO] Wrote: {frame_validity_csv}")
        print(f"[INFO] Wrote: {trial_validity_csv}")
        print(f"[INFO] Wrote: {validation_json}")
        print(f"[INFO] Wrote: {exercise_profiles_csv}")
        print(f"[INFO] Wrote: {exercise_profiles_json}")
        print(f"[INFO] Wrote: {exercise_distances_csv}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
