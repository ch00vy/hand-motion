from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

try:
    import torch
except ModuleNotFoundError:
    torch = None


DEFAULT_EXERCISE_NAMES = ("fist", "claw_stretch", "making_o")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_DIR = PROJECT_ROOT / "artifacts" / "graphs" / "sequences"


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


def _load_muscle_names_from_sequence(session_id: str, num_muscles: int) -> list[str]:
    if torch is None:
        return [f"muscle_{idx}" for idx in range(num_muscles)]
    pt_path = DEFAULT_SEQUENCE_DIR / f"exercise_{session_id}.pt"
    if not pt_path.exists():
        return [f"muscle_{idx}" for idx in range(num_muscles)]
    try:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        node_names = list(payload.get("node_names") or [])
        if len(node_names) == num_muscles:
            return [str(name) for name in node_names]
    except Exception:
        pass
    return [f"muscle_{idx}" for idx in range(num_muscles)]


def load_labelled_trial_csv(path: Path, allowed_exercises: set[str]) -> dict[str, Any] | None:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        field_names = list(reader.fieldnames or [])
        if not field_names:
            return None

        ctrl_columns = _indexed_columns(field_names, "ctrl_analysis_")
        if not ctrl_columns:
            return None

        exercise_values: list[str] = []
        trial_values: list[str] = []
        session_values: list[str] = []
        frame_indices: list[int] = []
        u_rows: list[list[float]] = []

        for row_idx, row in enumerate(reader):
            exercise_values.append(str(row.get("exercise_name") or ""))
            trial_values.append(str(row.get("trial_id") or ""))
            session_values.append(str(row.get("session_id") or ""))
            frame_indices.append(int(float(row.get("frame_index") or row.get("frame") or row_idx)))
            u_rows.append([float(row.get(name) or 0.0) for name in ctrl_columns])

    exercise_name = _unique_nonempty(exercise_values)
    trial_id = _unique_nonempty(trial_values)
    session_id = _unique_nonempty(session_values)
    if exercise_name is None or trial_id is None or session_id is None:
        return None
    if exercise_name not in allowed_exercises:
        return None

    u = np.asarray(u_rows, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] == 0:
        return None

    return {
        "source_csv": str(path),
        "exercise_name": exercise_name,
        "trial_id": trial_id,
        "session_id": session_id,
        "frame_indices": frame_indices,
        "u": u,
        "muscle_names": _load_muscle_names_from_sequence(session_id, int(u.shape[1])),
    }


def discover_trials(records_dir: Path, allowed_exercises: set[str]) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("*.csv")):
        try:
            item = load_labelled_trial_csv(path, allowed_exercises)
        except Exception as exc:
            print(f"[WARN] Skipping {path.name}: {type(exc).__name__}: {exc}")
            continue
        if item is not None:
            trials.append(item)
    return trials


def compute_trial_features(u: np.ndarray) -> tuple[np.ndarray, list[str]]:
    mean_vec = np.mean(u, axis=0)
    std_vec = np.std(u, axis=0)
    max_vec = np.max(u, axis=0)

    feature_vec = np.concatenate([mean_vec, std_vec, max_vec], axis=0).astype(np.float64, copy=False)
    num_muscles = int(u.shape[1])
    feature_names = (
        [f"mean_m{idx}" for idx in range(num_muscles)]
        + [f"std_m{idx}" for idx in range(num_muscles)]
        + [f"max_m{idx}" for idx in range(num_muscles)]
    )
    return feature_vec, feature_names


def build_trial_table(trials: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[str], list[str]]:
    if not trials:
        raise ValueError("No labelled trials found.")

    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    labels: list[str] = []
    feature_names: list[str] | None = None

    for trial in trials:
        feature_vec, current_feature_names = compute_trial_features(np.asarray(trial["u"], dtype=np.float64))
        if feature_names is None:
            feature_names = list(current_feature_names)
        row = {
            "exercise_name": str(trial["exercise_name"]),
            "trial_id": str(trial["trial_id"]),
            "session_id": str(trial["session_id"]),
            "source_csv": str(trial["source_csv"]),
            "num_frames": int(np.asarray(trial["u"]).shape[0]),
        }
        for name, value in zip(current_feature_names, feature_vec.tolist()):
            row[name] = float(value)
        rows.append(row)
        features.append(feature_vec)
        labels.append(str(trial["exercise_name"]))

    label_names = sorted(set(labels))
    label_to_index = {name: idx for idx, name in enumerate(label_names)}
    x = np.stack(features, axis=0)
    y = np.asarray([label_to_index[name] for name in labels], dtype=np.int64)
    return rows, x, y, feature_names or [], label_names


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((int(y.shape[0]), int(num_classes)), dtype=np.float64)
    out[np.arange(int(y.shape[0])), y.astype(np.int64)] = 1.0
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    num_classes: int,
    reg_strength: float = 1e-2,
    max_iter: int = 400,
) -> np.ndarray:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float64)], axis=1)
    y_one_hot = _one_hot(y_train, num_classes)
    num_features = int(x_aug.shape[1])

    def objective(flat_w: np.ndarray) -> tuple[float, np.ndarray]:
        w = flat_w.reshape(num_features, num_classes)
        logits = x_aug @ w
        probs = _softmax(logits)
        loss = -np.sum(y_one_hot * np.log(np.clip(probs, 1e-12, 1.0))) / x_aug.shape[0]
        loss += 0.5 * float(reg_strength) * np.sum(w[:-1] * w[:-1])
        grad = (x_aug.T @ (probs - y_one_hot)) / x_aug.shape[0]
        grad[:-1] += float(reg_strength) * w[:-1]
        return float(loss), grad.reshape(-1)

    init = np.zeros((num_features, num_classes), dtype=np.float64).reshape(-1)
    result = minimize(
        fun=lambda flat_w: objective(flat_w)[0],
        x0=init,
        jac=lambda flat_w: objective(flat_w)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter)},
    )
    if not result.success:
        print(f"[WARN] Optimizer status: {result.message}")
    return np.asarray(result.x, dtype=np.float64).reshape(num_features, num_classes)


def predict_logistic_regression(weights: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x_test, np.ones((x_test.shape[0], 1), dtype=np.float64)], axis=1)
    logits = x_aug @ weights
    return np.argmax(logits, axis=1).astype(np.int64)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true.tolist(), y_pred.tolist()):
        conf[int(true_idx), int(pred_idx)] += 1
    return conf


def compute_metrics(conf: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    total = int(np.sum(conf))
    accuracy = float(np.trace(conf) / total) if total > 0 else 0.0
    per_class: dict[str, Any] = {}
    for idx, label in enumerate(label_names):
        tp = int(conf[idx, idx])
        fp = int(np.sum(conf[:, idx]) - tp)
        fn = int(np.sum(conf[idx, :]) - tp)
        support = int(np.sum(conf[idx, :]))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": accuracy,
        "num_trials": total,
        "per_class": per_class,
    }


def run_leave_one_trial_out_classifier(
    x: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, Any]],
    label_names: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    num_classes = len(label_names)
    if x.shape[0] < max(3, num_classes):
        raise ValueError("Not enough labelled trials for leave-one-trial-out evaluation.")

    class_counts = {label_names[idx]: int(np.sum(y == idx)) for idx in range(num_classes)}
    too_small = [name for name, count in class_counts.items() if count < 2]
    if too_small:
        raise ValueError(f"Each class needs at least 2 trials for leave-one-trial-out. Too small: {too_small}")

    predictions: list[int] = []
    prediction_rows: list[dict[str, Any]] = []

    for held_out_idx in range(x.shape[0]):
        train_mask = np.ones(x.shape[0], dtype=bool)
        train_mask[held_out_idx] = False
        x_train = x[train_mask]
        y_train = y[train_mask]
        x_test = x[~train_mask]
        y_test = y[~train_mask]

        x_train_std, x_test_std = _standardize_train_test(x_train, x_test)
        weights = fit_logistic_regression(x_train_std, y_train, num_classes=num_classes)
        pred = predict_logistic_regression(weights, x_test_std)
        pred_idx = int(pred[0])
        true_idx = int(y_test[0])
        predictions.append(pred_idx)

        trial_row = dict(rows[held_out_idx])
        trial_row["predicted_label"] = str(label_names[pred_idx])
        trial_row["true_label"] = str(label_names[true_idx])
        trial_row["correct"] = int(pred_idx == true_idx)
        prediction_rows.append(trial_row)

    y_pred = np.asarray(predictions, dtype=np.int64)
    conf = compute_confusion_matrix(y, y_pred, num_classes=num_classes)
    metrics = compute_metrics(conf, label_names)
    metrics["evaluation"] = "leave_one_trial_out"
    metrics["class_counts"] = class_counts
    return conf, prediction_rows, metrics


def write_trial_feature_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    field_names = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix_csv(conf: np.ndarray, label_names: list[str], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["actual/predicted"] + list(label_names))
        for idx, label in enumerate(label_names):
            writer.writerow([label] + conf[idx].astype(int).tolist())


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def build_exercise_profiles(trials: list[dict[str, Any]], *, top_k: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not trials:
        return [], {}, []

    muscle_names = [str(name) for name in list(trials[0].get("muscle_names") or [])]
    profiles_csv_rows: list[dict[str, Any]] = []
    profiles_json: dict[str, Any] = {}
    mean_profiles: dict[str, np.ndarray] = {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        grouped.setdefault(str(trial["exercise_name"]), []).append(trial)

    for exercise_name in sorted(grouped.keys()):
        exercise_trials = grouped[exercise_name]
        u_all = np.concatenate([np.asarray(trial["u"], dtype=np.float64) for trial in exercise_trials], axis=0)
        mean_profile = np.mean(u_all, axis=0)
        std_profile = np.std(u_all, axis=0)
        mean_profiles[exercise_name] = mean_profile.astype(np.float64, copy=False)

        dominant_indices = np.argsort(mean_profile)[-max(1, int(top_k)) :][::-1]
        dominant_muscles = [
            {
                "muscle_name": muscle_names[idx] if idx < len(muscle_names) else f"muscle_{idx}",
                "mean_activation": float(mean_profile[idx]),
                "std_activation": float(std_profile[idx]),
            }
            for idx in dominant_indices.tolist()
        ]

        csv_row = {
            "exercise_name": exercise_name,
            "num_trials": int(len(exercise_trials)),
            "num_frames": int(u_all.shape[0]),
        }
        for idx, muscle_name in enumerate(muscle_names):
            csv_row[f"mean_{muscle_name}"] = float(mean_profile[idx])
        profiles_csv_rows.append(csv_row)

        profiles_json[exercise_name] = {
            "exercise_name": exercise_name,
            "num_trials": int(len(exercise_trials)),
            "num_frames": int(u_all.shape[0]),
            "dominant_muscles": dominant_muscles,
            "mean_activation_profile": {
                muscle_names[idx] if idx < len(muscle_names) else f"muscle_{idx}": float(mean_profile[idx])
                for idx in range(len(mean_profile))
            },
        }

    distance_rows: list[dict[str, Any]] = []
    exercise_names = sorted(mean_profiles.keys())
    for idx, source_name in enumerate(exercise_names):
        for jdx in range(idx + 1, len(exercise_names)):
            target_name = exercise_names[jdx]
            source_vec = mean_profiles[source_name]
            target_vec = mean_profiles[target_name]
            distance_rows.append(
                {
                    "exercise_a": source_name,
                    "exercise_b": target_name,
                    "euclidean_distance": float(np.linalg.norm(source_vec - target_vec)),
                    "cosine_similarity": float(_cosine_similarity(source_vec, target_vec)),
                }
            )

    return profiles_csv_rows, profiles_json, distance_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline analysis for labelled ctrl_analysis trials.")
    parser.add_argument("--records-dir", type=str, default="records", help="Directory containing labelled recording CSV files.")
    parser.add_argument("--output-dir", type=str, default="artifacts/research_u", help="Directory to save analysis outputs.")
    parser.add_argument(
        "--exercise-names",
        type=str,
        default=",".join(DEFAULT_EXERCISE_NAMES),
        help="Comma-separated exercise labels to include.",
    )
    parser.add_argument("--top-k-muscles", type=int, default=5, help="Number of dominant muscles to report per exercise.")
    args = parser.parse_args()

    records_dir = Path(args.records_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_exercises = {item.strip() for item in str(args.exercise_names).split(",") if item.strip()}
    trials = discover_trials(records_dir, allowed_exercises)
    if not trials:
        raise SystemExit("No labelled trials found. Record labelled CSVs first.")

    trial_rows, x, y, _, label_names = build_trial_table(trials)
    conf, prediction_rows, metrics = run_leave_one_trial_out_classifier(x, y, trial_rows, label_names)

    trial_features_csv = output_dir / "trial_features.csv"
    predictions_csv = output_dir / "trial_predictions.csv"
    confusion_csv = output_dir / "confusion_matrix.csv"
    metrics_json = output_dir / "metrics.json"
    exercise_profiles_csv = output_dir / "exercise_profiles.csv"
    exercise_profiles_json = output_dir / "exercise_profiles.json"
    exercise_distances_csv = output_dir / "exercise_distances.csv"

    write_trial_feature_csv(trial_rows, trial_features_csv)
    write_trial_feature_csv(prediction_rows, predictions_csv)
    write_confusion_matrix_csv(conf, label_names, confusion_csv)
    profile_rows, profile_json, distance_rows = build_exercise_profiles(trials, top_k=int(args.top_k_muscles))
    write_trial_feature_csv(profile_rows, exercise_profiles_csv)
    write_trial_feature_csv(distance_rows, exercise_distances_csv)
    exercise_profiles_json.write_text(json.dumps(profile_json, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_json.write_text(
        json.dumps(
            {
                "labels": label_names,
                "confusion_matrix": conf.astype(int).tolist(),
                **metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Trials loaded: {len(trials)}")
    print(f"[INFO] Labels: {label_names}")
    print(f"[INFO] Accuracy: {metrics['accuracy']:.4f}")
    print(f"[INFO] Wrote: {trial_features_csv}")
    print(f"[INFO] Wrote: {predictions_csv}")
    print(f"[INFO] Wrote: {confusion_csv}")
    print(f"[INFO] Wrote: {metrics_json}")
    print(f"[INFO] Wrote: {exercise_profiles_csv}")
    print(f"[INFO] Wrote: {exercise_profiles_json}")
    print(f"[INFO] Wrote: {exercise_distances_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
