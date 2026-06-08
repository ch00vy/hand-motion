from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gcn_node_classifier import (  # noqa: E402
    GCNNodeClassifier,
    PRIMARY_NODE_CLASSES,
    build_samples_from_dataset,
    compute_class_weights,
    evaluate_prediction_table,
    fit_standardizer,
    predict_samples,
    run_epoch,
    save_gcn_outputs,
    set_seed,
    split_trial_samples,
    standardize_samples,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "offline_biomech_gcn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline GCN node classifier from biomechanical graph-sequence trials.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Graph-sequence dataset directory.")
    parser.add_argument("--reference-dir", type=str, required=True, help="Reference sequence directory.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for the GCN classifier.")
    parser.add_argument("--cohorts", type=str, nargs="*", default=["normal", "abnormal"], help="Cohorts to include from the dataset manifest.")
    parser.add_argument("--onset-ratio", type=float, default=0.20, help="Relative threshold for onset estimation during label construction.")
    parser.add_argument("--time-region-width", type=int, default=3, help="Window width for compare-to-reference summary.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width of the GCN.")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout inside the GCN.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--train-ratio", type=float, default=0.60, help="Trial-level train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Trial-level validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    cohorts = {str(item).strip().lower() for item in list(args.cohorts or []) if str(item).strip()}
    samples, manifest_df = build_samples_from_dataset(
        Path(args.dataset_dir),
        Path(args.reference_dir),
        cohorts=cohorts if cohorts else None,
        onset_ratio=float(args.onset_ratio),
        time_region_width=int(args.time_region_width),
    )
    if not samples:
        raise SystemExit("No graph samples available for GCN training.")

    splits = split_trial_samples(
        samples,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
    )
    if not splits["train"]:
        raise SystemExit("Empty train split.")

    feature_mean, feature_std = fit_standardizer(splits["train"])
    train_samples = standardize_samples(splits["train"], feature_mean, feature_std)
    val_samples = standardize_samples(splits["val"], feature_mean, feature_std) if splits["val"] else []
    test_samples = standardize_samples(splits["test"], feature_mean, feature_std) if splits["test"] else []
    feature_names = list(train_samples[0].feature_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNNodeClassifier(
        in_dim=int(train_samples[0].x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_classes=int(len(PRIMARY_NODE_CLASSES)),
        dropout=float(args.dropout),
    ).to(device)
    class_weights = compute_class_weights(train_samples, num_classes=len(PRIMARY_NODE_CLASSES)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_val_macro = -1.0
    best_epoch = -1
    for epoch_idx in range(int(args.epochs)):
        train_stats = run_epoch(model, train_samples, optimizer=optimizer, criterion=criterion, device=device)
        if val_samples:
            val_pred_df = predict_samples(model, val_samples, device=device)
            val_metrics = evaluate_prediction_table(val_pred_df)
            val_macro = float(val_metrics["macro_f1"])
        else:
            val_metrics = {"accuracy": train_stats["accuracy"], "macro_f1": train_stats["accuracy"]}
            val_macro = float(val_metrics["macro_f1"])
        if val_macro >= best_val_macro:
            best_val_macro = val_macro
            best_epoch = int(epoch_idx)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch_idx in {0, int(args.epochs) - 1} or ((epoch_idx + 1) % 20) == 0:
            print(
                f"[GCN] epoch={epoch_idx + 1}/{int(args.epochs)} "
                f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} "
                f"val_macro_f1={val_macro:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_pred_df = predict_samples(model, train_samples, device=device)
    val_pred_df = predict_samples(model, val_samples, device=device) if val_samples else train_pred_df.iloc[0:0].copy()
    test_pred_df = predict_samples(model, test_samples, device=device) if test_samples else train_pred_df.iloc[0:0].copy()

    metrics = {
        "best_epoch": int(best_epoch),
        "train": evaluate_prediction_table(train_pred_df),
        "val": evaluate_prediction_table(val_pred_df),
        "test": evaluate_prediction_table(test_pred_df),
        "num_train_trials": int(len(train_samples)),
        "num_val_trials": int(len(val_samples)),
        "num_test_trials": int(len(test_samples)),
        "device": str(device),
    }
    outputs = save_gcn_outputs(
        Path(args.output_dir),
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=feature_names,
        sample_manifest_df=manifest_df,
        split_tables={
            "train": train_pred_df,
            "val": val_pred_df,
            "test": test_pred_df,
        },
        metrics=metrics,
    )
    print(f"[INFO] GCN summary: {outputs['summary_json_path']}")
    print(f"[INFO] GCN report: {outputs['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
