from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_friends_style_binary as friends_binary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run friends-style binary LOSO CNN for a chosen number of epochs.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of CNN epochs per LOSO fold (default: 5).")
    args = parser.parse_args()
    cnn_epochs = int(args.epochs)

    workspace_root = Path(__file__).resolve().parent
    outputs_dir = workspace_root / friends_binary.OUTPUT_DIR_NAME
    outputs_dir.mkdir(exist_ok=True)

    rows = friends_binary.build_window_rows(workspace_root)
    data_frame = pd.DataFrame(rows)

    friends_binary.CNN_EPOCHS = cnn_epochs
    cnn_result = friends_binary.evaluate_cnn(data_frame)

    metrics_path = outputs_dir / f"{friends_binary.FRIENDS_PREFIX}_cnn_metrics.json"
    predictions_path = outputs_dir / f"{friends_binary.FRIENDS_PREFIX}_cnn_predictions.csv"
    confusion_csv_path = outputs_dir / f"{friends_binary.FRIENDS_PREFIX}_cnn_confusion_matrix.csv"
    confusion_png_path = outputs_dir / f"{friends_binary.FRIENDS_PREFIX}_cnn_confusion_matrix.png"
    history_path = outputs_dir / f"{friends_binary.FRIENDS_PREFIX}_cnn_training_history.csv"
    summary_path = outputs_dir / f"{friends_binary.FRIENDS_PREFIX}_summary.json"

    metrics_path.write_text(json.dumps(cnn_result["metrics"], indent=2), encoding="utf-8")
    cnn_result["predictions"].to_csv(predictions_path, index=False)
    matrix = np.asarray(cnn_result["metrics"]["confusion_matrix"], dtype=np.int64)
    friends_binary.write_confusion_matrix_csv(matrix, confusion_csv_path)
    friends_binary.write_confusion_matrix_svg(matrix, confusion_png_path, title="CNN LOSO Confusion Matrix")
    if "history" in cnn_result:
        cnn_result["history"].to_csv(history_path, index=False)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}

    summary["pipeline"] = {
        "filter": {"low_hz": friends_binary.FILTER_LOW, "high_hz": friends_binary.FILTER_HIGH, "order": friends_binary.FILTER_ORDER},
        "window_seconds": friends_binary.WINDOW_SECONDS,
        "step_seconds": friends_binary.STEP_SECONDS,
        "normalization": "per-subject z-score on filtered BVP",
        "mixed_windows": "discarded",
        "evaluation": "Leave-One-Subject-Out (LOSO)",
        "feature_count": 24,
        "cnn_epochs": cnn_epochs,
    }
    summary["dataset"] = {
        "num_windows": int(len(data_frame)),
        "num_subjects": int(data_frame["subject"].nunique()),
        "class_counts": data_frame["target_label"].value_counts().to_dict(),
        "source_label_counts": data_frame["source_label"].value_counts().to_dict(),
    }
    summary["cnn"] = cnn_result["metrics"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Completed friends-style binary CNN LOSO run with {cnn_epochs} epochs.")
    print(f"Accuracy: {cnn_result['metrics']['accuracy']:.4f}")
    print(f"Macro F1: {cnn_result['metrics']['macro_f1']:.4f}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
