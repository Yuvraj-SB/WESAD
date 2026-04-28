from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "range",
    "energy",
    "rms",
    "skewness",
    "kurtosis",
]

SVM_EPOCHS = 50


def standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def train_binary_linear_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = SVM_EPOCHS,
    learning_rate: float = 0.01,
    regularization: float = 0.001,
) -> tuple[np.ndarray, float]:
    num_samples, num_features = x_train.shape
    weights = np.zeros(num_features, dtype=np.float64)
    bias = 0.0
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        indices = rng.permutation(num_samples)
        epoch_learning_rate = learning_rate / (1.0 + 0.05 * epoch)

        for index in indices:
            x_i = x_train[index]
            y_i = y_train[index]
            margin = y_i * (np.dot(weights, x_i) + bias)

            weights *= 1.0 - epoch_learning_rate * regularization
            if margin < 1.0:
                weights += epoch_learning_rate * y_i * x_i
                bias += epoch_learning_rate * y_i

    return weights, bias


def train_one_vs_rest_svm(
    x_train: np.ndarray, y_train_labels: np.ndarray, class_labels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    weight_rows: list[np.ndarray] = []
    bias_values: list[float] = []

    for class_name in class_labels:
        binary_targets = np.where(y_train_labels == class_name, 1.0, -1.0)
        weights, bias = train_binary_linear_svm(x_train, binary_targets)
        weight_rows.append(weights)
        bias_values.append(bias)

    return np.vstack(weight_rows), np.asarray(bias_values, dtype=np.float64)


def predict_one_vs_rest(
    x_values: np.ndarray, weights: np.ndarray, biases: np.ndarray, class_labels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    scores = x_values @ weights.T + biases
    predicted_indices = np.argmax(scores, axis=1)
    predicted_labels = np.asarray([class_labels[index] for index in predicted_indices])
    return predicted_labels, scores


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str]
) -> tuple[np.ndarray, dict[str, int]]:
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_index[str(true_label)], label_to_index[str(pred_label)]] += 1

    return matrix, label_to_index


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str]
) -> tuple[dict[str, object], np.ndarray]:
    confusion_matrix, label_to_index = compute_confusion_matrix(y_true, y_pred, class_labels)
    total = int(confusion_matrix.sum())
    correct = int(np.trace(confusion_matrix))
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []

    for label, index in label_to_index.items():
        tp = int(confusion_matrix[index, index])
        fp = int(confusion_matrix[:, index].sum() - tp)
        fn = int(confusion_matrix[index, :].sum() - tp)
        support = int(confusion_matrix[index, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1_scores.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    metrics = {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_order": class_labels,
        "num_test_samples": total,
    }
    return metrics, confusion_matrix


def write_confusion_matrix_csv(
    matrix: np.ndarray, class_labels: list[str], output_path: Path
) -> None:
    rows = []
    for row_label, row_values in zip(class_labels, matrix):
        row = {"true_label": row_label}
        for column_label, value in zip(class_labels, row_values):
            row[column_label] = int(value)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_confusion_matrix_svg(
    matrix: np.ndarray, class_labels: list[str], output_path: Path, title: str
) -> None:
    cell_size = 90
    width = 240 + cell_size * len(class_labels)
    height = 220 + cell_size * len(class_labels)
    left_margin = 150
    top_margin = 100
    max_value = int(matrix.max()) if matrix.size else 1

    def cell_color(value: int) -> str:
        intensity = value / max_value if max_value else 0.0
        blue = int(255 - 140 * intensity)
        red = int(245 - 120 * intensity)
        green = int(248 - 140 * intensity)
        return f"rgb({red},{green},{blue})"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="Arial" font-size="24">{title}</text>',
        f'<text x="{width / 2}" y="70" text-anchor="middle" font-family="Arial" font-size="14">Predicted label</text>',
        f'<text x="34" y="{height / 2}" transform="rotate(-90 34 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">True label</text>',
    ]

    for column_index, label in enumerate(class_labels):
        x = left_margin + column_index * cell_size + cell_size / 2
        lines.append(
            f'<text x="{x}" y="{top_margin - 18}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>'
        )

    for row_index, label in enumerate(class_labels):
        y = top_margin + row_index * cell_size + cell_size / 2 + 5
        lines.append(
            f'<text x="{left_margin - 16}" y="{y}" text-anchor="end" font-family="Arial" font-size="13">{label}</text>'
        )

    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            x = left_margin + column_index * cell_size
            y = top_margin + row_index * cell_size
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{cell_color(int(value))}" stroke="#666" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2 + 6}" text-anchor="middle" font-family="Arial" font-size="18">{int(value)}</text>'
            )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_dataset_variant(
    data_frame: pd.DataFrame, output_path: Path
) -> None:
    data_frame.to_csv(output_path, index=False)


def run_variant(
    variant_name: str,
    data_frame: pd.DataFrame,
    class_labels: list[str],
    outputs_dir: Path,
    train_subjects: set[str],
    test_subjects: set[str],
) -> dict[str, object]:
    train_frame = data_frame[data_frame["subject"].isin(train_subjects)].copy()
    test_frame = data_frame[data_frame["subject"].isin(test_subjects)].copy()

    x_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    x_test = test_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_train = train_frame["target_label"].to_numpy(dtype=str)
    y_test = test_frame["target_label"].to_numpy(dtype=str)

    x_train_scaled, x_test_scaled, mean, std = standardize_train_test(x_train, x_test)
    weights, biases = train_one_vs_rest_svm(x_train_scaled, y_train, class_labels)
    y_pred, scores = predict_one_vs_rest(x_test_scaled, weights, biases, class_labels)
    metrics, confusion_matrix = compute_metrics(y_test, y_pred, class_labels)

    prefix = f"svm_{variant_name}"
    metrics_output_path = outputs_dir / f"{prefix}_metrics.json"
    confusion_csv_path = outputs_dir / f"{prefix}_confusion_matrix.csv"
    confusion_svg_path = outputs_dir / f"{prefix}_confusion_matrix.svg"
    predictions_output_path = outputs_dir / f"{prefix}_test_predictions.csv"
    model_info_output_path = outputs_dir / f"{prefix}_model_info.json"
    dataset_output_path = outputs_dir / f"{prefix}_dataset.csv"

    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_confusion_matrix_csv(confusion_matrix, class_labels, confusion_csv_path)
    write_confusion_matrix_svg(
        confusion_matrix,
        class_labels,
        confusion_svg_path,
        title=f"Linear SVM Confusion Matrix ({variant_name})",
    )

    predictions_frame = test_frame[
        [
            "subject",
            "segment_index",
            "target_label",
            "window_start_seconds",
            "window_end_seconds",
        ]
    ].copy()
    predictions_frame["predicted_label"] = y_pred
    for class_index, class_name in enumerate(class_labels):
        predictions_frame[f"score_{class_name}"] = scores[:, class_index]
    predictions_frame.to_csv(predictions_output_path, index=False)

    save_dataset_variant(data_frame, dataset_output_path)

    model_info = {
        "variant": variant_name,
        "model_type": "one-vs-rest linear SVM trained with NumPy SGD",
        "feature_columns": FEATURE_COLUMNS,
        "class_order": class_labels,
        "train_subjects": sorted(train_subjects, key=lambda value: int(value[1:])),
        "test_subjects": sorted(test_subjects, key=lambda value: int(value[1:])),
        "num_train_samples": int(len(train_frame)),
        "num_test_samples": int(len(test_frame)),
        "standardization_mean": mean.tolist(),
        "standardization_std": std.tolist(),
        "weights": weights.tolist(),
        "biases": biases.tolist(),
    }
    model_info_output_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    return {
        "variant": variant_name,
        "metrics_path": str(metrics_output_path),
        "confusion_svg_path": str(confusion_svg_path),
        "dataset_path": str(dataset_output_path),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "num_train_samples": len(train_frame),
        "num_test_samples": len(test_frame),
    }


def main() -> None:
    workspace_root = Path(__file__).resolve().parent
    outputs_dir = workspace_root / "outputs"

    features_path = outputs_dir / "all_subjects_window_features.csv"
    split_path = outputs_dir / "all_subjects_split.json"

    data_frame = pd.read_csv(features_path)
    split_config = json.loads(split_path.read_text(encoding="utf-8"))

    train_subjects = set(split_config["train_subjects"])
    test_subjects = set(split_config["test_subjects"])

    three_class_frame = data_frame[
        data_frame["label_name"].isin(["baseline", "stress", "amusement"])
    ].copy()
    three_class_frame["target_label"] = three_class_frame["label_name"]

    binary_frame = three_class_frame.copy()
    binary_frame["target_label"] = np.where(
        binary_frame["label_name"] == "stress", "stress", "non-stress"
    )

    summaries = []
    summaries.append(
        run_variant(
            variant_name="3class",
            data_frame=three_class_frame,
            class_labels=["baseline", "stress", "amusement"],
            outputs_dir=outputs_dir,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
        )
    )
    summaries.append(
        run_variant(
            variant_name="binary",
            data_frame=binary_frame,
            class_labels=["non-stress", "stress"],
            outputs_dir=outputs_dir,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
        )
    )

    summary_output_path = outputs_dir / "svm_variants_summary.json"
    summary_output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("Completed SVM training for 3-class and binary variants.")
    for summary in summaries:
        print(
            f"{summary['variant']}: accuracy={summary['accuracy']:.4f}, "
            f"macro_f1={summary['macro_f1']:.4f}, "
            f"train={summary['num_train_samples']}, test={summary['num_test_samples']}"
        )
    print(f"Saved summary to: {summary_output_path}")


if __name__ == "__main__":
    main()
