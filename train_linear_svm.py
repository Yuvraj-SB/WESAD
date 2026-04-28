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

CLASS_LABELS = ["baseline", "stress", "amusement", "meditation"]


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
    epochs: int = 60,
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
    x_train: np.ndarray, y_train_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weight_rows: list[np.ndarray] = []
    bias_values: list[float] = []

    for class_name in CLASS_LABELS:
        binary_targets = np.where(y_train_labels == class_name, 1.0, -1.0)
        weights, bias = train_binary_linear_svm(x_train, binary_targets)
        weight_rows.append(weights)
        bias_values.append(bias)

    return np.vstack(weight_rows), np.asarray(bias_values, dtype=np.float64)


def predict_one_vs_rest(
    x_values: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = x_values @ weights.T + biases
    predicted_indices = np.argmax(scores, axis=1)
    predicted_labels = np.asarray([CLASS_LABELS[index] for index in predicted_indices])
    return predicted_labels, scores


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    matrix = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_index[str(true_label)], label_to_index[str(pred_label)]] += 1

    return matrix, label_to_index


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    confusion_matrix, label_to_index = compute_confusion_matrix(y_true, y_pred)
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

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_order": CLASS_LABELS,
        "num_test_samples": total,
    }


def write_confusion_matrix_csv(
    matrix: np.ndarray, output_path: Path
) -> None:
    rows = []
    for row_label, row_values in zip(CLASS_LABELS, matrix):
        row = {"true_label": row_label}
        for column_label, value in zip(CLASS_LABELS, row_values):
            row[column_label] = int(value)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_confusion_matrix_svg(
    matrix: np.ndarray, output_path: Path, title: str
) -> None:
    width = 780
    height = 520
    left_margin = 150
    top_margin = 100
    cell_size = 90
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

    for column_index, label in enumerate(CLASS_LABELS):
        x = left_margin + column_index * cell_size + cell_size / 2
        lines.append(
            f'<text x="{x}" y="{top_margin - 18}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>'
        )

    for row_index, label in enumerate(CLASS_LABELS):
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


def main() -> None:
    workspace_root = Path(__file__).resolve().parent
    outputs_dir = workspace_root / "outputs"

    features_path = outputs_dir / "all_subjects_window_features.csv"
    split_path = outputs_dir / "all_subjects_split.json"

    data_frame = pd.read_csv(features_path)
    split_config = json.loads(split_path.read_text(encoding="utf-8"))

    train_subjects = set(split_config["train_subjects"])
    test_subjects = set(split_config["test_subjects"])

    train_frame = data_frame[data_frame["subject"].isin(train_subjects)].copy()
    test_frame = data_frame[data_frame["subject"].isin(test_subjects)].copy()

    x_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    x_test = test_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_train = train_frame["label_name"].to_numpy(dtype=str)
    y_test = test_frame["label_name"].to_numpy(dtype=str)

    x_train_scaled, x_test_scaled, mean, std = standardize_train_test(x_train, x_test)
    weights, biases = train_one_vs_rest_svm(x_train_scaled, y_train)
    y_pred, scores = predict_one_vs_rest(x_test_scaled, weights, biases)

    metrics = compute_metrics(y_test, y_pred)
    confusion_matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    metrics_output_path = outputs_dir / "svm_metrics.json"
    confusion_csv_path = outputs_dir / "svm_confusion_matrix.csv"
    confusion_svg_path = outputs_dir / "svm_confusion_matrix.svg"
    predictions_output_path = outputs_dir / "svm_test_predictions.csv"
    model_info_output_path = outputs_dir / "svm_model_info.json"

    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_confusion_matrix_csv(confusion_matrix, confusion_csv_path)
    write_confusion_matrix_svg(
        confusion_matrix,
        confusion_svg_path,
        title="Linear SVM Confusion Matrix",
    )

    predictions_frame = test_frame[
        [
            "subject",
            "segment_index",
            "label_name",
            "window_start_seconds",
            "window_end_seconds",
        ]
    ].copy()
    predictions_frame["predicted_label"] = y_pred
    for class_index, class_name in enumerate(CLASS_LABELS):
        predictions_frame[f"score_{class_name}"] = scores[:, class_index]
    predictions_frame.to_csv(predictions_output_path, index=False)

    model_info = {
        "model_type": "one-vs-rest linear SVM trained with NumPy SGD",
        "feature_columns": FEATURE_COLUMNS,
        "class_order": CLASS_LABELS,
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

    print("Trained one-vs-rest linear SVM on all-subject features.")
    print(f"Train samples: {len(train_frame)}")
    print(f"Test samples: {len(test_frame)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Saved metrics to: {metrics_output_path}")
    print(f"Saved confusion matrix CSV to: {confusion_csv_path}")
    print(f"Saved confusion matrix SVG to: {confusion_svg_path}")
    print(f"Saved test predictions to: {predictions_output_path}")
    print(f"Saved model info to: {model_info_output_path}")


if __name__ == "__main__":
    main()
