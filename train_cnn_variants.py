from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from process_s2_pipeline import (
    apply_butterworth_bandpass_fft,
    extract_valid_segments,
    load_subject_pickle,
    map_labels_to_bvp_timeline,
)


VALID_LABELS = {"baseline", "stress", "amusement", "meditation"}
WINDOW_SECONDS = 20
STEP_SECONDS = 10
BVP_SAMPLING_RATE = 64
LABEL_SAMPLING_RATE = 700
RANDOM_SEED = 42
CNN_EPOCHS = 50


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_subject_ids(workspace_root: Path) -> list[str]:
    subject_ids = []
    for path in sorted(
        workspace_root.iterdir(),
        key=lambda item: (
            0,
            int(item.name[1:]),
        )
        if item.is_dir() and item.name.startswith("S") and item.name[1:].isdigit()
        else (1, item.name),
    ):
        if path.is_dir() and (path / f"{path.name}.pkl").exists():
            subject_ids.append(path.name)
    return subject_ids


def normalize_window(window_signal: np.ndarray) -> np.ndarray:
    mean = np.mean(window_signal)
    std = np.std(window_signal)
    if std == 0.0:
        std = 1.0
    return ((window_signal - mean) / std).astype(np.float32)


def build_raw_window_rows(workspace_root: Path) -> list[dict[str, object]]:
    subject_ids = discover_subject_ids(workspace_root)
    window_samples = WINDOW_SECONDS * BVP_SAMPLING_RATE
    step_samples = STEP_SECONDS * BVP_SAMPLING_RATE

    rows: list[dict[str, object]] = []
    for subject_id in subject_ids:
        data = load_subject_pickle(subject_id, workspace_root)
        bvp = np.asarray(data["signal"]["wrist"]["BVP"]).flatten()
        labels_700hz = np.asarray(data["label"])

        filtered_bvp = apply_butterworth_bandpass_fft(
            signal=bvp,
            sampling_rate=BVP_SAMPLING_RATE,
            low_cutoff_hz=0.5,
            high_cutoff_hz=4.0,
            order=4,
        )
        labels_on_bvp_timeline = map_labels_to_bvp_timeline(
            labels_700hz=labels_700hz,
            bvp_length=len(filtered_bvp),
            bvp_sampling_rate=BVP_SAMPLING_RATE,
            label_sampling_rate=LABEL_SAMPLING_RATE,
        )
        segments = extract_valid_segments(labels_on_bvp_timeline, subject_id)

        for segment_index, segment in enumerate(segments, start=1):
            if segment.label_name not in VALID_LABELS:
                continue
            segment_signal = filtered_bvp[segment.start_sample : segment.end_sample + 1]
            if len(segment_signal) < window_samples:
                continue

            for offset in range(0, len(segment_signal) - window_samples + 1, step_samples):
                window_signal = segment_signal[offset : offset + window_samples]
                global_start = segment.start_sample + offset
                global_end = global_start + window_samples - 1
                rows.append(
                    {
                        "subject": subject_id,
                        "segment_index": segment_index,
                        "label_name": segment.label_name,
                        "window_start_seconds": round(global_start / BVP_SAMPLING_RATE, 3),
                        "window_end_seconds": round((global_end + 1) / BVP_SAMPLING_RATE, 3),
                        "signal": normalize_window(window_signal),
                    }
                )
    return rows


def compute_confusion_matrix(
    y_true_indices: np.ndarray, y_pred_indices: np.ndarray, class_labels: list[str]
) -> np.ndarray:
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    for true_index, pred_index in zip(y_true_indices, y_pred_indices):
        matrix[int(true_index), int(pred_index)] += 1
    return matrix


def compute_metrics(
    y_true_indices: np.ndarray, y_pred_indices: np.ndarray, class_labels: list[str]
) -> tuple[dict[str, object], np.ndarray]:
    confusion_matrix = compute_confusion_matrix(y_true_indices, y_pred_indices, class_labels)
    total = int(confusion_matrix.sum())
    correct = int(np.trace(confusion_matrix))
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []

    for index, label in enumerate(class_labels):
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


class SimpleCNN1D(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x_values))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int = CNN_EPOCHS,
) -> list[dict[str, float]]:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0
        correct = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_x)
            total_samples += len(batch_x)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == batch_y).sum().item())

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_samples,
                "train_accuracy": correct / total_samples,
            }
        )
    return history


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_indices: list[int] = []
    pred_indices: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            pred_indices.extend(predictions.tolist())
            true_indices.extend(batch_y.numpy().tolist())

    return np.asarray(true_indices, dtype=np.int64), np.asarray(pred_indices, dtype=np.int64)


def build_dataloaders(
    rows: list[dict[str, object]],
    class_labels: list[str],
    train_subjects: set[str],
    test_subjects: set[str],
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, np.ndarray]:
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    frame = pd.DataFrame(rows)
    train_frame = frame[frame["subject"].isin(train_subjects)].copy()
    test_frame = frame[frame["subject"].isin(test_subjects)].copy()

    x_train = np.stack(train_frame["signal"].to_list()).astype(np.float32)
    x_test = np.stack(test_frame["signal"].to_list()).astype(np.float32)
    y_train = train_frame["target_label"].map(label_to_index).to_numpy(dtype=np.int64)
    y_test = test_frame["target_label"].map(label_to_index).to_numpy(dtype=np.int64)

    x_train_tensor = torch.from_numpy(x_train.copy()).unsqueeze(1)
    x_test_tensor = torch.from_numpy(x_test.copy()).unsqueeze(1)
    y_train_tensor = torch.from_numpy(y_train.copy())
    y_test_tensor = torch.from_numpy(y_test.copy())

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    class_counts = np.bincount(y_train, minlength=len(class_labels)).astype(np.float32)
    class_weights = class_counts.sum() / (len(class_labels) * np.maximum(class_counts, 1.0))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_frame, test_frame, class_weights


def run_variant(
    variant_name: str,
    rows: list[dict[str, object]],
    class_labels: list[str],
    outputs_dir: Path,
    train_subjects: set[str],
    test_subjects: set[str],
    device: torch.device,
) -> dict[str, object]:
    train_loader, test_loader, train_frame, test_frame, class_weights = build_dataloaders(
        rows, class_labels, train_subjects, test_subjects
    )

    model = SimpleCNN1D(num_classes=len(class_labels)).to(device)
    history = train_model(model, train_loader, device, torch.from_numpy(class_weights))
    y_true_indices, y_pred_indices = evaluate_model(model, test_loader, device)
    metrics, confusion_matrix = compute_metrics(y_true_indices, y_pred_indices, class_labels)

    prefix = f"cnn_{variant_name}"
    metrics_output_path = outputs_dir / f"{prefix}_metrics.json"
    confusion_csv_path = outputs_dir / f"{prefix}_confusion_matrix.csv"
    confusion_svg_path = outputs_dir / f"{prefix}_confusion_matrix.svg"
    predictions_output_path = outputs_dir / f"{prefix}_test_predictions.csv"
    model_info_output_path = outputs_dir / f"{prefix}_model_info.json"
    history_output_path = outputs_dir / f"{prefix}_training_history.csv"

    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_confusion_matrix_csv(confusion_matrix, class_labels, confusion_csv_path)
    write_confusion_matrix_svg(
        confusion_matrix,
        class_labels,
        confusion_svg_path,
        title=f"CNN Confusion Matrix ({variant_name})",
    )

    predictions_frame = test_frame[
        ["subject", "segment_index", "target_label", "window_start_seconds", "window_end_seconds"]
    ].copy()
    predictions_frame["predicted_label"] = [class_labels[index] for index in y_pred_indices]
    predictions_frame.to_csv(predictions_output_path, index=False)
    pd.DataFrame(history).to_csv(history_output_path, index=False)

    model_info = {
        "variant": variant_name,
        "model_type": "1D CNN using PyTorch",
        "feature_input": "filtered raw BVP windows normalized per window",
        "class_order": class_labels,
        "train_subjects": sorted(train_subjects, key=lambda value: int(value[1:])),
        "test_subjects": sorted(test_subjects, key=lambda value: int(value[1:])),
        "num_train_samples": int(len(train_frame)),
        "num_test_samples": int(len(test_frame)),
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "sampling_rate_hz": BVP_SAMPLING_RATE,
        "epochs": len(history),
        "batch_size": 64,
        "class_weights": class_weights.tolist(),
        "device": str(device),
    }
    model_info_output_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    return {
        "variant": variant_name,
        "metrics_path": str(metrics_output_path),
        "confusion_svg_path": str(confusion_svg_path),
        "history_path": str(history_output_path),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "num_train_samples": len(train_frame),
        "num_test_samples": len(test_frame),
    }


def main() -> None:
    set_seed()
    workspace_root = Path(__file__).resolve().parent
    outputs_dir = workspace_root / "outputs"

    split_config = json.loads((outputs_dir / "all_subjects_split.json").read_text(encoding="utf-8"))
    train_subjects = set(split_config["train_subjects"])
    test_subjects = set(split_config["test_subjects"])

    device = torch.device("cpu")
    all_rows = build_raw_window_rows(workspace_root)

    three_class_rows = []
    for row in all_rows:
        if row["label_name"] in {"baseline", "stress", "amusement"}:
            new_row = dict(row)
            new_row["target_label"] = row["label_name"]
            three_class_rows.append(new_row)

    binary_rows = []
    for row in three_class_rows:
        new_row = dict(row)
        new_row["target_label"] = "stress" if row["label_name"] == "stress" else "non-stress"
        binary_rows.append(new_row)

    summaries = []
    summaries.append(
        run_variant(
            variant_name="3class",
            rows=three_class_rows,
            class_labels=["baseline", "stress", "amusement"],
            outputs_dir=outputs_dir,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            device=device,
        )
    )
    summaries.append(
        run_variant(
            variant_name="binary",
            rows=binary_rows,
            class_labels=["non-stress", "stress"],
            outputs_dir=outputs_dir,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            device=device,
        )
    )

    summary_output_path = outputs_dir / "cnn_variants_summary.json"
    summary_output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("Completed CNN training for 3-class and binary variants.")
    for summary in summaries:
        print(
            f"{summary['variant']}: accuracy={summary['accuracy']:.4f}, "
            f"macro_f1={summary['macro_f1']:.4f}, "
            f"train={summary['num_train_samples']}, test={summary['num_test_samples']}"
        )
    print(f"Saved summary to: {summary_output_path}")


if __name__ == "__main__":
    main()
