from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from process_s2_pipeline import apply_butterworth_bandpass_fft, load_subject_pickle, map_labels_to_bvp_timeline
from train_random_forest_variants import RandomForestClassifierScratch
from train_svm_variants import standardize_train_test, train_one_vs_rest_svm, predict_one_vs_rest
from train_cnn_variants import SimpleCNN1D, train_model, evaluate_model


OUTPUT_DIR_NAME = "outputs"
FRIENDS_PREFIX = "friends_style_binary"
FILTER_LOW = 0.7
FILTER_HIGH = 3.7
FILTER_ORDER = 3
WINDOW_SECONDS = 60
STEP_SECONDS = 5
BVP_SAMPLING_RATE = 64
LABEL_SAMPLING_RATE = 700
RANDOM_SEED = 42
CNN_EPOCHS = 8
BATCH_SIZE = 256
CLASS_LABELS = ["non-stress", "stress"]
VALID_LABELS = {1: "baseline", 2: "stress", 3: "amusement"}


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


def zscore_subject(signal: np.ndarray) -> np.ndarray:
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    if std == 0.0:
        std = 1.0
    return ((signal - mean) / std).astype(np.float32)


def detect_peaks(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    if len(signal) < 3:
        return np.asarray([], dtype=np.int64)

    smooth = np.convolve(signal, np.ones(5) / 5.0, mode="same")
    threshold = float(np.mean(smooth) + 0.2 * np.std(smooth))
    min_distance = max(1, int(0.25 * sampling_rate))
    peaks: list[int] = []
    last_peak = -min_distance

    for index in range(1, len(smooth) - 1):
        if index - last_peak < min_distance:
            continue
        if smooth[index] > threshold and smooth[index] >= smooth[index - 1] and smooth[index] > smooth[index + 1]:
            peaks.append(index)
            last_peak = index

    return np.asarray(peaks, dtype=np.int64)


def compute_skewness(signal: np.ndarray) -> float:
    std = float(np.std(signal))
    if std == 0.0:
        return 0.0
    centered = (signal - np.mean(signal)) / std
    return float(np.mean(centered**3))


def compute_kurtosis(signal: np.ndarray) -> float:
    std = float(np.std(signal))
    if std == 0.0:
        return 0.0
    centered = (signal - np.mean(signal)) / std
    return float(np.mean(centered**4) - 3.0)


def extract_feature_vector(window_signal: np.ndarray) -> dict[str, float]:
    peaks = detect_peaks(window_signal, BVP_SAMPLING_RATE)
    peak_values = window_signal[peaks] if len(peaks) else np.asarray([], dtype=np.float32)
    ibi = np.diff(peaks) / BVP_SAMPLING_RATE if len(peaks) > 1 else np.asarray([], dtype=np.float32)
    hr = 60.0 / ibi if len(ibi) else np.asarray([], dtype=np.float32)
    ibi_diff = np.diff(ibi) if len(ibi) > 1 else np.asarray([], dtype=np.float32)

    def safe_mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else 0.0

    def safe_std(values: np.ndarray) -> float:
        return float(np.std(values)) if len(values) else 0.0

    def safe_min(values: np.ndarray) -> float:
        return float(np.min(values)) if len(values) else 0.0

    def safe_max(values: np.ndarray) -> float:
        return float(np.max(values)) if len(values) else 0.0

    nn50 = int(np.sum(np.abs(ibi_diff) > 0.05)) if len(ibi_diff) else 0
    pnn50 = float(nn50 / len(ibi_diff)) if len(ibi_diff) else 0.0
    rmssd = float(np.sqrt(np.mean(ibi_diff**2))) if len(ibi_diff) else 0.0

    return {
        "mean_hr": safe_mean(hr),
        "std_hr": safe_std(hr),
        "min_hr": safe_min(hr),
        "max_hr": safe_max(hr),
        "mean_ibi": safe_mean(ibi),
        "std_ibi": safe_std(ibi),
        "min_ibi": safe_min(ibi),
        "max_ibi": safe_max(ibi),
        "rmssd": rmssd,
        "sdnn": safe_std(ibi),
        "nn50": float(nn50),
        "pnn50": pnn50,
        "num_peaks": float(len(peaks)),
        "peak_density": float(len(peaks) / WINDOW_SECONDS),
        "mean_peak_amp": safe_mean(peak_values),
        "std_peak_amp": safe_std(peak_values),
        "mean_signal": float(np.mean(window_signal)),
        "std_signal": float(np.std(window_signal)),
        "min_signal": float(np.min(window_signal)),
        "max_signal": float(np.max(window_signal)),
        "range_signal": float(np.max(window_signal) - np.min(window_signal)),
        "rms_signal": float(np.sqrt(np.mean(window_signal**2))),
        "skewness": compute_skewness(window_signal),
        "kurtosis": compute_kurtosis(window_signal),
    }


def build_window_rows(workspace_root: Path) -> list[dict[str, object]]:
    subject_ids = discover_subject_ids(workspace_root)
    window_samples = WINDOW_SECONDS * BVP_SAMPLING_RATE
    step_samples = STEP_SECONDS * BVP_SAMPLING_RATE
    rows: list[dict[str, object]] = []

    for subject_id in subject_ids:
        data = load_subject_pickle(subject_id, workspace_root)
        bvp = np.asarray(data["signal"]["wrist"]["BVP"]).flatten()
        labels_700hz = np.asarray(data["label"])

        filtered = apply_butterworth_bandpass_fft(
            signal=bvp,
            sampling_rate=BVP_SAMPLING_RATE,
            low_cutoff_hz=FILTER_LOW,
            high_cutoff_hz=FILTER_HIGH,
            order=FILTER_ORDER,
        )
        normalized = zscore_subject(filtered)
        labels_64hz = map_labels_to_bvp_timeline(
            labels_700hz=labels_700hz,
            bvp_length=len(normalized),
            bvp_sampling_rate=BVP_SAMPLING_RATE,
            label_sampling_rate=LABEL_SAMPLING_RATE,
        )

        segment_index = 0
        for start in range(0, len(normalized) - window_samples + 1, step_samples):
            end = start + window_samples
            window_labels = labels_64hz[start:end]
            unique = np.unique(window_labels)
            if len(unique) != 1:
                continue
            label_id = int(unique[0])
            if label_id not in VALID_LABELS:
                continue

            segment_index += 1
            label_name = "stress" if label_id == 2 else "non-stress"
            signal_window = normalized[start:end].copy()
            feature_values = extract_feature_vector(signal_window)
            rows.append(
                {
                    "subject": subject_id,
                    "segment_index": segment_index,
                    "source_label": VALID_LABELS[label_id],
                    "target_label": label_name,
                    "window_start_seconds": round(start / BVP_SAMPLING_RATE, 3),
                    "window_end_seconds": round(end / BVP_SAMPLING_RATE, 3),
                    "signal": signal_window,
                    **feature_values,
                }
            )

    return rows


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    matrix = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_index[str(true_label)], label_to_index[str(pred_label)]] += 1
    return matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    matrix = compute_confusion_matrix(y_true, y_pred)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for index, label in enumerate(CLASS_LABELS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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
        "confusion_matrix": matrix.tolist(),
        "class_order": CLASS_LABELS,
        "num_test_samples": total,
    }


def write_confusion_matrix_csv(matrix: np.ndarray, output_path: Path) -> None:
    rows = []
    for row_label, row_values in zip(CLASS_LABELS, matrix):
        row = {"true_label": row_label}
        for column_label, value in zip(CLASS_LABELS, row_values):
            row[column_label] = int(value)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_confusion_matrix_svg(matrix: np.ndarray, output_path: Path, title: str) -> None:
    from generate_final_project_report import get_font
    from PIL import Image, ImageDraw

    cell_size = 110
    width = 520
    height = 420
    left_margin = 150
    top_margin = 100
    max_value = int(matrix.max()) if matrix.size else 1
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(22)
    label_font = get_font(14)
    value_font = get_font(18)

    draw.text((90, 18), title, fill="black", font=title_font)

    def color_for(value: int) -> tuple[int, int, int]:
        intensity = value / max_value if max_value else 0.0
        blue = int(255 - 140 * intensity)
        red = int(245 - 120 * intensity)
        green = int(248 - 140 * intensity)
        return red, green, blue

    for col, label in enumerate(CLASS_LABELS):
        draw.text((left_margin + col * cell_size + 18, top_margin - 28), label, fill="black", font=label_font)
    for row, label in enumerate(CLASS_LABELS):
        draw.text((20, top_margin + row * cell_size + 42), label, fill="black", font=label_font)

    for row, row_values in enumerate(matrix):
        for col, value in enumerate(row_values):
            x0 = left_margin + col * cell_size
            y0 = top_margin + row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color_for(int(value)), outline=(80, 80, 80), width=2)
            draw.text((x0 + 36, y0 + 42), str(int(value)), fill="black", font=value_font)

    image.save(output_path)


def evaluate_svm(data_frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    predictions = []
    for left_out_subject in sorted(data_frame["subject"].unique(), key=lambda value: int(value[1:])):
        train_frame = data_frame[data_frame["subject"] != left_out_subject].copy()
        test_frame = data_frame[data_frame["subject"] == left_out_subject].copy()
        x_train = train_frame[feature_columns].to_numpy(dtype=np.float64)
        x_test = test_frame[feature_columns].to_numpy(dtype=np.float64)
        y_train = train_frame["target_label"].to_numpy(dtype=str)
        y_test = test_frame["target_label"].to_numpy(dtype=str)

        x_train_scaled, x_test_scaled, _, _ = standardize_train_test(x_train, x_test)
        weights, biases = train_one_vs_rest_svm(x_train_scaled, y_train, CLASS_LABELS)
        y_pred, _ = predict_one_vs_rest(x_test_scaled, weights, biases, CLASS_LABELS)
        fold_frame = test_frame[["subject", "segment_index", "target_label", "window_start_seconds", "window_end_seconds"]].copy()
        fold_frame["predicted_label"] = y_pred
        predictions.append(fold_frame)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics = compute_metrics(
        prediction_frame["target_label"].to_numpy(dtype=str),
        prediction_frame["predicted_label"].to_numpy(dtype=str),
    )
    return {"metrics": metrics, "predictions": prediction_frame}


def evaluate_rf(data_frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    predictions = []
    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}

    for left_out_subject in sorted(data_frame["subject"].unique(), key=lambda value: int(value[1:])):
        train_frame = data_frame[data_frame["subject"] != left_out_subject].copy()
        test_frame = data_frame[data_frame["subject"] == left_out_subject].copy()
        x_train = train_frame[feature_columns].to_numpy(dtype=np.float64)
        x_test = test_frame[feature_columns].to_numpy(dtype=np.float64)
        y_train = train_frame["target_label"].map(label_to_index).to_numpy(dtype=np.int64)

        forest = RandomForestClassifierScratch(
            class_labels=CLASS_LABELS,
            n_trees=25,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=max(1, int(np.sqrt(len(feature_columns)))),
            max_thresholds_per_feature=12,
            random_state=42,
        )
        forest.fit(x_train, y_train)
        y_pred_indices = forest.predict(x_test)
        y_pred = [CLASS_LABELS[index] for index in y_pred_indices]
        fold_frame = test_frame[["subject", "segment_index", "target_label", "window_start_seconds", "window_end_seconds"]].copy()
        fold_frame["predicted_label"] = y_pred
        predictions.append(fold_frame)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics = compute_metrics(
        prediction_frame["target_label"].to_numpy(dtype=str),
        prediction_frame["predicted_label"].to_numpy(dtype=str),
    )
    return {"metrics": metrics, "predictions": prediction_frame}


def evaluate_cnn(data_frame: pd.DataFrame) -> dict[str, object]:
    set_seed()
    device = torch.device("cpu")
    predictions = []
    history_rows = []
    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}

    for fold_index, left_out_subject in enumerate(sorted(data_frame["subject"].unique(), key=lambda value: int(value[1:])), start=1):
        train_frame = data_frame[data_frame["subject"] != left_out_subject].copy()
        test_frame = data_frame[data_frame["subject"] == left_out_subject].copy()

        x_train = np.stack(train_frame["signal"].to_list()).astype(np.float32)
        x_test = np.stack(test_frame["signal"].to_list()).astype(np.float32)
        y_train = train_frame["target_label"].map(label_to_index).to_numpy(dtype=np.int64)
        y_test = test_frame["target_label"].map(label_to_index).to_numpy(dtype=np.int64)

        train_dataset = TensorDataset(
            torch.from_numpy(x_train.copy()).unsqueeze(1),
            torch.from_numpy(y_train.copy()),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(x_test.copy()).unsqueeze(1),
            torch.from_numpy(y_test.copy()),
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        class_counts = np.bincount(y_train, minlength=len(CLASS_LABELS)).astype(np.float32)
        class_weights = class_counts.sum() / (len(CLASS_LABELS) * np.maximum(class_counts, 1.0))

        model = SimpleCNN1D(num_classes=len(CLASS_LABELS)).to(device)
        history = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            class_weights=torch.from_numpy(class_weights),
            epochs=CNN_EPOCHS,
        )
        for entry in history:
            history_rows.append({"fold": fold_index, "left_out_subject": left_out_subject, **entry})

        y_true_indices, y_pred_indices = evaluate_model(model, test_loader, device)
        fold_frame = test_frame[["subject", "segment_index", "target_label", "window_start_seconds", "window_end_seconds"]].copy()
        fold_frame["predicted_label"] = [CLASS_LABELS[index] for index in y_pred_indices]
        predictions.append(fold_frame)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics = compute_metrics(
        prediction_frame["target_label"].to_numpy(dtype=str),
        prediction_frame["predicted_label"].to_numpy(dtype=str),
    )
    return {"metrics": metrics, "predictions": prediction_frame, "history": pd.DataFrame(history_rows)}


def main() -> None:
    workspace_root = Path(__file__).resolve().parent
    outputs_dir = workspace_root / OUTPUT_DIR_NAME
    outputs_dir.mkdir(exist_ok=True)

    rows = build_window_rows(workspace_root)
    data_frame = pd.DataFrame(rows)
    feature_columns = [
        "mean_hr",
        "std_hr",
        "min_hr",
        "max_hr",
        "mean_ibi",
        "std_ibi",
        "min_ibi",
        "max_ibi",
        "rmssd",
        "sdnn",
        "nn50",
        "pnn50",
        "num_peaks",
        "peak_density",
        "mean_peak_amp",
        "std_peak_amp",
        "mean_signal",
        "std_signal",
        "min_signal",
        "max_signal",
        "range_signal",
        "rms_signal",
        "skewness",
        "kurtosis",
    ]

    svm_result = evaluate_svm(data_frame, feature_columns)
    rf_result = evaluate_rf(data_frame, feature_columns)

    results = {
        "pipeline": {
            "filter": {"low_hz": FILTER_LOW, "high_hz": FILTER_HIGH, "order": FILTER_ORDER},
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "normalization": "per-subject z-score on filtered BVP",
            "mixed_windows": "discarded",
            "evaluation": "Leave-One-Subject-Out (LOSO)",
            "feature_count": len(feature_columns),
        },
        "dataset": {
            "num_windows": int(len(data_frame)),
            "num_subjects": int(data_frame["subject"].nunique()),
            "class_counts": data_frame["target_label"].value_counts().to_dict(),
            "source_label_counts": data_frame["source_label"].value_counts().to_dict(),
        },
        "svm": svm_result["metrics"],
        "random_forest": rf_result["metrics"],
    }

    for model_name, result in [("svm", svm_result), ("random_forest", rf_result)]:
        metrics_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_metrics.json"
        predictions_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_predictions.csv"
        confusion_csv_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_confusion_matrix.csv"
        confusion_svg_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_confusion_matrix.png"

        metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        result["predictions"].to_csv(predictions_path, index=False)
        matrix = np.asarray(result["metrics"]["confusion_matrix"], dtype=np.int64)
        write_confusion_matrix_csv(matrix, confusion_csv_path)
        write_confusion_matrix_svg(matrix, confusion_svg_path, title=f"{model_name.upper()} LOSO Confusion Matrix")

    data_frame.drop(columns=["signal"]).to_csv(outputs_dir / f"{FRIENDS_PREFIX}_dataset_features.csv", index=False)

    print("Completed alternate friends-style LOSO evaluation for classical models.")
    print(f"Total windows: {len(data_frame)}")
    print(f"SVM accuracy: {svm_result['metrics']['accuracy']:.4f}, macro F1: {svm_result['metrics']['macro_f1']:.4f}")
    print(f"Random Forest accuracy: {rf_result['metrics']['accuracy']:.4f}, macro F1: {rf_result['metrics']['macro_f1']:.4f}")

    cnn_result = evaluate_cnn(data_frame)
    results["cnn"] = cnn_result["metrics"]
    for model_name, result in [("cnn", cnn_result)]:
        metrics_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_metrics.json"
        predictions_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_predictions.csv"
        confusion_csv_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_confusion_matrix.csv"
        confusion_svg_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_confusion_matrix.png"
        metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        result["predictions"].to_csv(predictions_path, index=False)
        matrix = np.asarray(result["metrics"]["confusion_matrix"], dtype=np.int64)
        write_confusion_matrix_csv(matrix, confusion_csv_path)
        write_confusion_matrix_svg(matrix, confusion_svg_path, title=f"{model_name.upper()} LOSO Confusion Matrix")
    if "history" in cnn_result:
        cnn_result["history"].to_csv(outputs_dir / f"{FRIENDS_PREFIX}_cnn_training_history.csv", index=False)

    summary_path = outputs_dir / f"{FRIENDS_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"CNN accuracy: {cnn_result['metrics']['accuracy']:.4f}, macro F1: {cnn_result['metrics']['macro_f1']:.4f}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
