from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import evaluate_friends_style_binary as base
from train_cnn_variants import SimpleCNN1D, evaluate_model, train_model
from train_random_forest_variants import RandomForestClassifierScratch
from train_svm_variants import predict_one_vs_rest, standardize_train_test, train_one_vs_rest_svm


FRIENDS_PREFIX = "friends_style_3class"
CLASS_LABELS = ["baseline", "stress", "amusement"]
VALID_LABELS = {1: "baseline", 2: "stress", 3: "amusement"}
CNN_EPOCHS = 5


def build_window_rows(workspace_root: Path) -> list[dict[str, object]]:
    subject_ids = base.discover_subject_ids(workspace_root)
    window_samples = base.WINDOW_SECONDS * base.BVP_SAMPLING_RATE
    step_samples = base.STEP_SECONDS * base.BVP_SAMPLING_RATE
    rows: list[dict[str, object]] = []

    for subject_id in subject_ids:
        data = base.load_subject_pickle(subject_id, workspace_root)
        bvp = np.asarray(data["signal"]["wrist"]["BVP"]).flatten()
        labels_700hz = np.asarray(data["label"])

        filtered = base.apply_butterworth_bandpass_fft(
            signal=bvp,
            sampling_rate=base.BVP_SAMPLING_RATE,
            low_cutoff_hz=base.FILTER_LOW,
            high_cutoff_hz=base.FILTER_HIGH,
            order=base.FILTER_ORDER,
        )
        normalized = base.zscore_subject(filtered)
        labels_64hz = base.map_labels_to_bvp_timeline(
            labels_700hz=labels_700hz,
            bvp_length=len(normalized),
            bvp_sampling_rate=base.BVP_SAMPLING_RATE,
            label_sampling_rate=base.LABEL_SAMPLING_RATE,
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
            label_name = VALID_LABELS[label_id]
            signal_window = normalized[start:end].copy()
            feature_values = base.extract_feature_vector(signal_window)
            rows.append(
                {
                    "subject": subject_id,
                    "segment_index": segment_index,
                    "target_label": label_name,
                    "window_start_seconds": round(start / base.BVP_SAMPLING_RATE, 3),
                    "window_end_seconds": round(end / base.BVP_SAMPLING_RATE, 3),
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


def evaluate_svm(data_frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    predictions = []
    for left_out_subject in sorted(data_frame["subject"].unique(), key=lambda value: int(value[1:])):
        train_frame = data_frame[data_frame["subject"] != left_out_subject].copy()
        test_frame = data_frame[data_frame["subject"] == left_out_subject].copy()
        x_train = train_frame[feature_columns].to_numpy(dtype=np.float64)
        x_test = test_frame[feature_columns].to_numpy(dtype=np.float64)
        y_train = train_frame["target_label"].to_numpy(dtype=str)

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


def evaluate_cnn(data_frame: pd.DataFrame, cnn_epochs: int) -> dict[str, object]:
    base.set_seed()
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

        train_dataset = TensorDataset(torch.from_numpy(x_train.copy()).unsqueeze(1), torch.from_numpy(y_train.copy()))
        test_dataset = TensorDataset(torch.from_numpy(x_test.copy()).unsqueeze(1), torch.from_numpy(y_test.copy()))
        train_loader = DataLoader(train_dataset, batch_size=base.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=base.BATCH_SIZE, shuffle=False)

        class_counts = np.bincount(y_train, minlength=len(CLASS_LABELS)).astype(np.float32)
        class_weights = class_counts.sum() / (len(CLASS_LABELS) * np.maximum(class_counts, 1.0))

        model = SimpleCNN1D(num_classes=len(CLASS_LABELS)).to(device)
        history = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            class_weights=torch.from_numpy(class_weights),
            epochs=cnn_epochs,
        )
        for entry in history:
            history_rows.append({"fold": fold_index, "left_out_subject": left_out_subject, **entry})

        _, y_pred_indices = evaluate_model(model, test_loader, device)
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
    parser = argparse.ArgumentParser(description="Run friends-style 3-class LOSO evaluation.")
    parser.add_argument("--cnn-epochs", type=int, default=5, help="Number of CNN epochs per LOSO fold (default: 5).")
    args = parser.parse_args()
    cnn_epochs = int(args.cnn_epochs)

    workspace_root = Path(__file__).resolve().parent
    outputs_dir = workspace_root / base.OUTPUT_DIR_NAME
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
    cnn_result = evaluate_cnn(data_frame, cnn_epochs)

    results = {
        "pipeline": {
            "filter": {"low_hz": base.FILTER_LOW, "high_hz": base.FILTER_HIGH, "order": base.FILTER_ORDER},
            "window_seconds": base.WINDOW_SECONDS,
            "step_seconds": base.STEP_SECONDS,
            "normalization": "per-subject z-score on filtered BVP",
            "mixed_windows": "discarded",
            "evaluation": "Leave-One-Subject-Out (LOSO)",
            "feature_count": len(feature_columns),
            "cnn_epochs": cnn_epochs,
        },
        "dataset": {
            "num_windows": int(len(data_frame)),
            "num_subjects": int(data_frame["subject"].nunique()),
            "class_counts": data_frame["target_label"].value_counts().to_dict(),
        },
        "svm": svm_result["metrics"],
        "random_forest": rf_result["metrics"],
        "cnn": cnn_result["metrics"],
    }

    for model_name, result in [("svm", svm_result), ("random_forest", rf_result), ("cnn", cnn_result)]:
        metrics_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_metrics.json"
        predictions_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_predictions.csv"
        confusion_csv_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_confusion_matrix.csv"
        confusion_png_path = outputs_dir / f"{FRIENDS_PREFIX}_{model_name}_confusion_matrix.png"

        metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        result["predictions"].to_csv(predictions_path, index=False)
        matrix = np.asarray(result["metrics"]["confusion_matrix"], dtype=np.int64)
        base.write_confusion_matrix_csv(matrix, confusion_csv_path)
        base.write_confusion_matrix_svg(matrix, confusion_png_path, title=f"{model_name.upper()} LOSO Confusion Matrix")

    data_frame.drop(columns=["signal"]).to_csv(outputs_dir / f"{FRIENDS_PREFIX}_dataset_features.csv", index=False)
    cnn_result["history"].to_csv(outputs_dir / f"{FRIENDS_PREFIX}_cnn_training_history.csv", index=False)
    summary_path = outputs_dir / f"{FRIENDS_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Completed friends-style 3-class LOSO evaluation.")
    print(f"Total windows: {len(data_frame)}")
    print(f"SVM accuracy: {svm_result['metrics']['accuracy']:.4f}, macro F1: {svm_result['metrics']['macro_f1']:.4f}")
    print(f"Random Forest accuracy: {rf_result['metrics']['accuracy']:.4f}, macro F1: {rf_result['metrics']['macro_f1']:.4f}")
    print(f"CNN accuracy: {cnn_result['metrics']['accuracy']:.4f}, macro F1: {cnn_result['metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
