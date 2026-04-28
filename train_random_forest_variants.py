from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass
class TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    prediction_index: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.prediction_index is not None


def gini_impurity(y_indices: np.ndarray, num_classes: int) -> float:
    if len(y_indices) == 0:
        return 0.0
    counts = np.bincount(y_indices, minlength=num_classes).astype(np.float64)
    probabilities = counts / counts.sum()
    return 1.0 - float(np.sum(probabilities**2))


def majority_class_index(y_indices: np.ndarray, num_classes: int) -> int:
    counts = np.bincount(y_indices, minlength=num_classes)
    return int(np.argmax(counts))


class DecisionTreeClassifierScratch:
    def __init__(
        self,
        num_classes: int,
        max_depth: int = 8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: int | None = None,
        max_thresholds_per_feature: int = 12,
        random_state: int = 42,
    ) -> None:
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.random_state = random_state
        self.root: TreeNode | None = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, x_values: np.ndarray, y_indices: np.ndarray) -> None:
        self.root = self._build_tree(x_values, y_indices, depth=0)

    def _build_tree(
        self, x_values: np.ndarray, y_indices: np.ndarray, depth: int
    ) -> TreeNode:
        node_prediction = majority_class_index(y_indices, self.num_classes)
        unique_classes = np.unique(y_indices)

        if (
            depth >= self.max_depth
            or len(y_indices) < self.min_samples_split
            or len(unique_classes) == 1
        ):
            return TreeNode(prediction_index=node_prediction)

        split = self._find_best_split(x_values, y_indices)
        if split is None:
            return TreeNode(prediction_index=node_prediction)

        feature_index, threshold = split
        left_mask = x_values[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_node = self._build_tree(x_values[left_mask], y_indices[left_mask], depth + 1)
        right_node = self._build_tree(x_values[right_mask], y_indices[right_mask], depth + 1)
        return TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )

    def _find_best_split(
        self, x_values: np.ndarray, y_indices: np.ndarray
    ) -> tuple[int, float] | None:
        num_samples, num_features = x_values.shape
        if num_samples < self.min_samples_split:
            return None

        feature_count = self.max_features or num_features
        feature_count = min(feature_count, num_features)
        feature_indices = self._rng.choice(
            num_features, size=feature_count, replace=False
        )

        parent_impurity = gini_impurity(y_indices, self.num_classes)
        best_gain = 0.0
        best_split: tuple[int, float] | None = None

        for feature_index in feature_indices:
            feature_values = x_values[:, feature_index]
            unique_values = np.unique(feature_values)
            if len(unique_values) <= 1:
                continue

            if len(unique_values) > self.max_thresholds_per_feature:
                quantiles = np.linspace(0.1, 0.9, self.max_thresholds_per_feature)
                thresholds = np.unique(np.quantile(feature_values, quantiles))
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                left_count = int(left_mask.sum())
                right_count = int(right_mask.sum())

                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                ):
                    continue

                left_impurity = gini_impurity(y_indices[left_mask], self.num_classes)
                right_impurity = gini_impurity(y_indices[right_mask], self.num_classes)
                weighted_impurity = (
                    left_count / num_samples * left_impurity
                    + right_count / num_samples * right_impurity
                )
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_split = (int(feature_index), float(threshold))

        return best_split

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("Decision tree must be fitted before prediction.")
        return np.asarray([self._predict_row(row, self.root) for row in x_values], dtype=np.int64)

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> int:
        current = node
        while not current.is_leaf:
            assert current.feature_index is not None
            assert current.threshold is not None
            assert current.left is not None
            assert current.right is not None
            if row[current.feature_index] <= current.threshold:
                current = current.left
            else:
                current = current.right
        assert current.prediction_index is not None
        return current.prediction_index


class RandomForestClassifierScratch:
    def __init__(
        self,
        class_labels: list[str],
        n_trees: int = 25,
        max_depth: int = 8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: int | None = None,
        max_thresholds_per_feature: int = 12,
        random_state: int = 42,
    ) -> None:
        self.class_labels = class_labels
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.random_state = random_state
        self.trees: list[DecisionTreeClassifierScratch] = []
        self._rng = np.random.default_rng(random_state)

    def fit(self, x_values: np.ndarray, y_indices: np.ndarray) -> None:
        num_samples = len(x_values)
        num_classes = len(self.class_labels)
        self.trees = []

        for tree_index in range(self.n_trees):
            bootstrap_indices = self._rng.integers(0, num_samples, size=num_samples)
            x_bootstrap = x_values[bootstrap_indices]
            y_bootstrap = y_indices[bootstrap_indices]

            tree = DecisionTreeClassifierScratch(
                num_classes=num_classes,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_thresholds_per_feature=self.max_thresholds_per_feature,
                random_state=self.random_state + tree_index + 1,
            )
            tree.fit(x_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Random forest must be fitted before prediction.")

        tree_predictions = np.vstack([tree.predict(x_values) for tree in self.trees])
        final_predictions = []
        num_classes = len(self.class_labels)

        for sample_votes in tree_predictions.T:
            vote_counts = np.bincount(sample_votes, minlength=num_classes)
            final_predictions.append(int(np.argmax(vote_counts)))

        return np.asarray(final_predictions, dtype=np.int64)


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


def run_variant(
    variant_name: str,
    data_frame: pd.DataFrame,
    class_labels: list[str],
    outputs_dir: Path,
    train_subjects: set[str],
    test_subjects: set[str],
) -> dict[str, object]:
    label_to_index = {label: index for index, label in enumerate(class_labels)}

    train_frame = data_frame[data_frame["subject"].isin(train_subjects)].copy()
    test_frame = data_frame[data_frame["subject"].isin(test_subjects)].copy()

    x_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    x_test = test_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_train_indices = train_frame["target_label"].map(label_to_index).to_numpy(dtype=np.int64)
    y_test_indices = test_frame["target_label"].map(label_to_index).to_numpy(dtype=np.int64)

    max_features = max(1, int(np.sqrt(len(FEATURE_COLUMNS))))
    forest = RandomForestClassifierScratch(
        class_labels=class_labels,
        n_trees=25,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=max_features,
        max_thresholds_per_feature=12,
        random_state=42,
    )
    forest.fit(x_train, y_train_indices)
    y_pred_indices = forest.predict(x_test)
    metrics, confusion_matrix = compute_metrics(y_test_indices, y_pred_indices, class_labels)

    prefix = f"rf_{variant_name}"
    metrics_output_path = outputs_dir / f"{prefix}_metrics.json"
    confusion_csv_path = outputs_dir / f"{prefix}_confusion_matrix.csv"
    confusion_svg_path = outputs_dir / f"{prefix}_confusion_matrix.svg"
    predictions_output_path = outputs_dir / f"{prefix}_test_predictions.csv"
    model_info_output_path = outputs_dir / f"{prefix}_model_info.json"

    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_confusion_matrix_csv(confusion_matrix, class_labels, confusion_csv_path)
    write_confusion_matrix_svg(
        confusion_matrix,
        class_labels,
        confusion_svg_path,
        title=f"Random Forest Confusion Matrix ({variant_name})",
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
    predictions_frame["predicted_label"] = [class_labels[index] for index in y_pred_indices]
    predictions_frame.to_csv(predictions_output_path, index=False)

    model_info = {
        "variant": variant_name,
        "model_type": "random forest trained with local NumPy implementation",
        "feature_columns": FEATURE_COLUMNS,
        "class_order": class_labels,
        "train_subjects": sorted(train_subjects, key=lambda value: int(value[1:])),
        "test_subjects": sorted(test_subjects, key=lambda value: int(value[1:])),
        "num_train_samples": int(len(train_frame)),
        "num_test_samples": int(len(test_frame)),
        "n_trees": forest.n_trees,
        "max_depth": forest.max_depth,
        "min_samples_split": forest.min_samples_split,
        "min_samples_leaf": forest.min_samples_leaf,
        "max_features": forest.max_features,
    }
    model_info_output_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    return {
        "variant": variant_name,
        "metrics_path": str(metrics_output_path),
        "confusion_svg_path": str(confusion_svg_path),
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

    summary_output_path = outputs_dir / "rf_variants_summary.json"
    summary_output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("Completed Random Forest training for 3-class and binary variants.")
    for summary in summaries:
        print(
            f"{summary['variant']}: accuracy={summary['accuracy']:.4f}, "
            f"macro_f1={summary['macro_f1']:.4f}, "
            f"train={summary['num_train_samples']}, test={summary['num_test_samples']}"
        )
    print(f"Saved summary to: {summary_output_path}")


if __name__ == "__main__":
    main()
