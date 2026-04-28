from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from generate_final_loso_report import (
    build_final_loso_report,
    create_loso_model_comparison_png,
    create_model_pipeline_png,
    create_processed_segments_png,
    create_report_confusion_matrix_png,
    summary_row_3class,
    summary_row_binary,
)
from generate_final_project_report import OUTPUT_DIR_NAME, create_cnn_architecture_png, create_signal_comparison_png, load_json
import numpy as np


def build_table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e8fb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def metrics_rows(metrics: dict[str, object]) -> list[list[str]]:
    rows = [["Overall", f"{metrics['accuracy']:.4f}", f"{metrics['macro_f1']:.4f}", "-", "-"]]
    for label in metrics["class_order"]:
        rows.append(
            [
                label,
                f"{metrics['per_class'][label]['precision']:.4f}",
                f"{metrics['per_class'][label]['recall']:.4f}",
                f"{metrics['per_class'][label]['f1']:.4f}",
                str(metrics["per_class"][label]["support"]),
            ]
        )
    return rows


def build_final_loso_pdf(workspace_root: Path) -> Path:
    outputs_dir = workspace_root / OUTPUT_DIR_NAME
    signal_png = outputs_dir / "report_signal_comparison.png"
    processed_segments_png = outputs_dir / "report_processed_segments.png"
    svm_pipeline_png = outputs_dir / "report_svm_pipeline.png"
    rf_pipeline_png = outputs_dir / "report_random_forest_pipeline.png"
    cnn_architecture_png = outputs_dir / "report_cnn_architecture.png"
    comparison_png = outputs_dir / "report_loso_model_comparison.png"
    binary_svm_cm = outputs_dir / "report_binary_svm_confusion.png"
    binary_rf_cm = outputs_dir / "report_binary_random_forest_confusion.png"
    binary_cnn_cm = outputs_dir / "report_binary_cnn_confusion.png"
    class3_svm_cm = outputs_dir / "report_3class_svm_confusion.png"
    class3_rf_cm = outputs_dir / "report_3class_random_forest_confusion.png"
    class3_cnn_cm = outputs_dir / "report_3class_cnn_confusion.png"

    create_signal_comparison_png(workspace_root, signal_png)
    create_processed_segments_png(workspace_root, processed_segments_png)
    create_model_pipeline_png(
        "SVM Pipeline Used in This Project",
        [
            ("Input", "60 s filtered\nBVP window"),
            ("Feature Extraction", "24 handcrafted\nPPG / HRV-style\nfeatures"),
            ("Standardization", "mean = 0\nstd = 1 using\ntraining fold"),
            ("Classifier", "One-vs-rest\nlinear SVM"),
            ("Output", "baseline /\nstress /\namusement\nor binary class"),
        ],
        svm_pipeline_png,
    )
    create_model_pipeline_png(
        "Random Forest Pipeline Used in This Project",
        [
            ("Input", "60 s filtered\nBVP window"),
            ("Feature Extraction", "24 handcrafted\nPPG / HRV-style\nfeatures"),
            ("Tree Ensemble", "25 trees\nmax depth = 8\nmin split = 10\nmin leaf = 5"),
            ("Voting", "aggregate\npredictions from\nall trees"),
            ("Output", "baseline /\nstress /\namusement\nor binary class"),
        ],
        rf_pipeline_png,
    )
    create_cnn_architecture_png(cnn_architecture_png)
    create_loso_model_comparison_png(outputs_dir, comparison_png)

    svm_binary = load_json(outputs_dir / "friends_style_binary_svm_metrics.json")
    rf_binary = load_json(outputs_dir / "friends_style_binary_random_forest_metrics.json")
    cnn_binary = load_json(outputs_dir / "friends_style_binary_cnn_metrics.json")
    svm_3class = load_json(outputs_dir / "friends_style_3class_svm_metrics.json")
    rf_3class = load_json(outputs_dir / "friends_style_3class_random_forest_metrics.json")
    cnn_3class = load_json(outputs_dir / "friends_style_3class_cnn_metrics.json")
    binary_summary = load_json(outputs_dir / "friends_style_binary_summary.json")
    class3_summary = load_json(outputs_dir / "friends_style_3class_summary.json")
    binary_frame = pd.read_csv(outputs_dir / "friends_style_binary_dataset_features.csv")
    class3_frame = pd.read_csv(outputs_dir / "friends_style_3class_dataset_features.csv")

    binary_counts = binary_frame["target_label"].value_counts().to_dict()
    class3_counts = class3_frame["target_label"].value_counts().to_dict()
    subject_count = int(binary_frame["subject"].nunique())

    create_report_confusion_matrix_png(np.asarray(svm_binary["confusion_matrix"]), svm_binary["class_order"], "SVM Confusion Matrix", binary_svm_cm)
    create_report_confusion_matrix_png(np.asarray(rf_binary["confusion_matrix"]), rf_binary["class_order"], "Random Forest Confusion Matrix", binary_rf_cm)
    create_report_confusion_matrix_png(np.asarray(cnn_binary["confusion_matrix"]), cnn_binary["class_order"], "CNN Confusion Matrix", binary_cnn_cm)
    create_report_confusion_matrix_png(np.asarray(svm_3class["confusion_matrix"]), svm_3class["class_order"], "SVM Confusion Matrix", class3_svm_cm)
    create_report_confusion_matrix_png(np.asarray(rf_3class["confusion_matrix"]), rf_3class["class_order"], "Random Forest Confusion Matrix", class3_rf_cm)
    create_report_confusion_matrix_png(np.asarray(cnn_3class["confusion_matrix"]), cnn_3class["class_order"], "CNN Confusion Matrix", class3_cnn_cm)

    output_path = outputs_dir / "Final_Report_LOSO.pdf"
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontName="Times-Roman", fontSize=10, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="SectionHead", parent=styles["Heading1"], fontName="Times-Bold", fontSize=15, leading=18, spaceBefore=10, spaceAfter=8))

    story = []
    story.append(Paragraph("Final Report", styles["Title"]))
    story.append(Paragraph("Course 591", styles["BodySmall"]))
    story.append(Paragraph("Name : - Yuvraj Singh Bhatia (ybhatia2)", styles["BodySmall"]))
    story.append(Paragraph("Date: April 27, 2026", styles["BodySmall"]))

    story.append(Paragraph("1. Overview", styles["SectionHead"]))
    story.append(Paragraph("This report presents emotion and stress recognition experiments on the WESAD dataset using wrist-based photoplethysmography (PPG). The objective was to evaluate whether the wrist blood volume pulse signal can distinguish baseline, stress, and amusement states, and whether stress can be separated from non-stress under a subject-independent setting.", styles["BodySmall"]))
    story.append(Paragraph("To make the evaluation realistic, Leave-One-Subject-Out (LOSO) validation was used. In this setting, each subject is held out once for testing while the remaining subjects are used for training. Three models were evaluated: Support Vector Machine (SVM), Random Forest, and a one-dimensional Convolutional Neural Network (CNN). Results are reported for both 3-class classification and binary classification.", styles["BodySmall"]))

    story.append(Paragraph("2. Data Processing", styles["SectionHead"]))
    story.append(Paragraph("The synchronized SX.pkl files from WESAD were used. Only the wrist BVP signal and the protocol labels were retained. The signal was filtered using a Butterworth-style bandpass filter with a low cutoff of 0.7 Hz, a high cutoff of 3.7 Hz, and order 3. This step reduces baseline drift and high-frequency noise while preserving the frequency range that carries useful cardiac information.", styles["BodySmall"]))
    story.append(Paragraph("After filtering, each subject signal was normalized with subject-wise z-score normalization. Fixed-length windows of 60 seconds were extracted with a step size of 5 seconds, and windows containing mixed labels were discarded. For the binary setting, baseline and amusement windows were merged into a non-stress class, while stress windows remained unchanged. For the 3-class setting, baseline, stress, and amusement were kept as separate labels.", styles["BodySmall"]))
    story.append(build_table([["Item", "Value"], ["Subjects used", str(subject_count)], ["Signal", "Wrist BVP"], ["Filter", "0.7-3.7 Hz, order 3"], ["Window size", "60 seconds"], ["Step size", "5 seconds"], ["Evaluation", "LOSO"], ["3-class windows", str(class3_summary["dataset"]["num_windows"])], ["Binary windows", str(binary_summary["dataset"]["num_windows"])]], col_widths=[2.0 * inch, 4.3 * inch]))
    story.append(Paragraph(f"3-class label counts: baseline {class3_counts.get('baseline', 0)}, stress {class3_counts.get('stress', 0)}, amusement {class3_counts.get('amusement', 0)}. Binary label counts: non-stress {binary_counts.get('non-stress', 0)}, stress {binary_counts.get('stress', 0)}.", styles["BodySmall"]))
    story.append(Image(str(signal_png), width=6.2 * inch, height=3.6 * inch))
    story.append(Paragraph("Figure 1. Example of raw and filtered wrist BVP signal used in preprocessing.", styles["BodySmall"]))
    story.append(Image(str(processed_segments_png), width=6.2 * inch, height=3.76 * inch))
    story.append(Paragraph("Figure 2. Example processed PPG segments for baseline, stress, and amusement after filtering.", styles["BodySmall"]))

    story.append(Paragraph("3. Methodology", styles["SectionHead"]))
    story.append(Paragraph("Three classification models were implemented in order to compare feature-based learning with end-to-end deep learning. The first two models, SVM and Random Forest, used the same handcrafted PPG feature set. The CNN used normalized raw windows directly as input.", styles["BodySmall"]))
    story.append(Paragraph("Support Vector Machine (SVM): The SVM model was implemented as a one-vs-rest linear classifier. Input features were standardized before training. This model provides a strong margin-based baseline and is well suited to smaller physiological datasets.", styles["BodySmall"]))
    story.append(Image(str(svm_pipeline_png), width=6.2 * inch, height=1.9 * inch))
    story.append(Paragraph("Figure 3. SVM processing pipeline used in the LOSO experiments.", styles["BodySmall"]))
    story.append(Paragraph("Random Forest: The Random Forest model was trained on the same handcrafted features. The forest used 25 trees, a maximum depth of 8, a minimum split size of 10, a minimum leaf size of 5, and square-root feature sampling at each split. This model is useful because it can capture nonlinear feature interactions without requiring extensive feature scaling.", styles["BodySmall"]))
    story.append(Image(str(rf_pipeline_png), width=6.2 * inch, height=1.9 * inch))
    story.append(Paragraph("Figure 4. Random Forest processing pipeline used in the LOSO experiments.", styles["BodySmall"]))
    story.append(Paragraph("CNN: The CNN model was designed as a one-dimensional convolutional network over 60-second normalized BVP windows. It used stacked convolutional layers, ReLU activations, pooling, and a small fully connected classifier. For the binary LOSO setting, the final CNN was trained for 10 epochs per fold. For the 3-class LOSO setting, the final CNN was also trained for 10 epochs per fold. Weighted loss was used to reduce the effect of class imbalance.", styles["BodySmall"]))
    story.append(Image(str(cnn_architecture_png), width=6.2 * inch, height=2.1 * inch))
    story.append(Paragraph("Figure 5. CNN architecture used for the LOSO experiments.", styles["BodySmall"]))

    story.append(Paragraph("4. Experiments and Analysis", styles["SectionHead"]))
    story.append(Paragraph("Performance was measured using accuracy, macro F1 score, and confusion matrices. Accuracy indicates overall correctness, while macro F1 emphasizes balance across classes. This is especially important in physiological classification tasks where class behavior is not equally easy to learn.", styles["BodySmall"]))
    story.append(Paragraph("3-Class Evaluation Metrics", styles["BodySmall"]))
    story.append(
        build_table(
            [["Model", "Accuracy", "Macro F1", "F1-Baseline", "F1-Stress", "F1-Amusement"]]
            + [
                summary_row_3class("SVM", svm_3class),
                summary_row_3class("Random Forest", rf_3class),
                summary_row_3class("CNN", cnn_3class),
            ],
            col_widths=[1.75 * inch, 0.9 * inch, 0.95 * inch, 1.0 * inch, 0.9 * inch, 1.0 * inch],
        )
    )
    story.append(Paragraph("Binary Evaluation Metrics", styles["BodySmall"]))
    story.append(
        build_table(
            [["Model", "Accuracy", "Macro F1", "F1-Non-Stress", "F1-Stress"]]
            + [
                summary_row_binary("SVM", svm_binary),
                summary_row_binary("Random Forest", rf_binary),
                summary_row_binary("CNN", cnn_binary),
            ],
            col_widths=[1.85 * inch, 0.95 * inch, 0.95 * inch, 1.25 * inch, 1.0 * inch],
        )
    )
    story.append(Image(str(comparison_png), width=6.2 * inch, height=3.72 * inch))
    story.append(Paragraph("Figure 6. Comparison of accuracy and macro F1 across models.", styles["BodySmall"]))

    story.append(Paragraph("Binary Classification Metrics", styles["SectionHead"]))
    story.append(Paragraph("SVM", styles["BodySmall"]))
    story.append(build_table([["Class", "Precision/Accuracy", "Recall/Macro F1", "F1", "Support"]] + metrics_rows(svm_binary), col_widths=[1.3 * inch, 1.4 * inch, 1.4 * inch, 0.8 * inch, 0.8 * inch]))
    story.append(Paragraph("Random Forest", styles["BodySmall"]))
    story.append(build_table([["Class", "Precision/Accuracy", "Recall/Macro F1", "F1", "Support"]] + metrics_rows(rf_binary), col_widths=[1.3 * inch, 1.4 * inch, 1.4 * inch, 0.8 * inch, 0.8 * inch]))
    story.append(Paragraph("CNN", styles["BodySmall"]))
    story.append(build_table([["Class", "Precision/Accuracy", "Recall/Macro F1", "F1", "Support"]] + metrics_rows(cnn_binary), col_widths=[1.3 * inch, 1.4 * inch, 1.4 * inch, 0.8 * inch, 0.8 * inch]))

    story.append(Paragraph("3-Class Classification Metrics", styles["SectionHead"]))
    story.append(Paragraph("SVM", styles["BodySmall"]))
    story.append(build_table([["Class", "Precision/Accuracy", "Recall/Macro F1", "F1", "Support"]] + metrics_rows(svm_3class), col_widths=[1.3 * inch, 1.4 * inch, 1.4 * inch, 0.8 * inch, 0.8 * inch]))
    story.append(Paragraph("Random Forest", styles["BodySmall"]))
    story.append(build_table([["Class", "Precision/Accuracy", "Recall/Macro F1", "F1", "Support"]] + metrics_rows(rf_3class), col_widths=[1.3 * inch, 1.4 * inch, 1.4 * inch, 0.8 * inch, 0.8 * inch]))
    story.append(Paragraph("CNN", styles["BodySmall"]))
    story.append(build_table([["Class", "Precision/Accuracy", "Recall/Macro F1", "F1", "Support"]] + metrics_rows(cnn_3class), col_widths=[1.3 * inch, 1.4 * inch, 1.4 * inch, 0.8 * inch, 0.8 * inch]))

    story.append(Paragraph(f"In the binary LOSO setting, SVM achieved the strongest overall result with accuracy {svm_binary['accuracy']:.4f} and macro F1 {svm_binary['macro_f1']:.4f}. The CNN improved substantially after training for 10 epochs and reached accuracy {cnn_binary['accuracy']:.4f} with macro F1 {cnn_binary['macro_f1']:.4f}, which made it highly competitive with the classical baselines.", styles["BodySmall"]))
    story.append(Paragraph(f"In the 3-class LOSO setting, SVM gave the highest accuracy at {svm_3class['accuracy']:.4f}, while CNN achieved the highest macro F1 at {cnn_3class['macro_f1']:.4f}. This suggests that SVM made the most correct predictions overall, whereas CNN produced the most balanced performance across baseline, stress, and amusement.", styles["BodySmall"]))

    story.append(Paragraph("Confusion Matrix Analysis", styles["SectionHead"]))
    story.append(Paragraph("The confusion matrices provide a more detailed view of model behavior than accuracy and macro F1 alone. In the binary setting, the main source of error is confusion between stress and non-stress windows. In the 3-class setting, amusement is the most challenging class, while baseline and stress are generally recognized more consistently.", styles["BodySmall"]))
    story.append(Paragraph("Binary Confusion Matrices", styles["BodySmall"]))
    story.append(Paragraph("SVM Confusion Matrix", styles["BodySmall"]))
    story.append(Image(str(binary_svm_cm), width=4.9 * inch, height=3.65 * inch))
    story.append(Paragraph("Random Forest Confusion Matrix", styles["BodySmall"]))
    story.append(Image(str(binary_rf_cm), width=4.9 * inch, height=3.65 * inch))
    story.append(Paragraph("CNN Confusion Matrix", styles["BodySmall"]))
    story.append(Image(str(binary_cnn_cm), width=4.9 * inch, height=3.65 * inch))
    story.append(Paragraph("3-Class Confusion Matrices", styles["BodySmall"]))
    story.append(Paragraph("SVM Confusion Matrix", styles["BodySmall"]))
    story.append(Image(str(class3_svm_cm), width=4.9 * inch, height=4.3 * inch))
    story.append(Paragraph("Random Forest Confusion Matrix", styles["BodySmall"]))
    story.append(Image(str(class3_rf_cm), width=4.9 * inch, height=4.3 * inch))
    story.append(Paragraph("CNN Confusion Matrix", styles["BodySmall"]))
    story.append(Image(str(class3_cnn_cm), width=4.9 * inch, height=4.3 * inch))

    story.append(Paragraph("5. Conclusion", styles["SectionHead"]))
    story.append(Paragraph("This study showed that wrist-based PPG from WESAD can support both binary stress recognition and 3-class emotion recognition under a strict LOSO evaluation protocol. The preprocessing pipeline, longer windows, and subject-wise normalization produced stable inputs for all three models.", styles["BodySmall"]))
    story.append(Paragraph("Among the evaluated models, SVM was the strongest binary classifier and also produced the highest 3-class accuracy. The CNN benefited significantly from longer training, and after 10 epochs it achieved the highest 3-class macro F1, indicating improved class balance. Overall, the results show that both classical machine learning and deep learning are viable for PPG-based affect recognition, with the best model depending on whether overall accuracy or balanced class performance is prioritized.", styles["BodySmall"]))

    try:
        doc.build(story)
        return output_path
    except PermissionError:
        for suffix in ["_updated", "_v2", "_v3", "_v4", "_final"]:
            fallback_path = output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
            try:
                doc = SimpleDocTemplate(
                    str(fallback_path),
                    pagesize=letter,
                    leftMargin=0.7 * inch,
                    rightMargin=0.7 * inch,
                    topMargin=0.7 * inch,
                    bottomMargin=0.7 * inch,
                )
                doc.build(story)
                return fallback_path
            except PermissionError:
                continue
        raise


def main() -> None:
    workspace_root = Path(__file__).resolve().parent
    build_final_loso_report(workspace_root)
    output_path = build_final_loso_pdf(workspace_root)
    print(f"Saved LOSO PDF report to: {output_path}")


if __name__ == "__main__":
    main()
