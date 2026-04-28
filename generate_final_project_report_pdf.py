from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from generate_final_project_report import (
    OUTPUT_DIR_NAME,
    create_cnn_architecture_png,
    create_model_comparison_png,
    create_signal_comparison_png,
    load_json,
)


def build_table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e8fb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
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


def add_metric_rows(model_name: str, three_metrics: dict, binary_metrics: dict) -> list[str]:
    return [
        model_name,
        f"{three_metrics['accuracy']:.4f}",
        f"{three_metrics['macro_f1']:.4f}",
        f"{binary_metrics['accuracy']:.4f}",
        f"{binary_metrics['macro_f1']:.4f}",
    ]


def build_pdf_report(workspace_root: Path) -> Path:
    outputs_dir = workspace_root / OUTPUT_DIR_NAME
    signal_png = outputs_dir / "report_signal_comparison.png"
    comparison_png = outputs_dir / "report_model_comparison.png"
    cnn_architecture_png = outputs_dir / "report_cnn_architecture.png"
    if not signal_png.exists():
        create_signal_comparison_png(workspace_root, signal_png)
    if not comparison_png.exists():
        create_model_comparison_png(outputs_dir, comparison_png)
    if not cnn_architecture_png.exists():
        create_cnn_architecture_png(cnn_architecture_png)

    split = load_json(outputs_dir / "all_subjects_split.json")
    meta = load_json(outputs_dir / "all_subjects_pipeline_metadata.json")
    svm3 = load_json(outputs_dir / "svm_3class_metrics.json")
    rf3 = load_json(outputs_dir / "rf_3class_metrics.json")
    cnn3 = load_json(outputs_dir / "cnn_3class_metrics.json")
    svmb = load_json(outputs_dir / "svm_binary_metrics.json")
    rfb = load_json(outputs_dir / "rf_binary_metrics.json")
    cnnb = load_json(outputs_dir / "cnn_binary_metrics.json")
    subject_summary = pd.read_csv(outputs_dir / "all_subjects_subject_summary.csv")

    baseline_windows = int(subject_summary["baseline_windows"].sum())
    stress_windows = int(subject_summary["stress_windows"].sum())
    amusement_windows = int(subject_summary["amusement_windows"].sum())

    output_path = outputs_dir / "wesad_final_project_report.pdf"
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHead",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            spaceBefore=8,
            spaceAfter=6,
        )
    )

    story = []
    story.append(Paragraph("Emotion Recognition from Photoplethysmography (PPG) Signals Using WESAD", styles["Title"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Student Name(s): ______________________________", styles["BodySmall"]))
    story.append(Paragraph("Course: CSC 491/591", styles["BodySmall"]))
    story.append(Paragraph("Date: April 27, 2026", styles["BodySmall"]))

    story.append(Paragraph("1. Introduction", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Stress and affect recognition from wearable physiological signals is an important problem in mobile health. Among wrist-worn sensors, photoplethysmography (PPG) is especially attractive because it is widely available in consumer devices such as smartwatches. In this project, the WESAD dataset was used to build emotion and stress classification models from the wrist blood volume pulse (BVP) signal.",
            styles["BodySmall"],
        )
    )
    story.append(
        Paragraph(
            "The main goals were to preprocess the wrist BVP signal, convert the long recordings into labeled fixed-length windows, and compare several classification models under a subject-wise evaluation setting. Following both the project guidance and the WESAD literature, the main focus was placed on 3-class classification (baseline vs stress vs amusement) and binary classification (stress vs non-stress).",
            styles["BodySmall"],
        )
    )

    story.append(Paragraph("2. Dataset", styles["SectionHead"]))
    story.append(
        Paragraph(
            "The Wearable Stress and Affect Detection (WESAD) dataset contains multimodal physiological recordings from 15 valid subjects. For this project, only the synchronized SX.pkl files were used because they provide aligned sensor data and labels. From each subject file, the wrist BVP signal and the study-protocol labels were extracted.",
            styles["BodySmall"],
        )
    )
    dataset_table = build_table(
        [
            ["Item", "Value"],
            ["Subjects used", "15 (S2-S17 excluding missing S1 and S12)"],
            ["Signal used", "Wrist BVP (PPG-derived blood volume pulse)"],
            ["BVP sampling rate", "64 Hz"],
            ["Original label sampling rate", "700 Hz"],
            ["Primary class setup", "3-class: baseline, stress, amusement"],
            ["Binary class setup", "stress vs non-stress"],
        ],
        col_widths=[2.1 * inch, 4.4 * inch],
    )
    story.append(dataset_table)
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        Paragraph(
            f"After preprocessing and windowing, the combined dataset contained {meta['num_total_windows']} windows across all subjects before class filtering. For the 3-class experiments, meditation windows were removed, leaving {baseline_windows + stress_windows + amusement_windows} windows in total: {baseline_windows} baseline, {stress_windows} stress, and {amusement_windows} amusement windows.",
            styles["BodySmall"],
        )
    )

    story.append(Paragraph("3. Data Processing", styles["SectionHead"]))
    story.append(
        Paragraph(
            "The raw wrist BVP signal was filtered to reduce baseline wander, high-frequency noise, and motion-related artifacts. A Butterworth-style bandpass filter was applied with a low cutoff of 0.5 Hz, a high cutoff of 4.0 Hz, and order 4. These settings preserve the physiologically useful heart-related component while suppressing slow drift and high-frequency noise.",
            styles["BodySmall"],
        )
    )
    story.append(
        Paragraph(
            "After filtering, the 700 Hz labels were mapped onto the 64 Hz BVP timeline. The continuous recordings were segmented into valid emotion/stress sections, and fixed-length windows were created using a window size of 20 seconds and a step size of 10 seconds. For the final experiments, the standard WESAD benchmark classes baseline, stress, and amusement were used; meditation was removed from the final 3-class and binary tasks.",
            styles["BodySmall"],
        )
    )
    story.append(Image(str(signal_png), width=6.3 * inch, height=3.68 * inch))
    story.append(Paragraph("Figure 1. Example of raw and filtered wrist BVP signal for subject S2 (first 30 seconds).", styles["BodySmall"]))

    story.append(Paragraph("4. Methodology", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Two modeling approaches were used. The first approach was feature-based classification, where each BVP window was summarized by handcrafted statistical features: mean, standard deviation, minimum, maximum, median, range, energy, RMS, skewness, and kurtosis. These features were used by SVM and Random Forest models. The second approach was end-to-end deep learning, where filtered raw BVP windows were used directly as input to a 1D CNN.",
            styles["BodySmall"],
        )
    )
    story.append(Paragraph("4.1 SVM", styles["SubHead"]))
    story.append(
        Paragraph(
            "The SVM baseline was implemented as a one-vs-rest linear classifier. Feature vectors were standardized using the training data statistics. The model was trained using a margin-based optimization procedure with 60 epochs, learning rate 0.01, and regularization 0.001.",
            styles["BodySmall"],
        )
    )
    story.append(Paragraph("4.2 Random Forest", styles["SubHead"]))
    story.append(
        Paragraph(
            "The Random Forest model was trained on the same handcrafted feature vectors. The final configuration used 25 trees, maximum depth 8, minimum samples split 10, minimum samples leaf 5, and square-root feature sampling at each split.",
            styles["BodySmall"],
        )
    )
    story.append(Paragraph("4.3 1D CNN", styles["SubHead"]))
    story.append(
        Paragraph(
            "The deep learning model was a 1D CNN operating on filtered raw BVP windows. Each window was normalized independently using z-score normalization. The architecture used three convolutional blocks followed by adaptive average pooling and fully connected layers. Weighted cross-entropy loss was used to reduce the effect of class imbalance, and the model was trained with Adam on CPU for 15 epochs with batch size 64.",
            styles["BodySmall"],
        )
    )
    cnn_table = build_table(
        [
            ["Layer Group", "Details"],
            ["Input", "1 x 1280 normalized BVP samples (20 s at 64 Hz)"],
            ["Conv Block 1", "Conv1d(1,16,k=7), ReLU, MaxPool1d(2)"],
            ["Conv Block 2", "Conv1d(16,32,k=5), ReLU, MaxPool1d(2)"],
            ["Conv Block 3", "Conv1d(32,64,k=5), ReLU, AdaptiveAvgPool1d(1)"],
            ["Classifier", "Flatten, Dropout, Linear(64->32), ReLU, Linear(32->classes)"],
        ],
        col_widths=[2.0 * inch, 4.5 * inch],
    )
    story.append(cnn_table)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Image(str(cnn_architecture_png), width=6.3 * inch, height=2.16 * inch))
    story.append(Paragraph("Figure 2. 1D CNN architecture used in the end-to-end deep learning model.", styles["BodySmall"]))

    story.append(Paragraph("5. Experimental Setup", styles["SectionHead"]))
    story.append(
        Paragraph(
            "To avoid data leakage, the data was split by subject rather than by windows. The training subjects were S2, S3, S4, S5, S6, S7, S8, S9, S10, and S11, while the test subjects were S13, S14, S15, S16, and S17. Accuracy, macro F1 score, and confusion matrices were used as evaluation metrics.",
            styles["BodySmall"],
        )
    )
    split_table = build_table(
        [
            ["Split", "Subjects"],
            ["Train", ", ".join(split["train_subjects"])],
            ["Test", ", ".join(split["test_subjects"])],
        ],
        col_widths=[1.0 * inch, 5.5 * inch],
    )
    story.append(split_table)

    story.append(Paragraph("6. Results", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Figure 3 summarizes the overall comparison among SVM, Random Forest, and CNN. Table 1 reports the final quantitative results.",
            styles["BodySmall"],
        )
    )
    story.append(Image(str(comparison_png), width=6.3 * inch, height=3.78 * inch))
    story.append(Paragraph("Figure 3. Comparison of SVM, Random Forest, and CNN on 3-class and binary tasks.", styles["BodySmall"]))
    result_table = build_table(
        [
            ["Model", "3-Class Accuracy", "3-Class Macro F1", "Binary Accuracy", "Binary Macro F1"],
            add_metric_rows("SVM", svm3, svmb),
            add_metric_rows("Random Forest", rf3, rfb),
            add_metric_rows("1D CNN", cnn3, cnnb),
        ],
        col_widths=[1.6 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch],
    )
    story.append(result_table)

    story.append(Paragraph("6.1 3-Class Task", styles["SubHead"]))
    story.append(
        Paragraph(
            "For the 3-class task, Random Forest achieved the best overall performance with accuracy 0.6943 and macro F1 0.5922. The SVM achieved moderate performance but failed to identify amusement reliably. The 1D CNN served as a deep learning comparison model, but in its current form it underperformed the feature-based Random Forest model.",
            styles["BodySmall"],
        )
    )
    task3_table = build_table(
        [
            ["Model", "Accuracy", "Macro F1", "Observation"],
            ["SVM", f"{svm3['accuracy']:.4f}", f"{svm3['macro_f1']:.4f}", "Strong on baseline/stress, failed on amusement"],
            ["Random Forest", f"{rf3['accuracy']:.4f}", f"{rf3['macro_f1']:.4f}", "Best overall 3-class model"],
            ["1D CNN", f"{cnn3['accuracy']:.4f}", f"{cnn3['macro_f1']:.4f}", "Very high stress recall, weak baseline separation"],
        ],
        col_widths=[1.2 * inch, 1.0 * inch, 1.0 * inch, 3.2 * inch],
    )
    story.append(task3_table)

    story.append(Paragraph("6.2 Binary Task", styles["SubHead"]))
    story.append(
        Paragraph(
            "For the binary task, all three models performed better than on the 3-class task, which is expected because binary stress recognition is easier than fine-grained emotion separation. Random Forest again performed best overall, while the CNN achieved high recall for the stress class but produced more false positives on the non-stress class.",
            styles["BodySmall"],
        )
    )
    binary_table = build_table(
        [
            ["Model", "Accuracy", "Macro F1", "Observation"],
            ["SVM", f"{svmb['accuracy']:.4f}", f"{svmb['macro_f1']:.4f}", "Good non-stress recognition, weaker stress recall"],
            ["Random Forest", f"{rfb['accuracy']:.4f}", f"{rfb['macro_f1']:.4f}", "Best binary model overall"],
            ["1D CNN", f"{cnnb['accuracy']:.4f}", f"{cnnb['macro_f1']:.4f}", "Very high stress recall, lower non-stress recall"],
        ],
        col_widths=[1.2 * inch, 1.0 * inch, 1.0 * inch, 3.2 * inch],
    )
    story.append(binary_table)

    story.append(Paragraph("7. Discussion", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Several patterns are clear from the results. First, the binary stress vs non-stress problem is easier than the 3-class baseline vs stress vs amusement problem. This agrees with both the original WESAD paper and the hybrid CNN paper, which also report better performance for binary classification. Second, among the current models, Random Forest is the strongest overall approach, suggesting that the handcrafted statistical features already capture useful stress-related information in the wrist BVP signal.",
            styles["BodySmall"],
        )
    )
    story.append(
        Paragraph(
            "The CNN did not outperform Random Forest in the present implementation. This does not invalidate the deep learning approach; instead, it indicates that the current CNN is a baseline end-to-end model and may need stronger feature support, longer windows, LOSO validation, or hybrid feature augmentation to match more advanced literature. The hybrid CNN paper is particularly relevant here because it shows that combining handcrafted features and CNN-learned representations can improve performance over a plain CNN.",
            styles["BodySmall"],
        )
    )
    story.append(
        Paragraph(
            "There are important differences between this project and the reference papers. The current work uses 20-second windows with 10-second overlap, while the reference papers often use 60-second windows and denser sliding steps. The current project also uses a fixed subject-wise train/test split instead of leave-one-subject-out cross-validation. These choices are acceptable for the course project because the TA emphasized subject-wise splitting and did not require exact replication of the papers.",
            styles["BodySmall"],
        )
    )

    story.append(Paragraph("8. Conclusion", styles["SectionHead"]))
    story.append(
        Paragraph(
            "This project built an end-to-end wrist-PPG emotion/stress recognition pipeline using the WESAD dataset. The raw BVP signal was filtered, segmented, windowed, and converted into both feature-based and raw-signal model inputs. Three models were evaluated: SVM, Random Forest, and 1D CNN. Across the final experiments, Random Forest achieved the strongest performance for both the 3-class and binary tasks, while the CNN served as a meaningful deep learning comparison baseline. Overall, the results show that wrist-based PPG contains enough information to support emotion and stress recognition, especially in the binary stress vs non-stress setting.",
            styles["BodySmall"],
        )
    )

    story.append(Paragraph("References", styles["SectionHead"]))
    references = [
        "[1] Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. Proceedings of the 20th ACM International Conference on Multimodal Interaction, 2018.",
        "[2] Nafiul Rashid, Luke Chen, Manik Dautta, Abel Jimenez, Peter Tseng, and Mohammad Abdullah Al Faruque. Feature Augmented Hybrid CNN for Stress Recognition Using Wrist-based Photoplethysmography Sensor. 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), 2021.",
        "[3] CSC 491/591 Course Project Slides and TA Project Explanation Transcript, Spring 2026.",
    ]
    for reference in references:
        story.append(Paragraph(reference, styles["BodySmall"]))

    doc.build(story)
    return output_path


def main() -> None:
    workspace_root = Path(__file__).resolve().parent
    output_path = build_pdf_report(workspace_root)
    print(f"Saved final PDF report to: {output_path}")


if __name__ == "__main__":
    main()
