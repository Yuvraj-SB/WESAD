from __future__ import annotations

from pathlib import Path

from docx import Document


SOURCE_DOC = Path(r"C:\Users\Aparjyot S Bhatia\Downloads\Final_Report_LOSO_v2.docx")
OUTPUT_DOC = Path(r"C:\Users\Aparjyot S Bhatia\Downloads\WESAD\WESAD\outputs\Final_Report_LOSO_corrected.docx")


REPLACEMENTS = {
    "The synchronized SX.pkl files from the WESAD dataset were used. Only the wrist-based Blood Volume Pulse (BVP) signal and protocol labels were retained, as this study focuses on evaluating the discriminative power of a single peripheral cardiovascular signal without relying on additional modalities such as EDA or respiration. The BVP signal was recorded at 64 Hz, while the ground-truth labels were recorded at a higher sampling rate. Since the SX.pkl files provide synchronized signals and labels, direct alignment was not required in this implementation.":
    "The synchronized SX.pkl files from the WESAD dataset were used. Only the wrist-based Blood Volume Pulse (BVP) signal and protocol labels were retained, as this study focuses on evaluating the discriminative power of a single peripheral cardiovascular signal without relying on additional modalities such as EDA or respiration. The BVP signal was recorded at 64 Hz, while the ground-truth labels were recorded at a higher sampling rate. Because the signals are synchronized in the SX.pkl files, the labels were mapped onto the BVP timeline so that each retained BVP window had a consistent condition label.",

    "The filtered BVP signal was segmented into fixed-length windows of 60 seconds, corresponding to 3840 samples at 64 Hz. A sliding window with a step size of 5 seconds was used, resulting in a high degree of overlap between consecutive windows. The 60-second window length was chosen to ensure sufficient temporal context for capturing stable physiological patterns in the PPG signal. Shorter windows may not capture enough cardiac cycles, while longer windows reduce the number of training samples. A step size of 5 seconds was used to increase the number of training samples and improve model robustness, while still maintaining temporal continuity. Each window was assigned a label based on the dominant condition within the window. Windows containing mixed labels, typically occurring near transitions between conditions, were discarded to avoid introducing label ambiguity. This ensures that each training sample corresponds to a single, well-defined emotional state.":
    "The filtered BVP signal was segmented into fixed-length windows of 60 seconds, corresponding to 3840 samples at 64 Hz. A sliding window with a step size of 5 seconds was used, resulting in a high degree of overlap between consecutive windows. The 60-second window length was chosen to ensure sufficient temporal context for capturing stable physiological patterns in the PPG signal. Shorter windows may not capture enough cardiac cycles, while longer windows reduce the number of training samples. A step size of 5 seconds was used to increase the number of training samples and improve model robustness, while still maintaining temporal continuity. Only windows containing a single valid label throughout the full interval were retained. Windows containing mixed labels, typically occurring near transitions between conditions, were discarded to avoid introducing label ambiguity. This ensures that each training sample corresponds to a single, well-defined emotional state.",

    "Three classification models were implemented to compare feature-based learning with end-to-end deep learning approaches on wrist-based PPG data. The first two models, Support Vector Machine (SVM) and Random Forest, used the same handcrafted statistical feature set extracted from each window. In contrast, the CNN model operated directly on normalized raw PPG windows, allowing it to learn features automatically from the signal.":
    "Three classification models were implemented to compare feature-based learning with end-to-end deep learning approaches on wrist-based PPG data. The first two models, Support Vector Machine (SVM) and Random Forest, used the same handcrafted 24-feature representation extracted from each window. In contrast, the CNN model operated directly on normalized raw PPG windows, allowing it to learn features automatically from the signal.",

    "For each 60-second filtered PPG window, a set of statistical features was extracted, including mean, standard deviation, minimum, maximum, median, range, signal energy, root mean square (RMS), skewness, and kurtosis. These features capture both the central tendency and variability of the signal, as well as its distribution shape and signal power. This provides a compact but informative representation of the PPG waveform without relying on complex peak detection or domain-specific assumptions, making it a stable and interpretable baseline.":
    "For each 60-second filtered PPG window, a 24-feature handcrafted representation was extracted. This included heart-rate and inter-beat-interval statistics (mean_hr, std_hr, min_hr, max_hr, mean_ibi, std_ibi, min_ibi, max_ibi), variability measures (rmssd, sdnn, nn50, pnn50), peak-based features (num_peaks, peak_density, mean_peak_amp, std_peak_amp), and signal statistics (mean_signal, std_signal, min_signal, max_signal, range_signal, rms_signal, skewness, kurtosis). These features provide a richer summary of both cardiovascular timing patterns and waveform shape than simple global statistics alone.",

    "The Random Forest model was trained on the same handcrafted statistical feature set. The forest consisted of 25 decision trees with a maximum depth of 8, a minimum split size of 10, a minimum leaf size of 5, and square-root feature sampling at each split. These hyperparameters were chosen to balance model complexity and generalization, preventing overfitting while still allowing the model to capture meaningful patterns.":
    "The Random Forest model was trained on the same handcrafted 24-feature representation. The forest consisted of 25 decision trees with a maximum depth of 8, a minimum split size of 10, a minimum leaf size of 5, and square-root feature sampling at each split. These hyperparameters were chosen to balance model complexity and generalization, preventing overfitting while still allowing the model to capture meaningful patterns.",

    "In the binary LOSO setting, SVM achieved the strongest overall result with accuracy 0.8496 and macro F1 0.8185. The CNN improved substantially after training for 10 epochs and reached accuracy 0.8363 with macro F1 0.8206, which made it highly competitive with the classical baselines.":
    "In the binary LOSO setting, SVM achieved the highest accuracy at 0.8496, while CNN achieved the highest macro F1 score at 0.8206. The CNN improved substantially after training for 10 epochs and reached accuracy 0.8363, making it highly competitive with the classical baselines.",

    "The deep learning results were mixed. In the binary LOSO setting, CNN performed strongly after 10 epochs, reaching an accuracy of 0.836 and a macro F1 score of 0.821. However, SVM still achieved the best overall binary result with 0.850 accuracy and 0.819 macro F1. Random Forest followed with 0.824 accuracy and 0.785 macro F1. This shows that both feature-based and end-to-end approaches were effective, with SVM remaining the strongest overall binary model.":
    "The deep learning results were mixed. In the binary LOSO setting, CNN performed strongly after 10 epochs, reaching an accuracy of 0.836 and a macro F1 score of 0.821. SVM achieved the highest binary accuracy at 0.850, while CNN achieved the highest binary macro F1. Random Forest followed with 0.824 accuracy and 0.785 macro F1. This shows that both feature-based and end-to-end approaches were effective, with SVM remaining the strongest accuracy-based binary model and CNN providing the most balanced binary performance.",

    "In the binary LOSO setting, SVM achieved the best overall performance with an accuracy of 0.8496 and macro F1 of 0.8185, indicating that the simple statistical features extracted from the PPG signal are highly effective for separating stress from non-stress. Random Forest performed slightly worse, while CNN achieved comparable macro F1 (0.8206) with slightly lower accuracy (0.8363), showing more balanced performance across classes. Notably, CNN demonstrated higher recall for the stress class, suggesting it is better at detecting stress episodes, likely due to its ability to capture temporal patterns and waveform variations that are not fully represented by handcrafted features.":
    "In the binary LOSO setting, SVM achieved the highest accuracy at 0.8496, indicating that the handcrafted feature representation is highly effective for separating stress from non-stress. Random Forest performed slightly worse, while CNN achieved the highest macro F1 (0.8206) with slightly lower accuracy (0.8363), showing more balanced performance across classes. Notably, CNN demonstrated higher recall for the stress class, suggesting it is better at detecting stress episodes, likely due to its ability to capture temporal patterns and waveform variations that are not fully represented by handcrafted features.",
}


def replace_paragraph_text(paragraph, new_text: str) -> None:
    if not paragraph.runs:
        paragraph.text = new_text
        return
    paragraph.runs[0].text = new_text
    for run in paragraph.runs[1:]:
        run.text = ""


def main() -> None:
    document = Document(str(SOURCE_DOC))
    replaced = 0

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text in REPLACEMENTS:
            replace_paragraph_text(paragraph, REPLACEMENTS[text])
            replaced += 1

    output_candidates = [
        OUTPUT_DOC,
        OUTPUT_DOC.with_name("Final_Report_LOSO_corrected_v2.docx"),
        OUTPUT_DOC.with_name("Final_Report_LOSO_corrected_v3.docx"),
    ]
    saved_path = None
    for candidate in output_candidates:
        try:
            document.save(str(candidate))
            saved_path = candidate
            break
        except PermissionError:
            continue

    if saved_path is None:
        raise PermissionError("Could not save corrected report because all output filenames are locked.")

    print(f"Saved corrected report to: {saved_path}")
    print(f"Paragraphs updated: {replaced}")


if __name__ == "__main__":
    main()
