# WESAD PPG Emotion Recognition Project

## Overview
This repository contains a CSC 591 course project on emotion and stress recognition using the WESAD dataset and wrist-based photoplethysmography (PPG/BVP) signals.

The project includes two parallel experiment tracks:

- A subject-wise train/test split pipeline with `SVM`, `Random Forest`, and `1D CNN`
- A leave-one-subject-out (LOSO) pipeline using the same model families

The main goals were:

- preprocess wrist BVP signals from WESAD
- build both feature-based and end-to-end models
- evaluate binary and 3-class classification settings
- generate submission-ready reports with figures and result summaries

## What Is Included
- preprocessing scripts for single-subject and all-subject workflows
- LOSO evaluation scripts for binary and 3-class settings
- model training scripts for `SVM`, `Random Forest`, and `CNN`
- generated result files in [`outputs`](./outputs)
- final reports in Word and PDF form

## What Is Not Pushed
The raw WESAD files are too large for a normal GitHub repository, so the following files are intentionally excluded:

- `S*/S*.pkl`
- `S*/*_respiban.txt`
- `S*/*_E4_Data.zip`

These files are part of the original dataset and must be downloaded separately by anyone who wants to rerun the full pipeline.

Small metadata files such as `*_readme.txt` and `*_quest.csv` can remain, but the raw sensor payload is not tracked here.

## Dataset Layout Expected By The Scripts
The scripts expect the WESAD folder structure to look like this:

```text
WESAD/
|-- S2/
|   |-- S2.pkl
|   |-- S2_readme.txt
|   `-- S2_quest.csv
|-- S3/
|   |-- S3.pkl
|   `-- ...
|-- outputs/
|-- evaluate_friends_style_binary.py
|-- evaluate_friends_style_3class.py
`-- ...
```

If the excluded raw files are missing, the analysis scripts will not be able to rebuild the datasets from scratch, but the generated outputs already committed to this repo will still be viewable.

## Project Pipelines

### 1. Main Subject-Wise Split Pipeline
This track uses the preprocessed window features and trains:

- `SVM`
- `Random Forest`
- `1D CNN`

Key scripts:

- [`process_s2_pipeline.py`](./process_s2_pipeline.py)
- [`process_all_subjects_pipeline.py`](./process_all_subjects_pipeline.py)
- [`train_svm_variants.py`](./train_svm_variants.py)
- [`train_random_forest_variants.py`](./train_random_forest_variants.py)
- [`train_cnn_variants.py`](./train_cnn_variants.py)

Key outputs:

- [`outputs/Final_ReportCSC591.pdf`](./outputs/Final_ReportCSC591.pdf)
- [`outputs/Final_ReportCSC591.docx`](./outputs/Final_ReportCSC591.docx)

### 2. LOSO Pipeline
This track uses leave-one-subject-out evaluation and compares:

- `SVM`
- `Random Forest`
- `CNN`

Binary task:

- `stress` vs `non-stress`

3-class task:

- `baseline`
- `stress`
- `amusement`

Key scripts:

- [`evaluate_friends_style_binary.py`](./evaluate_friends_style_binary.py)
- [`run_friends_style_binary_cnn_5epochs.py`](./run_friends_style_binary_cnn_5epochs.py)
- [`evaluate_friends_style_3class.py`](./evaluate_friends_style_3class.py)
- [`generate_final_loso_report.py`](./generate_final_loso_report.py)
- [`generate_final_loso_report_pdf.py`](./generate_final_loso_report_pdf.py)

Key outputs:

- [`outputs/Final_Report_LOSO.pdf`](./outputs/Final_Report_LOSO.pdf)
- [`outputs/Final_Report_LOSO.docx`](./outputs/Final_Report_LOSO.docx)

## Dependencies
This project uses lightweight custom implementations for several models, so it does not depend on `scikit-learn`.

Install the main Python packages with:

```bash
pip install -r requirements.txt
```

## How To Run

### Rebuild The Single-Subject Exploration
```bash
python explore_s2_raw_bvp.py
python process_s2_pipeline.py
```

### Rebuild The Main Multi-Subject Pipeline
```bash
python process_all_subjects_pipeline.py
python train_svm_variants.py
python train_random_forest_variants.py
python train_cnn_variants.py
```

### Rebuild The LOSO Binary Experiments
```bash
python evaluate_friends_style_binary.py
python run_friends_style_binary_cnn_5epochs.py --epochs 10
```

### Rebuild The LOSO 3-Class Experiments
```bash
python evaluate_friends_style_3class.py --cnn-epochs 10
```

### Regenerate Reports
```bash
python generate_final_project_report.py
python generate_final_project_report_pdf.py
python generate_final_loso_report.py
python generate_final_loso_report_pdf.py
```

## Important Notes
- The raw dataset is not included in this repo because of GitHub file-size limits.
- The generated results inside [`outputs`](./outputs) were kept so the project remains reviewable even without the raw WESAD data.
- If you want a fully reproducible rerun, place the original WESAD raw subject files back into the `S*` folders before running the preprocessing or evaluation scripts.

## Repository Contents At A Glance
- `process_*` scripts: data loading, filtering, segmentation, and feature extraction
- `train_*` scripts: main split experiments
- `evaluate_friends_style_*` scripts: LOSO experiments
- `generate_*report*` scripts: final report generation
- `outputs/`: saved metrics, confusion matrices, plots, and final reports

## Final Reports
- Main report: [`outputs/Final_ReportCSC591.pdf`](./outputs/Final_ReportCSC591.pdf)
- LOSO report: [`outputs/Final_Report_LOSO.pdf`](./outputs/Final_Report_LOSO.pdf)
