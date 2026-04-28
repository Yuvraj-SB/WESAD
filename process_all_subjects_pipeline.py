from __future__ import annotations

import csv
import json
from pathlib import Path

from process_s2_pipeline import (
    VALID_LABEL_NAMES,
    apply_butterworth_bandpass_fft,
    build_window_rows,
    extract_valid_segments,
    load_subject_pickle,
    map_labels_to_bvp_timeline,
)

import numpy as np


def discover_subject_ids(workspace_root: Path) -> list[str]:
    subject_ids: list[str] = []
    for path in sorted(
        workspace_root.iterdir(),
        key=lambda item: (
            0,
            int(item.name[1:]),
        )
        if item.name.startswith("S") and item.name[1:].isdigit()
        else (1, item.name),
    ):
        if not path.is_dir():
            continue
        if not path.name.startswith("S"):
            continue
        if (path / f"{path.name}.pkl").exists():
            subject_ids.append(path.name)
    return subject_ids


def write_csv(file_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_subject_split(subject_ids: list[str]) -> dict[str, list[str]]:
    train_subjects = subject_ids[:10]
    test_subjects = subject_ids[10:]
    return {
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
    }


def main() -> None:
    workspace_root = Path(__file__).resolve().parent
    outputs_dir = workspace_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    bvp_sampling_rate = 64
    label_sampling_rate = 700
    window_seconds = 20
    step_seconds = 10

    subject_ids = discover_subject_ids(workspace_root)
    subject_split = build_subject_split(subject_ids)
    subject_to_split = {
        subject_id: "train" for subject_id in subject_split["train_subjects"]
    }
    subject_to_split.update(
        {subject_id: "test" for subject_id in subject_split["test_subjects"]}
    )

    all_segment_rows: list[dict[str, object]] = []
    all_window_rows: list[dict[str, object]] = []
    subject_summary_rows: list[dict[str, object]] = []

    for subject_id in subject_ids:
        data = load_subject_pickle(subject_id, workspace_root)
        bvp = np.asarray(data["signal"]["wrist"]["BVP"]).flatten()
        labels_700hz = np.asarray(data["label"])

        filtered_bvp = apply_butterworth_bandpass_fft(
            signal=bvp,
            sampling_rate=bvp_sampling_rate,
            low_cutoff_hz=0.5,
            high_cutoff_hz=4.0,
            order=4,
        )
        labels_on_bvp_timeline = map_labels_to_bvp_timeline(
            labels_700hz=labels_700hz,
            bvp_length=len(filtered_bvp),
            bvp_sampling_rate=bvp_sampling_rate,
            label_sampling_rate=label_sampling_rate,
        )
        segments = extract_valid_segments(labels_on_bvp_timeline, subject_id)
        window_rows = build_window_rows(
            filtered_bvp=filtered_bvp,
            segments=segments,
            sampling_rate=bvp_sampling_rate,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )

        for segment in segments:
            all_segment_rows.append(
                {
                    "subject": segment.subject,
                    "split": subject_to_split[subject_id],
                    "label_id": segment.label_id,
                    "label_name": segment.label_name,
                    "start_sample": segment.start_sample,
                    "end_sample": segment.end_sample,
                    "start_seconds": round(segment.start_sample / bvp_sampling_rate, 3),
                    "end_seconds": round((segment.end_sample + 1) / bvp_sampling_rate, 3),
                    "num_samples": segment.num_samples,
                    "duration_seconds": round(segment.num_samples / bvp_sampling_rate, 3),
                }
            )

        window_counts_by_label: dict[str, int] = {}
        for row in window_rows:
            label_name = str(row["label_name"])
            window_counts_by_label[label_name] = window_counts_by_label.get(label_name, 0) + 1
            row["split"] = subject_to_split[subject_id]
            all_window_rows.append(row)

        mapped_label_counts = {
            VALID_LABEL_NAMES[label_id]: int(np.sum(labels_on_bvp_timeline == label_id))
            for label_id in VALID_LABEL_NAMES
        }
        subject_summary_rows.append(
            {
                "subject": subject_id,
                "split": subject_to_split[subject_id],
                "num_segments": len(segments),
                "num_windows": len(window_rows),
                "baseline_samples": mapped_label_counts["baseline"],
                "stress_samples": mapped_label_counts["stress"],
                "amusement_samples": mapped_label_counts["amusement"],
                "meditation_samples": mapped_label_counts["meditation"],
                "baseline_windows": window_counts_by_label.get("baseline", 0),
                "stress_windows": window_counts_by_label.get("stress", 0),
                "amusement_windows": window_counts_by_label.get("amusement", 0),
                "meditation_windows": window_counts_by_label.get("meditation", 0),
            }
        )

    segment_output_path = outputs_dir / "all_subjects_segment_summary.csv"
    windows_output_path = outputs_dir / "all_subjects_window_features.csv"
    subject_summary_output_path = outputs_dir / "all_subjects_subject_summary.csv"
    split_output_path = outputs_dir / "all_subjects_split.json"
    metadata_output_path = outputs_dir / "all_subjects_pipeline_metadata.json"

    write_csv(
        segment_output_path,
        all_segment_rows,
        [
            "subject",
            "split",
            "label_id",
            "label_name",
            "start_sample",
            "end_sample",
            "start_seconds",
            "end_seconds",
            "num_samples",
            "duration_seconds",
        ],
    )
    write_csv(
        windows_output_path,
        all_window_rows,
        [
            "subject",
            "split",
            "segment_index",
            "label_id",
            "label_name",
            "window_start_sample",
            "window_end_sample",
            "window_start_seconds",
            "window_end_seconds",
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
        ],
    )
    write_csv(
        subject_summary_output_path,
        subject_summary_rows,
        [
            "subject",
            "split",
            "num_segments",
            "num_windows",
            "baseline_samples",
            "stress_samples",
            "amusement_samples",
            "meditation_samples",
            "baseline_windows",
            "stress_windows",
            "amusement_windows",
            "meditation_windows",
        ],
    )

    split_output_path.write_text(json.dumps(subject_split, indent=2), encoding="utf-8")

    metadata = {
        "subjects_processed": subject_ids,
        "dataset_files_modified": [],
        "dataset_files_read": [
            str(workspace_root / subject_id / f"{subject_id}.pkl")
            for subject_id in subject_ids
        ],
        "bvp_sampling_rate_hz": bvp_sampling_rate,
        "label_sampling_rate_hz": label_sampling_rate,
        "filter": {
            "type": "Butterworth-style bandpass via NumPy FFT response",
            "low_cutoff_hz": 0.5,
            "high_cutoff_hz": 4.0,
            "order": 4,
        },
        "windowing": {
            "window_seconds": window_seconds,
            "step_seconds": step_seconds,
            "window_samples": window_seconds * bvp_sampling_rate,
            "step_samples": step_seconds * bvp_sampling_rate,
        },
        "valid_labels": VALID_LABEL_NAMES,
        "subject_split": subject_split,
        "num_total_segments": len(all_segment_rows),
        "num_total_windows": len(all_window_rows),
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Processed subjects: {subject_ids}")
    print("Dataset files read only. No dataset files were modified.")
    print(f"Train subjects: {subject_split['train_subjects']}")
    print(f"Test subjects: {subject_split['test_subjects']}")
    print(f"Total segments: {len(all_segment_rows)}")
    print(f"Total windows: {len(all_window_rows)}")
    print(f"Saved segment summary to: {segment_output_path}")
    print(f"Saved window features to: {windows_output_path}")
    print(f"Saved subject summary to: {subject_summary_output_path}")
    print(f"Saved split definition to: {split_output_path}")
    print(f"Saved metadata to: {metadata_output_path}")


if __name__ == "__main__":
    main()
