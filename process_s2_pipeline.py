from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


VALID_LABEL_NAMES = {
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}


@dataclass(frozen=True)
class Segment:
    subject: str
    label_id: int
    label_name: str
    start_sample: int
    end_sample: int

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample + 1


def load_subject_pickle(subject_id: str, workspace_root: Path) -> dict:
    subject_path = workspace_root / subject_id / f"{subject_id}.pkl"
    with subject_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def apply_butterworth_bandpass_fft(
    signal: np.ndarray,
    sampling_rate: int,
    low_cutoff_hz: float = 0.5,
    high_cutoff_hz: float = 4.0,
    order: int = 4,
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
    spectrum = np.fft.rfft(signal)

    safe_frequencies = np.where(frequencies == 0.0, 1e-12, frequencies)
    highpass_response = 1.0 / np.sqrt(
        1.0 + (low_cutoff_hz / safe_frequencies) ** (2 * order)
    )
    lowpass_response = 1.0 / np.sqrt(
        1.0 + (safe_frequencies / high_cutoff_hz) ** (2 * order)
    )
    combined_response = highpass_response * lowpass_response
    combined_response[0] = 0.0

    filtered_spectrum = spectrum * combined_response
    filtered_signal = np.fft.irfft(filtered_spectrum, n=len(signal))
    return filtered_signal.astype(np.float64, copy=False)


def map_labels_to_bvp_timeline(
    labels_700hz: np.ndarray,
    bvp_length: int,
    bvp_sampling_rate: int,
    label_sampling_rate: int,
) -> np.ndarray:
    bvp_indices = np.arange(bvp_length)
    label_indices = np.floor(
        bvp_indices * label_sampling_rate / bvp_sampling_rate
    ).astype(np.int64)
    label_indices = np.clip(label_indices, 0, len(labels_700hz) - 1)
    return labels_700hz[label_indices]


def extract_valid_segments(labels_on_bvp_timeline: np.ndarray, subject: str) -> list[Segment]:
    valid_mask = np.isin(labels_on_bvp_timeline, list(VALID_LABEL_NAMES))
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return []

    segments: list[Segment] = []
    start = int(valid_indices[0])
    current_label = int(labels_on_bvp_timeline[start])
    previous_index = start

    for current_index in valid_indices[1:]:
        current_index = int(current_index)
        label_id = int(labels_on_bvp_timeline[current_index])
        is_new_segment = current_index != previous_index + 1 or label_id != current_label

        if is_new_segment:
            segments.append(
                Segment(
                    subject=subject,
                    label_id=current_label,
                    label_name=VALID_LABEL_NAMES[current_label],
                    start_sample=start,
                    end_sample=previous_index,
                )
            )
            start = current_index
            current_label = label_id

        previous_index = current_index

    segments.append(
        Segment(
            subject=subject,
            label_id=current_label,
            label_name=VALID_LABEL_NAMES[current_label],
            start_sample=start,
            end_sample=previous_index,
        )
    )
    return segments


def compute_window_features(window_signal: np.ndarray) -> dict[str, float]:
    mean_value = float(np.mean(window_signal))
    std_value = float(np.std(window_signal))
    centered = window_signal - mean_value

    if std_value == 0.0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        normalized = centered / std_value
        skewness = float(np.mean(normalized**3))
        kurtosis = float(np.mean(normalized**4) - 3.0)

    squared = window_signal**2
    return {
        "mean": mean_value,
        "std": std_value,
        "min": float(np.min(window_signal)),
        "max": float(np.max(window_signal)),
        "median": float(np.median(window_signal)),
        "range": float(np.max(window_signal) - np.min(window_signal)),
        "energy": float(np.sum(squared)),
        "rms": float(np.sqrt(np.mean(squared))),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def build_window_rows(
    filtered_bvp: np.ndarray,
    segments: list[Segment],
    sampling_rate: int,
    window_seconds: int,
    step_seconds: int,
) -> list[dict[str, float | int | str]]:
    window_samples = window_seconds * sampling_rate
    step_samples = step_seconds * sampling_rate
    rows: list[dict[str, float | int | str]] = []

    for segment_index, segment in enumerate(segments, start=1):
        segment_signal = filtered_bvp[segment.start_sample : segment.end_sample + 1]

        if len(segment_signal) < window_samples:
            continue

        for offset in range(0, len(segment_signal) - window_samples + 1, step_samples):
            window_signal = segment_signal[offset : offset + window_samples]
            global_start = segment.start_sample + offset
            global_end = global_start + window_samples - 1

            row: dict[str, float | int | str] = {
                "subject": segment.subject,
                "segment_index": segment_index,
                "label_id": segment.label_id,
                "label_name": segment.label_name,
                "window_start_sample": global_start,
                "window_end_sample": global_end,
                "window_start_seconds": round(global_start / sampling_rate, 3),
                "window_end_seconds": round((global_end + 1) / sampling_rate, 3),
            }
            row.update(compute_window_features(window_signal))
            rows.append(row)

    return rows


def write_csv(file_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    workspace_root = Path(__file__).resolve().parent
    outputs_dir = workspace_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    subject_id = "S2"
    bvp_sampling_rate = 64
    label_sampling_rate = 700
    window_seconds = 20
    step_seconds = 10

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

    segment_rows = [
        {
            "subject": segment.subject,
            "label_id": segment.label_id,
            "label_name": segment.label_name,
            "start_sample": segment.start_sample,
            "end_sample": segment.end_sample,
            "start_seconds": round(segment.start_sample / bvp_sampling_rate, 3),
            "end_seconds": round((segment.end_sample + 1) / bvp_sampling_rate, 3),
            "num_samples": segment.num_samples,
            "duration_seconds": round(segment.num_samples / bvp_sampling_rate, 3),
        }
        for segment in segments
    ]

    segment_output_path = outputs_dir / "s2_segment_summary.csv"
    windows_output_path = outputs_dir / "s2_window_features.csv"
    metadata_output_path = outputs_dir / "s2_pipeline_metadata.json"

    write_csv(
        segment_output_path,
        segment_rows,
        [
            "subject",
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
        window_rows,
        [
            "subject",
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

    label_counts = {
        VALID_LABEL_NAMES[label_id]: int(np.sum(labels_on_bvp_timeline == label_id))
        for label_id in VALID_LABEL_NAMES
    }
    window_counts: dict[str, int] = {}
    for row in window_rows:
        label_name = str(row["label_name"])
        window_counts[label_name] = window_counts.get(label_name, 0) + 1

    metadata = {
        "subject": subject_id,
        "dataset_files_read": [str(workspace_root / subject_id / f"{subject_id}.pkl")],
        "dataset_files_modified": [],
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
        "mapped_bvp_label_counts": label_counts,
        "num_valid_segments": len(segment_rows),
        "num_windows": len(window_rows),
        "window_counts_by_label": window_counts,
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Processed subject: {subject_id}")
    print("Dataset file read only:")
    print(f"  {workspace_root / subject_id / f'{subject_id}.pkl'}")
    print("Filter applied:")
    print("  4th-order Butterworth-style bandpass (0.5 Hz to 4.0 Hz)")
    print("Valid labels kept:")
    for label_id, label_name in VALID_LABEL_NAMES.items():
        print(f"  {label_id} = {label_name}")
    print(f"Mapped valid BVP label counts: {label_counts}")
    print(f"Number of valid contiguous segments: {len(segment_rows)}")
    print(f"Number of windows created: {len(window_rows)}")
    print(f"Window counts by label: {window_counts}")
    print(f"Saved segment summary to: {segment_output_path}")
    print(f"Saved window features to: {windows_output_path}")
    print(f"Saved pipeline metadata to: {metadata_output_path}")


if __name__ == "__main__":
    main()
