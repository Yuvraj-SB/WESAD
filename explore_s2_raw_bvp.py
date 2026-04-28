from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def load_subject_pickle(subject_id: str, workspace_root: Path) -> dict:
    subject_path = workspace_root / subject_id / f"{subject_id}.pkl"
    with subject_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def summarize_labels(labels: np.ndarray) -> dict[int, int]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique_labels, counts)}


def apply_butterworth_bandpass_fft(
    signal: np.ndarray,
    sampling_rate: int,
    low_cutoff_hz: float = 0.5,
    high_cutoff_hz: float = 4.0,
    order: int = 4,
) -> np.ndarray:
    """Approximate a Butterworth bandpass filter in the frequency domain.

    SciPy is not available in this workspace, so we build the Butterworth-style
    magnitude response directly with NumPy and apply it to the FFT spectrum.
    This is good enough for our visualization and preprocessing walkthrough.
    """
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


def write_svg_plot(time_axis: np.ndarray, signal: np.ndarray, output_path: Path, title: str) -> None:
    width = 1200
    height = 400
    left_margin = 70
    right_margin = 20
    top_margin = 40
    bottom_margin = 50
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin

    x_min = float(time_axis.min())
    x_max = float(time_axis.max()) if len(time_axis) > 1 else x_min + 1.0
    y_min = float(signal.min())
    y_max = float(signal.max()) if len(signal) > 1 else y_min + 1.0

    if y_max == y_min:
        y_max = y_min + 1.0

    def scale_x(value: float) -> float:
        return left_margin + ((value - x_min) / (x_max - x_min)) * plot_width

    def scale_y(value: float) -> float:
        return top_margin + (1.0 - (value - y_min) / (y_max - y_min)) * plot_height

    polyline_points = " ".join(
        f"{scale_x(float(x)):.2f},{scale_y(float(y)):.2f}"
        for x, y in zip(time_axis, signal)
    )

    x_ticks = 6
    y_ticks = 5
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<line x1="{left_margin}" y1="{top_margin + plot_height}" x2="{left_margin + plot_width}" y2="{top_margin + plot_height}" stroke="black" stroke-width="1" />',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + plot_height}" stroke="black" stroke-width="1" />',
    ]

    for tick in range(x_ticks + 1):
        value = x_min + (x_max - x_min) * tick / x_ticks
        x = scale_x(value)
        svg_lines.append(
            f'<line x1="{x:.2f}" y1="{top_margin}" x2="{x:.2f}" y2="{top_margin + plot_height}" stroke="#d9d9d9" stroke-width="1" />'
        )
        svg_lines.append(
            f'<text x="{x:.2f}" y="{top_margin + plot_height + 20}" text-anchor="middle" font-family="Arial" font-size="12">{value:.1f}</text>'
        )

    for tick in range(y_ticks + 1):
        value = y_min + (y_max - y_min) * tick / y_ticks
        y = scale_y(value)
        svg_lines.append(
            f'<line x1="{left_margin}" y1="{y:.2f}" x2="{left_margin + plot_width}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1" />'
        )
        svg_lines.append(
            f'<text x="{left_margin - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{value:.1f}</text>'
        )

    svg_lines.extend(
        [
            f'<polyline fill="none" stroke="#1565c0" stroke-width="1.5" points="{polyline_points}" />',
            f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-family="Arial" font-size="14">Time (seconds)</text>',
            f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">Amplitude</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load one WESAD subject and plot the first raw wrist BVP segment."
    )
    parser.add_argument(
        "--subject",
        default="S2",
        help="Subject folder / file prefix to inspect (default: S2).",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=30,
        help="How many seconds of raw BVP to plot (default: 30).",
    )
    parser.add_argument(
        "--output-format",
        default="svg",
        choices=["svg"],
        help="Plot output format. SVG is used to avoid extra plotting dependencies.",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent
    data = load_subject_pickle(args.subject, workspace_root)

    bvp = np.asarray(data["signal"]["wrist"]["BVP"]).flatten()
    labels = np.asarray(data["label"])
    bvp_sampling_rate = 64
    label_sampling_rate = 700
    filtered_bvp = apply_butterworth_bandpass_fft(
        signal=bvp,
        sampling_rate=bvp_sampling_rate,
        low_cutoff_hz=0.5,
        high_cutoff_hz=4.0,
        order=4,
    )

    plot_samples = min(len(bvp), args.seconds * bvp_sampling_rate)
    time_axis = np.arange(plot_samples) / bvp_sampling_rate

    print(f"Subject: {data['subject']}")
    print(f"BVP shape after flatten: {bvp.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"BVP duration (seconds): {len(bvp) / bvp_sampling_rate:.2f}")
    print(f"Label duration (seconds): {len(labels) / label_sampling_rate:.2f}")
    print(f"Unique labels and counts: {summarize_labels(labels)}")

    outputs_dir = workspace_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    raw_output_path = outputs_dir / f"{args.subject.lower()}_raw_bvp_first_{args.seconds}s.{args.output_format}"
    filtered_output_path = outputs_dir / f"{args.subject.lower()}_filtered_bvp_first_{args.seconds}s.{args.output_format}"

    write_svg_plot(
        time_axis=time_axis,
        signal=bvp[:plot_samples],
        output_path=raw_output_path,
        title=f"Raw Wrist BVP Signal - First {args.seconds} Seconds ({args.subject})",
    )
    write_svg_plot(
        time_axis=time_axis,
        signal=filtered_bvp[:plot_samples],
        output_path=filtered_output_path,
        title=f"Filtered Wrist BVP Signal - First {args.seconds} Seconds ({args.subject})",
    )
    print("Applied Butterworth-style bandpass filter:")
    print("  low cutoff = 0.5 Hz")
    print("  high cutoff = 4.0 Hz")
    print("  order = 4")
    print(f"Saved raw plot to: {raw_output_path}")
    print(f"Saved filtered plot to: {filtered_output_path}")


if __name__ == "__main__":
    main()
