from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm


MAIN_FREQUENCY_CHOICES_HZ = [250, 500, 1000, 2000, 4000, 6000]
FREQUENCY_GRID_STEP_HZ = 500


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio, int(sample_rate)


def magnitude_spectrum(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.array([0.0]), np.array([0.0])
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)
    spectrum_db = 20.0 * np.log10(spectrum / (np.max(spectrum) + 1e-12) + 1e-12)
    return freqs, spectrum_db


def spectrogram_db(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(1024, max(128, audio.size // 8))
    noverlap = nperseg // 2
    spec_freqs, spec_times, spec = signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    spec_db = 20.0 * np.log10(spec / (np.max(spec) + 1e-12) + 1e-12)
    return spec_freqs, spec_times, spec_db


def phase_difference_trace(audio: np.ndarray, sample_rate: int, max_points: int = 2400) -> tuple[np.ndarray, np.ndarray]:
    if audio.size == 0 or sample_rate <= 0:
        return np.array([0.0]), np.array([0.0])
    analytic = signal.hilbert(audio)
    phase = np.unwrap(np.angle(analytic))
    phase_diff = np.diff(phase, prepend=phase[0])
    smooth_window = max(1, int(round(sample_rate * 0.002)))
    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        phase_diff = np.convolve(phase_diff, kernel, mode="same")
    phase_diff = np.clip(phase_diff, -np.pi, np.pi)
    step = max(1, int(np.ceil(audio.size / max_points)))
    time_axis = np.arange(audio.size) / sample_rate
    return time_axis[::step], phase_diff[::step]


def frequency_ticks(freq_limit: float) -> np.ndarray:
    return np.arange(0, freq_limit + FREQUENCY_GRID_STEP_HZ, FREQUENCY_GRID_STEP_HZ)


def robust_phase_limits(values: np.ndarray, min_span: float = 0.3) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -min_span / 2, min_span / 2
    low, high = np.percentile(finite, [0.5, 99.5])
    center = float((low + high) / 2)
    span = max(float(high - low) * 1.2, min_span)
    return max(-np.pi, center - span / 2), min(np.pi, center + span / 2)


def save_composite(audio_path: Path, output_path: Path, max_freq: float | None = None) -> None:
    audio, sample_rate = load_mono(audio_path)
    duration = audio.size / sample_rate if sample_rate else 0.0
    time_axis = np.arange(audio.size) / sample_rate
    freqs, spectrum_db = magnitude_spectrum(audio, sample_rate)
    nperseg = min(1024, max(128, audio.size // 8))
    noverlap = nperseg // 2
    spec_freqs, spec_times, spec = signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    spec_db = 20.0 * np.log10(spec / (np.max(spec) + 1e-12) + 1e-12)
    freq_limit = max_freq or sample_rate / 2

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(time_axis, audio, linewidth=0.8, color="#1f77b4")
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, max(duration, 1e-6))
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(freqs, spectrum_db, linewidth=0.9, color="#d62728")
    axes[1].set_title("Magnitude Spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Relative magnitude (dB)")
    axes[1].set_xlim(0, freq_limit)
    axes[1].set_ylim(-90, 5)
    axes[1].grid(True, alpha=0.25)

    mesh = axes[2].pcolormesh(spec_times, spec_freqs, spec_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
    axes[2].set_title("Linear-Frequency Spectrogram")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_ylim(0, freq_limit)
    fig.colorbar(mesh, ax=axes[2], label="Relative magnitude (dB)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def add_frequency_guides(ax: plt.Axes, freqs: list[int], *, annotate: bool = True) -> None:
    for freq in freqs:
        ax.axvline(freq, color="#444444", linestyle="--", linewidth=0.8, alpha=0.55)
        if annotate:
            ax.text(
                freq,
                2,
                f"{freq} Hz",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="#222222",
            )


def add_horizontal_frequency_guides(ax: plt.Axes, freqs: list[int]) -> None:
    for freq in freqs:
        ax.axhline(freq, color="white", linestyle="--", linewidth=0.7, alpha=0.65)
        ax.text(0.01, freq, f"{freq} Hz", va="bottom", ha="left", fontsize=8, color="white")


def save_mainfreq_guided(audio_path: Path, output_path: Path, max_freq: float | None = None) -> None:
    audio, sample_rate = load_mono(audio_path)
    duration = audio.size / sample_rate if sample_rate else 0.0
    time_axis = np.arange(audio.size) / sample_rate
    freqs, spectrum_db = magnitude_spectrum(audio, sample_rate)
    nperseg = min(1024, max(128, audio.size // 8))
    noverlap = nperseg // 2
    spec_freqs, spec_times, spec = signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    spec_db = 20.0 * np.log10(spec / (np.max(spec) + 1e-12) + 1e-12)
    freq_limit = max_freq or sample_rate / 2
    zoom_limit = min(2500.0, freq_limit)
    low_guides = [freq for freq in MAIN_FREQUENCY_CHOICES_HZ if freq <= zoom_limit]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
    axes[0].plot(time_axis, audio, linewidth=0.8, color="#1f77b4")
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, max(duration, 1e-6))
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(freqs, spectrum_db, linewidth=0.9, color="#d62728")
    add_frequency_guides(axes[1], MAIN_FREQUENCY_CHOICES_HZ)
    axes[1].set_title("Magnitude Spectrum with Candidate Frequency Guides")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Relative magnitude (dB)")
    axes[1].set_xlim(0, freq_limit)
    axes[1].set_ylim(-90, 5)
    axes[1].set_xticks([0, *MAIN_FREQUENCY_CHOICES_HZ, int(freq_limit)])
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(freqs, spectrum_db, linewidth=1.0, color="#d62728")
    add_frequency_guides(axes[2], low_guides)
    axes[2].set_title("Low/Mid Frequency Zoom Spectrum")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Relative magnitude (dB)")
    axes[2].set_xlim(0, zoom_limit)
    axes[2].set_ylim(-90, 5)
    axes[2].set_xticks([0, *low_guides, int(zoom_limit)])
    axes[2].grid(True, alpha=0.35)

    mesh = axes[3].pcolormesh(spec_times, spec_freqs, spec_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
    add_horizontal_frequency_guides(axes[3], low_guides)
    axes[3].set_title("Low/Mid Frequency Zoom Spectrogram")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Frequency (Hz)")
    axes[3].set_ylim(0, zoom_limit)
    axes[3].set_yticks([0, *low_guides, int(zoom_limit)])
    fig.colorbar(mesh, ax=axes[3], label="Relative magnitude (dB)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_spectrogram_only(audio_path: Path, output_path: Path, max_freq: float | None = None) -> None:
    audio, sample_rate = load_mono(audio_path)
    spec_freqs, spec_times, spec_db = spectrogram_db(audio, sample_rate)
    freq_limit = max_freq or sample_rate / 2
    major_ticks = frequency_ticks(freq_limit)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    mesh = ax.pcolormesh(spec_times, spec_freqs, spec_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
    ax.set_title("Linear-Frequency Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, freq_limit)
    ax.set_yticks(major_ticks)
    ax.grid(axis="y", color="white", linestyle="-", linewidth=0.35, alpha=0.35)
    fig.colorbar(mesh, ax=ax, label="Relative magnitude (dB)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_diagnostic_triplet(audio_path: Path, output_path: Path, max_freq: float | None = None) -> None:
    audio, sample_rate = load_mono(audio_path)
    freqs, spectrum_db = magnitude_spectrum(audio, sample_rate)
    phase_times, phase_diff = phase_difference_trace(audio, sample_rate)
    spec_freqs, spec_times, spec_db = spectrogram_db(audio, sample_rate)
    freq_limit = max_freq or sample_rate / 2
    freq_major_ticks = frequency_ticks(freq_limit)
    duration = audio.size / sample_rate if sample_rate else 0.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    axes[0].plot(freqs, spectrum_db, linewidth=0.9, color="#d62728")
    axes[0].set_title("Magnitude Spectrum")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Relative magnitude (dB)")
    axes[0].set_xlim(0, freq_limit)
    axes[0].set_ylim(-90, 5)
    axes[0].set_xticks(freq_major_ticks)
    axes[0].grid(True, color="#777777", linestyle="-", linewidth=0.35, alpha=0.35)

    axes[1].plot(phase_times, phase_diff, linewidth=0.7, color="#1f77b4")
    axes[1].set_title("Instantaneous Phase Difference")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Phase increment (rad/sample)")
    axes[1].set_xlim(0, max(duration, 1e-6))
    axes[1].set_ylim(*robust_phase_limits(phase_diff))
    axes[1].grid(True, color="#777777", linestyle="-", linewidth=0.35, alpha=0.35)

    mesh = axes[2].pcolormesh(spec_times, spec_freqs, spec_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
    axes[2].set_title("Linear-Frequency Spectrogram")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_ylim(0, freq_limit)
    axes[2].set_yticks(freq_major_ticks)
    axes[2].grid(axis="y", color="white", linestyle="-", linewidth=0.35, alpha=0.35)
    fig.colorbar(mesh, ax=axes[2], label="Relative magnitude (dB)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VLM-readable diagnostic images for the probe dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data") / "zero_shot_probe" / "v1")
    parser.add_argument(
        "--image-kind",
        default="composite",
        choices=["composite", "mainfreq_guided", "spectrogram_only", "diagnostic_triplet"],
    )
    parser.add_argument("--max-freq", type=float, default=8000.0)
    parser.add_argument("--manifest-name", default="vlm_image_manifest.jsonl")
    args = parser.parse_args()

    eval_items = read_jsonl(args.dataset / "eval_items.jsonl")
    by_sample: dict[str, str] = {}
    for item in eval_items:
        by_sample[item["sample_id"]] = item["audio_path"]

    manifest = []
    for sample_id, rel_audio in tqdm(sorted(by_sample.items()), desc="generate probe images"):
        audio_path = args.dataset / rel_audio
        image_rel = Path("images") / args.image_kind / f"{sample_id}.png"
        image_path = args.dataset / image_rel
        if args.image_kind == "spectrogram_only":
            save_spectrogram_only(audio_path, image_path, max_freq=args.max_freq)
        elif args.image_kind == "diagnostic_triplet":
            save_diagnostic_triplet(audio_path, image_path, max_freq=args.max_freq)
        elif args.image_kind == "mainfreq_guided":
            save_mainfreq_guided(audio_path, image_path, max_freq=args.max_freq)
        else:
            save_composite(audio_path, image_path, max_freq=args.max_freq)
        manifest.append({"sample_id": sample_id, "audio_path": rel_audio, "image_path": str(image_rel).replace("\\", "/")})

    manifest_path = args.dataset / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Generated {len(manifest)} images under {args.dataset / 'images' / args.image_kind}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
