"""Generate a strict zero-shot probe set for basic underwater-acoustic checks.

The generated set is for evaluation only, not for fine-tuning. Every signal sample is
available under a small matrix of conditions:

- noise_level: clean / real_noise_light / real_noise_heavy
- channel_condition: direct / bellhop

Noise, when present, is cut from real local WAV recordings and mixed into the signal
at the target SNR. Bellhop samples are produced by writing a real Bellhop .env file,
running the repository's bellhop.exe, parsing the .arr arrivals, and applying the
arrival amplitudes/delays to the synthetic source waveform.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


FREQUENCY_CHOICES_HZ = [250, 500, 1000, 2000, 4000, 6000]
SIGNAL_TYPE_CHOICES = ["CW", "LFM", "FSK", "BPSK"]
NOISE_LEVELS: dict[str, float | None] = {
    "clean": None,
    "real_noise_light": 20.0,
    "real_noise_heavy": 0.0,
}
CHANNEL_CONDITIONS = ["direct", "bellhop"]


@dataclass(frozen=True)
class ProbeConfig:
    sample_rate_hz: int = 16000
    duration_s: float = 2.0
    seed: int = 20260929
    cw_repeats: int = 1
    lfm_repeats: int = 1
    fsk_repeats: int = 1
    bpsk_count: int = 5
    noise_levels: tuple[str, ...] = ("clean", "real_noise_light", "real_noise_heavy")
    channel_conditions: tuple[str, ...] = ("direct", "bellhop")
    real_noise_root: str = "exclude"
    bellhop_exe: str = "bellhop.exe"


@dataclass(frozen=True)
class BellhopEnvironment:
    name: str
    source: str
    depth_m: tuple[float, ...]
    temperature_c: tuple[float, ...]
    salinity_psu: tuple[float, ...]
    source_depth_m: float
    receiver_depth_m: float
    receiver_range_km: float
    bottom_sound_speed_mps: float = 1600.0
    bottom_density_g_cm3: float = 1.8
    top_option: str = "SVW"
    bottom_option: str = "A"
    run_type: str = "A"
    n_beams: int = 101
    min_angle_deg: float = -30.0
    max_angle_deg: float = 30.0


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def normalize(x: np.ndarray, peak: float = 0.85) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs <= 0:
        return x
    return x / max_abs * peak


def envelope(n: int, sample_rate_hz: int, fade_s: float = 0.03) -> np.ndarray:
    fade_n = max(1, int(fade_s * sample_rate_hz))
    fade_n = min(fade_n, n // 2)
    env = np.ones(n)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_n))
    env[:fade_n] = ramp
    env[-fade_n:] = ramp[::-1]
    return env


def gen_cw(freq_hz: float, cfg: ProbeConfig, rng: np.random.Generator) -> np.ndarray:
    n = int(cfg.sample_rate_hz * cfg.duration_s)
    t = np.arange(n) / cfg.sample_rate_hz
    phase = rng.uniform(0, 2 * np.pi)
    return normalize(np.sin(2 * np.pi * freq_hz * t + phase) * envelope(n, cfg.sample_rate_hz))


def gen_lfm(start_hz: float, end_hz: float, cfg: ProbeConfig) -> np.ndarray:
    n = int(cfg.sample_rate_hz * cfg.duration_s)
    t = np.arange(n) / cfg.sample_rate_hz
    x = signal.chirp(t, f0=start_hz, f1=end_hz, t1=cfg.duration_s, method="linear")
    return normalize(x * envelope(n, cfg.sample_rate_hz))


def gen_fsk(freqs_hz: list[float], cfg: ProbeConfig, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    n = int(cfg.sample_rate_hz * cfg.duration_s)
    symbol_rate = int(rng.choice([20, 25, 40, 50]))
    samples_per_symbol = max(1, int(cfg.sample_rate_hz / symbol_rate))
    symbol_count = int(math.ceil(n / samples_per_symbol))
    symbols = rng.integers(0, len(freqs_hz), size=symbol_count)
    inst_freq = np.repeat([freqs_hz[s] for s in symbols], samples_per_symbol)[:n]
    phase = 2 * np.pi * np.cumsum(inst_freq) / cfg.sample_rate_hz
    return normalize(np.sin(phase) * envelope(n, cfg.sample_rate_hz)), symbols.astype(int).tolist()


def gen_bpsk(carrier_hz: float, cfg: ProbeConfig, rng: np.random.Generator) -> tuple[np.ndarray, str, int]:
    n = int(cfg.sample_rate_hz * cfg.duration_s)
    bit_rate = int(rng.choice([20, 25, 40, 50]))
    samples_per_bit = max(1, int(cfg.sample_rate_hz / bit_rate))
    bit_count = int(math.ceil(n / samples_per_bit))
    bits = rng.integers(0, 2, size=bit_count)
    phase_bits = np.repeat(np.where(bits == 1, 0.0, np.pi), samples_per_bit)[:n]
    t = np.arange(n) / cfg.sample_rate_hz
    x = np.sin(2 * np.pi * carrier_hz * t + phase_bits) * envelope(n, cfg.sample_rate_hz)
    return normalize(x), "".join(str(int(bit)) for bit in bits), bit_rate


def signal_center_frequency_hz(label: dict[str, Any]) -> float:
    signal_type = label["signal_type"]
    if signal_type == "CW":
        return float(label["main_frequency_hz"])
    if signal_type == "LFM":
        return float((label["start_frequency_hz"] + label["end_frequency_hz"]) / 2)
    if signal_type == "FSK":
        return float(np.mean(label["frequencies_hz"]))
    if signal_type == "BPSK":
        return float(label["carrier_frequency_hz"])
    raise ValueError(f"Cannot select Bellhop centre frequency for {signal_type}")


def discover_real_noise_files(root: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*.wav") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No real noise WAV files found under {root}")
    return files


def read_real_noise_snippet(path: Path, sample_rate_hz: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    rate, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    original_dtype = data.dtype
    data = data.astype(np.float64)
    if np.issubdtype(original_dtype, np.integer):
        data = data / max(abs(np.iinfo(original_dtype).min), np.iinfo(original_dtype).max)
    data = normalize(data, peak=1.0)
    if rate != sample_rate_hz:
        gcd = math.gcd(int(sample_rate_hz), int(rate))
        data = signal.resample_poly(data, sample_rate_hz // gcd, rate // gcd)
    if data.size < n_samples:
        repeats = int(math.ceil(n_samples / max(1, data.size)))
        data = np.tile(data, repeats)
    start = int(rng.integers(0, data.size - n_samples + 1))
    snippet = data[start : start + n_samples]
    if rms(snippet) <= 0:
        raise ValueError(f"Noise snippet is silent: {path}")
    return snippet


def mix_real_noise_at_snr(
    clean_signal: np.ndarray,
    noise_files: list[Path],
    cfg: ProbeConfig,
    rng: np.random.Generator,
    target_snr_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    noise_path = noise_files[int(rng.integers(0, len(noise_files)))]
    noise = read_real_noise_snippet(noise_path, cfg.sample_rate_hz, clean_signal.size, rng)
    sig_rms = rms(clean_signal)
    noise_rms = rms(noise)
    if sig_rms <= 0 or noise_rms <= 0:
        raise ValueError("Cannot mix SNR for silent signal or noise")
    scaled_noise = noise / noise_rms * (sig_rms / (10 ** (target_snr_db / 20)))
    mixed = clean_signal + scaled_noise
    actual_snr_db = 20 * np.log10(sig_rms / rms(scaled_noise))
    return normalize(mixed), {
        "noise_source_path": str(noise_path),
        "target_snr_db": target_snr_db,
        "actual_snr_db": round(float(actual_snr_db), 4),
    }


def sound_speed_mackenzie(depth_m: np.ndarray, temperature_c: np.ndarray, salinity_psu: np.ndarray) -> np.ndarray:
    return (
        1449.2
        + 4.6 * temperature_c
        - 0.055 * temperature_c**2
        + 0.00029 * temperature_c**3
        + (1.34 - 0.01 * temperature_c) * (salinity_psu - 35)
        + 0.016 * depth_m
    )


def make_bellhop_environments() -> tuple[BellhopEnvironment, ...]:
    """Use the same water-column template as the existing repository Bellhop scripts."""

    depth = (0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0)
    return (
        BellhopEnvironment(
            name="repo_template_scs_1km_sd20_rd50",
            source="generate_cw_example.m Bellhop template",
            depth_m=depth,
            temperature_c=(25.0, 24.2, 22.8, 19.5, 16.8, 15.4, 15.0),
            salinity_psu=(34.2, 34.3, 34.5, 34.6, 34.7, 34.8, 34.8),
            source_depth_m=20.0,
            receiver_depth_m=50.0,
            receiver_range_km=1.0,
        ),
        BellhopEnvironment(
            name="repo_template_shallow_0p5km_sd40_rd20",
            source="make_cw_parameter_set.m style bounded condition",
            depth_m=depth,
            temperature_c=(26.0, 25.0, 23.3, 20.7, 18.1, 16.2, 14.8),
            salinity_psu=(34.0, 34.1, 34.25, 34.45, 34.65, 34.78, 34.9),
            source_depth_m=40.0,
            receiver_depth_m=20.0,
            receiver_range_km=0.5,
        ),
        BellhopEnvironment(
            name="repo_template_longer_1p5km_sd60_rd50",
            source="make_cw_parameter_set.m style bounded condition",
            depth_m=depth,
            temperature_c=(23.0, 22.4, 21.0, 18.8, 16.6, 15.0, 13.8),
            salinity_psu=(34.4, 34.45, 34.52, 34.62, 34.72, 34.82, 34.9),
            source_depth_m=60.0,
            receiver_depth_m=50.0,
            receiver_range_km=1.5,
        ),
    )


def write_bellhop_env(env_path: Path, title: str, frequency_hz: float, env: BellhopEnvironment) -> dict[str, Any]:
    depth = np.array(env.depth_m, dtype=float)
    temp = np.array(env.temperature_c, dtype=float)
    salt = np.array(env.salinity_psu, dtype=float)
    c = sound_speed_mackenzie(depth, temp, salt)
    box_z = float(depth[-1] + 10)
    box_r = float(env.receiver_range_km + 0.2)
    lines = [
        f"'{title}' ! Title",
        f"{frequency_hz:8.2f}               ! Frequency (Hz)",
        "    1                    ! NMedia",
        f"'{env.top_option}'                 ! Top Option",
        f"{len(depth):5d} 0.00 {depth[-1]:6.2f}    ! N sigma depth",
    ]
    for z_value, c_value in zip(depth, c):
        lines.append(f"\t {z_value:6.2f} {c_value:6.2f}   0.00   1.00   0.00   0.00 /   ! z c cs rho")
    lines.extend(
        [
            f"'{env.bottom_option}'   0.00          ! Bottom Option, sigma",
            f"    {depth[-1]:6.2f} {env.bottom_sound_speed_mps:6.2f}   0.00   {env.bottom_density_g_cm3:4.2f}   0.00   0.00 /   ! lower halfspace",
            "    1                    ! NSD",
            f"    {env.source_depth_m:6f}  /        ! SD(1)  ... (m)",
            "    1                    ! NRD",
            f"    {env.receiver_depth_m:6f}  /      ! RD(1)  ... (m)",
            "    1                    ! NRR",
            f"    {env.receiver_range_km:6f}  /     ! RR(1)  ... (km)",
            f"'{env.run_type}'                    ! Run Type",
            f"{env.n_beams:d}                    ! Nbeams",
            f"{env.min_angle_deg:f} {env.max_angle_deg:f} /    ! angles (degrees)",
            f"0.000000 {box_z:f} {box_r:f}    ! deltas (m) Box.z (m) Box.r (km)",
        ]
    )
    env_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "environment_name": env.name,
        "environment_source": env.source,
        "depth_m": list(env.depth_m),
        "temperature_c": list(env.temperature_c),
        "salinity_psu": list(env.salinity_psu),
        "sound_speed_mps": [round(float(value), 4) for value in c],
        "source_depth_m": env.source_depth_m,
        "receiver_depth_m": env.receiver_depth_m,
        "receiver_range_km": env.receiver_range_km,
        "bottom_sound_speed_mps": env.bottom_sound_speed_mps,
        "bottom_density_g_cm3": env.bottom_density_g_cm3,
        "n_beams": env.n_beams,
        "angle_span_deg": [env.min_angle_deg, env.max_angle_deg],
    }


def run_bellhop(case_dir: Path, stem: str, bellhop_exe: Path) -> Path:
    exe = bellhop_exe.resolve()
    if not exe.is_file():
        raise FileNotFoundError(f"Bellhop executable not found: {exe}")
    result = subprocess.run([str(exe), stem], cwd=case_dir, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Bellhop failed for {stem}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    arr_path = case_dir / f"{stem}.arr"
    if not arr_path.is_file():
        raise FileNotFoundError(f"Bellhop did not create {arr_path}")
    return arr_path


def parse_bellhop_arrivals(arr_path: Path, max_arrivals: int = 500) -> dict[str, Any]:
    tokens = arr_path.read_text(encoding="ascii", errors="ignore").split()
    cursor = 0

    def take_float() -> float:
        nonlocal cursor
        value = float(tokens[cursor])
        cursor += 1
        return value

    def take_int() -> int:
        return int(round(take_float()))

    frequency_hz = take_float()
    nsd = take_int()
    nrd = take_int()
    nrr = take_int()
    source_depths_m = [take_float() for _ in range(nsd)]
    receiver_depths_m = [take_float() for _ in range(nrd)]
    receiver_ranges_m = [take_float() for _ in range(nrr)]
    all_arrivals: list[dict[str, Any]] = []
    for source_index in range(nsd):
        _max_for_source = take_int()
        for receiver_depth_index in range(nrd):
            for receiver_range_index in range(nrr):
                count = min(take_int(), max_arrivals)
                for _ in range(count):
                    amplitude = take_float()
                    phase_deg = take_float()
                    delay_s = take_float()
                    delay_imag_s = take_float()
                    src_angle_deg = take_float()
                    rcv_angle_deg = take_float()
                    top_bounces = take_int()
                    bottom_bounces = take_int()
                    all_arrivals.append(
                        {
                            "amplitude": amplitude,
                            "phase_deg": phase_deg,
                            "delay_s": delay_s,
                            "delay_imag_s": delay_imag_s,
                            "source_angle_deg": src_angle_deg,
                            "receiver_angle_deg": rcv_angle_deg,
                            "top_bounces": top_bounces,
                            "bottom_bounces": bottom_bounces,
                            "source_index": source_index,
                            "receiver_depth_index": receiver_depth_index,
                            "receiver_range_index": receiver_range_index,
                        }
                    )
    if not all_arrivals:
        raise ValueError(f"No Bellhop arrivals found in {arr_path}")
    return {
        "frequency_hz": frequency_hz,
        "source_depths_m": source_depths_m,
        "receiver_depths_m": receiver_depths_m,
        "receiver_ranges_m": receiver_ranges_m,
        "arrivals": all_arrivals,
    }


def apply_bellhop_arrivals(x: np.ndarray, sample_rate_hz: int, arrival_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arrivals = arrival_data["arrivals"]
    min_delay = min(arrival["delay_s"] for arrival in arrivals)
    analytic = signal.hilbert(x)
    y = np.zeros_like(x, dtype=np.float64)
    compact_arrivals: list[dict[str, Any]] = []
    for arrival in arrivals:
        relative_delay = max(0.0, float(arrival["delay_s"] - min_delay))
        shift = int(round(relative_delay * sample_rate_hz))
        if shift >= x.size:
            continue
        amp = float(arrival["amplitude"]) * np.exp(1j * np.deg2rad(float(arrival["phase_deg"])))
        path_signal = np.zeros_like(x, dtype=np.float64)
        path_signal[shift:] = np.real(amp * analytic[: x.size - shift])
        y += path_signal
        compact_arrivals.append(
            {
                "relative_delay_s": round(relative_delay, 8),
                "absolute_delay_s": round(float(arrival["delay_s"]), 8),
                "amplitude": float(arrival["amplitude"]),
                "phase_deg": float(arrival["phase_deg"]),
                "top_bounces": int(arrival["top_bounces"]),
                "bottom_bounces": int(arrival["bottom_bounces"]),
            }
        )
    if rms(y) <= 0:
        raise ValueError("Bellhop arrivals produced a silent received signal")
    return normalize(y), {
        "arrival_count": len(compact_arrivals),
        "reference_delay_s": round(float(min_delay), 8),
        "arrivals": compact_arrivals,
    }


def apply_bellhop_channel(
    x: np.ndarray,
    label: dict[str, Any],
    cfg: ProbeConfig,
    sample_id: str,
    channel_dir: Path,
    env: BellhopEnvironment,
) -> tuple[np.ndarray, dict[str, Any]]:
    channel_dir.mkdir(parents=True, exist_ok=True)
    frequency_hz = signal_center_frequency_hz(label)
    stem = sample_id
    env_path = channel_dir / f"{stem}.env"
    env_meta = write_bellhop_env(env_path, stem, frequency_hz, env)
    arr_path = run_bellhop(channel_dir, stem, Path(cfg.bellhop_exe))
    arrival_data = parse_bellhop_arrivals(arr_path)
    y, arrival_meta = apply_bellhop_arrivals(x, cfg.sample_rate_hz, arrival_data)
    return y, {
        "channel_condition": "bellhop",
        "bellhop_processed": True,
        "bellhop_frequency_hz": frequency_hz,
        "env_path": str(env_path),
        "arr_path": str(arr_path),
        "environment": env_meta,
        **arrival_meta,
    }


def save_wav(path: Path, sample_rate_hz: int, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = normalize(x, peak=0.9)
    wavfile.write(path, sample_rate_hz, (x * np.iinfo(np.int16).max).astype(np.int16))


def save_spectrogram(path: Path, sample_rate_hz: int, x: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.specgram(x, NFFT=512, Fs=sample_rate_hz, noverlap=384, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, sample_rate_hz / 2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_eval_items(sample: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = sample["id"]
    audio_path = sample["audio_path"]
    signal_type = sample["label"]["signal_type"]
    items: list[dict[str, Any]] = [
        {
            "id": f"{sample_id}_type_choice",
            "sample_id": sample_id,
            "audio_path": audio_path,
            "task": "signal_type_choice",
            "prompt": "<audio>这段信号属于 CW、LFM、FSK、BPSK 中的哪一种？只回答一个选项。",
            "choices": SIGNAL_TYPE_CHOICES,
            "expected_answer": signal_type,
            "scoring": {"type": "choice_exact"},
            "condition": sample["condition"],
        }
    ]
    if signal_type == "CW":
        freq = sample["label"]["main_frequency_hz"]
        items.append(
            {
                "id": f"{sample_id}_main_frequency_choice",
                "sample_id": sample_id,
                "audio_path": audio_path,
                "task": "main_frequency_choice",
                "prompt": "<audio>这段单频信号的主频最接近哪个：250 Hz、500 Hz、1000 Hz、2000 Hz、4000 Hz、6000 Hz？只回答一个选项。",
                "choices": [f"{value} Hz" for value in FREQUENCY_CHOICES_HZ],
                "expected_answer": f"{freq} Hz",
                "scoring": {"type": "choice_exact"},
                "condition": sample["condition"],
            }
        )
    elif signal_type == "LFM":
        direction = sample["label"]["direction"]
        items.append(
            {
                "id": f"{sample_id}_lfm_direction_choice",
                "sample_id": sample_id,
                "audio_path": audio_path,
                "task": "lfm_direction_choice",
                "prompt": "<audio>这段 LFM 是升频还是降频？只回答：升频 或 降频。",
                "choices": ["升频", "降频"],
                "expected_answer": direction,
                "scoring": {"type": "choice_exact"},
                "condition": sample["condition"],
            }
        )
    elif signal_type == "FSK":
        order = sample["label"]["fsk_order"]
        items.append(
            {
                "id": f"{sample_id}_fsk_order_choice",
                "sample_id": sample_id,
                "audio_path": audio_path,
                "task": "fsk_order_choice",
                "prompt": "<audio>这段 FSK 更像两频跳变还是四频跳变？只回答：2FSK 或 4FSK。",
                "choices": ["2FSK", "4FSK"],
                "expected_answer": f"{order}FSK",
                "scoring": {"type": "choice_exact"},
                "condition": sample["condition"],
            }
        )
    return items


def build_sample(
    out_dir: Path,
    sample_id: str,
    sample_rate_hz: int,
    x: np.ndarray,
    label: dict[str, Any],
    condition: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    audio_rel = Path("audio") / f"{sample_id}.wav"
    spec_rel = Path("spectrogram") / f"{sample_id}.png"
    meta_rel = Path("metadata") / f"{sample_id}.json"
    save_wav(out_dir / audio_rel, sample_rate_hz, x)
    save_spectrogram(out_dir / spec_rel, sample_rate_hz, x, sample_id)
    sample = {
        "id": sample_id,
        "audio_path": audio_rel.as_posix(),
        "spectrogram_path": spec_rel.as_posix(),
        "metadata_path": meta_rel.as_posix(),
        "sample_rate_hz": sample_rate_hz,
        "duration_s": round(len(x) / sample_rate_hz, 6),
        "label": label,
        "condition": condition,
        "generation": generation,
        "dataset_role": "zero_shot_probe_not_training",
    }
    write_json(out_dir / meta_rel, sample)
    return sample


def generate_base_signals(cfg: ProbeConfig, rng: np.random.Generator) -> list[tuple[np.ndarray, dict[str, Any], dict[str, Any]]]:
    base: list[tuple[np.ndarray, dict[str, Any], dict[str, Any]]] = []
    for freq in FREQUENCY_CHOICES_HZ:
        for _ in range(cfg.cw_repeats):
            base.append((gen_cw(freq, cfg, rng), {"signal_type": "CW", "main_frequency_hz": freq}, {"generator": "cw"}))
    lfm_pairs = [(500, 2500), (2500, 500), (800, 3200), (3200, 800), (1000, 4000), (4000, 1000)]
    for start_hz, end_hz in lfm_pairs:
        for _ in range(cfg.lfm_repeats):
            direction = "升频" if end_hz > start_hz else "降频"
            label = {"signal_type": "LFM", "start_frequency_hz": start_hz, "end_frequency_hz": end_hz, "direction": direction}
            base.append((gen_lfm(start_hz, end_hz, cfg), label, {"generator": "lfm"}))
    fsk_sets = [[800, 2200], [1000, 3000], [700, 1500, 2300, 3100], [1000, 2000, 3000, 4000]]
    for freqs in fsk_sets:
        for _ in range(cfg.fsk_repeats):
            x, symbols = gen_fsk(freqs, cfg, rng)
            label = {"signal_type": "FSK", "fsk_order": len(freqs), "frequencies_hz": freqs}
            base.append((x, label, {"generator": "fsk", "symbol_sequence_prefix": symbols[:32]}))
    bpsk_carriers = [800, 1200, 2000, 3200, 4500]
    for index in range(cfg.bpsk_count):
        carrier_hz = bpsk_carriers[index % len(bpsk_carriers)]
        x, bits, bit_rate = gen_bpsk(carrier_hz, cfg, rng)
        label = {"signal_type": "BPSK", "carrier_frequency_hz": carrier_hz, "symbol_rate": bit_rate}
        base.append((x, label, {"generator": "bpsk", "bit_sequence_prefix": bits[:64]}))
    return base


def generate_dataset(out_dir: Path, cfg: ProbeConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    noise_files = discover_real_noise_files(Path(cfg.real_noise_root))
    bellhop_envs = make_bellhop_environments()
    samples: list[dict[str, Any]] = []
    counter = 1

    for base_x, label, generation in generate_base_signals(cfg, rng):
        for channel_condition in cfg.channel_conditions:
            channel_clean = base_x
            base_condition: dict[str, Any]
            channel_generation = dict(generation)
            sample_id_for_channel = f"probe_{counter:06d}_{channel_condition}"
            if channel_condition == "direct":
                base_condition = {"channel_condition": "direct", "bellhop_processed": False}
            elif channel_condition == "bellhop":
                env = bellhop_envs[(counter - 1) % len(bellhop_envs)]
                channel_clean, bellhop_meta = apply_bellhop_channel(
                    base_x,
                    label,
                    cfg,
                    sample_id_for_channel,
                    out_dir / "bellhop" / sample_id_for_channel,
                    env,
                )
                base_condition = bellhop_meta
            else:
                raise ValueError(f"Unsupported channel condition: {channel_condition}")

            for noise_level in cfg.noise_levels:
                if noise_level not in NOISE_LEVELS:
                    raise ValueError(f"Unsupported noise level: {noise_level}")
                sample_id = f"probe_{counter:06d}"
                counter += 1
                snr_db = NOISE_LEVELS[noise_level]
                condition = {"noise_level": noise_level, **base_condition}
                if snr_db is None:
                    x = channel_clean
                    condition.update({"noise_added": False, "target_snr_db": None, "actual_snr_db": None})
                else:
                    x, noise_meta = mix_real_noise_at_snr(channel_clean, noise_files, cfg, rng, snr_db)
                    condition.update({"noise_added": True, **noise_meta})
                samples.append(build_sample(out_dir, sample_id, cfg.sample_rate_hz, x, label, condition, channel_generation))

    eval_items: list[dict[str, Any]] = []
    for sample in samples:
        eval_items.extend(make_eval_items(sample))

    write_jsonl(out_dir / "manifest.jsonl", samples)
    write_jsonl(out_dir / "eval_items.jsonl", eval_items)
    write_json(out_dir / "dataset_config.json", asdict(cfg))
    write_json(
        out_dir / "noise_sources.json",
        {"real_noise_root": cfg.real_noise_root, "files": [str(path) for path in noise_files]},
    )
    return samples, eval_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a strict zero-shot underwater-acoustic probe set.")
    parser.add_argument("--output", type=Path, default=Path("output") / "zero_shot_probe" / "v1")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--cw-repeats", type=int, default=1)
    parser.add_argument("--lfm-repeats", type=int, default=1)
    parser.add_argument("--fsk-repeats", type=int, default=1)
    parser.add_argument("--bpsk-count", type=int, default=5)
    parser.add_argument("--noise-levels", default=",".join(NOISE_LEVELS), help="Comma-separated levels from: clean,real_noise_light,real_noise_heavy.")
    parser.add_argument("--channel-conditions", default=",".join(CHANNEL_CONDITIONS), help="Comma-separated conditions from: direct,bellhop.")
    parser.add_argument("--real-noise-root", default="exclude")
    parser.add_argument("--bellhop-exe", default="bellhop.exe")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = ProbeConfig(
        sample_rate_hz=args.sample_rate,
        duration_s=args.duration,
        seed=args.seed,
        cw_repeats=args.cw_repeats,
        lfm_repeats=args.lfm_repeats,
        fsk_repeats=args.fsk_repeats,
        bpsk_count=args.bpsk_count,
        noise_levels=tuple(value.strip() for value in args.noise_levels.split(",") if value.strip()),
        channel_conditions=tuple(value.strip() for value in args.channel_conditions.split(",") if value.strip()),
        real_noise_root=args.real_noise_root,
        bellhop_exe=args.bellhop_exe,
    )
    samples, eval_items = generate_dataset(args.output, cfg)
    print(f"Generated {len(samples)} audio samples and {len(eval_items)} eval items in {args.output}")
    print(f"Sample manifest: {args.output / 'manifest.jsonl'}")
    print(f"Evaluation items: {args.output / 'eval_items.jsonl'}")


if __name__ == "__main__":
    main()
