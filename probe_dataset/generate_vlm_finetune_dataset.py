"""Generate a balanced VLM SFT dataset for underwater-acoustic diagnostic images.

This script is intentionally separate from ``generate_probe_dataset.py`` because the
probe set is for zero-shot evaluation only. The fine-tuning set uses the same signal,
real-noise, and Bellhop building blocks, but writes Qwen2.5-VL style conversation
JSONL files for supervised fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from generate_probe_dataset import (
    FREQUENCY_CHOICES_HZ,
    NOISE_LEVELS,
    SIGNAL_TYPE_CHOICES,
    ProbeConfig,
    apply_bellhop_channel,
    discover_real_noise_files,
    envelope,
    gen_cw,
    gen_fsk,
    gen_lfm,
    make_bellhop_environments,
    mix_real_noise_at_snr,
    normalize,
    save_wav,
    write_json,
    write_jsonl,
)
from generate_vlm_probe_images import save_diagnostic_triplet


LFM_PAIRS = [(500, 2500), (2500, 500), (800, 3200), (3200, 800), (1000, 4000), (4000, 1000)]
FSK_SETS = {
    2: [[800, 2200], [1000, 3000], [700, 1700], [1200, 3600]],
    4: [[700, 1500, 2300, 3100], [1000, 2000, 3000, 4000], [600, 1400, 2600, 3800], [900, 1800, 2700, 3600]],
}
BPSK_CARRIERS_HZ = [800, 1200, 2000, 3200, 4500]
BPSK_SYMBOL_RATES = [20, 25, 40, 50]
SPLIT_WEIGHTS = {"train": 0.8, "val": 0.1, "test": 0.1}
CONDITION_WEIGHTS = {
    ("clean", "direct"): 0.7,
    ("real_noise_light", "direct"): 0.1,
    ("real_noise_heavy", "direct"): 0.1,
    ("clean", "bellhop"): 0.1,
}
FIELD_COUNT_WEIGHTS = {1: 0.65, 2: 0.25, 3: 0.10}
SINGLE_FORMAT_WEIGHTS = {"choice": 0.35, "open_short": 0.65}
FIELD_WEIGHTS = {
    "signal_type": 0.22,
    "main_frequency_hz": 0.20,
    "lfm_direction": 0.093,
    "fsk_order": 0.093,
    "bpsk_symbol_rate": 0.094,
    "noise_level": 0.12,
    "channel_condition": 0.08,
    "bio_species_or_group": 0.10,
}
MAX_QA_PER_IMAGE = 3


@dataclass(frozen=True)
class FinetuneConfig:
    output: Path = Path("data") / "vlm_finetune" / "v1"
    sample_rate_hz: int = 16000
    duration_s: float = 2.0
    seed: int = 20261001
    samples_per_class: int = 1000
    qa_per_task: int = 1000
    target_qa_items: int | None = None
    real_noise_root: str = "exclude"
    bellhop_exe: str = "bellhop.exe"
    max_freq_hz: float = 8000.0
    overwrite: bool = False


def balanced_counts(total: int, labels: list[Any]) -> dict[Any, int]:
    base = total // len(labels)
    remainder = total % len(labels)
    return {label: base + (1 if index < remainder else 0) for index, label in enumerate(labels)}


def weighted_counts(total: int, weights: dict[Any, float]) -> dict[Any, int]:
    raw = {key: total * weight for key, weight in weights.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def shuffled_cycle(values: list[Any], count: int, rng: np.random.Generator) -> list[Any]:
    repeated = [values[index % len(values)] for index in range(count)]
    rng.shuffle(repeated)
    return repeated


def gen_bpsk_fixed(carrier_hz: float, symbol_rate: int, cfg: ProbeConfig, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    n = int(cfg.sample_rate_hz * cfg.duration_s)
    samples_per_symbol = max(1, int(cfg.sample_rate_hz / symbol_rate))
    symbol_count = int(math.ceil(n / samples_per_symbol))
    bits = rng.integers(0, 2, size=symbol_count)
    phase_bits = np.repeat(np.where(bits == 1, 0.0, np.pi), samples_per_symbol)[:n]
    t = np.arange(n) / cfg.sample_rate_hz
    x = np.sin(2 * np.pi * carrier_hz * t + phase_bits) * envelope(n, cfg.sample_rate_hz)
    return normalize(x), "".join(str(int(bit)) for bit in bits)


def make_condition_schedule(total: int, rng: np.random.Generator) -> list[dict[str, str]]:
    counts = weighted_counts(total, CONDITION_WEIGHTS)
    rows: list[dict[str, str]] = []
    for (noise_level, channel_condition), count in counts.items():
        rows.extend({"noise_level": noise_level, "channel_condition": channel_condition} for _ in range(count))
    rng.shuffle(rows)
    return rows


def make_class_specs(signal_type: str, total: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if signal_type == "CW":
        counts = balanced_counts(total, FREQUENCY_CHOICES_HZ)
        for freq, count in counts.items():
            specs.extend({"signal_type": "CW", "main_frequency_hz": freq, "balance_label": str(freq)} for _ in range(count))
    elif signal_type == "LFM":
        direction_counts = balanced_counts(total, ["升频", "降频"])
        up_pairs = [pair for pair in LFM_PAIRS if pair[1] > pair[0]]
        down_pairs = [pair for pair in LFM_PAIRS if pair[1] < pair[0]]
        for direction, count in direction_counts.items():
            pairs = up_pairs if direction == "升频" else down_pairs
            for start_hz, end_hz in shuffled_cycle(pairs, count, rng):
                specs.append(
                    {
                        "signal_type": "LFM",
                        "start_frequency_hz": start_hz,
                        "end_frequency_hz": end_hz,
                        "direction": direction,
                        "balance_label": direction,
                    }
                )
    elif signal_type == "FSK":
        order_counts = balanced_counts(total, [2, 4])
        for order, count in order_counts.items():
            for freqs in shuffled_cycle(FSK_SETS[order], count, rng):
                specs.append({"signal_type": "FSK", "fsk_order": order, "frequencies_hz": freqs, "balance_label": str(order)})
    elif signal_type == "BPSK":
        rate_counts = balanced_counts(total, BPSK_SYMBOL_RATES)
        for rate, count in rate_counts.items():
            carriers = shuffled_cycle(BPSK_CARRIERS_HZ, count, rng)
            for carrier_hz in carriers:
                specs.append(
                    {
                        "signal_type": "BPSK",
                        "carrier_frequency_hz": carrier_hz,
                        "symbol_rate": rate,
                        "balance_label": str(rate),
                    }
                )
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    conditions = make_condition_schedule(total, rng)
    rng.shuffle(specs)
    for spec, condition in zip(specs, conditions, strict=True):
        spec.update(condition)
    return specs


def assign_splits(specs: list[dict[str, Any]], rng: np.random.Generator) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        groups[(spec["signal_type"], spec["balance_label"])].append(spec)

    for rows in groups.values():
        rng.shuffle(rows)
        split_counts = weighted_counts(len(rows), SPLIT_WEIGHTS)
        cursor = 0
        for split, count in split_counts.items():
            for row in rows[cursor : cursor + count]:
                row["split"] = split
            cursor += count


def synthesize_signal(spec: dict[str, Any], cfg: ProbeConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    signal_type = spec["signal_type"]
    if signal_type == "CW":
        freq = int(spec["main_frequency_hz"])
        return gen_cw(freq, cfg, rng), {"signal_type": "CW", "main_frequency_hz": freq}, {"generator": "cw"}
    if signal_type == "LFM":
        start_hz = int(spec["start_frequency_hz"])
        end_hz = int(spec["end_frequency_hz"])
        direction = spec["direction"]
        return (
            gen_lfm(start_hz, end_hz, cfg),
            {"signal_type": "LFM", "start_frequency_hz": start_hz, "end_frequency_hz": end_hz, "direction": direction},
            {"generator": "lfm"},
        )
    if signal_type == "FSK":
        freqs = [int(value) for value in spec["frequencies_hz"]]
        x, symbols = gen_fsk(freqs, cfg, rng)
        return x, {"signal_type": "FSK", "fsk_order": len(freqs), "frequencies_hz": freqs}, {"generator": "fsk", "symbol_sequence_prefix": symbols[:32]}
    if signal_type == "BPSK":
        carrier_hz = int(spec["carrier_frequency_hz"])
        symbol_rate = int(spec["symbol_rate"])
        x, bits = gen_bpsk_fixed(carrier_hz, symbol_rate, cfg, rng)
        return (
            x,
            {"signal_type": "BPSK", "carrier_frequency_hz": carrier_hz, "symbol_rate": symbol_rate},
            {"generator": "bpsk", "bit_sequence_prefix": bits[:64]},
        )
    raise ValueError(f"Unsupported signal type: {signal_type}")


def save_sample_metadata(path: Path, sample: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate_samples(out_dir: Path, cfg: FinetuneConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    probe_cfg = ProbeConfig(
        sample_rate_hz=cfg.sample_rate_hz,
        duration_s=cfg.duration_s,
        seed=cfg.seed,
        real_noise_root=cfg.real_noise_root,
        bellhop_exe=cfg.bellhop_exe,
    )
    noise_files = discover_real_noise_files(Path(cfg.real_noise_root))
    bellhop_envs = make_bellhop_environments()

    specs: list[dict[str, Any]] = []
    for signal_type in SIGNAL_TYPE_CHOICES:
        specs.extend(make_class_specs(signal_type, cfg.samples_per_class, rng))
    assign_splits(specs, rng)
    rng.shuffle(specs)

    samples: list[dict[str, Any]] = []
    bellhop_counter = 0
    for index, spec in enumerate(specs, start=1):
        sample_id = f"vlmft_{index:06d}"
        x, label, generation = synthesize_signal(spec, probe_cfg, rng)

        channel_condition = spec["channel_condition"]
        if channel_condition == "direct":
            channel_clean = x
            condition: dict[str, Any] = {"channel_condition": "direct", "bellhop_processed": False}
        elif channel_condition == "bellhop":
            env = bellhop_envs[bellhop_counter % len(bellhop_envs)]
            bellhop_counter += 1
            channel_clean, condition = apply_bellhop_channel(
                x,
                label,
                probe_cfg,
                sample_id,
                out_dir / "bellhop" / sample_id,
                env,
            )
            generation = {**generation, "bellhop_environment_name": env.name}
        else:
            raise ValueError(f"Unsupported channel condition: {channel_condition}")

        noise_level = spec["noise_level"]
        condition = {"noise_level": noise_level, **condition}
        snr_db = NOISE_LEVELS[noise_level]
        if snr_db is None:
            y = channel_clean
            condition.update({"noise_added": False, "target_snr_db": None, "actual_snr_db": None})
        else:
            y, noise_meta = mix_real_noise_at_snr(channel_clean, noise_files, probe_cfg, rng, snr_db)
            condition.update({"noise_added": True, **noise_meta})

        audio_rel = Path("audio") / f"{sample_id}.wav"
        image_rel = Path("images") / f"{sample_id}.png"
        metadata_rel = Path("metadata") / f"{sample_id}.json"
        save_wav(out_dir / audio_rel, probe_cfg.sample_rate_hz, y)
        save_diagnostic_triplet(out_dir / audio_rel, out_dir / image_rel, max_freq=cfg.max_freq_hz)

        sample = {
            "id": sample_id,
            "base_id": sample_id,
            "split": spec["split"],
            "audio_path": audio_rel.as_posix(),
            "image_path": image_rel.as_posix(),
            "metadata_path": metadata_rel.as_posix(),
            "sample_rate_hz": probe_cfg.sample_rate_hz,
            "duration_s": cfg.duration_s,
            "label": label,
            "condition": condition,
            "generation": generation,
            "dataset_role": "vlm_finetune_sft",
            "image_kind": "diagnostic_triplet",
        }
        save_sample_metadata(out_dir / metadata_rel, sample)
        samples.append(sample)
        if index == 1 or index % 100 == 0 or index == len(specs):
            print(f"Generated {index}/{len(specs)} samples", flush=True)
    return samples


def diagnostic_prefix() -> str:
    return "图中从上到下分别是幅频图、瞬时相位差图、线性频率时频图。"


def answer_signal_type(sample: dict[str, Any]) -> str:
    return sample["label"]["signal_type"]


def answer_main_frequency(sample: dict[str, Any]) -> str:
    return f"{sample['label']['main_frequency_hz']} Hz"


def answer_lfm_direction(sample: dict[str, Any]) -> str:
    return sample["label"]["direction"]


def answer_fsk_order(sample: dict[str, Any]) -> str:
    return f"{sample['label']['fsk_order']}FSK"


def answer_bpsk_symbol_rate(sample: dict[str, Any]) -> str:
    return f"{sample['label']['symbol_rate']} symbol/s"


def answer_noise_level(sample: dict[str, Any]) -> str:
    return sample["condition"]["noise_level"]


def answer_channel_condition(sample: dict[str, Any]) -> str:
    return sample["condition"]["channel_condition"]


def field_definitions() -> dict[str, dict[str, Any]]:
    return {
        "signal_type": {
            "choice_task": "signal_type_choice",
            "open_task": "signal_type_open_short",
            "choices": SIGNAL_TYPE_CHOICES,
            "answer_fn": answer_signal_type,
            "candidate_fn": lambda sample: True,
            "choice_prompt": lambda choices: diagnostic_prefix() + "这段水声信号属于 CW、LFM、FSK、BPSK 中的哪一种？只回答一个选项。",
            "open_prompt": lambda: diagnostic_prefix() + "这段水声信号属于哪一种调制/信号类型？只回答短答案。",
        },
        "main_frequency_hz": {
            "choice_task": "main_frequency_choice",
            "open_task": "main_frequency_open_short",
            "choices": [f"{value} Hz" for value in FREQUENCY_CHOICES_HZ],
            "answer_fn": answer_main_frequency,
            "candidate_fn": lambda sample: sample["label"].get("signal_type") == "CW",
            "choice_prompt": lambda choices: diagnostic_prefix() + "这段 CW 单频信号的主频最接近哪个：" + "、".join(choices) + "？只回答一个选项。",
            "open_prompt": lambda: diagnostic_prefix() + "这段 CW 单频信号的主频是多少？只回答短答案，格式如 1000 Hz。",
        },
        "lfm_direction": {
            "choice_task": "lfm_direction_choice",
            "open_task": "lfm_direction_open_short",
            "choices": ["升频", "降频"],
            "answer_fn": answer_lfm_direction,
            "candidate_fn": lambda sample: sample["label"].get("signal_type") == "LFM",
            "choice_prompt": lambda choices: diagnostic_prefix() + "这段 LFM 信号是升频还是降频？只回答：升频 或 降频。",
            "open_prompt": lambda: diagnostic_prefix() + "这段 LFM 信号的扫频方向是什么？只回答短答案。",
        },
        "fsk_order": {
            "choice_task": "fsk_order_choice",
            "open_task": "fsk_order_open_short",
            "choices": ["2FSK", "4FSK"],
            "answer_fn": answer_fsk_order,
            "candidate_fn": lambda sample: sample["label"].get("signal_type") == "FSK",
            "choice_prompt": lambda choices: diagnostic_prefix() + "这段 FSK 信号更像两频跳变还是四频跳变？只回答：2FSK 或 4FSK。",
            "open_prompt": lambda: diagnostic_prefix() + "这段 FSK 信号的频移键控阶数是什么？只回答短答案。",
        },
        "bpsk_symbol_rate": {
            "choice_task": "bpsk_symbol_rate_choice",
            "open_task": "bpsk_symbol_rate_open_short",
            "choices": [f"{value} symbol/s" for value in BPSK_SYMBOL_RATES],
            "answer_fn": answer_bpsk_symbol_rate,
            "candidate_fn": lambda sample: sample["label"].get("signal_type") == "BPSK",
            "choice_prompt": lambda choices: diagnostic_prefix() + "这段 BPSK 信号的相位跳变码元率最接近哪个：" + "、".join(choices) + "？只回答一个选项。",
            "open_prompt": lambda: diagnostic_prefix() + "这段 BPSK 信号的码元率是多少？只回答短答案，格式如 25 symbol/s。",
        },
        "noise_level": {
            "choice_task": "noise_level_choice",
            "open_task": "noise_level_open_short",
            "choices": list(NOISE_LEVELS),
            "answer_fn": answer_noise_level,
            "candidate_fn": lambda sample: True,
            "choice_prompt": lambda choices: diagnostic_prefix() + "这段样本的噪声等级最接近哪个：" + "、".join(choices) + "？只回答一个选项。",
            "open_prompt": lambda: diagnostic_prefix() + "这段样本的噪声等级是什么？只回答短答案。",
        },
        "channel_condition": {
            "choice_task": "channel_condition_choice",
            "open_task": "channel_condition_open_short",
            "choices": ["direct", "bellhop"],
            "answer_fn": answer_channel_condition,
            "candidate_fn": lambda sample: True,
            "choice_prompt": lambda choices: diagnostic_prefix() + "这段样本更像直达信道还是 Bellhop 多径信道：" + "、".join(choices) + "？只回答一个选项。",
            "open_prompt": lambda: diagnostic_prefix() + "这段样本的信道条件是什么？只回答短答案。",
        },
    }


def task_definitions() -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for field, spec in field_definitions().items():
        definitions[spec["choice_task"]] = {**spec, "field": field, "answer_format": "choice"}
        definitions[spec["open_task"]] = {**spec, "field": field, "answer_format": "open_short"}
    definitions["multi_field_open_json"] = {"field": "multi", "answer_format": "open_json"}
    return definitions


def available_fields(sample: dict[str, Any]) -> list[str]:
    return [field for field, spec in field_definitions().items() if spec["candidate_fn"](sample)]


def normalized_field_weights(fields: list[str]) -> dict[str, float]:
    weights = {field: FIELD_WEIGHTS[field] for field in fields if field in FIELD_WEIGHTS and field != "bio_species_or_group"}
    total = sum(weights.values())
    if total <= 0:
        return {field: 1.0 / len(fields) for field in fields}
    return {field: weight / total for field, weight in weights.items()}


def choose_fields(fields: list[str], count: int, rng: np.random.Generator) -> list[str]:
    weights = normalized_field_weights(fields)
    ordered_fields = list(weights)
    probabilities = np.array([weights[field] for field in ordered_fields], dtype=float)
    probabilities = probabilities / probabilities.sum()
    selected = rng.choice(ordered_fields, size=count, replace=False, p=probabilities)
    return [str(field) for field in selected]


def target_qa_count(cfg: FinetuneConfig) -> int:
    if cfg.target_qa_items is not None:
        return cfg.target_qa_items
    return cfg.qa_per_task * 5


def make_choice_prompt(field: str) -> str:
    spec = field_definitions()[field]
    return spec["choice_prompt"](list(spec["choices"]))


def make_open_prompt(field: str) -> str:
    return field_definitions()[field]["open_prompt"]()


def make_multi_prompt(fields: list[str]) -> str:
    return (
        diagnostic_prefix()
        + "请同时判断以下字段："
        + "、".join(fields)
        + "。只输出 JSON 对象，字段名必须与题目中给出的字段名一致，不要解释。"
    )


def field_answer(sample: dict[str, Any], field: str) -> str:
    return str(field_definitions()[field]["answer_fn"](sample))


def field_answers_json(sample: dict[str, Any], fields: list[str]) -> str:
    return json.dumps({field: field_answer(sample, field) for field in fields}, ensure_ascii=False, separators=(",", ":"))


def make_single_qa(
    sample: dict[str, Any],
    field: str,
    answer_format: str,
    sequence: int,
) -> dict[str, Any]:
    spec = field_definitions()[field]
    task = spec["choice_task"] if answer_format == "choice" else spec["open_task"]
    choices = list(spec["choices"]) if answer_format == "choice" else []
    return {
        "id": f"{sample['id']}_{task}_{sequence:04d}",
        "sample_id": sample["id"],
        "base_id": sample["base_id"],
        "split": sample["split"],
        "image_path": sample["image_path"],
        "audio_path": sample["audio_path"],
        "task": task,
        "prompt": make_choice_prompt(field) if answer_format == "choice" else make_open_prompt(field),
        "choices": choices,
        "fields": [field],
        "field_count": 1,
        "answer_format": answer_format,
        "expected_answer": field_answer(sample, field),
        "scoring": {"type": "choice_exact" if answer_format == "choice" else "open_short_exact"},
        "condition": sample["condition"],
        "label": sample["label"],
    }


def make_multi_qa(sample: dict[str, Any], fields: list[str], sequence: int) -> dict[str, Any]:
    signature = "+".join(fields)
    return {
        "id": f"{sample['id']}_multi_field_open_json_{sequence:04d}",
        "sample_id": sample["id"],
        "base_id": sample["base_id"],
        "split": sample["split"],
        "image_path": sample["image_path"],
        "audio_path": sample["audio_path"],
        "task": "multi_field_open_json",
        "prompt": make_multi_prompt(fields),
        "choices": [],
        "fields": fields,
        "field_count": len(fields),
        "field_signature": signature,
        "answer_format": "open_json",
        "expected_answer": field_answers_json(sample, fields),
        "scoring": {"type": "json_exact_fields", "fields": fields},
        "condition": sample["condition"],
        "label": sample["label"],
    }


def sorted_by_reuse(candidates: list[dict[str, Any]], image_counts: Counter[str], rng: np.random.Generator) -> list[dict[str, Any]]:
    order = list(candidates)
    rng.shuffle(order)
    order.sort(key=lambda sample: image_counts[sample["image_path"]])
    return order


def select_samples_for_answer(
    samples: list[dict[str, Any]],
    field: str,
    answer: str,
    count: int,
    image_counts: Counter[str],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidates = [
        sample
        for sample in samples
        if field in available_fields(sample) and field_answer(sample, field) == answer and image_counts[sample["image_path"]] < MAX_QA_PER_IMAGE
    ]
    if not candidates:
        raise ValueError(f"No candidates for field={field!r}, answer={answer!r}")
    while len(selected) < count:
        usable = [sample for sample in candidates if image_counts[sample["image_path"]] < MAX_QA_PER_IMAGE]
        if not usable:
            raise ValueError(f"Not enough reusable images for field={field!r}, answer={answer!r}; need {count}, got {len(selected)}")
        for sample in sorted_by_reuse(usable, image_counts, rng):
            selected.append(sample)
            image_counts[sample["image_path"]] += 1
            if len(selected) == count:
                break
    return selected


def make_single_qa_items(
    samples: list[dict[str, Any]],
    count: int,
    answer_format: str,
    image_counts: Counter[str],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    field_weights = normalized_field_weights(list(field_definitions()))
    field_counts = weighted_counts(count, field_weights)
    qa_items: list[dict[str, Any]] = []
    sequence = 0
    for field, field_count in field_counts.items():
        spec = field_definitions()[field]
        answers = [answer for answer in spec["choices"] if any(field in available_fields(sample) and field_answer(sample, field) == answer for sample in samples)]
        answer_counts = balanced_counts(field_count, answers)
        for answer, answer_count in answer_counts.items():
            selected = select_samples_for_answer(samples, field, str(answer), answer_count, image_counts, rng)
            for sample in selected:
                sequence += 1
                qa_items.append(make_single_qa(sample, field, answer_format, sequence))
    return qa_items


def make_multi_qa_items(
    samples: list[dict[str, Any]],
    count: int,
    field_count: int,
    image_counts: Counter[str],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    qa_items: list[dict[str, Any]] = []
    sequence = 0
    attempts = 0
    candidates = [sample for sample in samples if len(available_fields(sample)) >= field_count]
    while len(qa_items) < count:
        attempts += 1
        if attempts > count * 100:
            raise ValueError(f"Unable to sample {count} multi-field QA items with field_count={field_count}")
        usable = [sample for sample in candidates if image_counts[sample["image_path"]] < MAX_QA_PER_IMAGE]
        if not usable:
            raise ValueError(f"No reusable images left for multi-field QA with field_count={field_count}")
        sample = sorted_by_reuse(usable, image_counts, rng)[0]
        fields = choose_fields(available_fields(sample), field_count, rng)
        image_counts[sample["image_path"]] += 1
        sequence += 1
        qa_items.append(make_multi_qa(sample, fields, sequence))
    return qa_items


def make_qa_items(samples: list[dict[str, Any]], cfg: FinetuneConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed + 17)
    total = target_qa_count(cfg)
    field_count_counts = weighted_counts(total, FIELD_COUNT_WEIGHTS)
    single_format_counts = weighted_counts(field_count_counts[1], SINGLE_FORMAT_WEIGHTS)
    image_counts: Counter[str] = Counter()
    qa_items: list[dict[str, Any]] = []
    qa_items.extend(make_single_qa_items(samples, single_format_counts["choice"], "choice", image_counts, rng))
    qa_items.extend(make_single_qa_items(samples, single_format_counts["open_short"], "open_short", image_counts, rng))
    qa_items.extend(make_multi_qa_items(samples, field_count_counts[2], 2, image_counts, rng))
    qa_items.extend(make_multi_qa_items(samples, field_count_counts[3], 3, image_counts, rng))
    rng.shuffle(qa_items)
    return qa_items


def to_qwen_record(qa: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": qa["id"],
        "images": [qa["image_path"]],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": qa["image_path"]},
                    {"type": "text", "text": qa["prompt"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": qa["expected_answer"]}]},
        ],
        "metadata": {
            "sample_id": qa["sample_id"],
            "base_id": qa["base_id"],
            "task": qa["task"],
            "choices": qa["choices"],
            "fields": qa["fields"],
            "field_count": qa["field_count"],
            "answer_format": qa["answer_format"],
            "field_signature": qa.get("field_signature"),
            "scoring": qa["scoring"],
            "condition": qa["condition"],
            "label": qa["label"],
        },
    }


def counter_dict(values: list[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def validate_dataset(samples: list[dict[str, Any]], qa_items: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    class_counts = Counter(sample["label"]["signal_type"] for sample in samples)
    if max(class_counts.values()) - min(class_counts.values()) > 1:
        errors.append(f"Signal classes are imbalanced: {dict(class_counts)}")

    if not qa_items:
        errors.append("No QA items were generated")
        return errors

    target_total = len(qa_items)
    expected_field_counts = weighted_counts(target_total, FIELD_COUNT_WEIGHTS)
    actual_field_counts = Counter(item["field_count"] for item in qa_items)
    for field_count, expected in expected_field_counts.items():
        actual = actual_field_counts[field_count]
        tolerance = max(2, int(math.ceil(expected * 0.1)))
        if abs(actual - expected) > tolerance:
            errors.append(f"Field-count distribution drifted for {field_count}: expected about {expected}, got {actual}")

    single_rows = [item for item in qa_items if item["field_count"] == 1]
    expected_single_formats = weighted_counts(len(single_rows), SINGLE_FORMAT_WEIGHTS)
    actual_single_formats = Counter(item["answer_format"] for item in single_rows)
    for answer_format, expected in expected_single_formats.items():
        actual = actual_single_formats[answer_format]
        tolerance = max(2, int(math.ceil(expected * 0.1)))
        if abs(actual - expected) > tolerance:
            errors.append(f"Single-answer format distribution drifted for {answer_format}: expected about {expected}, got {actual}")

    base_splits: dict[str, set[str]] = defaultdict(set)
    image_splits: dict[str, set[str]] = defaultdict(set)
    for item in qa_items:
        if item["answer_format"] == "choice" and item["expected_answer"] not in item["choices"]:
            errors.append(f"Answer not in choices for {item['id']}")
        if item["answer_format"] == "open_short" and ("{" in item["expected_answer"] or "。" in item["expected_answer"]):
            errors.append(f"Open-short answer is not short for {item['id']}: {item['expected_answer']!r}")
        if item["answer_format"] == "open_json":
            try:
                parsed = json.loads(item["expected_answer"])
            except json.JSONDecodeError as exc:
                errors.append(f"JSON answer is invalid for {item['id']}: {exc}")
            else:
                if set(parsed) != set(item["fields"]):
                    errors.append(f"JSON fields mismatch for {item['id']}: expected {item['fields']}, got {sorted(parsed)}")
        if len(item["fields"]) != item["field_count"]:
            errors.append(f"Field count mismatch for {item['id']}")
        base_splits[item["base_id"]].add(item["split"])
        image_splits[item["image_path"]].add(item["split"])
    leaked_bases = [base_id for base_id, splits in base_splits.items() if len(splits) > 1]
    leaked_images = [image for image, splits in image_splits.items() if len(splits) > 1]
    if leaked_bases:
        errors.append(f"Base IDs leak across splits: {leaked_bases[:5]}")
    if leaked_images:
        errors.append(f"Images leak across splits: {leaked_images[:5]}")

    image_qa_counts = Counter(item["image_path"] for item in qa_items)
    overused_images = [image for image, count in image_qa_counts.items() if count > MAX_QA_PER_IMAGE]
    if overused_images:
        errors.append(f"Images exceed max QA reuse ({MAX_QA_PER_IMAGE}): {overused_images[:5]}")

    for sample in samples:
        condition = sample["condition"]
        if condition["noise_level"] != "clean" and condition.get("actual_snr_db") is None:
            errors.append(f"Missing SNR metadata for noisy sample {sample['id']}")
        if condition["channel_condition"] == "bellhop" and not condition.get("bellhop_processed"):
            errors.append(f"Missing Bellhop flag for {sample['id']}")
    return errors


def write_split_files(out_dir: Path, qa_items: list[dict[str, Any]]) -> None:
    for split in SPLIT_WEIGHTS:
        rows = [to_qwen_record(item) for item in qa_items if item["split"] == split]
        write_jsonl(out_dir / f"{split}.jsonl", rows)


def write_report(out_dir: Path, cfg: FinetuneConfig, samples: list[dict[str, Any]], qa_items: list[dict[str, Any]], errors: list[str]) -> None:
    sample_condition_counts = Counter(f"{sample['condition']['noise_level']}/{sample['condition']['channel_condition']}" for sample in samples)
    single_rows = [item for item in qa_items if item["field_count"] == 1]
    field_counts = Counter(field for item in qa_items for field in item["fields"])
    single_field_counts = Counter(item["fields"][0] for item in single_rows)
    answer_format_counts = Counter(item["answer_format"] for item in qa_items)
    field_count_counts = Counter(item["field_count"] for item in qa_items)
    image_reuse_counts = Counter(Counter(item["image_path"] for item in qa_items).values())
    lines = [
        "# VLM Fine-tuning Dataset Report",
        "",
        "## Config",
        "",
        f"- samples_per_class: `{cfg.samples_per_class}`",
        f"- qa_per_task: `{cfg.qa_per_task}`",
        f"- target_qa_items: `{target_qa_count(cfg)}`",
        f"- sample_rate_hz: `{cfg.sample_rate_hz}`",
        f"- duration_s: `{cfg.duration_s}`",
        f"- image_kind: `diagnostic_triplet`",
        f"- field_count_weights: `{FIELD_COUNT_WEIGHTS}`",
        f"- single_format_weights: `{SINGLE_FORMAT_WEIGHTS}`",
        f"- max_qa_per_image: `{MAX_QA_PER_IMAGE}`",
        "",
        "## Sample Distribution",
        "",
        f"- total samples: `{len(samples)}`",
        f"- by signal type: `{counter_dict([sample['label']['signal_type'] for sample in samples])}`",
        f"- by split: `{counter_dict([sample['split'] for sample in samples])}`",
        f"- by condition: `{dict(sorted(sample_condition_counts.items()))}`",
        "",
        "## QA Distribution",
        "",
        f"- total QA: `{len(qa_items)}`",
        f"- by task: `{counter_dict([item['task'] for item in qa_items])}`",
        f"- by answer format: `{dict(sorted(answer_format_counts.items()))}`",
        f"- by field count: `{dict(sorted(field_count_counts.items()))}`",
        f"- by field occurrence: `{dict(sorted(field_counts.items()))}`",
        f"- by single field: `{dict(sorted(single_field_counts.items()))}`",
        f"- by QA reuse per image: `{dict(sorted(image_reuse_counts.items()))}`",
        f"- by split: `{counter_dict([item['split'] for item in qa_items])}`",
        "",
        "## Answer Distribution By Task",
        "",
    ]
    for task in sorted(set(item["task"] for item in qa_items)):
        task_rows = [item for item in qa_items if item["task"] == task]
        if task == "multi_field_open_json":
            lines.append(f"- {task} signatures: `{counter_dict([item['field_signature'] for item in task_rows])}`")
        else:
            lines.append(f"- {task}: `{counter_dict([item['expected_answer'] for item in task_rows])}`")
    lines.extend(["", "## Validation", ""])
    if errors:
        lines.extend(f"- ERROR: {error}" for error in errors)
    else:
        lines.append("- OK: all validation checks passed.")
    (out_dir / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_dataset(cfg: FinetuneConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = cfg.output
    if out_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = generate_samples(out_dir, cfg)
    qa_items = make_qa_items(samples, cfg)
    errors = validate_dataset(samples, qa_items)

    write_jsonl(out_dir / "manifest.jsonl", samples)
    write_jsonl(out_dir / "qa_items.jsonl", qa_items)
    write_split_files(out_dir, qa_items)
    write_json(
        out_dir / "split_manifest.json",
        {
            "sample_splits": {sample["id"]: sample["split"] for sample in samples},
            "split_weights": SPLIT_WEIGHTS,
            "condition_weights": {f"{noise}/{channel}": weight for (noise, channel), weight in CONDITION_WEIGHTS.items()},
            "field_count_weights": FIELD_COUNT_WEIGHTS,
            "single_format_weights": SINGLE_FORMAT_WEIGHTS,
            "field_weights": FIELD_WEIGHTS,
            "max_qa_per_image": MAX_QA_PER_IMAGE,
        },
    )
    write_json(out_dir / "dataset_config.json", {**asdict(cfg), "output": str(cfg.output)})
    write_report(out_dir, cfg, samples, qa_items, errors)

    if errors:
        raise RuntimeError("Dataset validation failed; see dataset_report.md")
    return samples, qa_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a balanced Qwen2.5-VL SFT dataset for underwater-acoustic diagnostic images.")
    parser.add_argument("--output", type=Path, default=Path("data") / "vlm_finetune" / "v1")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--qa-per-task", type=int, default=1000, help="Compatibility default used as 5 * qa_per_task when --target-qa-items is omitted.")
    parser.add_argument("--target-qa-items", type=int, default=None, help="Total QA items to generate with the v2 question-type probability distribution.")
    parser.add_argument("--real-noise-root", default="exclude")
    parser.add_argument("--bellhop-exe", default="bellhop.exe")
    parser.add_argument("--max-freq", type=float, default=8000.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = FinetuneConfig(
        output=args.output,
        sample_rate_hz=args.sample_rate,
        duration_s=args.duration,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        qa_per_task=args.qa_per_task,
        target_qa_items=args.target_qa_items,
        real_noise_root=args.real_noise_root,
        bellhop_exe=args.bellhop_exe,
        max_freq_hz=args.max_freq,
        overwrite=args.overwrite,
    )
    samples, qa_items = generate_dataset(cfg)
    print(f"Generated {len(samples)} samples and {len(qa_items)} QA items in {cfg.output}")
    print(f"Qwen2.5-VL files: {cfg.output / 'train.jsonl'}, {cfg.output / 'val.jsonl'}, {cfg.output / 'test.jsonl'}")
    print(f"Report: {cfg.output / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
