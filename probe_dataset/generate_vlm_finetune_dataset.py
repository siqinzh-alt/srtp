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
from typing import Any, Callable

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


@dataclass(frozen=True)
class FinetuneConfig:
    output: Path = Path("output") / "vlm_finetune" / "v1"
    sample_rate_hz: int = 16000
    duration_s: float = 2.0
    seed: int = 20261001
    samples_per_class: int = 1000
    qa_per_task: int = 1000
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
        image_rel = Path("images") / "diagnostic_triplet" / f"{sample_id}.png"
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


def task_prompt(task: str, choices: list[str]) -> str:
    prefix = "图中从上到下分别是幅频图、瞬时相位差图、线性频率时频图。"
    if task == "signal_type_choice":
        return prefix + "这段水声信号属于 CW、LFM、FSK、BPSK 中的哪一种？只回答一个选项。"
    if task == "main_frequency_choice":
        return prefix + "这段 CW 单频信号的主频最接近哪个：" + "、".join(choices) + "？只回答一个选项。"
    if task == "lfm_direction_choice":
        return prefix + "这段 LFM 信号是升频还是降频？只回答：升频 或 降频。"
    if task == "fsk_order_choice":
        return prefix + "这段 FSK 信号更像两频跳变还是四频跳变？只回答：2FSK 或 4FSK。"
    if task == "bpsk_symbol_rate_choice":
        return prefix + "这段 BPSK 信号的相位跳变码元率最接近哪个：" + "、".join(choices) + "？只回答一个选项。"
    raise ValueError(f"Unsupported task: {task}")


def task_definitions() -> dict[str, dict[str, Any]]:
    return {
        "signal_type_choice": {
            "answers": SIGNAL_TYPE_CHOICES,
            "answer_fn": lambda sample: sample["label"]["signal_type"],
            "candidate_fn": lambda sample: True,
        },
        "main_frequency_choice": {
            "answers": [f"{value} Hz" for value in FREQUENCY_CHOICES_HZ],
            "answer_fn": lambda sample: f"{sample['label']['main_frequency_hz']} Hz",
            "candidate_fn": lambda sample: sample["label"]["signal_type"] == "CW",
        },
        "lfm_direction_choice": {
            "answers": ["升频", "降频"],
            "answer_fn": lambda sample: sample["label"]["direction"],
            "candidate_fn": lambda sample: sample["label"]["signal_type"] == "LFM",
        },
        "fsk_order_choice": {
            "answers": ["2FSK", "4FSK"],
            "answer_fn": lambda sample: f"{sample['label']['fsk_order']}FSK",
            "candidate_fn": lambda sample: sample["label"]["signal_type"] == "FSK",
        },
        "bpsk_symbol_rate_choice": {
            "answers": [f"{value} symbol/s" for value in BPSK_SYMBOL_RATES],
            "answer_fn": lambda sample: f"{sample['label']['symbol_rate']} symbol/s",
            "candidate_fn": lambda sample: sample["label"]["signal_type"] == "BPSK",
        },
    }


def select_by_answer_and_split(
    samples: list[dict[str, Any]],
    answers: list[str],
    answer_fn: Callable[[dict[str, Any]], str],
    candidate_fn: Callable[[dict[str, Any]], bool],
    per_answer: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if candidate_fn(sample):
            buckets[(answer_fn(sample), sample["split"])].append(sample)

    for answer in answers:
        split_targets = weighted_counts(per_answer, SPLIT_WEIGHTS)
        for split, count in split_targets.items():
            candidates = buckets[(answer, split)]
            if len(candidates) < count:
                raise ValueError(f"Not enough candidates for answer={answer!r}, split={split!r}: need {count}, got {len(candidates)}")
            order = np.arange(len(candidates))
            rng.shuffle(order)
            selected.extend(candidates[int(index)] for index in order[:count])
    return selected


def make_qa_items(samples: list[dict[str, Any]], cfg: FinetuneConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed + 17)
    qa_items: list[dict[str, Any]] = []
    for task, task_def in task_definitions().items():
        answers = task_def["answers"]
        total_for_task = cfg.qa_per_task - (cfg.qa_per_task % len(answers))
        per_answer = total_for_task // len(answers)
        choices = list(answers)
        selected = select_by_answer_and_split(
            samples,
            answers,
            task_def["answer_fn"],
            task_def["candidate_fn"],
            per_answer,
            rng,
        )
        for sample in selected:
            expected = task_def["answer_fn"](sample)
            qa_items.append(
                {
                    "id": f"{sample['id']}_{task}",
                    "sample_id": sample["id"],
                    "base_id": sample["base_id"],
                    "split": sample["split"],
                    "image_path": sample["image_path"],
                    "audio_path": sample["audio_path"],
                    "task": task,
                    "prompt": task_prompt(task, choices),
                    "choices": choices,
                    "expected_answer": expected,
                    "scoring": {"type": "choice_exact"},
                    "condition": sample["condition"],
                    "label": sample["label"],
                }
            )
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

    task_counts = Counter(item["task"] for item in qa_items)
    if max(task_counts.values()) - min(task_counts.values()) > len(FREQUENCY_CHOICES_HZ):
        errors.append(f"Tasks are imbalanced: {dict(task_counts)}")

    for task in task_definitions():
        rows = [item for item in qa_items if item["task"] == task]
        answer_counts = Counter(item["expected_answer"] for item in rows)
        if answer_counts and max(answer_counts.values()) - min(answer_counts.values()) > 1:
            errors.append(f"Answers are imbalanced for {task}: {dict(answer_counts)}")

    base_splits: dict[str, set[str]] = defaultdict(set)
    image_splits: dict[str, set[str]] = defaultdict(set)
    for item in qa_items:
        if item["expected_answer"] not in item["choices"]:
            errors.append(f"Answer not in choices for {item['id']}")
        base_splits[item["base_id"]].add(item["split"])
        image_splits[item["image_path"]].add(item["split"])
    leaked_bases = [base_id for base_id, splits in base_splits.items() if len(splits) > 1]
    leaked_images = [image for image, splits in image_splits.items() if len(splits) > 1]
    if leaked_bases:
        errors.append(f"Base IDs leak across splits: {leaked_bases[:5]}")
    if leaked_images:
        errors.append(f"Images leak across splits: {leaked_images[:5]}")

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
    lines = [
        "# VLM Fine-tuning Dataset Report",
        "",
        "## Config",
        "",
        f"- samples_per_class: `{cfg.samples_per_class}`",
        f"- qa_per_task: `{cfg.qa_per_task}`",
        f"- sample_rate_hz: `{cfg.sample_rate_hz}`",
        f"- duration_s: `{cfg.duration_s}`",
        f"- image_kind: `diagnostic_triplet`",
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
        f"- by split: `{counter_dict([item['split'] for item in qa_items])}`",
        "",
        "## Answer Distribution By Task",
        "",
    ]
    for task in task_definitions():
        task_rows = [item for item in qa_items if item["task"] == task]
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
        },
    )
    write_json(out_dir / "dataset_config.json", {**asdict(cfg), "output": str(cfg.output)})
    write_report(out_dir, cfg, samples, qa_items, errors)

    if errors:
        raise RuntimeError("Dataset validation failed; see dataset_report.md")
    return samples, qa_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a balanced Qwen2.5-VL SFT dataset for underwater-acoustic diagnostic images.")
    parser.add_argument("--output", type=Path, default=Path("output") / "vlm_finetune" / "v1")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--qa-per-task", type=int, default=1000)
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
