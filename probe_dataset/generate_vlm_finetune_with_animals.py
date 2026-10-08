"""Add public animal acoustic samples to an existing VLM SFT dataset.

The base fine-tuning dataset is synthetic underwater-acoustic diagnostic images.
This script keeps that dataset intact, builds a new output directory, downloads
Watkins Marine Mammal Sound Database audio, and renders it with the same
diagnostic-triplet image pipeline used by the synthetic samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf
from scipy import signal

from generate_probe_dataset import normalize, save_wav, write_json, write_jsonl
from generate_vlm_finetune_dataset import (
    FIELD_COUNT_WEIGHTS,
    MAX_QA_PER_IMAGE,
    SINGLE_FORMAT_WEIGHTS,
    SPLIT_WEIGHTS,
    counter_dict,
    to_qwen_record,
    weighted_counts,
)
from generate_vlm_probe_images import save_diagnostic_triplet


WATKINS_METADATA_URL = "https://marine-mammal.soundwave.cl/wmmsdb.20190601.b32a76a.json.xz"
WATKINS_AUDIO_URL = "https://archive.org/download/wmmsdb/{record_number}.flac"
WATKINS_DATA_PAGE = "https://marine-mammal.soundwave.cl/data.html"
ANIMAL_FIELD_WEIGHTS = {
    "bio_species_or_group": 0.77,
    "bio_species_code": 0.23,
}
ANIMAL_FIELD_LABELS = {
    "bio_species_or_group": "动物物种/类群",
    "bio_species_code": "物种代码",
}
ANIMAL_QA_LABEL_EXCLUDED_FIELDS = {"bio_call_description"}
IMAGE_KIND = "diagnostic_triplet"
IMAGE_DIR = "images"
SOURCE_DATASET_NAME = "Watkins Marine Mammal Sound Database"
USER_AGENT = "srtp-vlm-dataset-builder/1.0"


@dataclass(frozen=True)
class AnimalMergeConfig:
    base_dataset: Path = Path("data") / "vlm_finetune" / "v1"
    output: Path = Path("data") / "vlm_finetune" / "v3"
    source_cache: Path = Path("data") / "sources" / "watkins"
    animal_samples: int = 1000
    animal_qa_items: int | None = None
    seed: int = 20261006
    sample_rate_hz: int = 16000
    duration_s: float = 2.0
    max_freq_hz: float = 8000.0
    species_count: int = 20
    min_cut_size_s: float = 0.5
    request_timeout_s: int = 30
    download_retries: int = 3
    download_workers: int = 8
    overwrite: bool = False
    copy_mode: str = "hardlink"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_concatenated_json(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        obj, end = decoder.raw_decode(text, cursor)
        rows.append(obj)
        cursor = end
    return rows


def ensure_download(url: str, path: Path, timeout_s: int, retries: int) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, timeout=timeout_s, headers=headers, stream=True) as response:
                response.raise_for_status()
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
                tmp_path.replace(path)
                if path.stat().st_size <= 0:
                    path.unlink(missing_ok=True)
                    raise RuntimeError(f"Downloaded file is empty: {path}")
                return
        except Exception as exc:  # noqa: BLE001 - keep retry reason for final failure.
            last_error = exc
            time.sleep(min(2.0 * attempt, 6.0))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def load_watkins_rows(cfg: AnimalMergeConfig) -> list[dict[str, Any]]:
    metadata_path = cfg.source_cache / "wmmsdb.20190601.b32a76a.json.xz"
    ensure_download(WATKINS_METADATA_URL, metadata_path, cfg.request_timeout_s, cfg.download_retries)
    text = lzma.open(metadata_path, "rt", encoding="utf-8").read()
    return parse_concatenated_json(text)


def first_species(record: dict[str, Any]) -> dict[str, str] | None:
    genus = record.get("animal", {}).get("genus") or []
    if not genus:
        return None
    name = str(genus[0].get("name") or "").strip()
    code = str(genus[0].get("species_code") or "").strip()
    if not name or name.lower() == "unknown" or not code:
        return None
    return {"name": name, "code": code}


def normalize_note(note: str, species_code: str) -> str:
    text = re.sub(r"\s+", " ", note or "").strip()
    if species_code:
        text = re.sub(rf"^{re.escape(species_code)}\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" .;")
    if not text:
        return "unknown"
    parts = [part.strip(" .") for part in re.split(r"[.;]", text) if part.strip(" .")]
    if not parts:
        return text[:80]
    return "; ".join(parts[:2])[:80]


def candidate_records(rows: list[dict[str, Any]], cfg: AnimalMergeConfig) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        species = first_species(row)
        cut_size = row.get("signal", {}).get("cut_size")
        record_number = str(row.get("record_number") or "").strip()
        if species is None or not record_number:
            continue
        if not isinstance(cut_size, (int, float)) or float(cut_size) < cfg.min_cut_size_s:
            continue
        candidates.append(row)
    return candidates


def select_balanced_records(rows: list[dict[str, Any]], cfg: AnimalMergeConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    candidates = candidate_records(rows, cfg)
    species_counts = Counter(first_species(row)["name"] for row in candidates if first_species(row))
    selected_species = [name for name, _ in species_counts.most_common(cfg.species_count)]
    species_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        species = first_species(row)
        if species and species["name"] in selected_species:
            species_to_rows[species["name"]].append(row)

    target_with_spares = min(len(candidates), max(cfg.animal_samples, math.ceil(cfg.animal_samples * 1.5)))
    counts = weighted_counts(target_with_spares, {name: 1.0 / len(selected_species) for name in selected_species})
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for species_name in selected_species:
        rows_for_species = list(species_to_rows[species_name])
        rng.shuffle(rows_for_species)
        take = min(counts[species_name], len(rows_for_species))
        selected.extend(rows_for_species[:take])
        leftovers.extend(rows_for_species[take:])
    if len(selected) < target_with_spares:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_with_spares - len(selected)])
    if len(selected) < cfg.animal_samples:
        raise RuntimeError(f"Only {len(selected)} Watkins records available after filtering; requested {cfg.animal_samples}")
    rng.shuffle(selected)
    return selected


def split_records(records: list[dict[str, Any]], cfg: AnimalMergeConfig) -> dict[str, str]:
    rng = np.random.default_rng(cfg.seed + 1)
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        species = first_species(record)
        by_species[species["name"]].append(record)
    split_by_record: dict[str, str] = {}
    for rows in by_species.values():
        rng.shuffle(rows)
        counts = weighted_counts(len(rows), SPLIT_WEIGHTS)
        cursor = 0
        for split, count in counts.items():
            for row in rows[cursor : cursor + count]:
                split_by_record[str(row["record_number"])] = split
            cursor += count
    return split_by_record


def link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def flat_image_path(image_path: str) -> str:
    return (Path(IMAGE_DIR) / Path(image_path).name).as_posix()


def source_image_candidates(base_dir: Path, image_path: str) -> list[Path]:
    filename = Path(image_path).name
    candidates = [base_dir / image_path, base_dir / IMAGE_DIR / filename, base_dir / IMAGE_DIR / IMAGE_KIND / filename]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def copy_base_assets(base_dir: Path, out_dir: Path, base_samples: list[dict[str, Any]], copy_mode: str, max_freq_hz: float) -> None:
    copied_paths: set[str] = set()
    for sample in base_samples:
        for rel_path in (sample.get("audio_path"), sample.get("metadata_path")):
            if not rel_path or rel_path in copied_paths:
                continue
            src_path = base_dir / rel_path
            if src_path.exists():
                link_or_copy_file(src_path, out_dir / rel_path, copy_mode)
                copied_paths.add(rel_path)

        image_rel = sample["image_path"]
        if image_rel in copied_paths:
            continue
        image_dst = out_dir / image_rel
        for image_src in source_image_candidates(base_dir, image_rel):
            if image_src.exists():
                link_or_copy_file(image_src, image_dst, copy_mode)
                copied_paths.add(image_rel)
                break
        if image_rel in copied_paths:
            continue

        audio_rel = sample.get("audio_path")
        audio_src = base_dir / audio_rel if audio_rel else None
        audio_dst = out_dir / audio_rel if audio_rel else None
        if sample.get("image_kind") == IMAGE_KIND and audio_src is not None and audio_src.exists() and audio_dst is not None:
            if not audio_dst.exists():
                link_or_copy_file(audio_src, audio_dst, copy_mode)
            save_diagnostic_triplet(audio_dst, image_dst, max_freq=max_freq_hz)
            copied_paths.add(image_rel)
            continue

        raise FileNotFoundError(f"Missing base image and cannot regenerate it: {base_dir / image_rel}")


def audio_cache_path(record_number: str, cfg: AnimalMergeConfig) -> Path:
    return cfg.source_cache / "audio_flac" / f"{record_number}.flac"


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise RuntimeError("Audio is empty or has unsupported shape")
    if not np.all(np.isfinite(audio)):
        raise RuntimeError("Audio contains non-finite values")
    return audio


def resample_to_target(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate <= 0:
        raise RuntimeError(f"Invalid source sample rate: {source_sample_rate}")
    if source_sample_rate == target_sample_rate:
        return audio
    divisor = math.gcd(source_sample_rate, target_sample_rate)
    return signal.resample_poly(audio, target_sample_rate // divisor, source_sample_rate // divisor)


def deterministic_segment_seed(record_number: str, cfg: AnimalMergeConfig) -> int:
    digest = hashlib.sha256(f"{cfg.seed}:{record_number}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def fixed_length_segment(audio: np.ndarray, record_number: str, cfg: AnimalMergeConfig) -> tuple[np.ndarray, dict[str, Any]]:
    target_samples = max(1, int(round(cfg.duration_s * cfg.sample_rate_hz)))
    if audio.size >= target_samples:
        max_start = audio.size - target_samples
        rng = np.random.default_rng(deterministic_segment_seed(record_number, cfg))
        start = int(rng.integers(0, max_start + 1)) if max_start else 0
        segment = audio[start : start + target_samples]
        padded = False
    else:
        start = 0
        repeats = math.ceil(target_samples / audio.size)
        segment = np.tile(audio, repeats)[:target_samples]
        padded = True
    return normalize(segment, peak=0.85), {
        "segment_start_s": round(start / cfg.sample_rate_hz, 6),
        "segment_end_s": round((start + target_samples) / cfg.sample_rate_hz, 6),
        "segment_padded": padded,
    }


def prepare_watkins_audio(record: dict[str, Any], cfg: AnimalMergeConfig) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    record_number = str(record["record_number"])
    flac_path = audio_cache_path(record_number, cfg)
    try:
        ensure_download(WATKINS_AUDIO_URL.format(record_number=record_number), flac_path, cfg.request_timeout_s, cfg.download_retries)
        audio, source_sample_rate = sf.read(flac_path, always_2d=False)
        source_sample_rate = int(source_sample_rate)
        audio = to_mono_float(audio)
        original_duration_s = audio.size / source_sample_rate
        audio = resample_to_target(audio, source_sample_rate, cfg.sample_rate_hz)
        segment, segment_meta = fixed_length_segment(audio, record_number, cfg)
    except Exception as exc:  # noqa: BLE001 - keep retry/download/decode reason for skip logging.
        return record, None, str(exc)
    return record, {
        "segment": segment,
        "source_flac_path": flac_path,
        "original_sample_rate_hz": source_sample_rate,
        "original_duration_s": round(original_duration_s, 6),
        **segment_meta,
    }, None


def make_animal_samples(
    records: list[dict[str, Any]], split_by_record: dict[str, str], out_dir: Path, cfg: AnimalMergeConfig, id_start: int
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    skipped = 0
    scanned = 0
    max_workers = max(1, cfg.download_workers)
    record_iter = iter(records)
    pending: set[Future[tuple[dict[str, Any], dict[str, Any] | None, str | None]]] = set()

    def submit_more(executor: ThreadPoolExecutor) -> None:
        while len(pending) < max_workers * 2:
            try:
                record = next(record_iter)
            except StopIteration:
                return
            pending.add(executor.submit(prepare_watkins_audio, record, cfg))

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        submit_more(executor)
        while pending and len(samples) < cfg.animal_samples:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                scanned += 1
                record, audio_result, error = future.result()
                submit_more(executor)
                if len(samples) >= cfg.animal_samples:
                    break
                species = first_species(record)
                assert species is not None
                record_number = str(record["record_number"])
                if error is not None:
                    skipped += 1
                    print(f"Skipped Watkins record {record_number}: {error}", flush=True)
                    continue

                index = len(samples) + 1
                sample_id = f"vlmft_{id_start + index:06d}"
                audio_rel = Path("audio") / f"{sample_id}.wav"
                image_rel = Path(IMAGE_DIR) / f"{sample_id}.png"
                metadata_rel = Path("metadata") / f"{sample_id}.json"
                note = normalize_note(str(record.get("note") or ""), species["code"])
                assert audio_result is not None
                try:
                    save_wav(out_dir / audio_rel, cfg.sample_rate_hz, audio_result["segment"])
                    save_diagnostic_triplet(out_dir / audio_rel, out_dir / image_rel, max_freq=cfg.max_freq_hz)
                except Exception as exc:  # noqa: BLE001 - skip bad audio/rendering without stopping the full merge.
                    skipped += 1
                    (out_dir / audio_rel).unlink(missing_ok=True)
                    (out_dir / image_rel).unlink(missing_ok=True)
                    print(f"Skipped Watkins record {record_number}: failed to render diagnostic image ({exc})", flush=True)
                    continue

                location_names = record.get("location", {}).get("name") or []
                sample = {
                    "id": sample_id,
                    "base_id": f"watkins:{record_number}",
                    "split": split_by_record[record_number],
                    "audio_path": audio_rel.as_posix(),
                    "image_path": image_rel.as_posix(),
                    "metadata_path": metadata_rel.as_posix(),
                    "sample_rate_hz": cfg.sample_rate_hz,
                    "duration_s": cfg.duration_s,
                    "label": {
                        "signal_type": "Animal",
                        "bio_taxon_group": "marine_mammal",
                        "bio_species_or_group": species["name"],
                        "bio_species_code": species["code"],
                        "bio_call_description": note,
                    },
                    "condition": {
                        "noise_level": "field_recording",
                        "channel_condition": "archival_recording",
                        "noise_added": False,
                        "bellhop_processed": False,
                    },
                    "generation": {
                        "source_dataset": SOURCE_DATASET_NAME,
                        "source_data_page": WATKINS_DATA_PAGE,
                        "record_number": record_number,
                        "source_audio_url": WATKINS_AUDIO_URL.format(record_number=record_number),
                        "source_audio_cache": str(audio_result["source_flac_path"]),
                        "original_sample_rate_hz": audio_result["original_sample_rate_hz"],
                        "original_duration_s": audio_result["original_duration_s"],
                        "segment_start_s": audio_result["segment_start_s"],
                        "segment_end_s": audio_result["segment_end_s"],
                        "segment_padded": audio_result["segment_padded"],
                        "observation_date": record.get("observation_date"),
                        "location_name": location_names[0] if location_names else None,
                    },
                    "dataset_role": "vlm_finetune_sft",
                    "image_kind": IMAGE_KIND,
                }
                (out_dir / metadata_rel).parent.mkdir(parents=True, exist_ok=True)
                (out_dir / metadata_rel).write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                samples.append(sample)
                if index == 1 or index % 50 == 0 or index == cfg.animal_samples:
                    print(f"Generated animal samples {index}/{cfg.animal_samples} (scanned {scanned}, skipped {skipped})", flush=True)
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
    if len(samples) < cfg.animal_samples:
        raise RuntimeError(f"Only generated {len(samples)} animal samples after scanning {len(records)} records; requested {cfg.animal_samples}")
    return samples


def animal_prefix(sample: dict[str, Any]) -> str:
    return "图中是由海洋哺乳动物录音生成的统一声学诊断图。"


def animal_field_answer(sample: dict[str, Any], field: str) -> str:
    return str(sample["label"][field])


def animal_qa_label(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sample["label"].items() if key not in ANIMAL_QA_LABEL_EXCLUDED_FIELDS}


def animal_field_choices(samples: list[dict[str, Any]], field: str) -> list[str]:
    values = sorted({animal_field_answer(sample, field) for sample in samples})
    return values


def animal_prompt(sample: dict[str, Any], field: str, answer_format: str, choices: list[str]) -> str:
    prefix = animal_prefix(sample)
    label = ANIMAL_FIELD_LABELS[field]
    if answer_format == "choice":
        return prefix + f"这段动物声学样本的{label}最接近哪个：" + "、".join(choices) + "？只回答一个选项。"
    if field == "bio_species_or_group":
        return prefix + "这段动物声学样本对应的物种/类群是什么？只回答拉丁名短答案。"
    return prefix + f"这段动物声学样本的{label}是什么？只回答短答案。"


def animal_multi_prompt(fields: list[str]) -> str:
    field_labels = [f"{field}({ANIMAL_FIELD_LABELS[field]})" for field in fields]
    return (
        "图中是由海洋哺乳动物录音生成的统一声学诊断图。"
        + "请同时判断以下字段："
        + "、".join(field_labels)
        + "。只输出 JSON 对象，字段名必须与题目中给出的英文字段名一致，不要解释。"
    )


def make_animal_single_qa(sample: dict[str, Any], field: str, answer_format: str, choices: list[str], sequence: int) -> dict[str, Any]:
    task = f"{field}_{'choice' if answer_format == 'choice' else 'open_short'}"
    return {
        "id": f"{sample['id']}_{task}_{sequence:04d}",
        "sample_id": sample["id"],
        "base_id": sample["base_id"],
        "split": sample["split"],
        "image_path": sample["image_path"],
        "audio_path": sample.get("audio_path", ""),
        "task": task,
        "prompt": animal_prompt(sample, field, answer_format, choices),
        "choices": choices if answer_format == "choice" else [],
        "fields": [field],
        "field_count": 1,
        "answer_format": answer_format,
        "expected_answer": animal_field_answer(sample, field),
        "scoring": {"type": "choice_exact" if answer_format == "choice" else "open_short_exact"},
        "condition": sample["condition"],
        "label": animal_qa_label(sample),
    }


def make_animal_multi_qa(sample: dict[str, Any], fields: list[str], sequence: int) -> dict[str, Any]:
    return {
        "id": f"{sample['id']}_multi_field_open_json_{sequence:04d}",
        "sample_id": sample["id"],
        "base_id": sample["base_id"],
        "split": sample["split"],
        "image_path": sample["image_path"],
        "audio_path": sample.get("audio_path", ""),
        "task": "multi_field_open_json",
        "prompt": animal_multi_prompt(fields),
        "choices": [],
        "fields": fields,
        "field_count": len(fields),
        "field_signature": "+".join(fields),
        "answer_format": "open_json",
        "expected_answer": json.dumps({field: animal_field_answer(sample, field) for field in fields}, ensure_ascii=False, separators=(",", ":")),
        "scoring": {"type": "json_exact_fields", "fields": fields},
        "condition": sample["condition"],
        "label": animal_qa_label(sample),
    }


def normalized_animal_field_weights(fields: list[str]) -> dict[str, float]:
    weights = {field: ANIMAL_FIELD_WEIGHTS[field] for field in fields}
    total = sum(weights.values())
    return {field: weight / total for field, weight in weights.items()}


def animal_field_count_weights(fields: list[str]) -> dict[int, float]:
    max_field_count = len(fields)
    weights: dict[int, float] = {}
    for field_count, weight in FIELD_COUNT_WEIGHTS.items():
        weights[min(field_count, max_field_count)] = weights.get(min(field_count, max_field_count), 0.0) + weight
    return weights


def make_animal_qa_items(samples: list[dict[str, Any]], cfg: AnimalMergeConfig) -> list[dict[str, Any]]:
    total = cfg.animal_qa_items or round(cfg.animal_samples * 1.25)
    rng = np.random.default_rng(cfg.seed + 17)
    fields = list(ANIMAL_FIELD_WEIGHTS)
    field_count_counts = weighted_counts(total, animal_field_count_weights(fields))
    single_format_counts = weighted_counts(field_count_counts[1], SINGLE_FORMAT_WEIGHTS)
    field_counts_by_format = {
        "choice": weighted_counts(single_format_counts["choice"], normalized_animal_field_weights(fields)),
        "open_short": weighted_counts(single_format_counts["open_short"], normalized_animal_field_weights(fields)),
    }
    image_counts: Counter[str] = Counter()
    qa_items: list[dict[str, Any]] = []
    sequence = 0

    for answer_format in ("choice", "open_short"):
        for field, count in field_counts_by_format[answer_format].items():
            candidates = [sample for sample in samples if image_counts[sample["image_path"]] < MAX_QA_PER_IMAGE]
            if not candidates:
                raise RuntimeError("No animal images left for single-field QA")
            rng.shuffle(candidates)
            choices = animal_field_choices(samples, field)
            selected = candidates[:count]
            if len(selected) < count:
                raise RuntimeError(f"Not enough animal images for {field}/{answer_format}: need {count}, got {len(selected)}")
            for sample in selected:
                sequence += 1
                image_counts[sample["image_path"]] += 1
                qa_items.append(make_animal_single_qa(sample, field, answer_format, choices, sequence))

    for field_count in (2, 3):
        count = field_count_counts[field_count]
        for _ in range(count):
            candidates = [sample for sample in samples if image_counts[sample["image_path"]] < MAX_QA_PER_IMAGE]
            if not candidates:
                raise RuntimeError(f"No animal images left for {field_count}-field QA")
            candidates.sort(key=lambda sample: image_counts[sample["image_path"]])
            least_used = [sample for sample in candidates if image_counts[sample["image_path"]] == image_counts[candidates[0]["image_path"]]]
            sample = least_used[int(rng.integers(0, len(least_used)))]
            weights = normalized_animal_field_weights(fields)
            selected_fields = list(
                rng.choice(list(weights), size=field_count, replace=False, p=np.array([weights[field] for field in weights], dtype=float))
            )
            sequence += 1
            image_counts[sample["image_path"]] += 1
            qa_items.append(make_animal_multi_qa(sample, [str(field) for field in selected_fields], sequence))

    rng.shuffle(qa_items)
    return qa_items


def validate_merged_dataset(samples: list[dict[str, Any]], qa_items: list[dict[str, Any]], out_dir: Path) -> list[str]:
    errors: list[str] = []
    sample_ids = {sample["id"] for sample in samples}
    base_splits: dict[str, set[str]] = defaultdict(set)
    image_splits: dict[str, set[str]] = defaultdict(set)
    image_counts = Counter(item["image_path"] for item in qa_items)
    for item in qa_items:
        if item["sample_id"] not in sample_ids:
            errors.append(f"QA references unknown sample: {item['id']} -> {item['sample_id']}")
        if not (out_dir / item["image_path"]).exists():
            errors.append(f"Missing image: {item['image_path']}")
        if item["answer_format"] == "choice" and item["expected_answer"] not in item.get("choices", []):
            errors.append(f"Answer not in choices: {item['id']}")
        if item["answer_format"] == "open_json":
            try:
                parsed = json.loads(item["expected_answer"])
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON answer for {item['id']}: {exc}")
            else:
                if set(parsed) != set(item["fields"]):
                    errors.append(f"JSON fields mismatch for {item['id']}")
        base_splits[item["base_id"]].add(item["split"])
        image_splits[item["image_path"]].add(item["split"])
    leaked_bases = [base_id for base_id, splits in base_splits.items() if len(splits) > 1]
    leaked_images = [image for image, splits in image_splits.items() if len(splits) > 1]
    overused = [image for image, count in image_counts.items() if count > MAX_QA_PER_IMAGE]
    if leaked_bases:
        errors.append(f"Base IDs leak across splits: {leaked_bases[:5]}")
    if leaked_images:
        errors.append(f"Images leak across splits: {leaked_images[:5]}")
    if overused:
        errors.append(f"Images exceed max QA reuse ({MAX_QA_PER_IMAGE}): {overused[:5]}")
    return errors


def write_split_files(out_dir: Path, qa_items: list[dict[str, Any]]) -> None:
    for split in SPLIT_WEIGHTS:
        rows = [to_qwen_record(item) for item in qa_items if item["split"] == split]
        write_jsonl(out_dir / f"{split}.jsonl", rows)


def write_report(out_dir: Path, cfg: AnimalMergeConfig, samples: list[dict[str, Any]], qa_items: list[dict[str, Any]], errors: list[str]) -> None:
    animal_samples = [sample for sample in samples if sample["label"].get("signal_type") == "Animal"]
    base_samples = [sample for sample in samples if sample["label"].get("signal_type") != "Animal"]
    answer_format_counts = Counter(item["answer_format"] for item in qa_items)
    field_count_counts = Counter(item["field_count"] for item in qa_items)
    field_counts = Counter(field for item in qa_items for field in item.get("fields", []))
    image_reuse_counts = Counter(Counter(item["image_path"] for item in qa_items).values())
    lines = [
        "# VLM Fine-tuning Dataset Report",
        "",
        "## Config",
        "",
        f"- base_dataset: `{cfg.base_dataset}` (filtered to non-Animal samples before rewrite)",
        f"- animal_samples_requested: `{cfg.animal_samples}`",
        f"- animal_qa_items_requested: `{cfg.animal_qa_items or round(cfg.animal_samples * 1.25)}`",
        f"- sample_rate_hz: `{cfg.sample_rate_hz}`",
        f"- duration_s: `{cfg.duration_s}`",
        f"- max_freq_hz: `{cfg.max_freq_hz}`",
        f"- source_dataset: `{SOURCE_DATASET_NAME}`",
        f"- source_data_page: `{WATKINS_DATA_PAGE}`",
        f"- image_kind: `{IMAGE_KIND}`",
        f"- field_count_weights: `{FIELD_COUNT_WEIGHTS}`",
        f"- single_format_weights: `{SINGLE_FORMAT_WEIGHTS}`",
        f"- animal_field_weights: `{ANIMAL_FIELD_WEIGHTS}`",
        "",
        "## Sample Distribution",
        "",
        f"- total samples: `{len(samples)}`",
        f"- base synthetic samples: `{len(base_samples)}`",
        f"- animal samples: `{len(animal_samples)}`",
        f"- by signal type: `{counter_dict([sample['label'].get('signal_type', 'unknown') for sample in samples])}`",
        f"- by image kind: `{counter_dict([sample.get('image_kind', 'unknown') for sample in samples])}`",
        f"- by split: `{counter_dict([sample['split'] for sample in samples])}`",
        f"- animal by species: `{counter_dict([sample['label']['bio_species_or_group'] for sample in animal_samples])}`",
        "",
        "## QA Distribution",
        "",
        f"- total QA: `{len(qa_items)}`",
        f"- animal QA: `{sum(1 for item in qa_items if item.get('label', {}).get('signal_type') == 'Animal')}`",
        f"- by task: `{counter_dict([item['task'] for item in qa_items])}`",
        f"- by answer format: `{dict(sorted(answer_format_counts.items()))}`",
        f"- by field count: `{dict(sorted(field_count_counts.items()))}`",
        f"- by field occurrence: `{dict(sorted(field_counts.items()))}`",
        f"- by QA reuse per image: `{dict(sorted(image_reuse_counts.items()))}`",
        f"- by split: `{counter_dict([item['split'] for item in qa_items])}`",
        "",
        "## Validation",
        "",
    ]
    if errors:
        lines.extend(f"- ERROR: {error}" for error in errors)
    else:
        lines.append("- OK: all validation checks passed.")
    (out_dir / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_dataset(cfg: AnimalMergeConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if cfg.output.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output directory already exists: {cfg.output}. Use --overwrite to replace it.")
        shutil.rmtree(cfg.output)
    cfg.output.mkdir(parents=True, exist_ok=True)

    raw_base_samples = read_jsonl(cfg.base_dataset / "manifest.jsonl")
    for sample in raw_base_samples:
        sample["image_path"] = flat_image_path(sample["image_path"])
    base_samples = [sample for sample in raw_base_samples if sample["label"].get("signal_type") != "Animal"]
    base_sample_ids = {sample["id"] for sample in base_samples}
    base_qa = [item for item in read_jsonl(cfg.base_dataset / "qa_items.jsonl") if item["sample_id"] in base_sample_ids]
    for item in base_qa:
        item["image_path"] = flat_image_path(item["image_path"])
    print(f"Loaded base dataset: {len(base_samples)} samples, {len(base_qa)} QA", flush=True)
    if len(base_samples) != len(raw_base_samples):
        print(f"Filtered existing animal samples from base dataset: {len(raw_base_samples) - len(base_samples)}", flush=True)
    copy_base_assets(cfg.base_dataset, cfg.output, base_samples, cfg.copy_mode, cfg.max_freq_hz)
    print("Base image/metadata assets linked or copied", flush=True)

    watkins_rows = load_watkins_rows(cfg)
    selected_records = select_balanced_records(watkins_rows, cfg)
    split_by_record = split_records(selected_records, cfg)
    animal_samples = make_animal_samples(selected_records, split_by_record, cfg.output, cfg, id_start=len(base_samples))
    animal_qa = make_animal_qa_items(animal_samples, cfg)

    samples = base_samples + animal_samples
    qa_items = base_qa + animal_qa
    errors = validate_merged_dataset(samples, qa_items, cfg.output)
    write_jsonl(cfg.output / "manifest.jsonl", samples)
    write_jsonl(cfg.output / "qa_items.jsonl", qa_items)
    write_split_files(cfg.output, qa_items)
    write_json(
        cfg.output / "split_manifest.json",
        {
            "sample_splits": {sample["id"]: sample["split"] for sample in samples},
            "split_weights": SPLIT_WEIGHTS,
            "base_dataset": str(cfg.base_dataset),
            "animal_source_dataset": SOURCE_DATASET_NAME,
            "animal_source_data_page": WATKINS_DATA_PAGE,
            "animal_field_weights": ANIMAL_FIELD_WEIGHTS,
            "field_count_weights": FIELD_COUNT_WEIGHTS,
            "single_format_weights": SINGLE_FORMAT_WEIGHTS,
            "max_qa_per_image": MAX_QA_PER_IMAGE,
            "base_filter": "label.signal_type != Animal",
        },
    )
    write_json(
        cfg.output / "dataset_config.json",
        {
            **asdict(cfg),
            "base_dataset": str(cfg.base_dataset),
            "output": str(cfg.output),
            "source_cache": str(cfg.source_cache),
            "base_filter": "label.signal_type != Animal",
        },
    )
    write_report(cfg.output, cfg, samples, qa_items, errors)
    if errors:
        raise RuntimeError("Dataset validation failed; see dataset_report.md")
    return samples, qa_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add Watkins animal audio samples to a VLM SFT dataset using diagnostic-triplet images.")
    parser.add_argument("--base-dataset", type=Path, default=Path("data") / "vlm_finetune" / "v1")
    parser.add_argument("--output", type=Path, default=Path("data") / "vlm_finetune" / "v3")
    parser.add_argument("--source-cache", type=Path, default=Path("data") / "sources" / "watkins")
    parser.add_argument("--animal-samples", type=int, default=1000)
    parser.add_argument("--animal-qa-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--max-freq", type=float, default=8000.0)
    parser.add_argument("--species-count", type=int, default=20)
    parser.add_argument("--min-cut-size", type=float, default=0.5)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = AnimalMergeConfig(
        base_dataset=args.base_dataset,
        output=args.output,
        source_cache=args.source_cache,
        animal_samples=args.animal_samples,
        animal_qa_items=args.animal_qa_items,
        seed=args.seed,
        sample_rate_hz=args.sample_rate,
        duration_s=args.duration,
        max_freq_hz=args.max_freq,
        species_count=args.species_count,
        min_cut_size_s=args.min_cut_size,
        request_timeout_s=args.request_timeout,
        download_retries=args.download_retries,
        download_workers=args.download_workers,
        overwrite=args.overwrite,
        copy_mode=args.copy_mode,
    )
    samples, qa_items = merge_dataset(cfg)
    print(f"Generated merged dataset: {len(samples)} samples and {len(qa_items)} QA items in {cfg.output}")
    print(f"Qwen2.5-VL files: {cfg.output / 'train.jsonl'}, {cfg.output / 'val.jsonl'}, {cfg.output / 'test.jsonl'}")
    print(f"Report: {cfg.output / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
