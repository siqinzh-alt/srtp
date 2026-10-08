"""Run zero-shot evaluation on the generated probe set with Qwen2-Audio.

This script reads `eval_items.jsonl`, feeds each local WAV plus its text prompt to
Qwen2-Audio-7B-Instruct, writes raw predictions, and computes simple choice-task
accuracy grouped by task/noise/channel.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import librosa
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("２", "2").replace("４", "4")
    text = text.replace("赫兹", "hz").replace(" ", "")
    return re.sub(r"[。．.，,：:；;！!？?\n\t]", "", text)


def score_choice(response: str, expected: str, choices: list[str]) -> bool:
    normalized_response = normalize_text(response)
    normalized_expected = normalize_text(expected)
    if normalized_response == normalized_expected:
        return True
    choice_hits = [choice for choice in choices if normalize_text(choice) in normalized_response]
    if len(choice_hits) == 1:
        return normalize_text(choice_hits[0]) == normalized_expected
    return normalized_expected in normalized_response


def make_conversation(prompt: str, audio_path: Path) -> list[dict[str, Any]]:
    text = prompt.replace("<audio>", "").strip()
    return [
        {"role": "system", "content": "You are a careful acoustic signal analysis assistant. Answer exactly as requested."},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": text},
            ],
        },
    ]


def load_audio(path: Path, sample_rate: int) -> Any:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def run_batch(
    batch: list[dict[str, Any]],
    dataset_root: Path,
    processor: AutoProcessor,
    model: Qwen2AudioForConditionalGeneration,
    max_new_tokens: int,
) -> list[str]:
    conversations = [make_conversation(item["prompt"], dataset_root / item["audio_path"]) for item in batch]
    texts = [processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False) for conv in conversations]
    audios = [load_audio(dataset_root / item["audio_path"], processor.feature_extractor.sampling_rate) for item in batch]
    inputs = processor(text=texts, audios=audios, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        correct = bool(row["correct"])
        keys = {
            "overall": "overall",
            "task": row["task"],
            "noise": row["condition"].get("noise_level", "unknown"),
            "channel": row["condition"].get("channel_condition", "unknown"),
            "task_noise": f"{row['task']}|{row['condition'].get('noise_level', 'unknown')}",
            "task_channel": f"{row['task']}|{row['condition'].get('channel_condition', 'unknown')}",
        }
        for group_name, group_key in keys.items():
            groups[f"{group_name}:{group_key}"]["total"] += 1
            groups[f"{group_name}:{group_key}"]["correct"] += int(correct)
    return {
        key: {
            "correct": value["correct"],
            "total": value["total"],
            "accuracy": round(value["correct"] / value["total"], 6) if value["total"] else 0.0,
        }
        for key, value in sorted(groups.items())
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2-Audio zero-shot on a probe dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data") / "zero_shot_probe" / "v1")
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset
    output_path = args.output or dataset_root / "qwen2audio_predictions.jsonl"
    summary_path = args.summary or dataset_root / "qwen2audio_summary.json"
    items = read_jsonl(dataset_root / "eval_items.jsonl")
    if args.limit is not None:
        items = items[: args.limit]

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(items), args.batch_size), desc="zero-shot eval"):
        batch = items[start : start + args.batch_size]
        responses = run_batch(batch, dataset_root, processor, model, args.max_new_tokens)
        for item, response in zip(batch, responses):
            correct = score_choice(response, item["expected_answer"], item.get("choices", []))
            rows.append({**item, "response": response, "correct": correct})
        write_jsonl(output_path, rows)
        summary_path.write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summarize(rows).get("overall:overall", {}), ensure_ascii=False))
    print(f"Predictions: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
