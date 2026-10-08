from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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


def make_prompt(original_prompt: str, task: str, image_view: str) -> str:
    prompt = original_prompt.replace("<audio>", "").strip()
    if image_view == "spectrogram_only":
        if task == "main_frequency_choice":
            prefix = "图中只包含同一段信号的线性频率时频图。请根据最稳定、最亮的水平频率轨迹所在的 Hz 坐标判断。"
        else:
            prefix = "图中只包含同一段信号的线性频率时频图。请根据频率随时间变化的轨迹判断。"
    elif image_view == "diagnostic_triplet":
        if task == "main_frequency_choice":
            prefix = "图中从上到下依次是同一段信号的幅频图、瞬时相位差分图和线性频率时频图。请主要根据幅频图最高峰和时频图最亮轨迹所在的 Hz 坐标判断。"
        elif task == "lfm_direction_choice":
            prefix = "图中从上到下依次是同一段信号的幅频图、瞬时相位差分图和线性频率时频图。请主要根据时频图中频率轨迹随时间上升还是下降判断。"
        elif task == "fsk_order_choice":
            prefix = "图中从上到下依次是同一段信号的幅频图、瞬时相位差分图和线性频率时频图。请根据幅频图中的离散频率峰和时频图中的跳频轨迹判断 FSK 阶数。"
        else:
            prefix = "图中从上到下依次是同一段信号的幅频图、瞬时相位差分图和线性频率时频图。幅频图用于看频率峰，相位差分图用于看相位跳变，时频图用于看能量轨迹。请综合判断。"
    else:
        if task == "main_frequency_choice":
            prefix = "图中包含同一段信号的波形、幅度谱和线性频率时频图。请主要根据幅度谱最高峰所在的 Hz 坐标判断。"
        elif task in {"lfm_direction_choice", "fsk_order_choice"}:
            prefix = "图中包含同一段信号的波形、幅度谱和线性频率时频图。请主要根据时频图中的频率轨迹判断。"
        else:
            prefix = "图中包含同一段信号的波形、幅度谱和线性频率时频图。请根据这些信号特征判断。"
    return prefix + prompt


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


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def run_batch(batch: list[dict[str, Any]], dataset_root: Path, processor: AutoProcessor, model: Qwen2_5_VLForConditionalGeneration, max_new_tokens: int, image_view: str) -> list[str]:
    messages = []
    images = []
    for item in batch:
        image_path = dataset_root / item["image_path"]
        images.append(load_image(image_path))
        messages.append([
            {"role": "system", "content": "You are a careful acoustic signal analysis assistant. Answer exactly as requested."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.as_posix()},
                    {"type": "text", "text": make_prompt(item["prompt"], item["task"], image_view)},
                ],
            },
        ])
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL zero-shot on generated probe images.")
    parser.add_argument("--dataset", type=Path, default=Path("data") / "zero_shot_probe" / "v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image-manifest", default="vlm_image_manifest.jsonl")
    parser.add_argument("--image-view", choices=["composite", "spectrogram_only", "diagnostic_triplet"], default="composite")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    dataset_root = args.dataset
    output_path = args.output or dataset_root / "qwen25vl_predictions.jsonl"
    summary_path = args.summary or dataset_root / "qwen25vl_summary.json"
    image_manifest = {row["sample_id"]: row["image_path"] for row in read_jsonl(dataset_root / args.image_manifest)}
    items = read_jsonl(dataset_root / "eval_items.jsonl")
    for item in items:
        item["image_path"] = image_manifest[item["sample_id"]]
    if args.limit is not None:
        items = items[: args.limit]

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(items), args.batch_size), desc="qwen2.5-vl eval"):
        batch = items[start : start + args.batch_size]
        responses = run_batch(batch, dataset_root, processor, model, args.max_new_tokens, args.image_view)
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
