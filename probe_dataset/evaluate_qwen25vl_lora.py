from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


SYSTEM_PROMPT = "You are a careful acoustic signal analysis assistant. Answer exactly as requested."


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


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def row_to_eval_item(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    user_content = row["messages"][0]["content"]
    assistant_content = row["messages"][1]["content"]
    prompt = next(part["text"] for part in user_content if part.get("type") == "text")
    expected = assistant_content[0]["text"]
    return {
        "id": row["id"],
        "image_path": row["images"][0],
        "prompt": prompt,
        "expected_answer": expected,
        "choices": metadata.get("choices", []),
        "task": metadata.get("task", "unknown"),
        "condition": metadata.get("condition", {}),
        "label": metadata.get("label", {}),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        correct = int(bool(row["correct"]))
        condition = row.get("condition", {})
        keys = {
            "overall": "overall",
            "task": row.get("task", "unknown"),
            "noise": condition.get("noise_level", "unknown"),
            "channel": condition.get("channel_condition", "unknown"),
            "task_noise": f"{row.get('task', 'unknown')}|{condition.get('noise_level', 'unknown')}",
            "task_channel": f"{row.get('task', 'unknown')}|{condition.get('channel_condition', 'unknown')}",
        }
        for group_name, group_key in keys.items():
            key = f"{group_name}:{group_key}"
            groups[key]["total"] += 1
            groups[key]["correct"] += correct
    return {
        key: {
            "correct": value["correct"],
            "total": value["total"],
            "accuracy": round(value["correct"] / value["total"], 6) if value["total"] else 0.0,
        }
        for key, value in sorted(groups.items())
    }


def run_batch(
    batch: list[dict[str, Any]],
    dataset_root: Path,
    processor: AutoProcessor,
    model: torch.nn.Module,
    max_new_tokens: int,
) -> list[str]:
    messages = []
    images = []
    for item in batch:
        image_path = dataset_root / item["image_path"]
        images.append(load_image(image_path))
        messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.as_posix()},
                    {"type": "text", "text": item["prompt"]},
                ],
            },
        ])
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen2.5-VL LoRA adapter on the VLM fine-tuning QA set.")
    parser.add_argument("--dataset", type=Path, default=Path("output") / "vlm_finetune" / "v1")
    parser.add_argument("--data-file", default="test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter", type=Path, default=Path("output") / "qwen25vl_lora" / "v1" / "final_adapter")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--min-pixels", type=int, default=3136)
    args = parser.parse_args()

    dataset_root = args.dataset
    output_path = args.output or args.adapter.parent / f"{Path(args.data_file).stem}_predictions.jsonl"
    summary_path = args.summary or args.adapter.parent / f"{Path(args.data_file).stem}_summary.json"
    rows = read_jsonl(dataset_root / args.data_file)
    items = [row_to_eval_item(row) for row in rows]
    if args.limit is not None:
        items = items[: args.limit]

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    quantization_config = None
    if args.load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.dtype != "float16" else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(
        args.adapter if args.adapter.exists() else args.model,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    predictions: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(items), args.batch_size), desc="qwen2.5-vl lora eval"):
        batch = items[start : start + args.batch_size]
        responses = run_batch(batch, dataset_root, processor, model, args.max_new_tokens)
        for item, response in zip(batch, responses):
            correct = score_choice(response, item["expected_answer"], item.get("choices", []))
            predictions.append({**item, "response": response, "correct": correct})
        write_jsonl(output_path, predictions)
        summary_path.write_text(json.dumps(summarize(predictions), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = summarize(predictions).get("overall:overall", {})
    print(json.dumps(overall, ensure_ascii=False))
    print(f"Predictions: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
