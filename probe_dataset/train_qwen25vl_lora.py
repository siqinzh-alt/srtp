from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)


SYSTEM_PROMPT = "You are a careful acoustic signal analysis assistant. Answer exactly as requested."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class QwenVlSftDataset(Dataset):
    def __init__(self, dataset_root: Path, jsonl_path: Path, limit: int | None = None) -> None:
        self.dataset_root = dataset_root
        rows = read_jsonl(jsonl_path)
        self.rows = rows[:limit] if limit is not None else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        user_text = next(part["text"] for part in row["messages"][0]["content"] if part["type"] == "text")
        answer = row["messages"][1]["content"][0]["text"]
        image_path = self.dataset_root / row["images"][0]
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.as_posix()},
                {"type": "text", "text": user_text},
            ],
        }
        return {
            "id": row["id"],
            "image": load_rgb(image_path),
            "prompt_messages": [{"role": "system", "content": SYSTEM_PROMPT}, user_message],
            "full_messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                user_message,
                {"role": "assistant", "content": answer},
            ],
        }


@dataclass
class TrainConfig:
    dataset: str
    model: str
    output_dir: str
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_pixels: int = 1003520
    min_pixels: int = 3136
    use_4bit: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 20261001
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    save_steps: int = 200
    eval_steps: int = 100
    log_steps: int = 10
    num_workers: int = 0


class QwenVlCollator:
    def __init__(self, processor: AutoProcessor) -> None:
        self.processor = processor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [example["image"] for example in examples]
        full_texts = [
            self.processor.apply_chat_template(example["full_messages"], tokenize=False, add_generation_prompt=False)
            for example in examples
        ]
        prompt_texts = [
            self.processor.apply_chat_template(example["prompt_messages"], tokenize=False, add_generation_prompt=True)
            for example in examples
        ]
        full_inputs = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100
        full_inputs["labels"] = labels
        return full_inputs


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def evaluate_loss(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int | None = None) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            outputs = model(**to_device(batch, device))
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(model: torch.nn.Module, processor: AutoProcessor, output_dir: Path, step: int, metrics: dict[str, Any]) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    dataset_root = Path(cfg.dataset)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    processor = AutoProcessor.from_pretrained(
        cfg.model,
        trust_remote_code=True,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    quantization_config = None
    if cfg.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    train_dataset = QwenVlSftDataset(dataset_root, dataset_root / cfg.train_file, cfg.max_train_samples)
    val_dataset = QwenVlSftDataset(dataset_root, dataset_root / cfg.val_file, cfg.max_val_samples)
    collator = QwenVlCollator(processor)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collator)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    device = next(model.parameters()).device
    scaler_enabled = not cfg.bf16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    log_rows: list[dict[str, Any]] = []

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg.epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        running_loss = 0.0
        running_examples = 0
        for micro_step, batch in enumerate(progress, start=1):
            batch_size = int(batch["input_ids"].shape[0])
            batch = to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=scaler_enabled):
                loss = model(**batch).loss / cfg.grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * cfg.grad_accum * batch_size
            running_examples += batch_size
            if micro_step % cfg.grad_accum == 0 or micro_step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % cfg.log_steps == 0 or global_step == 1:
                    avg_loss = running_loss / max(1, running_examples)
                    lr = scheduler.get_last_lr()[0]
                    row = {"step": global_step, "epoch": epoch + 1, "train_loss": avg_loss, "learning_rate": lr}
                    log_rows.append(row)
                    write_jsonl(output_dir / "train_log.jsonl", log_rows)
                    progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                    running_loss = 0.0
                    running_examples = 0
                if cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                    val_loss = evaluate_loss(model, val_loader, device, max_batches=100)
                    log_rows.append({"step": global_step, "epoch": epoch + 1, "val_loss": val_loss})
                    write_jsonl(output_dir / "train_log.jsonl", log_rows)
                    print(json.dumps({"step": global_step, "val_loss": val_loss}, ensure_ascii=False), flush=True)
                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    save_checkpoint(model, processor, output_dir, global_step, {"epoch": epoch + 1, "step": global_step})

    final_val_loss = evaluate_loss(model, val_loader, device, max_batches=None)
    save_checkpoint(model, processor, output_dir, global_step, {"final_val_loss": final_val_loss, "step": global_step})
    model.save_pretrained(output_dir / "final_adapter")
    processor.save_pretrained(output_dir / "final_adapter")
    print(json.dumps({"final_step": global_step, "final_val_loss": final_val_loss}, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-VL on underwater-acoustic diagnostic images.")
    parser.add_argument("--dataset", default="output/vlm_finetune/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", default="output/qwen25vl_lora/v1")
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--val-file", default="val.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = TrainConfig(
        dataset=args.dataset,
        model=args.model,
        output_dir=args.output_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        use_4bit=not args.no_4bit,
        bf16=not args.fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        num_workers=args.num_workers,
    )
    train(cfg)


if __name__ == "__main__":
    main()
