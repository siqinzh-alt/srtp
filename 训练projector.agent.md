训练projector：
from transformers import LlavaForConditionalGeneration

model_path = "/root/autodl-tmp/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="cpu",
    local_files_only=True,
)

print(model.model.multi_modal_projector)

for name, parameter in model.named_parameters():
    if "multi_modal_projector" in name:
        print(name, tuple(parameter.shape))



1. 安装依赖：!pip install -U transformers accelerate pillow
2. 加载数据：
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_DIR = "/root/autodl-tmp/llava-1.5-7b-hf"
DATA_DIR = Path("/root/autodl-tmp/projector/data/llava_multiclass_v1")

TRAIN_JSON = DATA_DIR / "alignment_train.jsonl"
VAL_JSON = DATA_DIR / "alignment_val.jsonl"

assert TRAIN_JSON.is_file()
assert VAL_JSON.is_file()

processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.to("cuda")
3. 打印可训练参数：
for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.model.multi_modal_projector.parameters():
    parameter.requires_grad = True

trainable_count = 0

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name, tuple(parameter.shape))
        trainable_count += parameter.numel()

print(f"可训练参数总数：{trainable_count:,}")

4. 建立数据读取器
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class AlignmentDataset(Dataset):
    def __init__(self, json_path, image_root):
        self.rows = read_jsonl(json_path)
        self.image_root = Path(image_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        question = row["conversations"][0]["value"]
        question = question.replace("<image>\n", "").replace("<image>", "").strip()
        answer = row["conversations"][1]["value"].strip()

        image = Image.open(self.image_root / row["image"]).convert("RGB")

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }


class AlignmentCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        images = [item["image"] for item in examples]

        prompt_texts = [
            f"USER: <image>\n{item['question']}\nASSISTANT:"
            for item in examples
        ]

        full_texts = [
            f"USER: <image>\n{item['question']}\nASSISTANT: {item['answer']}"
            f"{self.processor.tokenizer.eos_token}"
            for item in examples
        ]

        full_batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()

        for i, prompt_length in enumerate(prompt_lengths):
            labels[i, :int(prompt_length)] = -100

        full_batch["labels"] = labels
        return full_batch