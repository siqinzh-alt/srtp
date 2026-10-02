#!/usr/bin/env bash
set -euo pipefail

cd "${SRTP_ROOT:-/root/autodl-tmp/srtp}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$HF_HOME" output/qwen25vl_lora

python -m pip install -U pip
python -m pip install -r probe_dataset/requirements_finetune.txt

python probe_dataset/train_qwen25vl_lora.py \
  --dataset output/vlm_finetune/v1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --output-dir output/qwen25vl_lora/v1 \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-4 \
  --max-pixels 1003520 \
  --save-steps 200 \
  --eval-steps 100 \
  --log-steps 10

python probe_dataset/evaluate_qwen25vl_lora.py \
  --dataset output/vlm_finetune/v1 \
  --data-file test.jsonl \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter output/qwen25vl_lora/v1/final_adapter \
  --output output/qwen25vl_lora/v1/test_predictions.jsonl \
  --summary output/qwen25vl_lora/v1/test_summary.json \
  --batch-size 1 \
  --max-pixels 1003520
