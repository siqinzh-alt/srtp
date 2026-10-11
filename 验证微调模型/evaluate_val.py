#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_val.py
============================================================================
对 LLaMA-Factory 微调后的 LLaVA-1.5-7B LoRA 做验证集批量推理，
并把模型输出与验证集标准答案（BPSK / CW / LFM / FSK / SHIP_NOISE）比对。

产出（写入 --out_dir）：
  predictions.jsonl   每条样本的 {image, question, gt, pred, raw_output, exact, correct}
  metrics.json        总体准确率、每类 P/R/F1、混淆矩阵、未解析数
  report.txt          可读文本报告
  confusion_matrix.png 混淆矩阵图（若装有 matplotlib）

典型用法（在 AutoDL、训练用的同一个 conda/venv 环境里）：
  # 1) 先小样本试跑 20 条，确认环境/模板/路径都正常
  python evaluate_val.py --limit 20
  # 2) 全量评估（支持中断续跑）
  python evaluate_val.py
============================================================================
"""

import argparse
import gc
import json
import os
import re
import sys

# ----------------------------- 常量配置 ----------------------------- #
# 五个封闭集标签（与训练 / 验证集一致，勿随意改名）
LABELS = ["BPSK", "CW", "LFM", "FSK", "SHIP_NOISE"]

# 默认路径（可用命令行参数覆盖）
DEFAULT_ADAPTER = "/root/autodl-tmp/LlamaFactory/saves/LLaVA-1.5-7B-Chat/lora/train_2026-10-08-20-53-14"
DEFAULT_DATA = "/root/autodl-tmp/LlamaFactory/data/val_llamafactory.jsonl"
DEFAULT_OUT = "/root/autodl-tmp/LlamaFactory/eval_results"

# LLaVA-1.5（Vicuna 模板）推理前缀，与 LLaMA-Factory 的 llava 模板一致
PROMPT_TEMPLATE = "USER: <image>\n{question} ASSISTANT:"


# ----------------------------- 工具函数 ----------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="LLaVA LoRA 验证集准确率评估")
    ap.add_argument("--adapter_path", default=DEFAULT_ADAPTER,
                    help="LoRA 微调输出目录（含 adapter_config.json）")
    ap.add_argument("--base_model", default="",
                    help="基座模型路径/HF id；留空则自动读 adapter_config.json")
    ap.add_argument("--data", default=DEFAULT_DATA, help="验证集 jsonl 路径")
    ap.add_argument("--out_dir", default=DEFAULT_OUT, help="结果输出目录")
    ap.add_argument("--limit", type=int, default=0,
                    help="只评估前 N 条（0 表示全量）；建议先 --limit 20 试跑")
    ap.add_argument("--max_new_tokens", type=int, default=16,
                    help="每条最大生成 token 数（分类标签很短，16 足够）")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--no_resume", action="store_true",
                    help="不使用断点续跑，覆盖旧的 predictions.jsonl")
    return ap.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_sample(rec):
    """从 LLaMA-Factory sharegpt 风格记录里取出 (图片路径, 纯问题, 标准答案)。"""
    msgs = rec["messages"]
    question, gt = "", ""
    for m in msgs:
        if m["role"] == "user":
            question = m["content"].replace("<image>", "").strip()
        elif m["role"] == "assistant":
            gt = m["content"].strip()
    image = rec["images"][0] if rec.get("images") else ""
    return image, question, gt


def resolve_image(image_path, data_file):
    """jsonl 里一般是绝对路径；若不存在，则按数据文件所在目录校正。"""
    if os.path.exists(image_path):
        return image_path
    data_root = os.path.dirname(os.path.abspath(data_file))
    if "/data/" in image_path:
        rel = image_path.split("/data/")[-1]          # 例如 val/ocean/sample_xxx.png
        cand = os.path.join(data_root, rel)
        if os.path.exists(cand):
            return cand
    cand = os.path.join(data_root, os.path.basename(image_path))
    if os.path.exists(cand):
        return cand
    return image_path  # 返回原路径，交由打开处报错


def _norm(text):
    """归一化：大写、去掉空格/下划线/连字符/常见标点，便于宽松匹配标签。"""
    return re.sub(r"[\s_\-\.,，。:：;；!！?？、]+", "", str(text).upper())


_NORM_MAP = {_norm(l): l for l in LABELS}


def parse_prediction(raw):
    """
    从模型原始输出中解析出标签。
    返回 (pred, exact)：
      pred  : LABELS 之一，或 'UNPARSED'
      exact : 是否“干净地”只输出了标签（用于衡量模型是否听话）
    """
    t = str(raw).strip()
    nt = _norm(t)
    if nt in _NORM_MAP:
        return _NORM_MAP[nt], True
    # 宽松匹配：在整段输出里找标签（长标签优先，避免子串问题）
    for key in sorted(_NORM_MAP, key=len, reverse=True):
        if key in nt:
            return _NORM_MAP[key], False
    return "UNPARSED", False


def get_base_model(adapter_path, override):
    if override:
        return override
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path", "")
    if not base:
        raise RuntimeError(f"未能从 {cfg_path} 读到 base_model_name_or_path，请用 --base_model 手动指定")
    return base


# ----------------------------- 主流程 ----------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pred_path = os.path.join(args.out_dir, "predictions.jsonl")

    # ---- 读取验证集 ----
    records = read_jsonl(args.data)
    if args.limit > 0:
        records = records[:args.limit]
    print(f"[INFO] 验证集共 {len(records)} 条：{args.data}")

    # ---- 断点续跑：跳过已写入 predictions.jsonl 的图片 ----
    done = set()
    if os.path.exists(pred_path) and not args.no_resume:
        for r in read_jsonl(pred_path):
            done.add(os.path.basename(r.get("image", "")))
        if done:
            print(f"[INFO] 断点续跑：已完成 {len(done)} 条，将跳过")

    # ---- 加载模型 ----
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import LlavaForConditionalGeneration as ModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as ModelCls
    from peft import PeftModel

    base_model = get_base_model(args.adapter_path, args.base_model)
    print(f"[INFO] 基座模型：{base_model}")
    print(f"[INFO] LoRA 目录：{args.adapter_path}")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = ModelCls.from_pretrained(
        base_model, torch_dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    print(f"[INFO] 模型加载完成，device={device}, dtype={model_dtype}")

    # ---- 逐条推理 ----
    try:
        from tqdm import tqdm
        iterator = tqdm(records, desc="infer", ncols=88)
    except Exception:
        iterator = records

    fout = open(pred_path, "a", encoding="utf-8")
    n_done = 0
    try:
        for rec in iterator:
            image_path, question, gt = extract_sample(rec)
            img_name = os.path.basename(image_path)
            if img_name in done:
                continue

            img = Image.open(resolve_image(image_path, args.data)).convert("RGB")
            prompt = PROMPT_TEMPLATE.format(question=question)
            inputs = processor(images=img, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=pad_id)
            n_in = inputs["input_ids"].shape[1]
            raw = processor.decode(out[0, n_in:], skip_special_tokens=True).strip()

            pred, exact = parse_prediction(raw)
            row = {
                "image": image_path,
                "question": question,
                "gt": gt,
                "pred": pred,
                "raw_output": raw,
                "exact": exact,
                "correct": int(pred == gt),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
    finally:
        fout.close()
    print(f"[INFO] 本次新推理 {n_done} 条，明细已写入 {pred_path}")

    # ---- 汇总统计（基于全量 predictions.jsonl，保证续跑后统计完整）----
    all_rows = read_jsonl(pred_path)
    if args.limit > 0:
        all_rows = all_rows[:len(records)] if len(all_rows) >= len(records) else all_rows
    summarize(all_rows, args)

    # 释放显存
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def summarize(rows, args):
    n = len(LABELS)
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = [[0] * n for _ in range(n)]      # 行=真值, 列=预测
    unparsed_by_gt = {l: 0 for l in LABELS}
    exact_cnt = 0
    correct_cnt = 0
    bad_gt = 0

    for r in rows:
        gt, pred = r["gt"], r["pred"]
        if r.get("exact"):
            exact_cnt += 1
        if pred == gt:
            correct_cnt += 1
        if gt not in idx:
            bad_gt += 1
            continue
        if pred in idx:
            cm[idx[gt]][idx[pred]] += 1
        else:  # UNPARSED
            unparsed_by_gt[gt] += 1

    total = len(rows)
    acc = correct_cnt / total if total else 0.0
    exact_rate = exact_cnt / total if total else 0.0
    unparsed_total = sum(unparsed_by_gt.values())

    per_class = {}
    for i, l in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp
        support = sum(cm[i]) + unparsed_by_gt[l]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn + unparsed_by_gt[l]) if (tp + fn + unparsed_by_gt[l]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[l] = {"precision": round(precision, 4), "recall": round(recall, 4),
                        "f1": round(f1, 4), "support": support,
                        "unparsed": unparsed_by_gt[l]}

    metrics = {
        "total": total,
        "accuracy": round(acc, 4),
        "exact_label_rate": round(exact_rate, 4),
        "unparsed": unparsed_total,
        "labels": LABELS,
        "confusion_matrix": cm,
        "per_class": per_class,
        "gt_not_in_labels": bad_gt,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---- 文本报告 ----
    lines = []
    lines.append("=" * 60)
    lines.append("LLaVA LoRA 验证集评估报告")
    lines.append("=" * 60)
    lines.append(f"数据文件     : {args.data}")
    lines.append(f"LoRA 目录    : {args.adapter_path}")
    lines.append(f"样本总数     : {total}")
    lines.append(f"总体准确率   : {acc:.4f}  ({correct_cnt}/{total})")
    lines.append(f"干净标签率   : {exact_rate:.4f}  (模型只输出标签的比例)")
    lines.append(f"未解析条数   : {unparsed_total}  (输出里找不到任何标签，计为错误)")
    lines.append("")
    header = f"{'类别':<12}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}{'unparsed':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for l in LABELS:
        m = per_class[l]
        lines.append(f"{l:<12}{m['precision']:>8.4f}{m['recall']:>8.4f}"
                     f"{m['f1']:>8.4f}{m['support']:>9d}{m['unparsed']:>10d}")
    lines.append("")
    lines.append("混淆矩阵 (行=真值, 列=预测; 顺序 " + "/".join(LABELS) + ")：")
    lines.append(f"{'gt\\pred':<12}" + "".join(f"{l[:6]:>9}" for l in LABELS))
    for i, l in enumerate(LABELS):
        lines.append(f"{l:<12}" + "".join(f"{cm[i][j]:>9d}" for j in range(n)))
    report = "\n".join(lines)
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n" + report)

    # ---- 混淆矩阵图（可选）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        arr = np.array(cm, dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
        ax.set_title(f"Confusion matrix  acc={acc:.4f}  unparsed={unparsed_total}")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(arr[i, j]), ha="center", va="center",
                        color="black" if arr[i, j] < arr.max() * 0.6 else "white",
                        fontsize=9)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)
        print(f"[INFO] 混淆矩阵图已保存")
    except Exception as e:
        print(f"[WARN] 未生成混淆矩阵图（{e}），不影响 json/txt 结果")


if __name__ == "__main__":
    main()
