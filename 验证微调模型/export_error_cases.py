#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_error_cases.py
============================================================================
根据 evaluate_val.py 生成的 predictions.jsonl，把所有错例（pred != gt）的
时频图汇总成一个可直接双击打开的 HTML 错例图集，并把错例图片复制到单独目录。

特点：
  - 按 “真值 -> 预测” 分组，错误越多的组排越前（SHIP_NOISE->BPSK 会在最前）；
  - 每张卡片显示：时频图、样本 id、真值、预测、模型原始输出、问题原文；
  - 顶部有 [全部] / [只看误判为 BPSK] 筛选按钮；
  - 默认把图片 base64 内嵌进单个 HTML，拷到本地/发给别人都能看，不依赖原图路径；
    加 --no_embed 则用相对路径引用（HTML 更小，需连同 error_cases 图片目录一起拷）。

典型用法（AutoDL，训练用的同一环境即可，只需 Pillow 之外几乎无依赖）：
  python export_error_cases.py
  # 结果：
  #   /root/autodl-tmp/LlamaFactory/eval_results/error_cases.html
  #   /root/autodl-tmp/LlamaFactory/eval_results/error_cases/*.png
============================================================================
"""

import argparse
import base64
import html
import json
import os
import shutil
from collections import defaultdict

DEFAULT_PRED = "/root/autodl-tmp/LlamaFactory/eval_results/predictions.jsonl"
DEFAULT_DATA = "/root/autodl-tmp/LlamaFactory/data/val_llamafactory.jsonl"
DEFAULT_OUT_DIR = "/root/autodl-tmp/LlamaFactory/eval_results"


def parse_args():
    ap = argparse.ArgumentParser(description="导出验证集错例 HTML 图集")
    ap.add_argument("--pred", default=DEFAULT_PRED, help="predictions.jsonl 路径")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="验证集 jsonl 路径（仅用于校正图片根目录）")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="输出目录")
    ap.add_argument("--html", default="", help="HTML 输出路径（默认 out_dir/error_cases.html）")
    ap.add_argument("--no_embed", action="store_true",
                    help="不内嵌 base64，改用相对路径引用图片")
    return ap.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_image(image_path, data_file):
    if os.path.exists(image_path):
        return image_path
    data_root = os.path.dirname(os.path.abspath(data_file))
    if "/data/" in image_path:
        rel = image_path.split("/data/")[-1]
        cand = os.path.join(data_root, rel)
        if os.path.exists(cand):
            return cand
    cand = os.path.join(data_root, os.path.basename(image_path))
    if os.path.exists(cand):
        return cand
    return image_path


def esc(s):
    return html.escape(str(s), quote=True)


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
       margin: 0; background: #f4f5f7; color: #1f2329; }
header { position: sticky; top: 0; z-index: 10; background: #fff; border-bottom: 1px solid #e5e6eb;
         padding: 16px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
h1 { margin: 0 0 8px; font-size: 20px; }
.summary { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 14px; }
.pill { background: #f2f3f5; border-radius: 999px; padding: 4px 12px; }
.pill b { color: #d83931; }
button { border: 1px solid #c9cdd4; background: #fff; border-radius: 6px; padding: 5px 14px;
         margin-right: 8px; cursor: pointer; font-size: 13px; }
button.active { background: #165dff; color: #fff; border-color: #165dff; }
main { padding: 20px 24px 60px; }
.group-header { grid-column: 1 / -1; margin: 22px 0 6px; font-size: 16px; font-weight: 700;
                padding: 8px 12px; border-left: 5px solid #d83931; background: #fff;
                border-radius: 4px; }
.group-header .cnt { color: #d83931; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
.card { background: #fff; border: 1px solid #e5e6eb; border-radius: 8px; overflow: hidden;
        display: flex; flex-direction: column; }
.card img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; background: #000;
            display: block; }
.missing { width: 100%; aspect-ratio: 1/1; display: flex; align-items: center; justify-content: center;
           background: #fbecec; color: #cb2634; font-size: 12px; padding: 10px; text-align: center; }
.meta { padding: 10px 12px 12px; font-size: 12.5px; line-height: 1.5; }
.badges { display: flex; gap: 6px; margin-bottom: 6px; flex-wrap: wrap; }
.tag { border-radius: 4px; padding: 2px 8px; font-weight: 700; color: #fff; font-size: 12px; }
.tag.gt { background: #00b42a; }
.tag.pred { background: #f53f3f; }
.id { color: #86909c; font-family: ui-monospace, Consolas, monospace; }
.field { margin-top: 5px; color: #4e5969; }
.field .k { color: #86909c; }
.raw { font-family: ui-monospace, Consolas, monospace; background: #f7f8fa; border-radius: 4px;
       padding: 2px 6px; word-break: break-all; }
.q { color: #6b7280; }
"""

JS = """
function applyFilter(pred) {
  document.querySelectorAll('.card, .group-header').forEach(function (el) {
    var show = (pred === '__ALL__') || (el.dataset.pred === pred);
    el.style.display = show ? '' : 'none';
  });
  document.querySelectorAll('.filters button').forEach(function (b) {
    b.classList.toggle('active', b.dataset.pred === pred);
  });
}
"""


def card_html(rec, img_src):
    gt, pred = rec["gt"], rec["pred"]
    if img_src is None:
        media = f'<div class="missing">图片未找到<br>{esc(rec["image"])}</div>'
    else:
        media = f'<img src="{img_src}" alt="{esc(gt)}->{esc(pred)}">'
    return f"""
    <div class="card" data-pred="{esc(pred)}">
      {media}
      <div class="meta">
        <div class="badges">
          <span class="tag gt">真值 {esc(gt)}</span>
          <span class="tag pred">预测 {esc(pred)}</span>
        </div>
        <div class="id">{esc(os.path.basename(rec["image"]))}</div>
        <div class="field"><span class="k">模型原话：</span><span class="raw">{esc(rec.get("raw_output",""))}</span></div>
        <div class="field q"><span class="k">问题：</span>{esc(rec.get("question",""))}</div>
      </div>
    </div>"""


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    html_path = args.html or os.path.join(args.out_dir, "error_cases.html")
    img_dir = os.path.join(args.out_dir, "error_cases")
    os.makedirs(img_dir, exist_ok=True)

    rows = read_jsonl(args.pred)
    errors = [r for r in rows if r.get("pred") != r.get("gt")]
    total = len(rows)
    print(f"[INFO] 总样本 {total}，错例 {len(errors)}")

    # 按 (gt -> pred) 分组，组按数量降序
    groups = defaultdict(list)
    for r in errors:
        groups[(r["gt"], r["pred"])].append(r)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))

    # 顶部汇总
    summary_items = "".join(
        f'<span class="pill">{esc(gt)} → {esc(pr)}：<b>{len(items)}</b></span>'
        for (gt, pr), items in ordered)

    blocks = []
    seq = 0
    for (gt, pr), items in ordered:
        items = sorted(items, key=lambda r: os.path.basename(r["image"]))
        blocks.append(
            f'<div class="group-header" data-pred="{esc(pr)}">'
            f'{esc(gt)} → {esc(pr)}　<span class="cnt">×{len(items)}</span></div>')
        cards = []
        for r in items:
            seq += 1
            src_path = resolve_image(r["image"], args.data)
            sample_id = os.path.splitext(os.path.basename(r["image"]))[0]
            dest_name = f"{seq:03d}_{gt}_to_{pr}_{sample_id}.png"
            dest_path = os.path.join(img_dir, dest_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
                if args.no_embed:
                    rel = os.path.relpath(dest_path, os.path.dirname(html_path)).replace("\\", "/")
                    img_src = rel
                else:
                    with open(src_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    img_src = f"data:image/png;base64,{b64}"
            else:
                img_src = None
            cards.append(card_html(r, img_src))
        blocks.append('<div class="grid">' + "".join(cards) + "</div>")

    acc = (total - len(errors)) / total if total else 0.0
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>验证集错例图集</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>验证集错例图集（共 {len(errors)} 个错误 / {total} 个样本，准确率 {acc*100:.2f}%）</h1>
  <div class="summary filters" style="margin-bottom:10px">
    <button data-pred="__ALL__" class="active" onclick="applyFilter('__ALL__')">全部</button>
    <button data-pred="BPSK" onclick="applyFilter('BPSK')">只看误判为 BPSK</button>
  </div>
  <div class="summary">{summary_items}</div>
</header>
<main>
  {''.join(blocks)}
</main>
<script>{JS}</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[INFO] HTML 已生成：{html_path}")
    print(f"[INFO] 错例图片已复制到：{img_dir}")
    if not args.no_embed:
        print("[INFO] HTML 已内嵌图片（单文件即可带走）；想要更小文件可加 --no_embed")


if __name__ == "__main__":
    main()
