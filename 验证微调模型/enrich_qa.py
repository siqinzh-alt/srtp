#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_qa.py
============================================================================
把 LLaMA-Factory 水声分类数据集里“只输出类别”的问答，增强为
“自然多样的问题 + 含类别名和时频特征描述的完整句子回答”。

设计要点
  - 每张图仍只产出 1 条问答（不扩量），顺序、images 路径与原文件一一对应；
  - 默认 20% 样本原样保留（问题+单个标签），保住模型“只输出标签”的能力；
    其余 80% 调用大模型改写成完整描述（比例可用 --keep_label_ratio 调整）；
  - 5 类时频“事实卡”内置，模型只能据此改写、类别强制锁定；
  - 回答必须包含正确英文标签，否则自动重试，最终失败回退到离线模板（标签必对）；
  - OpenAI 兼容接口，云端免费 API（智谱/硅基流动等）与本地 vLLM 通用；
  - 支持断点续跑（_enrich_cache.jsonl）、并发、--offline 纯模板离线生成。

云端免费 API 示例（二选一，model 以官网“免费模型”列表为准）
  智谱 GLM      --base_url https://open.bigmodel.cn/api/paas/v4 --model glm-4.7-flash
  硅基流动      --base_url https://api.siliconflow.cn/v1        --model Qwen/Qwen2.5-7B-Instruct
本地 vLLM 示例
  --base_url http://localhost:8000/v1 --model Qwen2.5-7B-Instruct --api_key EMPTY

用法
  python enrich_qa.py --in  /root/autodl-tmp/LlamaFactory/data/train_llamafactory.jsonl \
                      --base_url https://open.bigmodel.cn/api/paas/v4 \
                      --api_key 你的KEY --model glm-4.7-flash
  # 不联网 / 先验证流程（纯模板，标签必对）
  python enrich_qa.py --in train_llamafactory.jsonl --offline
============================================================================
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------- 5 类时频事实卡 ----------------------------- #
FACT = {
    "CW": {
        "title": "CW 单频载波脉冲",
        "fact": "能量在脉冲持续的时间段内集中于单一固定频率，形成一条水平、平直、连续的亮线；"
                "频率从头到尾不发生变化，仅在脉冲起止处有轻微淡入淡出，脉冲之外是安静背景（加噪后有均匀颗粒）。",
        "change": "频率不随时间变化，是一条恒定频率的水平线。",
        "contrast": "LFM 的线是倾斜的、频率连续变化，而 CW 水平不变；BPSK 是加宽、发毛的亮带，而 CW 是纤细干净的单线。",
    },
    "LFM": {
        "title": "LFM 线性调频脉冲",
        "fact": "能量沿一条随时间线性倾斜的亮线分布，频率从起始频率连续、匀速地扫到终止频率，"
                "可能由低到高（上行）或由高到低（下行），整体像一道斜杠；线条连续、不分叉、不跳变。",
        "change": "频率随时间呈线性（匀速）连续变化，表现为一条斜线。",
        "contrast": "CW 是频率恒定的水平线，LFM 是斜线；FSK 是在两个频率间台阶式跳变，LFM 则是连续滑动。",
    },
    "FSK": {
        "title": "2FSK 二进制频移键控",
        "fact": "用两个离散、固定的频率代表码元，能量在这两个频率高度之间来回阶跃切换，"
                "表现为上下两排交替出现的水平短划线/阶梯状图案；任一时刻能量只出现在其中一个频率上。",
        "change": "频率随时间在两个固定高度之间台阶式（阶跃）突变，码元速率越高切换越密集。",
        "contrast": "BPSK 始终围绕同一个载波频率展宽、不会跳到第二个高度；LFM 是连续斜线，FSK 是两个频率间的阶跃。",
    },
    "BPSK": {
        "title": "BPSK 二进制相移键控",
        "fact": "载波的频率和幅度恒定、相位在 0/π 间按码元翻转；相位翻转使频谱展宽，因此能量集中在"
                "单一载波频率附近、形成一条具有一定带宽、边缘发毛的水平亮带（不是 CW 那样纤细的单线）；"
                "码元速率越高亮带越宽、纹理越粗糙，码元切换瞬间可能有细微竖直瞬变。",
        "change": "中心频率不随时间移动、也不会跳到第二个频率，只是围绕该频率的能量被相位调制展宽。",
        "contrast": "CW 是纤细干净的水平线，BPSK 是加宽、毛糙的亮带；FSK 会跳到第二个频率高度，BPSK 只有一个中心频率。",
    },
    "SHIP_NOISE": {
        "title": "SHIP_NOISE 船舶辐射噪声",
        "fact": "来自真实船舶录音的宽带背景噪声，而非通信信号；表现为全程持续、无规则、不均匀的云雾状/"
                "颗粒状纹理，能量通常在低频更集中，可能夹杂模糊的机械窄带线谱，但不构成规则通信图案。",
        "change": "没有明确脉冲起止，也没有恒定载波或规律扫频/跳频，纹理随时间随机起伏。",
        "contrast": "BPSK 等通信信号有集中、规则、有限带宽的几何图案；船舶噪声是弥散、随机、宽带的背景。",
    },
}

# 任务类型：让问题/答案样式更丰富
TASKS = [
    ("classify_reason", "请先判断这是什么水声信号，再说明你是依据时频图上的哪些特征判断出来的。"),
    ("describe", "请具体描述这张时频图上能量的形态，以及它随时间是怎样变化的，最后点明类别。"),
    ("trend", "请重点说明图中信号的频率随时间如何变化（恒定/线性变化/阶跃跳变/无规则），并据此给出类别。"),
    ("contrast", "请把它和最容易混淆的另一类信号对比，说明区别后给出该图的类别。"),
]

# 离线兜底模板（不调用模型也能产出含标签、含特征的完整句）；按样本稳定选择
OFFLINE = {
    "CW": [
        ("这张时频图属于哪一类水声信号？判断依据是什么？",
         "这是 CW 单频载波脉冲信号。图中在脉冲持续时间内呈现一条频率恒定、不随时间变化的水平亮线，起止处有轻微淡入淡出。"),
        ("图中信号的频率随时间如何变化？",
         "该信号为 CW，频率从头到尾保持不变，在时频图上是一条平直连续的水平亮线。"),
    ],
    "LFM": [
        ("这张时频图属于哪一类水声信号？判断依据是什么？",
         "这是 LFM 线性调频信号。图中能量沿一条随时间线性倾斜的亮线分布，频率连续上升或下降，呈明显扫频轨迹。"),
        ("图中信号的频率随时间如何变化？",
         "该信号为 LFM，其频率随时间匀速、连续地变化，在时频图上表现为一道斜向亮线。"),
    ],
    "FSK": [
        ("这张时频图属于哪一类水声信号？判断依据是什么？",
         "这是 2FSK 频移键控信号。图中能量在两个固定频率高度之间台阶式来回跳变，形成上下两排交替的水平短划线。"),
        ("图中信号的频率随时间如何变化？",
         "该信号为 FSK，频率在两个离散频点之间阶跃突变而非连续滑动，码元速率越高切换越密集。"),
    ],
    "BPSK": [
        ("这张时频图属于哪一类水声信号？判断依据是什么？",
         "这是 BPSK 相移键控信号。图中能量集中在单一载波频率附近，被相位调制展宽成一条有带宽、边缘发毛的水平亮带，中心频率不发生跳变。"),
        ("如何把它和 FSK 区分开？",
         "该信号为 BPSK，它始终围绕同一个中心频率展宽，不会像 FSK 那样跳到第二个频率高度。"),
    ],
    "SHIP_NOISE": [
        ("这张时频图属于哪一类？请描述其特征。",
         "这是 SHIP_NOISE 船舶辐射噪声。图中是全程持续、宽频且无规则的云雾状颗粒背景，低频能量偏重，没有通信信号那种规则几何线条。"),
        ("如何区分它与通信信号？",
         "该记录为 SHIP_NOISE，其纹理弥散、随机且宽带，没有恒定载波或规律的扫频、跳频图案。"),
    ],
}

LABEL_ALIASES = {"SHIPNOISE": "SHIP_NOISE"}


def norm(s):
    return re.sub(r"[\s_\-]+", "", str(s).upper())


def label_in_text(label, text):
    """回答中必须原样包含正确英文标签（归一化匹配）。"""
    return norm(label) in norm(text)


def split_user_assistant(rec):
    q, a = "", ""
    for m in rec["messages"]:
        if m["role"] == "user":
            q = m["content"].replace("<image>", "").strip()
        elif m["role"] == "assistant":
            a = m["content"].strip()
    return q, a


def stable_hash(s):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def offline_qa(label, image):
    pool = OFFLINE[label]
    return pool[stable_hash(image) % len(pool)]


def build_prompt(label, task_name, task_instr):
    c = FACT[label]
    contrast = ""
    if task_name == "contrast":
        contrast = "\n【对比要点（请自然融入回答）】" + c["contrast"]
    return (
        "你是水声信号处理与训练数据构建专家。请根据下面给定的【信号类别】和【权威时频特征事实】，"
        "为一张水声时频图生成 1 个中文问题和 1 段中文回答。\n\n"
        f"【信号类别】{label}（{c['title']}）\n"
        f"【权威时频特征事实】{c['fact']}\n"
        f"【频率随时间变化】{c['change']}{contrast}\n\n"
        f"【本次问答任务】{task_instr}\n\n"
        "严格要求：\n"
        "1) 只能依据上述事实，不得杜撰具体频率、时间、带宽等数值，绝不允许改变类别；\n"
        "2) 问题要口语化、自然、措辞与给定任务相符，且问题中不要出现答案；\n"
        "3) 回答为 1~3 句通顺中文，必须具体描述时频图上的形态和随时间的变化，"
        f"并且必须原样包含英文类别标签 {label}；\n"
        "4) 只输出 JSON，不要任何多余文字，格式为：\n"
        '   {"question": "……", "answer": "……"}\n'
    )


SYS_PROMPT = "你是水声信号处理与训练数据构建专家，只依据给定事实作答，严格输出 JSON。"


def parse_json_content(content):
    content = content.strip()
    content = re.sub(r"^```(json)?|```$", "", content).strip()
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


class LLMClient:
    def __init__(self, base_url, api_key, model, temperature, timeout):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        import requests  # noqa
        self.requests = requests

    def chat(self, user_prompt):
        url = self.base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        r = self.requests.post(url, headers=headers, json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


class RateLimiter:
    """全局限速闸门：无论多少线程，每秒最多放行 qps 个请求（线程安全）。"""
    def __init__(self, qps):
        self.min_interval = (1.0 / qps) if qps and qps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_time = 0.0

    def acquire(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait = self.next_time - now
            self.next_time = max(now, self.next_time) + self.min_interval
        if wait > 0:
            time.sleep(wait)


def gen_one(label, image, client, max_retry, stats, limiter):
    # 稳定地随机选任务类型
    task_name, task_instr = TASKS[stable_hash(image + str(stats.get("salt", ""))) % len(TASKS)]
    prompt = build_prompt(label, task_name, task_instr)

    if client is None:  # 离线模式
        q, a = offline_qa(label, image)
        return q, a, "offline"

    last_err, rate_limited = "", False
    for attempt in range(max_retry + 1):
        limiter.acquire()
        try:
            content = client.chat(prompt)
            obj = parse_json_content(content)
            if obj and "question" in obj and "answer" in obj and label_in_text(label, obj["answer"]):
                return str(obj["question"]).strip(), str(obj["answer"]).strip(), "llm"
            last_err = "返回缺少字段或未包含正确标签"
            stats["retry"] += 1
            time.sleep(1.5)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            last_err = str(e)
            stats["retry"] += 1
            if status == 429:  # 触发速率限制 / 模型过载：按 Retry-After 长等待
                rate_limited = True
                resp = getattr(e, "response", None)
                body_txt = ""
                try:
                    body_txt = resp.text[:200] if resp is not None else ""
                except Exception:
                    body_txt = ""
                ra = resp.headers.get("Retry-After") if resp is not None else None
                try:
                    wait_s = min(max(float(ra), 1.0), 60.0) if ra else min(3.0 * (2 ** attempt), 45.0)
                except Exception:
                    wait_s = min(3.0 * (2 ** attempt), 45.0)
                print(f"  [429 限流] {body_txt}；等待 {wait_s:.0f}s 后重试（第 {attempt+1} 次）")
                time.sleep(wait_s)
            else:
                time.sleep(min(2 ** attempt * 0.8, 10))
    # 重试耗尽：用离线模板占位，保证标签必对、不丢样本
    q, a = offline_qa(label, image)
    stats["fallback_err"] = last_err
    if rate_limited:
        stats["rate_limited"] += 1
        return q, a, "rate_limited"   # 不写缓存：下次重跑会重新请求模型补全
    stats["fallback"] += 1
    return q, a, "fallback"


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        for line in open(path, "r", encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                cache[r["image"]] = r
    return cache


def main():
    ap = argparse.ArgumentParser(description="把分类 QA 增强为含时频描述的完整句子")
    ap.add_argument("--in", dest="inp", default="/root/autodl-tmp/LlamaFactory/data/train_llamafactory.jsonl")
    ap.add_argument("--out", default="", help="输出 jsonl（默认在输入旁加 _enriched）")
    ap.add_argument("--base_url", default="")
    ap.add_argument("--api_key", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=1, help="并发在途请求数（免费档建议 1，确认并发额度够再调高）")
    ap.add_argument("--qps", type=float, default=1.0, help="每秒最多发起请求数（免费档建议 ≤1）")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--max_retry", type=int, default=8)
    ap.add_argument("--keep_label_ratio", type=float, default=0.2,
                    help="保留为‘只输出标签’原始问答的比例")
    ap.add_argument("--offline", action="store_true", help="不调用模型，纯离线模板生成")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()

    if not args.offline and (not args.base_url or not args.model):
        print("[ERROR] 在线模式需要 --base_url 和 --model（或改用 --offline）。")
        sys.exit(1)

    out_path = args.out or (os.path.splitext(args.inp)[0] + "_enriched.jsonl")
    cache_path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_enrich_cache.jsonl")

    all_rows = [json.loads(l) for l in open(args.inp, "r", encoding="utf-8") if l.strip()]
    n_total = len(all_rows)
    row_range = range(args.limit) if args.limit else range(n_total)
    print(f"[INFO] 读入 {n_total} 条：{args.inp}"
          + (f"，本次仅处理前 {args.limit} 条" if args.limit else ""))

    # 保留样本抽样始终基于“全量数据”，保证先 --limit 试跑、再全量续跑时结果一致
    rng = random.Random(args.seed)
    by_label = {}
    for i, r in enumerate(all_rows):
        _, a = split_user_assistant(r)
        by_label.setdefault(a.strip(), []).append(i)
    keep_idx = set()
    for label, idxs in by_label.items():
        idxs = sorted(idxs)
        rng.shuffle(idxs)
        keep_idx.update(idxs[:round(len(idxs) * args.keep_label_ratio)])

    todo_all = []
    for i, r in enumerate(all_rows):
        if i in keep_idx:
            continue
        _, a = split_user_assistant(r)
        todo_all.append((i, a.strip(), r["images"][0]))
    todo = [t for t in todo_all if t[0] in row_range]
    print(f"[INFO] 全量保留纯标签问答 {len(keep_idx)} 条；本次需生成 {len(todo)} 条")

    client = None
    if not args.offline:
        client = LLMClient(args.base_url, args.api_key, args.model, args.temperature, args.timeout)
    stats = {"retry": 0, "fallback": 0, "rate_limited": 0, "fallback_err": "", "salt": args.seed}
    limiter = RateLimiter(args.qps if client is not None else 0)
    lock = threading.Lock()
    gen_map = {}   # image -> (question, answer, source)

    def work(item):
        _i, label, image = item
        q, ans, src = gen_one(label, image, client, args.max_retry, stats, limiter)
        return image, q, ans, src

    if client is None:
        # 离线模式：确定性模板生成，不读写缓存
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for fut in as_completed(futs):
                image, q, ans, _src = fut.result()
                gen_map[image] = (q, ans, _src)
    else:
        cache = load_cache(cache_path)
        for image, rec in cache.items():
            gen_map[image] = (rec["question"], rec["answer"], rec.get("source", "llm"))
        pending = [t for t in todo if t[2] not in gen_map]
        skipped = len(todo) - len(pending)
        if skipped:
            print(f"[INFO] 断点续跑：缓存命中 {skipped} 条")
        print(f"[INFO] 本次实际请求模型 {len(pending)} 条，并发 {args.workers}")
        cache_f = open(cache_path, "a", encoding="utf-8")
        done_cnt = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, t) for t in pending]
            for fut in as_completed(futs):
                image, q, ans, src = fut.result()
                with lock:
                    gen_map[image] = (q, ans, src)
                    if src != "rate_limited":   # 限流临时占位不缓存，下次重跑会重新请求模型
                        cache_f.write(json.dumps(
                            {"image": image, "question": q, "answer": ans, "source": src},
                            ensure_ascii=False) + "\n")
                        cache_f.flush()
                    done_cnt += 1
                    if done_cnt % 25 == 0 or done_cnt == len(pending):
                        print(f"  进度 {done_cnt}/{len(pending)}，重试 {stats['retry']}，"
                              f"回退 {stats['fallback']}，限流占位 {stats['rate_limited']}")
        cache_f.close()

    # 组装最终输出（仅 row_range，顺序与原文件一致）
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n_keep_out = n_gen = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in row_range:
            r = all_rows[i]
            if i in keep_idx:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")  # 原样保留
                n_keep_out += 1
                continue
            image = r["images"][0]
            q, ans, _src = gen_map[image]
            new_rec = {
                "messages": [
                    {"content": "<image>" + q, "role": "user"},
                    {"content": ans, "role": "assistant"},
                ],
                "images": r["images"],
            }
            f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            n_gen += 1

    print("\n[DONE] 输出：", out_path)
    print(f"       本次写出 {n_keep_out + n_gen} 条（纯标签保留 {n_keep_out}，增强描述 {n_gen}）；"
          f"重试 {stats['retry']}，回退 {stats['fallback']}，限流占位 {stats['rate_limited']}")
    if stats["rate_limited"]:
        print("       [提示] 有 {} 条因 429 限流暂时用模板占位（未写入缓存）。".format(stats["rate_limited"]))
        print("              等几分钟（或次日额度恢复后）原样重跑本命令，会自动把它们补成模型生成版本。")
    if stats["fallback"] or stats["rate_limited"]:
        print("       最后一次失败原因：", stats["fallback_err"][:200])


if __name__ == "__main__":
    main()
