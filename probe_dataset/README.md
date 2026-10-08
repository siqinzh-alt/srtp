# Zero-shot 水声基础能力 Probe Set

这个目录用于生成小规模、可复现、**不用于训练**的 zero-shot 评测集。目标是先测试基座音频大模型能否识别最基础的水声信号能力，而不是微调模型。

## 核心原则

- 只生成带真实标签的评测样本，不作为训练集。
- 噪声不单独作为一类；噪声只作为背景，按 SNR 混入信号。
- 加噪必须使用真实本地 WAV 噪声片段，默认从 `exclude` 下搜索。
- 多径条件必须调用仓库中的 `bellhop.exe`，由环境参数计算 `.arr` 到达结构后再卷积信号。
- 文件名不泄露答案，真实参数只写入 `metadata/` 和 `manifest.jsonl`。

## 测试目标

- 主频分辨：单频/CW 信号的候选主频选择。
- 类型识别：CW、LFM、FSK、BPSK 四选一。
- LFM 方向：升频/降频。
- FSK 阶数：2FSK/4FSK。
- 鲁棒性分层：比较无噪声、轻度真实噪声、重度真实噪声下的表现。
- 信道影响分层：比较 direct 与 Bellhop 多径传播后的表现。

## 难度与信道分层

默认每个基础信号会展开成 `3 × 2` 个版本：

| 维度 | 取值 | 含义 |
|---|---|---|
| 噪声难度 | `clean` | 不混入噪声 |
| 噪声难度 | `real_noise_light` | 从真实 WAV 噪声中截取片段，按约 `20 dB SNR` 混入 |
| 噪声难度 | `real_noise_heavy` | 从真实 WAV 噪声中截取片段，按约 `0 dB SNR` 混入 |
| 信道条件 | `direct` | 不经过 Bellhop，直接保存源信号/加噪信号 |
| 信道条件 | `bellhop` | 写 `.env`，调用 `bellhop.exe`，读取 `.arr`，按多径到达结构生成接收信号 |

## Bellhop 参数来源

当前版本先复用集体项目中已经跑通的 Bellhop 模板参数，来源包括：

- `generate_cw_example.m`
- `dataset_generator/generate_bellhop_sample.m`
- `dataset_generator/make_cw_parameter_set.m`
- `bellhop_tools/build_ssp.m`

Python 脚本按这些模板写 Bellhop `.env` 文件，包含：

- 水深剖面：`0, 10, 20, 40, 60, 80, 100 m`
- 温度/盐度剖面：来自现有脚本示例和同范围模板
- 源深/接收深/距离：使用现有脚本中的 `20/50 m, 1 km` 及同范围组合
- 声速公式：与 `build_ssp.m` 一致的 Mackenzie-style 公式
- 海底参数：底声速约 `1600 m/s`，密度约 `1.8 g/cm³`
- Bellhop 设置：`RunType='A'`，`Nbeams=101`，角度范围约 `-30°` 到 `30°`

后续如果补齐 WOA23 温度数据，可以把环境模板替换为 WOA 温盐剖面生成的真实海域声速剖面。

## 目录约定

- `data/`：正式数据集和外部数据缓存，例如 `data/vlm_finetune/v3`、`data/sources/watkins`。
- `output/`：模型训练、评测和推理产物，例如 `output/qwen25vl_lora/v3`。
- `.tmp/`：烟测、临时脚本和可随时删除的中间文件。

## 生成命令

生成默认完整分层 probe set：

```powershell
python .\probe_dataset\generate_probe_dataset.py
```

默认输出到：

```text
data/zero_shot_probe/v1/
```

生成一批更小的快速检查版本（少量 CW/LFM/FSK，1 条 BPSK）：

```powershell
python .\probe_dataset\generate_probe_dataset.py --output data\zero_shot_probe\tiny --cw-repeats 1 --lfm-repeats 1 --fsk-repeats 1 --bpsk-count 1
``` 

只生成最简单的无噪声、无 Bellhop 版本：

```powershell
python .\probe_dataset\generate_probe_dataset.py --output data\zero_shot_probe\clean_direct --noise-levels clean --channel-conditions direct
```

只生成 Bellhop + 三档噪声版本：

```powershell
python .\probe_dataset\generate_probe_dataset.py --output data\zero_shot_probe\bellhop_only --channel-conditions bellhop
```

## VLM 微调数据集

正式微调数据集使用独立入口，避免和 zero-shot probe 混用：

```powershell
python .\probe_dataset\generate_vlm_finetune_dataset.py --output data\vlm_finetune\v1
```

默认配置会生成：

- `CW/LFM/FSK/BPSK` 四类唯一样本均衡，每类 1000 张统一三联图，直接存放在 `images/` 下。
- 约 5000 条 QA，按固定题型概率分布生成，而不是简单五类选择题均分。
- 单题/多题比例约为 `65%/35%`；单题内部选择题/开放短答约为 `35%/65%`。
- 折算总体约为 `23%` 选择题、`42%` 单题开放短答、`35%` 多字段 JSON 开放题。
- 题目字段覆盖 `signal_type`、`main_frequency_hz`、`lfm_direction`、`fsk_order`、`bpsk_symbol_rate`、`noise_level`、`channel_condition`。
- 多题答案固定为 JSON；单题开放题只输出短答案，不要求解释。
- 条件比例约为 `70% clean/direct`、`10% real_noise_light/direct`、`10% real_noise_heavy/direct`、`10% clean/bellhop`。
- `train/val/test = 80/10/10`，同一图片和同一基础信号不会跨 split。
- 同一图片最多生成 3 条 QA，避免少数图片过度重复。

快速烟测可以使用更小规模：

```powershell
python .\probe_dataset\generate_vlm_finetune_dataset.py --output .tmp\vlm_finetune_smoke --samples-per-class 24 --qa-per-task 24 --overwrite
```

如果需要直接指定 QA 总数，可以使用：

```powershell
python .\probe_dataset\generate_vlm_finetune_dataset.py --output data\vlm_finetune\v2 --target-qa-items 5000
```

如果要在合成数据集基础上再加入 1000 条公开动物/生物声学信号样本，可以生成 `v3`：

```powershell
python .\probe_dataset\generate_vlm_finetune_with_animals.py --base-dataset data\vlm_finetune\v1 --output data\vlm_finetune\v3 --animal-samples 1000 --overwrite
```

`v3` 会保留基线的 4000 条合成 `CW/LFM/FSK/BPSK` 样本，并加入 Watkins Marine Mammal Sound Database 的动物声学样本：

- 动物样本下载 Watkins 原始 FLAC 音频，转为统一的 16 kHz/2 s WAV，再使用与合成样本相同的三联图格式，图片也直接存放在 `images/` 下。
- 动物题型覆盖 `bio_species_or_group`、`bio_species_code`、`bio_call_description`，仍保持单题/多题和选择/开放题的概率分布。
- 动物样本按物种尽量均衡抽样，默认选 Watkins 元数据中记录数最多的 20 个物种，每张图最多复用 3 条 QA。
- `train/val/test = 80/10/10`，同一 Watkins 录音号不会跨 split。
- 外部 FLAC 音频会缓存到 `data\sources\watkins\audio_flac`，便于中断后继续生成。

微调集输出结构：

```text
manifest.jsonl       # 唯一音频/图片样本清单
qa_items.jsonl       # 全部 QA 清单
train.jsonl          # Qwen2.5-VL SFT 训练集
val.jsonl            # Qwen2.5-VL SFT 验证集
test.jsonl           # Qwen2.5-VL SFT 测试集
split_manifest.json  # split 分组记录
dataset_report.md    # 分布统计与校验报告
```

## 输出结构

```text
audio/             # WAV 音频，文件名不泄露答案
spectrogram/       # 谱图，用于人工检查
metadata/          # 每条音频的隐藏真值、噪声来源、Bellhop 到达结构
bellhop/           # 每条 Bellhop 样本的 .env/.arr 等中间文件
manifest.jsonl     # 样本级清单
eval_items.jsonl   # 评测题目清单
dataset_config.json
noise_sources.json # 本次可用真实噪声文件列表
```

## `eval_items.jsonl` 示例

```json
{
  "id": "probe_000001_type_choice",
  "sample_id": "probe_000001",
  "audio_path": "audio/probe_000001.wav",
  "task": "signal_type_choice",
  "prompt": "<audio>这段信号属于 CW、LFM、FSK、BPSK 中的哪一种？只回答一个选项。",
  "choices": ["CW", "LFM", "FSK", "BPSK"],
  "expected_answer": "CW",
  "scoring": {"type": "choice_exact"},
  "condition": {
    "noise_level": "real_noise_light",
    "channel_condition": "bellhop",
    "bellhop_processed": true,
    "target_snr_db": 20.0,
    "actual_snr_db": 20.0
  }
}
```

