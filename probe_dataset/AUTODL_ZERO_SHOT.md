# AutoDL Zero-shot 实验运行手册

本手册只针对 **zero-shot 评测**，不涉及微调、不更新模型参数。

## 实验目标

在 AutoDL GPU 实例上运行 `Qwen/Qwen2-Audio-7B-Instruct`，评测它对当前水声 probe dataset 的基础识别能力：

- 信号类型识别：如单频、线性调频、脉冲串等。
- 主频/频率范围识别：先用选择题降低自动判分难度。
- 条件鲁棒性：比较无噪声、真实噪声轻度混合、真实噪声重度混合。
- 信道鲁棒性：比较直达信号和真实海洋参数 Bellhop 多径信道。

## 当前数据

本地已生成默认 probe set：

```text
data/zero_shot_probe/v1/
```

当前数据量很小，适合先做冒烟实验：

- 音频样本：`126` 条。
- 评测题目：`222` 条。
- 数据大小：约 `65 MB`。
- 条件矩阵：`direct/bellhop × clean/real_noise_light/real_noise_heavy`。

上传到 AutoDL 时至少需要：

```text
probe_dataset/
data/zero_shot_probe/v1/
```

如果只在 AutoDL 上跑 zero-shot，不需要上传 `bellhop.exe`、`exclude/` 原始噪声或海洋数据；因为音频已经在本地生成完毕。

## AutoDL 页面流程

已确认登录后的控制台入口：

```text
https://www.autodl.com/console/instance/list
```

页面路径：

```text
控制台 -> 容器实例 -> 租用新实例
```

创建实例页已看到的关键配置项：

- 计费方式：`按量计费`、`包日`、`包周`、`包月`。
- 地区：西北B区、北京B区、重庆A区、内蒙B区等。
- GPU 型号：`vGPU-32GB`、`RTX 5090`、`RTX PRO 6000`、`H800`、`RTX 4090D`、`RTX 4090`、`RTX 3090` 等。
- GPU 数量：先选 `1` 卡即可。
- 数据盘：默认免费 `50GB`，本次小数据集够用。
- 镜像：优先选基础镜像中的 `PyTorch + CUDA + Python` 环境。

## GPU 建议

`Qwen2-Audio-7B-Instruct` 是 7B 级别模型，建议按以下优先级租用：

- 首选：`RTX 4090 24GB`、`RTX 4090D 24GB`、`RTX 5090 32GB`、`vGPU-32GB`。
- 更稳：`RTX PRO 6000 96GB`、`H800 80GB`，但价格更高。
- 不建议：`RTX 3060 12GB`、`RTX A4000 16GB`，可能显存不足。
- 本轮只做小规模 zero-shot，先按量计费，跑通后立即关机。

## 镜像建议

优先选择基础镜像，关键词搜索：

```text
PyTorch / Python 3.10 / CUDA 12.x
```

推荐组合：

- Python：`3.10` 或 `3.11`。
- PyTorch：`2.x`。
- CUDA：`12.x`，与 AutoDL 页面显示的驱动兼容即可。

不需要一开始就定制镜像；先用基础镜像跑通，后续如果要反复实验，再保存为个人镜像。

## 上传方案

### 方案 A：压缩包上传

本地在仓库根目录打包最小运行包：

```powershell
tar -czf zero_shot_probe_autodl.tar.gz probe_dataset data/zero_shot_probe/v1
```

实例开机后，在 AutoDL 控制台复制 SSH 登录命令，格式大概是：

```text
ssh -p <port> root@<region-host>
```

然后在本地 PowerShell 上传压缩包：

```powershell
scp -P <port> zero_shot_probe_autodl.tar.gz root@<region-host>:/root/autodl-tmp/
```

上传到 AutoDL 后解压：

```bash
mkdir -p /root/autodl-tmp/srtp
tar -xzf /root/autodl-tmp/zero_shot_probe_autodl.tar.gz -C /root/autodl-tmp/srtp
cd /root/autodl-tmp/srtp
```

如果后续数据集变大且小文件很多，可以用 tar 流直传：

```powershell
tar cf - probe_dataset data/zero_shot_probe/v1 | ssh -p <port> root@<region-host> "mkdir -p /root/autodl-tmp/srtp && cd /root/autodl-tmp/srtp && tar xf -"
```

### 方案 B：Git + 单独上传数据

如果代码在 Git 仓库中：

```bash
cd /root/autodl-tmp
git clone <your-repo-url> srtp
cd srtp
```

然后只上传：

```text
data/zero_shot_probe/v1/
```

## 环境安装

进入项目目录后：

```bash
cd /root/autodl-tmp/srtp
python -m pip install -U pip
pip install -r probe_dataset/requirements_zero_shot.txt
```

建议把模型缓存放到数据盘，避免系统盘爆掉：

```bash
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
```

如果 Hugging Face 下载慢，再设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 冒烟测试

先只跑 5 道题，确认模型下载、音频读取和推理都正常：

```bash
python probe_dataset/evaluate_qwen2audio_zero_shot.py \
  --dataset data/zero_shot_probe/v1 \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --limit 5 \
  --batch-size 1 \
  --dtype float16
```

如果报显存不足，优先尝试：

```bash
python probe_dataset/evaluate_qwen2audio_zero_shot.py \
  --dataset data/zero_shot_probe/v1 \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --limit 5 \
  --batch-size 1 \
  --dtype float16 \
  --max-new-tokens 8
```

## 完整评测

冒烟测试通过后跑完整集：

```bash
python probe_dataset/evaluate_qwen2audio_zero_shot.py \
  --dataset data/zero_shot_probe/v1 \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --batch-size 1 \
  --dtype float16
```

如果 GPU 支持 BF16，也可以试：

```bash
python probe_dataset/evaluate_qwen2audio_zero_shot.py \
  --dataset data/zero_shot_probe/v1 \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --batch-size 1 \
  --dtype bfloat16
```

## 输出文件

默认输出在数据集目录下：

```text
data/zero_shot_probe/v1/qwen2audio_predictions.jsonl
data/zero_shot_probe/v1/qwen2audio_summary.json
```

`qwen2audio_predictions.jsonl` 每行包含：

- `task`：题型，如 `signal_type_choice`、`main_frequency_choice`。
- `prompt`：给模型的问题。
- `expected_answer`：标准答案。
- `response`：模型原始回答。
- `correct`：自动判分结果。
- `condition`：噪声与信道条件。

`qwen2audio_summary.json` 会按以下维度汇总：

- `overall`：总准确率。
- `task`：不同题型准确率。
- `noise`：不同噪声档位准确率。
- `channel`：`direct` vs `bellhop` 准确率。
- `task_noise` / `task_channel`：交叉分组准确率。

## 建议记录

每次实验至少记录：

```text
AutoDL 实例 GPU 型号：
镜像名：
Python/PyTorch/CUDA 版本：
transformers 版本：
模型名：Qwen/Qwen2-Audio-7B-Instruct
数据集路径：data/zero_shot_probe/v1
命令行参数：
运行耗时：
是否出现 OOM：
summary 文件路径：
```

查看版本命令：

```bash
python - <<'PY'
import torch, transformers
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('transformers:', transformers.__version__)
PY
```

## 结果解读

第一轮重点不是追求高分，而是判断基座模型有没有水声基础能力：

- 如果 `clean + direct` 的主频题都很差，说明模型很可能不能直接理解这类合成水声信号。
- 如果 `clean` 好但真实噪声差，后续数据集需要加强噪声鲁棒性训练。
- 如果 `direct` 好但 `bellhop` 差，后续需要重点做信道失真增强。
- 如果信号类型题好、主频题差，可以把微调任务拆成“分类”和“参数估计”两条线。

## 注意事项

- 创建实例会扣费；本轮建议只用按量计费，跑完立即关机或释放。
- AutoDL 页面显示实例连续关机一段时间后可能释放，重要结果要及时下载。
- 自动判分是选择题字符串匹配，正式汇报前建议抽样人工复核一部分模型回答。
- 这批 probe set 只用于 zero-shot 能力测试，不用于训练。
