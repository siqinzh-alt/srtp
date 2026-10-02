# AutoDL Qwen2.5-VL LoRA 微调记录

## 目标

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- 数据：`output/vlm_finetune/v1`
- 训练方式：单卡 32GB 上优先跑 4bit QLoRA。
- 任务：同一张诊断图可对应不同单问单答，但 train/val/test 图像不复用。

## 本地数据状态

- 训练集：`output/vlm_finetune/v1/train.jsonl`
- 验证集：`output/vlm_finetune/v1/val.jsonl`
- 测试集：`output/vlm_finetune/v1/test.jsonl`
- 数据报告：`output/vlm_finetune/v1/dataset_report.md`
- 图像目录：`output/vlm_finetune/v1/images`

当前报告显示：4000 张图、4996 条 QA，信号类型与问题类型基本均衡。

## 上传到 AutoDL

拿到当前实例 SSH 端口后，在本地仓库根目录执行：

```bash
tar cf - probe_dataset output/vlm_finetune/v1 | ssh -p <port> root@<region-host> "mkdir -p /root/autodl-tmp/srtp && cd /root/autodl-tmp/srtp && tar xf -"
```

如果 PowerShell 里没有 `tar`/`ssh` 兼容问题，也可以直接用同一条命令。`<port>` 和 `<region-host>` 以 AutoDL 控制台当前实例为准。

## 服务器启动

```bash
cd /root/autodl-tmp/srtp
bash probe_dataset/run_autodl_qwen25vl_lora.sh
```

默认输出：

- LoRA adapter：`output/qwen25vl_lora/v1/final_adapter`
- checkpoint：`output/qwen25vl_lora/v1/checkpoint-*`
- 训练日志：`output/qwen25vl_lora/v1/train_log.jsonl`
- 配置快照：`output/qwen25vl_lora/v1/train_config.json`

## 烟测命令

如果要先确认环境和数据读取正常：

```bash
cd /root/autodl-tmp/srtp
python -m pip install -r probe_dataset/requirements_finetune.txt
python -m py_compile probe_dataset/train_qwen25vl_lora.py
python probe_dataset/train_qwen25vl_lora.py \
  --dataset output/vlm_finetune/v1 \
  --output-dir output/qwen25vl_lora/smoke \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --epochs 1 \
  --save-steps 0 \
  --eval-steps 0 \
  --log-steps 1
```

## 备注

- 本机没有可用 NVIDIA GPU，因此实际训练应在 AutoDL 上跑。
- 32GB 显存优先保留 `--batch-size 1 --grad-accum 8 --max-pixels 1003520`。
- 如果显存不足，先把 `--max-pixels` 降到 `786432` 或 `602112`，不要先改数据集。
