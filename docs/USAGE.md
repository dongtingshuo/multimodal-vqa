# Usage Guide / 使用指南

This document describes the standard operation flow for the VQA project.

本文档说明 VQA 工程的标准使用流程。

## 1. Install Dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

For development / 开发环境：

```bash
pip install -r requirements-dev.txt
```

For Windows GPU environments, install a CUDA-enabled PyTorch package before running training.

Windows GPU 环境请先安装 CUDA 版 PyTorch，再安装其余依赖。

## 2. Prepare Data / 准备数据

```bash
python scripts/prepare_vqa_data.py --root data/vqa --full-coco-images --answer-vocab-size 3000
```

The command prepares VQA v2.0 annotations, question files, COCO 2014 images, and `data/answer_vocab.json`.

该命令会准备 VQA v2.0 标注、问题文件、COCO 2014 图片，以及 `data/answer_vocab.json`。

Known archives are downloaded with resume support and validated by size, checksum, ZIP CRC, and safe extraction path.

已知压缩包支持断点续传，并检查文件大小、校验和、ZIP CRC 与安全解压路径。

## 3. Download the Released Checkpoint / 下载已发布权重

Training is optional for inference and the web demo:

如果只进行推理或 Web 演示，可以直接下载已训练权重：

```bash
python scripts/download_checkpoint.py
```

The script verifies the published SHA256 digest before placing the file at `checkpoints/best.pt`.

脚本会校验发布页公布的 SHA256，然后将文件放置到 `checkpoints/best.pt`。

## 4. Train / 训练

```bash
python train.py --config configs/default.yaml
```

The best checkpoint is saved to:

最佳权重保存到：

```text
checkpoints/best.pt
```

The same directory also receives `training_history.csv`, `training_curves.png`, and `run_metadata.json`. Best-checkpoint selection uses validation `vqa_score` by default.

同一目录还会生成 `training_history.csv`、`training_curves.png` 和 `run_metadata.json`。默认按验证集 `vqa_score` 选择最佳权重。

## 5. Evaluate / 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

The output includes loss, hard Top-1 accuracy, VQA soft score, and Top-5 VQA score.

输出包含 loss、硬标签 Top-1 准确率、VQA soft score 和 Top-5 VQA score。

## 6. Run Inference / 执行推理

```bash
python infer.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt \
  --image data/vqa/val2014/COCO_val2014_000000000042.jpg \
  --question "What is in the image?" \
  --topk 5
```

## 7. Launch Web Demo / 启动 Web 演示

```bash
python demo.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

Open:

打开：

```text
http://127.0.0.1:8877
```

If port `8877` is busy, the demo automatically tries the next available port and prints it in the terminal.

如果 `8877` 被占用，演示程序会自动寻找后续可用端口，并在终端打印实际端口。

## Runtime Modes / 运行模式

### Full GPU Run / 全量 GPU 运行

```bash
python train.py --config configs/default.yaml
```

### CPU Smoke Test / CPU 连通性测试

```bash
python train.py --config configs/demo_cpu.yaml
```

### Offline Demo / 离线演示

```bash
python demo.py --offline
```

Use `--offline` only when Hugging Face model files are already cached locally.

仅当 Hugging Face 模型文件已存在于本地缓存时，才使用 `--offline`。

## Model Comparison / 模型对比

Run a lightweight offline comparison across `text_only`, `image_only`, `baseline_concat`, and `cross_attention`:

运行覆盖 `text_only`、`image_only`、`baseline_concat` 和 `cross_attention` 的轻量离线对比：

```bash
python scripts/run_model_comparison.py --config configs/demo_comparison.yaml
```

Reports are written to `outputs/comparison/`. The demo uses synthetic tensors and mock backbones, so the values validate the workflow rather than final model quality.

报告会写入 `outputs/comparison/`。该演示使用合成张量和 mock backbone，因此数值用于验证流程，而不是表示最终模型质量。

## Error Analysis / 错误分析

```bash
python scripts/run_error_analysis.py --predictions examples/toy_vqa_demo/toy_predictions.jsonl
```

Reports are written to `outputs/analysis/`.

报告会写入 `outputs/analysis/`。
