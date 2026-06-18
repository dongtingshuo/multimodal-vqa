# Usage Guide / 使用指南

This document describes the standard operation flow for the VQA project.

本文档说明 VQA 工程的标准使用流程。

## 1. Install Dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

For Windows GPU environments, install a CUDA-enabled PyTorch package before running training.

Windows GPU 环境请先安装 CUDA 版 PyTorch，再安装其余依赖。

## 2. Prepare Data / 准备数据

```bash
python scripts/prepare_vqa_data.py --root data/vqa --full-coco-images --answer-vocab-size 3000
```

The command prepares VQA v2.0 annotations, question files, COCO 2014 images, and `data/answer_vocab.json`.

该命令会准备 VQA v2.0 标注、问题文件、COCO 2014 图片，以及 `data/answer_vocab.json`。

## 3. Train / 训练

```bash
python train.py --config configs/default.yaml
```

The best checkpoint is saved to:

最佳权重保存到：

```text
checkpoints/best.pt
```

## 4. Evaluate / 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## 5. Run Inference / 执行推理

```bash
python infer.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt \
  --image data/vqa/val2014/COCO_val2014_000000000042.jpg \
  --question "What is in the image?" \
  --topk 5
```

## 6. Launch Web Demo / 启动 Web 演示

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
