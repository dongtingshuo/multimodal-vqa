# Multimodal Visual Question Answering (VQA)

[![CI](https://github.com/dongtingshuo/multimodal-vqa/actions/workflows/ci.yml/badge.svg)](https://github.com/dongtingshuo/multimodal-vqa/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](pyproject.toml)

基于多模态融合的视觉问答系统。工程提供数据准备、模型训练、评估、命令行推理、Gradio Web 演示、模型变体对比和错误分析流程，默认面向 VQA v2.0 与 COCO 2014 图像数据。

A research-oriented multimodal Visual Question Answering system with data preparation, model training, evaluation, command-line inference, a Gradio web demo, model-variant comparison, and error analysis. The default workflow targets VQA v2.0 with COCO 2014 images.

## Core Capabilities / 核心能力

- **Multimodal architecture / 多模态架构**: ResNet-50 image encoder, DistilBERT text encoder, bidirectional cross attention, and answer classification head.
- **Model variants / 模型变体**: `text_only`, `image_only`, `baseline_concat`, and `cross_attention` are available through a shared model factory.
- **Comparison workflow / 对比流程**: lightweight report generation for validating model-variant comparison without external downloads.
- **Question-type error analysis / 问题类型错误分析**: keyword-based diagnostics for color, count, object, location, yes/no, and other questions.
- **Full VQA workflow / 完整 VQA 流程**: data preparation, vocabulary construction, training, validation, checkpointing, inference, and web demo.
- **GPU-first training / GPU 优先训练**: CUDA, AMP mixed precision, DataLoader workers, pinned memory, and persistent workers are enabled in the default config.
- **Reproducible configuration / 可复现实验配置**: all runtime settings are centralized in YAML config files.
- **VQA-aware evaluation / VQA 评估**: soft-answer VQA score, Top-5 VQA score, hard Top-1 accuracy, and per-example multilabel loss.
- **Experiment tracking / 实验记录**: epoch history, curves, learning rate, elapsed time, environment versions, and Git commit metadata.
- **Self-describing checkpoints / 自描述权重**: architecture and preprocessing settings are loaded from checkpoint metadata while local paths remain configurable.
- **Toy demo / 玩具演示**: tiny example files support offline workflow checks for reports and diagnostics.
- **Operational demo / 可运行演示**: Gradio interface supports image upload, question input, and Top-k answer display.
- **Robust local runtime / 稳定本地运行**: demo handles missing checkpoints, occupied ports, CUDA fallback, and Hugging Face cache behavior with clear messages.

## Repository Layout / 工程结构

```text
.
├── configs/
│   ├── default.yaml          # Full GPU training config / 全量 GPU 训练配置
│   ├── cross_attention.yaml  # Cross-modal attention variant / 跨模态注意力变体
│   ├── baseline_concat.yaml  # Feature-concatenation baseline / 特征拼接基线
│   ├── text_only.yaml        # Text-only baseline / 纯文本基线
│   ├── image_only.yaml       # Image-only baseline / 纯图像基线
│   ├── demo_comparison.yaml  # Offline comparison demo / 离线对比演示
│   └── demo_cpu.yaml         # Lightweight smoke-test config / 轻量级 CPU 测试配置
├── examples/
│   └── toy_vqa_demo/         # Tiny public workflow sample / 极小公开流程样例
├── scripts/
│   ├── prepare_vqa_data.py   # VQA v2.0 and COCO data preparation / 数据准备脚本
│   ├── download_checkpoint.py # Verified Release download / 校验 Release 权重下载
│   ├── run_model_comparison.py
│   └── run_error_analysis.py
├── tests/
│   └── test_*.py             # Offline unit and workflow tests / 离线单元与流程测试
├── vqa_project/
│   ├── analysis/             # Error-analysis utilities / 错误分析工具
│   ├── answers.py            # Answer normalization and vocabulary / 答案规范化与词表
│   ├── config.py             # YAML config loading and device resolution / 配置加载与设备解析
│   ├── data.py               # Dataset and collator / 数据集与批处理
│   ├── engine.py             # Training, evaluation, checkpoints / 训练、评估与权重保存
│   ├── hf.py                 # Hugging Face model helpers / Hugging Face 加载辅助
│   ├── inference.py          # Single-image inference / 单图推理
│   ├── tracking.py           # History, curves, runtime metadata / 历史、曲线与运行元数据
│   └── model.py              # Multimodal VQA model / 多模态模型
├── train.py                  # Training entrypoint / 训练入口
├── evaluate.py               # Evaluation entrypoint / 评估入口
├── infer.py                  # CLI inference entrypoint / 命令行推理入口
├── demo.py                   # Gradio web demo / Web 演示入口
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Environment / 环境准备

Python 3.9+ is recommended. For GPU training, use a CUDA-enabled PyTorch build that matches your NVIDIA driver.

建议使用 Python 3.9+。如果进行 GPU 训练，请安装与 NVIDIA 驱动匹配的 CUDA 版 PyTorch。

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Development dependencies / 开发依赖：

```bash
pip install -r requirements-dev.txt
```

Check CUDA availability / 检查 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Data Preparation / 数据准备

Default data root / 默认数据目录：

```text
data/vqa
```

Full training requires:

全量训练需要：

- VQA v2.0 train/val annotations and questions
- COCO 2014 `train2014` and `val2014` images
- An answer vocabulary saved to `data/answer_vocab.json`

Prepare full data / 准备全量数据：

```bash
python scripts/prepare_vqa_data.py --root data/vqa --full-coco-images --answer-vocab-size 3000
```

Expected structure / 目标结构：

```text
data/
├── answer_vocab.json
└── vqa/
    ├── train2014/
    ├── val2014/
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    ├── v2_mscoco_train2014_annotations.json
    ├── v2_OpenEnded_mscoco_val2014_questions.json
    └── v2_mscoco_val2014_annotations.json
```

COCO image archives are large. Keep dataset files outside Git.

COCO 图片压缩包体积较大，数据文件不应提交到 Git。

The downloader resumes `.part` files and verifies known archive sizes, checksums, ZIP CRC values, and extraction paths.

下载器支持从 `.part` 文件断点续传，并校验已知压缩包大小、校验和、ZIP CRC 与解压路径。

## Trained Checkpoint / 已训练模型

Download and verify the published `v0.1.0` checkpoint / 下载并校验已发布的 `v0.1.0` 权重：

```bash
python scripts/download_checkpoint.py
```

- [GitHub Release](https://github.com/dongtingshuo/multimodal-vqa/releases/tag/v0.1.0)
- [Direct checkpoint download / 权重直链](https://github.com/dongtingshuo/multimodal-vqa/releases/download/v0.1.0/best.pt)
- SHA256: `d9638309c2e74a30479332eaabb0f27869555d967eefae0a1eaf981342c3f98c`
- Size / 大小: `427,216,108` bytes

## Training / 模型训练

Default full-training command / 默认全量训练命令：

```bash
python train.py --config configs/default.yaml
```

Default training config / 默认训练配置：

```yaml
device: cuda
answer_vocab_size: 3000
max_train_samples: null
max_val_samples: null
batch_size: 32
epochs: 10
use_amp: true
num_workers: 4
```

Checkpoint output / 权重输出：

```text
checkpoints/best.pt
```

Training artifacts / 训练产物：

```text
checkpoints/training_history.csv
checkpoints/training_curves.png
checkpoints/run_metadata.json
```

The best checkpoint is selected by validation `vqa_score` by default. A ReduceLROnPlateau scheduler tracks the same metric.

默认按验证集 `vqa_score` 选择最佳权重，ReduceLROnPlateau 学习率调度器也追踪该指标。

For a fast smoke test / 快速连通性测试：

```bash
python train.py --config configs/demo_cpu.yaml
```

## Evaluation / 模型评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

Evaluation reports per-example multilabel loss, hard Top-1 accuracy, VQA soft score, and Top-5 VQA score.

评估输出按样本归一化的多标签 loss、硬标签 Top-1 准确率、VQA soft score 和 Top-5 VQA score。

## Quick Demo / 快速演示

Run an offline model-variant comparison / 运行离线模型变体对比：

```bash
python scripts/run_model_comparison.py --config configs/demo_comparison.yaml
```

Run toy error analysis / 运行玩具错误分析：

```bash
python scripts/run_error_analysis.py --predictions examples/toy_vqa_demo/toy_predictions.jsonl
```

The generated reports are written under `outputs/`. These toy outputs validate the workflow and are not real benchmark results.

生成的报告会写入 `outputs/`。这些玩具输出用于验证流程，不是正式 benchmark 结果。

## Command-Line Inference / 命令行推理

```bash
python infer.py ^
  --config configs/default.yaml ^
  --checkpoint checkpoints/best.pt ^
  --image data/vqa/val2014/COCO_val2014_000000000042.jpg ^
  --question "What is in the image?" ^
  --topk 5
```

macOS/Linux:

```bash
python infer.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt \
  --image data/vqa/val2014/COCO_val2014_000000000042.jpg \
  --question "What is in the image?" \
  --topk 5
```

## Web Demo / Web 演示

```bash
python demo.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

Open / 打开：

```text
http://127.0.0.1:8877
```

Useful options / 常用参数：

```bash
python demo.py --server-port 8877 --inbrowser
python demo.py --offline
python demo.py --share
```

`--offline` only uses locally cached Hugging Face files. Do not use it if the model tokenizer has not been cached yet.

`--offline` 只读取本地 Hugging Face 缓存；如果 tokenizer 尚未缓存，请不要使用该参数。

## Runtime Artifacts / 运行产物

The following paths are runtime artifacts and should not be committed:

以下路径属于运行产物，不应提交：

```text
data/
checkpoints/
outputs/
*.pt
*.pth
*.ckpt
```

## Documentation / 文档

- [Usage Guide / 使用指南](docs/USAGE.md)
- [Architecture / 架构说明](docs/ARCHITECTURE.md)
- [Experiment Report / 实验报告](docs/EXPERIMENT_REPORT.md)
- [Troubleshooting / 故障排查](docs/TROUBLESHOOTING.md)
- [Toy Demo / 玩具演示](examples/toy_vqa_demo/README.md)
- [Model Card / 模型说明](MODEL_CARD.md)
- [Changelog / 变更记录](CHANGELOG.md)
- [Contributing / 贡献指南](CONTRIBUTING.md)
- [Security Policy / 安全政策](SECURITY.md)
- [Citation / 引用元数据](CITATION.cff)

## Development / 开发

Run tests / 运行测试：

```bash
python -m pytest -q
```

Run lint and package checks / 运行代码与打包检查：

```bash
python -m ruff check .
python -m build
```

Run syntax checks / 语法检查：

```bash
python -m compileall train.py evaluate.py infer.py demo.py vqa_project
```

## License / 许可证

This project is released under the MIT License. See [LICENSE](LICENSE).

本项目采用 MIT License 开源，详见 [LICENSE](LICENSE)。

## Notes / 说明

This repository implements a compact VQA classification pipeline for practical training, controlled evaluation, and local demonstration. Toy reports are not benchmarks. The runtime VQA score follows soft target credit from VQA annotations, while exact parity with the official evaluation toolkit still depends on answer normalization and submission protocol. Full training requires large datasets and suitable GPU resources. It is not a reproduction of large-scale vision-language models such as BLIP, Flamingo, or LLaVA.

本工程实现的是一个面向实际训练、受控评估和本地演示的紧凑型 VQA 分类管线。玩具报告不是 benchmark。运行时 VQA score 使用 VQA 标注中的软标签得分，但与官方评估工具完全一致仍取决于答案归一化和提交协议。全量训练需要大规模数据集和合适的 GPU 资源。本工程并非 BLIP、Flamingo、LLaVA 等大规模视觉语言模型的复现。
