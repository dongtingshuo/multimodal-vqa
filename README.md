# Multimodal Visual Question Answering (VQA)

基于多模态融合的视觉问答系统。工程提供完整的数据准备、模型训练、评估、命令行推理和 Gradio Web 演示流程，默认面向 VQA v2.0 与 COCO 2014 图像数据。

A production-style multimodal Visual Question Answering system with data preparation, model training, evaluation, command-line inference, and a Gradio web demo. The default workflow targets VQA v2.0 with COCO 2014 images.

## Highlights / 项目特性

- **Multimodal architecture / 多模态架构**: ResNet-50 image encoder, DistilBERT text encoder, bidirectional cross attention, and answer classification head.
- **Full VQA workflow / 完整 VQA 流程**: data preparation, vocabulary construction, training, validation, checkpointing, inference, and web demo.
- **GPU-first training / GPU 优先训练**: CUDA, AMP mixed precision, DataLoader workers, pinned memory, and persistent workers are enabled in the default config.
- **Reproducible configuration / 可复现实验配置**: all runtime settings are centralized in YAML config files.
- **Operational demo / 可运行演示**: Gradio interface supports image upload, question input, and Top-k answer display.
- **Robust local runtime / 稳定本地运行**: demo handles missing checkpoints, occupied ports, CUDA fallback, and Hugging Face cache behavior with clear messages.

## Repository Layout / 工程结构

```text
.
├── configs/
│   ├── default.yaml          # Full GPU training config / 全量 GPU 训练配置
│   └── demo_cpu.yaml         # Lightweight smoke-test config / 轻量级 CPU 测试配置
├── scripts/
│   └── prepare_vqa_data.py   # VQA v2.0 and COCO data preparation / 数据准备脚本
├── tests/
│   └── test_project_structure.py
├── vqa_project/
│   ├── answers.py            # Answer normalization and vocabulary / 答案规范化与词表
│   ├── config.py             # YAML config loading and device resolution / 配置加载与设备解析
│   ├── data.py               # Dataset and collator / 数据集与批处理
│   ├── engine.py             # Training, evaluation, checkpoints / 训练、评估与权重保存
│   ├── hf.py                 # Hugging Face model helpers / Hugging Face 加载辅助
│   ├── inference.py          # Single-image inference / 单图推理
│   └── model.py              # Multimodal VQA model / 多模态模型
├── train.py                  # Training entrypoint / 训练入口
├── evaluate.py               # Evaluation entrypoint / 评估入口
├── infer.py                  # CLI inference entrypoint / 命令行推理入口
├── demo.py                   # Gradio web demo / Web 演示入口
├── requirements.txt
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

For a fast smoke test / 快速连通性测试：

```bash
python train.py --config configs/demo_cpu.yaml
```

## Evaluation / 模型评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

The current evaluation reports validation loss and a simplified Top-1 answer accuracy.

当前评估输出验证集 loss 和简化 Top-1 答案准确率。

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
- [Troubleshooting / 故障排查](docs/TROUBLESHOOTING.md)
- [Model Card / 模型说明](MODEL_CARD.md)
- [Changelog / 变更记录](CHANGELOG.md)

## Development / 开发

Run tests / 运行测试：

```bash
python -m pytest -q
```

Run syntax checks / 语法检查：

```bash
python -m compileall train.py evaluate.py infer.py demo.py vqa_project
```

## License / 许可证

This project is released under the MIT License. See [LICENSE](LICENSE).

本项目采用 MIT License 开源，详见 [LICENSE](LICENSE)。

## Notes / 说明

This repository implements a compact VQA classification pipeline for practical training and deployment. It is not a reproduction of large-scale vision-language models such as BLIP, Flamingo, or LLaVA.

本工程实现的是一个可训练、可评估、可部署的紧凑型 VQA 分类管线，并非 BLIP、Flamingo、LLaVA 等大规模视觉语言模型的复现。
