# Model Card / 模型说明

This document describes the trained checkpoint used by this repository.

本文档描述本仓库配套使用的已训练模型权重。

This card currently covers the published `v0.1.0` checkpoint and the locally available Kaggle fine-tuning candidate. The Kaggle candidate has internal validation metrics, but it is not yet promoted as a release checkpoint until the official VQA evaluation and release gate are completed.

本文档当前覆盖已发布的 `v0.1.0` 权重和本地可用的 Kaggle 微调候选权重。Kaggle 候选权重已有内部验证指标，但在完成官方 VQA 评估和发布门槛前，尚未晋升为正式发布权重。

## Model Summary / 模型概要

- **Task / 任务**: Visual Question Answering (VQA)
- **Checkpoint / 权重文件**: `checkpoints/best.pt`
- **Release / 发布页**: [v0.1.0](https://github.com/dongtingshuo/multimodal-vqa/releases/tag/v0.1.0)
- **Checkpoint size / 权重大小**: 427,216,108 bytes
- **SHA256**: `d9638309c2e74a30479332eaabb0f27869555d967eefae0a1eaf981342c3f98c`
- **Model architecture / 模型结构**: ResNet-50 + DistilBERT + bidirectional cross attention + answer classifier
- **Answer vocabulary / 答案词表**: Top-3000 normalized answers
- **Input / 输入**: one RGB image and one English natural-language question
- **Output / 输出**: Top-k answer candidates and probabilities

## Training Configuration / 训练配置

```yaml
device: cuda
seed: 42

data:
  root: data/vqa
  train_split: train
  val_split: val
  answer_vocab_path: data/answer_vocab.json
  answer_vocab_size: 3000
  max_train_samples: null
  max_val_samples: null
  image_size: 224
  max_question_length: 32
  num_workers: 4

model:
  text_model_name: distilbert-base-uncased
  hidden_dim: 512
  num_attention_heads: 8
  dropout: 0.2
  freeze_backbones: true
  pretrained_cnn: true

train:
  batch_size: 32
  epochs: 10
  lr: 0.0001
  weight_decay: 0.01
  grad_clip_norm: 1.0
  use_amp: true
  pin_memory: true
  persistent_workers: true
```

## Training Data / 训练数据

The checkpoint was trained with the VQA v2.0 train split and COCO 2014 image data.

该权重基于 VQA v2.0 训练集和 COCO 2014 图像数据训练。

Expected local data layout:

本地数据目录结构：

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

## Validation Metrics / 验证指标

### Published v0.1.0 Checkpoint / 已发布 v0.1.0 权重

Checkpoint metadata:

权重内记录的验证结果：

```text
epoch: 10
val_loss: 0.001488
val_acc: 0.477510
```

`val_acc` is a simplified Top-1 answer accuracy. It is not the official VQA soft accuracy metric.

`val_acc` 是简化 Top-1 答案准确率，并不是 VQA 官方 soft accuracy 指标。

This legacy checkpoint predates the runtime `vqa_score` and per-example BCE-loss normalization. Its stored loss must not be compared directly with loss values produced by the current training code. Re-evaluation with the current code reports `vqa_score`, `top5_vqa_score`, hard accuracy, and the revised loss scale.

该历史权重生成于当前 `vqa_score` 和按样本归一化 BCE loss 实现之前，其中保存的 loss 不应与新版训练代码输出直接比较。使用当前代码重新评估会输出 `vqa_score`、`top5_vqa_score`、硬标签准确率和新 loss 尺度。

### Kaggle Fine-Tuning Candidate / Kaggle 微调候选权重

Local artifact:

本地文件：

```text
checkpoints/kaggle_finetune_best.pt
```

Checkpoint size / 权重大小: `660,441,108` bytes

SHA256:

```text
15e15b4a0194b073a153331ad2c6b38ee39400d87e489bb6f0fc77d91e7cb22c
```

Internal validation metrics from the completed Kaggle run:

Kaggle 完整运行得到的内部验证指标：

```text
epoch: 12
val_loss: 4.328347
val_acc: 0.523929
val_vqa_score: 0.623305
val_top5_vqa_score: 0.863991
evaluated_examples: 209410
exported_predictions: 214124
```

This candidate was trained with VQA v2 labels and a public Kaggle COCO image source after filtering samples whose images were absent from that image source. The metrics above are project-internal validation metrics, not official VQA toolkit scores.

该候选权重使用 VQA v2 标签和公开 Kaggle COCO 图片源训练，并过滤了该图片源中缺失图片的样本。上述指标是项目内部验证指标，不是官方 VQA toolkit 分数。

## Intended Use / 预期用途

This model is intended for:

该模型适用于：

- local VQA inference experiments / 本地视觉问答推理实验
- controlled validation on VQA-format data / VQA 格式数据的受控验证
- Gradio-based interactive demonstration / 基于 Gradio 的交互式演示
- further fine-tuning and architecture iteration / 后续微调与结构迭代

## Limitations / 局限性

- The model predicts answers from a fixed Top-3000 vocabulary.
- Questions should be written in English.
- The model is not a general-purpose large vision-language model.
- Performance depends on COCO/VQA-style image-question distributions.
- The released checkpoint metadata records simplified Top-1 accuracy only; current evaluation adds VQA soft scores.

- 模型只能从固定 Top-3000 答案词表中预测答案。
- 问题建议使用英文。
- 该模型不是通用大规模视觉语言模型。
- 模型表现依赖 COCO/VQA 风格的图像与问题分布。
- 已发布 checkpoint 的元数据仅记录简化 Top-1 准确率；当前评估代码已增加 VQA soft score。

## Distribution / 权重发布

Do not commit `checkpoints/best.pt` directly to the normal Git repository. The checkpoint is larger than GitHub's regular file size limit.

不要将 `checkpoints/best.pt` 直接提交到普通 Git 仓库。该权重文件超过 GitHub 普通文件大小限制。

The trained checkpoint is published as a verified GitHub Release asset while `checkpoints/` remains ignored by Git.

已训练权重作为可校验的 GitHub Release 附件发布，`checkpoints/` 仍保持 Git 忽略。

```bash
python scripts/download_checkpoint.py
```

[Direct download / 直接下载](https://github.com/dongtingshuo/multimodal-vqa/releases/download/v0.1.0/best.pt)

## Loading / 加载方式

Inference and evaluation use architecture and preprocessing settings stored inside the checkpoint. The supplied YAML continues to control local paths, device selection, and runtime batch settings.

推理与评估优先使用 checkpoint 内保存的模型架构和预处理配置；命令行传入的 YAML 继续控制本地路径、设备和运行时批大小。

```bash
python demo.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

```bash
python infer.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt \
  --image data/vqa/val2014/COCO_val2014_000000000042.jpg \
  --question "What is in the image?" \
  --topk 5
```

For the Kaggle fine-tuned candidate:

使用 Kaggle 微调候选权重：

```bash
python infer.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/kaggle_finetune_best.pt \
  --image data/vqa/val2014/COCO_val2014_000000000042.jpg \
  --question "What is in the image?" \
  --topk 5
```
