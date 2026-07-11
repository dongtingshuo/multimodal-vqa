# Model Card / 模型说明

This document describes the trained checkpoint used by this repository.

本文档描述本仓库配套使用的已训练模型权重。

This card covers the published `v0.1.0` checkpoint, two completed Kaggle cross-attention experiments, and the active ViLT continuation candidate. Only `v0.1.0` is a public Release asset. Internal candidates are not promoted until their training, prediction export, official VQA evaluation, and release gate are complete.

本文档覆盖已发布的 `v0.1.0` 权重、两个已完成的 Kaggle cross-attention 实验，以及正在续训的 ViLT 候选模型。只有 `v0.1.0` 是公开 Release 附件；内部候选必须完成训练、预测导出、官方 VQA 评估和发布门槛后才能晋升。

The ViLT route has completed two full epochs and currently leads the internal metrics. Because the configured 10-epoch run was interrupted by Kaggle's runtime limit, the result is explicitly reported as partial and resumable rather than complete.

ViLT 路线已完成两个完整 epoch，目前内部指标领先。由于配置的 10 epoch 任务受 Kaggle 最长运行时间中断，该结果明确标记为“部分完成、可续训”，而不是完整训练结果。

## Active Candidate Protocol / 当前候选协议

- **Architecture / 架构**: generic image-text pretrained `dandelin/vilt-b32-mlm-itm` plus a new Top-3000 VQA classifier
- **Config / 配置**: `configs/kaggle_vilt.yaml`
- **Data gate / 数据门槛**: 443,757 train questions and 214,354 validation questions; missing mirror images are repaired, not filtered
- **Internal targets / 内部目标**: hard accuracy `>= 0.55`, VQA score `>= 0.65`
- **Tracking / 追踪**: local CSV, PNG, JSON, and format-v3 checkpoints; W&B is optional and disabled in the maintained Kaggle runner
- **Current state / 当前状态**: epoch 2 completed; `latest.pt` resumes at epoch 3
- **Promotion / 晋升**: retain the staged candidate as the recommended completed model until the ViLT run and official evaluation finish

The source checkpoint is intentionally not VQAv2-fine-tuned. This keeps comparison claims separate from task-specific checkpoint transfer.

源 checkpoint 明确不使用 VQAv2 微调权重，以避免将任务特定 checkpoint 迁移误写成当前工程训练策略带来的提升。

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

### Strong Cross-Attention Ablation / 强化交叉注意力消融实验

The completed 24-epoch strong-model checkpoint is retained at:

24 epoch 强化模型实验权重保留于：

```text
checkpoints/kaggle_strong_best.pt
```

Checkpoint size / 权重大小: `811,846,462` bytes

SHA256:

```text
07d8bfb2c4de7e1f19a16c70b48b8043a78959498e8407d6f48486482f84aa7e
```

```text
epoch: 22
val_loss: 34.009508
val_acc: 0.496743
val_vqa_score: 0.595518
val_top5_vqa_score: 0.827068
evaluated_examples: 209410
exported_predictions: 214124
```

This ablation underperformed the staged fine-tuning candidate and is therefore not the
recommended checkpoint. It remains available for reproducibility and architecture comparison.

该消融实验的表现低于分阶段微调候选模型，因此不作为推荐 checkpoint，
仅用于可复现性与模型架构对比。

### ViLT Continuation Candidate / ViLT 续训候选模型

The current private Kaggle checkpoint is a format-v3 continuation artifact from the seed-42 run. It completed two epochs before the Kaggle session ended. Both `best.pt` and `latest.pt` record epoch 2; `latest.pt` also preserves optimizer, scheduler, AMP scaler, RNG, history, and global-step state for continuation from epoch 3.

当前私有 Kaggle checkpoint 来自 seed 42 的 format-v3 续训任务。Kaggle 会话结束前完成了两个 epoch；`best.pt` 与 `latest.pt` 均记录 epoch 2，其中 `latest.pt` 还保存优化器、调度器、AMP scaler、随机状态、历史和 global step，可从 epoch 3 继续。

| Epoch | Validation loss | Hard accuracy | VQA score | Top-5 VQA score |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.8765 | 0.5395 | 0.6368 | 0.8519 |
| 2 | **3.4327** | **0.5891** | **0.6879** | **0.8832** |

These are project-internal validation metrics from a partial run. They pass the internal `0.55 / 0.65` gates, but they are not official VQA toolkit scores and must not be compared as a finished benchmark until the remaining epochs, prediction export, and official evaluation complete.

以上是部分训练任务的项目内部验证指标，已通过 `0.55 / 0.65` 内部门槛，但并非官方 VQA toolkit 分数。在剩余 epoch、预测导出和官方评估完成前，不应将其作为完整 benchmark 结果比较。

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
