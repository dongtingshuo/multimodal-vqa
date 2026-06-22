# Architecture / 架构说明

This project implements a classification-based Visual Question Answering pipeline.

本工程实现基于答案分类的视觉问答管线。

## Pipeline / 处理流程

```text
Image + Question
      |
      v
COCO image transform + tokenizer
      |
      v
ResNet-50 image encoder + DistilBERT text encoder
      |
      v
Linear projections to shared hidden dimension
      |
      v
Variant-specific fusion
      |
      v
Answer classifier over Top-K answer vocabulary
      |
      v
Top-k answer probabilities
```

## Components / 组件

### Image Encoder / 图像编码器

`vqa_project/model.py` uses a pretrained ResNet-50 backbone and removes the final pooling/classification layers. The remaining convolutional feature map is flattened into visual tokens.

`vqa_project/model.py` 使用预训练 ResNet-50，并移除最后的池化和分类层。卷积特征图会被展开为视觉 token。

### Text Encoder / 文本编码器

The question is tokenized with a Hugging Face tokenizer and encoded by DistilBERT.

问题文本通过 Hugging Face tokenizer 分词，并由 DistilBERT 编码。

### Model Variants / 模型变体

`vqa_project/model.py` exposes a shared `build_model(config)` factory. If `model.name` is omitted, the default variant is `cross_attention`.

`vqa_project/model.py` 提供统一的 `build_model(config)` 工厂。如果省略 `model.name`，默认变体为 `cross_attention`。

- `text_only`: text encoder plus classifier / 文本编码器加分类器
- `image_only`: image encoder plus classifier / 图像编码器加分类器
- `baseline_concat`: pooled image and text features concatenated before classification / 池化图像与文本特征拼接后分类
- `cross_attention`: bidirectional cross-modal attention before classification / 分类前进行双向跨模态注意力融合

### Cross Attention Fusion / 交叉注意力融合

The fusion module applies two attention directions:

融合模块包含两个注意力方向：

- Text queries image tokens / 文本查询图像 token
- Image queries text tokens / 图像查询文本 token

The fused image and text representations are pooled and concatenated before classification.

融合后的图像和文本表示经过池化与拼接后输入分类头。

### Answer Classifier / 答案分类器

The classifier predicts logits over the Top-K answer vocabulary. The default vocabulary size is 3000.

分类器在 Top-K 答案词表上输出 logits。默认词表大小为 3000。

## Training Objective / 训练目标

VQA annotations often contain multiple human answers per question. The dataset converts answer counts into soft targets:

VQA 标注通常包含同一问题的多个标注答案。数据集会将答案计数转换为软标签：

```text
target_score = min(answer_count / 3, 1)
```

The training loss is summed across answer classes and normalized by batch size. This per-example `BCEWithLogitsLoss` scale remains comparable when the answer vocabulary is fixed and avoids diluting gradients across thousands of answer classes.

训练损失先对答案类别求和，再按 batch size 归一化。这种按样本定义的 `BCEWithLogitsLoss` 在答案词表固定时便于比较，也避免在数千个答案类别上稀释梯度。

Evaluation reports hard Top-1 accuracy and soft VQA credit. For a predicted answer index, `vqa_score` reads the corresponding soft target value; `top5_vqa_score` takes the highest soft credit among the five highest-ranked answers.

评估同时输出硬标签 Top-1 准确率和软标签 VQA 得分。`vqa_score` 取预测答案对应的软标签值；`top5_vqa_score` 取排名前五的答案中最高的软标签得分。

## Checkpoints / 模型权重

`train.py` saves the best checkpoint to:

`train.py` 将最佳权重保存到：

```text
checkpoints/best.pt
```

Checkpoint format version 2 includes model weights, optimizer state, epoch, metrics, full config, answer vocabulary, and runtime metadata. Inference reconstructs architecture and preprocessing from the stored config while preserving local data paths and device settings from the supplied runtime YAML.

checkpoint 格式版本 2 包含模型权重、优化器状态、epoch、指标、完整配置、答案词表和运行元数据。推理时使用内置配置重建模型架构与预处理，同时保留运行时 YAML 中的本地数据路径和设备设置。

Each run also writes `training_history.csv`, `training_curves.png`, and `run_metadata.json` beside the checkpoint.

每次训练还会在 checkpoint 同级目录写入 `training_history.csv`、`training_curves.png` 和 `run_metadata.json`。

## Deployment Surface / 部署入口

The project exposes three runtime surfaces:

工程提供三个运行入口：

- `infer.py`: command-line single-image inference / 命令行单图推理
- `evaluate.py`: validation evaluation / 验证集评估
- `demo.py`: Gradio web demo / Gradio Web 演示
