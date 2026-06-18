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
Bidirectional cross attention
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

The training loss is `BCEWithLogitsLoss`, which supports these soft multi-answer targets.

训练损失使用 `BCEWithLogitsLoss`，适配这种软标签答案分布。

## Checkpoints / 模型权重

`train.py` saves the best checkpoint to:

`train.py` 将最佳权重保存到：

```text
checkpoints/best.pt
```

The checkpoint includes model weights, optimizer state, epoch, metrics, config, and answer vocabulary.

checkpoint 包含模型权重、优化器状态、epoch、指标、配置和答案词表。

## Deployment Surface / 部署入口

The project exposes three runtime surfaces:

工程提供三个运行入口：

- `infer.py`: command-line single-image inference / 命令行单图推理
- `evaluate.py`: validation evaluation / 验证集评估
- `demo.py`: Gradio web demo / Gradio Web 演示
