# Architecture / 架构说明

This project implements a classification-based Visual Question Answering pipeline.

本工程实现基于答案分类的视觉问答管线。

## Pipeline / 处理流程

```text
Image + Question
      |
      v
Variant-specific image processor + tokenizer
      |
      v
ResNet-50 + DistilBERT, joint ViLT, or BEiT-3 Base encoder
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
- `strong_cross_attention`: gated attention pooling, residual normalization, and a deeper fusion classifier / 门控注意力池化、残差归一化与更深融合分类头
- `vilt`: jointly pretrained patch-and-text transformer with a Top-K VQA classifier / 联合预训练的图像 patch 与文本 Transformer，加 Top-K VQA 分类器
- `beit3_base`: optional BEiT-3 Base encoder with the upstream VQAv2 pooling/classification head / 可选 BEiT-3 Base 编码器与上游 VQAv2 池化/分类头

### ViLT Path / ViLT 路径

The ViLT variant uses `dandelin/vilt-b32-mlm-itm` as a generic image-text pretrained backbone and initializes a new Top-3000 answer classifier. The processor supplies pixel tensors, masks, token IDs, and attention masks. Gradient checkpointing and differential learning rates make full-backbone fine-tuning practical on Kaggle GPUs.

ViLT 变体使用 `dandelin/vilt-b32-mlm-itm` 作为通用图文预训练 backbone，并新建 Top-3000 答案分类器。processor 提供图像张量、图像 mask、token ID 与 attention mask；gradient checkpointing 和差分学习率用于在 Kaggle GPU 上完成全 backbone 微调。

### BEiT-3 Path / BEiT-3 路径

`vqa_project/beit3.py` implements the optional BEiT-3 Base path. It loads the generic 224 px pretraining checkpoint, verifies the pinned file size and SHA-256, checks complete encoder coverage, and interpolates visual position embeddings in FP32 for the 480 px VQA input. The official SentencePiece model is also checksum-verified. The input pipeline uses the upstream square bicubic resize and `0.5/0.5/0.5` channel normalization; no random crop, flip, or color augmentation is enabled. The head follows the upstream VQA pooler and two-layer classifier. The `mask_token` parameter is permanently frozen because VQA inference does not consume it, which keeps DDP's parameter graph static.

`vqa_project/beit3.py` 实现可选的 BEiT-3 Base 路径。它加载通用 224 px 预训练权重，校验固定文件大小和 SHA-256，并检查 encoder 参数完整覆盖；480 px VQA 输入所需的位置编码在 FP32 下插值。官方 SentencePiece 文件同样执行校验。图像输入采用上游的方形 bicubic resize 和 `0.5/0.5/0.5` 通道归一化，不启用随机裁剪、翻转或颜色增强。分类头遵循上游 VQA pooler 和两层分类结构。VQA 前向不使用 `mask_token`，因此该参数永久冻结，确保 DDP 参数图保持静态。

BEiT-3 has a dedicated tokenizer and official VQA answer normalization/leave-one-annotator scoring. Its Top-5000 vocabulary is built from VQA training annotations only. Checkpoint metadata stores the tokenizer and data fingerprints; model construction for checkpoint inference and resume does not redownload the large initialization weights. This path is experimental and does not replace the recommended ViLT release until official full-validation gates pass.

BEiT-3 使用专用 tokenizer 和官方 VQA 答案规范化/留一标注者计分，Top-5000 词表仅由 VQA 训练标注构建。checkpoint 元数据记录 tokenizer 与数据指纹；已有 checkpoint 的推理和续训不再重复下载大型初始化权重。此路径仍属实验候选，官方全量验证门槛通过前不会替换推荐 ViLT 发布版本。

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

VQA annotations often contain multiple human answers per question. Legacy model variants retain the historical count-based soft target:

```text
target_score = min(answer_count / 3, 1)
```

VQA 标注通常包含同一问题的多个标注答案；历史模型变体保留基于答案计数的软标签：

```text
target_score = min(answer_count / 3, 1)
```

BEiT-3 instead uses the exact leave-one-annotator-out score after applying the VQA-v2 evaluator's normalization rule. The training loss is summed across answer classes and normalized by batch size. This per-example `BCEWithLogitsLoss` scale remains comparable when the answer vocabulary is fixed and avoids diluting gradients across thousands of answer classes.

BEiT-3 则采用 VQA-v2 官方评测器的规范化规则，计算精确的留一标注者得分。训练损失先对答案类别求和，再按 batch size 归一化。这种按样本定义的 `BCEWithLogitsLoss` 在答案词表固定时便于比较，也避免在数千个答案类别上稀释梯度。

Evaluation reports hard Top-1 accuracy and soft VQA credit. Legacy models retain their historical target/scoring semantics. The BEiT-3 path uses the VQA-v2 answer normalizer and leave-one-annotator-out consensus score so its supervision matches the official metric protocol.

评估同时输出硬标签 Top-1 准确率和软标签 VQA 得分。历史模型保留原有目标与计分语义；BEiT-3 使用 VQA-v2 答案规范化和留一标注者共识得分，使监督与官方评测协议一致。

Dataset instances store only sparse answer indices and soft scores. The collator materializes dense answer targets per batch, reducing Python-process memory pressure without changing the model input contract.

Dataset 样本仅保存稀疏答案索引与软分数，collator 在每个 batch 内构造稠密目标，从而降低 Python 进程内存压力，同时保持模型输入接口不变。

## Fine-Tuning Strategy / 微调策略

The optional staged schedule keeps both pretrained backbones frozen initially, then unfreezes the final ResNet residual block and the last DistilBERT transformer layers. The classification/fusion head, image backbone, and text backbone use separate learning rates. Gradient accumulation increases the effective batch size without increasing per-step memory.

可选的分阶段方案先冻结两个预训练 backbone，随后解冻 ResNet 最后一个残差块和 DistilBERT 最后若干 Transformer 层。分类/融合头、图像 backbone 和文本 backbone 使用不同学习率；梯度累积在不增加单步显存的情况下提高有效 batch size。

## Checkpoints / 模型权重

`train.py` saves the best checkpoint to:

`train.py` 将最佳权重保存到：

```text
checkpoints/best.pt
```

Checkpoint format version 4 includes model weights, optimizer and scheduler states, AMP scaler state, per-rank RNG states, training history/state, epoch and in-epoch cursor, metrics, full config, answer vocabulary, data fingerprint, and runtime metadata. The engine continues to load earlier formats. `latest.pt` supports update-level BEiT-3 continuation, while `best.pt` tracks the configured official validation metric.

checkpoint 格式版本 4 包含模型权重、优化器与调度器状态、AMP scaler 状态、各 rank 随机状态、训练历史/状态、epoch 内游标、指标、完整配置、答案词表、数据指纹和运行元数据。engine 仍可加载早期格式。`latest.pt` 支持 BEiT-3 按 update 续训，`best.pt` 追踪所配置的官方验证指标。

Each run also writes `training_history.csv`, `training_curves.png`, and `run_metadata.json` beside the checkpoint.

每次训练还会在 checkpoint 同级目录写入 `training_history.csv`、`training_curves.png` 和 `run_metadata.json`。

## Deployment Surface / 部署入口

The project exposes four runtime surfaces:

工程提供四个运行入口：

- `infer.py`: command-line single-image inference / 命令行单图推理
- `evaluate.py`: validation evaluation / 验证集评估
- `demo.py`: Gradio web demo / Gradio Web 演示
- `notebooks/kaggle_train.ipynb`: reproducible Kaggle training and artifact export / 可复现 Kaggle 训练与产物导出
