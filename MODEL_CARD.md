# Model Card / 模型说明

## Overview / 概览

`multimodal-vqa` v0.3.0 provides a ViLT-based Visual Question Answering classifier trained on
VQA v2 with COCO 2014 images. It is the repository's recommended engineering checkpoint.

`multimodal-vqa` v0.3.0 提供基于 ViLT 的视觉问答分类模型，使用 VQA v2 与 COCO 2014
图像训练，是本仓库当前推荐的工程 checkpoint。

- **Task / 任务**: open-ended Visual Question Answering
- **Release / 发布**: [v0.3.0](https://github.com/dongtingshuo/multimodal-vqa/releases/tag/v0.3.0)
- **Checkpoint / 权重**: `best.pt`
- **Architecture / 架构**: ViLT joint vision-language transformer + Top-3000 classifier
- **Source backbone / 源模型**: `dandelin/vilt-b32-mlm-itm`
- **Processor / 处理器**: `dandelin/vilt-b32-finetuned-vqa`
- **Checkpoint format / 格式**: format v3
- **Size / 大小**: 1,409,005,580 bytes
- **SHA256**: `0cce251f02a7b5349b90c0a6e41850168cc01a700f36fe03663887bae7dbf213`

## Intended Use / 预期用途

The checkpoint is intended for English VQA research, reproducible internal evaluation,
single-image inference, and the included Gradio demonstration. It predicts answers from a fixed
Top-3000 normalized vocabulary. It is not a general-purpose generative VLM and is not intended
for safety-critical decisions.

该权重适用于英文 VQA 研究、可复现内部评估、单图推理和仓库内 Gradio 演示。模型从固定的
Top-3000 规范化答案词表中预测，不是通用生成式视觉语言模型，也不适用于安全关键决策。

## Training Data / 训练数据

- VQA v2 train/validation questions and annotations
- COCO 2014 `train2014` and `val2014` images
- 443,757 training questions and 214,354 validation questions passed strict data validation
- 434,635 training examples and 209,608 validation examples were internally scorable under the
  Top-3000 answer vocabulary

- VQA v2 训练/验证 questions 与 annotations
- COCO 2014 `train2014` 与 `val2014` 图像
- 443,757 条训练问题和 214,354 条验证问题通过严格数据校验
- Top-3000 答案词表下可参与内部评分的训练/验证样本分别为 434,635 / 209,608

## Training Configuration / 训练配置

```yaml
seed: 42
model:
  name: vilt
  pretrained_model_name: dandelin/vilt-b32-mlm-itm
  gradient_checkpointing: true
  trainable_layers: 12
data:
  processor_name: dandelin/vilt-b32-finetuned-vqa
  image_size: 384
  max_question_length: 40
  answer_vocab_size: 3000
train:
  batch_size: 4
  gradient_accumulation_steps: 8
  lr: 0.0001
  backbone_lr: 0.00002
  weight_decay: 0.01
  scheduler: warmup_plateau
  early_stopping_patience: 2
  selection_metric: vqa_score
  use_amp: true
```

Epochs 1-2 ran on a Kaggle P100. The format-v3 checkpoint was then continued on one AutoDL
RTX 4090D through epoch 7, where the configured early-stopping rule ended training. Optimizer,
scheduler, AMP scaler, RNG, history, and global-step state were restored at the epoch boundary.

epoch 1-2 在 Kaggle P100 上运行；随后通过 format-v3 checkpoint 在单卡 AutoDL RTX 4090D
上续训至 epoch 7，并按配置触发早停。续训在 epoch 边界恢复 optimizer、scheduler、AMP
scaler、随机数、历史和 global step 状态。

## Validation Results / 验证结果

| Epoch | Train VQA | Val loss | Hard accuracy | Internal VQA | Top-5 VQA | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 0.4591 | 3.8765 | 0.5395 | 0.6368 | 0.8519 | yes |
| 2 | 0.6949 | 3.4327 | 0.5891 | 0.6879 | 0.8832 | yes |
| 3 | 0.7629 | 3.3596 | 0.6050 | 0.7027 | 0.8888 | yes |
| 4 | 0.8090 | 3.3900 | 0.6103 | 0.7079 | 0.8899 | yes |
| **5** | **0.8427** | **3.5019** | **0.6126** | **0.7101** | **0.8908** | **yes** |
| 6 | 0.8672 | 3.6097 | 0.6126 | 0.7099 | 0.8895 | no |
| 7 | 0.8843 | 3.6601 | 0.6120 | 0.7089 | 0.8889 | no |

Epoch 5 is selected by validation VQA score. Epochs 6-7 increased training performance while
validation VQA score and loss worsened, providing direct evidence of overfitting and justifying
the early stop.

模型按验证 VQA score 选择 epoch 5。epoch 6-7 的训练指标继续提高，但验证 VQA score 与
loss 变差，直接表明过拟合，因此早停是合理的。

The final prediction export contains 214,354 unique records. Their question IDs exactly match
the complete official validation questions and annotations. A separate full project evaluation
of the selected checkpoint reported hard accuracy `0.6127`, internal VQA score `0.7102`, Top-5
VQA score `0.8908`, and loss `3.5019`.

最终预测文件包含 214,354 条唯一记录，question ID 与完整官方验证 questions 和 annotations
完全一致。对最佳权重的独立项目全量复评结果为：硬准确率 `0.6127`、内部 VQA score
`0.7102`、Top-5 VQA score `0.8908`、loss `3.5019`。

![Training curves](docs/assets/vilt-seed42-training-curves.png)

## Evaluation Scope / 评估范围

The official VQA Evaluation Toolkit reports `68.42` overall on all 214,354 VQA v2 validation
questions, with `85.88` yes/no, `60.40` other, and `48.54` number accuracy. The evaluation uses
official repository commit `a013f004` and the public validation annotations. Full provenance and
input hashes are recorded in
[`docs/evaluation/v0.3.0-official-vqa.md`](docs/evaluation/v0.3.0-official-vqa.md).

官方 VQA Evaluation Toolkit 在全部 214,354 条 VQA v2 验证问题上的 overall 为 `68.42`，
其中 yes/no `85.88`、other `60.40`、number `48.54`。评测使用官方仓库 commit `a013f004`
和公开验证标注；完整溯源与输入哈希见
[`docs/evaluation/v0.3.0-official-vqa.md`](docs/evaluation/v0.3.0-official-vqa.md)。

This is an official-protocol local validation result, not a test-dev/test-standard server
submission. No leaderboard or cross-project state-of-the-art claim is made.

这是采用官方协议的本地验证结果，不是 test-dev/test-standard 服务器提交成绩；不作
leaderboard 或跨项目 SOTA 声明。

## Comparison / 对比

| Checkpoint | Hard accuracy | Internal VQA | Official VQA val (%) | Status |
| --- | ---: | ---: | ---: | --- |
| ViLT seed 42 (`v0.3.0`) | **0.6126** | **0.7101** | **68.42** | Recommended engineering checkpoint |
| Staged cross-attention (`v0.2.0`) | 0.5239 | 0.6233 | not evaluated | Archived comparison release |
| Strong cross-attention ablation | 0.4967 | 0.5955 | not evaluated | Not promoted |
| Legacy checkpoint (`v0.1.0`) | 0.4775 | not recorded | not evaluated | Archived |

## Experimental Candidate: BEiT-3 Base

This section describes an implemented but not yet trained candidate. It has no validation result, release checkpoint, or promotion claim. The ViLT `v0.3.0` checkpoint remains the recommended model.

本节说明已接入工程、但尚未完成训练的候选模型。当前没有验证指标、发布权重或晋升结论；ViLT `v0.3.0` 仍是推荐模型。

| Property | Value |
| --- | --- |
| Model ID | `beit3_base` |
| Architecture | BEiT-3 Base, 12 encoder layers, hidden size 768, VQA pooling/classification head |
| Pretraining | Generic BEiT-3 Base pretraining; no VQA-finetuned or in-domain checkpoint |
| Image input | 480 x 480, official BEiT-3 VQA resize/bicubic and normalization path |
| Text tokenizer | Official `beit3.spm` SentencePiece model |
| Answer space | Top-5000 built from VQA v2 training annotations only, VQA-v2 answer normalization |
| Upstream implementation license | MIT, per the [Microsoft UniLM repository](https://github.com/microsoft/unilm) |
| TorchScale source | [`4d1e0e82e5adf86dd424f1463192635b73fc8efc`](https://github.com/microsoft/torchscale/tree/4d1e0e82e5adf86dd424f1463192635b73fc8efc) |
| Pretrained weights | [`beit3_base_patch16_224.pth`](https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_224.pth), 554,351,015 bytes |
| Weights SHA-256 | `12ebc3d487f418d4f0f000d09719edfe7dcc89079da427f5580a7fb86ee3943a` |
| Tokenizer SHA-256 | `6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4` |
| Training plan | A: VQA v2 only; B: screened Visual Genome QA; C: independent seed-123 replication |
| Status | Implemented; real pretrained load tested locally; automatic T4 micro-batch config and runtime estimate added; actual dual-T4 smoke, full training, and official evaluation pending |

Upstream VQAv2 preprocessing, answer normalization, and the classification-head layout are adapted from Microsoft's [BEiT-3 VQAv2 guide](https://github.com/microsoft/unilm/blob/master/beit3/get_started/get_started_for_vqav2.md) and [official implementation](https://github.com/microsoft/unilm/tree/master/beit3). The base checkpoint is downloaded from the upstream-linked release location and rejected if its expected size or SHA-256 does not match. Kaggle preflight benchmarks per-GPU batches 1/2/4/8, writes a runtime config preserving global batch 128, performs a real distributed checkpoint round-trip, and records a cautious throughput-based epoch/session estimate. The local integration test loaded all 12 encoder layers and verified the 480 px positional-embedding interpolation; these are implementation checks, not training results.

上游 VQAv2 图像预处理、答案规范化和分类头结构参考 Microsoft 的 [BEiT-3 VQAv2 指南](https://github.com/microsoft/unilm/blob/master/beit3/get_started/get_started_for_vqav2.md)及[官方实现](https://github.com/microsoft/unilm/tree/master/beit3)。Base 权重从上游指向的发布地址下载，文件大小或 SHA-256 不匹配时会拒绝加载。Kaggle 预检会测量每卡 batch 1/2/4/8、生成保持全局 batch 128 的运行配置、执行真实分布式 checkpoint 往返验证，并按吞吐记录审慎的 epoch/会话估算。本地集成测试已加载全部 12 层 encoder，并检查 480 px 位置编码插值；这些都只是实现检查，不代表训练结果。

The predeclared acceptance metric is official VQA validation overall score on all 214,354 questions. Internal metrics and partial exports cannot substitute for this gate. A candidate is promoted only after complete prediction-ID matching, repeated official evaluation, and an independent seed replication; until then, none of these configurations replaces the existing release.

预先声明的验收指标为全部 214,354 个问题上的官方 VQA validation overall。内部指标和部分预测不能替代此门槛。候选只有在预测 ID 完整匹配、官方评测可重复，并完成独立 seed 复现后才可晋升；此前任何 BEiT-3 配置都不会替换现有发布版本。

## Runtime Requirements / 运行要求

The release stores trained model parameters and checkpoint-owned preprocessing configuration.
Runtime construction also requires the Hugging Face ViLT backbone and processor assets. Cache
them once in an online environment before using `demo.py --offline`.

发布权重保存训练参数和 checkpoint 自带的预处理配置。运行时还需要 Hugging Face ViLT
backbone 与 processor 资源；首次联网缓存后，才可使用 `demo.py --offline`。

```bash
python scripts/download_checkpoint.py
python demo.py --checkpoint checkpoints/best.pt
```

## Limitations / 局限

- Fixed Top-3000 answer vocabulary cannot represent every valid answer.
- The default language path expects English questions.
- COCO/VQA domain bias and annotation bias remain present.
- Confidence values are classifier scores and are not calibrated guarantees.
- The `68.42` score is validation-set evaluation, not a held-out evaluation-server result.

- 固定 Top-3000 答案词表无法覆盖所有有效答案。
- 默认语言路径面向英文问题。
- 仍存在 COCO/VQA 数据域与标注偏差。
- 置信度是分类器分数，不是经过校准的保证。
- `68.42` 是验证集评测结果，不是隐藏测试集评测服务器成绩。

## Security / 安全

PyTorch checkpoints use pickle-based serialization. Download only from the official project
Release and verify the published SHA256 before loading.

PyTorch checkpoint 使用基于 pickle 的序列化。请仅从项目官方 Release 下载，并在加载前校验
发布的 SHA256。
