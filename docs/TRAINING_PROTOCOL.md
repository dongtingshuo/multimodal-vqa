# Training Protocol / 训练协议

## Objective / 目标

Evaluate whether staged backbone fine-tuning improves the official VQA validation score over a newly trained frozen Cross-Attention baseline under the same data, effective batch size, augmentation, and evaluation protocol.

在相同数据、有效 batch size、数据增强和评估协议下，验证分阶段 backbone 微调能否相比新训练的冻结 Cross-Attention 基线提升官方 VQA 验证得分。

## Controlled Runs / 对照实验

| Run | Config | Epochs | Seed | Effective Batch |
|---|---|---:|---:|---:|
| Feature concatenation | `configs/baseline_concat.yaml` | 10 | 42 | 32 |
| Frozen Cross-Attention | `configs/baseline_frozen.yaml` | 10 | 42 | 32 |
| Staged fine-tuning | `configs/kaggle_finetune.yaml` | up to 12 | 42 | 32 |
| Fine-tuning repeat | `configs/kaggle_finetune.yaml` | up to 12 | 1337 | 32 |

The repeat run is optional when the available Kaggle GPU quota is insufficient. It must be identified as missing rather than replaced by toy metrics.

当 Kaggle GPU 额度不足时，可省略重复微调实验。报告中应明确标注缺失，不得用玩具指标替代。

## Fine-Tuning Schedule / 微调时序

- Epochs 1–2: projections, Cross-Attention, and classifier only.
- Epochs 3–12: additionally train ResNet `layer4` and the final two DistilBERT transformer layers.
- Head learning rate: `1e-4`; image learning rate: `1e-5`; text learning rate: `5e-6`.
- Physical batch: 16; gradient accumulation: 2; effective batch: 32.
- Early stopping begins at epoch 6 with patience 3.

- Epoch 1–2：仅训练投影层、Cross-Attention 和分类器。
- Epoch 3–12：额外训练 ResNet `layer4` 和 DistilBERT 最后两个 transformer 层。
- 头部学习率 `1e-4`，图像层学习率 `1e-5`，文本层学习率 `5e-6`。
- 物理 batch 16，梯度累积 2，有效 batch 32。
- epoch 6 后启用 patience 3 的 early stopping。

## Evaluation and Release Gate / 评估与发布门槛

Each run must export validation predictions and be evaluated with the official VQA Evaluation Toolkit. Internal hard accuracy, VQA score, and Top-5 VQA score remain diagnostic metrics.

每个正式运行都必须导出验证集预测，并使用 VQA 官方 Evaluation Toolkit 评分。内部 hard accuracy、VQA score 和 Top-5 VQA score 仅作为诊断指标。

The fine-tuned checkpoint becomes the recommended v0.2.0 model only when its official VQA score exceeds the frozen Cross-Attention baseline by at least 1.0 absolute point.

仅当微调 checkpoint 的官方 VQA score 比冻结 Cross-Attention 基线至少高 1.0 个绝对分数点时，才将其作为 v0.2.0 推荐模型。

```bash
python scripts/summarize_experiments.py \
  --run frozen=/path/to/frozen \
  --run finetune=/path/to/finetune \
  --baseline frozen \
  --candidate finetune \
  --minimum-official-gain 1.0
```
