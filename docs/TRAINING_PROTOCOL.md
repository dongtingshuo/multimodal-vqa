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
| Strong Cross-Attention | `configs/kaggle_strong.yaml` | up to 24 | 42 | 32 |
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

## Strong Training Recipe / 强化训练方案

`configs/kaggle_strong.yaml` is the recommended candidate for the next performance run. It uses `strong_cross_attention`, gated bidirectional attention, attention pooling, lightweight image augmentation, warmup-cosine scheduling, label smoothing, and a longer staged fine-tuning window.

`configs/kaggle_strong.yaml` 是下一轮性能实验的推荐候选配置。它使用 `strong_cross_attention`、门控双向注意力、注意力池化、轻量图像增强、warmup-cosine 学习率调度、label smoothing 和更长的分阶段微调窗口。

```bash
python train.py --config configs/kaggle_strong.yaml --device cuda --wandb
python scripts/summarize_runs.py --runs-dir runs
```

W&B is optional. Use `WANDB_API_KEY` locally or as a Kaggle Secret; never write it into config files, notebooks, or commits.

W&B 是可选能力。本地或 Kaggle Secret 中设置 `WANDB_API_KEY` 即可启用；不要把密钥写入配置文件、notebook 或 commit。

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
