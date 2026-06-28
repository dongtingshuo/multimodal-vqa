# Experiment Report / 实验报告

## Project Scope / 项目范围

This repository provides a PyTorch-based multimodal Visual Question Answering workflow for studying image features, text features, and cross-modal fusion through data preparation, model training, evaluation, inference, model-variant comparison, and lightweight error analysis.

本仓库提供基于 PyTorch 的多模态视觉问答流程，关注图像特征、文本特征与跨模态融合，覆盖数据准备、模型训练、评估、推理、模型变体对比和轻量错误分析。

The system is designed for reproducible local experimentation on VQA-format data. It should not be interpreted as a large general-purpose vision-language model.

该系统面向 VQA 格式数据上的本地可复现实验，不应被理解为通用大规模视觉语言模型。

## Model Variants / 模型变体

- `text_only`: uses the question encoder and answer classifier only.
- `image_only`: uses the image encoder and answer classifier only.
- `baseline_concat`: concatenates pooled image and text features before classification.
- `cross_attention`: applies bidirectional cross-modal attention before answer classification.

- `text_only`：仅使用问题编码器和答案分类器。
- `image_only`：仅使用图像编码器和答案分类器。
- `baseline_concat`：分类前直接拼接池化后的图像与文本特征。
- `cross_attention`：答案分类前使用双向跨模态注意力。

## Evaluation Metrics / 评估指标

- `loss`: per-example multilabel BCE objective, summed across answer classes and normalized by batch size.
- `accuracy`: hard Top-1 answer accuracy.
- `vqa_score`: soft credit assigned to the highest-ranked answer from VQA annotation targets.
- `top5_vqa_score`: highest soft credit available among the five highest-ranked answers.
- `top-k prediction inspection`: optional review of the highest-probability answer candidates.
- `mock_loss`: cross-entropy from the toy comparison workflow.
- `mock_accuracy`: Top-1 accuracy from the toy comparison workflow.

- `loss`：在答案类别上求和、按 batch size 归一化的每样本多标签 BCE 目标。
- `accuracy`：硬标签 Top-1 答案准确率。
- `vqa_score`：从 VQA 标注软标签中读取最高排名答案的得分。
- `top5_vqa_score`：前五个候选答案中可获得的最高软标签得分。
- `top-k prediction inspection`：可选查看概率最高的候选答案。
- `mock_loss`：玩具对比流程中的交叉熵。
- `mock_accuracy`：玩具对比流程中的 Top-1 准确率。

`vqa_score` follows the soft-credit targets used by VQA training and is used for checkpoint selection. Published benchmark claims must use exported prediction JSON and the official VQA toolkit.

`vqa_score` 遵循 VQA 训练软标签计分方式，用于选择 checkpoint。公开 benchmark 结论必须使用导出的预测 JSON 和官方 VQA toolkit。

## Controlled Training Protocol / 受控训练协议

The current protocol compares a frozen cross-attention baseline against staged partial fine-tuning under the same data split, vocabulary, seed, and effective batch size. A second seed is optional for the selected candidate. See [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md) for exact configs and commands.

当前协议在相同数据划分、词表、随机种子和有效 batch size 下，对比冻结的 cross-attention 基线与分阶段部分微调。可对候选方案增加第二个随机种子。精确配置与命令见 [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)。

The fine-tuned checkpoint is promoted as the recommended release only when its official VQA score exceeds the frozen baseline by at least 1.0 absolute point. Missing official scores produce a blocked release decision rather than an inferred result.

只有当微调 checkpoint 的官方 VQA 分数比冻结基线至少高 1.0 个绝对点时，才会被提升为推荐发布版本。缺少官方分数时，发布决策会标记为阻塞，不会根据内部指标推断结果。

## Comparison Workflow / 对比实验流程

```bash
python scripts/run_model_comparison.py --config configs/demo_comparison.yaml
```

Outputs / 输出：

- `outputs/comparison/comparison_results.json`
- `outputs/comparison/comparison_results.csv`
- `outputs/comparison/comparison_report.md`

The demo comparison uses synthetic tensors and mock backbones, so it validates the reporting workflow without external downloads.

演示对比使用合成张量和 mock backbone，因此无需外部下载即可验证报告流程。

## Error Analysis Workflow / 错误分析流程

```bash
python scripts/run_error_analysis.py --predictions examples/toy_vqa_demo/toy_predictions.jsonl
```

Outputs / 输出：

- `outputs/analysis/error_analysis.json`
- `outputs/analysis/error_analysis.md`

The analyzer groups questions into color, count, object, location, yes/no, and other categories using simple keyword rules.

分析器通过简单关键词规则，将问题划分为颜色、数量、物体、位置、是/否和其他类别。

## Result Interpretation / 结果解释

The demo comparison uses small samples or mock data only to validate the workflow. Full performance evaluation requires real VQA v2.0 / COCO 2014 data. `text_only` and `image_only` help analyze single-modality contributions, while `cross_attention` is used to study cross-modal fusion.

demo comparison 使用小样本或 mock 数据，只用于验证流程。完整性能评估需要真实 VQA v2.0 / COCO 2014 数据。`text_only` 和 `image_only` 用于分析单模态信息贡献，`cross_attention` 用于分析跨模态融合效果。

Toy comparison results are not a real benchmark and should not be used to compare final model quality.

玩具对比结果不是正式 benchmark，不应用于比较最终模型质量。

Validation metrics from full training are more informative than toy metrics, but they still depend on dataset split, vocabulary size, random seed, training schedule, and checkpoint selection.

全量训练得到的验证指标比玩具指标更有参考价值，但仍会受到数据划分、答案词表规模、随机种子、训练方案和 checkpoint 选择的影响。

Error analysis reports are diagnostic tools for finding recurring failure patterns, not a replacement for a complete evaluation protocol.

错误分析报告是发现重复错误模式的诊断工具，不能替代完整评估流程。

## Kaggle Staged Fine-Tuning Run / Kaggle 分阶段微调运行

The Kaggle API run `dongtingshuo/multimodal-vqa-finetune` completed successfully on June 26, 2026. The run used `configs/kaggle_finetune.yaml`, P100-compatible PyTorch wheels, VQA v2 question/annotation files, and the public Kaggle COCO image source `sagnikkayalcse52/coco2014vqa`. Because that public image source is missing a small number of COCO validation images, the kernel filtered VQA samples to those with available local images before training and evaluation.

Kaggle API 任务 `dongtingshuo/multimodal-vqa-finetune` 已于 2026-06-26 成功完成。该运行使用 `configs/kaggle_finetune.yaml`、兼容 P100 的 PyTorch wheel、VQA v2 questions/annotations 文件，以及公开 Kaggle COCO 图片源 `sagnikkayalcse52/coco2014vqa`。由于该公开图片源缺少少量 COCO 验证图片，kernel 在训练和评估前过滤为本地实际存在图片的 VQA 样本。

| Metric | Value |
| --- | ---: |
| Epochs | 12 |
| Best epoch | 12 |
| Train loss | 3.1768 |
| Train hard accuracy | 0.6037 |
| Train VQA score | 0.7064 |
| Train Top-5 VQA score | 0.9074 |
| Validation loss | 4.3283 |
| Validation hard accuracy | 0.5239 |
| Validation VQA score | 0.6233 |
| Validation Top-5 VQA score | 0.8640 |
| Evaluated validation examples | 209,410 |
| Exported validation predictions | 214,124 |
| Total runtime | 28,847 seconds |

Local artifacts copied from Kaggle:

Kaggle 结果已复制到本地：

- `checkpoints/kaggle_finetune_best.pt`
- `checkpoints/kaggle_finetune_latest.pt`
- `checkpoints/kaggle_answer_vocab.json`
- `checkpoints/kaggle_training_history.csv`
- `checkpoints/kaggle_training_curves.png`
- `checkpoints/kaggle_run_metadata.json`

These are internal validation metrics from the project evaluator. They are useful for model selection but are not an official VQA leaderboard score.

以上为项目内部 evaluator 的验证指标，可用于模型选择，但不是官方 VQA leaderboard 分数。

## Kaggle Strong Cross-Attention Run / Kaggle 强化交叉注意力运行

The 24-epoch `strong_cross_attention` experiment completed on June 28, 2026 using
`configs/kaggle_strong.yaml`. Training resumed from epoch-boundary checkpoints across
Kaggle sessions. A separate evaluation-only run exported predictions for all validation
questions and completed the artifact archive.

24 epoch 的 `strong_cross_attention` 实验于 2026-06-28 完成，使用
`configs/kaggle_strong.yaml`。训练通过 epoch 边界 checkpoint 在多个 Kaggle 会话间续训，
并通过独立评估任务导出全部验证问题的预测与完整归档。

| Metric | Value |
| --- | ---: |
| Epochs | 24 |
| Best epoch | 22 |
| Train loss | 170.7379 |
| Train hard accuracy | 0.6868 |
| Train VQA score | 0.7801 |
| Train Top-5 VQA score | 0.8690 |
| Validation loss | 34.0095 |
| Validation hard accuracy | 0.4967 |
| Validation VQA score | 0.5955 |
| Validation Top-5 VQA score | 0.8271 |
| Evaluated validation examples | 209,410 |
| Exported validation predictions | 214,124 |
| Total training runtime | 83,741 seconds |

Local artifacts / 本地产物：

- `checkpoints/kaggle_strong_best.pt`
- `outputs/kaggle_strong_final/run_summary.json`
- `outputs/kaggle_strong_final/training_history.csv`
- `outputs/kaggle_strong_final/training_curves.png`
- `outputs/kaggle_strong_final/val_predictions.json`

This experiment did not exceed the staged fine-tuning candidate (`0.5239` hard accuracy,
`0.6233` VQA score), so it is retained as an ablation result and is not promoted as the
recommended checkpoint.

该实验未超过分阶段微调候选模型（硬准确率 `0.5239`、VQA score `0.6233`），
因此作为消融实验结果保留，不晋升为推荐 checkpoint。

## Current Limitations / 当前限制

- The answer space is a fixed Top-K vocabulary.
- Questions are expected to be English in the default tokenizer setup.
- Official evaluation-toolkit parity is not yet asserted because answer normalization and submission formatting may differ.
- Toy reports are workflow checks, not benchmark evidence.
- Small sample results are unstable and should be treated as smoke-test outputs.
- Pretrained model cache availability may affect local full-model runs.
- Full training requires COCO/VQA data and GPU resources for practical runtime.

- 答案空间是固定 Top-K 词表。
- 默认 tokenizer 设置下，问题建议使用英文。
- 目前不声称与官方评估工具完全一致，因为答案归一化和提交格式可能存在差异。
- 玩具报告是流程检查，不是 benchmark 证据。
- 小样本结果不稳定，应作为连通性检查输出看待。
- 预训练模型缓存可用性可能影响本地完整模型运行。
- 全量训练需要 COCO/VQA 数据和适合的 GPU 资源。

## Future Work / 后续工作

- Add BLIP / ViLT adapters as optional models.
- Add repeated-run experiment summaries with confidence intervals.
- Add richer question taxonomy and manual review fields.
- Add attention visualization.
- Improve answer normalization.
- Add optional remote experiment tracking for full training runs.
- Add additional vision and text backbone choices behind the same model factory.

- 将 BLIP / ViLT adapter 作为可选模型接入。
- 增加多次运行的实验汇总和置信区间。
- 增加更细的问题类型体系和人工复核字段。
- 增加注意力可视化。
- 改进答案归一化。
- 为全量训练增加可选的远程实验追踪。
- 在同一模型工厂下增加更多视觉和文本 backbone 选择。
