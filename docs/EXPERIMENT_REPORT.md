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

- `loss`: training or validation objective value.
- `accuracy`: simplified Top-1 answer accuracy used by the current runtime.
- `top-k prediction inspection`: optional review of the highest-probability answer candidates.
- `mock_loss`: cross-entropy from the toy comparison workflow.
- `mock_accuracy`: Top-1 accuracy from the toy comparison workflow.

- `loss`：训练或验证目标函数值。
- `accuracy`：当前运行流程使用的简化 Top-1 答案准确率。
- `top-k prediction inspection`：可选查看概率最高的候选答案。
- `mock_loss`：玩具对比流程中的交叉熵。
- `mock_accuracy`：玩具对比流程中的 Top-1 准确率。

The simplified Top-1 metric is useful for tracking local experiments, but it is not the official VQA soft accuracy metric.

简化 Top-1 指标适合追踪本地实验，但它不是 VQA 官方 soft accuracy 指标。

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

## Current Limitations / 当前限制

- The answer space is a fixed Top-K vocabulary.
- Questions are expected to be English in the default tokenizer setup.
- The current validation accuracy is simplified Top-1 accuracy, not official VQA scoring.
- Toy reports are workflow checks, not benchmark evidence.
- Small sample results are unstable and should be treated as smoke-test outputs.
- Pretrained model cache availability may affect local full-model runs.
- Full training requires COCO/VQA data and GPU resources for practical runtime.

- 答案空间是固定 Top-K 词表。
- 默认 tokenizer 设置下，问题建议使用英文。
- 当前验证准确率是简化 Top-1 准确率，不是 VQA 官方评分。
- 玩具报告是流程检查，不是 benchmark 证据。
- 小样本结果不稳定，应作为连通性检查输出看待。
- 预训练模型缓存可用性可能影响本地完整模型运行。
- 全量训练需要 COCO/VQA 数据和适合的 GPU 资源。

## Future Work / 后续工作

- Add official VQA soft-accuracy evaluation.
- Add BLIP / ViLT adapters as optional models.
- Add repeated-run experiment summaries with confidence intervals.
- Add richer question taxonomy and manual review fields.
- Add attention visualization.
- Improve answer normalization.
- Add optional experiment tracking for full training runs.
- Add additional vision and text backbone choices behind the same model factory.

- 增加 VQA 官方 soft accuracy 评估。
- 将 BLIP / ViLT adapter 作为可选模型接入。
- 增加多次运行的实验汇总和置信区间。
- 增加更细的问题类型体系和人工复核字段。
- 增加注意力可视化。
- 改进答案归一化。
- 为全量训练增加可选实验追踪。
- 在同一模型工厂下增加更多视觉和文本 backbone 选择。
