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

## Completed Training Versions / 已完成训练版本

Only completed model-training experiments are counted below. Kaggle kernel versions 12, 13,
and 14 belong to the same strong-model experiment: version 12 exposed a resume bug, version 13
completed training, and version 14 completed prediction export and artifact packaging. They are
operational sessions, not independent model results.

下表仅统计已完成的模型训练实验。Kaggle kernel 第 12、13、14 版属于同一次
strong 模型实验：第 12 版暴露续训问题，第 13 版完成训练，第 14 版完成预测导出与归档。
它们是运行会话，不是独立的模型结果。

| Version | Best epoch | Validation hard accuracy | Validation VQA score | Change from previous | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Published legacy checkpoint | 10 | 0.4775 | Not recorded | Reference | Historical baseline |
| Staged `cross_attention` | 12 | **0.5239** | **0.6233** | +0.0464 hard accuracy | Current internal candidate |
| `strong_cross_attention` | 22 | 0.4967 | 0.5955 | -0.0272 hard accuracy, -0.0278 VQA | Not promoted |

| Version | Training config | Training Git commit | Local checkpoint |
| --- | --- | --- | --- |
| Published legacy checkpoint | Embedded legacy config; source path unavailable | Unavailable | `checkpoints/best.pt` |
| Staged `cross_attention` | `configs/kaggle_finetune.yaml` | `e26ba287` | `checkpoints/kaggle_finetune_best.pt` |
| `strong_cross_attention` | `configs/kaggle_strong.yaml` | `36aa22f` (epochs 1-12), `794b037` (resumed epochs 13-24) | `checkpoints/kaggle_strong_best.pt` |

The strong run spans two training commits because a CUDA RNG restoration compatibility fix was
required before resuming. The fix changed checkpoint restoration, not the model, loss, data,
optimizer, or scheduler configuration.

strong 运行横跨两个训练 commit，原因是续训前需要修复 CUDA RNG 恢复兼容性。
该修复仅改变 checkpoint 恢复逻辑，未改变模型、目标函数、数据、优化器或调度器配置。

### Published Legacy Checkpoint / 已发布历史权重

**Outcome.** This run is the historical reference rather than a failed promotion attempt. It
completed 10 epochs and reached `0.4775` hard accuracy. Its checkpoint predates the current
soft VQA metrics and normalized multilabel BCE implementation, so its stored loss and the newer
loss values are not directly comparable.

**结果。** 该运行是历史参考基线，而不是一次晋升失败。它完成 10 epoch，硬准确率为
`0.4775`。该 checkpoint 早于当前 soft VQA 指标和归一化多标签 BCE 实现，
因此其保存的 loss 不能与新版 loss 直接比较。

**Why performance was limited.** The checkpoint configuration kept both pretrained backbones
frozen and trained for a fixed 10-epoch schedule. This restricted adaptation to VQA-specific
visual and linguistic patterns. Because no compatible epoch history or soft VQA score exists,
more specific causal claims would be speculation.

**性能受限原因。** checkpoint 配置始终冻结两个预训练 backbone，并使用固定的 10 epoch
调度，限制了模型对 VQA 特定视觉与语言模式的适配。由于没有可兼容的 epoch 历史和
soft VQA score，不对更细致的原因作无证据推断。

**Decision.** Retain for release compatibility and demonstration; do not use its legacy loss as
evidence when selecting current checkpoints.

**决策。** 保留该权重以保证发布兼容性和演示能力；选择当前 checkpoint 时，不将其历史 loss
作为比较证据。

### Staged Cross-Attention Candidate / 分阶段交叉注意力候选模型

**Outcome.** This run improved hard accuracy by `0.0464` absolute over the legacy checkpoint and
established the best completed internal result: `0.5239` hard accuracy and `0.6233` VQA score.
It is therefore a successful improvement, not a non-improving run.

**结果。** 该运行相比历史权重将硬准确率提高了 `0.0464` 个绝对点，并取得目前已完成实验中
最佳的内部结果：硬准确率 `0.5239`、VQA score `0.6233`。因此它是成功改进，不属于未提升运行。

**Why it did not reach the aspirational `0.55 / 0.65` target.** Validation VQA score improved at
every recorded epoch and epoch 12 was still the best checkpoint, while the train-validation VQA
gap was a moderate `0.0831`. The fixed 12-epoch stop therefore likely left some optimization
headroom. This is a high-confidence observation, but the amount of additional attainable gain
has not been established by a controlled continuation run. The Top-3000 answer vocabulary also
excludes some valid validation answers: 209,410 of 214,124 exported questions were internally
scorable.

**未达到预期 `0.55 / 0.65` 目标的原因。** 验证 VQA score 在所有已记录 epoch 中持续提升，
epoch 12 仍是最佳 checkpoint，此时训练与验证 VQA 差距为中等水平 `0.0831`。
因此，固定在 12 epoch 停止很可能保留了一定优化空间。该现象的证据可信度较高，
但可获得的额外收益尚未通过受控续训实验确定。Top-3000 答案词表也排除了部分有效验证答案：
214,124 个已导出问题中，209,410 个可在项目内部评分。

**Decision.** Keep as the current internal candidate. The next continuation should preserve this
architecture and objective, extend only the epoch budget, and use early stopping with a minimum
improvement threshold.

**决策。** 保留为当前内部候选模型。下一次续训应保持该架构与目标函数不变，仅延长 epoch 预算，
并使用带最小改进阈值的 early stopping。

### Strong Cross-Attention Candidate / 强化交叉注意力候选模型

**Outcome.** Despite completing 24 epochs, the strong model regressed by `0.0272` hard accuracy
and `0.0278` VQA score relative to the staged candidate. Its epoch-12 train VQA score was already
slightly higher than the staged model (`0.7117` versus `0.7064`), but validation VQA was lower
(`0.5905` versus `0.6233`). This rules out simple under-training as the primary explanation.

**结果。** 尽管完成了 24 epoch，strong 模型相比分阶段候选模型的硬准确率下降 `0.0272`，
VQA score 下降 `0.0278`。在 epoch 12，它的训练 VQA score 已略高于分阶段模型
（`0.7117` 对 `0.7064`），但验证 VQA 更低（`0.5905` 对 `0.6233`）。这排除了“单纯训练不足”是主因。

**Verified causes and defects.**

- Sparse-target label smoothing changed every zero target to `0.01`. With 2,999 negative
  classes, the theoretical negative-class BCE floor is approximately `167.95` per example,
  matching the observed loss plateau near `170.73`. The loss was therefore dominated by
  smoothed negatives instead of answer ranking.
- Trainable parameters increased from 34.75M to 52.09M. At the best strong epoch, train VQA
  reached `0.7801` while validation VQA remained `0.5955`, a gap of `0.1846`. From epoch 10 to
  epoch 22, train VQA gained `0.1038` while validation VQA gained only `0.0072`, directly
  demonstrating overfitting.
- The warmup-cosine scheduler did not enforce the configured `min_lr=1e-6`. Logged learning
  rates fell to `6.1e-7` at epoch 22 and `1.2e-12` at epochs 23-24, making the final epochs
  effectively inactive.

**已验证原因与缺陷。**

- 稀疏目标 label smoothing 将所有零目标改为 `0.01`。在 2,999 个负类别下，理论负类 BCE
  下限约为每样本 `167.95`，与观测到的 loss 平台 `170.73` 高度吻合。因此目标函数被平滑负类主导，
  而不是主要优化答案排名。
- 可训练参数从 34.75M 增加到 52.09M。strong 最佳 epoch 的训练 VQA 为 `0.7801`，验证 VQA
  仅为 `0.5955`，差距达 `0.1846`。从 epoch 10 到 epoch 22，训练 VQA 提高 `0.1038`，
  验证 VQA 仅提高 `0.0072`，直接证明明显过拟合。
- warmup-cosine 调度器未实现配置的 `min_lr=1e-6`。记录学习率在 epoch 22 降至 `6.1e-7`，
  epoch 23-24 降至 `1.2e-12`，最后两轮实际上已无有效学习。

**High-confidence contributing factors requiring ablation.** The strong run simultaneously
introduced gated attention pooling, a deeper classifier, more unfrozen backbone layers, stronger
random resized cropping, color jitter, higher head learning rate, a new scheduler, label
smoothing, and a longer schedule. The random crop can remove the object or spatial relation named
by a VQA question while retaining the original answer label. These changes are plausible
contributors, but their individual effects were not isolated and must not be presented as proven
causes.

**需要消融实验的高可信影响因素。** strong 运行同时引入了门控注意力池化、更深分类头、
更多解冻 backbone 层、更强随机缩放裁剪、颜色抖动、更高 head 学习率、新调度器、
label smoothing 和更长调度。随机裁剪可能移除 VQA 问题所指对象或空间关系，但仍保留原答案标签。
这些变化都可能影响性能，但它们的独立效应尚未隔离，不应写成已证实因果。

**Factors ruled out.** The data split, Top-3000 vocabulary size, seed, effective batch size, and
pretrained backbone families were unchanged. Kaggle interruptions are also not a metric cause:
format-v3 checkpoints restored optimizer, scheduler, scaler, RNG, history, and stage state at
epoch boundaries, and the resumed metric curve remained continuous.

**已排除因素。** 数据划分、Top-3000 词表大小、随机种子、有效 batch size 和预训练 backbone
类型均未改变。Kaggle 中断也不是指标下降原因：format-v3 checkpoint 在 epoch 边界恢复了
optimizer、scheduler、scaler、RNG、历史和解冻阶段，且续训后的指标曲线保持连续。

**Decision.** Do not continue this checkpoint for more epochs. Return to the staged
`cross_attention` candidate and test one change per run, starting with an unchanged-objective
continuation. Disable the current label smoothing before any further strong-model experiment and
fix scheduler minimum-LR enforcement first.

**决策。** 不再对该 checkpoint 继续增加 epoch。回到分阶段 `cross_attention` 候选模型，
每次运行只测试一项变更，首先执行不改目标函数的续训。在任何后续 strong 模型实验前，
应禁用当前 label smoothing，并先修复调度器的最小学习率约束。

## Experiment Documentation Policy / 实验记录规范

Every completed full-training run must record the config and Git commit, best epoch, core metrics,
comparison baseline, promotion decision, observed non-improvement evidence, and next controlled
test. Causal statements must be labeled as verified, high-confidence inference, or untested
hypothesis. Failed infrastructure attempts are documented only when they affect reproducibility;
they are not counted as model experiments.

每次完成的全量训练都必须记录配置与 Git commit、最佳 epoch、核心指标、对比基线、
晋升决策、未提升证据和下一个受控实验。因果结论必须标记为已验证、高可信推断或未验证假设。
基础设施失败仅在影响可复现性时记录，不计为模型实验。

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

## Next Controlled ViLT Experiment / 下一项受控 ViLT 实验

The next run changes the model family because the completed evidence shows that adding capacity to
the separate ResNet/DistilBERT fusion head increased overfitting without improving validation.
Run 1 therefore uses the jointly pretrained `dandelin/vilt-b32-mlm-itm` backbone, a fresh Top-3000
classifier, unchanged soft VQA targets, no label smoothing, and complete validation data. It does
not use a VQAv2-fine-tuned source checkpoint.

下一轮更换模型家族，是因为已完成实验表明：继续增加独立 ResNet/DistilBERT 融合头容量会加剧
过拟合，并未改善验证指标。首轮因此采用联合图文预训练的 `dandelin/vilt-b32-mlm-itm`
backbone、新建 Top-3000 分类器、保持 soft VQA target、不使用 label smoothing，并恢复完整验证数据。
源 checkpoint 不使用 VQAv2 微调权重。

The protocol permits at most two full runs. Run 1 is fixed by `configs/kaggle_vilt.yaml`; Run 2 is
generated by `scripts/select_vilt_followup.py` from recorded history and changes exactly one
variable. Local/Kaggle artifacts remain the source of record, and the Kaggle workflow no longer
depends on online W&B tracking.
The existing staged checkpoint remains recommended until ViLT reaches hard accuracy `0.55` and
VQA score `0.65` and completes official evaluation.

协议最多允许两次全量运行。首轮固定使用 `configs/kaggle_vilt.yaml`；第二轮由
`scripts/select_vilt_followup.py` 根据已记录历史生成，并且只改变一个变量。最终记录以
本地/Kaggle 产物为准，Kaggle 流程不再依赖 W&B 在线监控。在 ViLT 达到硬准确率 `0.55`、VQA score `0.65` 且完成
官方评估前，继续推荐现有分阶段 checkpoint。

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

- Evaluate the implemented ViLT route under the fixed two-run protocol.
- Add repeated-run experiment summaries with confidence intervals.
- Add richer question taxonomy and manual review fields.
- Add attention visualization.
- Improve answer normalization.
- Review persisted run artifacts after each full training run.
- Add additional vision and text backbone choices behind the same model factory.

- 按固定的两轮协议评估已实现的 ViLT 路线。
- 增加多次运行的实验汇总和置信区间。
- 增加更细的问题类型体系和人工复核字段。
- 增加注意力可视化。
- 改进答案归一化。
- 每次全量训练后审阅持久化 run 产物。
- 在同一模型工厂下增加更多视觉和文本 backbone 选择。
