# Training Protocol / 训练协议

## Objective / 目标

Test whether generic image-text pretrained ViLT can materially exceed the best completed staged Cross-Attention result without using a VQAv2-fine-tuned source checkpoint. The internal targets are hard accuracy `>= 0.55` and VQA score `>= 0.65`; official VQA evaluation remains the release metric.

验证仅使用通用图文预训练权重的 ViLT，能否显著超过当前最佳的分阶段 Cross-Attention 结果；禁止使用已在 VQAv2 上微调的源 checkpoint。内部目标为硬准确率 `>= 0.55`、VQA score `>= 0.65`，发布判断仍以官方 VQA 评估为准。

## Fixed Run 1 / 固定首轮

### Current execution state / 当前执行状态

The seed-42 run completed through epoch 7 and stopped under the configured early-stopping rule. Epoch 5 is the selected checkpoint with hard accuracy `0.6126` and internal VQA score `0.7101`. The run exported all 214,354 validation predictions and passed both internal gates.

seed 42 任务已完成至 epoch 7，并按配置触发早停。epoch 5 是最佳 checkpoint，硬准确率 `0.6126`、内部 VQA score `0.7101`。任务已导出全部 214,354 条验证预测，并通过两个内部门槛。

This completed result is the recommended engineering checkpoint. Official-toolkit evaluation on the public validation annotations is complete at `68.42` overall. This is not a test-dev/test-standard leaderboard submission. Epochs 6-7 show continued train-score gains while validation loss rises and validation VQA declines, so the next experiment must target generalization rather than extend training or simply increase epochs.

该完整结果作为推荐工程 checkpoint。官方 toolkit 已在公开验证标注上完成评测，overall 为 `68.42`；该结果不是 test-dev/test-standard leaderboard 提交成绩。epoch 6-7 的训练分数继续上升，但验证 loss 增大、验证 VQA 下降，因此下一轮应针对泛化问题，而不是单纯增加训练轮数。

| Item | Value |
| --- | --- |
| Config | `configs/kaggle_vilt.yaml` |
| Pretrained source | `dandelin/vilt-b32-mlm-itm` |
| Input processor | `dandelin/vilt-b32-finetuned-vqa` (tokenizer and image processor only) |
| Seed | 42 |
| Epoch budget | Up to 10 |
| Physical / effective batch | 4 / 32 |
| Backbone / head LR | `2e-5` / `1e-4` |
| Objective | Soft-target multilabel BCE, no label smoothing |
| Scheduler | 5% step warmup, then metric plateau reduction |
| Early stopping | Start epoch 3, patience 2, minimum delta `0.002` |
| Image processing | ViLT processor at 384 px, no semantic crop or color jitter |
| Tracking | Local CSV/PNG/JSON artifacts; W&B optional and disabled in maintained Kaggle runs |

Run 1 trains all ViLT layers with AMP, gradient checkpointing, gradient clipping, and parameter-group learning rates. `latest.pt` is saved every epoch and contains optimizer, scheduler, scaler, RNG, history, and stage state.

首轮使用 AMP、梯度 checkpoint、梯度裁剪和参数组学习率训练全部 ViLT 层。每个 epoch 保存 `latest.pt`，其中包含 optimizer、scheduler、scaler、随机状态、历史记录和训练阶段，可在中断后继续训练。

## Data Contract / 数据约束

The run must use the complete official VQA v2 train and validation question/annotation files with COCO 2014 images. Preflight must report exactly 443,757 train questions, 214,354 validation questions, 82,783 train images, and 40,504 validation images. Missing mirror images are repaired before training; samples are never silently removed.

训练必须使用完整的官方 VQA v2 train/validation questions、annotations 和 COCO 2014 图片。预检必须得到 443,757 条训练问题、214,354 条验证问题、82,783 张训练图和 40,504 张验证图。镜像缺图应在训练前补齐，禁止静默删除样本。

Training-time validation metrics are computed on the 209,608 examples with at least one Top-3000 answer. Prediction export and official evaluation use all 214,354 questions. Keep these metric populations explicit when comparing runs; the official score is the release comparison metric. Full-split answer-space and error diagnostics are in [v0.3.0-vilt-diagnostics.md](evaluation/v0.3.0-vilt-diagnostics.md).

训练阶段的验证指标只统计至少含一个 Top-3000 答案的 209,608 个问题；预测导出和官方评估覆盖全部 214,354 个问题。比较不同运行时必须明确指标样本范围；发布对比以官方分数为准。全验证集答案空间与错误诊断见 [v0.3.0-vilt-diagnostics.md](evaluation/v0.3.0-vilt-diagnostics.md)。

## Conditional Run 2 / 条件第二轮

At most one follow-up full run is allowed. `scripts/select_vilt_followup.py` reads Run 1 history and selects exactly one branch. Validation instability takes priority; otherwise, a train-validation VQA gap above `0.10` selects partial fine-tuning of the final six ViLT layers. This is the predeclared metric-improvement candidate for the current run. Seed replication is reserved for stable runs and measures variance, not expected score gain.

最多允许再执行一次全量训练。`scripts/select_vilt_followup.py` 读取首轮历史并只选择一个分支。优先处理验证不稳定；否则训练/验证 VQA 差距大于 `0.10` 时，仅微调 ViLT 最后 6 层。对当前运行而言，这是预先定义的提分候选。seed 复现仅用于稳定运行的方差估计，不承诺提分。

Current evidence selects the last-six-layer branch: the recorded maximum train-validation VQA gap is `0.1754`, with no repeated validation drops above `0.03`. The ready-to-run configuration is `configs/kaggle_vilt_last6.yaml`; it keeps seed, effective batch, optimizer, epoch budget, preprocessing, and evaluation fixed. The only model/training variable changed is the ViLT trainable layer scope; the run name and checkpoint directory are distinct to preserve Run 1 artifacts.

当前数据选中“最后 6 层”分支：记录到的最大训练/验证 VQA 差距为 `0.1754`，且没有连续出现超过 `0.03` 的验证分数下降。可直接使用 `configs/kaggle_vilt_last6.yaml`；它保持 seed、有效 batch、优化器、epoch 上限、预处理和评估不变。模型/训练变量仅改变 ViLT 可训练层范围；实验名和 checkpoint 目录单独设置，以保留首轮产物。

```bash
python train.py --config configs/kaggle_vilt_last6.yaml --device cuda
torchrun --standalone --nproc_per_node=2 train.py --config configs/kaggle_vilt_last6.yaml --device cuda
```

| Condition | Single change |
| --- | --- |
| Repeated validation drops greater than 0.03 | Backbone LR `1e-5` |
| Maximum train-validation VQA gap greater than 0.10 | Train only final 6 ViLT layers |
| Both internal targets met, with no instability or large gap | Repeat with seed 1337 for stability only |
| Stable but below target | Backbone LR `3e-5` |

All other data, objective, batch, epoch budget, preprocessing, evaluation, and artifact settings remain fixed. There is no third full run in this protocol.

除选中变量外，数据、目标函数、batch、epoch 上限、预处理、评估和产物设置全部保持不变。本协议不安排第三次全量训练。

## Evaluation And Promotion / 评估与晋升

Every completed run must produce `training_history.csv`, `training_curves.png`, `run_metadata.json`, `run_summary.json`, `best.pt`, `latest.pt`, and 214,354 validation predictions. `official_vqa_metrics.json` is additionally required for official-protocol comparison. ViLT Run 1 satisfies the engineering artifact gate and records official validation accuracy `68.42`.

每次完成的运行必须生成 `training_history.csv`、`training_curves.png`、`run_metadata.json`、`run_summary.json`、`best.pt`、`latest.pt` 和 214,354 条验证预测；官方协议对比还必须生成 `official_vqa_metrics.json`。ViLT Run 1 已通过工程产物门槛，并记录官方验证准确率 `68.42`。

Local/Kaggle artifacts and embedded checkpoint configuration are the source of record. The Kaggle workflow does not depend on external experiment-tracking credentials.

以本地/Kaggle 产物和 checkpoint 内嵌配置为准。Kaggle 流程不依赖外部实验追踪凭证。

## Dependency Compatibility / 依赖兼容策略

The maintained ViLT training path uses `transformers>=4.40,<4.49`; the AutoDL training environment pins `4.48.3`, and the Kaggle runner uses the same compatible minor range. Keep this boundary until a deliberate upgrade has passed checkpoint load/inference, processor compatibility, gradient-checkpointing, and a real one-epoch training smoke run. Do not independently widen only one requirements file.

当前维护的 ViLT 训练路径使用 `transformers>=4.40,<4.49`；AutoDL 训练环境固定为 `4.48.3`，Kaggle runner 使用相同兼容次版本范围。只有在 checkpoint 加载/推理、processor 兼容性、gradient checkpointing 和真实单 epoch 训练 smoke run 全部通过后才升级。禁止只单独放宽某一个依赖文件。
