# Training Protocol / 训练协议

## Objective / 目标

Test whether generic image-text pretrained ViLT can materially exceed the best completed staged Cross-Attention result without using a VQAv2-fine-tuned source checkpoint. The internal targets are hard accuracy `>= 0.55` and VQA score `>= 0.65`; official VQA evaluation remains the release metric.

验证仅使用通用图文预训练权重的 ViLT，能否显著超过当前最佳的分阶段 Cross-Attention 结果；禁止使用已在 VQAv2 上微调的源 checkpoint。内部目标为硬准确率 `>= 0.55`、VQA score `>= 0.65`，发布判断仍以官方 VQA 评估为准。

## Fixed Run 1 / 固定首轮

### Current execution state / 当前执行状态

The seed-42 run completed through epoch 7 and stopped under the configured early-stopping rule. Epoch 5 is the selected checkpoint with hard accuracy `0.6126` and internal VQA score `0.7101`. The run exported all 214,354 validation predictions and passed both internal gates.

seed 42 任务已完成至 epoch 7，并按配置触发早停。epoch 5 是最佳 checkpoint，硬准确率 `0.6126`、内部 VQA score `0.7101`。任务已导出全部 214,354 条验证预测，并通过两个内部门槛。

This completed result is the recommended engineering checkpoint. Official VQA toolkit evaluation remains required before any leaderboard-style benchmark claim; Run 2 is not justified because epochs 6-7 already show a widening train-validation gap.

该完整结果作为推荐工程 checkpoint。任何 leaderboard 式 benchmark 声明仍需官方 VQA toolkit；epoch 6-7 已显示训练与验证差距扩大，因此没有继续 Run 2 的依据。

该部分结果不会立即触发晋升或 Run 2。Run 1 必须先完整结束，或由配置的 early stopping 正常停止，然后导出全部验证预测并完成官方评估。

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

## Conditional Run 2 / 条件第二轮

At most one follow-up full run is allowed. `scripts/select_vilt_followup.py` reads Run 1 history and selects exactly one branch:

最多允许再执行一次全量训练。`scripts/select_vilt_followup.py` 读取首轮历史并只选择一个分支：

| Condition | Single change |
| --- | --- |
| Both internal targets met | Repeat with seed 1337 |
| Repeated validation drops greater than 0.03 | Backbone LR `1e-5` |
| Maximum train-validation VQA gap greater than 0.10 | Train only final 6 ViLT layers |
| Stable but below target | Backbone LR `3e-5` |

All other data, objective, batch, epoch budget, preprocessing, evaluation, and artifact settings remain fixed. There is no third full run in this protocol.

除选中变量外，数据、目标函数、batch、epoch 上限、预处理、评估和产物设置全部保持不变。本协议不安排第三次全量训练。

## Evaluation And Promotion / 评估与晋升

Every completed run must produce `training_history.csv`, `training_curves.png`, `run_metadata.json`, `run_summary.json`, `best.pt`, `latest.pt`, and 214,354 validation predictions. `official_vqa_metrics.json` is additionally required for official benchmark claims. ViLT Run 1 satisfies the engineering artifact gate; official evaluation remains pending.

每次完成的运行必须生成 `training_history.csv`、`training_curves.png`、`run_metadata.json`、`run_summary.json`、`best.pt`、`latest.pt` 和 214,354 条验证预测；正式 benchmark 声明还必须生成 `official_vqa_metrics.json`。ViLT Run 1 已通过工程产物门槛，官方评估仍待补充。

Local/Kaggle artifacts and embedded checkpoint configuration are the source of record. The Kaggle workflow does not depend on external experiment-tracking credentials.

以本地/Kaggle 产物和 checkpoint 内嵌配置为准。Kaggle 流程不依赖外部实验追踪凭证。
