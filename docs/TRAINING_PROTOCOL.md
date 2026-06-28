# Training Protocol / 训练协议

## Objective / 目标

Test whether generic image-text pretrained ViLT can materially exceed the best completed staged Cross-Attention result without using a VQAv2-fine-tuned source checkpoint. The internal targets are hard accuracy `>= 0.55` and VQA score `>= 0.65`; official VQA evaluation remains the release metric.

验证仅使用通用图文预训练权重的 ViLT，能否显著超过当前最佳的分阶段 Cross-Attention 结果；禁止使用已在 VQAv2 上微调的源 checkpoint。内部目标为硬准确率 `>= 0.55`、VQA score `>= 0.65`，发布判断仍以官方 VQA 评估为准。

## Fixed Run 1 / 固定首轮

| Item | Value |
| --- | --- |
| Config | `configs/kaggle_vilt.yaml` |
| Pretrained source | `dandelin/vilt-b32-mlm-itm` |
| Seed | 42 |
| Epoch budget | Up to 10 |
| Physical / effective batch | 4 / 32 |
| Backbone / head LR | `2e-5` / `1e-4` |
| Objective | Soft-target multilabel BCE, no label smoothing |
| Scheduler | 5% step warmup, then metric plateau reduction |
| Early stopping | Start epoch 3, patience 2, minimum delta `0.002` |
| Image processing | ViLT processor at 384 px, no semantic crop or color jitter |
| Tracking | W&B online required; checkpoint upload disabled |

Run 1 trains all ViLT layers with AMP, gradient checkpointing, gradient clipping, and parameter-group learning rates. `latest.pt` is saved every epoch and contains optimizer, scheduler, scaler, RNG, history, and W&B run ID state.

首轮使用 AMP、梯度 checkpoint、梯度裁剪和参数组学习率训练全部 ViLT 层。每个 epoch 保存 `latest.pt`，其中包含 optimizer、scheduler、scaler、随机状态、历史记录和 W&B run ID，可在中断后接回同一在线实验。

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

All other data, objective, batch, epoch budget, preprocessing, evaluation, and tracking settings remain fixed. There is no third full run in this protocol.

除选中变量外，数据、目标函数、batch、epoch 上限、预处理、评估和追踪设置全部保持不变。本协议不安排第三次全量训练。

## Evaluation And Promotion / 评估与晋升

Every completed run must produce `training_history.csv`, `training_curves.png`, `run_metadata.json`, `run_summary.json`, `best.pt`, `latest.pt`, 214,354 validation predictions, and `official_vqa_metrics.json`. The existing `checkpoints/kaggle_finetune_best.pt` remains recommended until a ViLT run passes both internal gates and its official score is verified.

每次完成的运行必须生成 `training_history.csv`、`training_curves.png`、`run_metadata.json`、`run_summary.json`、`best.pt`、`latest.pt`、214,354 条验证预测和 `official_vqa_metrics.json`。在 ViLT 同时通过两个内部门槛并完成官方分数验证前，继续推荐现有 `checkpoints/kaggle_finetune_best.pt`。

W&B is an observability system, not the source of record. Local/Kaggle artifacts and embedded checkpoint configuration remain authoritative. Real API keys must only be supplied through environment variables or Kaggle Secrets.

W&B 仅用于在线观测，不是最终记录来源。以本地/Kaggle 产物和 checkpoint 内嵌配置为准。真实 API Key 只能通过环境变量或 Kaggle Secrets 提供。
