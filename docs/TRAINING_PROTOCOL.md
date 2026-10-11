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

## Historical ViLT Run 2 Protocol / 历史 ViLT 第二轮协议

This section records the closed ViLT follow-up protocol that governed Kaggle Version 34. It is historical and does not authorize another unchanged ViLT run. Version 34 scored `68.08` under the official validation protocol and was not promoted.

本节记录已经结束的 ViLT 后续实验协议，对应 Kaggle Version 34。它仅作历史记录，不构成继续原样训练 ViLT 的授权。Version 34 官方验证分数为 `68.08`，未予晋升。

The historical ViLT protocol allowed at most one follow-up full run. `scripts/select_vilt_followup.py` read the prior run history and selected exactly one branch. Validation instability took priority; otherwise, a train-validation VQA gap above `0.10` selected partial fine-tuning of the final six ViLT layers. Seed replication was reserved for stable runs and measured variance, not expected score gain.

最多允许再执行一次全量训练。`scripts/select_vilt_followup.py` 读取首轮历史并只选择一个分支。优先处理验证不稳定；否则训练/验证 VQA 差距大于 `0.10` 时，仅微调 ViLT 最后 6 层。对当前运行而言，这是预先定义的提分候选。seed 复现仅用于稳定运行的方差估计，不承诺提分。

Version 34 used the last-six-layer branch because the recorded maximum train-validation VQA gap was `0.1754`, with no repeated validation drops above `0.03`. Its configuration was `configs/kaggle_vilt_last6.yaml`; seed, effective batch, optimizer, epoch budget, preprocessing, and evaluation remained fixed. The ViLT trainable layer scope was the only planned model/training variable change; artifacts used a separate run name and checkpoint directory.

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

All other data, objective, batch, epoch budget, preprocessing, evaluation, and artifact settings remain fixed. There is no third full run in this historical ViLT protocol.

除选中变量外，其他数据、目标函数、batch、epoch 预算、预处理、评估和产物设置全部保持不变。该历史 ViLT 协议不安排第三次全量训练。

除选中变量外，数据、目标函数、batch、epoch 上限、预处理、评估和产物设置全部保持不变。本协议不安排第三次全量训练。

## Evaluation And Promotion / 评估与晋升

Every completed run must produce `training_history.csv`, `training_curves.png`, `run_metadata.json`, `run_summary.json`, `best.pt`, `latest.pt`, and 214,354 validation predictions. `official_vqa_metrics.json` is additionally required for official-protocol comparison. ViLT Run 1 satisfies the engineering artifact gate and records official validation accuracy `68.42`.

每次完成的运行必须生成 `training_history.csv`、`training_curves.png`、`run_metadata.json`、`run_summary.json`、`best.pt`、`latest.pt` 和 214,354 条验证预测；官方协议对比还必须生成 `official_vqa_metrics.json`。ViLT Run 1 已通过工程产物门槛，并记录官方验证准确率 `68.42`。

Local/Kaggle artifacts and embedded checkpoint configuration are the source of record. The Kaggle workflow does not depend on external experiment-tracking credentials.

以本地/Kaggle 产物和 checkpoint 内嵌配置为准。Kaggle 流程不依赖外部实验追踪凭证。

## Dependency Compatibility / 依赖兼容策略

The supported runtime is Python 3.10+ with PyTorch 2.5.1+ and Transformers 4.57.6. Root requirements, AutoDL, and Kaggle pin `transformers==4.57.6`: Transformers 5.19.0 changed the XLM-R tokenizer constructor and currently fails when BEiT-3's verified SentencePiece model is passed through the upstream-compatible API. Keep these constraints aligned across every install path. Before upgrading Transformers again, verify tokenizer parity, checkpoint load/inference, processor compatibility, gradient-checkpointing, and a real training smoke run on Kaggle.

支持的运行环境为 Python 3.10+、PyTorch 2.5.1+ 和 Transformers 4.57.6。根目录依赖、AutoDL 和 Kaggle 统一固定 `transformers==4.57.6`：Transformers 5.19.0 更改了 XLM-R tokenizer 构造接口，当前会在 BEiT-3 传入已校验 SentencePiece 模型时失败。所有安装入口必须保持约束一致。下次升级 Transformers 前，须在 Kaggle 验证 tokenizer 编码一致性、checkpoint 加载/推理、processor 兼容性、gradient checkpointing 和真实训练 smoke run。

## BEiT-3 Improvement Program / BEiT-3 提分计划

### Baseline and target / 基线与目标

The unchanged comparison baseline is the recommended ViLT `v0.3.0` checkpoint at `68.42` official VQA v2 validation overall. The aspirational acceptance target is `>=75.00` for the selected experiment and `>=74.50` for an independent seed replication. These are targets, not predicted outcomes. The validation set is fixed at all 214,354 public questions and is never used for training or vocabulary construction.

固定对照基线为推荐 ViLT `v0.3.0`，官方 VQA v2 验证 overall 为 `68.42`。目标为优选实验 `>=75.00`，独立 seed 复现 `>=74.50`。这是验收目标，不是结果预测。验证集固定为全部 214,354 条公开问题，绝不用于训练或词表构建。

### Experiment matrix / 实验矩阵

| Run | Training source | Seed | Config |
| --- | --- | ---: | --- |
| A | VQA v2 train only | 42 | `configs/kaggle_beit3_a_t4x2.yaml` |
| B | VQA v2 train plus screened Visual Genome QA from COCO training images | 42 | `configs/kaggle_beit3_b_vg_t4x2.yaml` |
| C | Independently initialize the better A/B recipe | 123 | `configs/kaggle_beit3_c_seed123_t4x2.yaml` |

Use no more than these three full experiments. A/B keep the model and training recipe identical; the VG source is the only planned data change. Every B epoch traverses all VQA train examples and samples without replacement up to 25% as many screened VG QA pairs, with a deterministic `seed + epoch` sampler shared across ranks. After comparing A/B, generate C from the winner with `scripts/create_beit3_replication_config.py`; C then copies the selected recipe and changes only the seed. VG preparation must log source checksums and filter counts, map only to COCO images used by the VQA training split, and reject validation-image overlap, invalid mappings, duplicate normalized question/answer pairs, and answers outside the Top-5000 VQA-training vocabulary. B never uses VQA validation answers.

完整实验最多执行上述三组。A/B 的模型和训练配方一致，仅按计划改变 VG 数据源。B 每轮完整遍历所有 VQA 训练样本，并使用 `seed + epoch` 的跨 rank 确定性采样器，从筛选后的 VG 问答中无放回抽取不超过 VQA 数量 25% 的样本。比较 A/B 后，使用 `scripts/create_beit3_replication_config.py` 从胜出配置生成 C，确保 C 只改变随机种子。VG 准备流程必须记录源文件校验和与过滤计数，仅映射到 VQA 训练 split 使用的 COCO 图片，并排除验证图片重叠、无效映射、重复规范化问答及 Top-5000 VQA 训练词表外答案。B 不得使用 VQA 验证答案。

### Hardware gate / 硬件门槛

On the actual dual-T4 session, `scripts/benchmark_beit3_batch.py` measures per-GPU micro-batches 1/2/4/8. `scripts/prepare_beit3_runtime_config.py` selects the fastest candidate whose peak allocated and reserved memory stay below 88% of device memory, applies it to an isolated runtime YAML, and preserves global effective batch 128. The Kaggle runner and notebook invoke both automatically; resume must reuse the same report/config. Then `scripts/smoke_beit3_distributed.py` must show both GPUs completing two finite forward/backward optimizer steps, synchronized parameters, and restored model, optimizer, scaler, per-rank RNG, data-generator state, and step cursor. After full data validation, `scripts/estimate_beit3_runtime.py` records a planning estimate using measured throughput and the actual question counts; its 1.5x margin is explicitly approximate. No full training is allowed until hardware gates and strict full-data validation pass.

在目标双 T4 会话，`scripts/benchmark_beit3_batch.py` 测量每卡 micro-batch 1/2/4/8。`scripts/prepare_beit3_runtime_config.py` 会选择峰值已分配和预留显存均低于设备显存 88% 的最快配置，写入独立运行 YAML，并保持全局有效 batch 128。Kaggle runner 和 notebook 会自动调用，两者续训时必须复用同一报告和配置。随后 `scripts/smoke_beit3_distributed.py` 必须证明两张卡完成两次有限值前向/反向 optimizer step、参数同步，并恢复模型、优化器、scaler、各 rank RNG、数据生成器状态和 step 游标。全量数据严格校验后，`scripts/estimate_beit3_runtime.py` 会根据实测吞吐和实际问题数生成规划估算；其中 1.5 倍缓冲仅为近似值。硬件门槛与完整数据校验未通过前，禁止全量训练。

Kaggle Version 36 confirmed both T4 devices and completed the benchmark command, but its distributed smoke did not reach checkpoint round-trip: the default FP16 loss scale skipped both optimizer updates, leaving AdamW state uninitialized. The runner now starts BEiT-3 with a configurable loss scale of `1024`, retries bounded smoke/benchmark updates until real optimizer steps occur, and verifies the real-data smoke checkpoint has at least one successful update. Skipped AMP updates do not advance the optimizer-step cursor or update-based scheduler. Version 36 is a failed preflight, not a training result; benchmark numbers from runs without successful optimizer updates are not accepted.

Kaggle Version 36 确认检测到两张 T4 并执行了 benchmark 命令，但分布式 smoke 未进入 checkpoint 往返验证：FP16 默认 loss scale 导致两次 optimizer update 都被跳过，AdamW 状态因而尚未初始化。runner 现以可配置的 `1024` loss scale 启动 BEiT-3；smoke/benchmark 会在有限重试内确认真实 optimizer update 成功，并检查真实数据 smoke 的 checkpoint 至少包含一次成功更新。AMP 跳过的更新不会推进 optimizer-step 游标或按 update 调度的 scheduler。因此 Version 36 属于预检失败，不是训练结果；没有真实 optimizer update 的 benchmark 数值不予采纳。

Version 37 validated the AMP fix: the dual-T4 capacity benchmark selected a per-GPU batch size of 8, and both ranks completed two successful optimizer updates. The run then failed in the checkpoint smoke comparator, which called `torch.equal` on CPU and CUDA optimizer-state tensors. The comparator now normalizes tensors to CPU and checks shape, dtype, and exact values. No dataset preparation or full training started in Version 37; the hardware gate must be rerun.

Version 37 已验证 AMP 修复有效：双 T4 容量 benchmark 选择每卡 batch size 8，两个 rank 均完成两次成功的 optimizer update。随后 checkpoint smoke 比较器直接对位于 CPU 和 CUDA 的 optimizer state 张量调用 `torch.equal`，导致运行失败。现已修复为先将张量统一到 CPU，再核对形状、dtype 和精确值。Version 37 未开始数据准备或全量训练；仍须重新通过硬件门槛。

Version 38 passed the cross-device optimizer-state comparison, then exposed an omitted `scaler` argument in the smoke checkpoint save call. As a result, the checkpoint contained no AMP scaler state and the restore assertion correctly failed. The smoke now saves the scaler and explicitly rejects checkpoints without scaler state. Version 38 also stopped before dataset preparation and full training.

Version 38 已通过跨设备 optimizer state 比较，随后发现 smoke 保存 checkpoint 时漏传 `scaler` 参数，因此 checkpoint 未包含 AMP scaler 状态，恢复断言按预期失败。现已补充保存 scaler，并增加 checkpoint 必须包含该状态的检查。Version 38 同样在数据准备和全量训练前停止。

### Training and resume / 训练与续训

The initial recipe uses generic BEiT-3 Base pretraining, a fresh VQA head, 480 px official preprocessing, full-backbone fine-tuning, Top-5000 training-only vocabulary, AdamW (backbone LR `3e-5`, head LR `6e-4`, betas `(0.9, 0.98)`, weight decay `0.01`), gradient clipping `1.0`, 10% warmup followed by cosine decay, AMP, activation checkpointing, and up to 10 epochs. Random crop, flip, and color jitter remain disabled. From epoch 4, stop after three consecutive epochs without at least `0.1` official-score improvement.

初始配方使用通用 BEiT-3 Base 预训练权重和新建 VQA 分类头，采用 480 px 官方预处理、仅由训练数据构建的 Top-5000 词表、AdamW（骨干 LR `3e-5`、分类头 LR `6e-4`、betas `(0.9, 0.98)`、weight decay `0.01`）、梯度裁剪 `1.0`、10% warmup 后 cosine decay、AMP、activation checkpoint，最多 10 epoch。不启用随机裁剪、翻转或颜色扰动。从 epoch 4 起，若连续三轮官方分数提升均不足 `0.1`，则停止。

Write `latest.pt` every 500 optimizer updates and at each epoch boundary. A resumable checkpoint must include data/config/model fingerprints, optimizer, scheduler, scaler, all-rank RNG states, and the in-epoch sample cursor. Stop Kaggle sessions before the platform limit, preserve all run artifacts in a private output Dataset, and resume only if the fingerprint and world size match. W&B remains disabled.

每 500 次 optimizer update 和每个 epoch 边界写入 `latest.pt`。可续训 checkpoint 必须保存数据/配置/模型指纹、优化器、调度器、scaler、各 rank 随机状态和 epoch 内样本游标。Kaggle 会话须在平台限制前主动停止，并将完整产物保存在私有输出 Dataset 中；只有指纹和 world size 一致时才可恢复。W&B 保持关闭。

### Official evaluation and promotion / 官方评测与晋升

For each completed candidate, export predictions for all validation IDs, compare the IDs exactly against questions and annotations, and run the upstream VQA Evaluation Toolkit twice with identical output. Report overall plus `yes/no`, `number`, and `other`, full data/config/source hashes, history, curves, and checkpoints. If the acceptance targets are missed, report the actual results without substituting internal VQA metrics, ensembles, a partial split, or a changed protocol. Keep `v0.3.0` recommended until the best A/B run reaches `75.00`, the independent C run reaches `74.50`, and all artifact and evaluation checks pass.

每个完成的候选模型都须导出全部验证 ID 的预测，与 questions 和 annotations 精确比对，并运行上游 VQA Evaluation Toolkit 两次且结果一致。报告 overall、`yes/no`、`number`、`other`、完整数据/配置/来源哈希、训练历史、曲线与 checkpoint。若未达验收目标，按实际结果报告，不得以内部 VQA 指标、集成、部分数据或修改协议替代。优选 A/B 达到 `75.00`、独立 C 达到 `74.50` 且所有产物/评测检查通过前，`v0.3.0` 始终为推荐版本。
