# Single- and Multi-GPU Training / 单卡与多卡训练

## Supported Modes / 支持模式

The same training entry point supports CPU, Apple Silicon MPS, one CUDA GPU, and single-host multi-GPU CUDA on an NCCL-enabled Linux runtime (such as Kaggle) through PyTorch Distributed Data Parallel (DDP). Multi-node training is not configured.

同一训练入口支持 CPU、Apple Silicon MPS、单张 CUDA GPU，以及在支持 NCCL 的 Linux 环境（如 Kaggle）中通过 PyTorch Distributed Data Parallel（DDP）运行的单机多卡 CUDA。当前未配置多机训练。

## Local Commands / 本地命令

Single GPU:

```bash
python train.py --config configs/kaggle_vilt.yaml --device cuda
```

Two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/kaggle_vilt.yaml --device cuda
```

Kaggle T4 x2 follow-up run:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config configs/kaggle_vilt_last6_t4x2.yaml --device cuda
```

Resume on one GPU:

```bash
python train.py --config configs/kaggle_vilt.yaml --device cuda --resume checkpoints/latest.pt
```

Resume on two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/kaggle_vilt.yaml --device cuda --resume checkpoints/latest.pt
```

CPU/MPS continue through the ordinary single-process `python train.py` entry point.

CPU/MPS 继续通过普通的单进程 `python train.py` 入口运行。

## Batch Semantics / Batch 语义

By default, DDP preserves the configured one-GPU effective batch (`batch_size × gradient_accumulation_steps`). It derives the per-GPU microbatch and accumulation count for the selected GPU count. For the ViLT configuration, one GPU uses `4 × 8 = 32`; two GPUs use `4 × 4 × 2 = 32`. The actual values are recorded in `run_metadata.json`.

默认情况下，DDP 保持配置对应的单卡有效 batch（`batch_size × gradient_accumulation_steps`），并根据 GPU 数量推导每卡 microbatch 和累积步数。ViLT 配置单卡为 `4 × 8 = 32`，双卡为 `4 × 4 × 2 = 32`。实际设置会写入 `run_metadata.json`。

The dedicated T4 x2 config sets `batch_size: 16` and `gradient_accumulation_steps: 2`. With two GPUs this resolves to 16 samples per GPU and 1 accumulation step, preserving global effective batch 32 while avoiding redundant gradient-checkpoint recomputation and reducing synchronization overhead. AMP remains enabled; two DataLoader workers per process (four total) feed the GPUs. If a 16 GB T4 runs out of memory, use `batch_size: 8` and `gradient_accumulation_steps: 4` for 8 samples per GPU and 2 accumulation steps, or fall back to `batch_size: 4` and `gradient_accumulation_steps: 8` for 4 per GPU and 4 accumulation steps. All options preserve global effective batch 32.

T4 x2 专用配置设为 `batch_size: 16`、`gradient_accumulation_steps: 2`。双卡时换算为每卡 16 个样本、每卡累积 1 步，全局有效 batch 仍为 32；启用 AMP、关闭会引入重复计算的 gradient checkpointing，并减少同步开销。每个进程使用 2 个 DataLoader worker（总计 4 个）为 GPU 供数。若 16 GB T4 显存不足，依次回退到 `batch_size: 8`、累积 4 步，或 `batch_size: 4`、累积 8 步；双卡实际分别换算为每卡 8 / 4 个样本和每卡累积 2 / 4 步，全局有效 batch 都保持 32。

Set `train.preserve_effective_batch_size: false` only when intentionally increasing the global batch. If the requested global batch cannot be divided exactly among workers, the runtime selects the next valid batch and records that adjustment.

只有在明确希望增大全局 batch 时才设置 `train.preserve_effective_batch_size: false`。若全局 batch 不能被 worker 数整除，程序会选择下一个可用 batch，并记录调整结果。

## Distributed Behavior / 分布式行为

- Training data uses a shuffled distributed sampler with a deterministic epoch seed.
- Training metrics and gradients are synchronized; only rank 0 writes checkpoints, reports, and tracking events.
- Validation runs once on rank 0 over the full validation set, avoiding padded or duplicate validation examples.
- Latest checkpoints retain per-rank RNG states. Resume with the same GPU count and configuration for the closest continuation.
- ViLT gradient checkpointing uses the non-reentrant implementation so it remains compatible with DDP unused-parameter tracking during partial fine-tuning.
- DDP uses NCCL and requires one visible CUDA device per worker. Input data and checkpoints must be on a shared local filesystem.

- 训练数据使用按 epoch 固定随机种子的分布式 shuffle sampler。
- 训练梯度和指标会同步；只有 rank 0 写 checkpoint、报告和追踪事件。
- 验证只由 rank 0 对完整验证集执行，避免验证样本补齐或重复。
- latest checkpoint 保存各 rank 的随机状态。为尽量精确续训，建议使用相同 GPU 数量和配置恢复。
- ViLT 使用非重入式 gradient checkpointing，确保部分微调时与 DDP 的 unused-parameter 跟踪兼容。
- DDP 使用 NCCL，每个 worker 需要一张可见 CUDA GPU；输入数据和 checkpoint 必须位于共享的本机文件系统。

## Kaggle / Kaggle 使用

The repository's Kaggle API runner defaults to `configs/kaggle_vilt_last6_t4x2.yaml` and uses all attached CUDA GPUs by default. Set `VQA_NUM_GPUS=1` to force one GPU or `VQA_NUM_GPUS=2` to request two GPUs. The requested count is checked before training. Kaggle notebooks that invoke `train.py` directly can use the T4 x2 `torchrun` command above.

仓库的 Kaggle API runner 默认使用 `configs/kaggle_vilt_last6_t4x2.yaml`，并自动使用当前实例挂载的全部 CUDA GPU。设置 `VQA_NUM_GPUS=1` 可强制单卡，设置 `VQA_NUM_GPUS=2` 可指定双卡；训练前会校验 GPU 数量。直接调用 `train.py` 的 Kaggle Notebook 也可使用上面的 T4 x2 `torchrun` 命令。
