# Kaggle Training / Kaggle 训练

This workflow uses one training engine for local CPU, local CUDA, Apple MPS, and Kaggle CUDA. The published recommendation remains the completed ViLT checkpoint; the maintained Kaggle runner now defaults to the separate BEiT-3 A experiment route, which must pass its hardware and data gates before full training.

本流程在本地 CPU、本地 CUDA、Apple MPS 和 Kaggle CUDA 之间共用同一训练引擎。当前已发布推荐权重仍是 ViLT；维护中的 Kaggle runner 默认改为独立的 BEiT-3 A 实验路线，全量训练前必须通过硬件与数据门槛。

For direct `torchrun` commands, effective-batch semantics, and checkpoint resume behavior, see [Distributed Training / 单卡与多卡训练](DISTRIBUTED_TRAINING.md).

`torchrun` 命令、有效 batch 语义和 checkpoint 续训说明见[单卡与多卡训练文档](DISTRIBUTED_TRAINING.md)。

## Required Dataset Layout / 数据集目录

Attach a Kaggle Dataset and point `DATA_ROOT` to the directory containing:

挂载 Kaggle Dataset，并将 `DATA_ROOT` 指向包含以下文件的目录：

```text
vqa/
├── train2014/
├── val2014/
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_mscoco_train2014_annotations.json
├── v2_OpenEnded_mscoco_val2014_questions.json
└── v2_mscoco_val2014_annotations.json
```

Validate the mounted input before enabling the GPU run:

开始 GPU 训练前先校验挂载数据：

```bash
python scripts/validate_vqa_data.py \
  --root /kaggle/input/<dataset>/vqa \
  --sample-images 20 \
  --strict-full
```

## Runtime Selection / 运行方式

```bash
# Local automatic device selection
python train.py --config configs/default.yaml --device auto

# Local NVIDIA GPU
python train.py --config configs/default.yaml --device cuda

# CPU smoke test
python train.py --config configs/demo_cpu.yaml --device cpu

# Kaggle staged fine-tuning
python train.py \
  --config configs/kaggle_finetune.yaml \
  --device cuda \
  --data-root /kaggle/input/<dataset>/vqa \
  --answer-vocab-path /kaggle/working/answer_vocab.json \
  --checkpoint-dir /kaggle/working/multimodal-vqa/finetune

# Kaggle T4 x2 final-six-layer experiment
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config configs/kaggle_vilt_last6_t4x2.yaml \
  --device cuda \
  --data-root /kaggle/input/<dataset>/vqa \
  --answer-vocab-path /kaggle/working/answer_vocab.json \
  --checkpoint-dir /kaggle/working/multimodal-vqa/vilt-last6-t4x2 \
  --no-wandb

# Single-GPU generic ViLT experiment
python train.py \
  --config configs/kaggle_vilt.yaml \
  --device cuda \
  --data-root /kaggle/input/<dataset>/vqa \
  --answer-vocab-path /kaggle/working/answer_vocab.json \
  --checkpoint-dir /kaggle/working/multimodal-vqa/vilt \
  --no-wandb
```

Kaggle runs do not require W&B or Kaggle Secrets. Training progress and final results are recorded in `training_history.csv`, `training_curves.png`, `run_metadata.json`, `run_summary.json`, and the packaged checkpoint artifacts.

Kaggle 训练不需要 W&B 或 Kaggle Secrets。训练过程和最终结果会记录在 `training_history.csv`、`training_curves.png`、`run_metadata.json`、`run_summary.json` 以及打包的 checkpoint 产物中。

## BEiT-3 Experiments / BEiT-3 实验

The BEiT-3 route is separate from the recommended ViLT release. Do not start a full run until the code revision is pushed and pinned in `GIT_REF`, the complete VQA/COCO dataset passes strict validation, and both T4 GPUs pass the capacity and checkpoint smoke gates below. Enable Kaggle Internet for the first run so it can install the pinned source dependency and download the 554 MB pretrained checkpoint plus tokenizer; alternatively, pre-populate the configured BEiT-3 cache. The optional dependencies are isolated from existing model paths:

BEiT-3 路线与当前推荐的 ViLT 发布版本相互独立。在代码 revision 已推送并通过 `GIT_REF` 固定、完整 VQA/COCO 数据通过严格校验、两张 T4 均通过下述容量和 checkpoint smoke 门槛前，不要启动全量训练。首次运行需在 Kaggle 启用 Internet，以安装固定源码依赖并下载约 554 MB 的预训练权重及 tokenizer；也可预先填充配置的 BEiT-3 缓存目录。BEiT-3 依赖为可选项，不影响现有模型路径：

```bash
python -m pip install -e ".[beit3]"

# Benchmark per-GPU micro-batches 1/2/4/8 on both T4s.
torchrun --standalone --nproc_per_node=2 scripts/benchmark_beit3_batch.py \
  --output /kaggle/working/beit3_batch_benchmark.json

# Generate a run config that applies the fastest safe batch and keeps global batch 128.
python scripts/prepare_beit3_runtime_config.py \
  --config configs/kaggle_beit3_a_t4x2.yaml \
  --benchmark /kaggle/working/beit3_batch_benchmark.json \
  --output /kaggle/working/beit3-a/runtime_config.yaml \
  --world-size 2

# Real two-GPU forward/backward, parameter-sync, checkpoint save/restore test.
torchrun --standalone --nproc_per_node=2 scripts/smoke_beit3_distributed.py \
  --output-dir /kaggle/working
```

Keep a candidate only when both workers report a finite loss and synchronized parameters, and the benchmark selects a batch below the 88% allocated/reserved-memory ceiling. `prepare_beit3_runtime_config.py` applies `selected_batch_size_per_gpu` and the matching gradient-accumulation setting while preserving global batch 128; it rejects missing, unsafe, or wrong-world-size benchmark reports. The Kaggle runner and notebook do this automatically and reuse the saved runtime config on resume.

只有在两个 worker 都报告有限 loss、参数同步一致，且 benchmark 选出的 batch 低于 88% 已分配/预留显存上限时，才可进入全量训练。`prepare_beit3_runtime_config.py` 会应用 `selected_batch_size_per_gpu` 与对应的梯度累积设置，同时保持全局 batch 128；缺失、不安全或 world size 不匹配的报告会被拒绝。Kaggle runner 和 notebook 会自动完成此步骤，并在续训时复用运行配置。

The three configurations are isolated and must run as separate experiments:

三份配置相互隔离，必须作为独立实验运行：

| Run | Config | Training data | Seed | Use |
| --- | --- | --- | ---: | --- |
| A | `configs/kaggle_beit3_a_t4x2.yaml` | VQA v2 train only | 42 | Primary candidate |
| B | `configs/kaggle_beit3_b_vg_t4x2.yaml` | All VQA v2 train + deterministic sample of at most 25% as many screened VG QA pairs each epoch | 42 | Measure added-supervision effect |
| C | `configs/kaggle_beit3_c_seed123_t4x2.yaml` | Exact A/B winner recipe | 123 | Independent replication |

The checked-in C file is an A-shaped template. After comparing the official A/B results, generate the actual independent-replication config from the winner so the data source, sampling ratio, batch, and schedule match exactly:

已提交的 C 文件是基于 A 的模板。比较 A/B 官方结果后，须从胜出配置生成真正的独立复现配置，确保数据源、采样比例、batch 和训练计划完全一致：

```bash
python scripts/create_beit3_replication_config.py \
  --source-config configs/kaggle_beit3_b_vg_t4x2.yaml
```

Use the A config path instead if A wins. Record and pin the generated C config revision before starting that run.

如果 A 胜出，则将命令中的源配置改为 A。运行 C 前应记录并固定生成后的配置 revision。

The maintained Kaggle API runner defaults to experiment A; set `CONFIG_PATH`, `RUN_NAME`, and `GIT_REF` at the top of `kaggle_finetune_kernel/run_kaggle_finetune.py` or provide environment overrides. The legacy ViLT recipe remains available by explicitly setting `CONFIG_PATH=configs/kaggle_vilt_last6_t4x2.yaml`. For BEiT-3, the runner installs pinned optional dependencies, downloads and verifies pretrained assets, runs the batch benchmark, generates a separate runtime config, and completes the distributed save/restore smoke before data preparation and full training. After strict dataset validation and a separate 128-example training smoke through the real VQA dataloader, it prints and saves a throughput-based runtime/session estimate. On resume it reuses the saved batch report and runtime config, then re-runs the hardware and data smokes. For B, it also downloads Visual Genome metadata, filters QA to COCO images in the VQA training split, and writes a filter manifest before training; the manifest is included in the private resume artifact. The generic notebook defaults to A, supports A/C, and directs B to the maintained runner.

维护中的 Kaggle API runner 默认运行实验 A；可在 `kaggle_finetune_kernel/run_kaggle_finetune.py` 设置 `CONFIG_PATH`、`RUN_NAME`、`GIT_REF`，或通过环境变量覆盖。旧 ViLT 配方仍可通过显式设置 `CONFIG_PATH=configs/kaggle_vilt_last6_t4x2.yaml` 运行。BEiT-3 启动前会安装固定版本可选依赖、下载并校验预训练资源、测量 micro-batch、生成独立运行配置，并通过双卡保存/恢复 smoke；完整数据严格校验和使用真实 VQA dataloader 的独立 128 样本训练 smoke 通过后，才会输出并保存基于实测吞吐的耗时/会话估算。续训时复用原 batch 报告和运行配置，但会重新检查硬件与数据流程。B 还会下载 Visual Genome 元数据，仅保留映射到 VQA 训练 split COCO 图片的 QA，并在训练前生成过滤清单；该 manifest 会打包进私有续训产物，每轮遍历完整 VQA 训练集，再按 seed 和 epoch 无放回抽取不超过 25% 的 VG 样本。通用 notebook 默认运行 A，可运行 A/C；B 使用维护中的 runner。

`latest.pt` is saved every 500 optimizer updates and at epoch boundaries. If Kaggle ends a session near the configured 11-hour cutoff, preserve the entire run directory as a private Kaggle Dataset and attach it as the resume input for the next session. `--resume` validates the model, tokenizer, vocabulary, config signature, data fingerprint, and distributed world size before restoring optimizer, scheduler, scaler, RNG state, and the in-epoch data cursor. Never resume a partial run after changing its model, input resolution, vocabulary, global batch, optimizer, or data source.

每 500 次 optimizer update 及每个 epoch 边界都会保存 `latest.pt`。Kaggle 会话接近配置的 11 小时截止点时，将完整 run 目录保存为私有 Kaggle Dataset，并在下一会话中挂载续训。`--resume` 会先校验模型、tokenizer、词表、配置签名、数据指纹和分布式 world size，再恢复 optimizer、scheduler、scaler、随机状态与 epoch 内数据游标。模型、输入分辨率、词表、全局 batch、优化器或数据源发生变化时，不得续接旧的部分训练。

The official comparison requires predictions for all 214,354 validation question IDs and the upstream VQA Evaluation Toolkit. Do not use validation answers in vocabulary construction, Visual Genome filtering, or training. No BEiT-3 score is claimed until the full export is evaluated and reproduced from its saved artifacts.

官方对比必须覆盖全部 214,354 个验证 question ID，并运行上游 VQA Evaluation Toolkit。验证答案不得用于词表构建、Visual Genome 筛选或训练。BEiT-3 全量预测完成官方评测且结果能从保存产物复核前，不报告任何 BEiT-3 分数。

## Resume / 断点续训

`latest.pt` is written atomically after every epoch. Reattach the previous Notebook output as an Input or continue in the same saved session, then run:

`latest.pt` 在每个 epoch 结束后原子写入。将上一版 Notebook 输出重新挂载为 Input，或在同一保存会话中执行：

The maintained Kaggle script defaults to the private dataset slug
`multimodal-vqa-beit3-a-seed42-resume` for BEiT-3 A. Set `RESUME_ROOT` to use a different
attached Dataset. When present, it restores `latest.pt`, `best.pt`, the answer vocabulary,
run history, benchmark report, capacity estimate, and runtime config before resuming.
The historical ViLT continuation remains available by selecting its ViLT config and
matching `RESUME_ROOT`. BEiT-3 checkpoints include an in-epoch data cursor; the runner
preserves the same effective batch and only resumes under the recorded GPU world size.

维护中的 Kaggle 脚本对 BEiT-3 A 默认使用私有数据集 slug
`multimodal-vqa-beit3-a-seed42-resume`。可通过 `RESUME_ROOT` 指定其他已挂载数据集。
续训前会恢复 `latest.pt`、`best.pt`、答案词表、训练历史、batch benchmark、容量估算和运行配置。
历史 ViLT 续训仍可通过指定 ViLT config 与匹配的 `RESUME_ROOT` 使用。BEiT-3 checkpoint
包含 epoch 内数据游标；runner 保持相同有效 batch，并要求续训的 GPU world size 与记录一致。

The private Dataset `dongtingshuo/multimodal-vqa-vilt-resume` is the historical epoch-2
continuation artifact used to move the run off Kaggle. The run was subsequently completed on
AutoDL through the configured epoch-7 early stop. Keep this Dataset only for reproducibility;
use the v0.3.0 Release checkpoint for inference and evaluation.

私有 Dataset `dongtingshuo/multimodal-vqa-vilt-resume` 是用于将任务从 Kaggle 迁出的历史
epoch-2 续训产物。该任务随后已在 AutoDL 完成，并在 epoch 7 按配置早停。该 Dataset 仅用于
复现；推理与评估应使用 v0.3.0 Release 权重。

```bash
python train.py \
  --config configs/kaggle_vilt.yaml \
  --device cuda \
  --data-root /kaggle/input/<dataset>/vqa \
  --answer-vocab-path /kaggle/working/answer_vocab.json \
  --checkpoint-dir /kaggle/working/multimodal-vqa/vilt \
  --resume /kaggle/working/multimodal-vqa/vilt/latest.pt \
  --no-wandb
```

Resume permits different local paths, device, workers, logging location, and a larger total epoch count. Model, preprocessing, effective batch, optimizer rates, and fine-tune schedule must match.

断点续训允许更换本地路径、设备、worker、日志目录和更大的总 epoch 数。模型、预处理、有效 batch、优化器学习率和微调时序必须一致。

## Notebook / Notebook 模板

Import [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb). It defaults to BEiT-3 A on T4 x2, automatically selects and applies the safe micro-batch, runs the DDP checkpoint smoke, validates full data, performs a short data-loader training smoke, and prints the runtime estimate before the resumable full-training cell. Set `DATA_ROOT`, `CONFIG_PATH`, `RUN_NAME`, `TOTAL_EPOCHS`, and `GIT_REF` in the first code cell. Use a release tag or commit SHA for a final reproducible run.

导入 [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb)。模板默认在 T4 x2 上运行 BEiT-3 A，会自动选择并应用安全 micro-batch，运行 DDP checkpoint smoke、校验完整数据、执行短数据加载训练 smoke，并在可续训的全量训练单元前输出耗时估算。在首个代码单元设置 `DATA_ROOT`、`CONFIG_PATH`、`RUN_NAME`、`TOTAL_EPOCHS` 和 `GIT_REF`。正式运行应锁定 Release tag 或 commit SHA。

## Kaggle API Kernel / Kaggle API 远程任务

The repository also includes a script kernel in [`kaggle_finetune_kernel/`](../kaggle_finetune_kernel). It can be pushed from a local machine with Kaggle API credentials:

仓库也提供了 [`kaggle_finetune_kernel/`](../kaggle_finetune_kernel) 脚本任务目录，可在本机通过 Kaggle API 直接推送运行：

```bash
kaggle kernels push -p kaggle_finetune_kernel/coco_train_pack
kaggle kernels push -p kaggle_finetune_kernel/coco_val_pack
```

The original `sagnikkayalcse52/coco2014vqa` source contains more than 160,000 individual image files, and both Dataset and Notebook-output inputs have shown intermittent worker-side mount failures before Python starts. The two CPU-only pack kernels remain available as an optional path; they package train and validation images separately, each below Kaggle's 20 GB per-Notebook output limit. Kaggle supports reusing a Notebook's saved output as another Notebook's input ([Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks)). Keep both pack kernels private.

原始 `sagnikkayalcse52/coco2014vqa` 数据源包含 16 万多个独立图片文件；直接数据集输入和 Notebook 输出输入都出现过 worker 在 Python 启动前挂载失败的情况。两个仅使用 CPU 的打包 kernel 仍可作为可选路径，分别打包 train/val 图片，单个 Notebook 输出均低于 Kaggle 20 GB 上限。Kaggle 支持将 Notebook 保存的输出作为另一个 Notebook 的输入（见 [Kaggle Notebooks 文档](https://www.kaggle.com/docs/notebooks)）。请保持两个打包 kernel 为私有。

The main training kernel intentionally has no `dataset_sources` or `kernel_sources`, so it can reach Python without waiting on Kaggle's large-input mount. When no complete local mount or optional packed output is present, the runner downloads the official COCO 2014 `train2014.zip` and `val2014.zip` from the [COCO download host](https://cocodataset.org/#download) over resumable HTTPS, extracts them under `/tmp`, removes each archive after extraction, and validates the exact image counts before training. This transfers about 20.2 GB on each fresh session; interrupted Kaggle sessions may need to download the archives again. Set `VQA_NETWORK_SMOKE_ONLY=1` for a bounded check of both official download endpoints and both CUDA devices; it exits before downloading data, checking out source, installing dependencies, or training. `VQA_DATA_SMOKE_ONLY=1` remains available when packed outputs are deliberately attached. The regular run defaults to BEiT-3 A at `configs/kaggle_beit3_a_t4x2.yaml`, disables W&B, exports all 214,354 validation predictions, runs the official VQA toolkit, and packages training artifacts. Select the historical ViLT run explicitly with `CONFIG_PATH=configs/kaggle_vilt_last6_t4x2.yaml`.

主训练 kernel 刻意不配置 `dataset_sources` 或 `kernel_sources`，避免等待 Kaggle 大型输入挂载后才启动 Python。如果没有完整本地数据挂载或手动附加的打包输出，runner 会通过可断点续传的 HTTPS 从 [COCO 官方下载站](https://cocodataset.org/#download) 下载 `train2014.zip` 和 `val2014.zip`，在 `/tmp` 解压、每个 split 解压完成后删除 zip，并在训练前校验精确图片数量。每个全新会话约需传输 20.2 GB；Kaggle 会话中断后可能需要重新下载。设置 `VQA_NETWORK_SMOKE_ONLY=1` 可有界检查两个官方下载端点和两张 CUDA GPU，并在数据下载、源码检出、依赖安装和训练前退出。若明确附加打包输出，仍可使用 `VQA_DATA_SMOKE_ONLY=1` 做数据与 CUDA 预检。正式运行默认使用 BEiT-3 A 配置 `configs/kaggle_beit3_a_t4x2.yaml`，禁用 W&B，导出全部 214,354 条验证预测、运行官方 VQA toolkit，并打包训练产物。历史 ViLT 任务可显式设置 `CONFIG_PATH=configs/kaggle_vilt_last6_t4x2.yaml`。

This Kaggle kernel targets GPU T4 x2 and defaults to requiring two visible CUDA GPUs; it launches DDP across both cards and fails before training if the session exposes fewer. Set `VQA_NUM_GPUS=1` to intentionally run the same code on one GPU, or `VQA_NUM_GPUS=auto` to use every visible GPU. The configured effective batch is preserved across GPU counts. Kaggle's kernel metadata uses `machine_shape: NvidiaTeslaT4` for the GPU T4 x2 accelerator ([Kaggle CLI metadata reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)).

该 Kaggle kernel 面向 GPU T4 x2，默认要求检测到两张 CUDA GPU，并在双卡上启动 DDP；如果会话实际只分配到一张卡，会在训练前明确失败。可设置 `VQA_NUM_GPUS=1` 明确使用单卡，或设为 `VQA_NUM_GPUS=auto` 使用所有可见 GPU。不同卡数下都会尽量保持配置的有效 batch 不变。Kaggle kernel metadata 通过 `machine_shape: NvidiaTeslaT4` 选择 GPU T4 x2 加速器（见 [Kaggle CLI metadata reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)）。

The script writes timestamped stage and child-process output to Kaggle's live log and to `runner.log` plus `runner_status.json` under the run checkpoint directory. It emits a heartbeat every 60 seconds during quiet subprocesses, and Python subprocesses run unbuffered. If a Kaggle session is marked running but the first `bootstrap` line never appears, the source script has not produced evidence that it reached its entrypoint; check Kaggle's Active Events/session allocation before diagnosing the model or DataLoader.

脚本会把带时间戳的阶段状态和子进程输出写入 Kaggle 实时日志，并保存到 checkpoint 目录下的 `runner.log` 与 `runner_status.json`。子进程安静超过 60 秒时会输出心跳，Python 子进程使用无缓冲模式。如果 Kaggle 显示运行中，却连第一条 `bootstrap` 都没有，当前证据不能说明训练代码已进入入口；应先检查 Kaggle 的 Active Events 和会话资源分配，再判断模型或 DataLoader 是否卡住。

The runner reuses Kaggle's preinstalled `torch` and `torchvision` only after a real CUDA tensor operation succeeds on the assigned GPU. This catches architecture mismatches such as a P100 (`sm_60`) paired with a runtime built only for `sm_70` and newer. When the probe fails, the pinned stack is installed into an isolated working directory and activated through `PYTHONPATH`, leaving Kaggle's system packages untouched. Set `FORCE_TORCH_INSTALL=1` to force this fallback; its location and versions can be overridden with `PYTORCH_RUNTIME_DIR`, `TORCH_VERSION`, `TORCHVISION_VERSION`, and `PYTORCH_INDEX_URL`.

runner 仅在真实 CUDA 张量运算可在已分配 GPU 上成功执行时，才复用 Kaggle 预装的 `torch` 和 `torchvision`。这能识别 P100（`sm_60`）与仅支持 `sm_70` 及更新架构的运行时不兼容等问题。探测失败时，固定版本会安装到隔离的工作目录并通过 `PYTHONPATH` 启用，不修改 Kaggle 系统包。设置 `FORCE_TORCH_INSTALL=1` 可强制使用该回退；可通过 `PYTORCH_RUNTIME_DIR`、`TORCH_VERSION`、`TORCHVISION_VERSION` 和 `PYTORCH_INDEX_URL` 覆盖目录与版本。

## Controlled Second Run / 受控第二轮

After Run 1, generate the only permitted follow-up config from its history:

首轮结束后，根据训练历史生成唯一允许的第二轮配置：

```bash
python scripts/select_vilt_followup.py \
  --history /kaggle/working/multimodal-vqa/vilt/training_history.csv \
  --output-config /kaggle/working/kaggle_vilt_followup.yaml
```

The selector prioritizes instability and a train/validation VQA gap above 0.10 before considering seed-stability replication. For the recorded seed-42 run, the widening gap selects one controlled follow-up that trains only the final six ViLT layers. Seed replication is used only for stable runs and is not presented as an expected score-improvement method. No third full run is planned.

选择器优先处理训练不稳定和训练/验证 VQA 差距超过 0.10，再考虑 seed 稳定性复现。根据已记录的 seed-42 结果，扩大的差距会选择仅训练 ViLT 最后 6 层这一项受控后续实验。只有运行稳定时才进行 seed 复现，且不将其描述为预期提分方法。计划中不进行第三次全量训练。

## Official Evaluation / 官方评估

```bash
git clone https://github.com/GT-Vision-Lab/VQA.git /kaggle/working/VQA
python scripts/prepare_official_vqa_toolkit.py --toolkit-root /kaggle/working/VQA

python evaluate.py \
  --config configs/kaggle_finetune.yaml \
  --checkpoint /kaggle/working/multimodal-vqa/finetune/best.pt \
  --device cuda \
  --data-root /kaggle/input/<dataset>/vqa \
  --predictions-output /kaggle/working/multimodal-vqa/finetune/val_predictions.json

python scripts/run_official_vqa_eval.py \
  --toolkit-root /kaggle/working/VQA \
  --questions /kaggle/input/<dataset>/vqa/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations /kaggle/input/<dataset>/vqa/v2_mscoco_val2014_annotations.json \
  --predictions /kaggle/working/multimodal-vqa/finetune/val_predictions.json \
  --output /kaggle/working/multimodal-vqa/finetune/official_vqa_metrics.json
```
