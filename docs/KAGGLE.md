# Kaggle Training / Kaggle 训练

This workflow uses one training engine for local CPU, local CUDA, Apple MPS, and Kaggle CUDA. The recommended full experiment uses ViLT on Kaggle with strict data validation, epoch-boundary resume, online W&B tracking, prediction export, and official VQA evaluation.

本流程在本地 CPU、本地 CUDA、Apple MPS 和 Kaggle CUDA 之间共用同一训练引擎。推荐的全量实验在 Kaggle 上训练 ViLT，并执行严格数据校验、epoch 边界续训、W&B 在线监控、预测导出和官方 VQA 评估。

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

# Recommended Kaggle ViLT experiment with required online W&B tracking
python train.py \
  --config configs/kaggle_vilt.yaml \
  --device cuda \
  --data-root /kaggle/input/<dataset>/vqa \
  --answer-vocab-path /kaggle/working/answer_vocab.json \
  --checkpoint-dir /kaggle/working/multimodal-vqa/vilt \
  --wandb
```

Create a Kaggle Secret named `WANDB_API_KEY` and enable it for the notebook. `configs/kaggle_vilt.yaml` marks W&B as required: training stops before the first epoch if the secret cannot be read or verified. A successful launch prints `W&B run: <URL>`; that URL opens the live loss, accuracy, VQA score, learning-rate, stage, and timing charts. `WANDB_ENTITY` is optional. Checkpoints are not uploaded to W&B.

请在 Kaggle Secrets 中创建 `WANDB_API_KEY` 并为当前 notebook 启用。`configs/kaggle_vilt.yaml` 将 W&B 设为必需：如果密钥无法读取或在线验证，训练会在第一个 epoch 前停止。成功启动后日志会输出 `W&B run: <URL>`，打开该链接即可在线查看 loss、accuracy、VQA score、各参数组学习率、训练阶段和耗时。`WANDB_ENTITY` 可选，checkpoint 默认不上传 W&B。

## Resume / 断点续训

`latest.pt` is written atomically after every epoch. Reattach the previous Notebook output as an Input or continue in the same saved session, then run:

`latest.pt` 在每个 epoch 结束后原子写入。将上一版 Notebook 输出重新挂载为 Input，或在同一保存会话中执行：

The maintained Kaggle script also supports a private checkpoint Dataset mounted at
`/kaggle/input/multimodal-vqa-vilt-resume`. When present, it copies `latest.pt`,
`best.pt`, the answer vocabulary, and run history into `/kaggle/working` before training,
then resumes automatically. Kaggle checkpoints are saved at epoch boundaries, so an
interrupted partial epoch is repeated.

仓库内维护的 Kaggle 脚本还支持挂载到
`/kaggle/input/multimodal-vqa-vilt-resume` 的私有 checkpoint Dataset。如果存在，
脚本会在训练前将 `latest.pt`、`best.pt`、答案词表和训练历史复制到
`/kaggle/working`，并自动续训。Kaggle checkpoint 按 epoch 边界保存，因此中断时
未完成的 epoch 会重新训练。

```bash
python train.py \
  --config configs/kaggle_vilt.yaml \
  --device cuda \
  --data-root /kaggle/input/<dataset>/vqa \
  --answer-vocab-path /kaggle/working/answer_vocab.json \
  --checkpoint-dir /kaggle/working/multimodal-vqa/vilt \
  --resume /kaggle/working/multimodal-vqa/vilt/latest.pt \
  --wandb
```

Resume permits different local paths, device, workers, logging location, and a larger total epoch count. Model, preprocessing, effective batch, optimizer rates, and fine-tune schedule must match.

断点续训允许更换本地路径、设备、worker、日志目录和更大的总 epoch 数。模型、预处理、有效 batch、优化器学习率和微调时序必须一致。

## Notebook / Notebook 模板

Import [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb). Set `DATA_ROOT`, `CONFIG_PATH`, `RUN_NAME`, `TOTAL_EPOCHS`, and `GIT_REF` in the first code cell. Use a release tag or commit SHA for a final reproducible run.

导入 [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb)。在首个代码单元设置 `DATA_ROOT`、`CONFIG_PATH`、`RUN_NAME`、`TOTAL_EPOCHS` 和 `GIT_REF`。正式训练应锁定 Release tag 或 commit SHA。

## Kaggle API Kernel / Kaggle API 远程任务

The repository also includes a script kernel in [`kaggle_finetune_kernel/`](../kaggle_finetune_kernel). It can be pushed from a local machine with Kaggle API credentials:

仓库也提供了 [`kaggle_finetune_kernel/`](../kaggle_finetune_kernel) 脚本任务目录，可在本机通过 Kaggle API 直接推送运行：

```bash
kaggle kernels push -p kaggle_finetune_kernel
```

The script uses `sagnikkayalcse52/coco2014vqa` as its initial COCO image source, downloads official VQA v2 JSON files, repairs any referenced validation images missing from that mirror using the official COCO image host, and then enforces the official train/val counts. It runs `configs/kaggle_vilt.yaml`, verifies W&B before training, exports all 214,354 validation predictions, runs the official VQA toolkit, and packages the artifacts.

脚本将 `sagnikkayalcse52/coco2014vqa` 作为初始 COCO 图片源，下载官方 VQA v2 JSON；若镜像缺少验证集引用图片，则从 COCO 官方图片地址补齐，随后按官方 train/val 数量执行严格校验。任务默认运行 `configs/kaggle_vilt.yaml`，训练前验证 W&B，导出全部 214,354 条验证预测，运行官方 VQA toolkit，并打包所有产物。

The runner reuses Kaggle's preinstalled `torch` and `torchvision` when both import correctly and CUDA is available. This avoids replacing several gigabytes of compatible GPU packages on every launch. Set `FORCE_TORCH_INSTALL=1` only when the environment probe fails or a pinned fallback is required; the fallback versions can be overridden with `TORCH_VERSION`, `TORCHVISION_VERSION`, and `PYTORCH_INDEX_URL`.

当 Kaggle 预装的 `torch`、`torchvision` 可正常导入且 CUDA 可用时，runner 会直接复用该环境，避免每次启动都替换数 GB 的兼容 GPU 依赖。仅在环境探测失败或必须使用固定版本回退时设置 `FORCE_TORCH_INSTALL=1`；可通过 `TORCH_VERSION`、`TORCHVISION_VERSION` 和 `PYTORCH_INDEX_URL` 覆盖回退版本。

## Controlled Second Run / 受控第二轮

After Run 1, generate the only permitted follow-up config from its history:

首轮结束后，根据训练历史生成唯一允许的第二轮配置：

```bash
python scripts/select_vilt_followup.py \
  --history /kaggle/working/multimodal-vqa/vilt/training_history.csv \
  --output-config /kaggle/working/kaggle_vilt_followup.yaml
```

The selector chooses one predeclared branch: seed replication after meeting both gates, last-six-layer tuning for a train/validation gap above 0.10, backbone LR `1e-5` for repeated instability, or backbone LR `3e-5` for stable under-optimization. No third full run is planned.

选择器只会进入预先定义的一条分支：双指标达标后更换 seed 复现；训练/验证差距超过 0.10 时仅训练最后 6 层；重复不稳定时将 backbone LR 降至 `1e-5`；稳定但优化不足时将其升至 `3e-5`。计划中不进行第三次全量训练。

## Official Evaluation / 官方评估

```bash
git clone https://github.com/GT-Vision-Lab/VQA.git /kaggle/working/VQA

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
