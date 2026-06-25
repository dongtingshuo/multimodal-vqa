# Kaggle Training / Kaggle 训练

This workflow uses one training engine for local CPU, local CUDA, Apple MPS, and Kaggle CUDA. Kaggle-specific behavior is limited to input/output paths, resumable checkpoints, and the staged fine-tuning config.

本流程在本地 CPU、本地 CUDA、Apple MPS 和 Kaggle CUDA 之间共用同一训练引擎。Kaggle 专用部分仅包括输入输出路径、断点权重和分阶段微调配置。

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
  --sample-images 20
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
```

## Resume / 断点续训

`latest.pt` is written atomically after every epoch. Reattach the previous Notebook output as an Input or continue in the same saved session, then run:

`latest.pt` 在每个 epoch 结束后原子写入。将上一版 Notebook 输出重新挂载为 Input，或在同一保存会话中执行：

```bash
python train.py \
  --config configs/kaggle_finetune.yaml \
  --device cuda \
  --data-root /kaggle/input/<dataset>/vqa \
  --answer-vocab-path /kaggle/working/answer_vocab.json \
  --checkpoint-dir /kaggle/working/multimodal-vqa/finetune \
  --resume /kaggle/working/multimodal-vqa/finetune/latest.pt
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

The script uses the public Kaggle dataset `sagnikkayalcse52/coco2014vqa` as the COCO image source by default, downloads the official VQA v2 question/annotation JSON files at runtime, and normalizes everything into the internal VQA directory structure under `/kaggle/working/multimodal-vqa/vqa`.

脚本默认将公开 Kaggle 数据集 `sagnikkayalcse52/coco2014vqa` 作为 COCO 图片源，在运行时下载官方 VQA v2 questions/annotations JSON，并在 `/kaggle/working/multimodal-vqa/vqa` 下自动规范化为项目内部训练所需的 VQA 目录结构。

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
