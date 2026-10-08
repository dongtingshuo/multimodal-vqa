# Project Context

## Goal / 目标

Maintain a reproducible PyTorch visual question answering system for VQA v2.0 and COCO 2014, including training, evaluation, inference, and a Gradio demo.

维护面向 VQA v2.0 与 COCO 2014 的可复现 PyTorch 视觉问答系统，覆盖训练、评估、推理和 Gradio 演示。

## Scope / 范围

- In scope: multimodal model variants, data validation, single- and multi-GPU training, resumable checkpoints, official VQA evaluation, experiment summaries, and inference/demo workflows.
- Out of scope: committing COCO data or large checkpoints to Git; W&B is optional and disabled in the current Kaggle runner.

## Stack / 技术栈

- Python 3.10+; PyTorch/torchvision; Transformers; Pillow; PyYAML; Gradio 6.x; pytest and Ruff for development.
- Package metadata and dependency ranges are maintained in `pyproject.toml`, `requirements.txt`, and `requirements-dev.txt`.

## Architecture and Entry Points / 架构与入口

- `vqa_project/model.py`: model factory and multimodal architectures (`vilt`, text/image baselines, concat, and cross-attention variants).
- `vqa_project/data.py`, `answers.py`: VQA dataset, batching, answer normalization, and vocabulary.
- `vqa_project/engine.py`, `training.py`, `tracking.py`: train/evaluate loops, staged fine-tuning, checkpoint/resume, histories, curves, and run metadata.
- `train.py`, `evaluate.py`, `infer.py`, `demo.py`: user-facing training, evaluation, CLI inference, and Gradio entry points.
- `scripts/`: dataset preparation/validation, official evaluation adapter, experiment summaries, and error analysis.
- `kaggle_finetune_kernel/run_kaggle_finetune.py`: Kaggle orchestration; `autodl/`: AutoDL continuation workflow.

## Data and Models / 数据与模型

- Training data: VQA v2.0 questions/annotations plus COCO 2014 `train2014` and `val2014` images. Keep datasets outside Git.
- Kaggle COCO sources: when no complete image mount is available, the runner downloads the official `train2014.zip` and `val2014.zip` from COCO's S3 host into `/tmp`; the download is resumable within a running session.
- Optional private pack kernels: `dongtingshuo/multimodal-vqa-coco-train-pack` and `dongtingshuo/multimodal-vqa-coco-val-pack`. Their outputs are not attached to the main training kernel because Kaggle worker-side input mounts proved intermittent.
- Recommended published checkpoint: ViLT seed 42, release `v0.3.0`; see `README.md` and `docs/evaluation/` for metrics and provenance.

## Runtime and Tests / 运行与测试

- Local training: `python train.py --config configs/default.yaml --device auto`.
- Kaggle VQA fine-tuning: `kaggle_finetune_kernel/kernel-metadata.json` selects the private `multimodal-vqa-finetune` kernel, dual-T4 accelerator, and private epoch-4 resume-checkpoint dataset; the runner downloads COCO images when needed.
- Run tests with `pytest`; use `configs/smoke_train.yaml` or `configs/smoke_vilt.yaml` for bounded workflow checks.
- Validate real data before training with `python scripts/validate_vqa_data.py --root <data-root> --sample-images 20 --strict-full`.

## Current Status and Decisions / 当前状态与决策

- Directly attaching the raw COCO Dataset and attaching packed Notebook outputs both produced intermittent worker-side mount failures before Python startup. The main training kernel therefore has no `dataset_sources` or `kernel_sources`.
- The optional train and validation pack kernels completed successfully. Their archives contain 82,783 train images (13,889,986,198 source bytes) and 40,458 validation images (6,821,605,333 source bytes), plus VQA JSON files. A full packaged-input smoke also passed once on two T4s, but a later main-kernel attempt failed mounting the validation output after 30 retries; do not rely on this path as the default.
- The main runner now falls back to official COCO S3 ZIPs when neither a complete mounted dataset nor packed outputs are present. Downloads use HTTPS Range requests, retain partial files across retryable network errors, log byte progress, extract each split under `/tmp`, remove the ZIP after successful extraction, and validate the exact 82,783/40,504 image counts.
- Kaggle main kernel Version 31 ran a network-only smoke with no data sources attached: Python bootstrap logs appeared immediately, both Tesla T4 GPUs were visible and passed CUDA tensor operations, and COCO HEAD requests returned 200 for train (13,510,573,713 bytes) and validation (6,645,013,297 bytes) archives. No image download or training occurred.
- Production mode was restored and saved as Kaggle Version 32 without running. Its pulled remote metadata has empty `dataset_sources` and `kernel_sources`, GPU enabled, and `machine_shape: NvidiaTeslaT4`.
- Kaggle Version 33 cloned GitHub commit `35c4d92`, used both Tesla T4 GPUs, and strictly validated the complete COCO/VQA data (82,783 train images, 40,504 validation images; 443,757 train and 214,354 validation annotations). Kaggle stopped the run at its maximum execution duration after four completed epochs. The format-v3 `latest.pt` and `best.pt` were retrieved and verified at epoch 4: validation accuracy 0.6084, VQA score 0.7067, top-5 VQA score 0.8897; `global_step=54,332`. The resume signature matches `configs/kaggle_vilt_last6_t4x2.yaml`.
- The private Kaggle resume dataset `dongtingshuo/multimodal-vqa-vilt-last6-t4x2-resume` is ready (version 1) and contains `latest.pt`, `best.pt`, `README.md`, and `SHA256SUMS`; remote checkpoint sizes match the locally verified files. Kernel metadata attaches this dataset. The runner searches direct and owner-scoped Kaggle input mount layouts, and the next run must restore `latest.pt` and begin at epoch 5; do not restart from pretrained weights.
- Kaggle Version 34 is running from GitHub commit `37d4d50`. Its logs confirm both Tesla T4 GPUs, restored the private epoch-4 `latest.pt`, validated all COCO/VQA counts, and began `epoch=5/10`; no Version 34 validation metrics are available yet. Continue monitoring for the 12-hour limit and resume again from the newly saved `latest.pt` if required.
- The direct-download path transfers about 20.2 GB per fresh session. Partial HTTP downloads can resume within the current session; a new Kaggle session may need to download the archives again.
- The P100 ViLT checkpoint uses a different all-layer configuration, and the `strong_cross_attention` checkpoint is a different architecture; neither is used for this resume. Version 33's epoch-4 checkpoint matches the current ViLT last-six-layer T4x2 resume signature.
- Local verification: Ruff passed; the full suite passed 88 tests in the existing `pytorch` environment (Python 3.9, below the declared minimum); 5 upstream deprecation warnings remain. `git diff --check` passed.
- Continue the epoch-4 checkpoint through epoch 10, then verify the official VQA evaluation and retrieve the output package.
- Keep existing untracked `artifacts/` user data; do not clean it as part of routine work.

## Prioritized TODO / 优先事项

1. Monitor Kaggle Version 34 through completion or its execution limit; if interrupted, retrieve the newest compatible `latest.pt` and resume without restarting.
2. Continue through epoch 10, then verify the official VQA evaluation and retrieve the output package before updating published model claims.
3. If repeated session restarts make direct downloads too costly, evaluate a resumable COCO data source that avoids transferring the full archives every session.
4. Keep model-card and evaluation claims tied to checked-in reports and actual checkpoint provenance.
