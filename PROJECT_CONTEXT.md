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
- `vqa_project/beit3.py`, `vqa_project/vqa_normalization.py`: optional BEiT-3 Base model, pinned/checksummed assets, dedicated tokenizer, and VQA-v2 normalization/scoring.
- `vqa_project/data.py`, `answers.py`: VQA dataset, batching, answer normalization, and vocabulary.
- `vqa_project/engine.py`, `training.py`, `tracking.py`: train/evaluate loops, staged fine-tuning, checkpoint/resume, histories, curves, and run metadata.
- `train.py`, `evaluate.py`, `infer.py`, `demo.py`: user-facing training, evaluation, CLI inference, and Gradio entry points.
- `scripts/`: dataset preparation/validation, official evaluation adapter, experiment summaries, and error analysis.
- BEiT-3 preparation and hardware gates: `scripts/prepare_visual_genome_qa.py`, `scripts/create_beit3_replication_config.py`, `scripts/benchmark_beit3_batch.py`, `scripts/prepare_beit3_runtime_config.py`, `scripts/smoke_beit3_distributed.py`, and `scripts/estimate_beit3_runtime.py`.
- `kaggle_finetune_kernel/run_kaggle_finetune.py`: Kaggle orchestration; `autodl/`: AutoDL continuation workflow.

## Data and Models / 数据与模型

- Training data: VQA v2.0 questions/annotations plus COCO 2014 `train2014` and `val2014` images. Keep datasets outside Git.
- Kaggle COCO sources: when no complete image mount is available, the runner downloads the official `train2014.zip` and `val2014.zip` from COCO's S3 host into `/tmp`; the download is resumable within a running session.
- Optional private pack kernels: `dongtingshuo/multimodal-vqa-coco-train-pack` and `dongtingshuo/multimodal-vqa-coco-val-pack`. Their outputs are not attached to the main training kernel because Kaggle worker-side input mounts proved intermittent.
- Recommended published checkpoint: ViLT seed 42, release `v0.3.0`; see `README.md` and `docs/evaluation/` for metrics and provenance.
- Experimental candidate: BEiT-3 Base, generic pretrained weights only; `configs/kaggle_beit3_a_t4x2.yaml`, `configs/kaggle_beit3_b_vg_t4x2.yaml`, and `configs/kaggle_beit3_c_seed123_t4x2.yaml` define VQA-only A, screened Visual Genome B, and seed-123 replication C. No BEiT-3 training score exists yet.

## Runtime and Tests / 运行与测试

- Local training: `python train.py --config configs/default.yaml --device auto`.
- Kaggle training: the maintained runner defaults to BEiT-3 A on dual T4, selects and applies a safe micro-batch while preserving global batch 128, runs a real DDP save/restore smoke, and estimates runtime after strict data validation. The historical ViLT recipe remains selectable through `CONFIG_PATH`; the runner downloads COCO images when needed.
- BEiT-3 is an optional dependency extra (`pip install -e '.[beit3]'`); its two-T4 batch benchmark and save/resume smoke must pass on Kaggle before any full run.
- Run tests with `pytest`; use `configs/smoke_train.yaml` or `configs/smoke_vilt.yaml` for bounded workflow checks.
- Validate real data before training with `python scripts/validate_vqa_data.py --root <data-root> --sample-images 20 --strict-full`.

## Current Status and Decisions / 当前状态与决策

- Directly attaching the raw COCO Dataset and attaching packed Notebook outputs both produced intermittent worker-side mount failures before Python startup. The main training kernel therefore has no `dataset_sources` or `kernel_sources`.
- The optional train and validation pack kernels completed successfully. Their archives contain 82,783 train images (13,889,986,198 source bytes) and 40,458 validation images (6,821,605,333 source bytes), plus VQA JSON files. A full packaged-input smoke also passed once on two T4s, but a later main-kernel attempt failed mounting the validation output after 30 retries; do not rely on this path as the default.
- The main runner now falls back to official COCO S3 ZIPs when neither a complete mounted dataset nor packed outputs are present. Downloads use HTTPS Range requests, retain partial files across retryable network errors, log byte progress, extract each split under `/tmp`, remove the ZIP after successful extraction, and validate the exact 82,783/40,504 image counts.
- Kaggle main kernel Version 31 ran a network-only smoke with no data sources attached: Python bootstrap logs appeared immediately, both Tesla T4 GPUs were visible and passed CUDA tensor operations, and COCO HEAD requests returned 200 for train (13,510,573,713 bytes) and validation (6,645,013,297 bytes) archives. No image download or training occurred.
- Production mode was restored and saved as Kaggle Version 32 without running. Its pulled remote metadata has empty `dataset_sources` and `kernel_sources`, GPU enabled, and `machine_shape: NvidiaTeslaT4`.
- Kaggle Version 33 cloned GitHub commit `35c4d92`, used both Tesla T4 GPUs, and strictly validated the complete COCO/VQA data (82,783 train images, 40,504 validation images; 443,757 train and 214,354 validation annotations). Kaggle stopped the run at its maximum execution duration after four completed epochs. The format-v3 `latest.pt` and `best.pt` were retrieved and verified at epoch 4: validation accuracy 0.6084, VQA score 0.7067, top-5 VQA score 0.8897; `global_step=54,332`. The resume signature matches `configs/kaggle_vilt_last6_t4x2.yaml`.
- The private Kaggle resume dataset `dongtingshuo/multimodal-vqa-vilt-last6-t4x2-resume` remains at version 1 with the verified epoch-4 `latest.pt` and `best.pt`. Version 34's epoch-6 `latest.pt`, run summary, history, curves, and predictions were retrieved from Kaggle outputs; publish the epoch-6 checkpoint as a new private dataset version only if further training is explicitly justified.
- Kaggle Version 34 resumed the Version 33 epoch-4 checkpoint from Git commit `37d4d50`, completed epochs 5-6 on two Tesla T4 GPUs, and stopped by the configured patience-2 rule. Best metrics remained epoch 4 (accuracy 0.6084, internal VQA 0.7067, top-5 VQA 0.8897); epochs 5-6 improved training VQA while validation VQA declined to 0.7062 and 0.7057. The Kaggle kernel's ERROR state came from the unavailable `lib2to3` post-processing dependency, not from model training. The internal evaluator completed for 209,608 examples and exported predictions for all 214,354 validation questions. The upstream official VQA evaluator was converted with Fissix and run twice locally against the complete validation questions/annotations; both runs scored 68.08 overall, with matching question IDs. Version 34 is 0.34 points below the recommended checkpoint's official validation score of 68.42 and is not promoted. The downloaded V34 `best.pt` was 0 bytes, so the verified Version 33 epoch-4 checkpoint is the selected-weight recovery source; its SHA-256 is `947551915da46f6225f2574c217728b8125491608fa55a54aabd8625fd5561c5`. V34 `latest.pt` is epoch 6 and is not the best model. Full official metrics are tracked in `docs/evaluation/kaggle-v34-official-vqa-metrics.json` and the experiment report.
- The direct-download path transfers about 20.2 GB per fresh session. Partial HTTP downloads can resume within the current session; a new Kaggle session may need to download the archives again.
- The P100 ViLT checkpoint uses a different all-layer configuration, and the `strong_cross_attention` checkpoint is a different architecture; neither is used for this resume. Version 33's epoch-4 checkpoint matches the current ViLT last-six-layer T4x2 resume signature.
- BEiT-3 Base integration now uses the generic upstream checkpoint and SentencePiece tokenizer with pinned TorchScale source; both files are SHA-256 verified and pretrained parameter loading is strict. A local CPU integration check loaded all 12/12 encoder layers and verified finite forward output and 480 px FP32 position interpolation. This is implementation evidence only, not a training result.
- BEiT-3 A/B/C configs, leakage-screened VG manifest generation, per-epoch deterministic sampling of at most 25% VG examples in B, C-config generation from the winner, per-update checkpointing, automatic micro-batch config generation, data-sized runtime estimation, and dual-GPU smoke/benchmark entrypoints are implemented. Resume artifacts retain the selected runtime config, benchmark report, and screened VG manifest. Kaggle Version 35 exposed a runner import-path error, fixed by explicitly adding the checkout to `sys.path`. Version 36 exposed skipped AMP optimizer updates; initial scale `1024`, bounded retries, and successful-update-only global-step/scheduler accounting fixed that fault. Version 37 confirmed both T4s, selected per-GPU batch size 8 in the capacity benchmark, and completed two real optimizer updates on both ranks. Version 37's CPU/CUDA state-comparison failure was fixed. Version 38 passed that comparison but found the smoke save call omitted the AMP scaler, so restore validation failed; the smoke now saves and requires the scaler state. Version 38 did not proceed to dataset preparation or full training. The dual-GPU save/restore gate and real-data training smoke must pass before A training is allowed. Keep ViLT v0.3.0 and its 68.42 official validation result unchanged as the recommendation.
- Latest local verification: 114 tests passed in the existing `pytorch` environment (Python 3.9, below the declared Python 3.10 minimum); 5 upstream deprecation warnings remain. Ruff on changed Python files, Python compilation, and `git diff --check` passed. The wheel built successfully earlier with system Python 3.14 and isolated build dependencies. Kaggle's next version must pass the dual-T4 save/restore smoke, strict full-data validation, and real-data training smoke before the A full run is allowed.
- Do not resume the same run under the current early-stopping configuration: epoch 6 is a completed early-stop state and validation did not improve. Keep the recommended checkpoint unchanged unless a controlled experiment exceeds its official validation score and internal metrics; document any revised training strategy before launching another run.
- Keep existing untracked `artifacts/` user data; do not clean it as part of routine work.

## Prioritized TODO / 优先事项

1. Run the BEiT-3 batch benchmark and two-GPU save/resume smoke on the actual Kaggle T4 x2 runtime; do not start a full run until both pass and the complete data preflight succeeds.
2. Execute only the predeclared A/B/C experiments, preserve each run separately, and evaluate all 214,354 validation predictions with the same official VQA toolkit.
3. Keep ViLT `v0.3.0` recommended unless the selected BEiT-3 candidate and independent seed replication satisfy the documented promotion gates; do not resume ViLT Version 34 unchanged.
