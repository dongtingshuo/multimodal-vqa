# AutoDL ViLT Continuation / AutoDL ViLT 续训

This directory provides the maintained single-GPU AutoDL workflow for continuing the seed-42 ViLT run from epoch 3 on an RTX 4090D/4090.

本目录提供维护中的单卡 AutoDL 工作流，用于在 RTX 4090D/4090 上从 epoch 3 继续 seed 42 ViLT 训练。

The upload archive is built with `scripts/build_autodl_bundle.py`. It records the exact Git commit and SHA256 digest of every packaged resume artifact.

上传包由 `scripts/build_autodl_bundle.py` 构建，并记录准确 Git commit 与每个续训产物的 SHA256。

## Supported Instance / 推荐实例

- 1 x RTX 4090D or RTX 4090, 24GB
- 16 CPU cores and at least 32GB RAM
- 150GB or larger data disk
- AutoDL PyTorch 2.5.1, Python 3.12, CUDA 12.4 base image

Do not change the model, preprocessing, batch size, gradient accumulation, optimizer rates, scheduler, or fine-tuning fields when resuming. Runtime paths and DataLoader workers may differ.

续训时不要修改模型、预处理、batch size、梯度累积、优化器学习率、调度器或微调字段。运行路径和 DataLoader worker 可以调整。

## Package Layout / 上传包结构

```text
multimodal-vqa-autodl/
├── repo/                       # Source snapshot / 代码快照
└── resume/
    ├── answer_vocab.json
    └── vilt-seed42/
        ├── latest.pt           # Format-v3 epoch-2 training state
        ├── training_history.csv
        ├── config.snapshot.json
        └── run_metadata.json
```

`setup.sh` copies `latest.pt` to the working run directory and initializes `best.pt` from the same epoch-2 state. The package intentionally contains one large checkpoint file to reduce upload size.

`setup.sh` 会将 `latest.pt` 复制到工作目录，并使用同一 epoch 2 状态初始化 `best.pt`。上传包只保留一份大型 checkpoint，以减少上传量。

## Run Order / 执行顺序

After extracting the package under `/root/autodl-tmp`:

将上传包解压到 `/root/autodl-tmp` 后执行：

```bash
cd /root/autodl-tmp/multimodal-vqa-autodl/repo
bash autodl/setup.sh
bash autodl/prepare_data.sh
bash autodl/start_resume.sh
```

Monitor training / 查看训练：

```bash
bash autodl/status.sh
```

The background pipeline validates the full dataset, resumes from epoch 3, evaluates the best checkpoint, exports validation predictions, and creates a result archive under `artifacts/`.

后台流程会验证完整数据集、从 epoch 3 续训、评估最佳 checkpoint、导出验证预测，并在 `artifacts/` 下生成结果压缩包。

## Billing Safety / 计费注意

Run `setup.sh` immediately after startup, then start data preparation. AutoDL charges while the instance is running. After the result archive is downloaded and verified, shut down the instance. Paid data-disk capacity may continue billing while the instance is off.

实例开机后应立即执行 `setup.sh` 并准备数据。AutoDL 在实例运行期间计费。下载并确认结果压缩包后应关机；付费数据盘在关机后仍可能继续计费。
