# Troubleshooting / 故障排查

## Missing COCO Image / 缺少 COCO 图片

Error:

错误：

```text
FileNotFoundError: data/vqa/train2014/COCO_train2014_xxxxxxxxxxxx.jpg
```

Cause:

原因：

The annotation file references an image that does not exist locally. This usually means only a subset of COCO images was downloaded while the config is using full training.

标注文件引用了本地不存在的图片。常见原因是只下载了 COCO 子集，但配置正在运行全量训练。

Fix:

修复：

```bash
python scripts/prepare_vqa_data.py --root data/vqa --full-coco-images --answer-vocab-size 3000
```

## COCO Download Stays at 0B / COCO 下载长期停在 0B

The official COCO image archives are large and may be slow depending on network routing.

COCO 官方图片压缩包体积较大，网络路由不佳时可能长时间无速度。

Recommended options:

推荐处理方式：

- Download `train2014.zip` and `val2014.zip` manually with a download manager.
- Put them under `data/vqa/downloads/`.
- Re-run the data preparation command.

```text
data/vqa/downloads/train2014.zip
data/vqa/downloads/val2014.zip
```

Interrupted automatic downloads remain as `.part` files and resume on the next run. Completed archives are checked by expected size and checksum before extraction; ZIP CRC is checked during extraction.

自动下载中断后会保留 `.part` 文件，下次运行时继续下载。完整压缩包在解压前会检查预期大小和校验和，解压时还会检查 ZIP CRC。

## Training Progress Stays at 0% / 训练进度长期停在 0%

Possible causes:

可能原因：

- DataLoader workers are still preparing the first batch.
- Windows multiprocessing is copying a large dataset object.
- Disk I/O is slow while reading the first batch of images.

Quick diagnostic config:

快速诊断配置：

```yaml
data:
  num_workers: 0

train:
  batch_size: 4
  persistent_workers: false
```

If training starts moving with this config, the issue is DataLoader startup pressure rather than model computation.

如果这样配置后训练开始推进，问题通常是 DataLoader 启动压力，而不是模型计算本身。

## Gradio Shows Model Not Loaded / Gradio 显示模型未加载

Possible causes:

可能原因：

- `checkpoints/best.pt` does not exist.
- Hugging Face model files are not cached and the environment is offline.
- The demo was launched with `--offline` before tokenizer/model files were cached.
- A legacy checkpoint does not contain architecture metadata, or the file is incomplete.

Fix:

修复：

```bash
pip install -r requirements.txt
python scripts/download_checkpoint.py
python demo.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

Current checkpoints are self-describing: model architecture and preprocessing settings come from checkpoint metadata, while local paths and device settings come from the runtime config.

当前 checkpoint 为自描述格式：模型架构和预处理参数来自权重元数据，本地路径和设备设置来自运行时配置。

Do not use `--offline` unless the required Hugging Face files are already cached.

除非所需 Hugging Face 文件已经缓存，否则不要使用 `--offline`。

## CUDA Configured but Not Available / 配置了 CUDA 但当前不可用

The demo, evaluation, and inference entrypoints automatically fall back to CPU when CUDA is unavailable.

当 CUDA 不可用时，演示、评估和推理入口会自动回退到 CPU。

Training remains GPU-first in `configs/default.yaml`. If you need CPU-only training, use:

训练配置默认 GPU 优先。如果需要 CPU 连通性测试，请使用：

```bash
python train.py --config configs/demo_cpu.yaml
```

## Gradio Port Is Busy / Gradio 端口被占用

The demo starts from port `8877` and automatically tries later ports if needed.

演示程序从 `8877` 开始尝试端口，如果被占用会自动尝试后续端口。

You can also specify a port:

也可以手动指定端口：

```bash
python demo.py --server-port 8890
```

## Dependency Drift / 依赖版本漂移

If the Gradio page fails with schema or Pydantic errors, reinstall pinned dependencies:

如果 Gradio 页面出现 schema 或 Pydantic 相关错误，请重新安装锁定依赖：

```bash
pip install -r requirements.txt
```
