# 基于多模态融合的视觉问答（VQA）

这是一个用于人工智能实战课程设计的 VQA 工程。模型输入一张图像和一个英文自然语言问题，输出候选答案概率。默认训练方案已调整为 **Windows + NVIDIA GPU + VQA v2.0 全量训练**：`ResNet-50 + DistilBERT + 双向 Cross Attention + 答案分类器`。

## 功能

- 支持 VQA v2.0 train/val 全量数据读取。
- 默认构建 Top-3000 高频答案词表。
- 使用 ResNet-50 提取图像区域特征。
- 使用 DistilBERT 提取问题 token 特征。
- 使用双向 Cross Attention 完成图像和文本融合。
- 支持 CUDA、AMP 混合精度、DataLoader 多进程读取。
- 提供训练、评估、命令行推理和 Gradio 演示界面。

## Windows 环境

建议使用 Anaconda/Miniconda，在 Windows 的 Anaconda Prompt 或 PyCharm Terminal 中执行：

```bat
conda activate pytorch
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果第一行不是 `True`，说明当前 PyTorch 不是 CUDA 版本，需要先安装与你显卡驱动匹配的 CUDA 版 PyTorch。

安装项目依赖：

```bat
pip install -r requirements.txt
```

## 全量数据准备

默认数据根目录为：

```text
data/vqa
```

全量训练需要 VQA v2.0 标注/问题文件，以及完整 COCO 2014 train/val 图片。可直接运行：

```bat
python scripts/prepare_vqa_data.py --root data/vqa --full-coco-images --answer-vocab-size 3000
```

该命令会下载并解压：

```text
data/vqa/train2014/
data/vqa/val2014/
data/vqa/v2_OpenEnded_mscoco_train2014_questions.json
data/vqa/v2_mscoco_train2014_annotations.json
data/vqa/v2_OpenEnded_mscoco_val2014_questions.json
data/vqa/v2_mscoco_val2014_annotations.json
data/answer_vocab.json
```

完整 COCO 图片体积很大，下载和解压需要较长时间，并需要几十 GB 磁盘空间。

## 默认全量 GPU 训练

默认配置文件是：

```text
configs/default.yaml
```

默认关键参数：

```yaml
device: cuda
answer_vocab_size: 3000
max_train_samples: null
max_val_samples: null
batch_size: 32
epochs: 10
use_amp: true
num_workers: 4
```

开始训练：

```bat
python train.py --config configs/default.yaml
```

训练会保存最佳模型到：

```text
checkpoints/best.pt
```

如果显存不足，优先把 `configs/default.yaml` 中的 `train.batch_size` 从 `32` 改成 `16` 或 `8`。如果你的 Windows 机器 DataLoader 卡住，可把 `data.num_workers` 改成 `0`。

## 评估

```bat
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## 命令行推理

```bat
python infer.py --config configs/default.yaml --checkpoint checkpoints/best.pt --image data/vqa/val2014/COCO_val2014_000000000042.jpg --question "What is in the image?" --topk 5
```

## Gradio 演示

```bat
python demo.py --config configs/default.yaml --checkpoint checkpoints/best.pt --server-port 8877
```

浏览器打开：

```text
http://127.0.0.1:8877
```

## 小样本快速测试

如果只想快速检查工程是否能跑通，可以使用：

```bat
python train.py --config configs/demo_cpu.yaml
```

`configs/demo_cpu.yaml` 是 CPU/小样本配置，不影响默认全量 GPU 训练配置。
