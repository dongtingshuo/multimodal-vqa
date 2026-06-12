# 基于多模态融合的视觉问答（VQA）

这是一个用于人工智能实战课程设计的 VQA 工程。模型输入一张图像和一个英文自然语言问题，输出候选答案概率。整体方案为 `ResNet-50 + DistilBERT + 双向 Cross Attention + 答案分类器`，并提供 Gradio 前端演示。

## 功能

- 支持 VQA v2.0 格式数据读取。
- 自动构建 Top-K 高频答案词表。
- 使用 ResNet-50 提取图像区域特征。
- 使用 DistilBERT 提取问题 token 特征。
- 使用双向 Cross Attention 完成图像和文本融合。
- 提供训练、评估、命令行推理和 Gradio 演示界面。

## 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果在国内网络环境下载 Hugging Face 或 PyTorch 依赖较慢，可以提前配置镜像，或在 Colab/Kaggle 上运行训练。

## 数据准备

默认数据根目录为 `data/vqa`。建议按以下结构放置 VQA v2.0 与 COCO 图片：

```text
data/vqa/
  train2014/
    COCO_train2014_000000000009.jpg
    ...
  val2014/
    COCO_val2014_000000000042.jpg
    ...
  v2_OpenEnded_mscoco_train2014_questions.json
  v2_mscoco_train2014_annotations.json
  v2_OpenEnded_mscoco_val2014_questions.json
  v2_mscoco_val2014_annotations.json
```

本项目默认只使用小样本子集：训练 5000 条、验证 1000 条，适合本机课程演示。可在 `configs/default.yaml` 中调整。

也可以直接运行内置准备脚本。它会下载 VQA v2.0 的 train/val 问题和标注，并只下载小样本需要的 COCO 图片：

```bash
python scripts/prepare_vqa_data.py --root data/vqa --train-samples 5000 --val-samples 1000
```

## 训练

```bash
python train.py --config configs/default.yaml
```

训练会自动构建答案词表，并保存最佳模型到：

```text
checkpoints/best.pt
```

## 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## 命令行推理

```bash
python infer.py \
  --checkpoint checkpoints/best.pt \
  --image data/demo.jpg \
  --question "What color is the car?" \
  --topk 5
```

## Gradio 演示

```bash
python demo.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

启动后打开终端显示的本地地址，通常是：

```text
http://127.0.0.1:7860
```

如果没有 checkpoint，页面会显示明确提示，不会直接崩溃。

## 说明

VQA v2.0 每个问题通常有 10 个人工答案。本工程训练时使用软标签：某个答案出现次数越多，目标分数越高，最高为 1。评估时采用 Top-1 命中高频答案的简化准确率，便于课程设计展示和解释。
