from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from vqa_project.config import load_config, resolve_device
from vqa_project.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight VQA model-variant comparison.")
    parser.add_argument("--config", default="configs/demo_comparison.yaml")
    return parser.parse_args()


def build_toy_batch(config: dict, device: torch.device) -> dict[str, torch.Tensor]:
    data_cfg = config["data"]
    torch.manual_seed(int(config.get("seed", 42)))
    num_samples = int(data_cfg.get("num_samples", 8))
    image_size = int(data_cfg.get("image_size", 32))
    sequence_length = int(data_cfg.get("max_question_length", 12))
    answer_vocab_size = int(data_cfg.get("answer_vocab_size", 8))

    images = torch.linspace(0.0, 1.0, steps=num_samples * 3 * image_size * image_size)
    images = images.view(num_samples, 3, image_size, image_size).to(device)
    input_ids = torch.arange(num_samples * sequence_length).view(num_samples, sequence_length)
    input_ids = input_ids.remainder(257).to(device)
    attention_mask = torch.ones(num_samples, sequence_length, dtype=torch.long, device=device)
    labels = torch.arange(num_samples, device=device).remainder(answer_vocab_size)
    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@torch.no_grad()
def evaluate_variant(variant_name: str, config: dict, batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    model_cfg = dict(config["model"])
    model_cfg["name"] = variant_name
    model = build_model(model_cfg, answer_vocab_size=int(config["data"]["answer_vocab_size"])).to(device)
    model.eval()
    logits = model(batch["images"], batch["input_ids"], batch["attention_mask"])
    loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == batch["labels"]).float().mean()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "variant": variant_name,
        "samples": int(batch["labels"].numel()),
        "mock_loss": round(float(loss.item()), 6),
        "mock_accuracy": round(float(accuracy.item()), 6),
        "parameters": int(parameter_count),
    }


def write_json(path: Path, rows: list[dict]) -> None:
    payload = {
        "notice": (
            "These values are from a tiny synthetic/mock run for workflow validation only. "
            "They are not a real VQA benchmark."
        ),
        "results": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "samples", "mock_loss", "mock_accuracy", "parameters"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], config_path: str) -> None:
    table = "\n".join(
        f"| {row['variant']} | {row['samples']} | {row['mock_loss']:.6f} | "
        f"{row['mock_accuracy']:.6f} | {row['parameters']} |"
        for row in rows
    )
    text = f"""# Model Comparison Report / 模型对比报告

## Experiment Objective / 实验目的

This report validates the model-comparison workflow across four VQA variants using a tiny synthetic batch.

本报告使用极小的合成批次验证视觉问答模型变体的对比流程。

## Model Variants / 模型变体

- `text_only`: question encoder and answer classifier only.
- `image_only`: image encoder and answer classifier only.
- `baseline_concat`: pooled image and text features are concatenated directly.
- `cross_attention`: image and text tokens interact through cross-modal attention.

- `text_only`：仅使用问题编码器和答案分类器。
- `image_only`：仅使用图像编码器和答案分类器。
- `baseline_concat`：直接拼接池化后的图像与文本特征。
- `cross_attention`：通过跨模态注意力融合图像与文本 token。

## Data Setup / 数据设置

Configuration: `{config_path}`.

The run uses deterministic synthetic tensors and mock backbones, so no dataset download, checkpoint, or network access is required.

配置文件：`{config_path}`。

本流程使用确定性的合成张量和 mock backbone，因此不需要下载数据集、checkpoint 或访问网络。

## Metrics / 指标说明

- `mock_loss`: cross-entropy on synthetic labels.
- `mock_accuracy`: Top-1 accuracy on synthetic labels.
- `parameters`: trainable and frozen parameter count in the constructed model.

- `mock_loss`：基于合成标签计算的交叉熵。
- `mock_accuracy`：基于合成标签计算的 Top-1 准确率。
- `parameters`：构建出的模型参数总量。

## Comparison Table / 对比表格

| Variant | Samples | Mock Loss | Mock Accuracy | Parameters |
|---|---:|---:|---:|---:|
{table}

## Preliminary Analysis / 初步分析

The comparison confirms that all variants share the same input contract and can be evaluated through a common workflow. Differences in the mock metrics are only sanity-check signals.

该对比确认四种变体具备统一输入接口，并可通过同一流程进行评估。mock 指标差异只用于流程检查。

## Current Limitations / 当前限制

These values are not a real benchmark and must not be interpreted as final model quality. Reliable conclusions require full VQA data, a fixed training schedule, multiple runs, and held-out evaluation.

这些数值不是正式 benchmark，不能解释为最终模型质量。可靠结论需要完整 VQA 数据、固定训练方案、多次运行以及独立验证集评估。
"""
    path.write_text(text, encoding="utf-8")


def run(config_path: str) -> dict[str, Path]:
    config = load_config(config_path)
    device = resolve_device(config.get("device", "cpu"), allow_fallback=True)
    output_dir = Path(config.get("output_dir", "outputs/comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = build_toy_batch(config, device)
    rows = [
        evaluate_variant(variant, config, batch, device)
        for variant in config.get("variants", ["text_only", "image_only", "baseline_concat", "cross_attention"])
    ]

    json_path = output_dir / "comparison_results.json"
    csv_path = output_dir / "comparison_results.csv"
    report_path = output_dir / "comparison_report.md"
    write_json(json_path, rows)
    write_csv(csv_path, rows)
    write_report(report_path, rows, config_path)
    return {"json": json_path, "csv": csv_path, "report": report_path}


def main() -> None:
    paths = run(parse_args().config)
    print(f"wrote json: {paths['json']}")
    print(f"wrote csv: {paths['csv']}")
    print(f"wrote report: {paths['report']}")


if __name__ == "__main__":
    main()
