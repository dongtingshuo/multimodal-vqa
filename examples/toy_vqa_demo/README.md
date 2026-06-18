# Toy VQA Demo / 玩具视觉问答演示

## Demo Purpose / 演示目的

This folder provides a tiny, public, deterministic sample for validating reporting and error-analysis workflows.

本目录提供一个极小、公开且确定性的样例，用于验证报告生成与错误分析流程。

## Files / 文件说明

- `toy_questions.json`: four toy question-answer records.
- `toy_predictions.jsonl`: toy prediction records for error analysis.

- `toy_questions.json`：四条玩具问答记录。
- `toy_predictions.jsonl`：用于错误分析的玩具预测记录。

## How to Run / 运行方式

```bash
python scripts/run_model_comparison.py --config configs/demo_comparison.yaml
python scripts/run_error_analysis.py --predictions examples/toy_vqa_demo/toy_predictions.jsonl
```

## Notes / 注意事项

The files are intentionally small so the workflow can run without COCO/VQA data, checkpoints, or network access.

这些文件刻意保持很小，因此无需 COCO/VQA 数据、checkpoint 或网络访问即可运行流程。

## Limitations / 限制

Toy samples are not a benchmark and should only be used for workflow validation.

玩具样例不是 benchmark，只应用于流程验证。
