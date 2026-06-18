# Changelog / 变更记录

## Unreleased / 未发布

### Added / 新增

- Bilingual professional documentation for project usage, architecture, and troubleshooting.
- 项目使用、架构和故障排查文档已补充为专业双语说明。
- Runtime documentation for data preparation, training, evaluation, inference, and Gradio demo.
- 增加数据准备、训练、评估、推理和 Gradio 演示的运行说明。
- Baseline comparison workflow for `text_only`, `image_only`, `baseline_concat`, and `cross_attention`.
- 增加覆盖 `text_only`、`image_only`、`baseline_concat` 和 `cross_attention` 的基线对比流程。
- Lightweight error analysis by question type with JSON and Markdown reports.
- 增加按问题类型统计的轻量错误分析，输出 JSON 与 Markdown 报告。
- Toy demo files for offline workflow validation.
- 增加用于离线流程验证的玩具演示文件。
- Dependency pinning guidance for stable Gradio execution.
- 增加稳定运行 Gradio 的依赖版本说明。
- Git ignore rules for datasets, checkpoints, caches, and generated outputs.
- 增加数据集、checkpoint、缓存和生成产物的 Git 忽略规则。

### Changed / 调整

- Reworked README into a professional bilingual project overview.
- 将 README 调整为专业、克制的双语项目概览。
- Clarified runtime artifacts and expected checkpoint location.
- 明确运行产物与默认 checkpoint 位置。
- Documented CUDA fallback behavior for inference, evaluation, and demo entrypoints.
- 说明推理、评估和演示入口的 CUDA 回退行为。
- Updated runtime entrypoints to construct models through the shared model factory.
- 将运行入口调整为通过统一模型工厂构建模型。

## 0.1.0 / 初始版本

- Implemented ResNet-50 + DistilBERT + bidirectional cross-attention VQA model.
- 实现 ResNet-50、DistilBERT 与双向跨模态注意力组成的 VQA 模型。
- Added VQA v2.0 data preparation script.
- 增加 VQA v2.0 数据准备脚本。
- Added training, evaluation, CLI inference, and Gradio demo entrypoints.
- 增加训练、评估、命令行推理和 Gradio 演示入口。
- Added smoke tests for answer vocabulary and dataset loading.
- 增加答案词表与数据加载的连通性测试。
