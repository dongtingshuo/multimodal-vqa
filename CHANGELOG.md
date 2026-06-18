# Changelog / 变更记录

## Unreleased / 未发布

### Added / 新增

- Bilingual professional documentation for project usage, architecture, and troubleshooting.
- Runtime documentation for data preparation, training, evaluation, inference, and Gradio demo.
- Dependency pinning guidance for stable Gradio execution.
- Git ignore rules for datasets, checkpoints, caches, and generated outputs.

### Changed / 调整

- Reworked README into a professional bilingual project overview.
- Clarified runtime artifacts and expected checkpoint location.
- Documented CUDA fallback behavior for inference, evaluation, and demo entrypoints.

## 0.1.0 / 初始版本

- Implemented ResNet-50 + DistilBERT + bidirectional cross-attention VQA model.
- Added VQA v2.0 data preparation script.
- Added training, evaluation, CLI inference, and Gradio demo entrypoints.
- Added smoke tests for answer vocabulary and dataset loading.
