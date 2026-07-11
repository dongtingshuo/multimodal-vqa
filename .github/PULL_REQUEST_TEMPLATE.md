## Summary / 变更摘要

Describe what changed and why. / 说明改动内容与原因。

## Validation / 验证

- [ ] `python -m ruff check .`
- [ ] `python -m pytest -q`
- [ ] User-facing documentation is bilingual where applicable. / 面向用户的文档已按需提供中英文。
- [ ] No datasets, checkpoints, generated outputs, or credentials are committed. / 未提交数据集、权重、生成产物或凭据。

## Experiment Evidence / 实验证据

For metric-affecting changes, record the baseline, config, commit, data split, seed, hardware, best epoch, internal metrics, and official VQA result when available. Mark synthetic and partial-run results explicitly.

若改动影响指标，请记录 baseline、配置、commit、数据划分、seed、硬件、最佳 epoch、内部指标及可用的官方 VQA 结果，并明确标注合成结果和部分训练结果。
