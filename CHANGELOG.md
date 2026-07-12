# Changelog / 变更记录

## Unreleased / 未发布

## 0.3.0 / ViLT Full-Validation Release

Released on 2026-07-12. / 发布于 2026-07-12。

### Added / 新增

- Published the completed seed-42 ViLT checkpoint selected at epoch 5 after configured early stopping at epoch 7.
- 发布 seed 42 ViLT 完整训练权重；任务在 epoch 7 按配置早停，最佳权重来自 epoch 5。
- Recorded full validation metrics, 214,354 exported predictions, AutoDL runtime metadata, and training curves.
- 记录全量验证指标、214,354 条导出预测、AutoDL 运行元数据与训练曲线。
- Added a single-GPU AutoDL continuation workflow for RTX 4090D/4090 instances, including environment setup, official data download and validation, format-v3 checkpoint preflight, background training, monitoring, evaluation, and artifact packaging.
- 增加面向 RTX 4090D/4090 单卡实例的 AutoDL 续训流程，覆盖环境安装、官方数据下载与校验、format-v3 checkpoint 预检、后台训练、状态监控、评估与产物归档。

### Changed / 调整

- Promoted ViLT as the recommended engineering checkpoint and retained `v0.2.0` staged cross-attention as the comparison release.
- 将 ViLT 晋升为推荐工程权重，并保留 `v0.2.0` staged cross-attention 作为对照版本。
- Updated the verified checkpoint downloader, model card, experiment report, usage guide, citation metadata, and package version.
- 更新校验下载器、模型卡、实验报告、使用指南、引用元数据与软件包版本。

### Evaluation Scope / 评估范围

- Added official VQA v2 validation evaluation: `68.42` overall, `85.88` yes/no, `60.40` other, and `48.54` number.
- 增加官方 VQA v2 验证评测：overall `68.42`、yes/no `85.88`、other `60.40`、number `48.54`。
- Added evaluation provenance, input hashes, a Python 3 toolkit preparation utility, and Kaggle integration. The result is not a test-dev/test-standard leaderboard submission.
- 增加评测溯源、输入哈希、Python 3 toolkit 准备工具与 Kaggle 集成；该结果不是 test-dev/test-standard leaderboard 提交成绩。

## 0.2.0 / Staged Cross-Attention Model Release

Released on 2026-07-11. / 发布于 2026-07-11。

### Added / 新增

- Published the completed 12-epoch staged cross-attention checkpoint with verified size and SHA256 checksum.
- 发布完成 12 epoch 训练的 staged cross-attention checkpoint，并提供文件大小与 SHA256 校验。
- Updated the default checkpoint downloader, model card, citation metadata, and release documentation for `v0.2.0`.
- 将默认权重下载器、模型卡、引用元数据和发布文档更新至 `v0.2.0`。

- ViLT seed-42 partial-run results, continuation status, and release-gate distinctions across the README, model card, experiment report, and training protocol.
- 在 README、模型卡、实验报告和训练协议中补充 ViLT seed 42 部分训练结果、续训状态与发布门槛说明。
- GitHub issue forms and pull-request template for reproducible bug reports, experiment proposals, and reviewable contributions.
- 增加 GitHub Issue 表单和 Pull Request 模板，用于可复现问题报告、实验建议与规范贡献。

- Unified CPU, CUDA, MPS, and Kaggle execution controls with command-line path and sample overrides.
- 增加统一的 CPU、CUDA、MPS 与 Kaggle 执行控制，以及命令行路径和样本数覆盖参数。
- Controlled frozen-baseline and staged partial fine-tuning configs with differential learning rates, gradient accumulation, and early stopping.
- 增加受控冻结基线与分阶段部分微调配置，支持差分学习率、梯度累积和早停。
- Format-v3 resumable checkpoints with optimizer, scheduler, AMP scaler, RNG, history, and stage state.
- 增加 format-v3 可续训 checkpoint，保存优化器、调度器、AMP scaler、随机数、历史和训练阶段状态。
- Kaggle notebook, data preflight validation, official VQA prediction export/evaluation adapter, and experiment release gate.
- 增加 Kaggle notebook、数据预检、官方 VQA 预测导出/评估适配器和实验发布门槛。

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
- VQA soft-score and Top-5 VQA metrics with per-example BCE loss normalization.
- 增加 VQA soft score、Top-5 VQA 指标和按样本归一化的 BCE loss。
- Persistent training history, curves, learning-rate records, timing, and runtime metadata.
- 增加训练历史、曲线、学习率、耗时和运行元数据持久化。
- Resumable, integrity-checked dataset and Release checkpoint downloads.
- 增加可断点续传并校验完整性的数据集和 Release 权重下载流程。
- GitHub Actions CI for linting, tests, compilation, dependency checks, and package builds.
- 增加覆盖代码检查、测试、编译、依赖检查和打包的 GitHub Actions CI。
- Git ignore rules for datasets, checkpoints, caches, and generated outputs.
- 增加数据集、checkpoint、缓存和生成产物的 Git 忽略规则。

### Changed / 调整

- Dataset targets are stored sparsely and materialized per batch to reduce worker memory pressure.
- Dataset 目标改为稀疏存储并在 batch 内构造，以降低 worker 内存压力。
- Removed horizontal image flipping because it can invalidate directional VQA answers.
- 移除水平翻转增强，避免方向类 VQA 答案失真。

- Reworked README into a professional bilingual project overview.
- 将 README 调整为专业、克制的双语项目概览。
- Clarified runtime artifacts and expected checkpoint location.
- 明确运行产物与默认 checkpoint 位置。
- Documented CUDA fallback behavior for inference, evaluation, and demo entrypoints.
- 说明推理、评估和演示入口的 CUDA 回退行为。
- Updated runtime entrypoints to construct models through the shared model factory.
- 将运行入口调整为通过统一模型工厂构建模型。
- Checkpoint loading now restores stored architecture and preprocessing configuration while preserving local runtime paths.
- checkpoint 加载会恢复内置架构与预处理配置，同时保留本地运行路径。
- Runtime and development dependencies are separated and declared in `pyproject.toml`.
- 运行与开发依赖已分离，并在 `pyproject.toml` 中完整声明。

## 0.1.0 / 初始版本

- Implemented ResNet-50 + DistilBERT + bidirectional cross-attention VQA model.
- 实现 ResNet-50、DistilBERT 与双向跨模态注意力组成的 VQA 模型。
- Added VQA v2.0 data preparation script.
- 增加 VQA v2.0 数据准备脚本。
- Added training, evaluation, CLI inference, and Gradio demo entrypoints.
- 增加训练、评估、命令行推理和 Gradio 演示入口。
- Added smoke tests for answer vocabulary and dataset loading.
- 增加答案词表与数据加载的连通性测试。
