# Contributing / 贡献指南

Thank you for improving `multimodal-vqa`. Keep changes focused, reproducible, and explicit about whether reported results come from synthetic checks, subsets, or full VQA data.

感谢改进 `multimodal-vqa`。请保持改动聚焦、可复现，并明确说明结果来自合成检查、数据子集还是完整 VQA 数据。

## Development / 开发

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
python -m build
```

Windows activation / Windows 激活命令：

```powershell
.venv\Scripts\activate
```

Do not commit datasets, checkpoints, generated outputs, local caches, or credentials. Tests must remain offline-safe and should use mock backbones unless a network-dependent integration test is explicitly marked.

请勿提交数据集、checkpoint、生成产物、本地缓存或凭据。测试应保持离线可运行，除非显式标记为网络集成测试，否则应使用 mock backbone。

## Pull Requests / 合并请求

- Describe behavioral and metric changes.
- Add or update tests for changed contracts.
- Update bilingual documentation when user-facing behavior changes.
- Do not present toy or synthetic outputs as benchmark evidence.

- 说明行为和指标变化。
- 为改动后的接口补充或更新测试。
- 面向用户的行为变化需要同步更新双语文档。
- 不得将玩具或合成数据输出表述为 benchmark 证据。
