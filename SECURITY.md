# Security Policy / 安全政策

## Reporting / 报告漏洞

Please report security-sensitive issues through GitHub's private security-advisory workflow instead of a public issue:

如发现安全敏感问题，请通过 GitHub 私有安全公告流程报告，不要直接创建公开 Issue：

https://github.com/dongtingshuo/multimodal-vqa/security/advisories/new

Include affected versions, reproduction steps, impact, and any proposed mitigation. Avoid attaching private datasets, credentials, or untrusted checkpoint files.

请包含受影响版本、复现步骤、影响和可选缓解方案。不要附带私有数据集、凭据或不可信的 checkpoint 文件。

## Model Files / 模型文件

PyTorch checkpoints may contain pickle-based data. Only load checkpoints from trusted sources and verify published checksums before use. The official Release checkpoint can be downloaded and verified with `python scripts/download_checkpoint.py`.

PyTorch checkpoint 可能包含基于 pickle 的数据。仅加载可信来源的权重，并在使用前校验已公布的校验和。官方 Release 权重可通过 `python scripts/download_checkpoint.py` 下载和校验。
