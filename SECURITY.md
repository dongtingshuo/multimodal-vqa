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

## Downloads and Archives / 下载与归档

Maintained download helpers accept HTTPS URLs only and validate archive members before extraction. Do not bypass these checks for third-party datasets or bundles. The AutoDL bundle builder accepts a trusted local format-v3 checkpoint; because full training state uses PyTorch serialization, never point it at an untrusted `.pt` file.

维护中的下载工具仅接受 HTTPS URL，并在解压前检查归档成员。不要为第三方数据集或 bundle 绕过这些检查。AutoDL bundle 构建器仅接受可信的本地 format-v3 checkpoint；完整训练状态使用 PyTorch 序列化，因此不要向其传入不可信 `.pt` 文件。

The reproducibility environment still pins a legacy Gradio/FastAPI/Starlette stack with known advisories. Until a coupled major-version migration is validated against the released ViLT checkpoint, training pipeline, and demo, run the web interface only with trusted inputs on a loopback or private interface; do not expose it as a public service.

为保持可复现性，当前环境仍固定了存在已知漏洞通告的历史 Gradio/FastAPI/Starlette 组合。在联动大版本迁移通过已发布 ViLT 权重、训练流程和演示验证之前，只在回环地址或私有网络中以可信输入运行 Web 界面；不要将其作为公开服务暴露。
