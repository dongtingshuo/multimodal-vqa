# BEiT-3 Base A: VQA v2 Inference Release

The recommended BEiT-3 A checkpoint scores **74.22** on all **214,354 VQA v2 public validation questions**. The upstream evaluation toolkit returned identical results on two local runs: yes/no 89.95, other 66.39, number 58.60. This is 5.80 points above ViLT `v0.3.0` on the same split. It is not a hidden test-server score.

首选 BEiT-3 A 在全部 **214,354 道 VQA v2 公开验证题**上取得 **74.22**，使用上游评测工具本地复评两次结果一致：yes/no 89.95、other 66.39、number 58.60。同一划分下较 ViLT `v0.3.0` 高 5.80 分。该成绩不是隐藏测试服务器成绩。

## Download and Verify / 下载与校验

Download `beit3-a-vqa-v2-inference.zip` and `SHA256SUMS.txt` from the release assets. The ZIP is **925,551,383 bytes** and contains `best.pt`, `beit3.spm`, and `manifest.json`. The checkpoint comes from epoch 9 (optimizer update 30,763). All 497 FP32 tensors were verified bit-for-bit against the original checkpoint; no quantization or retraining was applied. The export contains no optimizer, scheduler, or training-resume state and passed a local offline CPU inference smoke test.

从 Release 附件下载 `beit3-a-vqa-v2-inference.zip` 和 `SHA256SUMS.txt`。ZIP 大小为 **925,551,383 字节**，包含 `best.pt`、`beit3.spm` 和 `manifest.json`。权重来自第 9 轮（更新步 30,763），497 个 FP32 张量与原训练权重逐字节一致，没有量化或重新训练。该包不包含优化器、调度器或续训状态，已通过本地离线 CPU 推理验证。

```sh
shasum -a 256 -c SHA256SUMS.txt
unzip beit3-a-vqa-v2-inference.zip
```

| Asset / 文件 | SHA-256 |
| --- | --- |
| Inference ZIP / 推理包 | `05b21faa756d1a1b3c4e105b0155e521eb2a5079b648cb274d00aa4054b8323b` |
| Extracted checkpoint / 解压后的权重 | `8d40a588cdedb124cfe48b05cdc726fb479ae7f0830888ab0a4c32d49e2d9ffd` |
| Tokenizer / 分词器 | `6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4` |

## Inference / 推理

Install the BEiT-3 dependencies from the repository. Keep `beit3.spm` beside `best.pt`. / 在仓库中安装 BEiT-3 依赖，保持 `beit3.spm` 与 `best.pt` 同目录。

```sh
python -m pip install -e '.[beit3]'
python infer.py --checkpoint /path/to/beit3-a-vqa-v2/best.pt --device cpu \
  --image /path/to/image.jpg --question "What is in the image?"
```

The same file works with `demo.py --checkpoint` or VQA Studio Settings. ViLT `v0.3.0` remains available as a fallback. / 同一权重可用于 `demo.py --checkpoint` 或 VQA Studio 设置。ViLT `v0.3.0` 仍可作为备选。

## Evidence and Attribution / 证据与来源

- [Model card / 模型卡](https://github.com/dongtingshuo/multimodal-vqa/blob/beit3-a-v1/MODEL_CARD.md)
- [Evaluation provenance / 评测核验记录](https://github.com/dongtingshuo/multimodal-vqa/blob/beit3-a-v1/docs/evaluation/beit3-a-official-vqa-provenance.json)
- Base model and tokenizer: [Microsoft BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3), [accompanying upstream license notice](https://github.com/dongtingshuo/multimodal-vqa/releases/download/beit3-a-v1/LICENSE-Microsoft-UniLM.txt).
- Fine-tuning supervision and answer vocabulary: [VQA v2 annotations](https://visualqa.org/terms.html) (VQA Consortium, CC BY 4.0). No source images or raw annotation files are included. / 微调监督和答案词表来自 VQA v2 标注（VQA Consortium，CC BY 4.0），不包含原始图片或标注文件。
