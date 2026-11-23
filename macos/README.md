# VQA Studio for macOS / macOS 版 VQA Studio

VQA Studio is a native SwiftUI workspace for image-based questions. It uses the same Python inference code as the CLI and keeps the model loaded between questions. The app stores imported image copies, drafts, favorites and conversations locally.

VQA Studio 是原生 SwiftUI 视觉问答工作台，与命令行共用 Python 推理代码，连续提问时复用已加载的模型。应用在本地保存导入图片的副本、问题草稿、收藏和会话。

## Build / 构建

Requires macOS 14 or newer and a Swift toolchain with a compatible macOS SDK. For BEiT-3, install the optional dependencies in the project's Python environment and extract the [recommended inference bundle](../README.md#trained-checkpoint--已训练模型). In Settings select its `best.pt` and keep `beit3.spm` beside it. ViLT remains available as a fallback.

需要 macOS 14 或更新版本，以及有兼容 macOS SDK 的 Swift 工具链。使用 BEiT-3 时，在项目的 Python 环境中安装可选依赖并解压[首选推理包](../README.md#trained-checkpoint--已训练模型)。在设置中选中解压后的 `best.pt`，并保持 `beit3.spm` 与它同目录。ViLT 可作为备选。

From the repository root / 在仓库根目录运行：

```sh
python scripts/build_macos_app.py --python /path/to/project-env/bin/python
open "VQA Studio.app"
```

The build signs a local development bundle and embeds initial project and Python paths. It does not bundle the Python environment, weights or datasets, and is not a notarized distribution. Use `--sdk /path/to/installed/MacOSX.sdk` when the default SDK is incompatible. To place the bundle outside the checkout, pass `--output /path/to/folder`.

构建生成用于本地使用的签名应用，写入初始项目和 Python 路径。应用不包含 Python 环境、模型权重或数据集，也不属于公证分发包。默认 SDK 不兼容时可通过 `--sdk` 指定已安装的 SDK；`--output` 可指定其他输出目录。

## Settings and use / 配置与使用

1. In Settings, select the repository, Python executable, model config and checkpoint. Start on CPU; MPS availability depends on the environment. Turn off offline mode for the first load only if trusted model assets need downloading.
2. Import images or drag them into the workspace. Ask questions in English and inspect ranked answers. Use the library to find previous images or conversations.
3. Export conversations as JSON or Markdown. Closing the workspace keeps the model available; quitting stops the worker.

1. 在设置中选定仓库、Python 可执行文件、模型配置和权重。建议先用 CPU，MPS 是否可用取决于环境。只有首次加载需要下载可信模型资源时才关闭离线模式。
2. 导入或拖入图片，用英文提问并查看候选答案。可在图库中检索历史图片及会话。
3. 会话可导出为 JSON 或 Markdown。关闭工作区窗口会保留模型；退出应用会停止推理进程。

## Validation / 验证

```sh
python scripts/check_macos_app.py --sdk "$(xcrun --sdk macosx --show-sdk-path)"
python -m pytest -q tests/test_desktop_worker.py tests/test_macos_packaging.py
```

Native checks use isolated fixtures; they do not touch your installed app or personal library. / 原生检查使用隔离的测试数据，不改动已安装应用或个人图库。
