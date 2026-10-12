# Version 43 Interruption / Version 43 训练中断

Version 43 used `107d18b46eea47a70b4e6f4f1c9f000a1f62b8bd` and passed both
dual-T4 save/restore and real-data training smoke gates. Full training loaded
437,546 training examples and all 214,354 validation questions. The available
API log reaches epoch 1, batch 1,787/27,347, and ends inside a progress line.
The owner confirmed a failed run. The API returned `ERROR`, no failure message,
and no output files. No completed epoch or official score is available.

Version 43 通过双 T4 恢复检查和真实数据 smoke 后进入全量训练。API 日志至少
记录到第 1 轮的 1,787/27,347 个小批次，并在进度行中途结束。用户已确认失败，
API 返回 ERROR，但没有终止原因或输出文件。该实验没有完成的 epoch 或官方分数。

## Findings and Changes / 排查与修改

The precise termination cause is unconfirmed: the last available log may be
truncated, and Version 43 did not record container memory usage or OOM counters.
Host-memory pressure is a hypothesis, not a measured diagnosis.

A concrete retention issue was found in the runner: it loaded the real-data
smoke checkpoint onto CPU and retained that local variable throughout full
training. The checkpoint was approximately 2.77 GB on disk. Verification now
runs in a short-lived CPU subprocess and returns only JSON metadata. Version 44
showed that scope cleanup freed Python references but left approximately 2.44 GiB
of runner RSS; process isolation avoids retaining that allocator memory.

Each BEiT-3 recipe now uses two data workers per GPU rather than four. Model,
optimizer, answer vocabulary, effective batch, and training schedule are unchanged.
Recovery checkpoints are written at the first successful update and subsequently
at the configured interval. Container memory, peak usage, memory limit, runner
RSS, and OOM counters are recorded once per minute in `resource_usage.jsonl`.
Progress rendering is throttled to reduce console output; metric logging remains.

此次中断的精确原因尚未确认：可获取的末尾日志可能被截断，原运行没有记录
容器内存或 OOM 计数，因此宿主内存压力仍只是待验证的假设。

已发现并修复明确的内存滞留：主进程加载约 2.77 GB 的 smoke checkpoint 后，
一直保留其 CPU 张量。Version 44 实测表明，释放 Python 引用后主进程 RSS 仍
约为 2.44 GiB，因此改为独立 CPU 子进程验证，只返回 JSON 元数据，退出时
释放检查进程的内存。三组 BEiT-3 配方统一改为每卡两个数据进程；模型、优化器、
词表、有效 batch 和训练策略保持一致。首次成功更新即保存恢复点，之后保持
配置的保存间隔。新增每分钟内存与 OOM 记录，并降低进度条刷新频率。

## Acceptance / 验收

Local tests cover subprocess checkpoint verification, cgroup v1/v2 memory readings,
missing/unlimited metrics, initial and periodic checkpoint boundaries, and the
existing accumulated dual-process training regression. A bounded Kaggle run is
required to observe full-data memory usage and verify graceful checkpoint pause.
These changes do not prove Version 43's termination cause or improve its score.

本地测试覆盖子进程 checkpoint 验证、cgroup v1/v2、缺失或无限内存指标、首次及周期保存，
以及已有的双进程梯度累积回归。下一轮需限时实测完整数据内存占用和暂停恢复点；
不能据此提前宣布中断原因已确定或模型分数已提升。
