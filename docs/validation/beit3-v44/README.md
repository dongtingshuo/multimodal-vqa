# Version 44 Resource Diagnostic / Version 44 资源诊断

Kaggle Version 44 completed successfully at code commit
`dff253a0e947a9304000d223fb684cf41804914f`. This was a deliberately bounded
75-minute session, not a completed experiment A. Two Tesla T4s trained BEiT-3
with batch 8 per GPU, accumulation 8, and effective global batch 128.

Version 44 成功结束，代码固定为上述 commit。这是预设 75 分钟的限时诊断，
不是实验 A 训练完成。双 T4 使用每卡 batch 8、累积 8，全局有效 batch 128。

## Observed Result / 实测结果

- Full-data training paused at epoch 1, batch 2,672/27,347 (9.77%), update 334.
- The pause reason was `session_time_budget`, not an exception.
- Training time in the summary was 3,322.75 seconds; zero epochs completed.
- A resumable `latest.pt` was saved at an optimizer boundary before packaging.
- 66 resource samples recorded a maximum container usage of 26.90 GiB and a
  peak counter of 26.97 GiB against a 30 GiB limit. OOM and OOM-kill counts were zero.
- No full validation or official VQA score exists; the recommended ViLT model
  remains unchanged. Training progress metrics are not validation results.

- 全量训练在第 1 轮第 2,672/27,347 个 batch（9.77%）、第 334 次更新后暂停。
- 暂停原因是时间预算，日志明确记录正常保存，没有异常退出。
- 汇总中的训练耗时为 3,322.75 秒，完成 epoch 数为 0。
- 暂停时已在参数更新边界保存可续训的 `latest.pt`，随后完成打包。
- 66 条资源记录显示最高使用量 26.90 GiB、峰值计数 26.97 GiB，上限 30 GiB；
  OOM 与 OOM-kill 计数均为 0。
- 尚无完整验证和官方 VQA 成绩，不替换现有 ViLT 推荐模型。

## Scope / 证据边界

The downloaded checkpoint is 2,772,574,122 bytes and was loaded successfully on
CPU. Its incomplete-epoch state matches batch 2,672 and update 334; optimizer,
scheduler, AMP, both rank RNG states, and the 5,000-answer vocabulary are present.
SHA-256: `ad3dacea39aaccf0020f952ac5b2c6fba0dc52b29bd0a9fc3294bc5772c4cd4e`.
This verifies payload readability and recovery state, not a subsequent GPU resume.

下载文件大小为 2,772,574,122 字节，CPU 读取成功；未完成 epoch 状态与 batch、
更新次数一致，包含优化器、调度器、AMP、双 rank RNG 和 5,000 答案词表。
校验值见上。这确认文件和恢复状态有效，尚未执行该文件的后续 GPU 续训。

This run passed the previous observed batch-1,787 log boundary and the graceful
pause path, but does not identify Version 43's exact termination cause or prove
11-hour stability. The checkpoint-inspection subprocess fix in `e6d319d` was
committed after Version 44 started and was not exercised by this run.

本次超过了 Version 43 可获取日志的第 1,787 个 batch，并验证了正常暂停路径，
但不能据此确定旧版本中断原因或证明连续 11 小时稳定。`e6d319d` 的 checkpoint
检查进程隔离修复在本次启动后提交，本次运行不包含该修复。

The archived runner status still says packaging started because it was copied
before the final packaging-completed message. Kaggle's terminal `COMPLETE` and
the terminal log confirm packaging completed; do not interpret that snapshot as
an active job.

归档状态快照在打包开始时复制，因此仍显示该阶段运行中；Kaggle 最终状态与
终止日志确认打包成功，不应将此快照理解为任务仍在运行。
