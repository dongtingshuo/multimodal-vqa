# BEiT-3 Dual-T4 Validation / BEiT-3 双 T4 验证

Kaggle Version 42 completed successfully on 2026-10-12 with code commit
`107d18b46eea47a70b4e6f4f1c9f000a1f62b8bd`. It validated the production training loop
before experiment A was launched. The 128-example smoke is an execution check;
it does not establish a model accuracy or an official full-validation score.

Kaggle Version 42 于 2026-10-12 成功完成，验证代码版本为上述 commit。
该验证覆盖正式训练循环，是实验 A 启动前的执行检查；128 条样本的 smoke
不作为模型准确率或完整官方验证分数。

## Fix and Evidence / 修复与证据

- Version 41 used the older `283bcbd` code and failed at the first synchronized
  backward after seven accumulation micro-batches. The fix enables static-graph
  DDP for checkpointed, fixed-full BEiT-3 and synchronizes its first executed
  micro-batch before using `no_sync`. Unused parameters are detected by static DDP.
- Two-process CPU tests cover fresh and in-epoch resumed accumulation, two epochs,
  and partial final groups. Parameter updates match equivalent global batches.
- Actual dual-T4 validation used 480 px inputs, a 5,000-answer head, batch 8 per
  GPU, and eight accumulation steps. Both ranks completed two successful updates
  and restored model, optimizer, AMP scaler, and rank-specific RNG state.
- Complete COCO/VQA data passed strict validation. A real-data train/validation
  smoke saved both checkpoints, and `latest.pt` contained one successful update.

Version 41 使用旧代码，在累积七个小批次后的第一次同步反向传播失败。
修复启用固定全量微调专用的 DDP 静态图，并让首个执行的小批次先同步，
之后再使用 `no_sync`。本地测试验证了首次启动、中途恢复、连续两轮和尾批次，
参数更新与等效全局 batch 一致。真实双 T4 完成了两次更新与完整状态恢复；
完整数据检查和真实数据训练 smoke 也通过，保存的恢复点包含一次成功更新。

## Artifacts and Timing / 产物与耗时

`validation_summary.json` records observed gates. The benchmark, runtime config,
and capacity estimate were downloaded from Version 42 without editing their values.
The estimator uses training forward/backward throughput as a proxy for single-rank
validation and adds a 1.5x planning factor; its epoch/session counts are conservative
planning estimates, not measurements of a complete epoch.

`validation_summary.json` 记录实际通过的检查。其余三个文件直接下载自 Version 42，
未改写数值。估算工具用训练前向与反向吞吐近似单卡验证耗时，并增加 1.5 倍余量；
轮次耗时和会话数量属于保守规划估算，尚非完整 epoch 的实测结果。
