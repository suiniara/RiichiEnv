<!-- kar-module: riichienv.kar.learning.trainer -->
<!-- api: production_update, train_production -->
# learning.trainer — 正式完整 Epoch 训练器

源码：[trainer.py](../../../../src/riichienv/kar/learning/trainer.py) · [learning 索引](../README.md)

## 主要接口与入口

`train_production(dataset_dir, output_dir, config, authorization=None, resume=False, ledger_path=None, main_model_registry=None)` 运行明确配方的完整训练；`production_update(...)` 提交一个全局 accumulation group。顶层 `kar train` 进入本模块，支持 torchrun。

核心参数为 `--dataset --output --config --authorization`，恢复/共享账本/唯一主模型由对应参数和登记绑定。缺授权默认在数据/模型执行前拒绝，不能以 smoke 或小 batch 绕过。

## 执行合同

- 实际 train/validation、N_train/eligible、rho、tier、完整工作量预测与 recipe 必须匹配；M2 不可使用 probe subset。
- CPU/CUDA、FP32/BF16 和 FP32 主参数；无效精度不静默换设备。
- 全局 sampler、尾 batch、空 rank 连图 dummy；统一 global_N，dummy 不进真实计数。
- 每个 microbatch 的 materialization/forward/loss/backward、梯度/裁剪、step/scheduler 有协调边界；失败不提交采样游标或部分检查点。
- 每 quarter-epoch 及 Epoch 末完整 validation；第三轮只消费 cal_fit model_selection 和预算条件。

## 恢复、计费与产物

每 rank RNG、optimizer/scheduler、sampler、验证历史、条件决策、费用锚点与性能累计进入代际检查点。已有 finished 检查点可只重建完成发布，不再次更新。输出互斥租约防并发写同一运行。

新运行先原子发布含 `startup.json` 的目录，随后绑定原账本锚点并登记未完成计算，再初始化模型。模型就绪后先提交 step 0 checkpoint，才执行初始全量 validation。初始化或首份 checkpoint 写盘失败时，`--resume` 可在同一数据、配方、实现和原账本下重试；M1 也会将已有未完成目录转交此流程。无 checkpoint 且存在已提交标记、更新日志或后续代际证据时拒绝重新初始化。没有启动记录的旧残留目录不会被自动收编。

日志记录真实 tokens/s、samples/s、padding 和耗时；完成报告用累计工作量/累计时间，不平均每步速率。已发生异常计算仍记账；硬中断留下的未完成计算先按 [budget](../../budget/README.md) 的实际凭据核销，再恢复。普通初始化重试也保留原账本，不能清空费用。M2 原始入口重验登记和实际 M1 证据，不能换 output/账本重获预算。

## 测试与状态

对应 [production/test_full_trainer.py](../../../../tests/kar/production/test_full_trainer.py)、[test_training_contracts.py](../../../../tests/kar/production/test_training_contracts.py) 和 [test_training_startup.py](../../../../tests/kar/production/test_training_startup.py)。启动故障注入测试禁止 forward/backward/optimizer 更新，只验证初始化、step 0 保存与恢复以及拒绝不安全重启。**实际训练入口会更新参数，仍需完整授权。**
