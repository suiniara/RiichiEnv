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

新运行先从已核验的授权对象取得数据 manifest SHA，原子发布含 `startup.json` 的目录，绑定原账本锚点并持久登记未完成计算。此后才读取完整 manifest、核验 SQLite 全文件 SHA、物化 train/validation 索引及长度、核验 M2 前置证据、设置设备并创建进程组。实际数据必须与授权身份完全一致，完整 schema/contract/SHA 检查仍然执行；manifest 含逐游戏记录，因此不把它假定为固定大小元数据，也不增加数据规模上限。

预处理完成后先结算实测时间，检查已发生费用及剩余整份预测，再进入进程组/模型初始化；结算保留当前 pending attempt。旧 attempt 未核销时，在加载任何数据前拒绝。多 rank 通过共享输出父目录中的 [StartupCoordinator](../distributed/README.md) 收据等待 primary 的持久记账，预处理和设备异常也在进程组创建前协调，避免其他 rank 提前开始昂贵工作。

模型就绪后先提交 step 0 checkpoint，才执行初始全量 validation。预处理、初始化或首份 checkpoint 写盘失败时，`--resume` 可在同一数据、配方、实现和原账本下重试；M1 也会将已有未完成目录转交此流程。无 checkpoint 且存在已提交标记、更新日志或后续代际证据时拒绝重新初始化。没有启动记录的旧残留目录不会被自动收编。启动记录格式仍为 `kar_training_startup_v1`；实现身份变更不会被当作可直接续训的旧代码运行。

M1/M2 调度器另以同一 phase 账本记录自身设备/进程组启动，成功结算后才交给 trainer 开新的 attempt，不重复登记固定费用或重叠计算时间。调用方已经建立的外部 group 的历史开销无法由 trainer 回溯；调度器在进入该启动阶段前的 M1/M2 证据预检不属于本段计费范围。

日志记录真实 tokens/s、samples/s、padding 和耗时；完成报告用累计工作量/累计时间，不平均每步速率。已发生异常计算仍记账；硬中断留下的未完成计算先按 [budget](../../budget/README.md) 的实际凭据核销，再恢复。普通初始化重试也保留原账本，不能清空费用。M2 原始入口重验登记和实际 M1 证据，不能换 output/账本重获预算。

## 测试与状态

对应 [production/test_full_trainer.py](../../../../tests/kar/production/test_full_trainer.py)、[test_training_contracts.py](../../../../tests/kar/production/test_training_contracts.py)、[test_training_startup.py](../../../../tests/kar/production/test_training_startup.py)、[test_training_startup_sqlite.py](../../../../tests/kar/production/test_training_startup_sqlite.py) 和 [test_startup_coordination.py](../../../../tests/kar/production/test_startup_coordination.py)。启动回归以合成时钟、真实小型 SQLite 元数据、子进程硬退出和文件收据检验费用、恢复与多 rank 故障路径；禁止 forward/backward/optimizer 更新，不启动 GPU 或实际付费作业。**实际训练入口会更新参数，仍需完整授权。**
