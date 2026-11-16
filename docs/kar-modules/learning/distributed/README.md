<!-- kar-module: riichienv.kar.learning.distributed -->
<!-- api: DistributedContext, DistributedStageError -->
# learning.distributed — 进程组、全局计数和失败协调

源码：[distributed.py](../../../../src/riichienv/kar/learning/distributed.py) · [learning 索引](../README.md)

## 接口与生命周期

`DistributedContext.open(device, backend=None, gloo_bind_host=None)` 读取 torchrun 的 RANK/WORLD_SIZE/LOCAL_RANK，校验拓扑并选择目标设备/进程组。CUDA 优先可用 NCCL，否则使用明确 Gloo 路径；Windows 的显式 Gloo host 通过版本适配工厂创建。

提供 `primary`、`coordinate(error, stage)`、`gather(value)`、`global_count(local_count)`、`synchronize_model(model)`、`average_gradients(model)` 和 `close()`。

## 合同

所有 rank 以相同顺序参与集体操作；一个 rank 的前向/反向/数据错误应在对应边界协调为 `DistributedStageError`，不能只让主 rank 继续保存。已有 group 必须匹配本次拓扑，context 只销毁自己拥有的 group，避免嵌套 orchestration/trainer 误关父组。

正式更新在累积期间使用 no_sync，最后显式平均梯度一次；配合每 rank 的 `W/global_N` sum 目标，不重复平均或按本地标签数归一化。Gloo 下必要的 CPU transport 不改变数学目标。

## 使用与副作用

这是运行层，不是静态配置。open 可能设置 CUDA、分配控制 Tensor、初始化网络进程组；必须从已授权生产入口调用并用 finally 关闭。不能为调试关闭 SSH/系统安全设置或结束用户进程。

无独立 CLI，由 [trainer](../trainer/README.md)和 [orchestration](../orchestration/README.md)消费。无效拓扑、缺/非有限梯度或其他 rank 失败时停止，不重试部分权重。

## 测试与状态

对应 [production/test_training_contracts.py](../../../../tests/kar/production/test_training_contracts.py)。含真实多进程/更新的测试不在本轮执行。本轮未创建 group、查询 CUDA 或启动 worker。
