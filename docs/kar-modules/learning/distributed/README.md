<!-- kar-module: riichienv.kar.learning.distributed -->
<!-- api: DistributedContext, DistributedStageError, StartupCoordinator, launch_topology, selected_backend -->
# learning.distributed — 进程组、全局计数和失败协调

源码：[distributed.py](../../../../src/riichienv/kar/learning/distributed.py) · [learning 索引](../README.md)

## 接口与生命周期

`DistributedContext.open(device, backend=None, gloo_bind_host=None)` 读取 torchrun 的 RANK/WORLD_SIZE/LOCAL_RANK，校验拓扑并选择目标设备/进程组。CUDA 优先可用 NCCL，否则使用明确 Gloo 路径；Windows 的显式 Gloo host 通过版本适配工厂创建。

提供 `primary`、`coordinate(error, stage)`、`gather(value)`、`global_count(local_count)`、`synchronize_model(model)`、`average_gradients(model)` 和 `close()`。

`launch_topology()` 只读取并验证 rank/world/local rank；`selected_backend(...)` 只解析预期 backend 名称，不创建设备或进程组，供启动身份使用。

`StartupCoordinator.open(output)` 在设备/进程组存在前，用共享输出父目录的原子、不可覆盖收据协调启动。每个 rank 有独立 nonce；每阶段报告绑定本次 primary nonce，primary 的放行收据再绑定全部 rank nonce。旧的完整或半次收据、并发重复参与者和不同阶段不能被当作本次就绪，普通错误立即传播，缺失参与者最多等待默认 600 秒。单进程不创建协调文件。

多进程需要共享输出文件系统和唯一 launch ID。`torchrun --standalone` 自动提供 `TORCHELASTIC_RUN_ID`；重启还绑定 `TORCHELASTIC_RESTART_COUNT`。其他 launcher 在 torchrun ID 缺失或为 `none` 时需设置唯一 `KAR_DISTRIBUTED_RUN_ID`，每次新启动换新 ID。收据保留用于防止旧会话重放，不可删除后复用旧 ID。原始进程组创建仍受 PyTorch 的 10 分钟超时约束；该协调器处理进入进程组前的失败，不把底层初始化变成无限等待。

## 合同

所有 rank 以相同顺序参与集体操作；一个 rank 的前向/反向/数据错误应在对应边界协调为 `DistributedStageError`，不能只让主 rank 继续保存。已有 group 必须匹配本次拓扑，context 只销毁自己拥有的 group，避免嵌套 orchestration/trainer 误关父组。

正式更新在累积期间使用 no_sync，最后显式平均梯度一次；配合每 rank 的 `W/global_N` sum 目标，不重复平均或按本地标签数归一化。Gloo 下必要的 CPU transport 不改变数学目标。

## 使用与副作用

这是运行层，不是静态配置。open 可能设置 CUDA、分配控制 Tensor、初始化网络进程组；必须从已授权生产入口调用并用 finally 关闭。不能为调试关闭 SSH/系统安全设置或结束用户进程。

无独立 CLI，由 [trainer](../trainer/README.md)和 [orchestration](../orchestration/README.md)消费。无效拓扑、缺/非有限梯度或其他 rank 失败时停止，不重试部分权重。

## 测试与状态

对应 [production/test_training_contracts.py](../../../../tests/kar/production/test_training_contracts.py) 和 [test_startup_coordination.py](../../../../tests/kar/production/test_startup_coordination.py)。启动协调回归运行短生命周期的 CPU 子进程，只写控制收据，不创建 torch group、查询 CUDA 或更新参数。
