<!-- kar-module: riichienv.kar.learning.production_checkpoint -->
<!-- api: rank_rng_state, restore_rank_rng, save_production_checkpoint, read_production_checkpoint -->
# learning.production_checkpoint — 生产代际检查点

源码：[production_checkpoint.py](../../../../src/riichienv/kar/learning/production_checkpoint.py) · [learning 索引](../README.md)

## 职责与接口

`rank_rng_state(device)` / `restore_rank_rng(state, device)` 处理每 rank 的 Python/NumPy/Torch/已初始化 CUDA 随机状态。`save_production_checkpoint(...)` 保存模型、optimizer、scheduler、完整 release contract、trainer/sampler/费用状态、每 rank RNG、runtime 与 step。

`read_production_checkpoint(path, contract, runtime)` 核验逻辑 sidecar 指向的已提交 generation，受限读取并返回 payload；由 trainer 再恢复到实际对象。

## 原子发布与恢复

每次写新的 `checkpoint.g...` payload，完成并获得 hash 后只原子切换一个 JSON 指针，保留上一代完整数据；不让同名 payload 与 sidecar 分别覆盖出混合版本。路径必须留在当前运行目录、符合 generation 约定，不能通过 sidecar 跳到外部文件。

精确 resume 要求模型/Schema/数据/recipe/代码/native/设备/精度/world size 等 runtime 一致，不能 CUDA→CPU 或改变并行布局后仍称精确恢复。只使用可信产物和 `weights_only` 读取边界。

## 与完成发布的关系

检查点是权威训练状态；`finished=true` 的 generation 已存在而 run_manifest 尚未发布时，恢复应交给 [run_publication](../run_publication/README.md)补发布，不再更新权重。费用账本只能延续已保存锚点，不能恢复成新空账本。

无独立 CLI。保存/读取/恢复涉及文件、Tensor、优化器和 RNG 副作用，需在上层已批准运行中使用，不是模型元数据预览工具。

## 测试与状态

对应 [production/test_training_contracts.py](../../../../tests/kar/production/test_training_contracts.py)、[test_full_trainer.py](../../../../tests/kar/production/test_full_trainer.py)。本轮未保存、加载或恢复检查点，也未运行更新测试。
