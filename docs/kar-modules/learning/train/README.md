<!-- kar-module: riichienv.kar.learning.train -->
<!-- api: TrainConfig, parameter_groups, create_optimizer, create_scheduler, optimizer_update, evaluate, train_cpu -->
# learning.train — 历史 CPU 诊断与共享优化器原语

源码：[train.py](../../../../src/riichienv/kar/learning/train.py) · [learning 索引](../README.md)

## 定位

`train_cpu` 是有界本地 smoke，不是顶层 `kar train` 的正式实现。后者进入 [trainer](../trainer/README.md)。本模块仍为正式训练复用 AdamW/调度器构造原语。

## 主要接口

- `TrainConfig`：默认 Small、16 updates、有效 batch8、microbatch1；限制一轮、最多100 updates、最多2线程等诊断边界。
- `parameter_groups` / `create_optimizer`：value_head.weight 衰减 .05，其他矩阵 .01，bias/norm 0；AdamW 固定 beta/eps。
- `create_scheduler`：按声明总 updates 的 warmup/cosine，当前 warmup 为 max(100, ceil(.03*updates))。
- `optimizer_update`：有许可的累积/反向/裁剪/step 原语。
- `evaluate`：历史有界 validation 指标；不等同全量 `evaluate_complete`。
- `train_cpu(dataset_dir, output_dir, config, ...)`：诊断训练与检查点/日志产物。

## 执行合同

实际执行含 forward、backward 和 optimizer.step，必须匹配更新许可；叫 smoke 不会变成无训练。固定 manifest rho、完整合法/公开响应 mask、sentinel-safe loss 与统一有效 batch 目标仍须遵守。

`overfit_examples` 有意反复访问少量 train 行，仅供过拟合诊断，不能作为 validation/test curriculum。报告中的 update 数、有限 validation 范围和诊断配置必须保留。

## CLI 与产物

独立 `python -m riichienv.kar.learning.train` 的诊断 opt-in/参数由源码 `main()` 解释。输出 run manifest、更新/验证日志和历史检查点；不启动 GPU/云资源，也不保证费用计量覆盖正式阶段。

## 测试与状态

对应 [m1/test_training.py](../../../../tests/kar/m1/test_training.py)、[production/test_training_contracts.py](../../../../tests/kar/production/test_training_contracts.py)。本轮未运行诊断、更新或测试。历史曾有参数更新的证据不能表述为“从未训练”，也不能升级成当前完整验收。
