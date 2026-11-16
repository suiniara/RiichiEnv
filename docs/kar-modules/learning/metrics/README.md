<!-- kar-module: riichienv.kar.learning.metrics -->
<!-- api: MetricTotals -->
# learning.metrics — 历史有界诊断指标累计

源码：[metrics.py](../../../../src/riichienv/kar/learning/metrics.py) · [learning 索引](../README.md)

## 职责与接口

`MetricTotals` 是历史 CPU smoke 的轻量累计器。`add(outputs, batch, slices=True)` 在 no_grad 下复用 loss 合同检查，累计每头有效数量、NLL、Brier，并按公开对手立直数切片；`report()` 返回自然频率均值及数量。

action 使用完整合法集合，response 先用有效位选取允许 mask，rank 只使用观察到的真响应分支。无有效目标时均值是 None，不是已达到零误差。

## 与正式诊断的区别

它不输出完整 per-class 可靠性、21 分支游戏级区间或所有门控分布，也不是 calibrated-policy evidence。正式生产使用 [statistics.FullDiagnostics](../../statistics/README.md) 和 [evaluation](../evaluation/README.md)，不要用简化报告替代独立 cal_check。

`_subset` 是内部组批辅助，不是可忽略输入/标签一致性的独立 API。`MetricTotals.add` 不调用模型 forward 或 optimizer，但会执行 Tensor 指标计算。

## 使用与接线

由历史 [train](../train/README.md)的 `evaluate` 及诊断流程消费。需要 Torch，不提供独立 CLI，不写原始数据，不决定正式 epoch 或 tau。

## 测试与状态

相关来源回归见 [m1/test_training.py](../../../../tests/kar/m1/test_training.py)、[m1/test_losses.py](../../../../tests/kar/m1/test_losses.py)。本轮未执行指标累计或测试；历史 smoke 指标不代表当前完整产品验收。
