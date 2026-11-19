<!-- kar-module: riichienv.kar.learning.evaluation_startup -->
<!-- api: evaluation_job -->
# learning.evaluation_startup — 冻结评测的持久化准备阶段

源码：[evaluation_startup.py](../../../../src/riichienv/kar/learning/evaluation_startup.py) · [learning 索引](../README.md)

## 接口与边界

`evaluation_job(output, budget, operation, plan_sha256, authorization)` 是校准与 validation tau 选择的共同计费作用域。先核验独立授权、M1/M2 预算及计划身份；之后独占创建新输出目录，持久化 `startup.json`、预算账本和未完成 compute attempt，才允许调用方加载 checkpoint、创建设备 Tensor、核验数据库摘要或构建完整索引。

CLI 用延迟工厂把加载放入作用域。Python API 也允许传入已加载对象，但调用 API 之前发生的外部加载不属于这个作用域，调用方必须另行计费；需要覆盖整个作业时应传工厂。一个作业只持有一个 attempt，准备后先结算已耗时间并核验余额，再开始前向或对局。估算应覆盖整个作业，不能只测内核耗时。

## 失败与恢复规则

普通异常和可捕获中断会记录最后一段实际费用并关闭 attempt；超预算同样先记录实际费用，不将已发生开销丢弃。硬退出留下 pending，后续必须依据外部实际账单调用预算账本的 `reconcile_compute`，不能把新进程的单调时钟当作旧进程免费结束的证据。

这两个作业暂不恢复部分 logits 或部分 tau 选择。任何已存在的输出目录都会在工厂执行前拒绝，包括准备中断、校准失败和已完成目录；即使外部对账完成，也不会重新写入同一目录或覆盖旧 logits、费用和结果。对账只解决费用的未完成状态，不能凭此证明计算产物可续接。重新执行需要新的明确作业输出和含历史费用的预算登记，不能换目录重置阶段开销。

## 验证

[test_frozen_evaluation_startup.py](../../../../tests/kar/production/test_frozen_evaluation_startup.py) 使用假加载器、模拟时钟和主动硬退出子进程验证加载前持久化、单次记费、预算提前拒绝、pending 保留和对账后仍不覆盖输出。[test_calibration_budget_admission.py](../../../../tests/kar/production/test_calibration_budget_admission.py) 使用实际 SQLite 摘要/索引准备验证准备开销、失败结算和超预算时不进入模型前向。它们不执行正式评测或更新参数。
