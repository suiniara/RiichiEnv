<!-- kar-module: riichienv.kar.learning.evaluation -->
<!-- api: add_diagnostics, cal_fit_loss_summary, evaluate_complete -->
# learning.evaluation — 全 validation 报告与选择隔离

源码：[evaluation.py](../../../../src/riichienv/kar/learning/evaluation.py) · [learning 索引](../README.md)

## 主要接口

`evaluate_complete(model, dataset, context, microbatch=..., use_rule_features=..., ...)` 对非空 validation 的所有真实决策做 no_grad 前向；DDP 分片不重复、不遗漏，最后聚合总量/游戏 clusters。

`add_diagnostics(...)` 从三头输出与独立标签累计完整统计；可用 `public_fields` 提供未遮蔽公开状态，使 feature-off 的当前 rank/威胁切片不丢失。`cal_fit_loss_summary(outputs, batch)` 单独计算用于模型选择的固定角色损失。

## 数据角色与损失

完整 validation 包括 cal_fit 与 cal_check，继续报告自然分布诊断；返回的 `model_selection` **只含 cal_fit**，固定角色 salt，明确 `cal_check_used=false`。第三轮不能用外层全 validation 均值替代该子记录。

选择目标使用 train 固定 rho 与训练 T=1 loss 合同，不按 validation 有效比例重算。无有效目标不填零。可选标定诊断使用绑定的 calibration/support 和其原始后端，不能改设备后冒充同一次标定。

## 模式与可观测结果

返回全量计数、三头条件 NLL、完整 reliability/branch/utility/CI/gate 分布、角色化 model_selection 及标定状态；关闭随机特征置空。forward 结束（包括失败）恢复模型原训练模式。

数据必须是 validation；test 不能传给训练选择接口。全量报告只证明该调用实际覆盖的已核验数据，不把 probe subset 冒充正式全数据。

## 测试与状态

对应 [production/test_full_trainer.py](../../../../tests/kar/production/test_full_trainer.py)、[test_selection_and_roles.py](../../../../tests/kar/production/test_selection_and_roles.py)、[test_diagnostic_distributions.py](../../../../tests/kar/production/test_diagnostic_distributions.py)。本轮未执行前向、验证或统计。无独立 CLI，由已授权 trainer 调用。
