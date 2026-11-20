<!-- kar-module: riichienv.kar.learning.evaluation -->
<!-- api: add_diagnostics, cal_fit_loss_summary, evaluate_complete -->
# learning.evaluation — 全 validation 报告与选择隔离

源码：[evaluation.py](../../../../src/riichienv/kar/learning/evaluation.py) · [learning 索引](../README.md)

## 主要接口

`evaluate_complete(model, dataset, context, microbatch=..., use_rule_features=..., ...)` 对非空 validation 的所有真实决策做 no_grad 前向；DDP 分片不重复、不遗漏。只向 primary 汇集局部统计并生成一份全局报表，协调异常后把同一报告广播给所有 rank。`timings_seconds` 分列最大 rank 的本地前向阶段、primary 汇总及 bootstrap。

`add_diagnostics(...)` 从三头输出与独立标签累计完整统计；可用 `public_fields` 提供未遮蔽公开状态，使 feature-off 的当前 rank/威胁切片不丢失。`cal_fit_loss_summary(outputs, batch)` 单独计算用于模型选择的固定角色损失。

## 数据角色与损失

完整 validation 包括 cal_fit 与 cal_check，继续报告自然分布诊断；返回的 `model_selection` **只含 cal_fit**，固定角色 salt，明确 `cal_check_used=false`。第三轮不能用外层全 validation 均值替代该子记录。

选择目标使用 train 固定 rho 与训练 T=1 loss 合同，不按 validation 有效比例重算。无有效目标不填零。标定诊断要求匹配的 calibration/support、原始后端及冻结 IEEE FP32；不会继承外层 autocast，拒绝显式 BF16。所有显式 FP32 验证均检查实际数值后端，包括尚未加载校准的模型选择验证；非 IEEE 或前向期间后端改变时拒绝。未校准验证仍可用已配置的 BF16。

## 模式与可观测结果

返回全量计数、三头条件 NLL、完整 reliability/branch/utility/CI/gate 分布、角色化 model_selection 及标定状态；关闭随机特征置空。forward 结束（包括失败）恢复模型原训练模式。

`response_valid` 行的 `unsupported_mass_histogram` 用冻结温度、公开 mask 与运行时相同的标量概率计算，再按 `EventSupport.value_supported(...)` 求和。逐事件 `response_event_calibration` 在公开允许该事件的行上记录正负例、正例独立游戏数和概率指标；不在 evaluation 重新判定准入。没有绑定的校准时有效未支持质量为 `unavailable`；可单列 `train_only_unsupported_mass_histogram`。报告及分布式合并状态使用 `whole_game_cluster_statistics_v4_event_ovr`，拒绝旧格式。

已校准路径逐行复用 `FrozenModelPredictor.policy_logits_from_rows`，在 DECIDE 处裁掉 teacher ACTION 与尾部 padding，按部署相同的 batch size 1 前缀计算策略 logits；response/rank 仍保留批处理。未经校准的训练验证继续原有批量前向，不承担额外策略前缀调用。

校准后的动作 gate 复用 calibration/runtime 的 `legal_prior` 标量归一化和 `support_discards`，按包含非舍牌动作的完整合法先验筛选，并以 `math.fsum` 累计支持质量；阈值处仍使用原有的包含边界比较，不添加容差。FP32 `log_softmax` 继续用于损失诊断，但其归一化舍入结果不决定 gate 成员或支持质量。严格策略相等限于相同模型、前缀、设备和实现，不能推为三头或跨后端逐位相同。

数据必须是 validation；test 不能传给训练选择接口。全量报告只证明该调用实际覆盖的已核验数据，不把 probe subset 冒充正式全数据。

## 测试与状态

对应 [production/test_full_trainer.py](../../../../tests/kar/production/test_full_trainer.py)、[test_selection_and_roles.py](../../../../tests/kar/production/test_selection_and_roles.py)、[test_diagnostic_distributions.py](../../../../tests/kar/production/test_diagnostic_distributions.py)。[test_calibrated_gate_parity.py](../../../../tests/kar/production/test_calibrated_gate_parity.py) 用合成 logits 验证等概率阈值、非单位温度、非法动作遮罩及非舍牌先验质量在校准、验证、实际 RuntimePolicy 三条路径中的一致性；[test_calibration_prefix_parity.py](../../../../tests/kar/production/test_calibration_prefix_parity.py) 用真实小模型前向验证 microbatch=3 的已校准完整验证与部署在拟合阈值处支持集合相同。无独立 CLI，由已授权 trainer 调用；这些回归不构成正式校准或棋力实验。
