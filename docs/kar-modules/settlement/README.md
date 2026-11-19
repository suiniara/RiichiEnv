<!-- kar-module: riichienv.kar.settlement -->
<!-- api: WinFacts, SettlementError, SettlementUnavailableError, verify_settlements -->
# settlement — 独立番符与分数账本核验

源码：[settlement.py](../../../src/riichienv/kar/settlement.py) · [总索引](../README.md)

## 职责与接口

`verify_settlements(events, observed_final_scores=None)` 重建独立点数、立直棒、本场、和牌/流局及杠相关事实；用 Rust `HandEvaluator` 评估真实和牌，再通过 [payments](../payments/README.md)调用共享原生支付。`WinFacts` 保存和牌所需的前缀事实。听牌与等待查询显式传入冻结雀魂规则；下一局庄家、风局、本场和整场终止由 `native_round_transition(..., rule=self.ruleset.engine_rule())` 计算，与采用相同规则的四人在线状态机共用 [Rust 结算内核](../../../riichienv-core/src/rules/settlement.rs)。

这是离线审计器，不是策略价值头。“独立”指事实重建和计算不以服务器标签或原生 `apply_event` 分数为依据，不意味着保留另一份 Python 付款/终局公式。原始全知手牌可用于独立核验，但不能把它们转成模型输入或公开 response mask。

## 比较而不修复

服务器番符、支付、header 或下一局起始分数是比较目标，不是反推独立计算的依据。缺原始里宝/公开揭示或规则事实时保持不可核验；省略 delta 不自动填零。相邻的多人荣和一起处理，不能让后一个赢家失去共享窗口信息。

杠沿用共享的原生公共事务判据：未提交的加杠不提前进入已成立 meld；旧杠的延迟 dora 不关闭新抢杠窗口。Python 审计仍保存来源证据与和牌前缀事实，不将这些审计副本回写到原生规则状态。

本模块固定雀魂：本家碰加暗手第四枚的单骑不计形式听牌；西入普通荒牌结算达三万点时，不因庄家听牌但非第一而继续。天凤对这两个案例保留不同结果，原生通过 `allow_fifth_copy_tenpai`、`extension_dealer_repeat_priority` 区分，不能把雀魂修复套到通用默认值。无 rule 的旧 `HandEvaluator` API 保持天凤兼容，KAR 显式传 `MJSOUL_RULESET.engine_rule()`。这些变化可能改变听牌支付、完成证书和最终名次监督，旧报告不能只修改版本或规则 hash 后继续使用；[官方来源与差异表](../../kar-native-rule-boundary.md#3-平台差异与-kar-雀魂修复)说明适用范围。

## 输出与失败分类

结果包含独立结算列表、计算终分、来源比较和错误。`SettlementUnavailableError` 对应证据不足/BLOCKED，`SettlementError` 或数据矛盾对应 FAIL。最终分数相等本身不是完整半庄证书；特别是缺显式 end_game 时，比较只能按其证据角色报告。

由 [audit](../audit/README.md)调用，再与 [completion](../completion/README.md)共同完成 gate。没有独立 CLI，不更新模型，也不改原始记录。

## 测试与状态

版本为 `native_mjsoul_settlement_v2`。对应 [test_settlement.py](../../../tests/kar/test_settlement.py)、[test_payments.py](../../../tests/kar/test_payments.py)、[test_draw_obligations.py](../../../tests/kar/test_draw_obligations.py)、[test_native_settlement_kernel.py](../../../tests/kar/test_native_settlement_kernel.py)。实际构建/测试范围和重新验收要求见[边界迁移记录](../../kar-native-rule-boundary.md)。
