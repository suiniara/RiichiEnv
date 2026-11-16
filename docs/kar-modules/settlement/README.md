<!-- kar-module: riichienv.kar.settlement -->
<!-- api: WinFacts, SettlementError, SettlementUnavailableError, verify_settlements -->
# settlement — 独立番符与分数账本核验

源码：[settlement.py](../../../src/riichienv/kar/settlement.py) · [总索引](../README.md)

## 职责与接口

`verify_settlements(events, observed_final_scores=None)` 重建独立点数、立直棒、本场、和牌/流局及杠相关事实；用 HandEvaluator 评估真实和牌，再通过 [payments](../payments/README.md)计算款项。`WinFacts` 保存和牌所需的前缀事实。

这是离线审计器，不是策略价值头。原始全知手牌可用于独立核验，但不能把它们转成模型输入或公开 response mask。

## 比较而不修复

服务器番符、支付、header 或下一局起始分数是比较目标，不是反推独立计算的依据。缺原始里宝/公开揭示或规则事实时保持不可核验；省略 delta 不自动填零。相邻的多人荣和一起处理，不能让后一个赢家失去共享窗口信息。

杠沿用公共事务判据：未提交的加杠不提前进入已成立 meld；旧杠的延迟 dora 不关闭新抢杠窗口。

## 输出与失败分类

结果包含独立结算列表、计算终分、来源比较和错误。`SettlementUnavailableError` 对应证据不足/BLOCKED，`SettlementError` 或数据矛盾对应 FAIL。最终分数相等本身不是完整半庄证书；特别是缺显式 end_game 时，比较只能按其证据角色报告。

由 [audit](../audit/README.md)调用，再与 [completion](../completion/README.md)共同完成 gate。没有独立 CLI，不更新模型，也不改原始记录。

## 测试与状态

对应 [test_settlement.py](../../../tests/kar/test_settlement.py)、[test_payments.py](../../../tests/kar/test_payments.py)、[test_draw_obligations.py](../../../tests/kar/test_draw_obligations.py)。本轮未执行独立结算或真实数据核验。
