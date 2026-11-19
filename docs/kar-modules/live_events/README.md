<!-- kar-module: riichienv.kar.live_events -->
<!-- api: LiveEventProjection -->
# live_events — 原生规则事件增量到观察者历史

源码：[live_events.py](../../../src/riichienv/kar/live_events.py) · [总索引](../README.md)

## 职责与接口

`LiveEventProjection()` 消费四人原生 `env.rule_log` 的增量，管理规范 trace 与四份观察者账本。完整声明、提交、终局分数和实际应公开手牌由 Rust 状态机输出；Python 不另行补事件或决定麻将规则。legacy `mjai_log` 接口仍保留，但不能替代 KAR 所需的完整规则事件。

`capture_self_passes(choices)` 只捕获上下文；成功原子提交后才能将 receipt 传给 `observe_native(..., confirmed_self_passes=...)`，允许空事件回调。有效但尚未观察到关闭锚的 receipt 按条保留，已写入/幂等才移除。非法 receipt 报错，不吞掉其他待办。暗/加杠未被抢时只用岭上摸牌作锚，不用 ACK 或 Dora。各账本只在实际关闭锚插入本家 marker，公共流拒绝 raw self_pass。`snapshot(seat)` 返回观察者状态。

全局规则日志与规范 trace 用于存档/协调，可能包含四家初始手牌或私有摸牌；它们不是任意观察者都可见的输入。模型只能消费经 Rust `ObserverState` 过滤的本家 snapshot。Python 保留批次、来源、公共语义时钟及 receipt 协调，不成为另一个全知可见性规则层。

## 暗杠和多人和牌

合法暗杠在进入抢杠窗口时由 `rule_log` 输出声明，未被抢时由原生输出 `kan_commit`，不重复声明。兼容参数 `declared_ankan` 仅检查日志已包含相同声明；缺失时明确失败并要求当前扩展，不再根据调用方选中动作补造声明。

多荣和同一窗口共享规范语义裁定时钟，不由 native batch 决定。跨批次暂存不完整 hora 组，实际边界到达后才聚合公开支付并核对原生分数；期间拒绝决策 snapshot，不回写旧快照。赢家公开手牌直接取已经结算的原生 `public_hand`；普通荒牌、流局满贯和九种九牌使用原生 `public_hands`，Python 不再调用 `HandEvaluator` 推算应公开者或从此前账本恢复揭牌。九种九牌仍只公开宣布者，已证零结算生成 SCORE_DELTA；source provenance 不成为模型输入。

分数与供托由 Rust 观察状态跟随原生结算推进；Python 验证 delta/scores 的连续性，终局供托也只核对原生结果，不再自行分配。

## 使用与错误

由 [agent](../agent/README.md)在执行动作后推进。不能把最后一小段日志误当完整历史，也不能重复喂已消费事件。完整声明/确认不匹配、分数链矛盾或没有可验证推进时应 fail-closed，不能为了继续仿真制造事件。

版本为 `native_rule_event_projection_v6`。本模块不选择策略，不创建优化器，也没有独立 CLI；它依赖 [observation](../observation/README.md)、完整原生规则日志和公共杠判据。

## 测试与状态

对应 [test_pass_receipt_batches.py](../../../tests/kar/test_pass_receipt_batches.py)、[test_self_pass_history.py](../../../tests/kar/test_self_pass_history.py) 与 [core/test_live_agent.py](../../../tests/kar/core/test_live_agent.py)。本轮验证范围与旧产物要求见[边界迁移记录](../../kar-native-rule-boundary.md)；历史分批 Pass 报告只覆盖其原源码和事件接口，不自动替代当前规则日志验收。
