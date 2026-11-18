<!-- kar-module: riichienv.kar.live_events -->
<!-- api: LiveEventProjection -->
# live_events — 原生日志增量到观察者事件

源码：[live_events.py](../../../src/riichienv/kar/live_events.py) · [总索引](../README.md)

## 职责与接口

`LiveEventProjection()` 管理公共规范化 trace 与四份观察者账本。`capture_self_passes(choices)` 仅捕获上下文，无副作用；原子提交成功后将结果传给 `observe_native(..., confirmed_self_passes=...)`。公共批次完整处理后仅在各自账本的确认锚插入 marker；公共流拒绝 raw self_pass。`snapshot(seat)` 返回该观察者状态。

全局规范化 trace 用于存档/协调；模型应只消费本家的 snapshot，不能直接把全局 trace 当作任意观察者都可见的输入。

## 暗杠和多人和牌

合法可抢暗杠窗口可能已进入 WaitResponse，但原生暂不写声明。仅在调用方提供已选定并成功执行的完整暗杠、且原生实际处于相应窗口时补声明；后续迟到日志转换为 `kan_commit`，不重复声明。

多荣和同一窗口共享规范语义裁定时钟，不由native batch决定。跨批次暂存不完整hora组，实际边界到达后才聚合公开支付；期间拒绝决策snapshot，不回写旧快照。赢家公开手牌只在合法公开结算后从此前本家账本恢复，不提前传播私有牌。九种九牌与离线共用宣布者/首摸/牌张校验，已证零结算生成SCORE_DELTA；source provenance不成为模型输入。

## 使用与错误

由 [agent](../agent/README.md)在执行动作后推进。不能把最后一小段日志误当完整历史，也不能重复喂已消费事件。完整声明/确认不匹配、分数链矛盾或没有可验证推进时应 fail-closed，不能为了继续仿真制造事件。

版本为 `native_public_resolution_projection_v4_semantic_clock`。本模块不选择策略，不创建优化器，也没有独立 CLI；它依赖 [observation](../observation/README.md)和公共杠判据。

## 测试与状态

对应 [test_self_pass_history.py](../../../tests/kar/test_self_pass_history.py) 与 [core/test_live_agent.py](../../../tests/kar/core/test_live_agent.py)。本轮未启动生产阶段或独立棋力评测；默认回归中的有界原生小局/半庄与响应提交只是测试，实际范围见[修复报告](../../kar-semantic-branch-repair.md)。
