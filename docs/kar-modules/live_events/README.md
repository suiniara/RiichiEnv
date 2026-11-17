<!-- kar-module: riichienv.kar.live_events -->
<!-- api: LiveEventProjection -->
# live_events — 原生日志增量到观察者事件

源码：[live_events.py](../../../src/riichienv/kar/live_events.py) · [总索引](../README.md)

## 职责与接口

`LiveEventProjection()` 管理公共规范化 trace 与四份观察者账本。`capture_self_passes(choices)` 仅捕获上下文，无副作用；原子提交成功后将结果传给 `observe_native(..., confirmed_self_passes=...)`。公共批次完整处理后仅在各自账本的确认锚插入 marker；公共流拒绝 raw self_pass。`snapshot(seat)` 返回该观察者状态。

全局规范化 trace 用于存档/协调；模型应只消费本家的 snapshot，不能直接把全局 trace 当作任意观察者都可见的输入。

## 暗杠和多人和牌

合法可抢暗杠窗口可能已进入 WaitResponse，但原生暂不写声明。仅在调用方提供已选定并成功执行的完整暗杠、且原生实际处于相应窗口时补声明；后续迟到日志转换为 `kan_commit`，不重复声明。

多荣和同一窗口共享逻辑 record 组。赢家公开手牌只在合法公开结算后从此前本家账本恢复，不提前把私有手牌传播给其他观察者。里宝、流局揭示和终局供托也保持明确的公开时点。

## 使用与错误

由 [agent](../agent/README.md)在执行动作后推进。不能把最后一小段日志误当完整历史，也不能重复喂已消费事件。完整声明/确认不匹配、分数链矛盾或没有可验证推进时应 fail-closed，不能为了继续仿真制造事件。

版本为 `native_public_resolution_projection_v3_private_pass`。本模块不选择策略，不创建优化器，也没有独立 CLI；它依赖 [observation](../observation/README.md)和公共杠判据。

## 测试与状态

对应 [test_self_pass_history.py](../../../tests/kar/test_self_pass_history.py) 与 [core/test_live_agent.py](../../../tests/kar/core/test_live_agent.py)。本轮未运行完整半庄/阶段任务；仅有界原生响应提交、失效与隐私对拍。
