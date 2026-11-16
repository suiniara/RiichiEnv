<!-- kar-module: riichienv.kar.native_replay -->
<!-- api: NativeReplayBridge, NativeReplayInterfaceError, NativeReplayContextError -->
# native_replay — 原生重放上下文桥

源码：[native_replay.py](../../../src/riichienv/kar/native_replay.py) · [总索引](../README.md)

## 职责

为离线 `apply_event` 重放补齐由前缀事实推导的一发、双立直、岭上、漏和等原生上下文。复用冻结引擎的完整杠响应预览，不在 Python 中重新实现隐藏手牌合法性，也不修改 Rust Core 规则。

## 主要接口与顺序

`NativeReplayBridge(env)` 验证原生薄接口存在；`preview_kan(actor, action)` 在声明前获取完整响应并绑定该声明。每个事件严格遵循：

```text
prepare(event, 四份 snapshots_before)
→ 原生 apply_event 与对应观察者推进
→ after_apply(event)
```

`prepare` / `after_apply` 必须对应同一事件，不能跳步。`kan_responses` 返回复制的动作集合；不向调用方暴露模拟未来摸牌。`_get_legal_actions` 仅用于原生合法性上下文/响应预览，不得移作模型公开 response mask。

## 事务合同

暗/加杠必须先有匹配的完整预览；待抢期间不因旧杠 dora 关闭当前窗口。真正提交时关闭抢杠窗口并取消一发，补牌必须属于相同 actor。四份快照必须就 pending actor 达成一致。

缺接口用 `NativeReplayInterfaceError`，事件/座位/事务不一致用 `NativeReplayContextError`。不通过偷偷清 flag、注入真实响应或过滤赢家来让 replay 通过。

## 测试与状态

对应 [test_native_replay.py](../../../tests/kar/test_native_replay.py)、[test_kan_preview.py](../../../tests/kar/test_kan_preview.py)、[test_review_kan_transactions.py](../../../tests/kar/test_review_kan_transactions.py)。版本 `prefix_flags_bridge_v2`。无独立 CLI；本轮未创建原生环境或执行重放。
