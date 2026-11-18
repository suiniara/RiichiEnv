<!-- kar-module: riichienv.kar.native_replay -->
<!-- api: NativeReplayBridge, NativeReplayInterfaceError, NativeReplayContextError -->
# native_replay — 原生重放上下文桥

源码：[native_replay.py](../../../src/riichienv/kar/native_replay.py) · [总索引](../README.md)

## 职责

为离线 `apply_event` 重放补齐由前缀事实推导的一发、双立直和岭上。四人原生在线与原生 MJAI 舍牌共用候选生成及无役和牌形的同巡振听更新；MJAI 在普通摸牌或成立鸣牌关闭窗时原生记录未使用的合法 Ron。Python 桥不保存或回写振听标志，只把原生杠预览中未使用的 Ron 在已证明提交时交给 Rust 同一转换；不重写隐藏手牌合法性。

## 主要接口与顺序

`NativeReplayBridge(env)` 验证原生薄接口存在；`preview_kan(actor, action)` 在声明前获取完整响应并绑定该声明。每个事件严格遵循：

```text
prepare(event, 四份 snapshots_before)
→ 原生 apply_event
→ after_apply(event)
→ 各观察者推进当前公共账本
```

`prepare` / `after_apply` 必须对应同一事件，不能跳步。`kan_responses` 返回复制的动作集合；不向调用方暴露模拟未来摸牌。`_get_legal_actions` 仅用于原生合法性上下文/响应预览，不得移作模型公开 response mask。

## 事务合同

暗/加杠必须先有匹配的完整预览；待抢期间不因旧杠 dora 关闭当前窗口。真正提交时关闭抢杠窗口并取消一发，补牌必须属于相同 actor。四份快照必须就 pending actor 达成一致。

缺接口用 `NativeReplayInterfaceError`，事件/座位/事务不一致用 `NativeReplayContextError`。不通过偷偷清 flag、注入真实响应或过滤赢家来让 replay 通过。

## 测试与状态

对应 [合法完整半庄及 SQLite 导出](../../../tests/kar/test_native_furiten_export.py)、[原生上下文](../../../tests/kar/test_native_replay.py)及[杠预览](../../../tests/kar/test_kan_preview.py)。版本 `prefix_flags_bridge_v3_native_furiten`；旧扩展缺少负责抢杠见逃的薄接口，须重新构建后才能运行审计或导出，不能让旧扩展以新合同继续生产。
