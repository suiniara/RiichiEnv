<!-- kar-module: riichienv.kar.native_replay -->
<!-- api: NativeReplayBridge, NativeReplayInterfaceError, NativeReplayContextError -->
# native_replay — 原生重放上下文桥

源码：[native_replay.py](../../../src/riichienv/kar/native_replay.py)、[Rust ReplayContext](../../../riichienv-core/src/replay_context.rs) · [总索引](../README.md)

## 职责

为离线 `apply_event` 提供版本与事件配对协议；麻将上下文由 Rust `ReplayContext` 在原生事件入口直接推进。当前一发、双立直、岭上、待提交杠与已证明未使用的合法 Ron 均由 Rust 持有，KAR 四家手牌完整的原生 MJAI 重放会自动建立抢杠响应窗口。Python 不再保存或回写这些 flag，也不从观察者快照推断它们。

四人原生在线与 MJAI 舍牌共用候选生成及无役和牌形的同巡振听更新。公开客户端的 `?` 手牌不能被当作完整手牌或用于推断未知 Ron；本轮也未新增仅本家手牌完整时的本家自动抢杠响应能力。抢杠见逃在本次提交得到证明后由 Rust 更新，未使用某个合法 Ron 不等于已观察到一个可作为动作监督的私有 Pass。

## 主要接口与顺序

`NativeReplayBridge(env)` 验证接口与 `native_replay_context_v4` 一致；`preview_kan(actor, action)` 在声明前调用 `prepare_replay_kan`，获取原生完整响应并绑定该声明。每个事件严格遵循：

```text
prepare(event, snapshots_before=None)
→ 原生 apply_event
→ after_apply(event)
→ 各观察者推进当前公共账本
```

`prepare` / `after_apply` 必须对应同一事件，不能跳步。兼容调用可以传入四份快照，仅作观察者身份校验，不向 Rust 提供规则 flag；当前审计路径不再为此构造快照。`kan_responses` 返回分离的动作副本；不向调用方暴露模拟未来摸牌。真实合法动作和抢杠预览仅用于合法性审计及冻结机会证据，不得移作公开 response mask。

## 事务合同

桥的暗/加杠审计协议要求先有匹配的完整预览；待抢期间不因旧杠 dora 关闭当前窗口。真正提交时由 Rust 关闭抢杠窗口并取消一发，补牌必须属于相同 actor。预览必须对应声明前的相同前缀，不能跨中间事件复用。

`validate_replay_draw(actor)` 只读验证实际观察到的摸牌是否可以应用。替换摸牌本身可以证明待抢杠窗口已提交，因此审计不能只用 `needs_tsumo` 布尔值拒绝该事件；许可判断由 Rust 处理，Python 继续核对记录中的出牌轮转顺序。查询不生成摸牌、不推进状态，也不生成私有 Pass。

缺接口用 `NativeReplayInterfaceError`，事件/座位/事务不一致用 `NativeReplayContextError`。不通过偷偷清 flag、注入真实响应或过滤赢家来让 replay 通过。

## 测试与状态

对应 [原生规则边界](../../../tests/kar/test_native_rule_boundary.py)、[合法完整半庄及 SQLite 导出](../../../tests/kar/test_native_furiten_export.py)、[原生上下文](../../../tests/kar/test_native_replay.py)及[杠预览](../../../tests/kar/test_kan_preview.py)。版本 `native_replay_context_v4`；旧扩展必须重建，不能回退到 Python 写 flag 或给旧产物改版本。实际验证与迁移要求见[边界迁移记录](../../kar-native-rule-boundary.md)。
