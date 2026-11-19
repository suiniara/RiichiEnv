<!-- kar-module: riichienv.kar.observation -->
<!-- api: PublicMeld, PublicSnapshot, SelfPassTrigger, ObserverLedger, project_observer, extract_rule_features -->
# observation — 观察者账本与冻结快照

源码：[observation.py](../../../src/riichienv/kar/observation.py)、[Rust ObserverState](../../../riichienv-core/src/observer.rs) · [总索引](../README.md) · [原生规则边界](../../kar-native-rule-boundary.md)

## 接口

`ObserverLedger(observer, initial_order)` 封装 Rust `NativeObserverState`；`apply_event/snapshot/check_observation` 推进事实、冻结副本并核对原生本家手牌。手牌、公开舍牌历史/当前河、副露、分数、立直、阶段和杠事务由 Rust 持有，Python 读取分离的不可变投影，不再另写一份麻将状态转换。`PublicMeld` 表示已成立 meld，`PublicSnapshot` 不包含他家暗手。`project_observer` 只处理指定公共 cutoff；`extract_rule_features` 保留不同完整性依赖。

`ObserverState` 不接 `GameState`，不查询他家真实合法动作或牌山。它只保留所选观察者的私有牌，隐藏其他初始手牌与摸牌，并过滤对手暗杠的赤牌组成。来源端必须先验证终局 `public_hand(s)` 可公开；投影器不独立证明服务器揭牌真实性，也不补造缺失手牌。原生归约器拒绝事件时不提交部分事实。

`pass_trigger()` 返回本家的不可变 `SelfPassTrigger`，不是提交成功。`self_pass_ready(trigger, provenance=...)` 区分有合法关闭锚与仍开放等待；非法 owner/trigger/provenance 或已失去关闭锚时报错。`record_self_pass` 仍需真实已关闭且已确认的窗口，同一 trigger 重复 no-op。`apply_event(type=self_pass)` 同样校验 owner/source/confirmed/provenance/确认时间，失败不推进状态。

## 私有历史与公共域

Python 继续拥有模型历史、公共语义时钟、cutoff 和来源，以及本家 Pass 的确认记录。`event_id/cutoff_event_id` 始终是公共流下标，source record 编号不决定模型时间；所有账本共用 [event_semantics](../event_semantics/README.md) 的因果语义时钟，快照显式绑定 public_clock_version，旧版本不静默补齐。私有 SELF_PASS 仅插入本家 `events/history_positions` 的公共确认锚之后，使用该锚的逻辑位置，不推进公共计数或时钟；复合事件数组下标不再等于公共 ID。既有快照不改写，其他三个观察者逐字节不变。

来源只支持原生成功提交或冻结的 unique-next-draw 推断；不能推补隐藏意图。私有 marker 不广播到公共 trace。API 是可信调用方记录合同，不是数字认证。

## 规则与事务

舍牌历史与当前河分开；被叫走的牌保留历史但不重复物理计数。known_counts 仍要求 physical 和 visibility 完整，历史特征不被连带清空。起家顺同分破平不随当前庄家变化。

暗/加杠 pending 与提交由 Rust 区分；加杠在提交前必须保留原 PON。观察投影与重放使用同一原生公开提交判据，旧杠的延迟 dora 不能提交新杠。Pass 使用该观察者已过滤的公开触发牌，不泄露对手暗杠赤牌组成。

## 测试与边界

对应 [test_native_observer_boundary.py](../../../tests/kar/test_native_observer_boundary.py)、[test_self_pass_history.py](../../../tests/kar/test_self_pass_history.py)、[test_observation_response.py](../../../tests/kar/test_observation_response.py)、[test_token_cache.py](../../../tests/kar/test_token_cache.py)。本轮构建及测试结果见[边界迁移记录](../../kar-native-rule-boundary.md#7-验证位置与当前证据)；测试入口不代表真实数据已重新导出或验收。
