<!-- kar-module: riichienv.kar.observation -->
<!-- api: PublicMeld, PublicSnapshot, SelfPassTrigger, ObserverLedger, project_observer, extract_rule_features -->
# observation — 观察者账本与冻结快照

源码：[observation.py](../../../src/riichienv/kar/observation.py) · [总索引](../README.md) · [输入链路合同](../../kar-input-chain-repair.md)

## 接口

`ObserverLedger(observer, initial_order)` 重建本家白名单状态；`apply_event/snapshot/check_observation` 推进事实、冻结副本并核对原生本家手牌。`PublicMeld` 表示已成立 meld，`PublicSnapshot` 不包含他家暗手。`project_observer` 只处理指定公共 cutoff；`extract_rule_features` 保留不同完整性依赖。

`pass_trigger()` 返回本家的不可变 `SelfPassTrigger`，只是上下文，不能当作提交成功。`record_self_pass(trigger, provenance=...)` 只记录已知且已关闭窗口的确认 Pass；同一 trigger 重复 no-op。`apply_event(type=self_pass)` 同样要求 owner/source/confirmed/provenance/确认时间匹配，验证失败不推进状态。

## 私有历史与公共域

`event_id/cutoff_event_id` 始终是公共流下标，source record 编号不决定模型时间；所有账本共用 [event_semantics](../event_semantics/README.md) 的因果语义时钟，快照显式绑定 public_clock_version，旧版本不静默补齐。私有 SELF_PASS 仅插入本家 `events/history_positions` 的公共确认锚之后，使用该锚的逻辑位置，不推进公共计数或时钟；复合事件数组下标不再等于公共 ID。既有快照不改写，其他三个观察者逐字节不变。

来源只支持原生成功提交或冻结的 unique-next-draw 推断；不能推补隐藏意图。私有 marker 不广播到公共 trace。API 是可信调用方记录合同，不是数字认证。

## 规则与事务

舍牌历史与当前河分开；被叫走的牌保留历史但不重复物理计数。known_counts 仍要求 physical 和 visibility 完整，历史特征不被连带清空。起家顺同分破平不随当前庄家变化。

暗/加杠 pending 与提交分开；加杠必须保留原 PON。Pass 使用该观察者已过滤的公开触发牌，不泄露对手暗杠赤牌组成。

## 测试与边界

对应 [test_self_pass_history.py](../../../tests/kar/test_self_pass_history.py)、[test_observation_response.py](../../../tests/kar/test_observation_response.py)、[test_token_cache.py](../../../tests/kar/test_token_cache.py)。本轮没有阶段训练、backward、optimizer.step 或重导出；只做有界合成/原生响应及既有有限缓存回归。
