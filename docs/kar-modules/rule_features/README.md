<!-- kar-module: riichienv.kar.rule_features -->
<!-- api: RankFeatures, genbutsu, full_suji_exclusion, no_visible_possible_ryanmen_shape, known_count_base34, public_dora_types, current_rank_features -->
# rule_features — 确定性公开规则特征

源码：[rule_features.py](../../../src/riichienv/kar/rule_features.py) · [总索引](../README.md)

## 职责与接口

| 接口 | 语义 |
|---|---|
| `genbutsu(target, discard_history)` | 仅判断目标玩家自己的公开舍牌历史 |
| `full_suji_exclusion(target, history)` | 所有另一侧等待牌均满足，不能把 half-suji 当 full-suji |
| `no_visible_possible_ryanmen_shape(target, known_counts)` | 可见数量是否排除表内全部合法两面形状 |
| `known_count_base34(...)` | 本家暗手、当前河、副露、宝牌等物理区域的 base34 计数 |
| `public_dora_types(indicators)` | 指示牌到公开宝牌牌种 |
| `current_rank_features(scores, initial_order, observer)` | 固定起家顺破平的 `RankFeatures`：rank、上下相邻差与 present 位 |

## 不能混淆的边界

- Ryanmen 表中，目标 3 不包含 `(1,2)` 边张，目标 7 不包含 `(8,9)` 边张。
- 玩家暗杠/加杠含某牌不代表该玩家打过它，不形成 Genbutsu。
- 被叫走的舍牌保留在历史判断中，但不再留在当前河的物理区重复计数。
- 一位没有更高相邻名次，四位没有更低相邻名次；相邻同分的差是 present 的 0，而不是 absent。
- 证据未知可返回 `None`，不能把未知强改为 false/true。排除两面形状不等于排除所有和牌形状，更不是“绝对安全”。

## 使用与调用关系

这是纯规则函数层，不查询他家私有合法动作，不使用终局名次标签。推荐由 `observation.extract_rule_features` 在白名单状态上调用，由 tokenizer 写入适用的 HAND_COUNT/ROUND_STATE 字段。牌编码应先规范化到本接口约定的 base34，不能直接传入原生物理 ID。

无独立 CLI；数量、座位顺序、牌种范围或物理计数不一致时报告错误而不是修补牌数。

## 测试与状态

对应 [test_rule_features.py](../../../tests/kar/test_rule_features.py)、[test_golden.py](../../../tests/kar/test_golden.py)和特征模式相关模型测试。本轮未执行任何测试；规则字段不提供策略安全证书。
