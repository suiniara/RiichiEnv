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
- 只有每个合法两面搭子中至少一种牌四张可见，才排除全部两面形状。目标牌本身四张可见不能单独排除：计数包含自己的暗手，从中舍出一张仍可能补全对手的两面搭子。例如本家持四张 `1m`，对手仍可持 `23m` 等待 `1m/4m`。
- 玩家暗杠/加杠含某牌不代表该玩家打过它，不形成 Genbutsu。
- 被叫走的舍牌保留在历史判断中，但不再留在当前河的物理区重复计数。
- 一位没有更高相邻名次，四位没有更低相邻名次；相邻同分的差是 present 的 0，而不是 absent。
- 证据未知可返回 `None`，不能把未知强改为 false/true。排除两面形状不等于排除所有和牌形状，更不是“绝对安全”。

## 使用与调用关系

这是纯规则函数层，不查询他家私有合法动作，不使用终局名次标签。推荐由 `observation.extract_rule_features` 在白名单状态上调用，由 tokenizer 写入适用的 HAND_COUNT/ROUND_STATE 字段。牌编码应先规范化到本接口约定的 base34，不能直接传入原生物理 ID。

无独立 CLI；数量、座位顺序、牌种范围或物理计数不一致时报告错误而不是修补牌数。

## 版本与已有产物

本次修正使用 `feature_contract_version=visible_rule_features_v2_ryanmen_pairs` 与 `input_pipeline_version=semantic_public_clock_frozen_decisions_v6_ryanmen_pairs`。前者区分修正前后的规则特征，后者使模型前缀缓存和训练配方也绑定新输入语义。Token 布局、Schema、动作表、公开响应硬掩码和模型参数形状不变。

旧版派生数据、检查点与发布产物不能仅修改版本字段后复用；应从可信原始记录重新审计并导出。已有加载入口按完整合同拒绝旧产物，历史 Token 缓存遇到输入版本变化会重建，模型前缀缓存会拒绝旧输入版本。

## 测试与状态

对应 [test_rule_features.py](../../../tests/kar/test_rule_features.py)、[test_golden.py](../../../tests/kar/test_golden.py)和特征模式相关模型测试。物理牌山回归覆盖本家持四张目标牌、舍出其中一张、对手实际两面荣和，检查修正后的 HAND_COUNT 字段、缓存前缀与原生和牌结果；版本测试覆盖旧导出与旧模型前缀缓存的拒绝。规则字段不提供策略安全证书。
