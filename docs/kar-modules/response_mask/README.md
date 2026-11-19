<!-- kar-module: riichienv.kar.response_mask -->
<!-- api: public_rule_response_mask, validate_response_target -->
# response_mask — 公开规则允许的 21 类响应

源码：[response_mask.py](../../../src/riichienv/kar/response_mask.py)、[Rust public_rules](../../../riichienv-core/src/public_rules.rs)、[绑定](../../../riichienv-python/src/public_rules.rs) · [总索引](../README.md)

## 接口与职责

`public_rule_response_mask(snapshot, candidate_discard, ruleset=MJSOUL_RULESET)` 返回长度 21 的布尔 tuple。1 表示尚未被公开事实证明不可能，不表示对手确实能做该动作。`validate_response_target(allowed, target)` 检查有效监督没有被排除。

Python 先验证完整候选为 `0..147` 的整数且不是 bool，并拒绝 pending-kan 快照，再将公开字段白名单传给 `kar_public_response_mask`。麻将谓词由 Rust 的纯 `PublicRuleContext` 计算，绑定不接完整 snapshot 或 `GameState`。实际参数为候选牌种/立直位、actor、公开舍牌历史和当前河、副露种类、立直接受位、剩余摸牌数、历史完整位和规则开关；没有真实响应标签、未来事件、他家暗手或私有合法动作参数。

## 硬规则与顺序

mask v3 区分公开证明的途中流局与仅可能的终止：四风、多玩家四杠、候选四立直按原生优先级；确定终止排除 NEXT_DRAW 与所有吃碰杠，但不压掉公开仍可能的荣和。一人四杠不误判为多人四杠。remaining_draws=0 禁止吃、碰、大明杠；历史不完整不因未知强制终止，已知正/否定证据仍保留。雀魂流局满贯修正不回退。不能因为“对方大概没有对子”或模型概率低而 mask。迁入 Rust 保持上述语义及 21 类顺序，因此 `RESPONSE_MASK_VERSION` 仍为 `public_rule_response_mask_v3_abort_last_draw`。

```text
公开快照 + 候选 → allowed
response logits / T_E → mask 为 -inf → Softmax
                                       ↓
                               之后才计算 rho_U
```

被排除类概率严格为 0；允许集合和为 1。全 false 表示快照/生成器错误，不能回退为未掩码 softmax。训练 T=1，推理使用标定 T_E，调用相同 mask 逻辑。

## 错误与集成

有效 target 被 mask 排除应阻断 M0/导出或训练，不允许“按真值恢复该位”。无效训练行使用受 loss 有效位排除的 dummy mask，不能让全 -inf 产生 NaN。上游 [observation](../observation/README.md)，消费者为 `learning/export`、`losses` 与 `runtime/support`。

## 测试与状态

对应 [test_native_public_rules.py](../../../tests/kar/test_native_public_rules.py)、[test_response_mask_boundaries.py](../../../tests/kar/test_response_mask_boundaries.py)、[test_observation_response.py](../../../tests/kar/test_observation_response.py)、[m1/test_losses.py](../../../tests/kar/m1/test_losses.py)。新增合同以实际隐藏 Ron 能力不同、公开状态相同的环境检查 mask 不变，并检查隐藏/未来字段无法进入原生绑定。实际验证状态见[边界迁移记录](../../kar-native-rule-boundary.md#7-验证位置与当前证据)。
