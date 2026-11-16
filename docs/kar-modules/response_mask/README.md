<!-- kar-module: riichienv.kar.response_mask -->
<!-- api: public_rule_response_mask, validate_response_target -->
# response_mask — 公开规则允许的 21 类响应

源码：[response_mask.py](../../../src/riichienv/kar/response_mask.py) · [总索引](../README.md)

## 接口与职责

`public_rule_response_mask(snapshot, candidate_discard, ruleset=MJSOUL_RULESET)` 返回长度 21 的布尔 tuple。1 表示尚未被公开事实证明不可能，不表示对手确实能做该动作。`validate_response_target(allowed, target)` 检查有效监督没有被排除。

输入只有当前公开快照、完整候选舍牌和冻结规则；没有真实响应标签、未来事件、他家暗手或私有合法动作参数。

## 硬规则与顺序

mask 可利用吃的相对位置、公开永久振听、活牌山/终局条件、四风/四立直/四杠前提及三家和规则等。不能因为“对方大概没有对子”而排除 PON，也不能因模型预测概率小而 mask。

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

对应 [test_observation_response.py](../../../tests/kar/test_observation_response.py)、[test_golden.py](../../../tests/kar/test_golden.py)、[m1/test_losses.py](../../../tests/kar/m1/test_losses.py)。本轮未执行规则判断或测试；mask 是公开不可能性筛查，不是对隐藏手牌的推断器。
