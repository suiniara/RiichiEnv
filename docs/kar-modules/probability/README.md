<!-- kar-module: riichienv.kar.probability -->
<!-- api: finite_vector, complete_candidates, masked_log_softmax, masked_softmax, legal_prior, probability_vector -->
# probability — 稳定合法概率原语

源码：[probability.py](../../../src/riichienv/kar/probability.py) · [总索引](../README.md)

## 职责与接口

纯 Python 数值层，不依赖 Torch，不做动作选择或校准拟合。

| 接口 | 合同 |
|---|---|
| `finite_vector(values, size, name)` | 固定尺寸、非 bool 的有限原始数值 |
| `complete_candidates(values)` | 非空、不同、合法范围内的完整全局 ID，且不超过容量 |
| `masked_log_softmax(logits, allowed, temperature=1)` | 在 mask 前缩放温度，稳定归一化；被排除项为 -inf |
| `masked_softmax(...)` | 对应概率，被排除项严格为 0 |
| `legal_prior(logits, candidates, temperature=1)` | 546 全局空间中的先验与 log 先验 |
| `probability_vector(values, size, name)` | 验证范围与概率和，不偷偷归一化坏输入 |

## 数值边界

必须区分有限原始 logits 与含非法项 -inf 的 log 概率。允许集合为空、温度非法、尺寸错误、NaN/Inf、重复候选或坏概率和均拒绝。极小真值概率用稳定 log 值参与 NLL/门控，不以 `exp` 下溢后的零值替代。

```python
from riichienv.kar.probability import masked_softmax

p = masked_softmax([0.0, 0.0, 0.0], [True, False, True])
assert p == (0.5, 0.0, 0.5)
```

## 调用关系与状态

由 `calibration`、`support`、`policy`、`runtime` 共享，避免训练/标定/推理各自实现不同 mask 顺序。合法先验分母包含完整合法集合，不条件化成“仅舍牌”。无独立 CLI。

对应 [core/test_probability_policy.py](../../../tests/kar/core/test_probability_policy.py)、[core/test_calibration.py](../../../tests/kar/core/test_calibration.py)。本轮未执行示例或测试；概率归一化正确不等于模型校准或策略收益成立。
