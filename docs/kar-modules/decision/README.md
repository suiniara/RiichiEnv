<!-- kar-module: riichienv.kar.decision -->
<!-- api: ReplayDecision -->
# decision — 决策输入与监督标签的交接对象

源码：[decision.py](../../../src/riichienv/kar/decision.py) · [总索引](../README.md)

## 职责

`ReplayDecision` 是审计回调与导出之间的冻结载体：把 cutoff 的公开快照、已编码前缀、真实动作条件后缀和来源信息放在明确边界上，而不是让模型直接读取整场全知事件。

## 接口与数据流

数据类的六个字段为 `snapshot`、`prefix`、`conditional`、`action_target_global`、`trigger_event_id`、`action_provenance`。完整候选保留在编码前缀内。**该对象不包含 response/rank 目标；它们由导出器结合单独的实际窗口/终局事实生成。**

```text
原始事件 → audit/InputAudit → ReplayDecision 回调
                            ├─ snapshot/prefix/候选 → 模型输入
                            └─ conditional/真实动作/trigger/provenance → 条件输入及动作监督
导出器另外解析实际响应与完整终局 → 独立 E/R 监督列
```

它不提供独立 CLI，也不创建网络、优化器、数据目录或执行动作。通常由 `audit.replay_game(..., on_decision=...)` 生产，由 `learning.export.GameWriter` 消费，不建议手工构造生产决策。

## 关键合同

- 真实 ACTION 可以作为条件头后缀，但不允许污染 DECIDE 的因果前缀。
- response/value 有效位独立保存；未知响应不能伪装为 NEXT_DRAW，非舍牌不生成伪响应。
- 最终名次可用作标签，不能写回当前公开 rank/gap 特征。
- 所有候选保持全局身份，四个观察者仍属于同一个 game split。

## 测试与状态

由 [test_audit.py](../../../tests/kar/test_audit.py)、[test_tokenize.py](../../../tests/kar/test_tokenize.py)、[m1/test_dataset.py](../../../tests/kar/m1/test_dataset.py)覆盖相关交接合同。关联模块：[input_audit](../input_audit/README.md)、[export](../learning/export/README.md)。本轮未运行回放或测试。
