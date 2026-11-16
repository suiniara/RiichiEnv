<!-- kar-module: riichienv.kar.input_audit -->
<!-- api: InputAudit -->
# input_audit — 决策 cutoff 编码检查

源码：[input_audit.py](../../../src/riichienv/kar/input_audit.py) · [总索引](../README.md)

## 职责与接口

`InputAudit(on_decision=None)` 为四个观察者分别维护 `TokenHistoryCache`。`inspect(snapshot, candidates, target, provenance="explicit")` 构造决策前 prefix 和真实动作条件后缀，累计字节/hash/阶段数；`report()` 汇总状态、gaps 与输入摘要。

这是审计组件，不执行网络 forward、优化器或环境动作，也没有独立 CLI。

## 回调合同

若配置 `on_decision`，审计协调器必须先提供明确的 `trigger_event_id`。成功检查后回调得到 `ReplayDecision(snapshot, prefix, conditional, target, trigger_event_id, provenance)`。类本身不凭缺失数据猜 action target，也不计算未来 E/R 标签。

因必要可见历史缺失而发生的 `TokenizationUnavailableError` 进入 gaps；不是静默补造输入。其他非法字段/候选等合同错误仍向上传播。

## 输出含义

`report()` 返回 Schema 身份、`decisions_checked`、`bytes_checked`、prefix 与 conditional 摘要、分阶段数和 gaps。Schema 不可用为 FAIL，有 gap 或零决策为 BLOCKED，其余才是本组件 PASS。`labels_in_prefix=false` 是编码合同描述，不代表整场数据/完整产品已经验收通过。

调用链：[audit](../audit/README.md) → `InputAudit` → [tokenize](../tokenize/README.md) → 可选 [export](../learning/export/README.md)。模型输入和独立监督列必须继续分离。

## 测试与状态

对应 [test_audit.py](../../../tests/kar/test_audit.py)、[test_tokenize.py](../../../tests/kar/test_tokenize.py)、[m1/test_dataset.py](../../../tests/kar/m1/test_dataset.py)。本轮没有运行 inspect、导出或测试。
