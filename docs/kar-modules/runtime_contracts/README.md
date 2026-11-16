<!-- kar-module: riichienv.kar.runtime_contracts -->
<!-- api: RuntimeBinding, runtime_contract, sha256_value -->
# runtime_contracts — 推理身份和冻结版本

源码：[runtime_contracts.py](../../../src/riichienv/kar/runtime_contracts.py) · [总索引](../README.md)

## 职责与接口

`runtime_contract()` 汇总活动 Schema、规则、动作、架构、效用、策略、支持、校准、杠事务和校准角色版本。`sha256_value(value, name)` 验证明确的 64 位小写十六进制摘要。

`RuntimeBinding` 绑定模型权重、实际训练 manifest、schema/rules/action 与推理 recipe。`RuntimeBinding.current(...)` 用当前冻结公共合同构造；`to_record()` 和 `digest` 用于标定、支持、证据与运行结果的一致性比较。

## 重要区别

- 权重相同但模型配置/特征模式改变，也可能是不同推理策略；调用方应使用真实 inference recipe，而非随意标签。
- `RuntimeBinding` 不是整个策略身份。实际校准、支持、采样模式、后端和放大约束进一步由 `calibrated_policy_identity` 绑定。
- `UTILITIES=(1.5,.5,-.5,-1.5)`，温度与效用同单位；不能套用原始点数尺度。
- Schema 必须完整可用；未知/旧合同不能静默升级。SHA 是一致性工具，不把来源不明文件变可信。

## 使用与调用关系

由冻结 predictor 构造真实绑定后，向 [calibration](../calibration/README.md)、[support](../support/README.md)、[policy](../policy/README.md)和 [release](../release/README.md)传播。不要为了让外来 artifact 通过而改写它的绑定。

无独立 CLI、模型计算或文件写入；错误 hash 和公共合同不一致直接拒绝。

## 测试与状态

对应 [core/test_artifact_contracts.py](../../../tests/kar/core/test_artifact_contracts.py)、[core/test_frozen_inference.py](../../../tests/kar/core/test_frozen_inference.py)。本轮未运行任何绑定/模型验收；字段版本以源码常量为准。
