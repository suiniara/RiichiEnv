<!-- kar-module: riichienv.kar.learning.deployment -->
<!-- api: frozen_export_plan, validate_frozen_policy, save_frozen_release, load_training_model, FrozenRelease -->
# learning.deployment — 冻结发布与受限加载

源码：[deployment.py](../../../../src/riichienv/kar/learning/deployment.py) · [learning 索引](../README.md)

## 主要接口

| 接口 | 作用 |
|---|---|
| `frozen_export_plan(...)` | 绑定检查点、设备、policy 和校准/tau job 元数据 |
| `load_training_model(checkpoint, device="cpu")` | 从核验后的生产检查点恢复模型供冻结使用 |
| `validate_frozen_policy(...)` | 导出与加载共用的完整有限策略检查 |
| `save_frozen_release(directory, model, ...)` | 在新目录写 weights.pt 与 release.json，不带优化器 |
| `FrozenRelease(directory, device="cpu")` | 核对源码/native/文件/配置/绑定并受限加载 |
| `FrozenRelease.policy(seed, baseline=False)` | 创建独立策略；baseline 强制同标定先验与 infinity |

## 有限策略必须绑定

温度、门控、支持、原始后端、采样模式、Rmax和验证选择都属于实际策略身份。有限部署要求总体质量passed、门控有效，至少一个train支持且response/rank分支质量passed的响应类别，并通过生产tau的cal_fit配对收益gate。分支政策摘要与状态进入完整策略身份；缺新字段的旧校准/支持拒绝。权重相同不意味着另一份标定可以复用旧tau。

正式有限发布拒绝 diagnostic argmax，diagnostic 校准 fixture 不能冒充生产。没有有效逐状态上界不声称 certified；标准冻结 M3 使用有限随机经验策略。

## 文件与迁移

release 保存模型配置、真实 weights 文件 hash、RuntimeBinding、训练/构建身份、calibration/support/selection、policy 与明确证据状态。加载使用受限 weights_only；只能加载可信来源。目录已存在、版本不符、权重/元数据被改或把 CUDA 标定改标 CPU 都会拒绝。

跨源码、native、设备或协议不支持静默“精确复用”，应显式重新导出/校准或迁移。metadata plan 生成不等于执行授权。

## 测试与状态

对应 [core/test_frozen_inference.py](../../../../tests/kar/core/test_frozen_inference.py)、[production/test_review_policy_identity.py](../../../../tests/kar/production/test_review_policy_identity.py)。实际加载/保存会创建网络与文件；本轮未迁移真实旧发布；新临时合成产物的保存/加载及拒绝回归范围见[修复报告](../../../kar-semantic-branch-repair.md)，不代表真实模型重建或棋力验收。
