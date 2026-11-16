<!-- kar-module: riichienv.kar.policy -->
<!-- api: support_discards, tau_minimum, fixed_tau_grid, RestrictedUpdate, restricted_kl, BoundEvidence, one_sided_bounds, PolicyOptions, PolicyResult, sample_complete -->
# policy — 质量保持 KL、证据界与采样

源码：[policy.py](../../../src/riichienv/kar/policy.py) · [总索引](../README.md)

## 主要接口

- `support_discards(candidates, log_prior, log_threshold)`：只保留完整舍牌 ID，按权威 log 阈值门控。
- `restricted_kl(prior, actions, values, tau, ...)`：返回 `RestrictedUpdate`。
- `tau_minimum(Rmax)` / `fixed_tau_grid(...)`：相对放大约束和含 infinity 的预登记网格。
- `BoundEvidence` / `one_sided_bounds(...)`：消费当前模型/决策/候选覆盖的有效界，计算 E1/E2。
- `PolicyOptions`、`PolicyResult`：模式、温度、证据状态与最终分布。
- `sample_complete(probabilities, candidates, rng, diagnostic_argmax=False)`：在完整合法分布采样。

## 核心数学合同

令 M 为原先验在 S 的质量，q0 为其条件分布，qtau = softmax(log q0 + Q/tau)。只把 S 内改为 `M * qtau`，S 外逐动作概率和 S 总质量保持不变。tau=infinity 是精确先验；有限 tau 可要求 `tau >= 3 / ln(Rmax)`。

E1 的 unsupported 惩罚按 q0 加权，不能换成 qtau；E2 使用概率移动方向的单侧代价。未知 b/delta_pi 不可填零。误差界必须说明来源、覆盖和置信/确定性条件，模块不估计统一界。

## 使用与回退

通常由 `RuntimePolicy` 统一协调，不建议绕过其标定/支持/质量检查直接部署低层结果。默认 `PolicyOptions()` 为 prior-only。`certified + diagnostic_argmax` 拒绝：众数动作不是同一随机策略，不能继承证书。

候选重复、非法分布、非正有限 tau、放大上限冲突或证据范围不符时错误/回退，不静默剪裁成另一策略。

## 测试与状态

对应 [core/test_probability_policy.py](../../../tests/kar/core/test_probability_policy.py)、[core/test_runtime.py](../../../tests/kar/core/test_runtime.py)、[core/test_review_runtime.py](../../../tests/kar/core/test_review_runtime.py)。本轮未执行数学回归；经验收益不等于数学安全保证。
