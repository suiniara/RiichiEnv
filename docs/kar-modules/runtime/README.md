<!-- kar-module: riichienv.kar.runtime -->
<!-- api: LogitPredictor, RuntimePolicy -->
# runtime — 原始 logits 到最终策略

源码：[runtime.py](../../../src/riichienv/kar/runtime.py) · [总索引](../README.md)

## 职责与接口

`LogitPredictor` 协议要求模型/数据 `binding`、前缀 policy logits 和全部 S 候选的 response/rank logits。`RuntimePolicy(predictor, options=..., support=..., calibration=..., seed=...)` 统一消费这些原始输出。

`decide(snapshot, candidates, context, bounds=None)` 返回 `PolicyResult`：最终完整概率、采样 ID、evidence status、fallback reason 和逐决策 telemetry。

## 单次流程

1. 校验完整合法集合与模型绑定，计算 T_A 先验。
2. 由 log p_min 得到完整舍牌 S；无效门控、质量失败、S 太小等使用同一先验。
3. infinity 路径不调用候选预测器；有限路径对全部 S 评分，不 Top-K。
4. 公开 mask → T_E/T_R → unsupported -1.5 → 保守 Q → 质量保持 KL。
5. empirical_only 明确不附数学证书；certified 要求有效当前界且 E2>0，否则回退。
6. 使用独立 `random.Random` 采样完整分布。

## 可观测输出

记录温度、支持集、候选价值、rho_U、KL、q0/qtau、界、RNG 前后摘要、最终概率和回退原因。`distribution_changed` 比较实际浮点概率变化，不与“执行过候选前向”混淆；二者都不是收益证书。

缺标定/支持/独立质量证据不会自动启用价值。`allow_diagnostic_calibration` 只允许明确测试 fixture，不能作为生产发布资格。公开快照缺历史/可见性时应拒绝而非从未来补齐。

## 调用关系与状态

预测适配见 [learning/inference](../learning/inference/README.md)，消费见 [agent](../agent/README.md)。runtime 不创建优化器、拟合温度或启动对局；无独立 CLI。

对应 [core/test_runtime.py](../../../tests/kar/core/test_runtime.py)、[core/test_review_runtime.py](../../../tests/kar/core/test_review_runtime.py)。本轮未执行策略或模型；默认保守先验不等于相对真实行为策略的无损保证。
