<!-- kar-module: riichienv.kar.selection_quality -->
<!-- api: TauSelectionPolicy, paired_tau_evidence, validate_finite_tau_selection -->
# selection_quality — 有限 tau 的验证收益准入

源码：[selection_quality.py](../../../src/riichienv/kar/selection_quality.py) · [总索引](../README.md)

## 职责与接口

`TauSelectionPolicy` 固定最少独立 scenario game clusters、最低效用增益、bootstrap 次数/seed/置信度和登记来源。缺省保守配置至少 2 个 clusters、最低增益 0、名义置信度 .95、2000 次重采样。

`paired_tau_evidence(outcomes, grid, policy)` 在相同 scenario game ID 上比较每个有限 tau 与 infinity，按整场 cluster 计算配对效用差区间。`validate_finite_tau_selection(selection)` 验证生产选择的版本/摘要、独立数量、选中项、均值差和登记统计配方。

## 选择边界

只有区间下端大于预登记最低增益的有限项才合格。只有一个游戏、重复决策再多、均值略高但区间跨零都不能放行；[calibration.choose_validation_tau](../calibration/README.md)保留 infinity。没有合格有限项不是失败地“必须选一个”。

`frozen_validation_tau_grid_v3` 产物保存 policy 和配对证据。校验器拒绝 diagnostic fixture 冒充生产，或在取得结果后改变 confidence/seed/次数。固定小网格只使用 cal_fit，不读 cal_check/test。

## 调用关系与限制

由 [learning/validation_policy](../learning/validation_policy/README.md)预登记和执行，有限发布与 M2 扩规模前置证据再验证。本层不运行模型、不调整网络、不查询资源，只有元数据和统计计算，无独立 CLI。

“reproducible_gain” 是该预登记经验工程门槛的结果，不是逐状态误差证书、确定性收益保证或独立 M3 的替代。

## 测试与状态

对应 [production/test_selection_and_roles.py](../../../tests/kar/production/test_selection_and_roles.py)、[production/test_stage_prerequisites.py](../../../tests/kar/production/test_stage_prerequisites.py)。本轮未执行区间计算、收益筛选或测试。
