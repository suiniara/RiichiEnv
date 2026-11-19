<!-- kar-module: riichienv.kar.selection_quality -->
<!-- api: TauSelectionPolicy, cal_fit_scenario_manifest, paired_tau_evidence, validate_finite_tau_selection -->
# selection_quality — 有限 tau 的验证收益准入

源码：[selection_quality.py](../../../src/riichienv/kar/selection_quality.py) · [总索引](../README.md)

## 职责与接口

`TauSelectionPolicy` 固定最少独立 scenario game clusters、最低效用增益、置信度、描述性 bootstrap 次数/seed 和登记来源。默认独立数量下限为 2、最低增益为 0、置信度 .95、2000 次重采样；独立数量下限只是额外条件，2 个正样本不能通过默认置信下界。

`paired_tau_evidence(outcomes, grid, policy)` 在相同完整 scenario game ID 上比较每个有限 tau 与 infinity，保存描述性 bootstrap 区间和独立计算的收益置信下界。`cal_fit_scenario_manifest(...)` 绑定全部 cal_fit ID、数量、来源 manifest 和摘要。`validate_finite_tau_selection(selection, expected_game_ids=...)` 要求调用方从实际标定或 M1 报告提供完整清单，验证全部网格证据，并从保存的均值、数量和候选数重新计算准入下界。

## 选择边界

每场平均名次效用属于 [-1.5, 1.5]，候选减 infinity 的配对差属于 [-3, 3]。对预先固定的 n 个独立场景和 K 个有限候选，使用单侧 Hoeffding 界并对全部 K 个候选作 union bound：

`lower = max(-3, mean_gain - sqrt(18 * log(K / (1 - confidence)) / n))`

只有该下界严格大于预登记最低增益且数量达到登记下限的有限项才合格。bootstrap 区间只报告，不决定准入；两个同为 +1 的配对差即使 bootstrap 下端为 +1，默认 Hoeffding 下界仍为负，保留 infinity。只有 infinity 的网格不计算有限候选下界，也不会除以 0。证据不足时 [calibration.choose_validation_tau](../calibration/README.md) 返回 infinity，不自动追加对局或提高资源预算。

选择产物为 `frozen_validation_tau_grid_v4`，政策为 `preregistered_tau_selection_v2`，配对准入证据为 `paired_rank_hoeffding_union_v1`。旧版本不能改标签复用。生产路径要求完整预登记 cal_fit 清单和全部网格项，不能事后选择有利子集、删候选或达到显著后提前停止；不读 cal_check/test。不同候选之间可以相关，但不同场景块必须独立，且样本数、候选和策略须在结果出现前固定。

## 调用关系与限制

由 [learning/validation_policy](../learning/validation_policy/README.md)预登记和执行，有限发布与 M2 扩规模前置证据再验证。本层不运行模型、不调整网络、不查询资源，只有元数据和统计计算，无独立 CLI。

“reproducible_gain” 表示通过上述限定假设下的同时均值收益下界。它不是逐状态误差证书、确定性收益保证或独立 M3 的替代；`uniform_error_certificate` 保持 false。

## 测试与状态

对应 [production/test_selection_and_roles.py](../../../tests/kar/production/test_selection_and_roles.py)、[production/test_stage_prerequisites.py](../../../tests/kar/production/test_stage_prerequisites.py)和 [production/test_review_policy_identity.py](../../../tests/kar/production/test_review_policy_identity.py)。静态回归覆盖两个正样本的零收益反例、充分固定样本、网格多重比较、infinity 回退，以及子集、旧版本和重签篡改证据的拒绝；这些合成结果不代表实际模型棋力。
