<!-- kar-module: riichienv.kar.learning.validation_policy -->
<!-- api: validation_tau_plan, select_validation_tau -->
# learning.validation_policy — cal_fit 对局选择 tau

源码：[validation_policy.py](../../../../src/riichienv/kar/learning/validation_policy.py) · [learning 索引](../README.md)

## 接口与登记

`validation_tau_plan(subject, opponents, seed, grid=(infinity,1,.5,.25), selection_policy=None)` 固定实际校准策略身份、三个对手、cal_fit scenario IDs、网格、后端和收益判据。

`select_validation_tau(...)` 验证独立许可、plan/预算和数据角色后，对每个预登记 scenario/grid 项执行完整半庄。subject 必须有通过独立质量检查的标定与 train 支持，不从 cal_check/test 补场景。

含有限 tau 的网格还要求至少一个响应分支同时满足 train 支持和冻结 cal_check 分支质量。总体校准通过而有效价值分支为空时，规划立即拒绝；执行入口重建同一计划，因此不会创建输出目录、费用账本或启动对局。此时保留已校准先验，不通过放宽质量门槛继续调参。显式只含 infinity 的网格仍可用于预登记的先验基线评估，其选择结果为 infinity，不声明有限策略收益。

## 配对含义

scenario 使用 cal_fit game ID 稳定派生模拟墙 seed，**不是重放同一源游戏的暗手或轨迹**。网格项共享同一初始墙/座位和登记对手流，主体 RNG 独立；不同策略的后续轨迹可以分叉。

infinity 为同 checkpoint 的校准先验，有限项为 empirical_only 的受限 KL。完整候选评分/公开 mask/支持门控与线上运行相同。

## 收益 gate 与产物

[selection_quality](../../selection_quality/README.md)政策在对局前入 plan hash。均值较高但缺独立 clusters 或配对区间下端不足，保留 infinity，不强迫发布有限策略。

新目录写 `validation_matches.jsonl`、费用账本、`tau-selection.json`，记录 frozen plan、完整选择、RNG/seed 与不使用 test/cal_check 的声明。异常结算最后实际计算段，不将失败免费丢弃。

## 使用与状态

经 `jobs_cli` 的 `select-tau` 进入；实际执行会加载/运行已准备的冻结模型和原生对局，不更新网络但不是静态操作。对应 [production/test_selection_and_roles.py](../../../../tests/kar/production/test_selection_and_roles.py)、[test_review_job_guards.py](../../../../tests/kar/production/test_review_job_guards.py)、[test_review_policy_identity.py](../../../../tests/kar/production/test_review_policy_identity.py)及[test_tau_selection_support.py](../../../../tests/kar/production/test_tau_selection_support.py)。有效分支门禁回归使用合成静态 logits 和实际冻结发布的保存/加载；拒绝路径禁止账本和对局调用，不更新参数。
