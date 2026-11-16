<!-- kar-module: riichienv.kar.support -->
<!-- api: EventSupport, support_from_train_games, CandidateValue, candidate_value -->
# support — 独立 train 支持与保守候选价值

源码：[support.py](../../../src/riichienv/kar/support.py) · [总索引](../README.md)

## 接口与数据流

`EventSupport` 保存实际训练 manifest 身份、21 类独立 train game 数及来源；支持 `to_record/from_record` 和绑定校验。`support_from_train_games(train_manifest, observed_events, source_reference=...)` 由明确 train 游戏与观察事件构建证据。

`candidate_value(action_id, response_logits, rank_logits, ...)` 在公开 allowed mask、T_E/T_R 和支持 manifest 下计算 `CandidateValue`，返回响应概率、各分支效用、名义/保守价值及 unsupported mass。

## 冻结公式与边界

响应 softmax **先 mask**，再计算未启用事件概率质量 rho_U。名次效用为 `(1.5, .5, -.5, -1.5)`；未支持事件在保守价值中使用 -1.5，不是零分。21 个 rank 分布要与事件维度对应，不能拿真实响应标签挑选线上候选分支。

至少 500 个独立 train 完整游戏才启用对应分支；重复决策行不增加数量，499 train 加 1 validation 不能凑 500。计数、NLL、Brier 或 rho_U 都不是 delta_E / epsilon_m 的统一误差上界。

## 集成与失败处理

上游为真实训练选择及冻结标定，下游 [runtime](../runtime/README.md)对全部 S 候选评分。非法候选、错误尺寸、空 mask、非有限 logits 或不同 train binding 应拒绝，不修补统计或把缺失填零。

无独立 CLI；只消费数组/元数据，不创建网络或读他家暗手。生产统计也由 `learning.calibration_job.dataset_event_support` 从只读 SQLite 计算。

## 测试与状态

对应 [core/test_artifact_contracts.py](../../../tests/kar/core/test_artifact_contracts.py)、[core/test_runtime.py](../../../tests/kar/core/test_runtime.py)、[core/test_probability_policy.py](../../../tests/kar/core/test_probability_policy.py)。本轮未执行支持统计或候选评分。
