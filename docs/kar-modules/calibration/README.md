<!-- kar-module: riichienv.kar.calibration -->
<!-- api: CalibrationSplit, CalibrationRow, CalibrationArtifact, calibration_split, calibration_split_from_manifest, temperature_nll, select_static_temperature, gate_log_threshold, calibration_metrics, calibrate_static_rows, choose_validation_tau, calibrated_policy_identity, validate_policy_identity, calibration_rows_sha256 -->
# calibration — 温度、log 门控与冻结产物

源码：[calibration.py](../../../src/riichienv/kar/calibration.py) · [总索引](../README.md)

## 主要接口

| 接口 | 作用 |
|---|---|
| `CalibrationSplit` / `calibration_split_from_manifest` | 从实际 validation 清单派生固定 cal_fit/cal_check |
| `CalibrationRow` | 静态 logits、完整候选、有效目标、公开 mask 和分组信息 |
| `temperature_nll` / `select_static_temperature` | 在明确温度网格上评分三个 head |
| `gate_log_threshold` | 只在多完整舍牌子集标定权威 log p_min |
| `calibration_metrics` | 完整自然频率、分支、可靠性、CI 和门控诊断 |
| `calibrate_static_rows` / `CalibrationArtifact` | 拟合 fit、只检查 check，冻结来源和质量判定 |
| `choose_validation_tau` | 消费预登记验证对局结果，不自行执行对局 |
| `calibrated_policy_identity` / `validate_policy_identity` | 绑定校准、支持、后端和随机策略身份 |

## 关键合同

生产 split 校验完整 manifest hash、主 validation、排除 Golden 及固定角色 salt。三个温度和 p_min 只在 cal_fit 选择。门控概率分母是全部合法 C，而非仅舍牌；真实动作在门控前 C_D、至少两个完整舍牌且 action_valid 才参与。n 个 log 概率排序后取 0-based `n//100`，`>=` 保留 ties，无插值。

空标定子集使 gate 无效，不能伪造阈值。独立总体与逐响应分支质量由 [calibration_quality](../calibration_quality/README.md) 判定。`calibration_metrics` 以包含正负例、公开允许的行计算逐事件 response 概率指标；按真实 E 分组的分类诊断单列，不作准入。Rank 仍在真实 E 分支上评估。artifact 绑定 branch policy/report 及 21 个 value_branch_statuses，缺标准或正例独立 game 证据不得放行。

## 身份、使用与产物

`CalibrationArtifact` 保存绑定、support hash、split、温度、log 阈值、行摘要、深冻结诊断、质量政策和原始后端。版本为 `heldout_temperatures_log_gate_v6_event_ovr`。先计算基础指标、按 cal_check 预登记标准确定分支质量，再用同一冻结有效支持集对 fit/check 的 response_valid 行统计有效未支持质量；只读 logit archive 逐行再读，不在内存存整批概率。两份 train-only 分布各自独立于最终价值分布。加载校验统计版本、桶、有效观测数和实际支持位，不对旧报告补标签。diagnostic fixture 不可冒充生产。

这里的温度选择对调用方已经提供的静态数组工作，不创建网络或优化器；真实 logits 收集由 [learning/calibration_job](../learning/calibration_job/README.md)执行。生产有限 tau 还需 [selection_quality](../selection_quality/README.md)的登记收益政策；实际对局由 `validation_policy` 运行。

## 测试与状态

对应 [core/test_calibration.py](../../../tests/kar/core/test_calibration.py)、[core/test_artifact_contracts.py](../../../tests/kar/core/test_artifact_contracts.py)、[production/test_selection_and_roles.py](../../../tests/kar/production/test_selection_and_roles.py)。本轮未进行真实模型校准或阶段评测；合成静态数组、质量三态与版本拒绝单元的实际结果见[修复报告](../../kar-semantic-branch-repair.md)。
