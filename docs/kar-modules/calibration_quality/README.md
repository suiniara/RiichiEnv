<!-- kar-module: riichienv.kar.calibration_quality -->
<!-- api: CalibrationQualityPolicy, evaluate_quality -->
# calibration_quality — 独立标定质量判定

源码：[calibration_quality.py](../../../src/riichienv/kar/calibration_quality.py) · [总索引](../README.md)

## 职责与接口

`CalibrationQualityPolicy` 固定三个 head 的 NLL/Brier/ECE 上限、最少有效观察数、最少多舍牌门控观察数、最低保留率和登记来源。`to_record/from_record` 记录阈值与自身摘要；阈值映射被冻结，不能读取 cal_check 后继续修改。

`evaluate_quality(metrics, policy)` 返回 `passed`、`failed` 或 `unavailable`、原因、cal_check 分区及 `uniform_error_certificate=false`。没有 policy 不默认通过。

## 必需字段

- `max_nll / max_brier / max_ece` 都明确覆盖 action/response/rank。
- `minimum_head_observations`、`minimum_gate_observations` 为正整数。
- `minimum_gate_retention` 在 0..1，来源引用非空。
- 非有限/负值、Brier/ECE 非法上限、未知派生字段或摘要变化拒绝。

指标非空不等于质量通过；样本不足和缺指标是 unavailable，超登记限值是 failed。通过只是工程准入，不声称每个响应类或每个状态已经校准。

## 集成与使用

在 `calibration_job_plan` 之前由操作者预登记真实 criteria；[calibration](../calibration/README.md)保存 policy/report，[runtime](../runtime/README.md)和冻结发布消费它。失败时有限重排回到同一 checkpoint 的先验，不扩大模型掩盖问题。

纯元数据/数组接口，无模型、优化器或独立 CLI。本文不生成一套“保证通过”的默认阈值。

## 测试与状态

对应 [core/test_review_runtime.py](../../../tests/kar/core/test_review_runtime.py)、[production/test_review_policy_identity.py](../../../tests/kar/production/test_review_policy_identity.py)。本轮未执行质量判定或真实标定。
