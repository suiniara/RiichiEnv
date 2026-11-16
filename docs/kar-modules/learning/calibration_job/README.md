<!-- kar-module: riichienv.kar.learning.calibration_job -->
<!-- api: CalibrationRows, dataset_event_support, calibration_job_plan, collect_and_calibrate -->
# learning.calibration_job — 冻结 logits 收集与正式标定

源码：[calibration_job.py](../../../../src/riichienv/kar/learning/calibration_job.py) · [learning 索引](../README.md)

## 主要接口

`calibration_job_plan(predictor, dataset_manifest, validation_manifest, quality_policy=..., temperatures=...)` 绑定已加载模型、实际数据、原始后端、温度网格和预登记质量政策。它不自行加载网络，但需要调用方已准备的冻结 predictor。

`collect_and_calibrate(...)` 核验独立评测许可和完整 validation 范围后，实际运行模型并收集 logits。`CalibrationRows(path, partition)` 为 SQLite fit/check 的只读惰性 Sequence；使用后关闭连接。`dataset_event_support(dataset)` 只从数据库 train 部分按 distinct game_id 统计21类。

## 角色与因果合同

真实动作只用于条件头输入/监督；prefix 不含真实 E/R。fit/check 来自实际冻结 validation manifest，不允许悄悄使用 probe 子集。当前 rank/威胁分组读取未遮蔽公开输入，不用最终 rank 标签反推。

只有 cal_fit 拟合三头温度和 log 门控；cal_check 冻结后仅诊断/质量判定，不再调节。缺预登记 quality_policy 在前向前拒绝。

## 产物、计费和状态

在新目录写 `logits.sqlite3`、`costs.jsonl`、`calibration.json`。最终 job 保存 calibration/support/质量报告、完整诊断、真实数据范围和身份。质量不过为 `CALIBRATION_REJECTED`，不能因有样本就 PASS。

模型为 no_grad，验证冻结身份没有变化；不更新网络，但会进行前向和温度/统计搜索，必须单独获评测许可。计算范围包含成功/失败的最后费用结算，forecast 不是实际固定费发票。

## 测试与状态

对应 [core/test_calibration.py](../../../../tests/kar/core/test_calibration.py)、[production/test_review_job_guards.py](../../../../tests/kar/production/test_review_job_guards.py)、[test_diagnostic_distributions.py](../../../../tests/kar/production/test_diagnostic_distributions.py)。入口经 `jobs_cli` 的 calibrate；本轮未执行模型、标定或测试。
