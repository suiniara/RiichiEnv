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

预算的 `work_units` 以完整 validation 的决策行为单位，不以游戏数、microbatch 数或 cal_fit 子集计数。入口在创建输出和计算账本前要求预测工作量至少覆盖 `len(validation)`，同时保持 phase 与 recipe 的绑定检查；不足时不生成半完成作业目录。每批只预留该批实际处理的行数，最后一个不足 microbatch 的批次不会按满批虚增工作量。

吞吐与固定费用证据必须覆盖同一冻结 recipe 的 logits 收集、归档、温度拟合和 cal_check；完整行数是准入下限，不是“前向以外的计算免费”的声明。不任意按温度候选数增加模型前向次数：每行 logits 只收集一次，后续拟合和检查读取归档，实际全部耗时仍由外层账本结算。

模型为 no_grad，验证冻结身份没有变化；不更新网络，但会进行前向和温度/统计搜索，必须单独获评测许可。计算范围包含成功/失败的最后费用结算，forecast 不是实际固定费发票。

## 测试与状态

对应 [core/test_calibration.py](../../../../tests/kar/core/test_calibration.py)、[production/test_calibration_budget_admission.py](../../../../tests/kar/production/test_calibration_budget_admission.py)、[production/test_review_job_guards.py](../../../../tests/kar/production/test_review_job_guards.py)、[test_diagnostic_distributions.py](../../../../tests/kar/production/test_diagnostic_distributions.py)。预算回归使用合成数据、阻断计算入口或固定 logits 与模拟时钟，覆盖不足工作量拒绝、精确完整行数准入及末批预留；不构成真实吞吐测量或正式校准。入口经 `jobs_cli` 的 calibrate。
