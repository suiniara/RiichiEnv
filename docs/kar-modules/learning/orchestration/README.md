<!-- kar-module: riichienv.kar.learning.orchestration -->
<!-- api: run_m1_ablation, run_m2, torchrun_command, config_from_record -->
# learning.orchestration — M1 消融与唯一 M2

源码：[orchestration.py](../../../../src/riichienv/kar/learning/orchestration.py) · [learning 索引](../README.md)

## 主要接口

`run_m1_ablation(dataset, output, on, off, authorization=None, resume=False)` 固定 Small 两种特征各一完整 Epoch，共享阶段账本。`run_m2(dataset, output, config, forecasts, main_model_registry=..., m1_report=..., ...)` 选择并登记唯一主力模型，调用正式 trainer。

`torchrun_command(...)` 只构造显式多进程命令列表，不执行；`config_from_record` 使用带派生字段校验的配置解码，成本使用 `CostForecast.from_record`，不能直接展开 to_record。

## M1 合同

两臂 seed、实际数据、模型、样本顺序、batch、设备和精度一致；特征模式相反。预算在启动前联合检查，不能两臂各获得一份USD100。完成时核对真实初始权重和已提交顺序，输出 `M1-ablation.json` 与完整 arm 报告。

## M2 合同

按实际 N_train 和已测成本选一个 tier，不跑完整规模矩阵。必须消费完整 M1 证据；扩到 Medium/Large 还要对应 Small checkpoint/特征模式的验证收益。登记固定 output/training 与 M2 账本、配方及 prerequisites；原始 trainer 再验证。

恢复先核验现有 run/检查点/来源/账本，完成检查点可重建未发布结果，不能把损坏报告当理由从零训练或覆盖历史。

## CLI 与副作用

顶层 `m1` / `m2` 固定 phase，读取 `--dataset --output --config --authorization`，支持明确 resume。M2 配置包含唯一 registry、`m1_report`、可选 `m1_tau_selection_job` 和 forecasts。实际调用会训练、开进程组、写日志/费用/检查点，不能由文档自动启动。

## 测试与状态

对应 [production/test_planning.py](../../../../tests/kar/production/test_planning.py)、[test_stage_prerequisites.py](../../../../tests/kar/production/test_stage_prerequisites.py)、[test_jobs.py](../../../../tests/kar/production/test_jobs.py)。本轮未执行调度、训练或 torchrun。
