<!-- kar-module: riichienv.kar.simulation_contracts -->
<!-- api: benchmark_plan_from_records, qualify_benchmark_report, validate_m3_forecast -->
# simulation_contracts — 成对吞吐与成本证据合同

源码：[simulation_contracts.py](../../../src/riichienv/kar/simulation_contracts.py) · [总索引](../README.md)

## 职责与接口

纯元数据模块，不导入 Torch 或创建 agent。

- `benchmark_plan_from_records(subject, opponents, matches, seed, device, source_sha256, ...)`：从已核验 release 元数据登记具体基准。
- `qualify_benchmark_report(report)`：核验完成状态、两臂/四座数量、实测耗时、候选实际前向及报告身份。
- `validate_m3_forecast(paired, forecast, benchmark, source_sha256=...)`：确保正式成本预测使用这个测量，而非无来源的速率数值。

## 基准约束

默认 paired；`matches` 是两臂合计半庄数，至少 8 且是 8 的倍数。baseline 模式仍可明确登记诊断，但不具备 M3 成本资格。subject 必须是冻结随机有限 empirical candidate，不能是 argmax 或 infinity。

报告必须证明 baseline 零辅助前向、candidate 至少真正执行一次响应/价值候选前向；计时对应完整作业而非后缀样本数。记录实际 CPU 线程等 runtime 身份。

## M3 成本绑定

主体、三个对手、规则、代码、后端、完整对局 cap 与计划一致；M3 速率不得比匹配测量更快，引用为 `benchmark:<sha256>`。benchmark 与正式 test 墙 seed 拒绝重叠。预算完整工作量由 M3Config 再检查。

报告 hash 和版本正确不自动让恶意或虚假的测量可信；调用方仍需保留实际生产者/收据来源。这里不预测策略收益，不授予执行许可。

## 测试与状态

上游 [learning/simulation](../learning/simulation/README.md)生产实际测量，[planning](../planning/README.md)生成元数据配置，下游 M3 消费。对应 [production/test_paired_benchmark.py](../../../tests/kar/production/test_paired_benchmark.py)。本轮未执行测量、预测或测试。
