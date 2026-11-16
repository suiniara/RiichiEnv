<!-- kar-module: riichienv.kar.planning -->
<!-- api: plan_benchmark, plan_m3, main -->
# planning — 不加载模型的冻结计划生成

源码：[planning.py](../../../src/riichienv/kar/planning.py) · [总索引](../README.md)

## 职责与接口

`plan_benchmark(config)` 读取 subject/opponents 的 release.json，输出冻结的 benchmark plan 与 plan SHA。`plan_m3(config)` 消费合格成对测量，生成固定 10000 pairs、完整成本预测和作业配置。

只处理元数据和路径，不读取权重、不创建模型、不启动资源。结果明确保存 `planning_only=true`、`model_computation_performed=false`、`authorization_granted=false`。

## 输入配置

| 计划 | 主要字段 |
|---|---|
| benchmark | `subject_release`、三个 `opponent_releases`、`seed`，可给 `matches`（缺省8）、device/mode/max_decisions |
| M3 | 上述 release 路径、`benchmark_report`、`master_seed`、`resource_usd_per_hour`、`fixed_cost_usd`、`budget_cap_usd`、`budget_spent_usd`、`other_phase_spent_usd`、`run_output`、`test_registry` |

M3 可带固定费真实收据和预登记统计/safety 参数。不得把已知费用丢掉；超预算直接停止，不缩减 pairs。

## CLI 与输出

从仓库根目录：`plan-benchmark --config INPUT --output PLAN` 或 `plan-m3 --config INPUT --output CONFIG`，经顶层 KAR 入口分发。输出父目录须存在，文件用独占创建，拒绝覆盖。

benchmark 输出是**登记计划**，不是带授权/预算即可自动启动的隐藏作业。M3 输出可供 `jobs_cli` 读取，但仍要独立核验并提供用户批准。实际执行时还会验证文件、模型和运行时是否仍匹配。

## 失败、测试与状态

缺/过大 JSON、模型身份不符、prior-only 测量、候选前向未覆盖或成本超过上限均拒绝。对应 [production/test_metadata_planning.py](../../../tests/kar/production/test_metadata_planning.py)、[test_paired_benchmark.py](../../../tests/kar/production/test_paired_benchmark.py)。本轮未执行计划生成、读取 release 产物或启动任何作业。
