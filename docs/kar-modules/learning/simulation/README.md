<!-- kar-module: riichienv.kar.learning.simulation -->
<!-- api: M3Config, register_m3_run, open_m3_ledger, run_m3, simulation_runtime_identity, benchmark_plan, benchmark_matches -->
# learning.simulation — 完整成对基准与独立 M3

源码：[simulation.py](../../../../src/riichienv/kar/learning/simulation.py) · [learning 索引](../README.md)

## 基准接口

`benchmark_plan(...)` 只读 release 元数据登记计划；`benchmark_matches(..., matches, seed, output, budget, mode="paired", ...)` 实际加载模型并测完整作业。matches 是**两臂合计半庄数**，paired 至少8且为8倍数，四个主体座位平衡。

baseline 用同标定先验/infinity，candidate 保留冻结有限策略。模型加载、agent、完整对局、压缩 replay、收据和记账都纳入计时。只跑 prior 或 candidate 没有真正进行辅助前向的结果，不能作为 M3 成本资格。

`simulation_runtime_identity(device)` 记录实际主机/运行时/线程与适用 GPU 信息，调用它也不是本轮静态文档任务可擅自做的资源查询。

## 正式 M3

`M3Config` 固定配对计划、subject/三个对手路径、output、共享 test registry、预算、device、cap、统计及 benchmark_report。`run_m3` 先许可/计划/证据检查，严格固定 test、10000 pairs/20000场，不因费用不足减少。

`register_m3_run` 把唯一计划绑定实际 output/账本与原计价；换目录或降价不能重获USD250。`open_m3_ledger` 在任何新计算前验证全部完成/失败/最终收据的费用锚点，已登记运行丢账本拒绝重建。

## 配对、恢复与产物

两臂同墙/对手初始流，主体 RNG 分开，不宣称后续反事实状态相同。运行时需匹配测量，并恢复测量 CPU 线程数。写逐 arm receipt、gzip replay、费用与最终统计；恢复核验来源、block、arm、replay hash 和账本。孤儿 replay 不自动覆盖。

最终报告含平均顺位/点数/一四位/和牌放铳、配对游戏级区间、支持与实际概率变化覆盖、延迟直方图区间。所有结果为经验评测，不附统一误差证书。截断/不完整场不计完成。

## 测试与状态

对应 [production/test_paired_benchmark.py](../../../../tests/kar/production/test_paired_benchmark.py)、[test_m3_registration.py](../../../../tests/kar/production/test_m3_registration.py)、[test_review_job_guards.py](../../../../tests/kar/production/test_review_job_guards.py)。经 `jobs_cli` 进入；本轮未查询硬件、加载模型、计费或执行任何对局。
