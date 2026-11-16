<!-- kar-module: riichienv.kar.learning.benchmark -->
<!-- api: peak_process_memory_bytes, benchmark_candidates -->
# learning.benchmark — 单决策候选推理诊断

源码：[benchmark.py](../../../../src/riichienv/kar/learning/benchmark.py) · [learning 索引](../README.md)

## 接口与实际工作

`benchmark_candidates(dataset_dir, preset="small", repetitions=5)` 选择一条 validation 决策，创建**随机初始化**的 CPU 模型，分别测前缀和共享 K/V 的候选 ACTION 后缀。`peak_process_memory_bytes()` 读取当前进程峰值内存：Windows 用系统计数，其他支持平台用 resource，无法读取时为 None。

benchmark 在 no_grad 下执行真实 forward，不做 backward/optimizer.step，但会读取数据、分配模型并使用 CPU。它不是纯元数据工具；该低层函数不自行提供完整冻结作业许可门禁，调用方应在已批准的诊断范围使用。

## 输出与限制

结果包含 prefix tokens/seconds、候选数、`candidate_suffixes_per_second`、当前进程峰值内存、输出形状和明确随机权重标记。`simulation_matches_per_second=None`、`gpu_memory_measured=false`。

重复次数限制 1..100。只测一条决策，不测完整半庄、有限策略门控/规则推进或 GPU，也不证明棋力/标定。不能将该后缀速率拿去预测固定20000场 M3。

## 正确替代入口

完整生产成对吞吐见 [learning/simulation](../simulation/README.md) 的 `benchmark_matches` 和 [simulation_contracts](../../simulation_contracts/README.md)。历史 [probe](../probe/README.md)会调用本函数作架构诊断。

本模块无独立 CLI；不要通过调用内存计数器来擅自查询别人的进程或结束用户进程。

## 测试与状态

相关模型缓存/形状合同见 [m1/test_model.py](../../../../tests/kar/m1/test_model.py)和历史诊断测试。本轮没有调用内存 API、创建随机模型或运行 benchmark。
