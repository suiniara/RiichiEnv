<!-- kar-module: riichienv.kar.experiments -->
<!-- api: independent_seed, PairedSeedBlock, paired_seed_manifest, validate_paired_manifest -->
# experiments — 固定的 M3 配对种子计划

源码：[experiments.py](../../../src/riichienv/kar/experiments.py) · [总索引](../README.md)

## 职责与接口

`independent_seed(master, *parts)` 以明确命名空间产生稳定 seed。`PairedSeedBlock` 保存一对实验的墙、座位、主体两臂 RNG、对手 RNG/版本、起始点数和固定起家顺序。

`paired_seed_manifest(...)` 创建 **10000 blocks / 20000 半庄** 的计划；`validate_paired_manifest(document, require_benchmark=False)` 重新构造并比较全部字段，不只检查表面计数。

## 冻结内容

四个主体初始座位各 2500 pairs；baseline/candidate 同 checkpoint、同初始牌山和同固定对手流，主体策略 RNG 分开。不同动作会自然改变后续轨迹，不能声称后续状态反事实相同。

计划冻结 ruleset、device、完整对局 cap、统计 seed/次数/置信度、主体/对手 release IDs 和 benchmark SHA。版本 `balanced_paired_seed_blocks_v2`。

## 草稿与可执行计划

未给 `benchmark_sha256` 时允许构造 `execution_ready=false` 草稿供规划；`require_benchmark=True` 拒绝它。正式 `run_m3` 还要求 test 分区、匹配的成对测量、完整预算和独立用户许可，因此 execution_ready 不表示已授权。

这里仅构造/校验元数据，不跑模型或对局，不自行写文件。产物的写入由 [planning](../planning/README.md)或明确调用方负责；不支持默默减少 pair_count。

## 测试与状态

对应 [production/test_jobs.py](../../../tests/kar/production/test_jobs.py)、[production/test_paired_benchmark.py](../../../tests/kar/production/test_paired_benchmark.py)。本轮未生成或执行实际实验计划；hash 是一致性摘要，不是棋力或费用已验证的证明。
