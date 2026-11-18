<!-- kar-module: riichienv.kar.statistics -->
<!-- api: percentile, GameClusters, ReliabilityBins, LatencyDistribution, ClassificationMetrics, ResponseMassHistogram, FullDiagnostics -->
# statistics — 自然频率指标、聚类区间与分布

源码：[statistics.py](../../../src/riichienv/kar/statistics.py) · [总索引](../README.md)

## 主要接口

| 接口 | 作用 |
|---|---|
| `ClassificationMetrics` | NLL/Brier/top1、各类数量/召回/分布与可靠性 |
| `ReliabilityBins` | 固定分桶的预测/实际频率与 ECE |
| `GameClusters` | 每 game 的总量/权重，整场 bootstrap 区间 |
| `FullDiagnostics` | 三头、21 观察分支、威胁/rank 切片、utility、M/S/rho_U 分布 |
| `LatencyDistribution` | 可合并、有限内存的延迟直方图 |
| `percentile` | 已给定静态数值的插值分位数；空集合返回 None |

## 统计合同

决策行不是独立样本：同一半庄重复一千条仍只有一个 game cluster，不能产生虚假的窄区间。分类均值保持自然有效样本频率，不因罕见类重采样而冒充自然概率。零样本指标是 None/证据不足，不是完美的零损失。

`response_event_calibration` 在公开允许且响应可用的行上做逐事件 Bernoulli NLL/Brier/ECE；分布式合并总量与可靠性桶，对正例 game ID 取并集。`response_classification_by_true_event_intervals` 仅为真实类别下的多分类诊断，不是逐事件概率准入。Rank 仍用真实观察到的响应分支，名次分布与效用分开记录。`branch_calibration_claimed=false` 意味着报告不自动声明逐状态校准或安全证书。

## 分布与合并

`FullDiagnostics.add_gate` 记录 M 的 0.01 分桶、0..32 完整支持数及切片，不只保存均值。`ResponseMassHistogram` 在公开 mask/softmax 后统计，保留 support basis、具体支持位、状态、观测数和 101 个桶。有效 `unsupported_mass_histogram` 与 `train_only_unsupported_mass_histogram` 独立；无证据/观测时为 `unavailable`。`state_dict/merge` 供 DDP 聚合，拒绝旧格式或不同支持位的同口径统计，不能平均各 rank 的均值代替总量合并。

相同有序合格游戏集的指标共享 seeded bootstrap 索引；权重不同仍独立计算分母，样本矩阵不常驻内存。`FullDiagnostics.report` 可向调用方 `timings` 记录 bootstrap 耗时，计时不写入冻结校准报告内容。

延迟的 p50/p95 报告明确的直方图区间，不是精确点分位数；单局精确列表分位数与可合并区间不能混叫同一种统计。

## 接线与状态

由 `learning/evaluation`、`calibration_metrics` 和 `learning/simulation` 消费；无 Torch、文件读取或独立 CLI。对应 [production/test_statistics.py](../../../tests/kar/production/test_statistics.py)、[test_diagnostic_distributions.py](../../../tests/kar/production/test_diagnostic_distributions.py)。本轮未做真实模型校准；合成诊断与聚合回归的实际结果见[验收报告](../../kar-pass-rho-math-repair.md)。
