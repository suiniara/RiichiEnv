<!-- kar-module: riichienv.kar.learning.dataset -->
<!-- api: TrainingBatch, collate_examples, DecisionDataset -->
# learning.dataset — 只读共享历史决策数据

源码：[dataset.py](../../../../src/riichienv/kar/learning/dataset.py) · [learning 索引](../README.md)

## 职责与接口

`DecisionDataset(directory, split, verify_hash=True, history_cache_items=32)` 读取 `manifest.json` 和完整 `decisions.sqlite3`，只接受 train/validation/test。`__getitem__` 通过共享 game/observer 历史与该决策 state/ACTION 重建样本，不把四视角历史重复存成完整矩阵。

`collate_examples(examples)` 返回 `TrainingBatch(inputs, labels, game_ids)`；其中 `inputs` 是 `ModelBatch`，`labels` 是独立 `Labels`，可整体迁移设备。

## 数据与验证

默认校验数据库 hash、导出完成性、scope、Schema/规则/特征/损失/角色版本和分区信息。真实最终 rank 从外部 1..4 明确转换到训练 0..3。action/response/value 有效位及响应 allowed mask 分开保存。

SQLite 以只读方式使用；每个进程自己的连接与有限历史缓存不应在 fork/worker 之间错误复用。`verify_hash=False` 不是生产迁移办法，不能以省时理由信任未知数据库。

## 使用与接线

上游 [export](../export/README.md)，下游组批/训练/完整验证/标定。M2 必须读取实际完整 train 与 validation，而不能沿用 probe subset；这层提供证据，生产 trainer 再执行阶段约束。

读取数据不自动启动训练，但打开/核验大型文件仍有 I/O 副作用。无独立 CLI，不写回 SQLite，不修改 raw。

## 错误、测试与状态

缺 manifest、数据库 hash 或合同冲突、未知 split、非法 sample/候选/标签会拒绝，不恢复成随机样本。对应 [m1/test_dataset.py](../../../../tests/kar/m1/test_dataset.py)、[production/test_full_trainer.py](../../../../tests/kar/production/test_full_trainer.py)。本轮未打开实际数据集或运行测试。
