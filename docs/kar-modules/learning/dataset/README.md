<!-- kar-module: riichienv.kar.learning.dataset -->
<!-- api: TrainingBatch, collate_examples, DecisionDataset -->
# learning.dataset — 只读共享历史决策数据

源码：[dataset.py](../../../../src/riichienv/kar/learning/dataset.py) · [learning 索引](../README.md)

## 职责与接口

`DecisionDataset(directory, split, verify_hash=True, history_cache_items=32)` 读取 `manifest.json` 和完整 `decisions.sqlite3`，只接受 train/validation/test。`__getitem__` 通过共享 game/observer 历史与该决策 state/ACTION 重建样本，不把四视角历史重复存成完整矩阵。

`collate_examples(examples)` 返回 `TrainingBatch(inputs, labels, game_ids)`；其中 `inputs` 是 `ModelBatch`，`labels` 是独立 `Labels`，可整体迁移设备。

## 数据与验证

默认校验数据库 hash、导出完成性、scope、Schema/规则/特征/损失/角色版本和分区信息。现在还要求显式设计映射、`rule_feature_mask_version`、`training_plan_version` 与计数口径；SQLite 内部 scope 必须等于 manifest，实际 games 表的 split 计数必须等于导出计数。真实最终 rank 从外部 1..4 明确转换到训练 0..3。action/response/value 有效位及响应 allowed mask 分开保存。

协议身份要求 `protobuf_source_and_descriptor_v1`，且 scope 必须同时存在有效的顶层 source SHA 和含 imports 的 descriptor SHA。只填当前版本、缺少任一摘要的旧导出不能进入训练；需要重新审计、导出来源数据。

SQLite 以只读方式使用；每个进程自己的连接与有限历史缓存不应在 fork/worker 之间错误复用。`verify_hash=False` 不是生产迁移办法，不能以省时理由信任未知数据库。

## 使用与接线

上游 [export](../export/README.md)，下游组批/训练/完整验证/标定。M2 必须读取实际完整 train 与 validation，而不能沿用 probe subset；这层提供证据，生产 trainer 再执行阶段约束。

读取数据不自动启动训练，但打开/核验大型文件仍有 I/O 副作用。无独立 CLI，不写回 SQLite，不修改 raw。

## 私有回归夹具选择与更新

`tests/kar/m1/test_dataset.py` 可通过 `KAR_M1_TEST_DATASET_DIR` 显式选择测试数据；未指定时优先使用工作区 `kar-m1-data/cpu-current-contract`，只在它不存在时回退到历史 `cpu-probe-v2`。选中目录不完整或合同过期仍然报错，不能因为有旧资产而静默 skip；只有没有配置且两处私有资产都不存在时才沿用缺资产 skip。

当前合同数据必须从匹配本版显式身份的真实 M0 验收清单重新 replay/编码，不能改旧 manifest 字段假装兼容。`final-acceptance-20261116`、`cpu-current-contract` 等既有目录名是历史名称，不证明其内容符合当前 `kar_riichienv_contract_v2`。

本轮没有重导出或改写这些目录。后续若单独获准重建，须先重新取得当前合同的 M0 验收，按 [export](../export/README.md) 接口写入全新且位于 raw 目录之外的目录，再通过显式测试变量选择；不能覆盖旧证据或放宽 `DecisionDataset` 校验。

## 错误、测试与状态

缺 manifest、数据库 hash 或合同冲突、未知 split、非法 sample/候选/标签会拒绝，不恢复成随机样本。对应 [m1/test_dataset.py](../../../../tests/kar/m1/test_dataset.py)、[production/test_full_trainer.py](../../../../tests/kar/production/test_full_trainer.py)。示例不授予运行或更新许可；本轮未运行训练或 GPU 作业，实际 CPU 数据与测试验收以绑定来源和源码版本的结果为准。
