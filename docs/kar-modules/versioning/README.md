<!-- kar-module: riichienv.kar.versioning -->
<!-- api: artifact_contract, validate_artifact_contract, game_count_contract -->
# versioning — 设计引用、产物身份与半庄计数

源码：[versioning.py](../../../src/riichienv/kar/versioning.py) · [总索引](../README.md) · [设计一致性合同](../../kar-design-consistency.md)

## 职责与接口

`artifact_contract()` 返回数据、运行时、检查点和发布共用的显式身份。活动格式为 `kar_riichienv_contract_v3`，新增 `input_pipeline_version=legal_set_pool_private_self_pass_v1`；保留完整 host Schema、设计引用、Response Mask v2、独立特征/置空版本及 train 选型。输入 forward/历史变化不能仅因参数形状相同而兼容，旧 v2/缺 input 版本的记录明确拒绝。

`validate_artifact_contract(record, context=...)` 对每个必需字段检查精确值；缺失不填当前默认值，旧记录必须明确拒绝。

`game_count_contract(available_split_counts, exported_split_counts=None)` 分开保存完整审计 cohort 和实际导出子集，检查 train/validation/test 的整场计数。`N_eligible_half_games` 是切分前合格池总数，`N_train_half_games` 是当前导出实际 train 完整半庄数；主力选型固定使用后者。

## Schema 与迁移边界

`kar-v1.2.2` 是规范引用，`kar-riichienv-v1` 是独立完整注册表身份；overlay SHA 不是完整 canonical Schema hash。映射记录不声称 legacy 字节或 checkpoint 兼容，不改 ID、参数形状或输入编码。

读取合同会校验包内 Schema 文件，但本模块不导入 Torch、不创建模型、不写数据、不运行训练或自动迁移。所有 hash 都是内容/来源检查，不是秘密签名。

## 调用、错误与测试

上游：[schema](../schema/README.md)、[contracts](../contracts/README.md)。下游：[export](../learning/export/README.md)、[runtime_contracts](../runtime_contracts/README.md)、[release](../release/README.md)及两类 checkpoint 加载器。

字段缺失、旧版本、设计引用错误、bool/负数半庄计数、导出数量超过审计数量均报错。对应 [test_design_contracts.py](../../../tests/kar/test_design_contracts.py)、[m1/test_design_versions.py](../../../tests/kar/m1/test_design_versions.py)。本轮没有运行阶段作业。实际 CPU 合同验证状态见[实现覆盖报告](../../kar-implementation-coverage.md)；API 可调用不授予阶段运行许可。
