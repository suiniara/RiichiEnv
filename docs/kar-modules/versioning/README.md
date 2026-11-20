<!-- kar-module: riichienv.kar.versioning -->
<!-- api: artifact_contract, validate_artifact_contract, game_count_contract -->
# versioning — 设计引用、产物身份与半庄计数

源码：[versioning.py](../../../src/riichienv/kar/versioning.py) · [总索引](../README.md) · [设计一致性合同](../../kar-design-consistency.md)

## 职责与接口

`artifact_contract()` 返回数据、运行时、检查点和发布共用的显式身份。活动格式为 `kar_riichienv_contract_v4`、输入 `semantic_public_clock_frozen_decisions_v7_native_rules`、特征 `visible_rule_features_v2_ryanmen_pairs`；`native_rules_version=riichienv_native_rules_v4_nagashi_mask` 要求包含末巡确定流局满贯排除普通荒牌的原生扩展，保留四组面子容量与在线食替开关修复。仍绑定 `native_replay_version=native_replay_context_v4`、公开时钟、终局公开事实与 one-vs-rest 分支质量。保留完整 host Schema 和 N_train 口径；Response Mask 为 `public_rule_response_mask_v5_certain_nagashi`。旧扩展必须重建，旧 mask 数据及其绑定产物必须重新生成或核验；旧语义或缺字段的产物明确拒绝，不因参数形状相同而兼容。冻结推理另以 `fp32_ieee_single_prefix_v3` 绑定实际后端合同。完整表见[主设计](../../KAR-v1.2.2-DESIGN.md)。

`protocol_identity_version=protobuf_source_and_descriptor_v1` 要求源协议与完整描述符的双重身份。数据相关 scope 另携带 `protocol_sha256`（顶层原始 proto）和 `protocol_descriptor_sha256`（包含 imports 的编译结果）；协议依赖修改不能沿用旧审计、恢复记录或导出。Golden 与 completion 证据格式同步为 v2。没有这组身份的旧产物必须从 raw 重新验证，不能只重新签署旧内容。

`validate_artifact_contract(record, context=...)` 对每个必需字段检查精确值；缺失不填当前默认值，旧记录必须明确拒绝。

`game_count_contract(available_split_counts, exported_split_counts=None)` 分开保存完整审计 cohort 和实际导出子集，检查 train/validation/test 的整场计数。`N_eligible_half_games` 是切分前合格池总数，`N_train_half_games` 是当前导出实际 train 完整半庄数；主力选型固定使用后者。

## Schema 与迁移边界

`kar-v1.2.2` 是规范引用，`kar-riichienv-v1` 是独立完整注册表身份；overlay SHA 不是完整 canonical Schema hash。映射记录不声称 legacy 字节或 checkpoint 兼容，不改 ID、参数形状或输入编码。

读取合同会校验包内 Schema 文件和原生规则能力版本，但本模块不导入 Torch、不创建模型、不写数据、不运行训练或自动迁移。旧扩展须重建，旧验收/派生产物须按[规则迁移要求](../../kar-native-rule-boundary.md#6-旧数据与产物的迁移)重新验证或生成，不能补版本字段、换 hash 冒充迁移。所有 hash 都是内容/来源检查，不是秘密签名。

`Ruleset` 的两个平台开关显式绑定 KAR 雀魂的 `false` 值；原生 Tenhou/default 和无规则参数的旧听牌 API 保留兼容行为。通用原生 API 能读取旧规则对象，不表示旧对象或旧训练产物满足当前 KAR 雀魂合同。

## 调用、错误与测试

上游：[schema](../schema/README.md)、[contracts](../contracts/README.md)。下游：[export](../learning/export/README.md)、[runtime_contracts](../runtime_contracts/README.md)、[release](../release/README.md)及两类 checkpoint 加载器。

字段缺失、旧版本、设计引用错误、bool/负数半庄计数、导出数量超过审计数量均报错。对应 [test_design_contracts.py](../../../tests/kar/test_design_contracts.py)、[m1/test_design_versions.py](../../../tests/kar/m1/test_design_versions.py)。本轮没有运行阶段作业。实际 CPU 合同验证状态见[实现覆盖报告](../../kar-implementation-coverage.md)；API 可调用不授予阶段运行许可。
