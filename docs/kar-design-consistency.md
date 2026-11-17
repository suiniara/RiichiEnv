# KAR-v1.2.2 / RiichiEnv 设计一致性与迁移合同

本次基线：`e24c24bd4b0ace681a2feaa093eede94754aafa2`，分支 `kar/riichienv-v1-20261117`。
正式主设计为工作区设计包的 `KAR-v1.2.2-DESIGN.md`，文件 SHA256：
`4349039c44175ece77f6dddad3a2106fbd3e0e0339935122df242175c83cbdba`。
本轮与原始设计包逐字节核对一致，没有改写正式设计。

## 1. Schema 是显式宿主物化，不是改名兼容

| 身份 | 值 | 用途 |
|---|---|---|
| 设计规格引用 | `kar-v1.2.2` | 正式设计规定的语义与架构 |
| 完整活动 Schema | `kar-riichienv-v1` | RiichiEnv 的完整 kind/字段/二进制注册表 |
| 完整 canonical Schema hash | `fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673` | 数据、Token 和 checkpoint 的实际身份 |
| `reference_design_schema_id` | `kar-v1.2.2` | 引用设计，不是另一个可接受的活动 ID |
| `reference_overlay_sha256` | `90286d088014eb72e9f99f1b3c35d0a7553bcf5fec99e8aeb3ba5f02a010716a` | 原始部分 overlay 文件的真实字节 SHA，不是完整 Schema hash |
| `schema_mapping_version` | `riichienv_complete_v1_from_kar_v1_2_2_v1` | 明确记录宿主注册表与规格的关系 |
| legacy 兼容性 | `not_claimed` | 不自动转换旧 Token、BC/CQL 或 v1.2.1 checkpoint |

完整 JSON、活动 ID、139 字节 Token、944 Embedding 行、62 状态槽和 canonical hash 均保持不变。映射保留设计的字段/三态/presence/块注意力合同，并明确采用完整宿主注册表的 kind、编码和逻辑位置；它不是声称已有一份完整 Kanachan 父注册表可逐字节互换。仅形状或参数量相等不能证明 checkpoint 兼容。

`schema.py` 现在也显式检查设计 ID 与 overlay 引用。`versioning.artifact_contract()` 将这些引用与实际 host ID/hash 一起绑定到数据、运行时、检查点和发布。将 `schema_id` 换成 `kar-v1.2.2`、把 overlay SHA 当完整 hash，或省略映射版本，都拒绝；没有“改 ID 后重新算 hash”的迁移。

## 2. 半庄口径：正式设计内部冲突及用户裁定

正式设计 §7.5 明确将 `N_eligible_half_games` 定义为切分前的合格完整半庄池，§14.2 也引用它选型；但 §17 又明确要求按训练 split 的完整半庄数选型。本轮没有伪装成两处原文一致，也没有修改正式文件。

**用户本轮明确选择 §17 的保守 train 口径。** 实施合同如下：

- `N_eligible_half_games`：规则/cohort/完整性/Golden 排除等审计后、整场 split 前的合格池总数。
- `available_split_counts`：该完整合格池按 game_id 切分后的 train/validation/test 数量，总和等于 N_eligible。
- `exported_split_counts`：当前实际导出的完整游戏数，不能大于 available。
- `N_train_half_games`：当前实际导出 train 的完整半庄数；完整主力数据必须覆盖全部 available train/validation，探针子集不冒充全量。
- `model_tier_basis=N_train_half_games`，不得用小局、决策行、四视角或原始文件数量替代。

| N_train 完整半庄 | 数据允许的最大主力规模 | 有效 batch |
|---:|---|---:|
| <30000 | Small | 128 |
| 30000..49999 | Medium | 256 |
| >=50000 | Large reference | 512 |

预算不足只按已有 measured forecast 逐级降档，或停止；不因此改变损失、层数、宽度或 M3 数量。相同 N_train/N_eligible 与 basis 进入导出 manifest、生产配置、forecast recipe SHA、release contract 和 run_manifest。M1 仍固定 Small；它的临时数据子集与完整 M0 池分开记录。

`training_plan_version=budget_ladder_v1` 不变；配置序列化另明确使用 `training_config_version=explicit_train_split_scale_v2`。`from_record` 不再替缺失的旧版本字段补当前默认值。该裁定解决实现口径，但 §7.5/§14.2 与 §17 的正式文字仍需要后续文档勘误；不能把本轮结果称为逐字消除了原文矛盾。

## 3. Response Mask v2 是真实规则修正，不回退为 v1

活动版本为 `public_rule_response_mask_v2_mjsoul_nagashi`。其外部规则依据和资格矩阵见 [流局满贯规则合同](kar-nagashi-rules.md)。正式设计中 v1 标签属于参考版本，不允许据此把本轮已确认的雀魂规则修正回滚。

v2 公开事实白名单：observer 座位、候选舍牌、公开舍牌历史/当前河、已成立 meld、公开立直接受状态、活牌山余量、历史完整性与 pending-kan 标志。它不读暗手、隐藏牌山、未来事件、真实响应标签或他家私有合法动作列表。

- 雀魂三家和保留 `RON_MASK_7`，不作三家和流局；规则开关的相反配置仍有明确 mask 语义。
- 完整公开历史中的永久振听限制适用于相应荣和集合。
- 四风连打、四家立直和四杠终止只根据规定的公开条件判断；同一玩家四杠与多人四杠分开。
- 活牌山耗尽时禁止正常摸牌/大明杠，保留对应终局类别。
- 流局满贯按自己的幺九字牌且未被叫走判断，自己鸣牌/杠本身不取消；已知中张在历史不完整时仍能否定，真正未知保留可能性。
- 四家分别判断后 any；pending 杠窗口不冒充普通舍牌响应窗口。

导出与 runtime 直接引用同一个 `public_rule_response_mask` 函数。训练标签保留该 allowed mask，loss 在 Softmax/CE 前应用；推理在响应 Softmax 与 unsupported 质量计算前应用。mask/响应标签不是模型输入。

## 4. 分开特征语义与随机置空身份

过去 `FEATURE_VERSION=sample_mask_v1` 混淆了两个合同。现在：

- `feature_contract_version=visible_rule_features_v1_2_2`：Genbutsu、Full Suji、Ryanmen、公开 Dora、当前顺位和分差语义。
- `rule_feature_mask_version=sample_mask_v1`：sample-level Bernoulli(0.25) 增强。

随机流继续使用 `sample_mask_v1` 与 seed/epoch/game/decision/draw，未因改正语义标签而意外重播不同随机流。选中样本时 34 个 HAND_COUNT flags 同步 N/A，ROUND_STATE numeric[18:20] 同步归零并清 present；stage 和其余字段不变。全局 feature-off 另清 stage。验证/推理不随机置空；模型参数不变。

## 5. 统一产物身份与拒绝边界

`artifact_contract_version=kar_riichienv_contract_v2` 由 [versioning.py](../src/riichienv/kar/versioning.py) 集中定义。除了上述 Schema 及设计引用，还显式记录 architecture、ruleset、action manifest、feature、response mask、rule feature mask、training plan、loss 与 model-tier basis。

- M0 producer scope 带同一身份，跨分片 acceptance 不丢弃这些字段；旧 scope 不能自动补齐。
- Dataset 校验全部字段，SQLite 内部 scope 必须等于 manifest，games 表 split 计数必须符合声明；不能只改外层 manifest 的标签。
- 两类 checkpoint 保存/读取活动合同；production reader 即使调用方传入与旧文件相同、校验和有效的旧合同，也拒绝，而不是只比较二者是否相等。
- FrozenRelease 在加载权重前验证活动版本及 training contract。
- runtime、支持、校准和发布均携带统一合同；release 不再硬编码 `public_rule_response_mask_v1`。
- 训练配置、数据计数、预算 forecast recipe 必须一致，完成 run_manifest 显式保存 tier、basis 及两个 N。

v1 数据、缺字段的旧 host 数据、旧支持/校准、旧检查点或旧发布不能通过重签 hash、补字段或改目录名静默变成 v2。正常受支持加载器明确拒绝，重建/迁移须另行真实核验。hash 是内容与来源检查，不是秘密签名，也不能替可信来源或用户授权。

## 6. 实际验收及停止边界

本轮十五项 CPU 验证的逐项状态和用例入口见[实现覆盖报告](kar-implementation-coverage.md)。只有实际通过的条目标 PASS；58 项总实现表本身仍不是阶段运行证明。

实际执行包含 Small/Medium/Large 参数构造、临时小模型 forward/backward、真实两进程本机 Gloo 梯度比较、非均匀 microbatch 梯度累积、合成 SQLite/checkpoint/发布的负向版本验证与有限规则回归。没有启动生产/探针 launcher，没有给真实数据签发更新许可，没有调用阶段训练或 M3。

未运行完整阶段训练、GPU/BF16 硬件验收、全量真实数据重导出、真实校准/支持重建或棋力实验；不覆盖历史模型/账本/数据库，不以旧 M0 或 commit message 充作本轮证据。本轮结束后停止，不自动进入 M1/M2/M3。
