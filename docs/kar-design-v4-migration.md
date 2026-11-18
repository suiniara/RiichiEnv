# KAR-v1.2.2 / RiichiEnv V4：相对旧附件的差异与迁移

唯一活动正文是 [KAR-v1.2.2 / RiichiEnv V4 完整主设计](KAR-v1.2.2-DESIGN.md)，分发版是它的字节一致快照。本表是迁移说明，不定义第二套合同。实现基准：`suiniara/RiichiEnv`、`kar/riichienv-v1-20261117`、`f70684ef7aa69e56e6a530dcca13326fbf2c83b5`。

历史附件 `KAR-v1.2.2-DESIGN(3).md` 的 SHA-256 为 `4349039c44175ece77f6dddad3a2106fbd3e0e0339935122df242175c83cbdba`。本次对照读取的是本地历史包中 SHA-256 完全相同的 `KAR-v1.2.2-DESIGN.md`；不依赖括号序号。历史文件保持原状。下表「旧附件」节号均指该字节版本。

| 主题 | 旧附件 | 当前 RiichiEnv V4 | 迁移处理 |
|---|---|---|---|
| 实施目标与分发 | 页首以 Kanachan 为实施目标，RiichiEnv 为参考。 | RiichiEnv 是宿主；算法仍标识 `KAR-v1.2.2`，Kanachan 仅是牌与动作编号参考。 | 分发 V4 完整正文，不随附件序号切换规范；历史附件只用于追溯。 |
| Schema 身份与摘要 | 协议及 §14.4、§19 使用 `schema_id=kar-v1.2.2`，记录部分 overlay，并约定完整 registry 合并后再算 hash。 | 活动完整 Schema 为 `kar-riichienv-v1`；`kar-v1.2.2` 仅为来源引用。`schema_hash` 基于规范化的完整 Schema JSON，`reference_overlay_sha256` 基于历史 overlay 的原始字节。 | 两个身份及两个摘要不得互换；不能只改 `schema_id` 或替换摘要冒充兼容。见主设计 §0。 |
| 行数与位置 | §4.1、§6 按历史行数 H 取状态/DECIDE 的 H+1 和 ACTION 的 H+2。 | H 仅为历史 Token 行数（不含 BOS）；C 是公共语义时钟。长度仍是 H+64 / H+65，位置改用 C+1 / C+2。 | 受旧位置语义影响的 Token 和缓存必须按可核验事件重建；不是替换位置常数或源记录序号。见 §4.1、§6。 |
| 模型选型 | §7.5、§14.2 按切分前 `N_eligible_half_games`；§17 又写训练 split。 | `N_eligible_half_games` 是审计合格池总量；`available_split_counts` 是其完整切分，`exported_split_counts` 是实际导出切分。仅 `N_train_half_games=exported_split_counts["train"]` 用于选型。 | 计划与主模型登记均用实际导出的 train 完整半庄数；不能用可用池或探针子集冒充正式 M2 数据。见 §7.1。 |
| 输入、线上与规则版本 | §11.4、§19 记录 `public_rule_response_mask_v1`；没有 V4 输入和在线投影版本。 | 完整版本见主设计 §0：mask v3、语义时钟输入 v2、在线投影 v5（含分批 Pass receipt），以代码中的完整标识为准。 | 旧输入、规则及私有快照不能补标签变成当前版本。已丢失的私有 Pass 只能凭可信提交凭据重投影，不从公共轨迹猜测。 |
| `known_counts()` 与终局揭示 | §4.8.4 将任意 cutoff 前已公开的手牌及其他公开牌纳入计数；§21 列出 `other_public_reveals`。 | [`PublicSnapshot.known_counts()`](../src/riichienv/kar/observation.py) 只向计数函数传自己的暗手、当前河、已成立副露与公开指示牌。终局揭示走 `PUBLIC_HAND_REVEAL` 历史 Token；`closed` 不是允许决策阶段。 | 底层 [`known_count_base34()`](../src/riichienv/kar/rule_features.py) 即使有可选参数，也不扩大快照 API 的范围。不能把终局揭示额外加进当前决策计数，或用该计数代替任意 cutoff 的全公开牌查询。见 §4.6、§8.1。 |
| 有效事件支持 | §10.4、§11.4.3 保留 500 个独立完整游戏筛查及校准诊断，未定义 V4 的逐分支质量交集。 | `EventSupport.supported` 仅为独立 train 局数 ≥500；有效价值支持还需要同响应的 response/rank 两项预登记 `cal_check` 判据均通过。`failed`、`unavailable` 都不放行。 | 仅旧计数或总体质量通过不能启用有限策略。保留响应概率，非支持分支取 −1.5；没有合格分支则回退先验。见 §10.2、§11.3。 |
| 校准、统计与两种未支持指标 | §11.4.3 的 $\rho_U$ 对未支持事件求和，未区分当前双口径的持久化报告。 | 校准 v5、诊断 v3。`unsupported_mass_histogram` 使用最终有效支持；`train_only_unsupported_mass_histogram` 只使用 train 计数位，两者使用公开 mask 后的概率。 | 旧 train-only 桶不得当作有效质量。需要有效报告时，从来源核验过的冻结 logits、绑定支持与已预登记的质量判据重新计算；不直接改名或重签。见 §11.5、§19。 |
| 核心模型、损失与策略 | §4.7、§6–12 定义五因子、三头、固定分母、候选隔离、受限 KL 与误差/证书条件。 | 保留完整 LEGAL_SET 五表均值、三头及各规模参数数、FP32 读出、固定分母损失、分支隔离和概率质量保持。名义值误差范围与保守值误差范围分开写明。 | 不因文档或报告口径改动而改变权重、参数量、行为采样或证书门槛；登记的逐状态误差界与纯经验指标必须分开。 |
| 证据 | 历史设计及其测试要求。 | [本次文档校验](kar-design-v4-validation.md)仅验证当前正文、实现对应关系和分发身份。 | 历史运行证据保留其提交、版本和范围；不把其通过数用作本次结果。不运行真实数据、反向、GPU、校准或 M3。 |

已满足当前 V4 输入合同的 Token 不因本次文档同步而全量重导出。迁移前按实际 artifact/runtime/源码绑定核准；仅历史语义或绑定不符的产物才需在相应范围内重建或重新校验。本说明不执行任何旧产物迁移。