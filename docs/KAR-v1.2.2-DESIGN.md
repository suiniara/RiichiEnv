# KAR-v1.2.2 / RiichiEnv V4 完整主设计

**版本：** 算法 `KAR-v1.2.2` · 宿主合同 `RiichiEnv V4`（`kar_riichienv_contract_v4`） · 文档修订 `design-sync-6`。最后一项仅标识文档，不是新的算法、输入或产物版本。

**唯一正文：** `docs/KAR-v1.2.2-DESIGN.md`。对外分发时应从该正文制作逐字节快照，不另维护规范；相对链接以仓库 `docs/` 为基准。此前 `design-sync-1` 的分发文件只对应当时的 `f70684e`，不自动升级。

**适用实现：** `suiniara/RiichiEnv` 的 `kar/riichienv-v1-20261117`。保留此前以 `7626166246ec3ed698d5e6d95b27b2068afbe604` 为基线的[响应监督、冻结 FP32 和验证统计修订](kar-response-fp32-validation-repair.md)及[原生规则边界收敛](kar-native-rule-boundary.md)，并在 `4132c6ab21d0b17ecc901763fe56399c1a454e5b` 后追加计分、协议身份、冻结评测和检查点发布修复。宿主、重放、合法动作及运行时均为 RiichiEnv；Kanachan 仅是 546 动作和 tile37 编号来源。RiichiEnv 上游参考提交为 `0ee1236a283b15508f3ac28407027f340b81e1de`。

**历史来源：** 外部附件 `KAR-v1.2.2-DESIGN(3).md` 的字节 SHA-256 为 `4349039c44175ece77f6dddad3a2106fbd3e0e0339935122df242175c83cbdba`。它保留历史身份，不作为 V4 实施规范。见[相对旧附件的差异与迁移说明](kar-design-v4-migration.md)。

**证据范围：** [design-sync-1 校验](kar-design-v4-validation.md)、[分批 Pass／有效未支持质量报告](kar-pass-rho-math-repair.md)、[振听／逐事件校准报告](kar-furiten-event-calibration-repair.md)和[响应监督、冻结 FP32 与验证统计报告](kar-response-fp32-validation-repair.md)各保留原范围。本轮规则迁移的实现、验证与旧产物边界见[迁移记录](kar-native-rule-boundary.md)，历史通过数不自动覆盖当前原生构建。设计不授予真实数据处理、参数更新、GPU、校准或 M3 许可。

## 0. 冻结结论与活动版本

保持结构化观察、完整 LEGAL_SET、因子化 ACTION、546/21/21×4 三头、自然分布监督、既定损失、受限 KL 公式及随机采样。只对通过动作支持门控的舍牌重排；集合外各动作概率及集合内总概率质量不变。

| 合同 | 活动身份 | 源定义 |
|---|---|---|
| 完整宿主 Schema（`schema_id`） | `kar-riichienv-v1` | [contracts.py](../src/riichienv/kar/contracts.py) |
| 完整 canonical Schema SHA-256（`schema_hash`） | `fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673` | [schema.py](../src/riichienv/kar/schema.py) |
| 设计来源引用（`reference_design_schema_id`） | `kar-v1.2.2`，不是活动 Schema 的别名 | [contracts.py](../src/riichienv/kar/contracts.py) |
| 历史 overlay 字节 SHA-256（`reference_overlay_sha256`） | `90286d088014eb72e9f99f1b3c35d0a7553bcf5fec99e8aeb3ba5f02a010716a`，不是完整 Schema hash | [contracts.py](../src/riichienv/kar/contracts.py) |
| Schema 映射 | `riichienv_complete_v1_from_kar_v1_2_2_v1`，legacy compatibility=`not_claimed` | [versioning.py](../src/riichienv/kar/versioning.py) |
| architecture | `kar_kl_response_value_v1` | [versioning.py](../src/riichienv/kar/versioning.py) |
| artifact | `kar_riichienv_contract_v4` | [versioning.py](../src/riichienv/kar/versioning.py) |
| ruleset | `mahjong_soul_4p_red_half_v3_native_rules` | [contracts.py](../src/riichienv/kar/contracts.py) |
| native rules | `riichienv_native_rules_v4_nagashi_mask`，产物显式绑定 `native_rules_version` | [core/lib.rs](../riichienv-core/src/lib.rs)、[versioning.py](../src/riichienv/kar/versioning.py) |
| protocol identity | `protobuf_source_and_descriptor_v1`，同时绑定主 proto 源字节与完整描述符 | [mjsoul.py](../src/riichienv/kar/mjsoul.py) |
| input pipeline | `semantic_public_clock_frozen_decisions_v7_native_rules` | [versioning.py](../src/riichienv/kar/versioning.py) |
| native replay | `native_replay_context_v4` | [native_replay.py](../src/riichienv/kar/native_replay.py) |
| settlement | `native_mjsoul_settlement_v2` | [settlement.py](../src/riichienv/kar/settlement.py) |
| dataset / action labels | `kar_decision_sqlite_v1` / `explicit_and_unique_response_pass_v3` | [learning/export.py](../src/riichienv/kar/learning/export.py) |
| response opportunity evidence | `frozen_response_opportunities_v1` | [response_supervision.py](../src/riichienv/kar/response_supervision.py) |
| frozen model precision recipe | `fp32_ieee_single_prefix_v3`，绑定实际 IEEE FP32 后端合同 `frozen_fp32_backend_v1` | [learning/config.py](../src/riichienv/kar/learning/config.py) |
| public clock | `public_semantic_clock_v1` | [event_semantics.py](../src/riichienv/kar/event_semantics.py) |
| terminal public resolution | `verified_abort_public_resolution_v1` | [event_semantics.py](../src/riichienv/kar/event_semantics.py) |
| model computation identity | `kar_transformer_v3_semantic_inputs`，不改变参数形状 | [learning/config.py](../src/riichienv/kar/learning/config.py) |
| feature semantics / stochastic mask | `visible_rule_features_v2_ryanmen_pairs` / `sample_mask_v1` | [versioning.py](../src/riichienv/kar/versioning.py) |
| response mask | `public_rule_response_mask_v5_certain_nagashi`，保留四组面子容量约束，并在完整公开证据确定末巡流局满贯时排除普通荒牌 | [contracts.py](../src/riichienv/kar/contracts.py) |
| calibration quality / branch quality | `preregistered_calibration_quality_v3_event_ovr` / `preregistered_response_branch_quality_v2_ovr` | [calibration_quality.py](../src/riichienv/kar/calibration_quality.py) |
| event support | `train_independent_games_v2_branch_quality`（train 计数定义不变） | [runtime_contracts.py](../src/riichienv/kar/runtime_contracts.py) |
| calibration | `heldout_temperatures_log_gate_v6_event_ovr` | [runtime_contracts.py](../src/riichienv/kar/runtime_contracts.py) |
| diagnostics report / merge state | `whole_game_cluster_statistics_v4_event_ovr` | [statistics.py](../src/riichienv/kar/statistics.py) |
| live projection | `native_rule_event_projection_v6` | [live_events.py](../src/riichienv/kar/live_events.py) |
| runtime / finite policy identity | `restricted_kl_runtime_v3_event_ovr` / `calibrated_policy_identity_v3_event_ovr`，KL 数学不变 | [runtime_contracts.py](../src/riichienv/kar/runtime_contracts.py) |
| training configuration | `explicit_train_split_scale_v5_decision_time`，配方绑定当前输入版本 | [versioning.py](../src/riichienv/kar/versioning.py) |
| training startup / cost ledger | `kar_training_startup_v1` / `durable_compute_cost_ledger_v3` | [learning/production_checkpoint.py](../src/riichienv/kar/learning/production_checkpoint.py) / [budget.py](../src/riichienv/kar/budget.py) |
| frozen evaluation startup / calibration job | `frozen_evaluation_startup_v1` / `heldout_logits_calibration_job_v3_single_prefix` | [learning/evaluation_startup.py](../src/riichienv/kar/learning/evaluation_startup.py) / [learning/calibration_job.py](../src/riichienv/kar/learning/calibration_job.py) |
| bounded diagnostic / production checkpoint | `kar_exact_resume_v2` / `kar_production_exact_resume_v1`，共用不可变代际与原子指针发布 | [learning/checkpoint.py](../src/riichienv/kar/learning/checkpoint.py) / [learning/production_checkpoint.py](../src/riichienv/kar/learning/production_checkpoint.py) |
| validation tau selection / gain bound | `frozen_validation_tau_grid_v4` / `paired_rank_hoeffding_union_v1` | [selection_quality.py](../src/riichienv/kar/selection_quality.py) |
| tau preregistration / evaluation plan | `preregistered_tau_selection_v2` / `registered_validation_tau_matches_v4` | [selection_quality.py](../src/riichienv/kar/selection_quality.py) / [learning/validation_policy.py](../src/riichienv/kar/learning/validation_policy.py) |
| training plan / loss / decision bound | `budget_ladder_v1` / `sentinel_safe_v1_2_1` / `one_sided_v1_2_1` | [versioning.py](../src/riichienv/kar/versioning.py) / [runtime_contracts.py](../src/riichienv/kar/runtime_contracts.py) |
| model tier basis | `N_train_half_games` | [versioning.py](../src/riichienv/kar/versioning.py) |

完整宿主 kind、字段、139 字节 little-endian Token、944 Embedding 行及 62 状态槽已物化。`schema_hash` 对活动 [Schema JSON](../src/riichienv/kar/token_schema.riichienv.v1.json) 的规范化内容计算；`reference_overlay_sha256` 对历史 overlay 的原始文件字节计算，两者不可互换。不要求合并缺失的 Kanachan 父注册表，不把设计引用改写为活动 ID。

本轮 Rust/Python 职责收敛不改 546 动作 manifest、21 类响应、Token 布局或模型参数形状。KAR 雀魂规则与输入身份仍须升级：本家四枚单骑形式听牌及西入普通荒牌终止的规则选择可能改变合法性、公开揭牌、完成证书和最终名次监督。天凤在这两处允许不同结果，原生内核必须按 `GameRule` 区分；同 shape 不构成旧 KAR 产物兼容证明。

### 0.1 不声称什么

MLE 不保证有限网络校准；观察条件关联不自动等于干预价值；500 个 train 游戏不是逐状态误差证书；KL 不消除所有 OOD；RON 后最终名次并非低方差；粗响应不是完整下一状态；任何参数规模、预算或 Epoch 数都不保证棋力提升。

## 1. 目标、定义域与效用

给定观察者在 cutoff 当时可见的历史序列 $\mathcal H$、状态 $O$ 与完整合法集合 $\mathcal L$，模型估计行为先验、候选舍牌的即时响应分布，以及给定响应的终局名次分布。本文用 `H` 表示历史 Token 行数（不含 BOS），用 `C` 表示公共语义时钟；合法动作集合只记作 $\mathcal L$。

- 策略覆盖完整全局动作 0..545。
- 响应和条件名次仅对舍牌 0..147 有定义。
- 非舍牌仍训练动作头；不得为其伪造 NEXT_DRAW 或舍牌价值。
- 最终效用固定 `U=2.5-rank`，rank 为唯一的 1、2、3、4 位，故 `U∈{1.5,0.5,-0.5,-1.5}`。
- 小局净点数 `G_round` 仅作诊断标签，不增加第四个头、不复制成逐步奖励。
- 更改平台段位分、点数效用、平分名次或 uma 机制必须升级规则/效用合同。

## 2. 随机变量与因果边界

$X=\operatorname{tokenize}(\mathcal H,O,\mathcal L)$；A 是实际完整动作；E 是该舍牌的最终即时响应；R 是半庄终局名次。真实响应和终局名次不进入 X。

固定来源/cohort 的完整行为策略为 $\mu(A\mid X)$。若观察到完整的行为样本，给定 X 的总体动作交叉熵最优解是 $\mu$；实际公开牌谱有响应动作删失。令 $I=1$ 表示此机会有显式或唯一可证的动作而被保留，则现行普通 CE 对这些行学习的是 $P(A\mid X,I=1)$，一般不等于 $\mu$。补齐唯一可证 Pass 减少删失，不证明覆盖全部意图，仍有依赖 A 的保留机制；校准、任意重加权或重定义 $\mu$ 都不能辨认缺失的私有响应。本修订不引入删失或集合标签损失。对完整行为的误差仍定义 `δπ(x)=TV(π0(·|x),μ(·|x))`，不能拿保留子总体的误差替代它。

对舍牌 a，定义 `pμ(e|x,a)=P(E(a)=e|X=x)`、`mμ(x,a,e)=E[U^{a,μ}|X=x,E(a)=e]`，因此：

$$Q^\mu(x,a)=\sum_e p^\mu(e\mid x,a)m^\mu(x,a,e).$$

干预解释要求：实际动作与日志一致、给定完整可见 X 后条件可交换、目标动作有覆盖、后续本家策略和对手机制固定、标签完整且可追溯。混合未知技能/风格/规则会破坏这些假设；不能满足时只作观察分布相关性解释。

## 3. 三头概率模型

$$\pi_\theta(A\mid X)=\operatorname{Softmax}_{\mathcal L(X)}f_A(h_{DECIDE}).$$

$$p_\theta^{pub}(E\mid X,A)=\operatorname{Softmax}(\operatorname{mask}_{M^{pub}}f_E(h_{ACTION})).$$

$$v_\theta(R\mid X,A,e)=\operatorname{Softmax}_{r=1..4}f_V(h_{ACTION})[e,r].$$

名义分支效用 `m̂=Σr vθ(r|x,a,e)(2.5-r)`；名义候选价值 `Q̂=Σe p̂(e)m̂(e)`。实际重排使用第 10、11 节的保守评分，不以名义诊断值绕过分支支持。

## 4. 输入协议与可见性

观察者的手牌、河/历史、副露、立直、分数、阶段和杠事务由 Rust `ObserverState` 从规范事件归约；它不读取 `GameState`、隐藏牌山或对手真实合法动作，只保留所选观察者的私有牌。Python `ObserverLedger` 管理模型历史、公共语义时钟、cutoff、来源及本家 Pass 确认，读取分离的原生事实投影。终局揭示的来源真实性由在线原生裁定或离线审计先行验证，不能由投影器自行补全。

### 4.1 行布局与 Tensor

前缀为 `[BOS][历史 H 行][当前状态 S 62 槽][DECIDE]`，共 H+64 行；追加一个真实或假设 ACTION 后为 H+65 行。`H` 只决定行数、数组索引和长度，不决定 RoPE 位置；位置用公共语义时钟 `C`，见 §6.1。

| 字段 | 形状/取值 |
|---|---|
| kind | `[B,L]`，48 类，0=PAD |
| actor/source | 各 `[B,L]`，5 类相对座位，0=N/A、1=本家 |
| tiles | `[B,L,5]`，每列 39 类，各有独立 Embedding |
| stage/wind/slot | 各 `[B,L]`，16/5/17 类 |
| flags | `[B,L,16]`，0=Unknown/N/A、1=False、2=True |
| numeric/numeric_present | 各 `[B,L,20]`，FP32/bool |
| action_ids | `[B,L]`，0..545，546=N/A |
| action_type_ids | `[B,L]`，0..12 |
| action_tile37 | `[B,L]`，0=N/A，tile37+1，38=HIDDEN |
| action_tsumogiri/action_riichi | 各 `[B,L]`，0=N/A、1=False、2=True |
| block/positions/valid | 各 `[B,L]`，注意力块、语义位置、有效位 |
| candidates | `[B,32]`，完整动作，546=padding |

未知 numeric 先置零，再清 presence；真实零必须 present=true。不静默裁剪。字段含义由 `(kind,field,index,value)` 决定，stage/flags 的跨 kind 同值可共享表，不能用全局 enum 覆盖其他 kind 的语义。

### 4.2 Kind 注册表

| ID | Kind | ID | Kind |
|---:|---|---:|---|
|0|PAD|1|BOS|
|2|ROUND_STATE|3|PLAYER_STATE|
|4|HAND_COUNT|5|MELD_STATE|
|6|DORA_STATE|7|DECISION_CONTEXT|
|8|LEGAL_SET|9|DECIDE|
|10|ACTION|11|ROUND_START|
|12|DRAW_SELF|13|DRAW_OTHER|
|14|DISCARD|15|CHI_ACCEPT|
|16|PON_ACCEPT|17|DAIMINKAN_ACCEPT|
|18|ANKAN_DECLARE|19|ANKAN_ACCEPT|
|20|KAKAN_DECLARE|21|KAKAN_ACCEPT|
|22|RIICHI_DECLARE|23|RIICHI_ACCEPT|
|24|DORA_REVEAL|25|SCORE_DELTA|
|26|SELF_PASS|27|WIN_RESOLVED|
|28|ROUND_END|29|PUBLIC_HAND_REVEAL|

30..47 保留，不写入。完整合法编码以仓库 [Schema JSON](../src/riichienv/kar/token_schema.riichienv.v1.json) 为准；不得由部分 overlay 猜测其他 kind。MELD_STATE stage=8/9/10/11/12 对应 CHI/PON/DAIMINKAN/ANKAN/KAKAN。

### 4.3 当前状态 62 槽

ROUND_STATE 1、PLAYER_STATE 4、HAND_COUNT 34、MELD_STATE 16、DORA_STATE 5、DECISION_CONTEXT 1、LEGAL_SET 1。未用 meld/dora 槽为 PAD。自己的 34 基本牌计数包含摸牌恰一次，当前摸牌身份另放在 DECISION_CONTEXT。

### 4.4 牌种、数值与排名

冻结 tile37：0=赤5m，1..9=普通1m..9m；10=赤5p，11..19=普通1p..9p；20=赤5s，21..29=普通1s..9s；30..36=东南西北白发中。模型牌字段用 tile37+1，0=N/A、38=HIDDEN；形状按 base34 合并红五，但赤牌数独立。

| numeric 索引 | 语义 | 除数 |
|---:|---|---:|
|0|当前原始点数|50000|
|1|已发生点数变化|10000|
|2|与本家点数差|50000|
|3|本场|10|
|4|供托根数|10|
|5|公开剩余活牌山|70|
|6|风局编号|12|
|7|起始座位顺序|3|
|8|本家基本牌暗手数，含摸牌|4|
|9|本家该基本牌赤牌数|1|
|10|本家已知该基本牌总数|4|
|11|已知该基本牌赤牌数|1|
|12|当前暗手总张数|14|
|13|已成立 meld 数|4|
|14|本局舍牌次数|24|
|15|立直声明的本家舍牌序号|24|
|16|完整合法动作数|32|
|17|本局摸牌次数|70|
|18|紧邻上一名分数减本家分数|50000|
|19|本家分数减紧邻下一名分数|50000|

18/19 只在 ROUND_STATE 填充。一位没有上一名、四位没有下一名，缺失字段 presence=false；相邻同分为真实零且 present=true。

整场固定 `initial_seat_order_player_ids`。排序键为 `(-score,initial_seat_index)`，起家东优先于南、西、北，当前 oya 不参与破平；ROUND_STATE.stage=1..4 为本家当前名次。当前点数不预分未归属立直棒、不加终局 uma/oka。四家 25000 时为 1/2/3/4，所有存在的相邻差为零。不同平分规则必须另建 ruleset/utility。

### 4.5 ACTION 与完整 LEGAL_SET

舍牌 `a=4*tile37+2*tsumogiri+riichi`。ACTION 写完整 ID、族、舍牌牌种/摸切/立直；非舍牌的后三因子必须 N/A。

族：0=N/A、1=DISCARD、2=ANKAN、3=KAKAN、4=TSUMO、5=KYUSHU、6=PASS、7=CHI、8=PON、9=DAIMINKAN、10=RON，11..12 保留。

$$e(a)=E_{id}[a]+E_{type}[t]+E_{tile}[p]+E_{tsumogiri}[m]+E_{riichi}[r].$$

LEGAL_SET（kind 8）在既有字段向量之上加入 $|\mathcal L|^{-1}\sum_{a\in\mathcal L}e(a)$，复用 ACTION 五张表，无新参数。完整候选按全局 ID 去重、排序，非空且最多 32 项；过滤 padding=546，不合并摸切/手切/立直变体。保留 $\mathrm{numeric}[16]=|\mathcal L|/32$。dense 和 `encode_prefix` 共用输入实现，单 ACTION 后缀不重复池化集合；缓存精确绑定候选集合。

### 4.6 确定性可见特征与三态

HAND_COUNT.flags[0..2] 是对 P1/P2/P3 的 Genbutsu，3..5 是 full-suji exclusion，6 是 no-visible-possible-ryanmen，7 是公开 Dora 类型，8..15=N/A。字牌 flags[3..6]=N/A。

- Genbutsu 仅来自目标玩家自己的公开舍牌历史，被叫走的舍牌仍保留；暗杠/加杠不算舍牌，不推测临时振听。
- Full-suji 必须覆盖所有另一侧等待牌；4/5/6 只满足一侧为 False。它不排除单骑/双碰/边张/嵌张/七对子。
- [`PublicSnapshot.known_counts()`](../src/riichienv/kar/observation.py) 是决策特征的当前物理区计数，只计自己的暗手（含已进入暗手的当前摸牌）、当前河、已成立副露、公开指示牌四类，并按 base34 合并赤五和普通五。河中叫走牌只在副露区计一次；加杠提交后才新增第四张。该方法要求 `physical_ledger_complete` 与 `visibility_complete` 同时为真，否则返回 `None`（Unknown），不依赖 `history_complete`。舍牌历史相关特征仍独立依赖历史完整性。
- **终局公开不扩张计数 API。** 赢家或宣告者的终局公开手牌进入 `PUBLIC_HAND_REVEAL` 历史 Token，不作为第五类区传给计数。底层 `known_count_base34()` 可收显式 `other_public_reveals`，但快照方法没有传入它。[`encode_prefix()`](../src/riichienv/kar/tokenize.py) 只允许 `own_act`、`discard_response`、`kan_response`，`closed` 阶段没有决策前缀。即使某个终局快照能返回四区计数，也不是全公开牌查询。旧附件 §4.8.4 的任意 cutoff 广义计数范围不属于此 V4 API。
- Dora 只由 cutoff 前明确公开的指示牌计算，不使用未公开里宝牌。

冻结 Ryanmen 查表：

| 目标 | 两张搭子 P(v) | 另一等待 Opp(v) |
|---:|---|---|
|1|(2,3)|4|
|2|(3,4)|5|
|3|(4,5)|6|
|4|(2,3),(5,6)|1,7|
|5|(3,4),(6,7)|2,8|
|6|(4,5),(7,8)|3,9|
|7|(5,6)|4|
|8|(6,7)|5|
|9|(7,8)|6|

不跨花色、不循环。目标3的(1,2)和目标7的(8,9)是边张，不列入 Ryanmen。仅当每个 P(v) 至少一种搭子牌四张可见时，将 flags[6] 置 True；目标牌本身四张可见不构成充分条件，因为候选舍牌可以来自本家可见手牌，仍能完成对手的两面搭子。Full-suji 为 `Opp(v)⊆目标玩家舍牌历史`。这些标志不是绝对安全证书。旧特征与输入版本的派生审计、导出和缓存必须重新生成，不能只改版本字段复用。

## 5. 完整动作、响应与监督边界

### 5.1 动作表

| ID | 动作 |
|---|---|
|0..147|完整舍牌（含摸切与立直变体）|
|148..181|ANKAN|
|182..218|KAKAN|
|219|TSUMO|
|220|KYUSHU_KYUHAI|
|221|PASS|
|222..311|CHI|
|312..431|PON|
|432..542|DAIMINKAN|
|543..545|RON 来源|

不把原生紧凑 action index 或候选下标当完整 ID。普通5p为60..63，赤5p为40..43。动作 manifest 的内容 hash、原生完整 Action 映射与 round-trip 测试共同冻结身份。

### 5.2 响应表

| ID | 类别 |
|---:|---|
|0|NEXT_DRAW|
|1|CHI_P1|
|2..4|PON_P1/P2/P3|
|5..7|DAIMINKAN_P1/P2/P3|
|8..14|RON_MASK_1..7，ID=7+mask|
|15|EXHAUSTIVE_DRAW|
|16|ABORT_FOUR_WINDS|
|17|ABORT_FOUR_RIICHI|
|18|ABORT_FOUR_KANS|
|19|ABORT_TRIPLE_RON|
|20|NAGASHI_MANGAN|

P0 是舍牌者，P1/P2/P3 为下/对/上家，RON bit=1/2/4。响应是最终裁定，不是未公开意图。窗口依次由实际荣和集合、成立鸣牌、正常下家摸牌或终局关闭；缺失/冲突是 invalid，不当作 NEXT_DRAW。大明杠后的岭上自摸不把原舍牌响应改成 RON；暗/加杠抢杠窗口不属于 21 类舍牌响应。

### 5.3 观察者私有 SELF_PASS

在线只在原子提交成功后写 `native_committed_pass`。离线须冻结该观察者的非平凡完整候选（Pass 不是唯一合法项）与触发身份，读到真实公开裁定后，才可能给出唯一标签：

| 公开裁定与排除条件 | 离线 provenance |
|---|---|
| 普通舍牌后确实由下家正常摸牌，任何非 Pass 会阻止该摸牌 | `inferred_unique_next_draw_pass` |
| 暗／加杠原生预览只提供 Pass/Ron，宣告者实际完成岭上摸牌，排除 Ron | `inferred_unique_kan_replacement_pass` |
| 成立鸣牌的优先级严格低于该观察者的**全部**其他候选，例如 Chi 成立排除另一家的 Pon | `inferred_unique_call_priority_pass` |
| 其他候选只有 Ron，经已证无荣和的终局，或雀魂不抢截的完整多荣和赢家组中不含该观察者 | `inferred_unique_terminal_no_ron_pass` |

若较低或同优先级的私有动作仍可能得到相同公开结果（例如高优先级 Pon 覆盖了另一家的 Chi），该机会**删失**，不补成行动行或私有 marker。未观测到的意图、缺失关闭或不适用裁定同样不猜；不读取 `GameAction.user_input`。捕获触发身份本身不是提交证据。每局及导出 split 分别统计非平凡机会总数、显式、唯一推断及删失，按响应阶段与关闭结果交叉分组；这些计数不作为训练权重。

marker 锚定公共关闭事件之后、复用该事件位置，不增加公共 event ID 或时钟，不改其他观察者。未被抢的杠用实际公开 replacement draw 关闭私有窗口，transport ACK 和 Dora 均不可代替它；已成立抢杠由公开和牌裁定关闭。普通舍牌以实际关闭事件为锚。旧 snapshot/prefix 冻结不回写。source/provenance 是审计元数据，不作为 E/R 或未来信息输入。

成功提交的 receipt 可以随第一段事件或单独在空回调到达。投影按该 receipt 所属 ledger 的合法关闭锚逐条消费：已知且仍开放则保留，已写入或此前已幂等处理才移除。未知/错误 owner、触发、来源以及已经失去合法关闭锚的窗口必须报合同错误，不当作等待；一条记录失败不能清空其他有效待办。整批与合法分批在相同最终 cutoff 的历史、positions 和完整前缀应逐字节一致。

### 5.4 决策冻结、同巡振听与导出

回放在摸牌或吃碰取得动作权、或响应窗口真正打开时冻结本家观察、完整动作候选和历史。随后才公开的延迟 Dora、`reach_accepted` 等仍按真实顺序入账，只影响后续决策；监督来源事件不反写当前前缀。抢杠使用声明前原生预览；加杠窗口打开前已公开的上一个杠指示牌可见，本次窗口之后的指示牌不可见。

四人原生在线和 MJAI `dahai` 共用开窗规则。和牌形无役且当前没有荣和时，也可置该玩家的同巡振听；此时没有可谢绝的 RON，不能仅凭这个标志导出 Pass。同巡标志只在该玩家自己的舍牌（含鸣牌后的舍牌）或新局清除，不因他家鸣牌、摸牌或舍牌清除。实际有 RON 的窗若由正常摸牌或成立鸣牌关闭，原生重放把未和者记为同巡振听；已立直的未和者保留立直振听。抢杠未和只在原生预览的真 RON 窗提交时登记，桥接层不缓存或回写整份振听状态。有合法终局荣和时不按一次普通见逃关闭。

这些规则改变的是本家原生合法候选与可证行为，不泄露到他家公开 response mask。旧回放可能含多出的 RON、错误的 `inferred_unique_next_draw_pass` 训练行及污染后续前缀的 `SELF_PASS`；未检测出独立结算差异不能证明候选正确。

## 6. 规范化公共事件时钟与 Attention

### 6.1 语义边界

`event_id/cutoff_event_id` 是规范化公共事件流下标，复合 events 另可含本家 marker，二者不能混用。`source_record_index`、source hash、native callback 批次和文件分块仅作来源信息，**不得决定 RoPE positions**。

规范时钟从0开始，只看已观察到的事件：

1. start_game、start_kyoku、draw、reach、discard、reach_accepted、dora、成立鸣牌、杠声明、终止及 end_kyoku/end_game 等各推进一个语义 tick。一次事件展开多条 Token 全部共享位置。
2. 开局 start_kyoku 与庄家首次 draw 是两个语义事件，无论源 RecordNewRound 将其打包在一起还是分开发送。标准开局 start_game=1、start_kyoku=2、first draw=3，S/DECIDE=4。
3. reach 与其 discard 分别计时；reach_accepted 与之后的 draw 分别计时。原始 record 是否合并它们不改变时间。
4. 同一触发牌的连续多荣和是一个公开同时裁定，共享时钟，并按相对放铳者P1/P2/P3规范顺序展开；原始赢家列表顺序不改变输入。公共总支付只编码一次。在线跨批次暂存不完整win group，见到实际边界才发布完整裁定；未完整时拒绝决策snapshot，不利用尚未到达事件补写旧输入。
5. kan_commit 是可选transport确认，不占模型时钟，不单独生成历史接受Token；它仍可立即证明当前账本的杠已提交。为避免有无ACK及旧杠延迟Dora改变历史，历史接受行统一锚定共同可见的本次有效Dora或replacement draw证据；旧杠延迟Dora不能证明新杠提交。模型真实决策前两条路径的状态/历史/位置一致，不借未来证据回写已封存前缀。
6. SELF_PASS 不推进时钟，使用已观察关闭事件的锚。不同来源若本家私有事实不同，不强行要求相同历史，也不伪造离线无法知道的 Pass。
7. 同一公共cutoff有多个已证非平凡Pass监督样本时，按稳定初始座位顺序输出，不遍历原生hash-map顺序；每个观察者的输入不变，样本ID/聚合哈希/串行与两进程重放也必须确定一致。

`C` 是当前 cutoff 的最后公共语义位置，不是事件流下标，也不是历史 Token 行数。位置为 **BOS=0、所有状态槽与 DECIDE=C+1、ACTION=C+2**。历史中一个事件可展开多行、多个事件可共用一个 tick，本家私有 marker 也可增加 H 而不增加 C。见 [`TokenHistoryCache`、`encode_prefix()` 和 `append_action()`](../src/riichienv/kar/tokenize.py)。

`PublicSnapshot.public_clock_version` 必须显式为当前版本；缺失/旧时钟拒绝 Tokenize 和 runtime。旧历史/模型 K/V 必须重建，不能补一个常数、丢掉 positions、放宽误差或仅改版本标签。

### 6.2 块注意力

| Query / Key | 历史块（BOS 与历史） | S | D | A |
|---|---|---|---|---|
|历史块|因果|禁止|禁止|禁止|
|S|允许|双向|禁止|禁止|
|D|允许|允许|自身|禁止|
|A|允许|允许|允许|自身|

所有层相同；RMSNorm/FFN 逐 Token，不跨样本归一化。D 在任何层不能读 A。Padding 不作 Key，无效 Query 只读 BOS 后清零；SDPA bool True=允许。

同一决策候选只共享前缀 K/V，分支互不可见，不串接多个 ACTION。缓存绑定 owner、观察者/决策、参数与 buffer 版本、设备/dtype、完整候选、input/model 合同；不能跨决策或更新后复用。

## 7. Transformer 架构与规模口径

pre-norm decoder-only：字段 Embedding 求和与 numeric 投影；L 层 RMSNorm→无 bias QKV→RoPE→block SDPA→无 bias output+residual→RMSNorm→SwiGLU+residual；final RMSNorm。零 dropout，无 BatchNorm 或跨样本混合。

五个 tiles 列分别为独立 `Embedding(39,d)`，共195行。numeric 与 presence 转 FP32 后拼接40维，经 `Linear(40,d,bias=False)`。其他独立字段行数：kind48、actor5、source5、stage16、wind5、slot17、flags16×3、action_id547、type13、action_tile39、tsumogiri3、riichi3；总944行。

Policy=`Linear(d,546)`，Response=`Linear(d,21)`，Value=`Linear(d,84)` reshape21×4，均含 bias。dense、encode_prefix 的 Policy 以及 score_actions 的 Response/Rank **在线性投影本身禁用 autocast**，FP32 输入/计算/输出；不得仅在 BF16 投影后 `.float()`。BF16 主干不要求与完全 FP32 主干逐位相同。

$$P=L(4d^2+3df+2d)+1636d+651.$$

| preset | L | d | heads | f | 参数量 |
|---|---:|---:|---:|---:|---:|
|Tiny|4|192|3|512|2,085,771|
|Small|6|256|4|704|5,239,435|
|Medium|8|384|6|1024|14,790,795|
|Large|8|512|8|1408|26,536,587|

### 7.1 统一半庄统计

[`game_count_contract()`](../src/riichienv/kar/versioning.py) 按完整半庄 `game_id` 计数，区分审计池、整场切分和实际导出：

| 字段 | 含义 |
|---|---|
| `N_eligible_half_games` | 过滤后、切分前的完整合格池总数，等于 `available_split_counts` 各 split 之和。 |
| `available_split_counts` | 对整个合格池按 game_id 切分后，train/validation/test 的各自可用数。 |
| `exported_split_counts` | 当前产物实际导出的 train/validation/test 数；任一 split 不超过对应可用数。 |
| `N_train_half_games` | `exported_split_counts["train"]`，不是合格池、可用 train 或决策行总数。 |

**模型规模只按 `N_train_half_games` 选型**，门槛来自 [`model_tier()`](../src/riichienv/kar/contracts.py)。旧附件 §7.5、§14.2 与 §17 的口径冲突以此消解。

| N_train 完整半庄 | 数据允许最大主力 preset | effective batch |
|---:|---|---:|
|<30,000|Small|128|
|30,000..49,999|Medium|256|
|≥50,000|Large reference|512|

M1 固定 Small。正式 M2 必须覆盖全部 available train/validation，probe 子集不冒充全量。规模门槛不是学习理论充分条件；规则混杂、数据/支持不足不能靠扩模型解决。实测预算超限按 Large→Medium→Small 降档或停止，不缩 M3 样本伪装预算。

## 8. 数据与监督合同

主 cohort 为雀魂四人赤牌半庄，`category=2/mode=2/mode_id=12`；规则、技能/段位来源同质。Tenhou 或其他规则分开数据集。训练/验证/测试按整场 game_id 的固定 hash/salt 划分，四观察者视角同 split；真实 Golden 整场排除在训练、验证、测试及支持统计外。

主数据必须完整半庄、来源事件完整、所有采纳动作原生合法且可信、舍牌窗口裁定完整、独立结算和最终唯一名次可追溯。不完整数据仅作独立诊断，不混为同一价值分布。

样本保存 game/round/observer/decision/cutoff、初始座位、完整候选与实际全局动作、action provenance、响应适用/观察位和真实响应、最终名次有效位、小局点数诊断、规则/输入/Schema/mask/模型/产物版本与来源链。input 和 labels 分开构造，targets 的扰动不能改变 forward。

`action_valid`、`response_applicable`、`response_observed`、`value_valid` 分开。`value_valid⇒response_valid`；非舍牌不造响应，缺标签不填正常事件。有效 target 若被当前公开 mask 排除，定位规则/数据错误，不按标签 unmask。

### 8.1 终局公开信息

- Hora 仅在公开裁定后揭示赢家手牌；里宝只在规则实际公开时写。不能从全知初始手牌提前传播给其他观察者。
- 荒牌/流局满贯按冻结原生 tenpai 宣告机制揭示实际应公开手牌；不从支付猜听牌。
- 九种九牌只揭示宣布者。RecordLiuJu 使用 type、seat、tiles；不用 allplayertiles 机械补其他手牌。核实当前 draw actor、首次 draw、未舍牌、未鸣牌、14张、至少9种幺九字与牌张一致；源 tiles 存在必须匹配，否则拒绝。源 tiles 缺失时，只有已验证历史与冻结公开规则唯一重建才允许，保留明确 provenance。
- 揭示在终止时点进入 PUBLIC_HAND_REVEAL，所有视角获得同一公共事实；揭示前只有本家知道自己的手牌，旧快照不回写。通用终局若缺应公开且不可证明的信息，置visibility_complete=false并记录gap，后续前缀不可用。当前九种规范化入口要求完整前序牌账本；宣布者/牌张矛盾或重建前提不足直接RecordError，不输出伪装完整的终止事件，不填假手牌。
- 已验证的途中流局由冻结规则证明本次转移为零，记录零 deltas 和该时点已知 scores，四家均生成 present=true 的 SCORE_DELTA。它与缺少 NoTile/Hule 结算不同；后者保持缺失，不从下一局分数反推、不默认填零。已有立直棒扣分不撤回，也不被终止零转移覆盖。
- 终局影响必须一直对拍到下一局真实决策的完整历史、顺序、位置与状态，不能只比较 reason 字符串。

## 9. 损失与训练数学

[`Labels`、`loss_sums()`、`LossSums.objective()`](../src/riichienv/kar/learning/losses.py) 使用固定训练 manifest 比例 `ρE=P_D(response_valid)`、`ρV=P_D(value_valid)`，要求 $0<\rho_V\le\rho_E\le1$，不按 microbatch 或 validation 重算。令 $B$ 为决策样本总数（不是各头有效标签数）；$I_i^A$、$I_i^E$、$I_i^V$ 分别表示 `action_valid`、`response_valid`、`value_valid`，其中 `response_valid = action_valid & response_applicable & response_observed`。以下只对对应有效行求和：

$$L_A=-\frac1B\sum_{i:I_i^A=1}\log\pi_\theta(A_i\mid X_i,\mathcal L_i).$$
$$L_E=-\frac1{B\rho_E}\sum_{i:I_i^E=1}\log p_\theta^{pub}(E_i\mid X_i,A_i).$$
$$L_V=-\frac1{B\rho_V}\sum_{i:I_i^V=1}\log v_\theta(R_i\mid X_i,A_i,E_i).$$
$$L=L_A+0.25L_E+0.1L_V.$$

Response train temperature=1；校准后推理使用 TE，但 allowed mask 同一函数。Value 只取实际发生 E 的 rank CE，真实 rank1..4 在数据层明确转0..3。保持自然分布，不做类重加权、Focal、RWR/Profile S、CQL/TD 或额外 BC 阶段。

有效标签先严格验范围、合法动作和 allowed 位，再为无效行分别生成 safe_e、safe_e_v、safe_r；不对有效越界标签 clamp/取模/关 valid。无效行在 CE 前用有限 dummy logits 和全允许 dummy response mask，避免 NaN×0；gather 前不可接收 -1/255 等原始哨兵。全无有效辅助标签返回连图零，不伪称完美 NLL。

一次 update 的全局样本数 N、DDP world size W：各 rank local sums 乘 W/N、W/(NρE)、W/(NρV)，DDP 平均梯度后等于全局定义。不能各自除本地有效数或 microbatch 数。条件 NLL 用 sum/valid_count 单独报告。

## 10. 冻结概率、价值与受限 KL 数学

### 10.1 名义价值：需要全允许事件的误差界

链式法则 `P(A,E,R|X)=P(A|X)P(E|X,A)P(R|X,A,E)` 与全期望成立，不证明有限模型最优或因果识别。

固定 $(x,a)$，令 $A_{\mathrm{pub}}=\{e:M^{pub}_e=1\}$，$\delta_E=\mathrm{TV}(\hat p,p^\mu)$。公开规则不可能的事件在两个分布中都为零；真实有效标签被 mask 会证伪这个假设。对 $p^\mu_e=0$ 的允许事件，可取一个固定的区间内条件效用扩展；这不改 $Q^\mu$，但若 $\hat p_e>0$，它仍属于名义误差控制的范围。

第 3 节的名义 $\widehat Q$ 对所有允许事件求和。若所有这些事件的条件价值误差都由 $\varepsilon_{m,\mathrm{all}}$ 控制（特别是每个 $\hat p_e>0$ 的事件），则

$$\widehat Q-Q^\mu=\sum_{e\in A_{\mathrm{pub}}}\hat p_e(\hat m_e-m^\mu_e)+\sum_{e\in A_{\mathrm{pub}}}(\hat p_e-p^\mu_e)m^\mu_e,$$

$$|\widehat Q-Q^\mu|\le \varepsilon_{m,\mathrm{all}}+3\delta_E.$$

第二项用效用区间 $[-1.5,1.5]$ 的跨度 3 控制。**只控制支持分支不足以推出这个名义误差界。**

### 10.2 保守价值：仅需支持分支的条件价值误差

令 $E_{sup}$ 为第 11 节的冻结有效价值支持集合；$\varepsilon_{m,\mathrm{sup}}$ 只控制 $A_{\mathrm{pub}}\cap E_{sup}$ 上的条件价值误差。保留原公式（公开不允许的类别因 $\hat p_e=0$ 不作贡献）：

$$\widetilde Q=\sum_{e\in E_{sup}}\hat p_e\hat m_e+\sum_{e\notin E_{sup}}\hat p_e(-1.5),\quad
\rho_a=\sum_{e\notin E_{sup}}\hat p_e,\quad b_a=\varepsilon_{m,\mathrm{sup}}+3\delta_E.$$

拆分支持分支误差、未支持替代（其差值在 $[-3\rho_a,0]$）和响应概率误差，得到：

$$-b_a-3\rho_a\le\widetilde Q-Q^\mu\le b_a.$$

验算：$A_{\mathrm{pub}}=\{0,1\}$、$E_{sup}=\{0\}$、$\hat p=p^\mu=(0.5,0.5)$；事件 0 的真实/预测效用都为 0，事件 1 的真实/预测效用分别为 −1.5/+1.5。则 $\varepsilon_{m,\mathrm{sup}}=\delta_E=0$，但 $Q^\mu=-0.75$、$\widehat Q=+0.75$，名义绝对误差为 1.5（$\varepsilon_{m,\mathrm{all}}=3$）。用 −1.5 替代未支持事件后 $\widetilde Q=-0.75$、$\rho_a=0.5$，保守误差 0 落在 $[-1.5,0]$ 内。见[精确分数回归](../tests/kar/core/test_value_bound_domains.py)。

未支持替代本身不高估，但整个 Q̃ 不是自动具有置信覆盖率的 LCB。`rho_U` 是模型 masked 概率质量，不是 $\delta_E$ 或真实概率误差上界。分支质量 passed、样本数、普通校准指标不能提供或置零任何所需误差界，也不能签发 certified 证书。

500 个独立 train 完整游戏只是数据筛查。原均值 Hoeffding 核算 `ceil(72 ln840)=485` 在固定 iid 有界总体均值下算术成立，但不能推出条件价值模型的统一误差界。决策行相关、游戏聚合可能改变总体。

### 10.3 受限闭式更新

实际重排使用 $\widetilde Q$，不是第 3 节的名义 $\widehat Q$。KL 方向固定为 $D_{\mathrm{KL}}(q\Vert q_0)$；在固定 $S$ 上最大化 $\mathbb E_q[\widetilde Q]-\tau D_{\mathrm{KL}}(q\Vert q_0)$。见 [`restricted_kl()`](../src/riichienv/kar/policy.py)。

先按 $S=\{a\in\mathcal L:0\le a\le147,\ \log\pi_0(a)\ge\ell_{\min}\}$ 选完整舍牌，$M=\pi_0(S)$，$q_0=\pi_0(\cdot\mid S)$。若 $|S|<2$ 或 $M=0$ 直接使用先验；否则：

$$q_\tau=\operatorname{Softmax}_{a\in S}(\log q_0(a)+\widetilde Q_a/\tau),$$
$$\pi_\tau(a)=\begin{cases}M q_\tau(a),&a\in S,\\\pi_0(a),&a\notin S.\end{cases}$$

完整候选不 Top-K、不合并动作变体。分支质量只决定 Esup，不删除响应后重新归一化，不偷偷改变 S 的定义；因此集合外逐动作概率与 M 保持合同仍成立。

### 10.4 E1/E2

令 `d=qτ-q0`、`Δ̂=Σa d_a Q̃_a`。给定当前状态/候选的有效 $b_a=\varepsilon_{m,\mathrm{sup}}+3\delta_E$（第 10.2 节，而非名义界）以及 $\delta_\pi$：

$$E1=M[\tau KL(q_\tau\Vert q_0)-\mathbb E_{q_\tau}b-\mathbb E_{q_0}b-3\mathbb E_{q_0}\rho]-3\delta_\pi.$$

$$E2=M[\widehat\Delta-\sum_a|d_a|b_a-3\sum_a(q_0(a)-q_\tau(a))_+\rho_a]-3\delta_\pi.$$

依据 Q̃−Qμ 的单侧区间，真实增益下界至少为 E1，抵消同一动作误差得到不更松的 E2。不得把 E1 的 E_q0[rho] 换为 E_qτ[rho]。

反例：q0=(.5,.5)、τ=.1、真值(1.5,0)、Q̃=(-1.5,0)、rho=(1,0)、b=δπ=0，qτ≈(3.05902227e-7,.9999996941)。真实变化约−.7499995411；错误 qτ 加权界约+.0693133109；正确 E1≈−1.4306857714，E2 等于真实变化。该反例永久保留。

`certified` 必须有误差界来源、当前候选覆盖、置信/确定性声明且 E2>0。没有真实上界则 certificate_unavailable，不填零。普通 NLL/Brier/ECE、分桶均值、bootstrap、分支质量 passed 都不产生 certified。经验模式不附数学保证；先验回退仅停止本次额外重排，不证明 π0 相对真实 μ 无损。

整场非劣还要求相关可访问历史上局部条件成立、后续机制固定，并满足终止/可积条件；训练分布平均改善不自动满足。

### 10.5 温度与采样

因 Q̃∈[−1.5,1.5]，支持内相对先验倍率在 `[exp(−3/τ),exp(3/τ)]`。预登记 Rmax>1 时必须 τ≥3/ln Rmax；未配置 Rmax 不声称另有上限。τ=∞ 严格先验、零候选辅助前向；τ→0 不作首版默认。

默认从完整最终分布随机采样。argmax 是单独诊断策略，不继承随机策略的证明，不能作为 certified 或验证冻结的随机部署策略冒充发布。

## 11. 校准、分支质量与推理保护

### 11.1 角色与温度

完整 validation 按固定 game hash 分成 cal_fit/cal_check；test 不调参。冻结网络，仅 cal_fit 的固定温度网格选 TA、TE、TR；分别最小化合法动作、masked response、真实 E 分支 rank NLL。cal_check 只评估预登记判据，不能看完后调阈值。

冻结模型合同要求 FP32 参数及浮点缓冲，在校准收集、冻结 prefix/候选及已校准 validation 的完整前向边界关闭 autocast，不靠输出 `.float()` 补精度。推理 recipe 显式绑定 `fp32_ieee_single_prefix_v3` 及实际后端记录 `frozen_fp32_backend_v1`。检查实际生效的矩阵乘法精度，拒绝 CPU MKLDNN BF16/TF32 或 CUDA TF32 等降精度模式；不修改共享宿主的全局设置。后端、SDPA 选择及确定性设置在绑定后发生变化时，拒绝继续复用冻结预测器、缓存或校准。标定计划和发布记录保存同一后端身份，元数据重验使用已保存且通过验证的合同。

动作校准 logits 与部署共用单样本、无 padding、截至 DECIDE 的 `encode_prefix` 路径；收集时剔除教师 ACTION，不能用批量 dense policy logits 拟合严格门控。相同 FP32 dtype 不保证不同矩阵形状逐位相同；阈值附近的舍入差异足以改变 `>=` 支持集合，因此不以部署侧 epsilon 掩盖差异。Response/Rank 的 heldout logits 仍可批量收集，但完整前向同样验证冻结后端；其路径不构成动作门控数值一致性证明。

底层模型缓存检查 autocast 开关和 dtype；冻结 predictor 通过每次前后缀的后端检查，进一步禁止它管理的缓存跨实际数值后端使用。显式 FP32 验证要求 IEEE 后端，即使尚未加载校准；未校准的训练验证仍可用已配置的 BF16，已校准验证拒绝请求 BF16。若以后提供冻结 BF16，应单独绑定 recipe／cache／calibration／release 并在同一路径重做校准。

已校准完整验证的动作门控诊断也使用上述单前缀路径。逐元素一致性限定于相同设备、实现及前缀输入，不承诺跨 CPU/GPU、内核或库版本的逐位一致；跨运行环境发布仍须满足冻结运行身份与校准证据合同。

温度总体 NLL 与平均 forward KL 差一个未知条件熵，Pinsker/Jensen 仅给同一分布上的平均 TV 关系；有限验证 NLL 不是逐状态证书，不能忽略条件熵把原始 NLL 当真实 KL 数值。

### 11.2 动作门控

只用 cal_fit 中真实动作为舍牌、合法舍牌子集 $\mathcal L_D$ 至少两项且 `action_valid` 的样本；概率分母是完整合法集合 $\mathcal L$，不是舍牌子集。将真实动作 logπ0 升序，取 0-based `n//100`，`>=` 保留 ties，不插值；n<100 取最小值，不外推 99% 保证。log 阈值权威，避免 exp 下溢。无样本/非法概率则 gate unavailable、禁用重排。

cal_check 报告保留率、|S|<2比例、M与支持数分布、当前顺位/立直/副露威胁切片；99%是拟合经验约束，不是因果覆盖率。

### 11.3 显式逐分支质量合同

总体 action/response/rank 的 max_nll/max_brier/max_ece、最小观察数与门控保留率仍按预登记 policy 检查。

另由 `BranchQualityPolicy` 显式登记一组 `BranchQualityCriterion`：

- head 为 response 或 rank；event_id 为0..20。
- 每项必须给 max_nll、max_brier、max_ece、minimum_independent_games。
- 整组给非空 source_reference；序列化包括版本、指标总体、cal_check分区、计数单位、缺项语义、回退行为与完整内容 hash；登记内容进入计划/校准/有限策略身份。
- **Response：** 对每个事件 $e$，总体是 `cal_check` 中 `response_valid` 且公开 mask 允许 $e$ 的**全部**行。取冻结温度与公开 mask 后的 $p_e$，以 $y_e=1[E=e]$ 为真实值，包含正例和负例，计算二元 NLL $-y_e\log p_e-(1-y_e)\log(1-p_e)$、Brier $(p_e-y_e)^2$，以及按 $p_e$ 的 10 个等宽桶比较均值与真实频率的 ECE。分别报告合格总行数、实际正例行数、至少含一个实际正例的独立完整游戏数；只有最后一项能满足本头的最小独立游戏门槛。未知响应、或公开不允许 $e$ 的行不算负例。
- **Rank：** 只用 `value_valid` 且真实 $E=e$ 的行；保留 4 类 NLL/Brier 和 10-bin top-label ECE，独立游戏按真分支有名次行的 game_id 去重。100 行同一游戏仍为 1。总体 response 多分类指标仍照常报告；`response_classification_by_true_event` 的 21 类分类诊断（包括 `top_label_ece`）不作为 response 逐事件准入。
- 反例：同样的预测 $(.9,.1)$ 给 1000 行，其中 900 是 NEXT_DRAW、100 是 CHI_P1。总体 ECE 和两个事件的 one-vs-rest ECE 约为 0；按真实类别分组的 top-label ECE 分别为 .1 和 .9。用**仅合成测试**的 `max_ece=.2` 时旧口径错误拒绝 CHI；这不是生产阈值，也不涉及训练损失错误。
- 某 e 的 response 与 rank 两个判据都 passed，才算该分支质量 passed。每头缺正式判据、缺指标或独立游戏不足为 unavailable；证据足够但超限为 failed。合并时任一已证失败令分支 failed（仍保留另一头的 unavailable），无失败但有缺项才为 unavailable，不让已知失败被缺项掩盖。
- **没有正式默认阈值或默认最小验证游戏数。** 审查反例和单元测试数字不是正式生产标准。未登记就是 unavailable，不以总体平均放行罕见分支。

[`EventSupport.supported`](../src/riichienv/kar/support.py) 只表达每个响应类别在 train 中覆盖至少 500 个去重的完整半庄。定义 $E_{\mathrm{train}}=\{e:N_{\mathrm{train},e}\ge500\}$；`value_supported()` 另检查冻结 `cal_check` 分支状态，只有该响应的预登记 response、rank 两项均通过才纳入有效支持：

$$e\in E_{sup}\iff e\in E_{\mathrm{train}}\ \land\ \mathrm{quality}_{\mathrm{response},e}=\mathrm{passed}\ \land\ \mathrm{quality}_{\mathrm{rank},e}=\mathrm{passed}.$$

规则不可能、训练支持不足、质量 failed、质量 unavailable 分别报告，不混入公开规则 mask。无任何有效价值分支时 runtime 使用同一先验，原因 `no_quality_qualified_response_branches`，不做候选前向；仍有合格分支时对其他分支保留 masked 概率、取−1.5并计入 rho_U。有限生产发布同样要求至少一个训练支持且质量合格分支，以及既有总体质量和独立验证策略选型通过。

有限 τ 的验证规划复用同一有效分支判定；即使总体校准通过，若 $E_{sup}$ 为空，也在创建输出、账本和对局前拒绝含有限 τ 的网格。显式只含 infinity 的先验基线评估仍可运行，但不能形成有限策略收益证据。

### 11.4 公开规则 mask

mask 的1仅表示未被公开事实证明不可能；未知保守允许。Python 验证完整候选整数 ID 为 `0..147` 且不是 bool，并拒绝 pending-kan 快照；再将 observer、候选牌种/立直位、公开舍牌历史/当前河、副露种类、公开立直接受状态、remaining_draws、history_complete 和规则开关传入 Rust 纯 `PublicRuleContext`。绑定不接 snapshot 或 `GameState`，不读取他家暗手/私有合法列表/隐藏牌山/未来事件或真实标签。mask v5 保留四组面子的公开容量约束，并排除公开已确定流局满贯时的普通荒牌，21 类次序不变；旧 mask 必须从可信事件重新生成，不能只修改版本字段。

1. 字牌不能 CHI，只有相对下家可吃；已立直者或已有四组公开面子者不能吃碰大明杠。已声明的暗杠也计作一组；四组面子仍可单骑荣和，不能因此排除 RON。
2. 完整公开历史中，目标玩家自己舍过候选同基本牌种，排除包含该玩家的 RON mask；不猜隐藏等待或临时振听。
3. 三家和按雀魂正常多荣和，保留 RON_MASK_7、排除 ABORT_TRIPLE_RON；相反冻结开关则互换。
4. 候选之后四风连打、多玩家四杠、四家立直按公开证据分 possible/certain。证明任一途中流局必发生时，NEXT_DRAW/CHI/PON/DAIMINKAN及普通荒牌/流局满贯不再可能；**荣和仍先于途中流局**。终止原因优先级为四风→多人四杠→四立直，只有确定证据才压掉低优先级原因。
5. 多人四杠需要总杠≥4且有多个所有者；一人四杠不触发。四杠已满禁止再大明杠。第四家候选立直必须考虑该候选的立直位，不能只看动作前accepted状态。
6. remaining_draws=0 时 NEXT_DRAW 与所有吃、碰、大明杠为 False，保留公开规则尚可能的荣和/终局。remaining未知不因缺失断言耗尽。
7. 流局满贯按雀魂既定修正：自己的吃碰杠本身不取消；自己的中张舍牌或自己的舍牌被叫走才排除资格。先保留已知否定，再对不完整历史作保守判断。完整公开历史、remaining_draws=0 且候选之后至少一人确定合格时，无人荣和就必为流局满贯，因此关闭 EXHAUSTIVE_DRAW；保留荣和优先及确定途中流局优先级。必须计算候选对本家资格的影响，候选中张可以破坏其资格。历史或剩余牌数未知、nagashi 规则关闭时，不应用这条确定性排除。
8. 不完整历史不由未见事实强行判定终止；已明确公开的四立直/多人四杠证据仍可使用。不能通过扩大 Unknown 掩盖已知规则错误。

顺序固定：公开规则 mask → masked Softmax/TE → train+branch质量价值支持 → unsupported概率质量/−1.5 → KL。统计质量不足不改 allowed。全空 mask 是合同错误，不退回未掩码 Softmax；有效真实标签被排除时定位规则或数据错误，不临时放行标签。

### 11.5 报告中的有效未支持质量

冻结顺序：仅用 cal_fit 选择温度/log 门控；计算基础总体与逐分支指标及独立游戏数；按预登记标准从 cal_check 定出分支三态；再与 train 筛查求交得到同一个 $E_{sup}$。随后用冻结 $T_E$、同一公开 mask 和 `response_valid` 样本重算两分区的质量直方图，最后定稿并计算摘要。cal_fit 的质量分布是描述性结果，不作质量放行；两份派生统计均不参与温度、门控或质量判定。

两种指标都基于同一公开 mask 及温度处理后的 $\hat p$，但取不同补集：

| 报告字段 | 口径 | 用途与不可用情形 |
|---|---|---|
| `unsupported_mass_histogram` | 有效 $\rho_U=\sum_{e\notin E_{sup}}\hat p_e$ | 同第 10.2 节与 runtime。evaluation 使用绑定的冻结校准与事件支持；没有该证据时标 `unavailable`。如果冻结证据存在但没有任何合格分支，全部已允许概率均未支持，runtime 仍回退先验且没有候选前向。 |
| `train_only_unsupported_mass_histogram` | 仅计数 $\rho_U^{\mathrm{train}}=\sum_{e\notin E_{\mathrm{train}}}\hat p_e$ | 独立的 train 计数诊断，不用于悲观价值或证书。不能拿它替代有效 $\rho_U$；计数已达标但任一分支质量未通过时，两种指标可不同。 |

两个字段各自保存 support basis、支持位、观测数、状态和桶计数，宽度 0.01；1.0 使用桶 100。公开 mask 剔除的类别概率为 0，不贡献任何一种未支持概率质量。无观测不是实测 0。见 [`ResponseMassHistogram`](../src/riichienv/kar/statistics.py) 与 [evaluation](../src/riichienv/kar/learning/evaluation.py)。

报告和校准产物均检查版本绑定；旧 train-only 桶不能读作当前有效质量。从经核验的冻结 logit archive（若存在且获授权）真实重算，不能给旧 JSON 加字段或重签来迁移。

## 12. 决策执行合同

1. 校验完整可见前缀、时钟版本、完整合法集合 $\mathcal L$、模型/数据/recipe 绑定。
2. 前缀一次前向得到 546 logits，在 $\mathcal L$ 上取经 $T_A$ 校准的先验；构造 S、M。
3. prior-only/τ∞、缺有效门控、总体质量失败、无有效价值分支、S不足或M=0时直接相同先验。
4. 有限合格路径对全部 S 构造独立单 ACTION 后缀，共享同一前缀 K/V，无Top-K。
5. 每候选先公开mask，再TE响应Softmax；TR名次Softmax给分支效用；按Esup计算保守Q和rho，不对可靠分支重新归一化。
6. 受限KL；certified另校验当前误差界及E2，其余标empirical_only；从完整最终分布随机采样，映射回完整原生Action。
7. telemetry记录规则位、train计数支持、每分支质量三态、最终价值支持、masked概率、保守/名义诊断、rho、S/M、q0/qtau、KL、E1/E2、回退、RNG前后摘要。

### 12.1 候选动作价值的事实监督边界

训练集中的每条决策只包含牌谱中实际执行的一个 ACTION。response 与 rank 两个辅助头只在该 ACTION 对应的真实响应已观察、且监督适用时计算损失；未执行候选动作没有真实反事实标签。不得把专家最终选择的事件/名次复制给同一状态下的其他候选，也不得把模型自身预测回填成监督标签。

因此，推理时对多个合法舍牌分别计算 response/rank，只代表模型对条件输入 $(X,A)$ 的泛化预测，不代表这些候选在该状态下的反事实结果已被数据观察或识别。现有 500-game event support 和 cal_check 分支质量是事件类别层面的覆盖/预测质量门槛，不是候选动作级或状态条件级误差界；不得把它们表述为后者。

有限 $\tau$ 的 empirical-only 路径之所以仍可作为受控实验，是因为它还必须通过独立于逐行 NLL 的端到端策略选择：预先冻结完整有限 $\tau$ 网格和完整 cal_fit 场景清单，每个候选与 $\tau=\infty$ 在同一组独立模拟场景上配对，使用第 16 节的同时 Hoeffding 下界。该证据只约束预登记场景总体上的期望配对效用，不约束单个状态的 Q 误差，也不证明反事实模型已被识别。最终强度结论仍需独立 test/M3 评估。

若有限候选未通过该下界，必须保留 $\tau=\infty$ 的行为先验；不得通过增加未经登记的候选、场景或事后筛选场景来制造通过结果。若要求逐状态/逐候选的数学保证，必须另提供绑定当前模型、状态和完整候选集合的有效误差界，并满足 certified 的正 $E2$ 条件；当前 500-game 计数、温度校准或整体模拟收益均不能替代该误差界。由于反事实标签缺失，当前阶段不采用伪标签、同状态标签复制或未经验证的合成反事实扩增。

### 12.2 候选 ACTION 的分支监督（下一版数据合同）

当前 v1 数据集每个原始决策只有一个实际执行 ACTION。它足以监督行为先验 $\pi(A\mid X)$，也能监督该事实 ACTION 对应的响应/名次头；它**不足以直接监督同一状态下其余合法候选的反事实响应与名次**。因此，当前 v1 的 Response/Rank 头不得被描述为已对全部推理候选获得直接监督。仅靠把同一事实标签复制到其他 ACTION、随机替换 ACTION Token 或模型自蒸馏，均不能修复该缺口。

下一版候选价值数据必须通过原生引擎分支模拟产生，定义为：

$
\mathcal D_{branch}=\{(X_i,A_{ij},E_{ijr},U_{ijr},w_{ijr})\}
$

其中 $A_{ij}$ 是原决策时刻全部合法完整舍牌候选（global ID 0..147 的实际合法子集），$r$ 是分支续局 replicate，$E$ 是该分支首次舍牌响应窗口的最终裁定事件，$U=2.5-\mathrm{rank}$ 是完整终局名次效用。**只对真实执行的专家动作保留行为策略标签**；候选分支标签必须来自该候选自己的模拟结果，不得从事实轨迹复制。

分支生成合同：

当前实现版本固定为 `kar_candidate_action_branch_v2`，续局策略固定为 `native_riichi_baseline_v2`。导出默认开启候选分支监督，每局默认从整局时间轴的早、中、晚三个区段进行确定性分层哈希抽样，最多选择 8 个本家自摸/出牌决策状态，每个候选默认 1 个 replicate；不再固定取最早 8 个状态。manifest 必须记录抽样算法和各区段覆盖统计。 `enabled=true`、每局上限、策略版本和墙牌采样版本。显式传入 0 属于配置错误，不能静默关闭候选分支监督。该默认上限用于控制首轮生成成本，不代表只覆盖 8 个决策就足以证明全局候选价值质量。

续局策略不再对所有合法动作均匀随机：合法和牌优先；响应窗口优先 Ron，其他鸣牌仅以低概率采样，否则 Pass；自身出牌用可解释的手牌效率启发式，倾向保留对子/连接牌并避免切掉宝牌与赤牌，听牌时适度偏好立直。所有动作仍由 Rust 当前合法动作集合约束并通过原生执行接口提交。该策略是冻结的 baseline，不是已训练的强牌 AI；价值目标明确条件于此策略，后续更换策略必须提升策略版本并重新生成分支数据、support、calibration 和 release。

回放 `start_kyoku` 使用的墙牌可能是占位数据，不能直接拿来续局。分支执行前，必须调用 Rust `prepare_replay_continuation_wall(seed)`：从当前四家暗手、河牌和副露中扣除已确定的物理牌，保留已公开宝牌指示牌在死墙中的原生位置，按当前岭上牌计数和剩余牌数重采样未知墙牌；牌山长度、可摸牌数和已公开状态不变。同一源决策的同一 replicate 对所有候选使用同一 wall seed（common random numbers），策略动作 RNG 则按候选独立派生。Python 不构造牌池、不实现规则，只调用 Rust 原生状态方法。该重采样墙是一个明确的模拟条件，不声称还原牌谱中未公开的真实未来牌山。

1. 在可信牌谱 replay 到每个选定决策 cutoff 时，保存该时刻完整 Rust RiichiEnv 状态的深拷贝。当前 Rust GameState 实现 Clone，Python RiichiEnv 暴露 clone()；此状态包含隐藏手牌与牌山，只能作为标签生成器输入，绝不能序列化进模型特征或交给模型。
2. 对每个决策的每个合法舍牌候选，从同一冻结状态分别克隆分支，并通过现有原生 execute_candidate/Rust step 路径执行候选；禁止 Python 重新实现规则、修改隐藏状态或手工合成响应事件。
3. 原始后续牌谱不能用于替代分支的未来。候选一旦改变，之后轨迹通常已经分叉。每个分支必须由预先冻结、版本化的 continuation/opponent policy 继续完整对局，直至产生真实终局；响应事件由分支实际执行产生，名次由 Rust 终局结算产生。
4. continuation policy 的模型权重/规则版本、策略混合比例、采样温度、随机种子派生规则和版本摘要必须写入数据 manifest。训练目标明确是“在该冻结对手策略分布下的条件响应/名次”，不是不依赖对手的绝对价值。正式发布仍须在独立对手池和 M3 中检验迁移。
5. 同一源决策下的所有候选使用相同的源状态；replicate 的随机种子按 (source_game_id, cutoff, candidate_id, replicate, continuation_policy_id) 稳定派生。记录完整分支身份、执行动作、响应事件、终局、失败/截断原因。截断或非法分支不得伪装成普通终局标签。
6. 先按完整源 game_id 划分 train/validation/test，再展开候选和 replicate。一个源局的任何候选分支不得跨 split；禁止按分支行随机切分。
7. 同一源决策展开成 K×R 行时，行为策略损失 L_A 每个源决策只计算一次；Response/Rank 分支损失按源决策归一化权重，使每个源决策的总辅助头权重固定，不因候选数量或 replicate 数量较多而被过度加权。对同一 (X,A) 的多次 replicate，保留计数并用经验分布/软标签训练 p(E|X,A)，并对每个真实事件估计 m(X,A,E)；不得只保留一次随机结果而丢失方差信息。
8. 训练时可用重复 Prefix + 单 ACTION 的 dense batch 先验证数学正确性；后续如做共享 Prefix KV 的可微训练优化，必须与 dense reference 的输出和梯度对拍。现有 encode_prefix() 是 eval-only/no-grad，不能直接拿来训练，否则会切断 Prefix 梯度。
9. 价值监督支持必须按 (源状态/上下文, 候选动作, continuation policy) 报告分布与覆盖，不能只用 21 类事件的 500-game 计数替代。低覆盖候选不得被解释为已可靠估值；没有候选分支支持时维持 prior-only，不做有限-τ 重排。
10. 默认每局前 8 个本家自摸/出牌决策开启分支生成。必须报告每个决策候选数、分支完成率、每个候选 replicate 数、每局分支成本、标签方差、事件/名次覆盖和分支策略版本；扩展每局上限或增加 replicate 前，先核对这些统计与生成成本。

**损失合同**：L_A 仍使用每个原始状态唯一的专家 ACTION；L_E 和 L_V 改为只使用该源状态所有已完成、有效的候选分支行，按源状态归一化。旧 v1 value_valid 事实标签不能与新分支标签混在同一个未经区分的总体中。新数据必须提升 DATASET_VERSION、分支监督版本、训练 recipe 和模型/release binding；旧 checkpoint/calibration/support/tau-selection 不能通过改版本字段迁移。

**推理合同**：仅当模型绑定了对应分支监督版本、continuation policy 目标、候选覆盖证据，且独立校准与策略验证通过时，才允许有限-τ 候选价值重排。否则 Response/Rank 只能作为事实动作的诊断输出，策略保持 τ=∞ 的行为先验。事件级校准通过不自动意味着候选级价值通过。当前 held-out calibration 尚未收集 candidate-action 条件分支指标，因此生产推理必须拒绝有限-τ 重排并回退 prior-only，直到 candidate_action_conditional_validation 独立通过。

**验收测试**：
- 对同一冻结 Rust 状态，所有合法候选分支均从完全相同的源状态开始，原始环境不被任何分支修改。
- 每个候选实际执行的原生动作与 global ACTION ID 一一对应，非法/重复候选被拒绝。
- 分支响应事件与终局名次能由 Rust 日志/结算独立复核。
- 任意一个候选的未来事件变化不会修改其他候选的分支记录。
- 按 game split 后不存在跨 split 的 sibling branch。
- 行扩展前后每个源决策的 L_A 权重相同；辅助头按源决策而非按展开行数归一化。
- Dense 单 ACTION 训练前向与候选分支推理前向在相同权重/输入下输出一致；反向梯度与 dense reference 一致。
- 分支策略或其权重摘要变化时，旧数据、校准、support 和冻结 release 全部拒绝复用。

## 13. RiichiEnv/雀魂适配

离线 GameDetailRecords.actions 只读 `.result`，绝不读 `.user_input`。下载 pb2 不 import 执行；协议由可信 proto 静态字段和描述符核对。`protocol_sha256` 保留主 proto 文件字节身份，`protocol_descriptor_sha256` 绑定实际编译的完整 FileDescriptorSet（包括 imports）。解码、逐局审计、完成证书、Golden、审计恢复/合并、接受选择、导出和训练读取均须核对两者及 `protocol_identity_version`。只改被导入的协议也可能改变相同 wire payload 的解释；主文件 SHA 相同不能放行。缺少完整身份的旧派生产物拒绝使用，不能事后补一个 hash 迁移。source_record_index/hash保留来源，不影响模型时间。

离线 replay 在已知 cutoff 调观察；不得用将未来日志注入 current_claims 的 helper 构造决策前合法集合。原生 `ReplayContext` 在 MJAI 事件入口推进一发、双立直、岭上和杠见逃等规则上下文；Python 桥只验证版本、预览及事件配对，不从快照推导或回写 flag。真实合法响应只用于审计/冻结机会证据，不能输入公开 mask。

在线在完整原子执行成功后消费 Rust `rule_log`；原生输出暗杠声明/提交、终局实际应公开手牌及结算。legacy `mjai_log` 接口保留，Python 不再补暗杠声明、调用 `HandEvaluator` 决定在线听牌揭示或自行分配终局供托。全局规则日志可能包含私有牌张，模型只收经原生观察投影后的本家 snapshot，不收四家全知协调对象。

立直复合完整动作在声明前冻结输入；完整可舍牌来自保存环境规则的原生 Observation 的 `legal_riichi_discards()`，执行通过 `step_riichi_discard` 原子提交 reach 与对应 discard，失败不能留下部分声明。结构听牌即可，不因外部可见等待牌四张（空听）而排除合法立直。本节固定雀魂规则排除本家已持四枚的单骑形式听牌；不能将此规定扩大为所有平台的通用听牌定义。

暗/加杠分声明、抢杠窗口和提交；KAKAN声明时仍是原PON。仅有本次有效dora/替换draw/匹配ACK证据才提交，旧杠延迟Dora不证明新声明；被抢杠不提前改meld。对手暗杠红牌组成按可见性过滤，不从全知记录泄露。

冻结雀魂规则包括赤牌、三家和、抢暗杠国士、倍役满、包牌、喰替、起家顺破平、终局及流局满贯。规则hash与数据来源必须一致，不混用天凤规则。流局满贯外部依据见[雀魂规则取证](kar-nagashi-rules.md)。

和牌拆解按实际支付选优，普通累计番仍按单倍累计役满封顶，不能因宝牌使原始番数更高而覆盖真正的多倍役满；同支付时保留真正役满身份供后续规则与包牌处理。天和没有固定的和牌张，按[雀魂官方规则](https://mahjongsoul.com/news/46/)选择得分最高的解释，四暗刻单骑、纯正九莲和国士十三面不能由十四张牌的排列顺序决定。普通自摸、地和和荣和仍使用实际和牌张。四人和三人计分器共用拆解比较及天和解释逻辑；这不扩大为对三人整套平台规则的验收。

四人在线与独立审计共用 Rust `rules::settlement` 的支付、残余供托、连庄和终止内核，但平台规则由显式 `GameRule` 决定。KAR 固定雀魂；以下差异依据[雀魂官方段位战规则](https://mahjongsoul.com/news/46/)与[天凤官方四人段位战规则](https://tenhou.net/man/#RULE)：

| 条件 / 原生开关 | 天凤 / `GameRule::default()` | KAR 雀魂 |
|---|---|---|
| 本家碰三枚且暗手单张同牌、纯手未用四枚的形式听牌 / `allow_fifth_copy_tenpai` | 允许，`true` | 不允许本家四枚单骑，`false` |
| 西入普通荒牌后他家达三万点且庄家听牌但不是第一 / `extension_dealer_repeat_priority` | 连庄优先，`true` | 达标结束，`false`；保留官方庄家双荣和例外 |

无规则参数的 `HandEvaluator.is_tenpai/get_waits_u8/get_waits` 保留旧的天凤兼容枚数语义；Rust 显式 `*_with_rule(&GameRule)` 与高层 Python `is_tenpai(rule=...)`、`get_waits(rule=...)` 区分平台，KAR 显式传入 `MJSOUL_RULESET.engine_rule()`。回合推进同样显式传规则，旧序列化 `GameRule` 缺少这两个字段时恢复兼容的 `true`，不能把它误作新的 KAR 雀魂合同。

独立审计仍从原始牌张/事件重建事实，服务器支付、header 终分和 `apply_event` 分数仅作比较目标；共享公式不替代来源完整性、已知规则案例或完整半庄证据。三人状态机不在本轮迁移范围，不能声称其雀魂规则差异已修复或验收。

## 14. 训练配置与授权边界

从随机初始化，不继承旧 BC/CQL checkpoint。Small feature-on/off M1同split/seed/样本序，每模式最多1Epoch，effective batch128；feature-on训练整样本25%置空，feature-off清全部新增规则字段。

M2唯一主力默认2Epoch，只有验证继续改善、预登记各头无显著恶化且预算允许才第3Epoch。Small/Medium/Large batch128/256/512；AdamW lr3e-4、betas(.9,.95)、eps1e-8；global L2 clip1；warmup总updates3%且至少100，余下cosine至0.1峰值。每0.25Epoch和Epoch结束完整验证。主参数FP32，可用硬件BF16主干，但三头投影始终FP32。

weight decay：value_head.weight=.05；其他二维矩阵=.01；所有bias/norm与value_head.bias=0。不做首轮WD网格试错。

训练规则特征增强：每决策样本抽一次 Bernoulli(.25)，命中时该样本全部34个HAND_COUNT flags[0..7]同步N/A，ROUND_STATE numeric18/19清零且present=false；不清stage或其他字段。全局feature-off另清ROUND_STATE.stage。验证/测试/推理置空概率0。随机流按原sample_mask_v1和seed/epoch/game/decision/draw派生，不因DDP rank改变样本增强。

`updates_per_epoch=ceil(N_train_decisions/effective_batch)`，不是固定2000更新。正式配置同时保存N_train、N_eligible、available/exported counts、tier/basis、epochs、recipe和预算；无真实吞吐/单价不伪造测量。

Checkpoint保存模型、optimizer/scheduler、Python/NumPy/Torch/CUDA RNG、sampler/分桶游标、所有活动artifact字段、source/native构建指纹、完整训练manifest与split、固定rho/lambda、预算账本/登记身份、温度/log阈值/support/quality与验证策略引用、Rmax/tau_min、证书来源和环境版本。任何核心字段缺失/不匹配拒绝精确续训。

有界诊断检查点和正式检查点共用代际发布：先完整写入不可变 payload、flush/fsync，再原子替换同目录 JSON 指针，并在 POSIX 同步目录。新指针提交前不能覆盖或删除上一份已提交代际；成功后保留当前和上一代，未提交的孤立文件不自动删除。读取须验证指针所属命名空间、payload SHA 与 metadata SHA。有界诊断格式升级为 `kar_exact_resume_v2`，其旧单文件 v1 要求显式迁移；正式检查点保持原格式版本和完整训练状态合同。

输出目录与 `startup.json` 身份一起原子发布；身份绑定数据、配方、实现内容、运行布局、原登记及原账本路径。`--resume` 在首份 checkpoint 前仅允许核验过的初始化重入，保留原账本；创建时已有的共享账本还必须保留原始前缀。模型初始化后先保存 step 0 checkpoint，再执行初始完整验证。已提交标记、更新日志或非零 step 代际证据存在时，不得因 checkpoint 缺失而重新初始化；无 startup 记录的既有 checkpoint 仍走原精确恢复校验。

正式 trainer 先从已核验的授权记录取得数据 manifest 摘要，绑定启动身份并持久化原账本中的 compute attempt，然后才读取和校验完整 manifest、计算 SQLite 文件 SHA、加载索引、构造长度表、复验 M2 前置证据，以及初始化线程、设备和进程组。实际加载数据必须与授权摘要一致；授权摘要不能替代完整数据校验。预处理失败也保留启动与失败记录，并按本次计时结算；未核销的旧 attempt 在任何新数据加载前拒绝继续。

多进程在设备和进程组建立前，通过共享输出文件系统协调启动。torchrun 的独立 run ID（含 `--standalone`）或其他 launcher 显式提供的 `KAR_DISTRIBUTED_RUN_ID` 与 restart count 绑定一次启动；放行收据必须绑定本次 primary 和各 rank 的新 nonce，旧成功记录不能放行新 worker。所有 rank 等 primary 的账本登记完成才开始数据预处理，普通失败使各 rank 一起退出。M1/M2 外层创建进程组使用同一阶段账本的独立启动计时间隔，结算后再由 trainer 接续，避免重复收费；调用方自行提前创建进程组的既有成本不能由后进入的 trainer 回溯计时。

冻结校准与 τ 选型同样先验证预算/授权、持久化独占输出身份及 compute attempt，再经延迟加载工厂读取模型、数据摘要和索引。准备阶段先结算，超预算不能进入前向或对局；准备失败与计算失败共用同一账本，硬中断保留 pending attempt。既有输出不得被新启动覆盖；需先审计、对账并明确处理失败产物，不能静默续接部分 logits 或选型结果。API 接受已加载对象以保持调用兼容，但调用方在进入作用域前发生的开销无法追溯计时；正式 CLI 使用延迟工厂覆盖自身的加载成本。

总云资源USD1000独立封顶：M0/M1=100，M2=650，M3=250，包括CPU/GPU、存储、快照和传输。实际spent、未对账fixed reservations及未完成compute reservations分别记录。计算前持久化attempt，阶段内实测结算递减预留但保持attempt，正常结束或可捕获异常才关闭；没有独立初始化预测时，零金额预留仍是未完成标记。硬中断后不把新进程时钟当成旧计算为零的证据，必须通过外部实际费用凭据核销未记账余款后才能继续。收据唯一ID幂等，checkpoint绑定账本锚点，拒绝费用回退；旧账本版本和不完整尾部要求显式审计迁移，不自动重写。不能跨阶段挪用、缩M3样本或隐费用凑预算。

## 15. 计算与缓存成本

前缀一次；最多32个单Token候选后缀，只算门控舍牌。无MCTS、扩散或多步轨迹搜索。FP32参数/梯度/Adam两moment粗下界16P：Tiny31.8MiB、Small79.9MiB、Medium225.7MiB、Large404.9MiB，不含激活/workspace/缓存，不能据此声称GPU batch可运行。

近似训练量 `6NL(4d²+3df)+12LN²d`；真实成本必须测tokens/s、samples/s、长度桶padding、allocated/reserved显存、数据等待，以及prior-only/finite候选成对推理与完整模拟吞吐。纯prior benchmark不能为finite候选成本背书。

## 16. 统计、校准与棋力证据

报告自然频率action/response/rank NLL、Brier、top-label与每类可靠性、有效样本和独立game数；强制动作与多合法动作分开，立直/副露/顺位切片，响应按RON/CHI/PON/DAIMINKAN/terminal族与21分支，rank按真实E分支。效用分桶与独立game聚类区间不当作uniform错误界。报告所有failed/unavailable，不只总体平均。

完整验证分片覆盖所有合格行。只有 primary 汇总各 rank 的统计并构造全局报表；先协调错误，再广播同一结果。聚类 bootstrap 只在有相同有序合格 game ID 集的指标间复用固定种子抽样；每指标保留自己的游戏权重、分母与分位数，逐轮分配索引而不是缓存整张抽样矩阵。分别报告最大 rank 的本地 `forward` 墙钟（含该循环的数据准备与累加）、primary `aggregation` 和 `bootstrap` 时间；不缩小验证集。

同checkpoint比较τ∞先验与验证冻结finite策略，固定规则/对手版本/配置、座位轮换、成对初始牌山seed；策略采样RNG独立记录，轨迹分叉后不声称相同反事实。M3为10,000 paired seed blocks、共20,000半庄、四个初始座位平衡；按整场/paired block聚类区间，报告顺位/一四位率/点数/和放铳/覆盖/延迟/费用。未获可重复验证收益保留prior-only，不靠扩模型试错。

有限 τ 的准入使用预登记固定网格的全部有限候选数 $K$，以及完整 `cal_fit` 场景清单；清单的内容、数量和摘要随计划与选型证据一起保存。每个候选必须在同一组 $n$ 个独立模拟场景上与 τ∞ 配对。名次效用 $U=2.5-\mathrm{rank}\in[-1.5,1.5]$，故配对差 $D_i=U_{\tau,i}-U_{\infty,i}\in[-3,3]$。对预登记置信度 $1-\alpha$，使用单侧 Hoeffding 下界与网格联合界：

$$L_\tau=\max\!\left(-3,\overline D_\tau-\sqrt{\frac{18\log(K/\alpha)}{n}}\right).$$

只有独立场景数达到预登记下限，且 $L_\tau$ 严格超过预登记最小收益，候选才可通过；否则选择 τ∞。$K$ 不能只数事后表现较好的候选；仅含 τ∞ 时不计算有限策略下界。不同候选共享牌山产生的相关性不妨碍联合界，但不同场景之间的独立性、看结果前冻结网格和场景数是该置信解释的前提。该判据不支持看完结果再换子集、增加样本直到通过，或把多次重新选型当成同一次预登记试验。

配对整场 bootstrap 区间仍用于描述经验分布，不再单独批准有限策略或扩模型。两局配对收益都为正会产生退化的正 bootstrap 区间，却不足以使默认 95% 的上述下界为正。发布与 M2 规模门槛均核验完整场景清单，并从候选均值、样本数、网格和预登记策略重新计算收益下界；不能只信任序列化的 `reproducible_gain`。该分布无关下界较保守，未通过只表示当前证据不足，不等于真实收益为负；评测工作量和预算不因此自动增加。这里得到的是固定评测设计下的平均收益证据，不是逐状态误差证书，也不改变第 10 节的 `certified` 条件。

新选型证据有独立版本；旧证据必须按新契约重新生成，不能仅改版本或摘要继续使用。本次修复不改数据格式、Token 布局、模型参数形状或损失函数；实现内容指纹仍正常变化，旧 checkpoint 的精确续训和冻结发布须遵守原有来源一致性检查，不能据此宣称跨实现版本兼容。

## 17. 阶段顺序与运行许可

- M0：完整来源、结果解码、重放、独立结算、完整半庄、observer泄漏、动作/Token/clock/特征/规则mask/真实Golden与计数审计。合成测试不是新的真实Golden证书。
- M1：授权后Small三头loss/梯度/DDP/恢复与feature-on/off、资源测量。仅forward通过不代表梯度通过，更不代表阶段完成。
- M2：在M0/M1和预算证据齐全后，以N_train选唯一主力，显式登记唯一output/账本/recipe，完整数据训练2、条件3Epoch。
- 校准/finite发布：预登记overall+branch质量标准，cal_fit选择、cal_check只验收，校准/支持/后端/随机策略与验证选型完全绑定。
- M3：只有另获明确授权且完整20,000半庄预算预测通过才运行；不足不静默缩样本。

规则、隐私、动作或mask错误先停止；完整数据或价值支持不足保持unsupported；质量失败或不可用保守回退；无有效误差界不称certified；预算不足按预定降档或停止。文档、单元PASS、数据training_allowed和历史授权都不是当前任务的运行许可。

## 18. 必须保留的永久回归

- 三头形状/参数量、五张独立tile表、40维无bias数值投影；dense/cache FP32与外层CPU BF16投影边界；同hidden FP32参考，拒绝后置cast假修。
- 完整LEGAL_SET五表均值、等长不同集合影响raw policy、顺序/padding不变、非舍牌N/A、teacher不影响DECIDE、单ACTION与分支隔离、陈旧/跨上下文缓存拒绝。
- 同语义cutoff的离线/在线所有输入字段与positions对拍：开局/庄家首次draw、reach/discard、accept/draw、Dora、多Ron、杠声明/抢杠/提交、相同已知本家Pass；batch/incremental/合法分块，来源元数据另比。
- 九种九牌公开前不泄漏、source seat/tiles矛盾拒绝、规则重建来源、零结算presence、四视角和下一局实际决策完整历史/位置；缺终局可见信息不可伪装完整。
- 原生合法136牌墙四家立直、四风、多人/单人四杠、末巡和相邻边界；荣和优先、未知历史保守、有效标签不被排除，mask白名单/空mask错误。
- 总体指标通过但罕见分支失败；passed/failed/unavailable、独立game与重复行、规则不可能、train499/500分别表达；失败分支p保留但不能供应价值，S外概率/M不变，旧缺合同产物拒绝。
- 已确认的原生 Pass 在整批、`ankan | dora | tsumo`、receipt 先单独到达时最终输入一致，并在关闭锚前不写入；混合 ready/waiting、重复和非法 receipt 保持各自语义。
- 总体通过但罕见分支失败时，1000 局合成反例要求校准报告、加载后报告、evaluation 与 runtime 对同一冻结集合得出同样的有效质量分布 `{0: 990, 99: 10}`。train-only 必须独立，缺冻结质量的 evaluation 不能伪报实测有效质量。
- 安全哨兵、全无目标、固定rho、DDP/累积梯度等价及恢复、名义/保守误差域、完整有限集合KL/E1/E2反例；本轮禁止反向时不得执行梯度节点而伪记通过。
- red5/普通5完整动作、空听立直合法、声明前不扣棒、多Ron完整mask、PON优先窗口、DAIMINKAN后岭上自摸、被抢KAKAN不提交、对手暗杠赤牌隐私、同game四视角同split、未来/隐藏扰动不影响已封存输入。
- 公开状态相同但隐藏 Ron 能力不同仍产生相同公开 mask；原生 ObserverState 不保留对手暗手；完整立直失败不部分提交；直接 MJAI 重放不依赖 Python 写规则 flag；rule_log 的声明/提交及终局揭牌按批次投影一致。
- 同一本家成立碰加暗手第四枚单骑，雀魂不听而天凤/default 保留形式听牌；无参旧接口兼容、显式规则与旧序列化默认值可区分。合法第四枚/外部空听不误删；西入普通荒牌跨三万点时雀魂结束而天凤庄家听牌继续，KAR 在线、独立结算及完成证书显式使用雀魂并保持一致，保留平台规定的例外。
- Ryanmen九点查表、3/7边张排除、full-suji双侧要求、被叫走历史与物理并集去重、起家顺tie-break、真实零/缺失presence、kind-specific字段、whole-sample随机置空及参数量不变。
- 数据/SQLite/checkpoint/calibration/support/发布所有新字段的缺失/旧值拒绝，即使校验和有效也不能静默补字段迁移。

这些是设计要求，不是本次运行结果。各历史通过数只归属各自的提交和验证范围；本轮原生构建与回归结果单独记录在[边界迁移记录](kar-native-rule-boundary.md#7-验证位置与当前证据)，不能以历史文档校验替代。

## 19. 旧产物拒绝与真实重建

进入 V4 时，输入位置、终局公开历史与 mask 已改变，artifact v1/v2/v3、source-record 时钟、旧输入、mask v1/v2、旧 quality/support/calibration 与有限策略身份不能跨版本直接复用。旧输入的重建要求仍如下：

- 从只读原始result事件重新审计/规范化/导出新Token、mask和manifest，输出新目录，不覆盖旧库或重签旧hash。
- 所有旧历史缓存/KV重新生成；同Schema或同shape不证明输入语义一致。
- 旧权重即使state_dict可加载，也不是当前精确训练/校准/评测产物；标准source/native与版本加载检查拒绝，任何迁移须显式实现、单独授权与真实验收。
- 旧支持需在当前合格train选择重新统计；旧校准需固定新输入/模型、正式预登记overall与branch判据后重新计算；finite策略选型/吞吐与发布随之重建。
- 真实旧产物不得补字段、改标签或换 hash 冒充新产物。缺正式分支判据时使用 prior-only 或 unsupported，不声称已校准。

`c095084` → `f70684e` 的历史实现已处理分批 Pass 和有效 $\rho_U$。旧分批轨迹若丢了 receipt，只能凭原始、可信的已提交 receipt 重新投影，不能从公共 trace 逆猜私有 Pass。

后续冻结机会和原生振听修复改变了部分合法候选及训练行：从原始、可核验记录使用重新构建的原生扩展重新重放、审计和生成派生 SQLite、索引／选集、指纹及训练配方。错误 Pass 删除后其后的私有历史、行序号或有效样本计数也可能改变；不能只改 manifest 或因为支付一致而沿用。旧输入的缓存、精确续训 checkpoint 和发布都不能当作当前合同。

逐事件校准 v6 不能迁移 v5 的真实类别切片；需要在匹配当前数据、模型、mask 的冻结 logits 和独立 `cal_check` 上按新预登记语义重算。绑定旧输入的 logit archive 不能直接重算来代替输入重建。统计合并状态、训练支持产物、有限策略选择和发布按相应当前绑定重新生成或验证，不重贴阈值或 hash。实施未覆盖原始或既有训练产物，详见[本次迁移报告](kar-furiten-event-calibration-repair.md)。

以 `4c09e9e` 为基线的响应监督修订还会补入非平凡 Pass、改变可证私有历史，并记录不可辨认机会；旧 audit、接受快照、训练 SQLite、索引／样本序、计数与绑定指纹应由可信原始结果重建。固定 rho 的计算与使用方式不变；新导出如因补行改变 train 样本数，其 manifest 比例由实际训练集重新计算，而非在训练中动态再估。冻结 FP32 recipe hash 也已变化，旧缓存和旧绑定的静态 logits／校准／有限策略及发布证据不能重贴。新验收不覆盖原始数据或旧训练产物。

以 `7626166` 为基线的原生规则收敛进一步升级 KAR 雀魂 ruleset、input v7、native replay v4、settlement v2 和 live projection v6，并新增 `native_rules_version`。雀魂规则 hash 显式包含 `allow_fifth_copy_tenpai=false` 与 `extension_dealer_repeat_priority=false`；旧真实 Golden、audit/接受报告和原始数据必须用新扩展重新核验，不能由旧默认值推定平台。受影响的合法候选、终局公开历史、完成/名次标签及派生 SQLite、缓存、索引、统计和绑定需真实重建或验证。绑定旧规则/输入的 checkpoint、校准 logits、有限策略与发布证据不能因 546 Schema 和参数形状未变而直接复用。给旧 JSON 补字段或换 hash 不是迁移；本轮也不自动处理已有训练产物。具体范围见[原生规则边界](kar-native-rule-boundary.md)。

在 `4132c6a` 后的修复将 native rules 升到 `riichienv_native_rules_v2_scoring`，强制重新编译扩展；完整协议身份、Golden/完成证据 v2、单前缀冻结 recipe 与校准 job v3 均有显式绑定。旧计分、旧协议解码或旧 dense policy 校准产物不能直接作为当前训练/发布证据，须从可核验来源重新审计、导出或收集。回归仅证明已覆盖案例中的错误得到修复；未扫描真实历史训练产物，不据此断言已有训练权重或整个语料受污染。

后续鸣牌修复将 native rules 升到 `riichienv_native_rules_v3_meld_rules`，Response Mask 升到 `public_rule_response_mask_v4_meld_capacity`。四组面子的公开容量会改变部分响应概率的归一化范围；旧 mask 导出及绑定产物须由可信来源重新生成或核验，不能仅改版本。自定义 `kuikae_forbidden=False` 的三人碰牌及四人吃碰在线行为也已与重放统一；KAR 冻结雀魂配置仍使用 `True`。动作 Schema、参数形状和损失公式不变，具体回归见[原生规则边界第 9 节](kar-native-rule-boundary.md#9-后续鸣牌修复公开容量与食替开关)。

在 `85884ab` 后的本轮修复进一步升级 native rules 为 `riichienv_native_rules_v4_nagashi_mask`、mask 为 `public_rule_response_mask_v5_certain_nagashi`，排除公开已确定流局满贯时的普通荒牌；冻结推理 recipe 升为 `fp32_ieee_single_prefix_v3`，显式绑定实际数值后端。旧原生扩展、旧 mask 数据和上一版单前缀校准均明确拒绝；即使重算摘要也不能替代语义重建。实现和本轮验证范围见[流局满贯与冻结精度修复记录](kar-nagashi-precision-repair.md)。

## 20. 风险与声明边界

响应稀疏、条件混杂、有限网络误校准、行为先验偏差、对手分布漂移、长期随机性、粗事件粒度、独立游戏不足、规则/数据来源差异和同分规则漂移仍是风险。规则特征不增加理想预测器信息量；其可能作用只是有限模型归纳偏置。SHA是内容一致性与来源绑定，不是秘密签名或恶意发布者认证；checkpoint只从可信来源读取。

## 21. 实现入口与单一文档原则

| 模块 | 职责 |
|---|---|
|Rust `rules::settlement` / `HandEvaluator`|按显式平台规则共享四人支付/终局与听牌/和牌评估；KAR 使用雀魂|
|Rust `ObserverState` / `public_rules`|观察者事实与可见性投影、仅显式公开上下文的 21 类可能性|
|Rust `ReplayContext` / `GameState.rule_log`|重放 flag 与杠事务、在线规范声明/提交及终局揭示|
|`mjsoul`|可信result-only协议解码与离线规范化|
|`event_semantics`|公共语义时钟、九种九牌宣布者手牌验证|
|`live_events` / `observation`|规范事件批次、终局组、历史/时钟与本家 Pass 来源，适配原生观察事实|
|`tokenize` / `schema` / `actions`|完整宿主编码、历史缓存、546 ID 与原生完整动作适配|
|`response_mask` / `response`|公开规则白名单适配与实际响应标签，二者分开|
|`rule_features`|物理known-count、Genbutsu、Ryanmen/Suji、rank/gap|
|`calibration_quality` / `calibration`|预登记总体/逐分支判据、heldout指标和校准身份|
|`support` / `runtime` / `policy`|train支持∩质量准入、悲观值、受限KL与证书边界|
|`versioning` / `runtime_contracts`|统一活动身份，拒绝缺字段和旧合同|
|`learning`|数据、模型、损失、预算训练/恢复、冻结推理与发布|
|`audit` / `settlement` / `completion`|来源重建、原生合法性/共享规则调用、独立比较与完整半庄证据|

详细 API 见[模块 README 索引](kar-modules/README.md)。[原生规则边界](kar-native-rule-boundary.md)、[设计一致性](kar-design-consistency.md)、[输入链历史说明](kar-input-chain-repair.md)和[实现覆盖](kar-implementation-coverage.md)是来源/实现/证据说明，不是并行主设计；若存在旧版本文字，不能代替本文的活动规范。

## 22. 最终实施规格摘要

RiichiEnv宿主；完整Schema/hash固定；源record不决定时间；同语义cutoff和已知私有事实产生相同输入；终局只揭示规则实际公开且已验证的事实；62槽/944行/139字节；完整LEGAL_SET；私有Pass因果确认；三头FP32投影；N_train选型；自然三项CE；公开mask先于Softmax，质量不足独立进入unsupported；Esup同时满足train与逐分支质量；受限KL保持S外概率及M；随机采样；无误差证书不称certified；严格版本/source绑定；任何真实运行另需用户明确授权。

## 23. 来源

- 原始设计及其SHA见页首；Kanachan动作/tile37仅来源参考，非当前工程目标。
- [RiichiEnv 上游冻结基线](https://github.com/smly/RiichiEnv/tree/0ee1236a283b15508f3ac28407027f340b81e1de)；本仓库的 [原生合法动作](../riichienv-core/src/state/legal_actions.rs) 和 [状态规则](../riichienv-core/src/state/mod.rs) 定义原生鸣牌余量及荣和/途中流局优先级。
- [雀魂官方段位战规则](https://mahjongsoul.com/news/46/)与[天凤官方手册的四人规则](https://tenhou.net/man/#RULE)分别规定形式听牌和延长局连庄优先级；两者不能互相替代。平台差异及 API 兼容见[原生规则边界](kar-native-rule-boundary.md#3-平台差异与-kar-雀魂修复)。
- 雀魂 `RecordLiuJu` 源协议字段号包括 `type=1`、`gameend=2`、`seat=3`、`tiles=4`、`liqi=5`、`allplayertiles=6`。只采用公开语义所需字段；协议来源和当时证据见[历史 V4 实现报告](kar-semantic-branch-repair.md)，本次文档校验未重新审计真实牌谱。
- [雀魂官方FAQ](https://mahjongsoul.com/faq/)及[API](https://mahjongsoul.com/api/faq/list)的流局满贯条文，相关取证见仓库规则文档；自己鸣牌不取消资格。
- [Guo et al., 温度校准](https://proceedings.mlr.press/v70/guo17a.html)、[Canonne, KL/TV](https://arxiv.org/abs/2202.07198)、[torch.gather](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)、[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。这些方法不替代当前数据/误差/棋力验收。

**文档结束。**
