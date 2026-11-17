# KAR 实现覆盖与本轮设计一致性验收（2026-11-17）

逐模块职责、接口、数据合同、使用方式及测试位置见 [72个模块 README 总索引](kar-modules/README.md) 与 [learning 子包索引](kar-modules/learning/README.md)。

## 范围与状态

宿主为 **RiichiEnv**；依据 `KAR-v1.2.2-DESIGN.md` 及已批准的 `kar-riichienv-v1` 完整 Schema、雀魂 mode12 主 cohort、`N_train_half_games` 模型阶梯适配。Kanachan 仅保留动作编号参考。

下表沿用原 58 项编号，记录实现入口，不是全项目运行通过表。**未在末尾本轮验收矩阵列明的完整路径，一律为 IMPLEMENTED / NOT EXECUTED。** 本轮实际运行了指定的 CPU 合同/模型/梯度/规则/策略测试；没有启动 M1/M2/M3、正式训练、GPU、全量重导出或覆盖历史产物。旧版本历史验收不自动升级。

Schema 非别名映射、设计 §7.5/§17 冲突的用户裁定、版本前后值与失效边界见[设计一致性与迁移合同](kar-design-consistency.md)。本轮最终证据为 1348 passed、0 failed、0 errors、0 skipped，来自七组实际 JUnit；逐项测试名称保存在工作区证据包，不是从 commit message 推断。

## 数据、规则与输入（D01–D18）

路径以下均相对于 `src/riichienv/kar/`；测试路径相对于 `tests/kar/`。

| ID | 冻结职责 | 实现与接线 | 回归源码 |
|---|---|---|---|
| D01 | 实际协议、result-only 解码 | `mjsoul.py` | `test_mjsoul.py` |
| D02 | 单一 ruleset、主 cohort | `contracts.py`，导出/验收共同检查 | `test_contracts.py` |
| D03 | 完整重放、原生阶段合法性 | `audit.py` → `native_replay.py` | `test_audit.py`、`test_draw_obligations.py` |
| D04 | 观察者白名单与隐藏信息隔离 | `observation.py` → tokenizer | `test_observation_response.py`、`test_tokenize.py` |
| D05 | 完整 Schema、944 Embedding 行、62 状态槽 | `schema.py`、`token_schema.riichienv.v1.json` | `test_schema.py` |
| D06 | 历史序列化、分块与逻辑位置 | `tokenize.py`、`TokenHistoryCache` | `test_tokenize.py`、`test_token_cache.py` |
| D07 | Genbutsu/Suji/Ryanmen/物理去重 | `rule_features.py` | `test_rule_features.py` |
| D08 | 起家顺同分破平与相邻分差 | `rule_features.py` → ROUND_STATE | `test_rule_features.py` |
| D09 | 546 全局动作及红五变体 | `actions.py`、动作 manifest | `test_actions.py` |
| D10 | 完整动作执行、两步立直、原子多人响应 | `execute_candidate`、`execute_joint_responses` | `test_actions.py`、`core/test_live_agent.py` |
| D11 | 杠声明/提交及重放上下文 | `kan_transactions.py` → ledger/tokenizer/native bridge/settlement；`live_events.py` 延迟确认 | `test_native_replay.py`、`test_kan_preview.py`、`test_review_kan_transactions.py` |
| D12 | 21 类舍牌响应窗口 | `response.py` | `test_observation_response.py` |
| D13 | 只读取公开事实的响应 mask | `response_mask.py`，训练/推理共用 | `test_golden.py`、`test_observation_response.py` |
| D14 | 独立付款、终局与名次标签 | `payments.py` → `settlement.py` → `completion.py` | 对应 payments/settlement/completion 测试 |
| D15 | manifest/split/去重/恢复/Golden 排除 | `corpus_audit.py` → `acceptance.consolidate_shards` → `learning/export.py`；保留来源链与语义别名排除 | `test_corpus_audit.py`、`test_acceptance.py`、`test_acceptance_shards.py` |
| D16 | 真实来源绑定 Golden 核验 | `golden.py`、`golden_registry.v1.json` | `test_golden.py`；真实数据仍由执行验收提供 |
| D17 | 独立 train 游戏事件支持 | `audit.py`、`support.py`、`learning/calibration_job.dataset_event_support` | `test_audit.py`、`core/test_artifact_contracts.py` |
| D18 | 有界只读搬运与稀有场景定位 | `corpus_fetch.py`、`rare_search.py` | 对应 fetch/search 测试 |

## 模型与缓存（M01–M07）

| ID | 冻结职责 | 实现与接线 | 回归源码 |
|---|---|---|---|
| M01 | 五个独立 tile Embedding、40D 数值投影 | `learning/model.py` | `m1/test_model.py` |
| M02 | RMSNorm/QKV/RoPE/SwiGLU/零 dropout | `learning/model.py` | `m1/test_model.py` |
| M03 | 546 / 21 / 21×4 三头 FP32 logits | `learning/model.py` | `m1/test_model.py` |
| M04 | prefix/state/action 分块注意力与 padding | `learning/model.py`、`learning/batch.py` | `m1/test_model.py` |
| M05 | 真实 E/R 标签不进入 forward、候选分支隔离 | 输入/标签分离、共享前缀候选接口 | `m1/test_model.py`、`m1/test_losses.py` |
| M06 | 同一决策 KV 共享及失效检查 | `PrefixCache`、`FrozenModelPredictor` | `m1/test_model.py`、`core/test_frozen_inference.py` |
| M07 | Tiny/Small/Medium/Large 配方和精确参数断言 | `learning/config.py` → 模型构造 | `m1/test_model.py` |

## 监督、训练基础与诊断入口（L01–L10）

| ID | 冻结职责 | 实现与接线 | 回归源码 |
|---|---|---|---|
| L01 | 可信决策标签与独立有效位 | `learning/export.py` → `learning/losses.py` | `m1/test_dataset.py`、`m1/test_losses.py` |
| L02 | 输入/标签分离、名次 1..4 转 0..3 | `learning/export.py`、`Labels` | `m1/test_dataset.py` |
| L03 | 只读共享历史与全量导出 | `learning/dataset.py`、`export_dataset` | `m1/test_dataset.py`、`test_acceptance_shards.py` |
| L04 | 固定 train rho、masked CE、sentinel-safe gather | `learning/losses.py` | `m1/test_losses.py` |
| L05 | AdamW 分组、裁剪、warmup/cosine | `learning/train.py`，生产 trainer 复用 | `m1/test_training.py`、`production/test_training_contracts.py` |
| L06 | global sum、DDP 与累积等价 | `production_update`、`DistributedContext`、全局 sampler | `production/test_training_contracts.py` |
| L07 | 整条样本规则特征同步随机置空 | `learning/batch.py`、按 seed/epoch/game/row 生成 mask | `m1/test_model.py`、`production/test_training_contracts.py` |
| L08 | RNG/游标/优化器/调度器精确恢复 | `learning/checkpoint.py`、`production_checkpoint.py` | `m1/test_training.py`、`production/test_full_trainer.py` |
| L09 | 有界 CPU 更新/消融诊断，默认拒绝 | 历史 `learning/train.py`、`probe.py` 接口保留，不冒充全量训练 | `m1/test_training.py`、`core/test_authorization.py` |
| L10 | 有授权的数据绑定 CUDA 小样本预检 | `learning/gpu_preflight.py`；读取实际 manifest 后再次核对许可 | `production/test_review_job_guards.py`；本轮未运行 CUDA |

## 推理、标定与策略（P01–P12）

| ID | 冻结职责 | 实现与接线 | 回归源码 |
|---|---|---|---|
| P01 | 完整合法集合上的校准先验 | `probability.py` → `RuntimePolicy` | `core/test_probability_policy.py` |
| P02 | 固定 cal_fit/cal_check、三头温度 | `calibration_roles.py`、`calibration.py`、`calibration_job.py` | `core/test_calibration.py`、`production/test_selection_and_roles.py` |
| P03 | 多完整舍牌的 log p_min 门控及分布 | `gate_log_threshold` → `support_discards` → 完整 M/支持数直方图 | `core/test_calibration.py`、`production/test_diagnostic_distributions.py` |
| P04 | 冻结名次效用 | `runtime_contracts.py`、`support.py` | `core/test_probability_policy.py` |
| P05 | 响应边缘化与保守价值 | `candidate_value` | 核心概率/策略测试 |
| P06 | train-only 支持、-1.5、post-mask rho_U | `EventSupport` → mask → candidate value | `core/test_artifact_contracts.py`、`core/test_runtime.py` |
| P07 | S 总质量与 S 外逐动作概率保持 | `restricted_kl` | `core/test_probability_policy.py` |
| P08 | infinity/Rmax/tau_min/验证冻结选择 | `policy.py`、`TauSelectionPolicy`、`validation_policy.py`；没有合格收益保留 infinity | `production/test_selection_and_roles.py`、`core/test_artifact_contracts.py` |
| P09 | E1/E2、误差来源、certified/empirical/fallback | `BoundEvidence`、`one_sided_bounds`、`RuntimePolicy` | `core/test_runtime.py`、`core/test_probability_policy.py` |
| P10 | 完整分布采样与 RNG 审计 | `sample_complete`、agent/match 记录；certified 禁止 argmax | `core/test_runtime.py`、`core/test_review_runtime.py` |
| P11 | 模型到 RiichiEnv 在线 agent | `FrozenModelPredictor` → `KARAgent` → 完整动作执行及公开事件投影 | `core/test_frozen_inference.py`、`core/test_live_agent.py`、杠事务回归 |
| P12 | 门控/价值/KL/证据/实际概率变化 telemetry | `RuntimePolicy.decide` → match/replay/report | `core/test_runtime.py`、`production/test_paired_benchmark.py` |

## 生产闭环与入口（T01–T10、U01）

| ID | 冻结职责 | 实现与接线 | 回归源码 |
|---|---|---|---|
| T01 | 完整 Epoch CPU/CUDA FP32/BF16 trainer | `learning/trainer.py`，无诊断 update 上限 | `production/test_full_trainer.py` |
| T02 | 分布式全局计数、尾 batch、空 rank、失败协调 | sampler/distributed/production update | `production/test_training_contracts.py`、`test_planning.py` |
| T03 | quarter-epoch 全 validation 与独立模型选择 | `evaluate_complete` 报完整 validation；`cal_fit_loss_summary` 单独提供第三轮依据 | `production/test_selection_and_roles.py`、`test_planning.py` |
| T04 | 模型阶梯、真实成本、持久固定费预留、恢复锚点 | `budget.py` → 所有生产阶段 | `production/test_review_finance_and_resume.py`、`test_review_job_guards.py` |
| T05 | 完整发布身份、原子检查点与完成发布恢复 | `release.py`、`production_checkpoint.py`、`run_publication.py`、`deployment.py` | 完整 trainer/release 身份与恢复回归 |
| T06 | 自然频率分类、可靠性、威胁/rank 切片与分支指标 | `FullDiagnostics` → 完整验证与独立 cal_check | `production/test_statistics.py`、`test_diagnostic_distributions.py` |
| T07 | 游戏级 CI、效用校准与独立分支证据量 | `GameClusters`、21 事件分支区间；不当作统一误差证书 | `production/test_statistics.py`、`test_diagnostic_distributions.py` |
| T08 | 真正的 baseline/candidate 成对完整模型吞吐 | `benchmark_plan_from_records` → `benchmark_matches`；actual auxiliary coverage、replay、收据、运行时来源 | `production/test_paired_benchmark.py` |
| T09 | M1 完整消融 → 唯一 M2 → 条件第三轮 | `m2_prerequisites` → 登记 v3 → raw trainer 重验；扩模型必须有对应 Small 验证收益 | `production/test_stage_prerequisites.py`、`test_planning.py` |
| T10 | 固定 10000 pairs/四座/RNG/对手/费用/统计 | `paired_seed_manifest` v2 → 实测证据预算校验 → `run_m3`；不复用 test 墙作基准 | `production/test_jobs.py`、`test_paired_benchmark.py`、费用恢复回归 |
| U01 | 完整延迟导入 CLI 与元数据计划 | `__main__.py`、`planning.py`、`learning/jobs_cli.py`、orchestration CLI | `production/test_jobs.py`、`test_metadata_planning.py` |

## 不能由这张表替代的事项

- 完整训练阶段、GPU/BF16 硬件对照、全量数据重导出、真实标定/吞吐测量和 M3，均未在本轮运行；局部 CPU 单元回归不能代替这些阶段。
- 真实牌谱 Golden 不能由本轮合成回归源码替代；历史 M0 的证据范围和源码版本保持原记录。
- 设计没有要求统一逐状态误差证书估计器。已实现调用方提供有效界时的消费与回退；没有界时维持经验/不可用状态，不填零、不自动 certified。
- 构建/角色/标定/登记/基准合同不匹配时拒绝静默复用；代码补齐不会产生训练或评测许可。
- 58 项表是代码层覆盖；只有下面明确列出且本轮实际运行通过的项目标记 PASS。提交与发布状态以 Git 为准，本页不授予阶段运行许可。

## 本轮十五项设计一致性矩阵

执行基线 HEAD：`e24c24bd4b0ace681a2feaa093eede94754aafa2`，包含本轮未提交工作树补丁。PASS 的范围限于表内 CPU 验收，不表示棋力、真实完整数据或云端性能通过。

| # | 设计要求 | 实际执行证据 | 状态 |
|---|---|---|---|
| 1 | Small/Medium/Large 参数量 | [test_model](../tests/kar/m1/test_model.py)：实际构造并断言 5,239,435 / 14,790,795 / 26,536,587；另含 Tiny 对照 | PASS |
| 2 | 五张独立 Tile Embedding | 同文件检查五张 39×d 表、独立参数地址、944 总行数 | PASS |
| 3 | Numeric 40D 投影 | 同文件 hook 实际输入，核对 FP32 numeric/presence 拼接、40D、bias=False | PASS |
| 4 | 三头输出形状 | 同文件实际 forward/backward：546 / 21 / 21×4、FP32 | PASS |
| 5 | DECIDE / ACTION 隔离 | 同文件块 mask 真值、teacher ACTION 改变不影响策略输出 | PASS |
| 6 | 候选 ACTION 分支隔离 | 同文件共享 KV 与 dense 一致；改变中间候选不改变其余候选输出 | PASS |
| 7 | 三项 Loss / 安全哨兵 | [test_losses](../tests/kar/m1/test_losses.py)：实际 CE/梯度、非法索引拒绝、dummy 连图零梯度、固定 rho | PASS |
| 8 | DDP / 梯度累积等价 | [test_training](../tests/kar/m1/test_training.py) 的两进程本机 Gloo 梯度对照 + test_losses 非均匀 microbatch 累积；不调用阶段训练 | PASS |
| 9 | Genbutsu / Full Suji / Ryanmen Golden | [test_rule_features](../tests/kar/test_rule_features.py) 的 168 项确定性断言；不是全部真实牌谱 Golden 证书 | PASS |
| 10 | 同分顺位 / numeric presence | test_rule_features 固定起家排序和庄家轮换对照；[test_schema](../tests/kar/test_schema.py) 区分 present-zero 与 unknown | PASS |
| 11 | Response Mask 公开隔离 | [test_observation_response](../tests/kar/test_observation_response.py) 白名单/未来暗手扰动；[test_design_contracts](../tests/kar/test_design_contracts.py) 断言训练/推理同一个 mask 函数 | PASS |
| 12 | 规则特征随机置空 | test_model 核对整样本 flags、numeric/presence 同步清空、stage 保留、off 模式、不可变及 seed/顺序复现 | PASS |
| 13 | 受限 KL 概率质量 | [test_probability_policy](../tests/kar/core/test_probability_policy.py)：四种 tau 的 S 质量和 S 外逐动作概率保持 | PASS |
| 14 | E1/E2 单侧界 | 同文件固定随机种子的 150 组有限集合回归及 runtime 消费/回退单元；不生成真实状态误差证书 | PASS |
| 15 | 数据 / checkpoint 版本绑定 | [test_design_versions](../tests/kar/m1/test_design_versions.py)：57 项临时 SQLite、两类 checkpoint、发布的校验和有效旧合同拒绝 | PASS |

### 实际测试组

| 组 | 通过 | failed/errors/skipped |
|---|---:|---|
| 设计/Schema/计数/预算/登记/发布元数据 | 736 | 0/0/0 |
| 合成 M0 producer 与单/多分片合同 | 39 | 0/0/0 |
| 规则特征/Token/公开掩码/有界合法本局 | 400 | 0/0/0 |
| 现有 Golden 目标文件 | 31 | 0/0/0 |
| 模型/Loss/版本与合成序列化 | 93 | 0/0/0 |
| 真实两进程 CPU Gloo | 1 | 0/0/0 |
| KL/E1/E2/支持与静态校准单元 | 48 | 0/0/0 |
| **合计（不重复计入复跑）** | **1348** | **0/0/0** |

工作区证据目录：`kar-fix-reports/design-consistency-20261117`，含实际 argv/cwd、退出码、完整 stdout、JUnit、前后版本及逐项用例映射。源码里的 M0/M1/M2 标签或合成登记不表示运行了这些阶段；本轮没有生产更新授权，没有训练 launcher 调用。

### 仍为 IMPLEMENTED / NOT EXECUTED

| 范围 | 状态 | 原因 |
|---|---|---|
| 全量真实 M0 / 整个 Golden Registry 重新验收 | IMPLEMENTED / NOT EXECUTED | 本轮只执行列明的有限测试；历史证据不自动迁移 |
| 当前合同的真实数据重导出和旧训练产物迁移 | IMPLEMENTED / NOT EXECUTED | 不覆盖旧 manifest、SQLite、权重或账本 |
| 完整生产 optimizer-update / Epoch / 中断恢复 / M1 消融 | IMPLEMENTED / NOT EXECUTED | 梯度等价单元不代替阶段运行，且本轮禁止启动 |
| M2 正式训练与 M3 成对评测 | IMPLEMENTED / NOT EXECUTED | 按用户要求停止 |
| CUDA/BF16、云端吞吐与棋力验证 | NOT EXECUTED | 本轮仅 CPU；未使用 GPU 或付费资源 |

设计 §7.5/14.2 与 §17 的选型计数文字仍矛盾；本轮用户明确选择 `N_train_half_games`。代码链条按此裁定已验证一致，但不宣称正式文件本身已被勘误。Schema 仍是显式宿主物化，不能据模型形状一致而将旧设计 checkpoint 当作兼容输入。
