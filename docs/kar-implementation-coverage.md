# KAR 实现覆盖对照（代码层，2026-11-16）

逐模块职责、接口、数据合同、使用方式及测试位置见 [72个模块 README 总索引](kar-modules/README.md) 与 [learning 子包索引](kar-modules/learning/README.md)。

## 范围与状态

宿主为 **RiichiEnv**；依据 `KAR-v1.2.2-DESIGN.md` 及已批准的 `kar-riichienv-v1` 完整 Schema、雀魂 mode12 主 cohort、`N_train_half_games` 模型阶梯适配。Kanachan 仅保留动作编号参考。

下表沿用原 58 项编号，列出实现入口及回归源码。**已建立实现/接线不等于已通过运行验收，也不表示代码不可能再有缺陷。** 本轮只编辑代码、测试源码、文档并执行 Ruff / Ty / diff 静态检查；没有执行 pytest、模型、训练、预检、标定、仿真、对局或远端操作。旧版本的历史验收不自动升级为本次源码的验收。

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

- 当前源码的实际测试执行、GPU/CPU 对照、全量数据重导出、完整标定、吞吐测量和 M3，均未在本轮运行。
- 真实牌谱 Golden 不能由本轮合成回归源码替代；历史 M0 的证据范围和源码版本保持原记录。
- 设计没有要求统一逐状态误差证书估计器。已实现调用方提供有效界时的消费与回退；没有界时维持经验/不可用状态，不填零、不自动 certified。
- 构建/角色/标定/登记/基准合同不匹配时拒绝静默复用；代码补齐不会产生训练或评测许可。
- 本页仅记录代码层覆盖；提交与发布状态以 Git 历史为准，不由本页授予运行许可。
