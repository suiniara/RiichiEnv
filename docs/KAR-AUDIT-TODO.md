# KAR Audit To-Do List

> 审计基准提交：`kar/riichienv-v1-20261117` @ `b677d3e`
> 目标：按优先级逐项修复正确性缺陷、打通端到端链路、消除算法/统计/工程瓶颈，并规划模型能力重构。

---

## P0 — 致命正确性缺陷（阻断编译或端到端主链路）

- [x] **P0-1：修复 `riichienv-core` 单元测试编译失败（`types::Meld` 缺 `PartialEq`）**
  - 修复：为 `riichienv-core/src/types.rs` 中 `Meld` 派生 `PartialEq, Eq`。
  - 验证：`cargo test --manifest-path riichienv-core/Cargo.toml --no-default-features` 全部 110 项单元测试 + 全部集成测试编译并全绿。

- [x] **P0-2：修复真实 MJAI 牌谱回放切入原生引擎继续模拟（`prepare_replay_continuation_wall` 100% 失败 + 重复 tile ID + 状态字段缺失）**
  - 修复：
    1. 在 `riichienv-core/src/replay_context.rs` 的 `ReplayContext` 新增本局 136 张物理牌分配表 `allocated_tiles: [bool; 136]` 及 `allocate_replay_tile` / `take_replay_hand_tile` / `resolve_replay_called_tile` / `peek_replay_hand_tiles`，确保 `StartKyoku`、`Tsumo`、`Dora`、`Dahai`、`Pon`、`Chi`、`Kan`、`Ankan`、`Kakan` 保持唯一物理 `0..135` 牌 ID，消除与 `dora_indicators` 的冲突及 `actions.py:296` 的虚假歧义。
    2. 在 `riichienv-core/src/state/event_handler.rs` 与 `replay_context.rs` 补齐 `discard_from_hand`、`discard_is_riichi`、`riichi_declaration_index`、`melds[j].from_who`、`wall.pending_kan_dora_count`（`sync_replay_pending_kan_dora`）和包牌 `pao`（`record_replay_pao`）；同时为 `apply_log_action` 补充物理牌/同牌面兼容移除辅助函数。
  - 验证：新增 Rust 单测 `mjai_replay_continuation_wall_allocates_unique_physical_tiles_and_tracks_state` 与 Python 端到端回归测试 `test_p0_2_mjai_replay_candidate_branches_succeed_with_duplicates_and_dora_overlap`（含重复牌、赤 5、手牌与宝牌指示牌同牌面，分支 100% `PASS`）。

- [x] **P0-3：修复 `candidate_branches.py` 中原始 Rust `ryukyoku` 字符串未规范化导致的崩溃**
  - 修复：在 `src/riichienv/kar/response.py` 的 `_END_REASONS` 补全原生 Rust 名称别名 `"nagashimangan"`、`"sufuurenta"`、`"suukansansen"`、`"sanchaho"`。
  - 验证：`test_p0_3_resolve_discard_response_accepts_raw_rust_draw_reasons` 覆盖全部 4 种流局并通过。

- [x] **P0-4：打通生产环境有限 `τ` KL 重排死锁（真实采集 held-out 候选动作条件校准指标）**
  - 修复：
    1. 在 `src/riichienv/kar/calibration.py` 新增 `CandidateBranchCalibrationRow` 与 `_candidate_branch_metrics`，支持在 `cal_check` 严格隔离计算候选动作条件响应/顺位指标（绝不泄露至 `cal_fit`）。
    2. 在 `src/riichienv/kar/calibration_quality.py` 实现 `_evaluate_candidate_conditional_validation`，按实测指标与预注册阈值判定 `"passed"` / `"failed"` / `"unavailable"`（无分支数据时保持原有 `"heldout_candidate_branch_metrics_not_collected"` 语义）。
    3. 在 `src/riichienv/kar/learning/inference.py` 新增 `FrozenModelPredictor.score_candidate_branch_rows`（与部署一致的单前缀 FP32 算术）；在 `src/riichienv/kar/learning/calibration_job.py` 新增 `_collect_cal_check_branch_rows` 采集并归档 `candidate_branch_logits`。
    4. 在 `src/riichienv/kar/learning/validation_policy.py` 与 `src/riichienv/kar/learning/deployment.py` 对齐 `runtime.py`，在有限 `τ` 规划/部署前即校验 `candidate_action_conditional_validation == "passed"`。
  - 验证：`test_p0_4_end_to_end_calibration_job_collects_heldout_candidate_branches` 与 `test_production_candidate_branch_validation_enables_finite_tau_reranking` 全链路验证非诊断生产 `RuntimePolicy` 真实执行有限 `τ` 重排。

- [x] **P0-5：恢复 `loss_sums()` 中的响应掩码合法性检查**
  - 修复：将 `src/riichienv/kar/learning/losses.py` 的响应损失统一走 `_masked_ce(..., weights=labels.response_weight)`，恢复空掩码与越出规则掩码的 `ValueError` 校验（含 `response_weight == 0` 边界）。
  - 验证：`test_p0_5_loss_sums_rejects_empty_or_excluded_response_masks` 通过。

---

## P1 — 高危正确性缺陷

- [x] **P1-1：修复 `native_riichi_baseline_v2` 的 3 处逻辑错误**
  - 修复（`src/riichienv/kar/candidate_branches.py`）：
    1. `_dora_base` 区分 4 风（`% 4`）与 3 元（`% 3`）。
    2. `_discard_efficiency_score` 修正 `+ 0.6 * isolated` 符号并将 `counts[base]` 纳入数牌 `neighbours`。
    3. `_choose_own_candidate` 去掉 14 张 `WaitAct` 恒为 `False` 的 `observation.is_tenpai` 死条件，直接按 `candidate.spec.riichi` 加分。
  - 验证：`test_p1_1_baseline_dora_wrap_isolated_bonus_and_riichi_preference` 通过。

- [x] **P1-2：修复实时对局 `agent.py` 遍历 `HashMap` 导致的 `match_sha256` 非确定性**
  - 修复：`src/riichienv/kar/agent.py` 改为 `for seat in sorted(observations):`。
  - 验证：`test_p1_2_functional_match_hash_is_deterministic` 通过。

- [x] **P1-3：修复验证评估的“指标裂脑”与 `use_rule_features=False` 顺位切片坍塌**
  - 修复：`src/riichienv/kar/learning/evaluation.py` 重写 `_factual_validation_view`，按源决策真实槽位定位（不误过滤 `action_valid=False` 行），恢复带分支源决策上的事实辅助标签用于 `FullDiagnostics`，并向 `add_diagnostics` 传入未掩码的原始 `batch.inputs.fields` 作为 `public_fields`。
  - 验证：`test_p1_3_evaluation_preserves_current_rank_when_rule_features_disabled` 通过。

- [x] **P1-4：修复 `benchmark_candidates()` 在非切牌候选上崩溃**
  - 修复：`src/riichienv/kar/learning/benchmark.py` 仅筛选 `< 148` 的合法切牌候选并扫描首个含切牌候选的验证行。
  - 验证：`test_p1_4_benchmark_candidates_filters_non_discard_actions` 通过。

---

## P2 — 中危缺陷、统计/算法设计改进与工程性能修复

### P2-A：中危正确性修复
- [x] **P2-1**：补齐 `src/riichienv/kar/learning/calibration_job.py`（`dataset_event_support`）缺失的 4 项分支归属一致性校验，并核对 SQLite 实际扫描计数与 manifest `candidate_branch_counts`。
- [x] **P2-2**：修复 `src/riichienv/kar/support.py`（`support_from_train_games`），新增可选 `dataset_manifest` 绑定与一致性校验以读取导出数据集的 `candidate_branch_counts` 与 `manifest_sha256`。
- [x] **P2-3**：修复 `src/riichienv/kar/selection_quality.py` 硬编码 `split_salt`，支持传入 `main_split_salt` 并由 `deployment.py` 透传。
- [x] **P2-4**：修复 `src/riichienv/kar/learning/train.py` 学习率调度在 `step = warmup - 1` 与 `step = warmup` 连续重复峰值 `1.0` 的问题（验证见 `test_p2_4_scheduler_hits_peak_once_and_decays_to_ten_percent`）。
- [x] **P2-5**：修复 `src/riichienv/kar/learning/orchestration.py` `run_m2` 恢复读取未包在 `context.coordinate()` 的 DDP 挂起隐患。
- [ ] **P2-6**：`simulation.py` 孤儿回放与 `hostname` 绑定——注：`test_m3_crash_recovery.py` 明确将“拒绝覆盖孤儿回放”作为显式契约保留；后续如确需云主机抢占迁移续跑，再另行设计显式 `reconcile_orphan_replays` 工具与主机无关运行时契约。
- [x] **P2-7**：复核 `src/riichienv/kar/learning/precision.py`——确认 PyTorch 2.x 的 CPU SDPA 开关确由 `torch.backends.cuda.{flash,math}_sdp_enabled()` 暴露，属于预期实现（无需破坏性修改）。
- [x] **P2-8**：在 `riichienv-core/src/state/legal_actions.rs` 补全立直后暗杠的面子/雀头拆分不变性检验（`ankan_changes_mentsu_decomposition`），并通过新增 Rust 单测 `post_riichi_ankan_rejects_mentsu_decomposition_change_even_when_waits_match`。
- [x] **P2-9**：同步 `docs/KAR-v1.2.2-DESIGN.md` §0 活动版本表与 §12.2 分支抽样/推理合同描述，消除文档漂移。

### P2-B：算法与统计设计改进
- [x] **P2-10**：在 `riichienv-core/src/state/mod.rs` 与 `riichienv-python/src/env.rs` 新增 `prepare_observer_continuation_wall(observer, seed)`（固定本家手牌与已立直对手手牌，重采样其余对手暗手与未知牌山以边缘化 $P(H_{-i}\mid X)$），并在 `candidate_branches.py` 暴露 `resample_opponent_hands` 开关（验证见 `test_p2_10_and_p2_11_observer_hand_resampling_and_empirical_bernstein`）。注：用已训练 `π₀` 替代启发式续演与软标签聚合需配合新版本数据重新导出。
- [x] **P2-11**：在 `src/riichienv/kar/selection_quality.py` 新增有限样本无分布假设的 Maurer–Pontil 经验 Bernstein 界 `empirical_bernstein_gain_bound`（在低配对方差下较最坏方差 9 的 Hoeffding 半径缩小 ~2×）。注：解耦 `cal_fit` 对局数上限与 CRN RNG 重构涉及 M2 计划版本升级，待按需启用。
- [x] **P2-12**：在 `src/riichienv/kar/support.py` 的 `candidate_value` 新增 `unsupported_fallback="neutral_draws"` 选项，使流局满贯/中途流局/荒牌等非放铳稀缺分支不再因 `-1.5` 惩罚被策略自毁（默认保留 `"worst_case"` 兼容旧契约测试）。
- [x] **P2-13**：在 `src/riichienv/kar/calibration.py` 新增连续 1D 黄金分割温度优化 `optimize_static_temperature`（`select_static_temperature(..., "continuous")`），在 `src/riichienv/kar/statistics.py` 新增稀缺事件等质量自适应分箱 `quantile_ece`。

### P2-C：工程与性能优化
- [x] **P2-14**：修复 `src/riichienv/kar/learning/dataset.py:204` SQLite `branch_samples` 查询补全 `source_game_id=? AND source_decision_row_id=?`，命中复合索引消除每次 `__getitem__` 全表扫描。
- [ ] **P2-15**：压缩 `branch_samples` SQLite 存储结构（仅存 139 字节 `ACTION` token 增量，而非重复存完整上下文）——涉及 `DATASET_VERSION` 表结构迁移，建议配合下一次数据重导执行。
- [x] **P2-16**：在 `src/riichienv/kar/learning/trainer.py` 使 `GlobalEpochSampler` 长度感知分支展开倍数 `1 + branch_count`，消除 DDP 显存尖峰与负载倾斜。
- [x] **P2-17**：消除推理与评测主要冗余开销：
  - `src/riichienv/kar/tokenize.py` 新增 `encode_action_token`，`inference.py` 与 `benchmark.py` 不再每候选复制 350 KB 前缀缓冲；
  - `src/riichienv/kar/learning/inference.py` 用不可变 `PublicSnapshot` 引用快速路径消除每步 2 次全日志 `asdict` + JSON + SHA-256；
  - `src/riichienv/kar/learning/deployment.py` + `inference.py` 复用 `_verified_weights` 身份校验，消除 M3 中 80,000 次全量 CPU 权重拷贝哈希；
  - `src/riichienv/kar/runtime.py` 在 `RuntimePolicy.__init__` 缓存校准/分支报告并用紧凑 `branch_quality_summary` 替代每手遥测中 ~8 KB 的重复嵌套报告。
  - 剩余：跨巡 Block-0 增量 KV Cache 属于模型推理内核升级，归入下一阶段。

---

## 附：顺带修复的上游既有破损测试夹具

- [x] 修复 `tests/kar/core/test_runtime.py`（2 处生产态 `CalibrationArtifact` 夹具缺 `inference_device_type="cpu"` / `validation_manifest_sha256`）
- [x] 修复 `tests/kar/production/test_full_trainer.py`、`test_training_startup.py`、`test_review_job_guards.py`（13 处 `DecisionDataset` mock 未接收 `include_candidate_branches` 参数导致原始 HEAD 即报错）

---

## P3 — 模型架构与麻将棋力重构（中长期，需重新训练与升级模型版本）

- [ ] **P3-1：按局重置上下文**：事件流仅保留本局历史，跨局信息压缩为紧凑的局况/顺位/对手风格摘要 token，将序列长度从 ~2,000 降至 ~200。
- [ ] **P3-2：强化手牌与动作归纳偏置**：
  - 为 34 种牌增加花色 + 序数结构编码（或 1D 卷积/局部邻接编码）以及张数 0–4 离散 Embedding；
  - 在观测特征中补充向听数、进张/有效牌、听牌/待牌、基础役种/打点信息；
  - 为 395 个非切牌动作（吃/碰/杠/荣和）补全牌面与动作类型因子分解，替代单调查表与 `LEGAL_SET` 简单均值池化。
- [ ] **P3-3：增强动作评估与训练范式**：
  - 在 `ACTION` 反事实分支显式提供切牌后手牌状态差分；
  - 提升训练轮次并引入对手听牌/手牌辅助监督头，探索自博弈 RL（PPO/IMPALA）或将价值决策扩展至副露与多步规划。
