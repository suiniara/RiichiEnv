# KAR：离线振听、逐事件校准与训练夹具回归

审查基线 `f70684ef7aa69e56e6a530dcca13326fbf2c83b5`（`kar/riichienv-v1-20261117`，开工时本地与 fork 一致）。保留前一轮冻结决策机会及其它已有变更；同步活动[主设计 §5.4／§11.3](KAR-v1.2.2-DESIGN.md)。没有改变架构、动作／响应分类、公开 mask、训练损失、固定 rho 或受限 KL 公式。

## 1. 根因与实现

- **振听**：[四人原生舍牌](../riichienv-core/src/state/mod.rs)和 [MJAI 事件重放](../riichienv-core/src/state/event_handler.rs)共用候选及无役和牌形的同巡振听更新。原生在同一响应窗被正常摸牌或成立鸣牌关闭时处理未使用的合法 Ron；没有合法 Ron 的无役和牌形不是见逃行为，不产生 Pass。自己的舍牌清同巡，不清立直振听。[Python 桥](../src/riichienv/kar/native_replay.py)不再缓存振听位；只有经原生预览并获真实提交证据的抢杠窗，才调用 [Rust 薄接口](../riichienv-python/src/env.rs)登记未使用的 Ron。不从实际赢家构造候选，不向公开 mask 注入私有信息。
- **校准**：[统计](../src/riichienv/kar/statistics.py)、[校准](../src/riichienv/kar/calibration.py)与 [evaluation](../src/riichienv/kar/learning/evaluation.py)对每个事件使用同样的公开允许总体、标量掩码概率和二元 NLL／Brier／10-bin ECE。报告合格总行数、正例行数和正例独立 game 数。[门控](../src/riichienv/kar/calibration_quality.py)只用正例 game 满足该头最小证据条件，不改变真实分支 Rank，也不放宽 train≥500 或既有 failed／unavailable 保护。原 21 类真实类别切片单列为分类诊断，不参与部署准入。
- **训练夹具**：[FullSyntheticDataset](../tests/kar/production/test_full_trainer.py)沿用 `game_count_contract` 和 `learning_contract()`；新增默认、无更新用例，实际进入 `_dataset_contract` 和 `complete_release_contract`。完整 epoch、非有限值拒绝和完成恢复测试仍只在显式参数更新许可下执行。这不会把合同自洽等同于更新／恢复已验收。

## 2. 兼容与重算

- `native_replay_version=prefix_flags_bridge_v3_native_furiten`，`input_pipeline_version=semantic_public_clock_frozen_decisions_v4_furiten`，`action_label_version=explicit_and_unique_next_draw_pass_v2_furiten`。缺少新接口的旧 Python 原生扩展拒绝回放，必须重新构建。事件顺序、公开时钟、SQLite 表布局及 Token Schema 不变；训练配方通过输入版本自动变化。
- 旧产物受影响的是无役和牌形后的合法集合、实际可证 Pass、其后本家 `SELF_PASS`／前缀和行序号。两个缺陷叠加时还须含前一轮决策冻结。独立支付或完局通过不能证明样本正确。不声明所有旧游戏都含此反例，也未扫描真实全库。以当前合同交付时，应重新审计和重放原始记录，重新导出 SQLite、缓存、索引和 manifest；不能通过剔除一个目标行、改旧版本或 hash 来等价迁移。原始记录不需要修改。
- `preregistered_response_branch_quality_v2_ovr`、`preregistered_calibration_quality_v3_event_ovr`、`heldout_temperatures_log_gate_v6_event_ovr`、`whole_game_cluster_statistics_v4_event_ovr`、`restricted_kl_runtime_v3_event_ovr` 和 `calibrated_policy_identity_v3_event_ovr` 替代旧语义。正例切片没有负例信息，不能换标签变成逐事件校准。只可用匹配当前输入和冻结权重的可核验 logits／mask／标签，依新预登记标准重算独立 `cal_check`，其后重算有效支持与质量直方图；旧训练产物、校准、精确恢复与发布不能冒充新绑定。该边界不授予正式训练或真实数据重建许可。

## 3. 回归范围与文件

- [完整合法墙／重放／导出](../tests/kar/test_native_furiten_export.py)比较无役 `1s`、紧接的可和形 `4s`、自己摸牌／舍牌与再次可 Ron 的 `4s`，通过原始结果索引定位同一窗口。完整半庄同时检查假 Pass 及私有 marker 消失、真正被谢绝的 Ron 仍可产生 Pass，并调用实际 `GameWriter.finish`。另外用已有[普通见逃／立直振听／抢杠关闭](../tests/kar/test_native_replay.py)、[原生杠预览](../tests/kar/test_kan_preview.py)及[冻结时点](../tests/kar/test_decision_time_parity.py)回归守住边界。
- [逐分支质量测试](../tests/kar/core/test_branch_quality.py)区分未观察、规则 mask、负例与独立正例 game，校验饱和补事件的 log-space NLL；[质量／支持对拍](../tests/kar/production/test_value_support_mass.py)用固定的 900／100 例，实际走 calibration、分块 evaluation 合并和 runtime，不以总体平均放行。
- [版本表](../src/riichienv/kar/versioning.py)、[运行时与有限策略合同](../src/riichienv/kar/runtime_contracts.py)、[导出](../src/riichienv/kar/learning/export.py)与[Python 原生存根](../src/riichienv/_riichienv.pyi)随实现同步；[输入版本](../tests/kar/test_input_versions.py)和[语义版本](../tests/kar/test_semantic_versions.py)检查旧产物拒绝。更新相应模块 README；原 `design-sync-1` 的历史校验及分发快照不覆盖。

## 4. 实际验证

旧对照取开工时的原生扩展和 `f70684e` 的回放桥／校准代码，保留上一轮的冻结决策时点。玩家 3 的 `4s`（规范事件 9），在线玩家 1 的状态为同巡振听、无响应候选；旧离线为未振听、`(221,544)`，修复后与在线相同。旧版三个针对性断言失败，即便完局／独立结算及 SQLite 写入成功；旧 ECE 实测总体约 0、真类 NEXT/CHI 为 .1/.9，CHI 仅因合成 `max_ece=.2` 被拒。

同一 12 局合成半庄，旧导出 1014 行，假 Pass（事件 9）与真正见逃（事件 17）都留有 `SELF_PASS`；修复后导出 1013 行，Pass 和 marker 都只保留事件 17。`replay_game`、完整半庄证书、独立结算及 SQLite `GameWriter.finish` 均通过。没有读取真实全库或借用人工振听标志。

同一 900／100 概率反例、不变的合成门槛：新版逐事件 ECE 约 `1.51e-14`（NEXT）和 `1.42e-15`（CHI），两个二元 Brier 均约 `.09`，各自总体都是 1,000，正例／独立正例 game 为 900／900 与 100／100。原类别条件诊断仍为 `.1/.9`，不参加门控。合成校准重载、实际 evaluation 合并及 runtime 使用同一有效集合 `{NEXT_DRAW, CHI_P1}`；有负例但无正例的分支仍不可用。此结果不是生产概率质量或棋力验收。

| 最终运行 | 通过 | 失败 | 错误 | 跳过／忽略 |
|---|---:|---:|---:|---:|
| 默认 `pytest tests/kar`，无参数更新（含全部上列新增回归） | 2897 | 0 | 0 | 19 |
| `cargo test -p riichienv-core --lib --release --locked --offline` | 78 | 0 | 0 | 0 |

默认套件耗时约 373 秒。19 个跳过包括 15 个需要显式参数更新许可的用例（含完整 epoch、非有限值拒绝、完成 checkpoint 恢复三项），和 4 个不兼容的私有旧导出夹具。默认无更新的 `test_full_synthetic_fixture_satisfies_current_production_contract` 实际通过；三条更新路径没有执行、不计为通过。旧对照 0 通过／3 失败／0 跳过，另 1 个不适用于旧版接口的用例未选入，不计入最终。Ruff、Ty、Rust 格式、Git 空白检查及文档 28 个版本值／76 个本地链接检查通过。

测试用隔离的重新构建原生扩展；仓库原已安装扩展未替换，用它实际验证新桥会拒绝旧接口。最终源码 SHA‑256 `c0062fe1681bc7e6dfd6a480296fb57e8645a422092d530e55481ad3f7e53347`，测试扩展 SHA‑256 `f3b452b8173bb5d423c5a11e86125682ed44814e768531a755def2fc6f678592`。原始日志、JUnit、同窗候选、CPU 守卫与逐文件 SHA 见随附证据包。没有运行参数更新、正式训练、真实校准、GPU、真实数据重建，也未覆盖已有训练产物、提交或推送。
