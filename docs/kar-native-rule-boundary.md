# KAR 原生规则边界与产物迁移

本次改动以 `7626166246ec3ed698d5e6d95b27b2068afbe604` 为改动前基线，围绕 KAR 固定的雀魂四人赤牌半庄合同，把规则计算和状态转移收敛到 Rust。共享内核通过显式 `GameRule` 保留平台差异；Python 保留原始记录解码、证据核对、模型历史编码、监督、训练和产物管理。活动规范仍为 [KAR-v1.2.2 / RiichiEnv V4 主设计](KAR-v1.2.2-DESIGN.md)，本文说明实现边界、行为变化和迁移要求。

这次调整同时修正 KAR 雀魂链路的两项规则不匹配：本家碰加暗手第四枚的单骑形式听牌，以及西入后普通荒牌达到终止点数却继续连庄。天凤对这两个案例允许不同结果，不能把雀魂修复套到所有四人规则或无规则参数的旧 API。旧 KAR 审计结论、导出和发布身份需要按显式雀魂规则重新验证。

## 1. 当前职责

| 责任 | Rust 实现 | Python 保留的工作 |
|---|---|---|
| 和牌支付、流局支付、残余供托、连庄与半庄终止 | [`rules::settlement`](../riichienv-core/src/rules/settlement.rs)，四人在线状态机按其 `GameRule` 调用，KAR 审计显式选择雀魂 | `payments` 校验输入类型并适配独立评估事实；`settlement` 重建审计事实、比较来源；`completion` 检查完整性证据 |
| 观察者可见事实 | [`ObserverState`](../riichienv-core/src/observer.rs)：本家手牌、公开河与舍牌历史、副露、立直、分数、阶段和杠事务 | `ObserverLedger` 保存历史行、公共语义时钟、cutoff、来源及本家 Pass 确认证据，冻结 `PublicSnapshot` |
| 公开响应可能性 | [`public_rules`](../riichienv-core/src/public_rules.rs)：显式公开上下文到 21 位 mask | `response_mask` 校验完整候选 ID，提取白名单字段，调用绑定并检查有效 target |
| MJAI 重放规则上下文 | [`ReplayContext`](../riichienv-core/src/replay_context.rs)：一发、双立直、岭上和已证明未使用的合法荣和机会 | `NativeReplayBridge` 校验接口版本、预览及 prepare/after_apply 配对，不回写规则 flag |
| 完整立直候选与执行 | 原生 Observation 枚举立直舍牌；`step_riichi_discard` 原子执行完整动作 | `actions` 保留 546 ID、摸切/手切和赤牌身份、原生签名与过期检查 |
| 在线规则事件与终局揭牌 | `GameState.rule_log` 输出声明、提交、结算及公开揭示 | `LiveEventProjection` 处理事件批次、多荣和组、公开时钟及本家 Pass receipt，不补造暗杠声明或重新判断听牌 |

这三个 Rust 接口的权限不同，不能互换：`ObserverState` 不接收 `GameState`，只归约输入事件并保留选定观察者可见的事实；公开 mask 不接收完整事件流或环境对象；`ReplayContext` 属于原生状态机，可以在合法性审计中使用其真实手牌上下文，但它返回的真实合法响应不能进入公开 mask 或模型前缀。

自动建立抢杠响应窗口的原生重放范围是 KAR 四家手牌完整的记录。遮蔽 MJAI 的 `?` 手牌不能用于推断未知 Ron；本轮也未新增仅本家手牌完整时的本家自动抢杠响应能力，不能将完整信息审计的支持范围扩大为所有公开客户端。

## 2. 独立审计的含义

“独立结算”指计算不依赖服务器的番符、支付差额、header 终分或 `apply_event` 写入的原生分数。审计从原始牌张和事件前缀重建 `WinFacts`，调用 Rust `HandEvaluator` 得到和牌评估；涉及听牌/等待和回合推进时显式选择 `MJSOUL_RULESET.engine_rule()`，再通过共享支付与终局内核计算结果。服务器结果和下一局起始分数只用于比较，缺证据仍为不可核验，矛盾仍为失败。

因此审计仍有独立的**事实来源和比较过程**，但不再把另一份 Python 付款公式或终局规则当作第二个权威实现。共享内核能防止在线和审计的规则分叉，却不能单靠二者结果相等证明内核正确；已知规则案例、守恒检查、真实原始结果和明确反例仍是必要证据。`completion` 也继续检查完整起局、回合链、事件消费、独立结算覆盖和结束来源，不能仅凭 Rust 的 `match_end` 布尔值签发完整半庄证书。

保留在 Python 的来源解码、证据完整性、Unknown、FAIL/BLOCKED 分类、历史位置和训练产物管理是不同职责，不应为了减少 Python 文件数量而并入规则状态机。预算、恢复、注册和发布的幂等性问题不属于本轮规则边界改动；本轮不声称修复这些问题。

## 3. 平台差异与 KAR 雀魂修复

依据为[雀魂官方段位战规则](https://mahjongsoul.com/news/46/)的共同详细规则和四人规则，以及[天凤官方手册](https://tenhou.net/man/#RULE)的四人段位战规则、延长局结束补充说明。雀魂的朋友场或大会设置不自动属于本合同，天凤数据也不因此可混入 KAR 主训练集。

| 本轮涉及的条件 | 天凤 / 原生默认 | KAR 雀魂四人 | `GameRule` 开关 |
|---|---|---|---|
| 本家已有 `111m` 碰，暗手单张 `1m` 等待成雀头；纯手未使用四枚 | 官方允许这种第五枚形式听牌 | 官方排除本家持四枚的单骑空听形式 | `allow_fifth_copy_tenpai`：`true` / `false` |
| 已西入，普通荒牌结算后他家达到三万点、庄家听牌但不是第一 | 连庄优先，继续；庄家达标第一的和了止/听牌止仍另行判断 | 按达标条件结束；官方明确的庄家双荣和连庄例外保留 | `extension_dealer_repeat_priority`：`true` / `false` |

两个开关在 `GameRule::default_tenhou()` 与 `Default` 中均为 `true`，在 `GameRule::default_mjsoul()` 中均为 `false`。旧序列化规则缺少它们时按兼容的 `true` 恢复；KAR `Ruleset` 显式保存 `false` 并纳入规则 hash。共享公式和状态机意味着共用实现，不意味着各平台必须得到相同结果。

### 3.1 雀魂的本家四枚单骑与旧 API 兼容

例：暗手为 `1m 234567p 789s`，本家已有 `111m` 碰。原 KAR 沿用只检查纯手枚数的判断，会把该牌姿算为听牌；这符合上述天凤定义，却不符合 KAR 的雀魂目标，会影响不听罚符、终局揭牌和相关监督。不能仅用“第五枚不存在”否定天凤对形式听牌的明文定义。

Rust 的显式接口为 `is_tenpai_with_rule(&GameRule)`、`get_waits_u8_with_rule(&GameRule)`、`get_waits_with_rule(&GameRule)`。雀魂配置按本家 `full_hand`（暗手加成立副露）约束等待；天凤配置按纯手计数。无规则参数的 `is_tenpai/get_waits_u8/get_waits` 保留旧兼容语义。高层 Python `HandEvaluator.is_tenpai(rule=...)`、`get_waits(rule=...)` 选择相应原生接口，KAR 显式传雀魂规则；不能通过改变无参默认值偷偷迁移所有调用方。

这些接口不按全场公开牌或真实牌山删除形式听牌。自己的碰只有三枚且暗手可使用第四枚时仍可等待；外部已见完的空听也不能据此排除雀魂合法立直。

### 3.2 雀魂西入达标终止与天凤连庄优先

例：西一局庄家为座位 0，结算前点数为 `[25000, 29500, 22500, 23000]`，座位 0、1 听牌。普通荒牌支付后成为 `[26500, 31000, 21000, 21500]`。原 KAR 把达标终止与“不连庄”绑定，因庄家听牌而继续；这是天凤应保留的结果，却会改变雀魂目标下的整场长度及最终名次标签。

共享 `round_transition(context, end, &GameRule)` 显式区分和牌、普通荒牌和途中流局，并读取平台开关。该例在雀魂下结束，在天凤/default 下继续；原有庄家第一和了止/听牌止、途中流局及平台规定的例外单独处理。KAR `settlement` 和 `completion` 调用 Python `native_round_transition(..., rule=MJSOUL_RULESET.engine_rule())`，不依赖原生通用默认值。

共享结算内核覆盖四人在线自摸、整组荣和、流局满贯、不听罚符、回合推进及终局供托，并保留显式规则选择。三人状态机仍走原路径，本轮没有迁移或验收三人雀魂的上述差异，不能声称其规则已修复。

## 4. 观察、事件与规则的信任边界

### 观察者状态

Rust `ObserverState` 接收规范事件，不查询全局牌山、对手真实合法动作或对手暗手状态。起局只保留本家手牌，他家摸牌输出 `?`，他家暗杠的赤牌组成按原有可见性合同隐藏。状态字段由 Rust 持有；Python 读取的是分离的不可变投影，不能通过给 `ledger.hand`、`scores` 或规则 flag 赋值来改变原生事实。

它是可见性与事实转移边界，不是所有来源真实性的证明器。`hora/ryukyoku` 中的 `public_hand(s)` 必须先由在线原生结算或离线来源审计证明可以公开，投影器不会从隐藏牌中自行补全。终局后才进入公开历史，已经冻结的旧快照不回写。公共时钟和私有 Pass 的因果来源仍由 Python 历史层管理，确认前不能把 Pass 写成模型可见事实。

### 公开响应 mask

Python 适配器仅传候选 base34 牌种及立直位、actor、公开舍牌历史和当前河、副露种类、公开立直接受位、剩余摸牌数、历史完整位与规则开关。它先检查完整动作 ID 为 `0..147`，拒绝 bool、越界或非整数，也拒绝 pending-kan 快照。原生函数没有他家手牌、未来事件、真实响应标签或私有振听参数。

21 位的 `true` 仍表示“公开事实尚未排除”，不表示对方实际有合法动作。历史不完整仍保守保留 Unknown，已知中张舍牌等否定证据仍生效；确定途中流局不能压掉优先荣和。全空 mask 和被 mask 排除的有效 target 仍报错，不依据标签补回允许位。语义保持 response mask v3，迁移到 Rust 本身不改变类别次序或公开允许集。

### 在线规范事件

四人 `rule_log` 在原生状态机输出完整的暗杠声明与 `kan_commit`，以及和牌、荒牌和九种九牌实际应公开的手牌、分数和终局供托。legacy `mjai_log` 接口保留；KAR 在线协调使用 `rule_log`，不把延迟的旧日志行为当作 Python 补声明的理由。

`rule_log` 是协调层的完整事件来源，起局和摸牌可能仍带原生私有牌张；它不是可直接喂给任意观察者的模型输入。四份账本必须分别通过 Rust 的可见性投影。Python 仍对跨批多荣和进行分组并验证分数链、处理 receipt，但不调用 `HandEvaluator` 决定在线荒牌揭牌，不从此前全知协调对象恢复赢家手牌，也不自行分配终局供托。

## 5. 活动版本与不变的结构

| 合同 | 本轮活动值 | 变化原因 |
|---|---|---|
| `Ruleset.name` | `mahjong_soul_4p_red_half_v3_native_rules` | 显式绑定雀魂的两个 `false` 开关及其行为修复，规则 hash 随内容改变 |
| `native_rules_version` | `riichienv_native_rules_v1` | 新增共享原生规则及绑定能力身份 |
| `input_pipeline_version` | `semantic_public_clock_frozen_decisions_v7_native_rules` | 原生事实投影及修复后合法性、终局输入链 |
| `native_replay_version` | `native_replay_context_v4` | 规则上下文由 Rust 自行推进 |
| `SETTLEMENT_VERSION` | `native_mjsoul_settlement_v2` | 审计共享支付、回合和终止内核 |
| `LIVE_EVENT_VERSION` | `native_rule_event_projection_v6` | 在线依赖完整原生规则事件与揭牌 |
| `response_mask_version` | `public_rule_response_mask_v3_abort_last_draw` | 保留原有公开可能性语义，不因实现语言改变升级 |

完整 Schema 仍为 `kar-riichienv-v1`，动作 manifest 仍为 546 个 ID，模型输出仍为 `546 / 21 / 21×4`，候选容量、Token 布局和模型参数形状不变。公共时钟、规则特征和损失合同也未因这次职责收敛重定义。相同形状只说明张量结构相容，不证明旧输入、旧监督或旧 checkpoint 满足新语义。

规则与版本定义分别见 [contracts](../src/riichienv/kar/contracts.py)、[versioning](../src/riichienv/kar/versioning.py)、[原生绑定](../riichienv-python/src/lib.rs)；使用旧扩展时必须明确失败并重建，不能降级回 Python 的旧规则计算。

## 6. 旧数据与产物的迁移

1. 用当前 Rust 扩展从可信的原始 result 记录重新重放，验证动作、观察前缀、独立结算和完整半庄条件。历史真实 Golden 和接受报告保留历史身份，按当前实现重新验证后才能作为当前证据。
2. 对受影响的 KAR 雀魂数据重建规范事件、Token、mask、完整候选、最终名次监督、SQLite、索引、样本计数与 manifest。两项平台规则选择可能改变数据是否合格，不能因原始终分恰好相同就跳过核验，也不能把天凤的不同结果判成雀魂修复前的通用错误。
3. 重新生成输入缓存及其绑定，依据当前合格 train 选择验证或重建支持统计。旧模型形状虽相同，精确续训仍受规则、输入、源码与原生构建身份约束；任何权重迁移须另有明确实现与证据。
4. 绑定旧输入的校准 logits、校准/质量/有限策略、吞吐和发布证据不能只更新 hash 后复用。需要在其实际依赖满足新合同后重新验证或生成，并记录所用源码、原生构建和数据来源。

这不是给旧 JSON 补一个 `native_rules_version` 的格式修补。版本标签和摘要只能记录已经完成的重建或核验，不能代替它们。本轮代码与文档修改不自动迁移真实数据或既有训练产物，也不自动启动训练、GPU、校准或 M3。

## 7. 验证位置与当前证据

| 要证明的性质 | 测试入口 |
|---|---|
| 同牌姿/结算在雀魂与天凤下的不同结果、无参兼容、KAR 显式规则及保留的例外 | [test_native_settlement_kernel.py](../tests/kar/test_native_settlement_kernel.py)、[formal_tenpai_rules.rs](../riichienv-core/tests/formal_tenpai_rules.rs) 与 Rust `hand_evaluator`、`rules::settlement`、`rule` 单元测试 |
| 相同公开状态、实际 Ron 能力不同仍有相同 mask；隐藏/未来字段不进入绑定 | [test_native_public_rules.py](../tests/kar/test_native_public_rules.py) |
| 他家私有信息不进入观察状态，拒绝事件不推进事实，Python 不能改写规则状态 | [test_native_observer_boundary.py](../tests/kar/test_native_observer_boundary.py) |
| 原生完整立直、失败无部分提交、重放不依赖 Python 回写 flag | [test_native_rule_boundary.py](../tests/kar/test_native_rule_boundary.py) |
| 原有公开 mask、结算、终局、杠与 Pass 合同 | [response mask](../tests/kar/test_response_mask_boundaries.py)、[settlement](../tests/kar/test_settlement.py)、[completion](../tests/kar/test_completion.py)、[native replay](../tests/kar/test_native_replay.py)、[Pass 批次](../tests/kar/test_pass_receipt_batches.py) |

### 2026-11-19 集成结果

| 检查 | 实际结果 |
|---|---|
| `maturin develop --release` | 构建并安装当前扩展成功 |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --tb=short` | **3556 passed，42 skipped，2 failed**；两项失败均为下述既有 Gloo 环境阻塞 |
| 最终 release 的支付、规则分流、原子立直、版本与终局日志绑定复验 | **231 passed**；与全量测试重叠，不相加 |
| `cargo test -p riichienv-core --no-fail-fast` | **328 passed，0 failed** |
| `cargo test --all-targets --all-features --no-fail-fast` | **365 passed，0 failed**；所有基准目标的测试模式也成功 |
| `cargo clippy --all-targets --all-features -- -D warnings` | 通过 |
| `cargo fmt --all -- --check` | 通过 |
| `ty check` 覆盖全部修改的 Python 源码与 stub | 通过 |
| Ruff 对 33 个修改/新增 Python 与 stub 文件的基线对比 | **0 项新增诊断**；`audit.py::_finish_response` 原有 `PLR0917` 仍在 |
| `git diff --check` | 通过 |

两项环境失败均来自 `test_full_validation_only_primary_reports_broadcasts_and_coordinates_failure[False/True]`：Gloo 在 `create_device(hostname="127.0.0.1")` 处因 `Operation not permitted` 无法建立 TCP 设备，尚未进入被测验证逻辑。改动前基线已出现同样两项失败；没有修改它们的断言或增加跳过条件。本轮因此不声称默认 Python 全套零失败，仍需在允许 Gloo TCP 的执行环境复验这两项。

本地运行环境为 Python 3.12.14、CPU PyTorch 2.15.0，NumPy 2.3.5；后者低于项目 `kar-ml` 声明的 2.4 下限，故这些结果不构成完整锁定训练环境的验收。Rust 全 feature 测试使用实际解释器的 `lib` 目录设置 `LIBRARY_PATH` 与 `LD_LIBRARY_PATH`，修正当前运行环境中 Python `sysconfig` 残留的 `/install/lib` 路径；未因此修改仓库构建规则。

上述验证证明本轮规则分流、接口、事件时序和训练前缀回归的结果，不代表已迁移真实数据、重新训练模型或测得吞吐与策略强度提升。
