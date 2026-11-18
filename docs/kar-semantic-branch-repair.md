# KAR V4：公共输入、响应规则与分支质量修复报告

日期：2026-11-18。仓库 `suiniara/RiichiEnv`，分支 `kar/riichienv-v1-20261117`。

## 结论与证据边界

实际开始/结束 HEAD 均为 `9a4d1549719d36987309f9752aa4c3362914b757`，开始工作区干净。没有回退分支，没有commit/push。

**本轮最终默认KAR：2846 passed、0 failed、0 errors、19 skipped。最终定向五组：130 passed、0 failed/errors/skipped。两者重叠，不能相加。Ruff、Ty通过。**

所有最终组执行前后均为同一源码/测试/守卫/runner指纹，源码聚合SHA-256：

```text
80e9594d12698a13738049848fa6272be7d8eb6da1919c5bab0aff3c0c3e6e0f
```

完整记录在工作区 `kar-fix-reports/semantic-branch-20261118`，含实际argv/cwd/环境、原始stdout、退出码、JUnit、逐文件SHA与禁止调用守卫；不是仅在文档里追加PASS。审查基线的2673/40、历史1348/372/45均不作为本轮结果。

当前可访问工作区内未找到 `KAR-9a4d154-audit.zip`，没有声称执行包内脚本。永久回归按任务给出的反例，在仓库真实接口和原生合法136物理牌墙上重建；旧错误行为未被保留为“正确”断言。

## 1. 问题、原因、修改入口与行为

| 问题 | 原因 | 修改文件 | 修复行为 |
|---|---|---|---|
| 离线/在线逻辑位置 | 账本按ServerRecord/native记录分组推进时钟 | [event_semantics](../src/riichienv/kar/event_semantics.py)、[observation](../src/riichienv/kar/observation.py)、[tokenize](../src/riichienv/kar/tokenize.py)、[live_events](../src/riichienv/kar/live_events.py) | 共享因果公共语义时钟，source record/hash和native批次不决定RoPE位置；开局首draw独立tick；S/DECIDE=C+1、ACTION=C+2，H不冒充C |
| 终局/鸣牌互斥 | mask满足途中流局时仍允许NEXT_DRAW；末巡只禁杠未禁吃碰 | [response_mask](../src/riichienv/kar/response_mask.py) | possible/certain分开；确定终止排除普通推进与吃碰杠，保留公开仍可能的Ron及其优先级；多人四杠区别单人四杠；remaining=0禁所有开鸣 |
| 九种九牌历史 | RecordLiuJu只输出reason，丢失公开手牌与已证零转移 | [mjsoul](../src/riichienv/kar/mjsoul.py)、event_semantics、live_events | 读取/验证宣布者seat和tiles；只在公开终止后揭示该玩家，源tiles缺失仅允许冻结规则和已验证前序事件唯一重建；记录来源，规则证明零结算与缺字段分开 |
| 罕见分支质量 | 总体三头指标和train500筛查独自放行 | [calibration_quality](../src/riichienv/kar/calibration_quality.py)、[calibration](../src/riichienv/kar/calibration.py)、[support](../src/riichienv/kar/support.py)、[runtime](../src/riichienv/kar/runtime.py)、[deployment](../src/riichienv/kar/learning/deployment.py) | 显式预登记response/rank逐分支阈值、独立验证game数、来源与版本；passed/failed/unavailable；有效价值支持=train≥500∩两头分支质量passed |
| 全套暴露的输入顺序不稳定 | 原生observations来自hash-map，同cutoff多个Pass样本顺序跨进程变化 | [audit](../src/riichienv/kar/audit.py)、[两进程永久回归](../tests/kar/test_parallel_replay.py) | 按稳定初始座位输出样本；每个观察者输入保持不变，导出顺序/聚合SHA确定；保留原字节/哈希断言，不在测试里排序结果掩盖错误 |

### 时钟和事务的具体边界

- start_game、start_kyoku、庄家首次draw为位置1/2/3，S/DECIDE=4；不固定加减位置，不忽略positions，不放宽数值容差。
- reach/discard、reach_accepted/后续draw、Dora各按规范事件计时；单语义事件展开的多个Token共位置。
- 多Ron按相对放铳者P1/P2/P3规范顺序展开，共同裁定时间、公共总支付只编码一次；不完整native win group暂存并拒绝决策snapshot，不借未到达事件补写旧输入。
- 可选kan_commit ACK不占模型时间，仍可即时证明当前账本的已提交状态；历史接受Token锚定共同可见的本次有效Dora/replacement draw。旧杠延迟Dora不提交新杠，有无ACK不改变最终历史。
- SELF_PASS只入本家，确认锚使用已实际发生的关闭事件，不推进公共ID/时钟；杠的规范私有关闭锚为替换draw。旧snapshot/prefix不回写，其他观察者不变。不同来源provenance单独比较，不伪造一致来源或未知Pass意图。
- public_clock_version缺失/旧值拒绝Tokenize/runtime；历史与K/V缓存同时绑定当前input/model合同。

### 九种九牌公开信息与零值

实际proto的RecordLiuJu字段为type=1、seat=3、tiles=4、allplayertiles=6；本轮只消费必要公开字段，不从allplayertiles补其他玩家手牌。验证当前draw actor、首次draw、无本家舍牌/无鸣牌、14张、至少9种幺九字及源tiles一致。身份/牌张矛盾或重建前提不足直接RecordError，不返回伪装完整事件；通用终局可见性缺口仍用visibility_complete=false/gap阻断前缀。

终止前其他观察者看不到该手牌；终止后各视角取得相同公共事实。已证途中流局无转移时记录零deltas与当时已知scores，产生4个present=true的SCORE_DELTA；不撤回已成立立直扣棒。NoTile/Hule缺结算仍未知，未从下一局分数反推或统一填零。

### 分支回退与既有公式的关系

正式分支阈值与最小验证game数没有默认值，必须在计算/查看cal_check前登记。指标总体、NLL/Brier/10-bin top-label ECE、distinct validation game计数、来源与摘要均可序列化。重复100行同一游戏仍只算1个独立game。缺正式判据/证据为unavailable；充分证据超限为failed，各头状态分别保留。

质量不改公开规则mask。先mask→Softmax，再按train与质量交集决定分支是否供应预测效用；failed/unavailable分支的概率仍保留，取−1.5并计入rho_U，不删除后重归一化。S的定义不变，故受限KL的集合外逐动作概率和集合内总质量合同仍成立。无合格价值分支时直接同一先验、零候选前向，原因 `no_quality_qualified_response_branches`；有限发布也拒绝这种无质量支持状态。普通质量指标不产生certified。

## 2. 反例 → 正确永久断言

| 原反例/边界 | 正确断言与实际覆盖 |
|---|---|
| 开局64行positions不同 | 四视角完整Token字节一致，首draw=3、DECIDE=4；source不同合法打包/增量缓存一致 |
| 随机Small三头因位置改变 | 同一随机seed、eval、FP32、真实Small 5,239,435参数；开局与第四家立直输入全字段相同，三头最大绝对差均0.0 |
| 第四家立直candidate39、accepted=(T,T,T,F)、remaining66 | 真正原生合法四家立直构造；NEXT_DRAW与吃碰杠False、ABORT_FOUR_RIICHI允许；Ron只依公开规则而非暗手排除；最终真实标签仍允许 |
| 末巡remaining0仍允许PON | 实际完整原生小局最后舍牌及remaining1相邻边界；末巡所有吃碰杠False |
| 四风/四杠 | 真正合法136牌墙及每步原生提供动作；四风终止，多人四杠终止，一人四杠不误判；未知历史不强行终止，已证多人四杠仍有效 |
| 九种后下一局20/27历史行差异 | 实际合法九种终止并继续同一环境下一局draw；四视角、3种分块，公开手牌3行、零结算4行、历史/顺序/位置/完整输入字节一致 |
| 九种source字段或可见性不确定 | 合成真实wire结构检查seat/tiles，缺tiles受严格规则重建，错人/错牌/迟于首摸/非九种拒绝；poisoned allplayertiles不进入事件 |
| 多Ron跨批次/赢家顺序 | 原生合法双Ron；1/2/3/整批、正/反赢家顺序得到相同输入，公共支付只一次；未完整裁定拒绝snapshot，旧前缀不回写 |
| 旧杠延迟Dora及可选ACK | 声明/抢杠/提交既有测试保留，并补无ACK/有ACK时最终历史字节一致 |
| 总体rank NLL约0.081而RON_MASK_1约8.001 | 本机合成实测总体 **0.0802138026070063**、分支 **8.001005881814928**；总体passed但该分支failed，仍保留其masked概率，改变其rank预测不能影响最终受保护策略 |
| 分支证据/规则/train支持混淆 | passed/failed/unavailable、规则不可能、train499/500正交组合；未登记全部unavailable时零候选前向先验回退；旧缺合同校准/支持/策略身份拒绝 |
| 串行/两进程聚合SHA不同 | 实际只读2个既有完整原始牌谱，逐决策触发/观察者/目标/前缀跟踪与聚合哈希都一致，无skip |

原生场景是有界合成合法局面/小局，不冒充新的真实雀魂Golden。2个真实源重放也不是整个M0合格池重新验收。

## 3. 实际执行结果与命令

环境为本机Python 3.14、PyTorch **2.15.0+cpu**，torch.version.cuda=None；没有安装/升级依赖。最终默认套件实际耗时401.70秒。该CPU结果不外推为GPU/BF16硬件、真实校准或棋力结果。

| 最终标签 | passed | failed/errors/skipped |
|---|---:|---|
| inputs-final |34|0/0/0|
| mask-final |11|0/0/0|
| quality-final |23|0/0/0|
| versions-final |22|0/0/0|
| forward-final |40|0/0/0|
| 定向合计 |**130**|**0/0/0**|
| default-final（包括上述用例，不重复相加） |**2846**|**0/0/19**|
| ruff-final / ty-final |exit 0|通过|

默认套件内还实际执行：mjsoul104、observation/response201、Nagashi一致性13、既有杠事务8、SQLite/两类checkpoint/冻结发布版本序列化102、通用设计合同49、输入版本2、语义版本22、两进程原始重放1、分支质量23等。它们是默认2846的一部分，不另加总。

所有子命令cwd为 `D:\Workspace\RiichiEnv`。实际调用如下；完整子进程argv、显式环境、JUnit与临时目录参数见各同名JSON。证据只增不覆盖，复跑需换新标签。

```bash
& "D:\Workspace\RiichiEnv\.venv\Scripts\python.exe" -I "D:\Workspace\kar-fix-reports\semantic-branch-20261118\run_checks.py" inputs inputs-final
```
```bash
& "D:\Workspace\RiichiEnv\.venv\Scripts\python.exe" -I "D:\Workspace\kar-fix-reports\semantic-branch-20261118\run_checks.py" mask mask-final
```
```bash
& "D:\Workspace\RiichiEnv\.venv\Scripts\python.exe" -I "D:\Workspace\kar-fix-reports\semantic-branch-20261118\run_checks.py" quality quality-final
```
```bash
& "D:\Workspace\RiichiEnv\.venv\Scripts\python.exe" -I "D:\Workspace\kar-fix-reports\semantic-branch-20261118\run_checks.py" versions versions-final
```
```bash
& "D:\Workspace\RiichiEnv\.venv\Scripts\python.exe" -I "D:\Workspace\kar-fix-reports\semantic-branch-20261118\run_checks.py" forward forward-final
```
```bash
& "D:\Workspace\RiichiEnv\.venv\Scripts\python.exe" -I "D:\Workspace\kar-fix-reports\semantic-branch-20261118\run_checks.py" default default-final
```
```bash
& "D:\Workspace\RiichiEnv\.venv\Scripts\python.exe" -I "D:\Workspace\kar-fix-reports\semantic-branch-20261118\run_checks.py" ruff ruff-final
```
```bash
& "D:\Workspace\RiichiEnv\.venv\Scripts\python.exe" -I "D:\Workspace\kar-fix-reports\semantic-branch-20261118\run_checks.py" ty ty-final
```

Pytest在spawn-safe `forward_only.py`守卫下运行，未传 `--allow-parameter-updates`。最终默认与定向组：Tensor.backward/autograd.backward/autograd.grad/optimizer.step/CUDA初始化尝试均0；CUDA initialized=false、grad_enabled=false。两进程worker继承相同守卫，startup记录带守卫SHA，每次禁止调用会在抛错前另写事件；最终无worker禁止事件。worker startup不是终止时证书，不伪装为完整worker末态快照。

### 19个skip，均不计PASS

- **15个既有parameter_updates保护节点**：本轮禁止反向/更新，没有opt-in。
- **4个私有导出正向集成节点**：本地 `kar-m1-data/cpu-current-contract` 实际仍是旧ruleset/mask/输入合同，不因为目录叫current就兼容。先核对生产加载器严格拒绝、manifest自洽和数据库SHA完整，再明确skip为“V4真实重建未授权”。
- 原正向测试断言没有删除或放宽；新增5个fixture边界回归确保显式选择旧路径、缺失/损坏资产、损坏manifest或当前版本错误仍ValueError，不被skip掩盖。生产DecisionDataset验证器没有为这些skip放松。

因此当前V4真实私有导出的四项正向集成仍为 **NOT EXECUTED**，不能把旧8游戏数据换标签充当通过。

## 4. 版本变更与冻结保护

| 身份 | 9a4基线 | 当前 |
|---|---|---|
| artifact |kar_riichienv_contract_v3|kar_riichienv_contract_v4|
| input pipeline |legal_set_pool_private_self_pass_v1|semantic_public_clock_terminal_reveal_v2|
| public clock |按source record分组，未独立版本化|public_semantic_clock_v1|
| terminal public |九种稀疏终止历史，未独立版本化|verified_abort_public_resolution_v1|
| response mask |public_rule_response_mask_v2_mjsoul_nagashi|public_rule_response_mask_v3_abort_last_draw|
| calibration quality |preregistered_calibration_quality_v1|preregistered_calibration_quality_v2_branches|
| branch quality |缺合同/未接线|preregistered_response_branch_quality_v1|
| support |train_independent_games_v1|train_independent_games_v2_branch_quality|
| calibration |heldout_temperatures_log_gate_v3|heldout_temperatures_log_gate_v4_branches|
| runtime policy |restricted_kl_runtime_v1|restricted_kl_runtime_v2_branch_quality|
| finite policy identity |calibrated_policy_identity_v1|calibrated_policy_identity_v2_branches|
| model identity |kar_transformer_v2_legal_set|kar_transformer_v3_semantic_inputs|
| training config |explicit_train_split_scale_v3_inputs|explicit_train_split_scale_v4_semantics|
| live projection |native_public_resolution_projection_v3_private_pass|native_public_resolution_projection_v4_semantic_clock|

保持不变：宿主Schema/hash、设计引用及overlay、映射、ruleset、动作manifest、架构、特征/随机置空、loss/bound/训练plan、N_train选型口径。完整字段表见[单一主设计§0](KAR-v1.2.2-DESIGN.md)。模型结构源码、losses、policy、probability、schema及JSON、rule_features和kan_transactions均与基线逐文件SHA相同；native扩展未重建/未改变。

四个实际preset参数数仍为Tiny2,085,771、Small5,239,435、Medium14,790,795、Large26,536,587。正确LEGAL_SET、私有Pass、FP32投影不回退。随机Small及缓存精度测试记录临时模型权重/buffer指纹前后相同，没有optimizer更新。

## 5. 拒绝矩阵与真实重建需求

- **旧数据/SQLite**：缺任一新输入、时钟、终局、mask、branch字段或scope不匹配拒绝；不能只重签外层manifest，内部scope和实际split/game计数仍校验。
- **旧快照/历史/KV缓存**：旧/缺public_clock_version拒绝；input/model/candidate/参数/buffer绑定保留，必须重新生成。
- **旧checkpoint/发布**：两类checkpoint与FrozenRelease严格活动合同校验，即使调用方给匹配的旧合同且校验和有效也拒绝；source/native指纹检查未放宽。
- **旧支持/校准/finite选择**：旧版本、缺分支判据/report/states、策略身份缺新字段拒绝；validation/train选择manifest也校验完整活动合同。None分支政策是显式unavailable，不是兼容旧产物后自动passed。
- **真实重建清单，未自动执行**：在新输出目录重新审计/规范化与导出原始result数据及mask；重建历史/KV；按当前train重新统计支持；冻结新输入/模型后按正式预登记标准重做真实校准和finite验证选型；重新测吞吐、重建发布。旧权重同shape不表示精确兼容；任何迁移另行授权/验证，不重签旧hash伪装。

本轮仅用新临时合成SQLite、checkpoint、支持/校准/发布进行拒绝测试；旧真实权重/数据库/manifest/原始盘和历史报告不覆盖。

## 6. 单一完整主设计与文档入口

完整修订版：[docs/KAR-v1.2.2-DESIGN.md](KAR-v1.2.2-DESIGN.md)，共0..23节。已在正文统一RiichiEnv宿主完整Schema、H与C、事件/事务/私有时钟、九种公开/零结算、N_train阶梯、当前输入/规则/质量版本、质量三态与悲观支持公式。不是追加一份让实施者在N_eligible/N_train等冲突间自行选择的补丁。

原件/工作区原设计字节SHA保持 `4349039c44175ece77f6dddad3a2106fbd3e0e0339935122df242175c83cbdba`；当前仓库主文件是完整修订版，其新SHA在 `docs-final.json/evidence-index.json`，不把原SHA冒充修订后SHA。

根README、源码导航、设计一致性/输入链说明、核心/生产接口文档及相关模块README已同步。全部74个模块README存在，静态API/链接检查另有日志，不把文档检查说成模型/训练执行。

## 7. 初始失败与验证脚本修复不隐去

- inputs-before真实为24 failed/1 passed，其中一项是新fixture误取复合立直动作的首步None tile；修成最终discard步骤后继续验证。输入回归随后扩展到34项，不能把最终测试SHA冒充首次红灯SHA。
- mask-before为7 failed/4 passed；quality-before为1 failed（未登记分支仍做12个候选前向）。后续正确行为断言已通过。
- 默认初轮及诊断轮曾因外部包装器在Windows spawn导入时顶层执行pytest而递归启动测试，均为中断、NOT PASSED，未统计最终通过数。修包装器的__main__边界，不改/跳过真正两进程测试。
- spawn-safe后完整诊断轮暴露hash-map样本顺序问题；逐决策跟踪证明是同cutoff观察者顺序互换而非每个观察者的字段不同。修生产audit固定座位输出，保留哈希断言并新增顺序断言。
- 中间静态问题和失败日志保留。诊断全套期间发生过后续源码修正，明确不作为最终source-bound PASS；只有本报告列出的*-final与同一SHA用于最终结论。

## 8. 明确未执行及交付物

**NOT EXECUTED：** 当前V4真实私有导出正向集成4项；全量真实M0/整个Golden重新认证；真实数据/支持/校准/权重迁移重建；梯度、backward、autograd.grad、optimizer.step、DDP真实更新；GPU/CUDA/付费/云；训练launcher、M1/M2/M3、真实模型校准和独立棋力评测。默认套件中的有界原生小局/半庄是功能回归，不是M3。没有为真实数据签发参数更新或评测授权。

交付包括完整主设计、本报告、含所有新增文件的changes.patch、逐文件变更SHA、全部最终及失败/中断证据、原始JUnit/命令/守卫、文档检查与CRC/逐文件SHA核验后的ZIP。不包含临时权重或数据。Git只读diff检查及反向--check不应用补丁，不暂存、提交或推送。结束后停止，不自动进入训练/对局评测。
