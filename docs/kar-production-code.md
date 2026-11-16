# KAR 完整代码说明：实现先行，运行验收后置

本次补齐此前 T01–T10 的代码缺口。**代码已实现不等于运行验收已通过**：这一轮只写实现、接线、测试源码和静态检查；没有查询远端硬件、没有运行模型前向/反向、没有训练、标定、探针、仿真或 M2/M3。

宿主仍是 RiichiEnv，冻结 `kar-riichienv-v1`、546 动作、32 完整候选、公开规则 mask、固定起家同分顺位。没有修改 Rust Core 或原始牌谱。

逐个 Python 模块的职责、接口、合同、产物与测试位置见 [模块 README 总索引](kar-modules/README.md)，学习运行层见 [learning 子包索引](kar-modules/learning/README.md)。

## 完整运行链的代码

| 代码职责 | 模块与入口 |
|---|---|
| 完整数据导出 | `learning/export.py:export_dataset`（已有完整清单接口） |
| M1 相同初始化／相同样本顺序的 on/off 各一轮 | `learning/orchestration.py:run_m1_ablation` |
| 生产 CPU/CUDA、FP32/BF16 训练 | `learning/trainer.py:train_production,production_update` |
| 全局分桶／尾 batch／DDP 空 rank | `learning/sampling.py:GlobalEpochSampler` |
| torchrun、NCCL／显式 Gloo、同步失败与梯度 | `learning/distributed.py:DistributedContext` |
| 每 0.25 Epoch 与 Epoch 末的完整验证 | `learning/evaluation.py:evaluate_complete` |
| 模型阶梯与独立阶段预算 | `budget.py:CostForecast,BudgetPlan,BudgetLedger,choose_training_tier` |
| 原子代际检查点、每 rank RNG、精确状态恢复 | `learning/production_checkpoint.py` |
| 打包代码身份与完整发布合同 | `release.py` |
| 每类可靠性／召回、21 事件 rank、威胁切片、游戏级 CI | `statistics.py; learning/evaluation.py` |
| 冻结验证 logits、cal_fit/cal_check 与 train-only 支持 | `learning/calibration_job.py` |
| 验证对局固定 tau 网格选择 | `learning/validation_policy.py` |
| 冻结模型的发布、受限加载与配置核验 | `learning/deployment.py` |
| 固定模型／固定对手的真实仿真吞吐 | `learning/simulation.py:benchmark_matches` |
| 唯一主力模型、两轮与有条件第三轮 | `learning/orchestration.py:run_m2` |
| 10000 成对种子块、20000 半庄、四座平衡 | `experiments.py; learning/simulation.py:run_m3` |
| 全部命令的延迟导入入口 | `kar/__main__.py; learning/jobs_cli.py` |

### 训练器

- 按真实决策样本数计算完整 Epoch，不再把诊断用的 100 updates 当正式上限。
- Small/Medium/Large 有效 batch 分别为 128/256/512；microbatch 与 rank 划分不改变完整 update 的 global_N。
- 全局随机顺序后按局部窗口分桶；真实样本不为 DDP 补齐而重复。空 rank/microbatch 使用连图零损失，dummy 不计样本数或条件指标。
- 所有梯度在一个 accumulation group 后统一平均，使用 `W/global_N` 的 local sum 目标；不除以本地有效标签数。
- BF16 autocast、FP32 主参数；前向、反向、有限梯度、裁剪、优化器和 scheduler 各阶段协调失败。失败不推进采样游标或发布部分状态。
- 固定 AdamW 配方及 value_head 专用衰减；固定全训练 manifest 的 rho。
- quarter-epoch 验证覆盖完整 validation，并关闭随机特征置空；不能用 test/cal_check 决定第三轮。
- 检查点保存每 rank 的 CPU/CUDA RNG、全局 sampler 状态、optimizer/scheduler、验证历史、预算账本位置与完整来源合同。
- 每次保存写新 generation，完成后原子切换指针；保留上一代完整检查点，避免 payload 与 sidecar 中断时互相覆盖。

### 规模、预算与阶段

按 **N_train_half_games** 选择：小于 30000 为 Small，30000–49999 为 Medium，50000 以上才允许 Large。成本预测超预算可在训练前降级，但只登记一个主力模型，不能把三个规模全部训练。

M1/M2/M3 分别封顶 USD100/650/250，总额 USD1000。吞吐、单价、已有支出、完整工作量和测量来源必须明确提供；没有“默认实测值”。账本保留真实收费与引用；不跨阶段借预算，不通过减少 M3 pairs 或隐藏尾 batch 来伪装完成。

第三轮需要完整验证总目标继续改善、各头未超过预登记恶化阈值及预算仍有余额。拒绝第三轮是已登记的条件分支，不允许临时修改测试集或添加新训练矩阵。

### 校准、发布与评测

完整代码链为：

`完整训练 → 冻结 validation logits → cal_fit 拟合 / cal_check 仅检查 → 已校准 prior 发布 → cal_fit 对局选择 tau → 冻结有限策略发布 → 独立 M3`

- 校准 archive 用只读 SQLite 懒加载，避免把全部 logits 展开成巨大 Python 对象列表。
- 支持只统计真实训练分区的独立 game，不允许 validation/test 补 499→500。
- 生产清单校验与实际 M0 的 `manifest_sha256=hash(games)` 口径一致；scope、冻结规则和 split 另外检查。
- 模型、native 扩展、配置、特征模式、训练选择、校准／支持／tau 选择 artifact 都有独立 hash。
- 有限策略发布必须来自验证集冻结选择；没有证据时仍按核心规则退回先验。NLL、Brier 和 500-game 筛查都不成为统一误差证书。
- 发布文件不含优化器，使用受限 `weights_only` 加载；源码／native 构建不符需要显式迁移，不静默复用标定。
- 仿真吞吐使用完整模型和固定版本对手，不拿候选后缀吞吐冒充 matches/s。
- M3 固定 10000 pairs，四座各2500；baseline/candidate 同 checkpoint、同牌山 seed、同对手版本与座位，主体策略采样 RNG 分开记录。
- 中断恢复校验每个 arm 的 receipt、replay、代码版本及固定计划；不把截断算完整半庄，不静默减少样本。
- CI 按整场游戏或成对种子块聚类；重复决策行不变成独立样本。经验 utility 校准代理明确不作安全证书。

## 命令入口（这里只说明代码，没有执行）

`python -m riichienv.kar` 提供延迟导入命令：

- `accept-data` / `export-data`
- `train`
- `m1` / `m2`
- `calibrate` / `select-tau` / `release`
- `plan-benchmark` / `plan-m3`（仅元数据计划，不加载权重、不授予执行许可）
- `benchmark` / `m3`
- `stamp-build`

训练与评测入口需要独立明确授权和冻结配置。缺少许可默认拒绝；数据资格、代码实现、运行验收、资源就绪与用户许可不是同一件事。**代码补齐阶段不生成有效执行许可。**

`training_recipe_sha256` 可在提供价格前生成配方绑定；`CostForecast.from_record`、`BudgetPlan.from_record` 和 `ProductionTrainConfig.from_record` 支持带校验字段的往返序列化。

## 修复后的合同与兼容性

- `certified` 与诊断 `argmax` 不可同时启用；生产有限策略也拒绝诊断采样模式。
- tau 选择绑定完整标定、支持、后端和采样身份；导出与加载执行同一验证逻辑。
- 生产标定配置必须在计算 cal_check 前提供 `quality_policy`：三个 head 的 NLL/Brier/ECE 上限、最少观察数和 heldout 门控保留率要求。没有默认“通过阈值”。标准产物保存政策 hash、质量通过/失败状态及完整独立诊断；失败时运行时保留先验。
- 原生可抢暗杠的声明由已验证的实际动作补齐；延迟原生确认只生成提交事实，不重复声明。旧杠的宝牌揭示不能提交新杠。模型仍使用原有接受 kind，不增加 Schema 行或维度。
- M2 登记同时固定 `training_output` 与 `budget_ledger`；原始 train 入口不能换目录重置费用。已有完成检查点可以仅重建结果发布，不再次更新参数。
- 费用账本 v2 持久保留未开票固定费用的预留；`spent_usd` 是实际记账，`committed_usd` 是实际记账加未结预留，不能混报。实际非计算账单通过唯一 `FixedCostReceipt` / `BudgetLedger.charge` 对账，同一账单不能重复收费。
- M3 在任何新对局前核验全部既有收据的费用锚点；缺失或回退账本拒绝恢复。标定、tau 验证和基准的异常路径也结算失败计算。
- 标定合同升级为 v3，杠事务合同为 `public_kan_commit_evidence_v2`。输入字段与 canonical Schema hash 不变，但旧派生数据、旧账本或旧标定不能冒充新合同；通过显式重导出／迁移处理。原始数据和既有报告不被覆盖。
- `accept-data --shard CORPUS INPUT_MANIFEST` 可重复给出多个有界审计分片，汇总全局去重、Golden/语义别名排除、来源链、split 和独立支持计数。单片上限不再限制合并后的 Medium/Large 数据规模。

以上只说明代码合同的变更，不表示已经运行回归或重新验收历史数据。

## 2026-11-16 续补齐：端到端合同

本轮不再把单个组件存在当作整条生产链完成。58 项需求的实现与回归源码对应关系见 [代码覆盖对照](kar-implementation-coverage.md)。本轮只做静态检查，任何运行验收状态均未升级。

### 完整策略吞吐与 M3 计划

- `benchmark` 默认 `mode=paired`；`matches` 表示**总半庄数**，至少 8 且为 8 的倍数。每个 seed block 各跑一次 baseline 和有限 tau candidate，并在四个主体座位平衡。主体 RNG 分开、对手初始 RNG 与牌山配对，后续轨迹可以自然分叉。
- baseline 使用同 checkpoint 的校准先验与无限 tau；candidate 保留发布时冻结的有限策略，不再强制覆盖为 prior-only。明确指定 `mode=baseline` 仍可诊断，但该结果不能用于 M3 成本放行。
- 吞吐计时覆盖模型加载、agent 构造、完整对局、replay 压缩、收据与记账。保存分 arm 决策/延迟、实际辅助前向次数、座位计数和完整来源。
- candidate 未真正执行过响应/价值候选前向时，测量可以完成，但 `m3_forecast_eligible=false`；不能冒充完整候选推理吞吐。
- M3 配置必须提供 `benchmark_report`。配对计划冻结 `forecast_evidence.benchmark_sha256`；成本预测的引用为 `benchmark:<sha256>`，速率不得快于测量。主体、对手、代码、规则、后端、截断上限必须匹配，执行时核对机器/运行时并恢复测量的 CPU 线程数。
- M3 共享注册表同时固定计划、实际输出、费用账本和原定计价/额度；不能复制同一计划到新目录重置 USD250。已登记运行丢失账本时拒绝重建空历史，即使尚无完成对局收据。
- 基准墙 seed 与独立 M3 墙 seed 使用不同命名空间，并拒绝实际重叠。正式 M3 只接受 test 分区及随机有限经验策略；继续固定 10000 pairs、20000 半庄，不因预算不足缩减。
- `plan-benchmark --config INPUT --output PLAN` 与 `plan-m3 --config INPUT --output CONFIG` 只处理元数据。前者生成冻结基准计划；后者消费合格的成对测量，生成完整 seed/cost/job 配置。两者不启动模型、对局或资源，也不生成执行许可。

### 独立验证角色和阶段放行

- 固定 `kar-cal-fit-check-v1` 游戏级角色 hash。全 validation 仍完整报告，但用于第三轮判断的 `model_selection` 仅由 cal_fit 决策计算；cal_check 不参与 epoch、tau 或阈值调节。生产校准不得更换该 salt。
- 有限 tau 使用预登记 `TauSelectionPolicy`：至少两个独立验证 scenario game clusters、固定 bootstrap seed/次数/置信度与最低效用增益。只有配对效用区间下端超过登记门槛的有限项才可选择；均值较高但证据不足仍选 infinity。
- 该区间只是验证集上的经验工程放行，不是逐状态误差证书，也不是独立 M3 结果。
- M2 必须提供 `m1_report`，核验两种特征模式各一完整 Epoch、相同初始化/顺序、完整验证、实际检查点、共享费用账本及当前代码/native 构建。Medium/Large 另需 `m1_tau_selection_job`，且收益来自对应 M1 checkpoint/特征模式。没有可重复验证收益不扩大模型。
- 主模型登记 v3 封存这些前置证据；直接调用原始 trainer 也会重读并验证，不能通过草稿登记绕过。缺运行证据时只拒绝执行，不阻止继续编写代码。

### 完整诊断与可恢复性能记录

- 支持质量 M 保留固定宽度直方图，同时记录完整支持候选数分布和威胁/当前 rank 分组；不再只报均值。feature-off 的诊断分组读取未遮蔽的公开状态，而非最终名次标签。
- 响应类别及 21 个观察到的 rank 分支输出自然频率 NLL/Brier、分布、可靠性和按 game 聚类的区间；零样本/不足独立游戏明确报告证据不足，不声称已经校准或获得统一误差界。
- M3 记录可合并的决策延迟分布；p50/p95 是明确标注的直方图区间，不把近似区间伪装成精确点分位数。实际概率变化与执行过候选评分分开统计。
- 训练检查点持久化真实 tokens、真实样本、padding、提交更新耗时、数据等待及验证耗时。完成报告以总工作量除以总耗时计算吞吐，不对每步速率做错误的算术平均；恢复发布不重做更新。
- 输入角色合同、tau 选择 v3、M3 seed 计划 v2、完整策略基准 v2 和主模型登记 v3 均显式校验；旧派生产物不被静默迁移，原始数据与历史报告仍保持不变。

## 测试代码与验收状态

新增 `tests/kar/production/` 覆盖：

- 全局样本完整性、尾 batch、空 rank、sampler 精确恢复与 quarter-epoch 验证边界；
- 模型阶梯、预算封顶、独立收费与篡改拒绝、第三轮条件；
- 每类可靠性、观察分支 rank 指标、游戏级 bootstrap、哨兵／概率下溢；
- 固定10000 pairs、四座平衡、JSON 往返和计划篡改；
- 新训练／评测入口的默认拒绝；
- 真实两进程生产更新与单进程 global-sum 等价、dummy microbatch；
- 完整 Epoch / 全量 validation、非有限输出不更新参数或 scheduler、原子检查点。

含 backward/更新的测试保留显式标记，默认不会参与无更新验收。**这一轮没有运行任何测试用例**，也没有把源码中写好的测试算作已经通过。运行验收统一留在完整代码之后进行。
