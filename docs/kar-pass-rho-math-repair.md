# KAR：分批 Pass、有效未支持质量与误差界

唯一活动规范：[KAR-v1.2.2-DESIGN.md](KAR-v1.2.2-DESIGN.md)。本页是实现与验证记录，不是另一份设计。

分支 `kar/riichienv-v1-20261117`。开始时本地、跟踪分支与通过 `gh api` 读取的 fork 远端均为 `c095084b07e5eb8d43c0de79a2af1b53bd6d130f`，工作区及暂存区干净。结束时 HEAD 和远端仍为同一 SHA；工作区仅含本轮 25 个文件（21 个修改、4 个新增），暂存区为空。未回退、切换分支、提交或推送。实际路径和逐文件指纹在证据包 `changed-files.json` 中。

## 1. 已提交 Pass 的分批消费

**根因。** [live_events.py](../src/riichienv/kar/live_events.py) 原来仅为未完成的 hora 组延迟处理。它在写入前清空 receipt 队列，未判断每个 receipt 的所属账本有没有合法关闭锚。在 `ankan | dora | tsumo` 情形，第一个回调的 ACK 不是私有窗口关闭证据，`record_self_pass()` 抛错后 receipt 已丢失。正常 `run_match` 整批路径不受这个反例影响。

**修复。** [observation.py](../src/riichienv/kar/observation.py) 的 `self_pass_ready()` 共用记录时的身份、触发器、来源和锚验证。只有合法且仍开放的窗口返回等待；非法或已失去锚的窗口报错，直接 `record_self_pass()` 也不接受未关闭窗口。投影先校验新传入的 receipts，然后按条保留有效待办；只有已写入或幂等成功才删除该项。空回调可携带已获原子提交成功的 receipt，capture 本身仍无授权意义。

**实际反例。** [永久回归](../tests/kar/test_pass_receipt_batches.py) 用同一完整 136 物理牌墙，验证原生向玩家 1 提供合法国士抢暗杠机会。明确选择 Pass，原生包装器成功提交，随后确实产生 `['ankan', 'dora', 'tsumo']`。两种分批交付方式得到相同结果：

| 观测 | 修复前 | 修复后 |
|---|---|---|
| 整批本家 SELF_PASS | 1 | 1 |
| receipt 随 `ankan` 第一段或单独空回调 | `Private Pass trigger has no observed closing event` | 正常等待；不写 marker |
| 第一回调后 / Dora 后队列条数 | 0 / 0 | 1 / 1 |
| `tsumo` 后本家 SELF_PASS | 0 | 1 |
| 最终整批与分批 prefix 字节 / positions / 历史顺序 | 不一致 | 全部一致 |

其他观察者无 marker，公共 ID 和时钟不因插入而改变；旧 snapshot/prefix 仍冻结。重复 receipt 不增 Token。以先普通舍牌、后可抢暗杠的真实窗口验证 ready/waiting 混合处理；额外注入单条记录错误，确认异常未被伪装成等待、未丢其它待办。错误触发器、owner、来源和失去合法锚均有拒绝断言。

## 2. 统计与 runtime 的 rho_U 对齐

**根因。** [calibration.py](../src/riichienv/kar/calibration.py) 与 [learning/evaluation.py](../src/riichienv/kar/learning/evaluation.py) 原直方图只把 `EventSupport.supported`（独立 train 游戏 ≥500）视为支持。运行时的实际价值支持还要求冻结 `cal_check` 的 response/rank 分支质量均通过。

**修复。** 固定 `cal_fit` 的温度和门控后，先算基础指标和逐分支独立局计数。由已登记的 `cal_check` 判据定出质量三态，再用同一个 `EventSupport.value_supported()` 得到有效集合，以冻结温度和公开 mask 重读有效响应行，填完 fit/check 的统计才冻结和计算产物摘要。第二遍可流式重读现有只读 logit archive，不缓存整个概率矩阵；统计不参与温度拟合、log 门控和质量判定。evaluation 对 histogram 用和 runtime 相同的 masked 标量归一化，避免同一输入在 FP32 批量 softmax 的舍入下跨桶；原有 loss 和分类统计公式未改。

[statistics.py](../src/riichienv/kar/statistics.py) 保留独立 `unsupported_mass_histogram`（最终价值支持）和 `train_only_unsupported_mass_histogram`（训练计数诊断），含 support basis、支持位、状态、有效观测数及桶。没有冻结校准时最终口径明确 `unavailable`，不是实测 0；无分支放行的已冻结产物仍可描述全不支持的概率，runtime 必须先验回退、零候选前向。共享 [support.py](../src/riichienv/kar/support.py) 的概率求和不改概率值；规则剔除类质量严格 0，failed/unavailable 的可行类不删除、不重新归一化，仍使用 −1.5 和既有受限 KL。

**1000 局合成反例。** [永久回归](../tests/kar/production/test_value_support_mass.py) 用 990 常见分支/10 个 RON_MASK_1，独立 game ID，合成 train count 全 500。判据在造数据前登记，只是测试阈值。`T_E=T_R=2.0`；三者使用同一公开 mask，有意给被 mask 的规则不可能类高 logit，确认其概率仍为 0。evaluation 输入相同决策、游戏 ID、logits、标签和 mask；额外未观察响应行不入桶，观察到响应而缺 rank 的行仍入桶。实际总体 rank NLL/Brier/ECE 分别为 **0.0810058818 / 0.0199745377 / 0.0089946239**，总体 passed；罕见 rank NLL **8.0010058818**，分支 failed。两类 runtime 单行质量约为 **0.0046745230 / 0.9996661055**。

| 同一批有效响应 | 修复前 | 修复后 |
|---|---|---|
| 校准及 JSON 重载后的有效口径 | 原报告 `{0: 1000}`，实际只用 train count | `{0: 990, 99: 10}` |
| `evaluation.add_diagnostics` 和聚合结果 | `{0: 1000}` | `{0: 990, 99: 10}` |
| runtime 用实际有效集的质量 | `{0: 990, 99: 10}` | `{0: 990, 99: 10}`，实际数值保持 |
| 明确命名的 train-only 诊断 | 没有单独字段 | `{0: 1000}` |

完整拟合/诊断仍是合成 logits，不是已训练模型的实测校准。保留 train499/500、passed/failed/unavailable、规则允许位，以及 S 外概率和 S 内总质量的既有断言。校准加载会拒绝旧版本、旧统计、缺字段及与绑定支持不符的记录；聚合拒绝旧状态或同口径异支持集。两个原本手填指标的合成 fixture 现经实际 `calibrate_static_rows()` 构造当前合同，原负例断言保留。

## 3. 名义与保守误差域

[主设计第 10 节](KAR-v1.2.2-DESIGN.md) 正文区分 $\varepsilon_{m,\mathrm{all}}$（所有公开允许、尤其模型分配正质量的事件）与 $\varepsilon_{m,\mathrm{sup}}$（公开允许与有效支持集的交集）。后者只能用于保守单侧界的 $b_a=\varepsilon_{m,\mathrm{sup}}+3\delta_E$，不能用于名义 $\widehat Q$ 的全事件误差界。

[精确分数验算](../tests/kar/core/test_value_bound_domains.py) 取 `allowed={0,1}`、`E_sup={0}`、相同概率 `(1/2,1/2)`，允许但未支持的事件 1 的真值是 −1.5、预测是 +1.5。实算 $\delta_E=\varepsilon_{m,\mathrm{sup}}=0$，$\varepsilon_{m,\mathrm{all}}=3$，$Q^\mu=-3/4$、$\widehat Q=+3/4$、名义绝对误差 $3/2$。悲观替代后 $\widetilde Q=-3/4$，$\rho_U=1/2$，实际误差 0 落在 $[-3/2,0]$。保守单侧区间、E1/E2、随机采样与证书条件保持；没有用 passed、样本数或普通指标填零或生成证书。

## 4. 版本及真实重建边界

| 身份 | 基线 | 本轮 |
|---|---|---|
| 在线投影 | `native_public_resolution_projection_v4_semantic_clock` | `native_public_resolution_projection_v5_pass_receipts` |
| 校准产物／runtime 内绑定 | `heldout_temperatures_log_gate_v4_branches` | `heldout_temperatures_log_gate_v5_value_mass` |
| 诊断报告及 DDP 合并载体 | `whole_game_cluster_statistics_v2` | `whole_game_cluster_statistics_v3_value_mass` |

活动 artifact v4、宿主 Schema/input、公开时钟、response mask、规则、模型版本与形状没有升版；已正确整批投影的前缀/派生输入没有因报告字段而改变。旧丢失私有 marker 的分批 trace 若要补救，只能从可信的原始已提交 receipt 取得证据再投影，不能从公开流反猜。旧直方图只有 train-only 口径，有效价值口径必须真实重算；绑定完整 runtime 合同的旧支持记录、引用旧校准摘要的策略选择/发布、源码绑定的既有执行批准不能原样充当当前产物。完整发布仍检查源代码/原生构建指纹；本轮不绕过。可从获授权且来源核验通过的冻结 logits 重新计算，不能补字段、改版本、重签旧 JSON 来冒充统计更新。没有为统计修复要求全量重导出已经合格的 V4 Token；本地旧 V4 前的私有导出仍不能用于当前真实正向集成。

## 5. 实际验证

| 实际执行 | Passed | Failed | Errors | Skipped | 退出码 |
|---|---:|---:|---:|---:|---:|
| `pass-final`：分批 receipt、原有私有 Pass 及杠事务 | 37 | 0 | 0 | 0 | 0 |
| `rho-final`：质量、统计、版本、既有策略/E1/E2 回归 | 93 | 0 | 0 | 0 | 0 |
| `math-final`：精确分数验算 | 1 | 0 | 0 | 0 | 0 |
| `preserved-final`：输入、规则、版本拒绝、模型前向 | 120 | 0 | 0 | 0 | 0 |
| **定向合计** | **251** | **0** | **0** | **0** | — |
| **`default-final`：默认 `tests/kar`** | **2864** | **0** | **0** | **19** | **0** |

全部定向用例都在默认套件中通过，不与 2864 相加。默认套件耗时约 303.69 秒。19 个 skip 为 **15 个现有 `parameter_updates` 保护节点**和 **4 个现有私有旧导出正向集成节点**（V4 重建未执行）；原因和逐项名称在 `default-final.txt/.xml`、`verified-results.json` 中，不计 PASS。没有改选择或跳过条件。

基线上的 `pass-before` 为 7 failed、`rho-before` 为 6 failed，均退出 1；后者在 Pass 修复已在工作区、但统计生产源码仍为基线时执行。各自保留实际源码清单和精确反例属性。新回归随后补强，没将基线失败改作成功条件。

本机 Python 3.14 / PyTorch `2.15.0+cpu`。执行顺序：最终 `pass`、`rho`、`math`、`preserved` 与 Ruff/Ty 定向确认后运行一次默认套件。实际默认调用如下；全部组的子进程 argv、cwd、环境、退出码、原始日志和 JUnit 保存在 `kar-fix-reports/pass-rho-math-20261118` 及交付 ZIP 中。

```bash
& "D:\Workspace\RiichiEnv\.venv\Scripts\python.exe" -I "D:\Workspace\kar-fix-reports\pass-rho-math-20261118\run_checks.py" default default-final
```

守卫在 `D:\Workspace\RiichiEnv` 调用 `pytest -q` 的 `tests/kar`（含 JUnit 与临时目录参数），未向 pytest 传 `--allow-parameter-updates`。Ruff `check .`、Ty `check`、文档 AST/API/链接检查与 `git diff --check` 均通过。最终组测试前后源码、测试、pyproject 和主设计聚合 SHA-256：`973090ab05c291addbcad225482d994adacda4c94e1e5f65e461f5452254fe13`；主设计文件 SHA-256：`db0f19fe5805614a952c9b190432b9e2f4ecd2c8413448c986af040ce74c8537`。冻结模型、规则、时钟、mask、loss 和 KL 源码指纹均核对未变。

保留行为复验：语义时钟及来源分块、开局 `RecordNewRound` 规范化 vs native 输入一致且 DECIDE=4，原生末巡/途中流局/荣和优先、九种九牌四视角及下一局前缀、LEGAL_SET、参数量与三头 FP32。Small 两例使用同一固定 seed 的 5,239,435 参数随机模型，三个输出差均为 0.0；**第四家立直只比 native 中盘和同一 `live.events` 改来源元数据，不是独立经过雀魂 decoder 的双入口中盘验证**。未额外实现或宣称此双入口中盘场景。

最终守卫记录 backward、autograd.grad、optimizer.step 与 CUDA 初始化调用尝试全部为 0；CUDA 未初始化。实际两进程重放也通过，workers 有相同守卫启动记录且无禁止调用事件。未进入训练 launcher、真实参数更新、GPU、全量真实数据重建、真实模型校准或 M3。默认套件中已有的有界原生和只读原始数据功能回归不是完整 M0、正式对局或棋力验收。
