# KAR：可证响应监督、冻结 FP32 与完整验证统计

基线 `4c09e9eb1f02fde3bb4ce0a2fe4d769b634e3610`，`suiniara/RiichiEnv` 的 `kar/riichienv-v1-20261117`，开工时本地和 fork 一致。保留开工前 `docs/kar-modules/observation/README.md` 的改动，不提交或推送。

## 1. 响应监督：机会、标签证据与删失

旧 [replay_game](../src/riichienv/kar/audit.py) 只对普通 NEXT_DRAW 导出可证 Pass。暗／加杠遇到岭上摸牌、普通舍牌遇到成立鸣牌时，既有响应机会可能被清空；完整合法半庄和独立支付通过也不能发现漏行。现在 [ResponseWindow/OpportunityCounts](../src/riichienv/kar/response_supervision.py) 冻结原生选择、同席完整候选和原始窗口，直到真正关闭才分类。实际成立的动作照常是显式标签。只有所有非 Pass 候选都被规则及结果排除才推断，不按已见结果重算合法集合。

- 正常下家摸牌继续用 `inferred_unique_next_draw_pass`；原生只允许 Pass/Ron 的抢杠窗以实际岭上摸牌证明 `inferred_unique_kan_replacement_pass`。
- 严格优先级 `Ron > Pon/大明杠 > Chi`：若所有非 Pass 选择都严格压过公开成立的鸣牌，则是 `inferred_unique_call_priority_pass`。若仍可提交被覆盖的 Chi 等动作，不猜 Pass。对无截和的完整赢家组或已验证无荣和终局，只在其他合法候选全是 Ron、能够排除 Ron 时，使用单独的 `inferred_unique_terminal_no_ron_pass`。不猜相同优先级裁定或缺失结局。
- [ObserverLedger](../src/riichienv/kar/observation.py) 只在实际关闭锚后记录私有 `SELF_PASS`，不回写冻结 prefix，不推进公共时钟。确认窗的成熟时间可不同于声明：连续杠中先等此前杠的 Dora，再冻结合法抢杠输入。`trigger_event_id` 仍标识原始窗（已推断 Pass）或实际动作结果（显式），[GameWriter](../src/riichienv/kar/learning/export.py) 另核对并记录窗口和证据事件两个 ID。
- 每个非平凡窗口按席统计 `total = explicit + uniquely_inferred + censored`，按 `discard_response`／`kan_response` 及关闭结果交叉切片，出口校验计数与实际响应行动行相符。manifest 另汇总各 split。删失机会没有行动真值，不能制造训练行；Pass-only 不等于存在行为选择。

[完整半庄回归](../tests/kar/test_response_supervision.py) 从合法 136 张墙和实际原生选择生成轨迹，经规范化、重放、独立结算、完整半庄证书与 SQLite `GameWriter.finish`。抢杠和下位 Chi 反例比较当前冻结输入和后续 `SELF_PASS`、字段/positions/字节；竞争调用的 Chi 和 Pass 两种真实意图能产生同一公开结果，离线不应猜出在线的私有选择。

**解释边界：** 令 I 表示该机会有可辨认的行为而被保留；当前普通动作 CE 在模型容量足够时对保留样本学习 \(P(A\mid X,I=1)\)，不一定是完整 \(\mu(A\mid X)\)。更高优先级覆盖、同优先级仍可能的意图和缺失的裁定仍可能使保留依赖 A。温度标定不恢复被删失行为；不任意重加权、不改写 μ，也不引入未经论证的删失／集合标签损失。

## 2. 冻结精度与校准一致性

旧 [FrozenModelPredictor](../src/riichienv/kar/learning/inference.py) 的主干继承外层 autocast，而校准收集明确关闭。现在在 prefix 及候选的整个网络调用边界关闭 autocast，要求 FP32 参数及浮点缓冲；不靠输出 `.float()`。在 [model.py](../src/riichienv/kar/learning/model.py) 的原生 K/V cache 记录实际 autocast dtype／关闭状态，两种方向都拒绝跨精度后缀。已校准 `evaluate_complete` 匹配绑定的后端和 FP32，拒绝显式 BF16；未校准 BF16 验证保留。`fp32_no_autocast_v1` 进入统一推理 recipe，校准及发布依赖该绑定。未引入正式冻结 BF16 支持。

## 3. 只汇总一次并共享抽样

旧 [evaluate_complete](../src/riichienv/kar/learning/evaluation.py) 每 rank 拿到相同全局数据并生成同一报表。现在 [DistributedContext](../src/riichienv/kar/learning/distributed.py) 只向 primary 收集报告前的分片统计，primary 合并、生成后协调异常，再广播同一结果。`timings_seconds` 分报本地前向循环的最大 rank 墙钟（含数据准备与局部诊断）、primary 汇总和 bootstrap。

[GameClusters](../src/riichienv/kar/statistics.py) 对有相同有序合格游戏 ID 的指标共享 seeded 抽样，同权重可进一步复用分母，每指标保留自身权重；只保留一轮抽样向量，不预分配 games × replicates 矩阵。指标总体、验证完整覆盖、整场聚类、原种子、2000 默认重复和分位数定义不变。独立手写的旧估计器与新方法逐边界比对；[两进程 Gloo](../tests/kar/production/test_validation_execution.py) 验证所有 rank 得到同一报告及 primary 抛错时两边退出。

## 4. 选型隔离测试

[旧测试](../tests/kar/production/test_selection_and_roles.py) 修改 `final_rank_target`，而 `collate_examples` 实际读 `final_rank`；即便改对，均匀 rank logits 也可能产生相同 CE。新测试确实扰动入批的标签和非均匀 logits，校验未隔离的 CE 会变化、`cal_check` 的两种扰动都不影响 `cal_fit`，改 `cal_fit` 任何一种均改变选型损失。生产选型损失没有改定义。

## 5. 兼容及重建

`semantic_public_clock_frozen_decisions_v5_response_pass`、`explicit_and_unique_response_pass_v3`、`frozen_response_opportunities_v1` 和 `fp32_no_autocast_v1` 绑定输入、动作监督、机会统计和推理。SQLite 布局、Token Schema、546/21/21×4、模型规模、损失权重、公开 mask、按 train 固定 rho 与 KL 公式不变。受影响的是派生日志审计/验收、SQLite/历史/行 ID、索引与样本选集、训练比例证据/配方、缓存、精确续训、验证日志、冻结 logits/校准/策略/发布绑定。应从可信只读结果重放，输出全新的相应派生数据；固定 rho 仍从实际导出 train 一次确定，不拿旧计数或验证数据临时替代。旧权重同形状不意味着可按新合同精确恢复。不能仅修改 manifest、字段或 hash。

## 6. 实际运行

| 对照 | `4c09e9e` | 修复后 |
|---|---|---|
| 唯一活跃席抢杠 Pass | 完局导出 1016 行；指定 Pass 0 行，后续仅在线有标记 | 1017 行；指定 Pass 1 行，在线/离线均有标记，后续前缀逐字节相同 |
| 同前缀 `(221, 354)`：玩家 2 的 Pon/Pass | Pon 成立导出 1006 行；Pass 且 Chi 成立仅 1005 行，无该席私有标记 | 两分支均为 1006 行；显式 Pon 不变，Chi 分支新增可证 Pass 行与标记 |
| 高优先级 Pon 覆盖 Chi | 同一个公开结果无法区分 Chi 和 Pass | 两条真实原生轨迹仍同公开结果；离线标为删失，不写 Pass，不能冒充有在线私有 receipt |
| CPU 外层 BF16 | Policy 最大差 `1.71795e-4`；响应／名次最大差 `1.99785e-4`；raw 跨精度缓存两方向均未拒绝 | 三头在同一冻结权重和输入上相对不套外层 autocast 的结果均完全一致；raw 跨精度缓存两方向均拒绝 |
| Gloo 两个 rank 的全局报告调用 | `[1,1]`；primary 抛错时 worker 仍报成功 | `[1,0]`；两边收到同一报告，primary 抛错时两边收到协调异常 |

对拍的三样本双 rank 运行记录 `forward/aggregation/bootstrap` 约 `11.33/14.18/0.31 ms`，不是训练吞吐。另用相同 48 局合成统计、163 项区间、每组 2000 次重复，原／新报表哈希相同；随机抽样数 `10,464,000 → 192,000`，本机单次报表墙钟 `2.650 → 0.565 s`，非生产规模预测。

| 实际测试组 | 通过 | 失败 | 错误 | 跳过 |
|---|---:|---:|---:|---:|
| 最终 `tests/kar/core` | 94 | 0 | 0 | 0 |
| 最终半庄、精度、统计、Gloo、隔离与合同定向组（不含 core） | 496 | 0 | 0 | 4 |
| 最终合计 | **590** | **0** | **0** | **4** |
| 基线反例套件（不计入最终） | 0 | 8 | 0 | 0 |

4 个跳过均为不兼容的私有旧导出。基线新版 Pon/Pass 用例首先在缺机会报告字段处失败；另外的实跑前后对照直接证实旧版 Pass 行和私有标记缺失。早期测试入口有一次误写路径（0 项执行），混合显式路径的一轮有 15 个嵌套 fixture setup error；改为 core 目录与其他定向组各自发现，未修改 fixture、跳过条件或断言来消除错误。最终 Ruff、Ty、Git 空白、主设计 30 个合同值／72 个本地链接和模块索引通过。

同一源码 SHA-256 `a34ba2f179970e76bd2b286dd22824ac9d384341af44f334c0074fae5db2a746`；用隔离目录里与基线 Rust 源证据一致的此前实编扩展 `f3b452b8173bb5d423c5a11e86125682ed44814e768531a755def2fc6f678592`，现装扩展未替换。没有运行默认全套、参数更新、正式 M1/M2/M3、GPU、真实校准或全库重建，也没有改写既有产物。原始日志、JUnit、版本和完整文件清单在随附证据。
