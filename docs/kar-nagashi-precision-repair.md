# 末巡流局满贯与冻结后端精度修复

本轮以 `85884ab0be9b151e33d64c4fea45cb5056541cce` 为基线，修复审查确认的两个问题，并补充回归和产物迁移约束。记录日期：2026-11-21。它们分别影响响应类别的公开可行集合，以及冻结校准与推理是否采用相同数值设置；本轮没有执行真实数据训练，也没有据此推断已有权重的实际棋力损失。

## 1. 完整公开证据已经确定流局满贯时，排除普通荒牌

旧实现独立设置 `EXHAUSTIVE_DRAW` 和 `NAGASHI_MANGAN`：剩余摸牌数为零允许普通荒牌，公开资格又允许流局满贯。对于历史完整、候选舍牌之后仍至少一人满足流局满贯资格的末巡，普通荒牌实际上已经不可能：最后一张舍牌没有吃、碰、大明杠机会；无人荣和时，原生结算以流局满贯替代普通荒牌。

[Rust 公开规则](../riichienv-core/src/public_rules.rs)现在复用已有 `nagashi_player_possible` 判断，仅当 `allowed[NAGASHI_MANGAN]` 为真、历史完整且 `remaining_draws == Some(0)` 时关闭 `EXHAUSTIVE_DRAW`。复用已有位保留了规则开关及确定途中流局的优先级，也包含当前候选是否破坏本家资格、已有中张舍牌和舍牌被叫走的检查。Ron 位不因此关闭；历史或牌山状态未知时仍保守保留可能性。

这修正了训练行中的 `response_allowed`，使 masked CE 和推理响应混合不再给该场景中不可能的普通荒牌分配概率。类别顺序、动作 ID、网络形状、真实结算及支付没有改变。

### 回归证据

[完整物理牌局回归](../tests/kar/test_nagashi_consistency.py)分配全部 136 张牌且每张仅出现一次，逐步验证原生合法动作，再核对公开 mask、响应标签和独立结算。六种路径覆盖无副露、吃、碰、大明杠、暗杠和加杠；新增本人最后舍牌保留／破坏资格、三种未知上下文和关闭流局满贯规则的案例。Rust 单元测试还覆盖多名合格者、被叫走的舍牌、剩余一张、确定途中流局与 Ron 优先。

同一新增断言在旧实现上得到 **7 failed、5 passed**：六种合法轨迹及本人最后保留资格时，旧 mask 错误保留普通荒牌；控制案例通过。旧测试中“除流局满贯以外所有位不变”的假设也已改成明确验证普通荒牌随资格丢失恢复，其余类别继续逐位比较。

## 2. 冻结的是实际 IEEE FP32 后端，而不只是 Tensor dtype

参数和输出为 FP32、autocast 关闭，并不能排除矩阵运算内部使用 BF16 或 TF32。PyTorch 2.15 官方 [MKLDNN](https://docs.pytorch.org/docs/2.15/notes/mkldnn.html) 和 [CUDA](https://docs.pytorch.org/docs/2.15/notes/cuda.html) 文档说明了这些公共精度开关。此前的 CPU 小模型复现中，仅改变 MKLDNN matmul 模式，权重、Tensor dtype 和旧 binding 都不变，却使动作 logit 产生约 `4.47e-8` 的最大差异，足以让严格校准门槛附近的动作退出支持集合。

新增 [learning.precision](../src/riichienv/kar/learning/precision.py) 读取实际生效的后端设置。冻结入口只接受 IEEE FP32；将当前支持版本中默认 `none` 规范化为 `ieee`，拒绝有效的 BF16、TF32 等非 IEEE 模式。合同保存设备类型、Torch 版本、SDPA 后端集合、确定性算法开关，以及 CPU MKLDNN enabled 或 CUDA BLAS preference。实现不修改宿主的进程级后端设置。

[FrozenModelPredictor](../src/riichienv/kar/learning/inference.py)在构造时绑定合同，在每次前向之前和之后复验。单条 policy 前缀、候选后缀以及校准 archive 的 Response/Rank 批量辅助头均使用相同边界。前向期间改变后端时，该次结果会被拒绝，不返回策略 logits 或归档校准行。严格门控的比较式和阈值没有放宽。

[完整验证](../src/riichienv/kar/learning/evaluation.py)在显式 `precision="fp32"` 时也要求 IEEE，包括尚未校准的验证；原有未校准 `precision="bf16"` 和 BF16 训练路径保持可用。底层通用 `PrefixCache` 结构没有增加后端字段；冻结 predictor 通过包住前后缀调用的检查，禁止它管理的缓存跨后端设置复用。

校准计划、tau 计划和 [冻结发布](../src/riichienv/kar/learning/deployment.py)保存合同。M2 前置证据从归档计划解析该合同来重算 binding，不要求当前训练宿主等同于当时的评估宿主。元数据模块和 recipe 计算不导入 Torch。

### 回归证据

- [冻结后端测试](../tests/kar/m1/test_frozen_backend_precision.py)：非 IEEE 构造拒绝、门槛拟合后切换、前后缀之间切换、前向内部切换、IEEE 下 MKLDNN 开关变化、继承的全局精度、默认值规范化及无 Torch 的元数据解析。
- [完整前向保护](../tests/kar/production/test_backend_forward_guards.py)：校准辅助头不能写入错误精度的行；未校准 FP32 验证拒绝错误后端；BF16 验证继续通过；异常后恢复模型训练模式。
- [独立迁移回归](../tests/kar/production/test_review_frozen_backend_contract.py)：保持当前规则、源码和权重，只将 recipe 换回上一版并重算摘要，旧校准和旧 release 仍被拒绝。它验证精度迁移本身，避免仅靠本轮 mask 升级挡住旧产物的假阳性。

最初四个后端拒绝案例在原始基线上全部因没有抛错而失败，修复后通过。CPU 的配置与前向是实际执行；CUDA 分支只检查公共 TF32 配置及拒绝行为，没有初始化 CUDA，不构成 GPU 数值实测。

## 3. 活动合同与已有产物

| 合同 | 当前值 |
|---|---|
| `native_rules_version` | `riichienv_native_rules_v4_nagashi_mask` |
| `response_mask_version` | `public_rule_response_mask_v5_certain_nagashi` |
| frozen inference recipe | `fp32_ieee_single_prefix_v3` |
| backend metadata | `frozen_fp32_backend_v1` |

[版本回归](../tests/kar/test_semantic_versions.py)明确拒绝上一版 native v3 / mask v4，旧原生扩展在创建产物之前即提示重新编译。[数据集回归](../tests/kar/m1/test_design_versions.py)也覆盖 SQLite 元数据、manifest 和摘要全部一致但语义仍为旧版本的情况。支持统计、校准、checkpoint 和 release 通过完整合同拒绝不兼容产物。

升级顺序是：重新构建原生扩展，从可信事件重新生成或核验 mask 和依赖的数据导出，再按实际依赖重新生成支持统计、校准、tau 选择及发布证据。给旧 manifest 换版本或重算 hash 不能改变其语义。源码及原生构建身份原本就参与精确续训／发布校验；旧 checkpoint 即使参数形状相同，也不能直接声称满足新合同，权重迁移需要独立且明确的流程。

所有冻结 release（包括未校准的 prior-only）现在都绑定后端；更换设备类型或有效后端配置需要重新导出，已校准策略还需要在对应环境重新校准。本合同检查明确记录的运行设置，不承诺不同硬件、内核或底层库之间逐位相同。

## 4. 本轮验证范围

执行环境为 Python 3.12.14、PyTorch `2.15.0+cpu`、NumPy 2.4.2、Rust 1.92.0。构建使用当前 Rust 源码的 release 扩展。Python 动态库路径因宿主 `sysconfig` 保留旧安装目录而需在 Rust 全 feature 测试时显式指定实际目录；仓库构建规则没有因此修改。

| 检查 | 结果 |
|---|---|
| Rust pure core | 347 passed |
| Rust all targets / all features | 384 passed |
| 原生 release wheel | 构建成功，Python 已载入新 native v4 |
| 全部 `tests` 功能回归 | 3726 passed、42 skipped、2 deselected |
| 有界 CPU 梯度、训练和恢复检查 | 12 passed、1 skipped、33 deselected |
| 最终观察／后端／旧 recipe／阶段前置定向回归 | 222 passed；为上述套件的相关子集，不额外累加总数 |
| Ruff format / Ruff check / Ty | 全部通过 |
| Rust fmt / Clippy（all targets / all features，`-D warnings`） | 全部通过 |
| `git diff --check` | 通过 |

功能回归使用 `pytest -q tests -k 'not full_validation_only_primary_reports_broadcasts_and_coordinates_failure'`；排除的是先前实际遇到宿主 `Operation not permitted` 的两个 Gloo 多进程案例，没有把它们计为通过。有界更新采用仓库 CI 中的五文件门禁，显式开启 `--allow-parameter-updates -m parameter_updates`，并保留其 `not real_gloo and not real_production_global_sum` 选择条件；其中私有 CPU 数据集不可用的案例跳过。CI 配置的 Python 3.14 未在本地执行。

未运行真实数据训练、校准作业或棋力评估；没有修改已有数据集和模型产物。测试说明覆盖场景的代码行为已修正，不说明已经评估训练收益或完成所有部署硬件的数值验收。
