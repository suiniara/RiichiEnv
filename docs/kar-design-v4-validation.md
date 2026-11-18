# KAR-v1.2.2 / RiichiEnv V4 本次文档校验

[唯一活动主设计](KAR-v1.2.2-DESIGN.md) · [相对旧附件的差异与迁移](kar-design-v4-migration.md)

**对应实现：** `suiniara/RiichiEnv` · `kar/riichienv-v1-20261117` · `f70684ef7aa69e56e6a530dcca13326fbf2c83b5`。开始时本地 HEAD 与 fork 同名远端均为此提交，工作区干净；无后续实现提交需要合并。本轮仅改 6 个 Markdown 文件，结束时 HEAD 不变；无代码、Schema、权重或数据修改，未提交或推送。

## 实际校验结果

在本机已有 Python 3.14.9 中执行 `verify_design.py`，**10 项文档检查通过，退出码 0**。脚本用只读 AST、仓库对象检查及少量纯计算核对合同；不加载 KAR 或 Torch 模块，不把纯计算称作模型回归。

| 校验 | 实际结果 |
|---|---|
| HEAD、改动范围与空白 | HEAD 与审查提交一致，仅有预期 6 处 Markdown 变更；`git diff --check` 通过，暂存区为空。 |
| 历史附件与双版本 | 通过 SHA-256 确认历史原件及工作副本为同一字节版本；主设计标题和版本字段同时标识 `KAR-v1.2.2`、`RiichiEnv V4`，独立区分文档修订。 |
| 版本、Schema 与 overlay | 将主文档表格同源码的 26 个常量逐项比较。分别重算完整 Schema canonical SHA 和历史 overlay 原始字节 SHA，均匹配源码且两者不同。 |
| 行数、时钟与计数范围 | 核对 H+64/H+65 vs C+1/C+2；`known_counts()` 向底层传 4 个区且无额外公开牌参数。`closed` 不在 `encode_prefix()` 允许决策阶段，终局 `public_hand` 走 kind 29 历史行。 |
| 审计池、导出与选型 | 纯算术探针：eligible=65,000、available train=60,000、exported train=29,999，选 Small；29,999/30,000/49,999/50,000 的规模边界与表格一致。 |
| 模型参数与输入/读出 | 从源码公式和尺寸复算参数：Tiny 2,085,771、Small 5,239,435、Medium 14,790,795、Large 26,536,587。静态核对 944 行 Embedding、完整五表均值、三头 FP32 投影边界和注意力隔离。未构建模型。 |
| 固定分母与受限 KL | 静态核对有效样本位、固定分母和权重；对选取的纯函数用合成先验核对 S 外概率不变、S 内质量为 0.5，以及 E1/E2 与文档计算一致。非证书验收。 |
| 500 局门禁与双口径 | 静态核对 `≥500`、response/rank 预登记及 passed 交集；对合成分支概率算得 train-only 未支持质量 0.25、有效未支持质量 0.5，二者正确分开。未拟合温度或校准。 |
| 代码与文档链接 | 检查变更文档内 80 处本地链接、39 个不同目标；代码/Schema 目标在对应 Git 提交中存在。外部引用 URL 未做网络可用性验证。 |
| 完整分发 | 正文 24 节。`KAR-v1.2.2-RiichiEnv-V4-DESIGN.md` 与仓库正文逐字节一致（56,178 字节）。两者 SHA-256 均为 `c30fc5e8c564fbf307abe60c112116a41a77151908fbc9288e62bb78437d36a5`。 |

详细输出与复现脚本位于工作区 `kar-fix-reports/kar-design-v4-doc-sync-f70684e/validation.json` 和 `verify_design.py`。分发快照不单独维护；用于访问相对代码链接时，按目标仓库 `docs/` 目录解释路径。

**未执行：** KAR 测试套件、真实牌谱验收、模型前向与反向、参数更新、GPU、校准、M3 或旧产物迁移。此前运行报告保留其历史范围，不计入这 10 项。