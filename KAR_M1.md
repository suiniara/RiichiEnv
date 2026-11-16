# KAR M1 — 可运行的 Small CPU 学习闭环

> 状态更新：本文保留早期有界 CPU 诊断接口。完整训练、全量验证、发布和 M1/M2/M3 调度代码见 [完整代码说明](docs/kar-production-code.md)。所有更新入口现已默认关闭；下方旧诊断命令缺少独立授权时会拒绝执行，不是现在开始训练的建议。当前按用户要求先写齐代码，运行验收统一后置。

M1 学习代码位于 `src/riichienv/kar/learning/`。它是独立可选依赖模块，不复用现有 82 动作 Actor-Critic，不改变 M0 冻结 Schema 或 Rust Core。基础 `riichienv.kar` 导入不需要 PyTorch。

## 当前范围

已实现模型、共享历史训练数据、安全三头损失、优化器/累积、特征增强、检查点/精确恢复、条件指标、候选 K/V 推理和本机 CPU 有界探针。

**CPU smoke ≠ 完整 M1 探针完成。** 默认使用有效 batch 8、16 updates/特征模式及小型真实数据子清单；正式设计的 Small 有效 batch 128、feature-on/off 各最多一轮，以及 GPU 显存/成本和完整校准仍需要另行运行。没有启动付费资源、M2 正式训练或 M3 棋力评测。

## 环境

现有 Python 3.14 环境可直接使用；`kar-ml` 单独锁定 PyTorch 2.15 与 NumPy。Windows 本次使用 `torch 2.15.0+cpu`，不初始化 CUDA。

在 RiichiEnv 工程内：

```bash
uv sync --group kar-ml --inexact
```

旧 `riichienv-ml` 的 Ray/PPO/CQL/wandb 依赖不是 KAR M1 的必要依赖，不自动安装或上传实验日志。

## 数据导出

只接受 M0 PASS 的来源绑定验收与 train/validation/test 清单；Golden 禁入训练，整场切分不变。输出目录必须新建，且不能位于原始数据目录内。

```bash
uv run --no-sync python -m riichienv.kar.learning.export --acceptance ../kar-m0-reports/final-acceptance-20261116 --proto ../kar-m0-assets/mahjongsoul.proto --output ../kar-m1-data/new-probe --train-games 6 --validation-games 2 --test-games 0
```

`--all-games` 才展开完整已验收清单；不要在首次 CPU 验收中无提示触发大规模导出。`--resume` 必须保持相同输入、协议与导出实现指纹。

- 数据格式：`decisions.sqlite3` + `manifest.json`；数据只包含结构化 SQLite、JSON 和冻结 Token 字节，不用 pickle 保存数据。
- 同 game/observer 的历史只存一次；决策保存历史前缀长度、63 行状态/DECIDE、ACTION、完整候选、单独标签。
- 每场只有完整重放、独立结算、输入与终局全部通过才事务提交；未完成数据不能加载训练。
- `action_valid / response_applicable / response_observed / value_valid` 分开；0..3 rank 是数据层从最终 1..4 位明确转换。
- 仅实际舍牌有 response/rank 监督。可信执行动作标为 explicit；NEXT_DRAW 完整闭合且截止前有实际回应选择时，可唯一证明的 PASS 标为 `inferred_unique_next_draw_pass`。被高优先级覆盖的私有意图不猜。
- response public mask 从截止前 snapshot 重建，不读取实际响应反推；标签和来源身份不进入模型 Tensor。
- rho_E/rho_V 固定从实际导出的 train 子清单计算，不用 validation/test 或每个 microbatch 的比例替代。

## 精确模型与掩码

Small 为 **6 层 × 256、4 heads、SwiGLU 704，5,239,435 参数**。五个 tile 列与所有其他类别字段用独立 Embedding，共944行；numeric/presence FP32拼成40维，无bias投影。

逐层采用 RMSNorm、无bias QKV/输出投影、逻辑位置 RoPE、SwiGLU、零dropout。DECIDE 读546动作先验；ACTION读21响应和21×4名次。

- H因果，S可读H与双向S，D不可读ACTION，ACTION只读同决策前缀和自己。
- SDPA boolean mask 的 True 为允许；padding不是key，padding query只看BOS后清零。
- `encode_prefix` / `score_actions` 为eval/no-grad，同一决策前缀K/V共享，候选分支不互见。
- 缓存绑定模型身份/参数版本、观察者决策ID、device/dtype；参数更新、load或缓存修改后拒绝旧缓存。不跨决策复用当前状态K/V。

## 训练与特征对照

```bash
uv run --no-sync python -m riichienv.kar.learning.probe --dataset ../kar-m1-data/new-probe --output ../kar-m1-runs/new-cpu-acceptance
```

该命令运行 Small 的 on/off 各16步与单独8样本32步过拟合诊断。每个模式起始权重和实际样本顺序均以SHA记录并核对。过拟合诊断故意重复 train 样本，不代表自然分布训练或验证收益。

训练总目标为 `L_A + .25 L_E + .1 L_V`，各项使用 local sum、固定训练rho和完整update全局N归一化。辅助标签在gather/CE前隔离哨兵；有效越界、实际target被规则mask排除、原始非有限输出均硬失败。无有效标签保持连图零梯度，条件指标为N/A。

AdamW：lr3e-4、betas(.9,.95)、eps1e-8；value_head.weight decay .05，其他矩阵 .01，bias/norm 0；有限梯度检查、global norm1裁剪，成功step后更新scheduler。CPU进程使用至多2个Torch线程。

Feature-on 的训练样本级p=.25同步遮蔽34个HAND_COUNT flags0..7及ROUND_STATE numeric18/19/presence；off还把当前rank stage置N/A。历史、其他字段和模型结构不改。随机流由seed/epoch/game/decision派生，不受worker/rank改变；验证/推理随机遮蔽为0。基底M0 Token不被原地修改。

单模式诊断及精确恢复：

```bash
uv run --no-sync python -m riichienv.kar.learning.train --dataset ../kar-m1-data/new-probe --output ../kar-m1-runs/new-single --preset small --max-updates 16 --effective-batch 8 --microbatch 1
```

中断后加 `--resume` 并保持参数一致。检查点包含模型、optimizer/scheduler、Python/NumPy/Torch RNG、sampler/增强上下文、固定rho、合同/数据/实现指纹及运行版本。weights-only受限加载；哈希、数据、配置或代码不一致拒绝精确续训。不把旧checkpoint偷偷当成新协议模型。

## 输出与结论

- `cpu-probe.json`：有界管线验收及明确未运行项。
- `feature-on/`、`feature-off/`、`overfit-eight/`：`run_manifest.json`、`updates.jsonl`、`checkpoint.pt`、checkpoint JSON元数据。
- NLL/Brier以实际有效数报告；验证默认最多64行，标明是否覆盖完整验证集。小样本、少步数差异不作规则特征有效性或棋力结论。
- 候选吞吐基准只衡量一条验证决策上的CPU架构推理（随机初始化权重），不是实战对局速度。未测GPU显存、未拟合温度或未运行M3时明确标为未测。

## 测试

```bash
uv run --no-sync pytest -q tests/kar/m1
```

测试覆盖精确参数、前缀/候选隔离、dense/cache等价、特征置空、无效标签/NaN、AdamW分组、梯度累积、真实CPU Gloo DDP、精确检查点及中断恢复。Windows测试使用本机loopback的原生Gloo工厂（PyTorch 2.15默认工厂忽略Gloo options），不修改系统网络配置，不伪造分布式梯度结果。

无私有原始数据/探针数据的安装中，相应集成测试会明确skip，不能据此宣称真实CPU探针完成。
