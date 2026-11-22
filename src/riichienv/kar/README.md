# KAR V2 源码导航

本包以 RiichiEnv 为宿主，旧 KAR V1 模型/训练实现与 V1 历史文档已从活动分支移除（保留共用动作映射 `actions.py`、规则特征 `rule_features.py`、回放与核心校验模块）。

- **唯一活动工程合同**：[docs/KAR-V2-CONTRACT.md](../../../docs/KAR-V2-CONTRACT.md)（v2.5）
- **KAR V2 核心实现**：`src/riichienv/kar/v2/`
  - `model.py`：`KARV2Actor`（4 档配置）、独立 `OracleCriticV2`、`fp32` 掩码 PPO/KL/熵与 `belief_auxiliary_loss`
  - `grp.py`：`PureStateGRP`、`FrozenGRP`、全半庄连续 GAE 与 `evaluate_grp_calibration`
  - `rl_contract.py`：`CriticWarmupGate`、`PFSPOpponentPool`、`FrozenGAERolloutBatch`
  - `benchmark.py`：端到端 Rust 编码与 PyTorch 实测 Benchmark（`python -m riichienv.kar.v2.benchmark`）
- **Rust 原生特征引擎**：`riichienv-core/src/v2_features/` 与 `riichienv-python/src/v2_features.rs`
