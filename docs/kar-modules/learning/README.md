<!-- kar-module: riichienv.kar.learning -->
# learning 子包 README

源码：[learning/__init__.py](../../../src/riichienv/kar/learning/__init__.py) · [KAR 总索引](../README.md)

## 职责与依赖

这是 KAR 的可选学习运行层。包入口本身只保留说明，不自动导入 Torch 或创建网络；调用方显式导入需要的子模块。多数模型、训练、标定和仿真模块需要工程的 `kar-ml` 依赖（Torch、NumPy）；不能因此让基础 M0 规则/协议模块强制依赖 ML。

## 模块导航

| 模块 | README | 定位 |
|---|---|---|
| `batch` | [说明](batch/README.md) | 输入组批、padding、规则特征视图 |
| `config` | [说明](config/README.md) | 模型 preset 与参数合同 |
| `model` | [说明](model/README.md) | 三头 Transformer 与 prefix KV |
| `dataset` | [说明](dataset/README.md) | 只读决策数据集与标签组批 |
| `export` | [说明](export/README.md) | 从通过 gate 的 raw 重建数据 |
| `losses` | [说明](losses/README.md) | 固定 rho、mask、safe gather |
| `metrics` | [说明](metrics/README.md) | 历史有界诊断指标 |
| `train` | [说明](train/README.md) | 历史 CPU 诊断与复用优化器原语 |
| `checkpoint` | [说明](checkpoint/README.md) | 历史 CPU/预检检查点 |
| `benchmark` | [说明](benchmark/README.md) | 单决策候选吞吐，不是 matches/s |
| `probe` | [说明](probe/README.md) | 含参数更新的 CPU 诊断 |
| `gpu_preflight` | [说明](gpu_preflight/README.md) | 含参数更新的有界 CUDA 预检 |
| `training_config` | [说明](training_config/README.md) | 正式训练配方与条件第三轮 |
| `sampling` | [说明](sampling/README.md) | 完整全局 epoch、尾 batch、空 rank |
| `distributed` | [说明](distributed/README.md) | 设备/进程组及协调失败 |
| `trainer` | [说明](trainer/README.md) | 正式完整 Epoch 训练器 |
| `evaluation` | [说明](evaluation/README.md) | 全 validation 诊断和 cal_fit 选择 |
| `production_checkpoint` | [说明](production_checkpoint/README.md) | 代际原子检查点与每 rank RNG |
| `run_publication` | [说明](run_publication/README.md) | 完成发布、恢复和累计吞吐 |
| `orchestration` | [说明](orchestration/README.md) | M1 on/off 与唯一 M2 调度 |
| `prerequisites` | [说明](prerequisites/README.md) | M1→M2 实际证据重验 |
| `inference` | [说明](inference/README.md) | 冻结模型到原始 logits |
| `calibration_job` | [说明](calibration_job/README.md) | 冻结 validation archive 与质量判定 |
| `validation_policy` | [说明](validation_policy/README.md) | cal_fit 配对对局选择 tau |
| `deployment` | [说明](deployment/README.md) | 冻结发布和受限加载 |
| `simulation` | [说明](simulation/README.md) | 成对完整基准与固定 M3 |
| `jobs_cli` | [说明](jobs_cli/README.md) | 冻结模型作业 CLI 接线 |

加上本页，共覆盖 28 个 Python 模块。

## 正式链与历史诊断的区别

正式链：`export → trainer/orchestration → calibration_job → prior release → validation_policy → finite release → paired benchmark → M3`。检查点由 `production_checkpoint` 管理，完成报告由 `run_publication` 发布。

`train`、`probe`、`gpu_preflight`、`benchmark`、`metrics`、`checkpoint` 保留用于历史有界诊断或共享原语。它们的 Small 参数更新、8 样本过拟合和候选后缀吞吐不等于完整 M1/M2/M3，也不代表棋力。

## 训练代码的 CI 验证

GitHub Actions 的 `pytest (KAR ML)` 任务在安装与执行阶段都显式选择 `kar-ml` dependency group，并在收集测试前直接导入 Torch、NumPy。缺少这些依赖时任务失败，不能把 `pytest.importorskip("torch")` 跳过后的结果当成训练代码验收。基础 Python 测试仍可独立运行。

本地对应命令为 `uv run --locked --python 3.14 --all-extras --dev --group kar-ml pytest tests/kar`。这是功能测试入口；带 `parameter_updates` 标记的诊断仍遵循根测试配置中的显式 opt-in，不由 CI 的依赖安装自动授权。

## 使用前必须满足

- 使用已核验的完整数据与当前合同；不把 synthetic 测试数据当作真实 Golden。
- 正式更新需要源码/数据/phase/recipe 匹配的独立许可；评测需要 plan/source 匹配的独立许可。
- 用全局真实样本数和固定训练 rho 归一化；dummy 不算真实样本。
- cal_check 只报告，不参与 epoch/tau/门控调节；test 不调参。
- 价格、测量来源、固定成本和历史账本都要明确；不能换目录或更改计价重置阶段预算。

示例和测试位置用于说明接口，不表示本轮已执行。完整公共约束见 [KAR 总索引](../README.md#共同约束)。
