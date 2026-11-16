<!-- kar-module: riichienv.kar.learning.probe -->
<!-- api: cpu_probe -->
# learning.probe — 含更新的有界 CPU 管线诊断

源码：[probe.py](../../../../src/riichienv/kar/learning/probe.py) · [learning 索引](../README.md)

## 职责与接口

`cpu_probe(dataset, output, authorization=None)` 在默认拒绝许可后执行历史 Small CPU 诊断：feature-on/off 各16次更新、有效 batch8，再对8条 train 样本作32次更新的过拟合检查，最后运行一条决策的候选推理 benchmark。

它会真实 backward / optimizer.step，不能因名称 probe 或运行在 CPU 就说“没有训练”。没有完整许可时不应创建输出并开始计算。

## 对照与输出

两种特征模式必须相同初始化、split/seed/样本顺序且参数数相同；过拟合检查只使用 train。输出 `cpu-probe.json` 及各子运行日志/检查点，记录有限更新、过拟合下降、候选后缀速率、进程内存与来源 hash。

报告明确 `complete_M1_probe=false`、不声称棋力，并列出未做的全 train Epoch、GPU/费用、正式标定、M2 和 M3。报告的本地诊断零云资源字段不是外部云账单核验。

## CLI 与边界

独立 `python -m riichienv.kar.learning.probe` 接收 `--dataset --output --update-authorization`。输出须为新目录，不能覆写历史证据。正式同初始化/顺序的完整 Epoch 消融见 [orchestration](../orchestration/README.md)，不要用本结果替代 M2 的完整 M1 前置证据。

## 测试与状态

相关原语见 [m1/test_training.py](../../../../tests/kar/m1/test_training.py)、[core/test_authorization.py](../../../../tests/kar/core/test_authorization.py)。本轮未执行该入口、过拟合或模型更新；历史诊断结果仅对其原始范围与源码有效。
