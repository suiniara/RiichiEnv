<!-- kar-module: riichienv.kar.learning.gpu_preflight -->
<!-- api: representative_indices, run_preflight -->
# learning.gpu_preflight — 目标机器有界 CUDA 预检

源码：[gpu_preflight.py](../../../../src/riichienv/kar/learning/gpu_preflight.py) · [learning 索引](../README.md)

## 接口与授权顺序

`representative_indices(lengths)` 选择短/中位/长尾等代表长度；`run_preflight(dataset_dir, max_reserved_gib=4.5, scratch_dir=None, authorization=None)` 在目标设备进行 Small 小样本核验。

先核验更新许可，再打开实际 train 数据集并再次比较其 manifest，之后才能改随机种子、查询 CUDA/显存或创建模型。用 A 数据许可不能更新 B。

## 实际执行内容

包括 CPU/GPU FP32 logits/梯度对照、适用时 BF16 forward/backward、全部候选 dense/KV 一致性与速率、**一次实际 AdamW step**、陈旧缓存拒绝、CUDA RNG/检查点恢复。故属于有参数更新的诊断，不是“只读设备检查”。

当前历史目标机预检保留 1..4.5 GiB 进程 reserved 上限和其他进程余量等限制；它不是正式服务器 trainer 的通用显存上限。资源不满足应报告该预检未完成，不能代替代码编写，也不能悄悄改用控制机。

## CLI 与产物

独立入口接收 `--dataset --output --max-reserved-gib --update-authorization`。报告含实际设备、运行时、代表样本、精度差、峰值显存、参数数和 round-trip 结果；`full_epoch_started=false`。临时检查点放指定 scratch，拒绝覆盖已有结果。

## 测试与状态

数据许可前置的回归见 [production/test_review_job_guards.py](../../../../tests/kar/production/test_review_job_guards.py)。完整参数更新测试仍需显式许可。**本轮未查询 GPU/内存、未访问远端、未执行任何预检或更新。**
