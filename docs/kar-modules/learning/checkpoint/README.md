<!-- kar-module: riichienv.kar.learning.checkpoint -->
<!-- api: implementation_identity, save_checkpoint, load_checkpoint -->
# learning.checkpoint — 历史诊断检查点

源码：[checkpoint.py](../../../../src/riichienv/kar/learning/checkpoint.py) · [learning 索引](../README.md)

## 职责与接口

为有界 CPU/CUDA 预检保存精确状态。`save_checkpoint(path, model, optimizer, scheduler, run_contract, sampler_state, step)` 写 payload 和 JSON sidecar。`load_checkpoint(..., run_contract, restore_rng=True)` 核验后恢复模型、可选优化器/调度器、RNG，并返回 step/游标/metadata。

`implementation_identity()` 标识这套历史学习实现的文件/Git 来源，不等同整个生产包的完整 release 身份。

## 数据与读取边界

payload 包含模型、optimizer、scheduler、sampler 和 Python/NumPy/Torch RNG；CUDA RNG 仅在实际初始化后保存。CPU 不能静默消费一次要求精确 CUDA 恢复的状态。

先检查存在性/大小、checkpoint SHA、metadata SHA、模型/数据/规则/recipe/运行时与 payload 一致性，再使用 `weights_only=True` 受限加载。只允许可信来源；hash 不能认证恶意文件的发布者。

## 与生产检查点的区别

历史实现分别替换 payload 和 sidecar，不提供生产代际指针的联合发布语义。正式全 Epoch/DDP 使用 [production_checkpoint](../production_checkpoint/README.md)，不要把两个接口互换或声称旧格式具有相同恢复保证。

无独立 CLI。写入/恢复会改变文件、内存权重及随机流；即便没有 step，也不能当作无副作用信息查询。

## 测试与状态

对应 [m1/test_training.py](../../../../tests/kar/m1/test_training.py)及预检相关回归。版本 `kar_exact_resume_v1`。本轮未保存或读取任何模型检查点。
