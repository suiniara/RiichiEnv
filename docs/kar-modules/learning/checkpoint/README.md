<!-- kar-module: riichienv.kar.learning.checkpoint -->
<!-- api: implementation_identity, save_checkpoint, load_checkpoint -->
# learning.checkpoint — 有界诊断检查点

源码：[checkpoint.py](../../../../src/riichienv/kar/learning/checkpoint.py) · [learning 索引](../README.md)

## 职责与接口

为有界 CPU/CUDA 预检保存精确状态。`save_checkpoint(path, model, optimizer, scheduler, run_contract, sampler_state, step)` 写不可变代际 payload，再原子发布 JSON 指针。`path` 是逻辑检查点名，例如 `checkpoint.pt`；实际文件名由同目录 `checkpoint.json` 的 `checkpoint_file` 指定。`load_checkpoint(..., run_contract, restore_rng=True)` 核验指针和该代文件后恢复模型、可选优化器/调度器、RNG，并返回 step/游标/metadata。

`implementation_identity()` 标识这套历史学习实现的文件/Git 来源，不等同整个生产包的完整 release 身份。

## 数据与读取边界

payload 包含模型、optimizer、scheduler、sampler 和 Python/NumPy/Torch RNG；CUDA RNG 仅在实际初始化后保存。CPU 不能静默消费一次要求精确 CUDA 恢复的状态。

先检查代际文件名属于该检查点命名空间，再检查存在性/大小、checkpoint SHA、metadata SHA、模型/数据/规则/recipe/运行时与 payload 一致性，然后使用 `weights_only=True` 受限加载。只允许可信来源；hash 不能认证恶意文件的发布者。

payload 与指针临时文件分别 flush/fsync，POSIX 平台也同步目录。只有完整新 payload 已落盘后才替换指针；发布失败不会覆盖旧 payload。成功发布后保留当前和上一代，清理更旧的已核验代。中断留下的未提交文件保留供核查，不自动当作有效检查点。

## 与生产检查点的区别

诊断与 [production_checkpoint](../production_checkpoint/README.md) 共用代际发布机制，但 payload、元数据合同和 RNG 范围仍各自独立，不能交换两个读取接口。诊断格式为 `kar_exact_resume_v2`；旧 v1 的单 payload/sidecar 不会被静默改名或重新签署，加载时明确要求迁移。正式全 Epoch/DDP 仍使用生产接口。

无独立 CLI。写入/恢复会改变文件、内存权重及随机流；即便没有 step，也不能当作无副作用信息查询。

## 测试与状态

对应 [m1/test_checkpoint_publication.py](../../../../tests/kar/m1/test_checkpoint_publication.py)、[m1/test_training.py](../../../../tests/kar/m1/test_training.py)及预检相关回归。发布故障回归覆盖 payload 写入失败、指针替换失败、保留上一代、路径越界和旧代损坏；这些 I/O 用例不执行模型前向、反向或参数更新。
