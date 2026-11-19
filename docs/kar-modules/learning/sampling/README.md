<!-- kar-module: riichienv.kar.learning.sampling -->
<!-- api: UpdateGroup, GlobalEpochSampler, validation_updates -->
# learning.sampling — 全局完整 epoch 与精确游标

源码：[sampling.py](../../../../src/riichienv/kar/learning/sampling.py) · [learning 索引](../README.md)

## 主要接口

`GlobalEpochSampler(lengths, seed, effective_batch, microbatch, rank=0, world_size=1, bucket_window=...)` 以相同全局顺序组织真实样本，再分到各 rank。`peek()` 返回 `UpdateGroup`，成功更新后 `commit(group)`，epoch 完成后 `next_epoch()`。

`state_dict/load_state_dict` 恢复冻结次序、epoch 与已提交游标。数据长度采用只读 tuple，摘要在初始化时生成；每次生成新 epoch 排列同时计算原算法摘要。`state_dict` 及 trainer 的 epoch 记录读取缓存摘要，不重扫索引；恢复仍核对当前长度、重新生成排列并验证摘要及游标。`updates_per_epoch`、`exhausted` 供完整训练调度。`validation_updates(n)` 给出 quarter-epoch 及末端边界。

## 合同

长度分桶不丢样本，完整 epoch 每个真实样本恰好一次；尾 batch 不复制样本凑齐。空 rank/microbatch 由 trainer 生成连图 dummy，`UpdateGroup.global_examples` 仍只计真实样本，不能把 dummy 加进损失归一化或吞吐分子。

先 peek、后执行、再 commit。失败时不前移游标，不能保存“已经消费”但优化器未提交的状态。恢复必须位于合法全局更新边界并匹配长度/顺序与配置。

## 使用与边界

纯元数据模块，不导入 Torch、不执行训练或读数据库。消费者是 [trainer](../trainer/README.md)，历史 sampler 不能替代这个生产完整 epoch 合同。数据长度信息来自已核验 `DecisionDataset`。

## 测试与状态

对应 [production/test_planning.py](../../../../tests/kar/production/test_planning.py)、[production/test_sampler_digest_cache.py](../../../../tests/kar/production/test_sampler_digest_cache.py)。采样摘要、下一组恢复和内存实测见[性能报告](../../../kar-history-sampler-performance.md)；没有推进真实训练游标。
