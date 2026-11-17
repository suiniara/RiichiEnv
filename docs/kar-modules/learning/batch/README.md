<!-- kar-module: riichienv.kar.learning.batch -->
<!-- api: ModelBatch, canonical_candidate_ids, collate_inputs, feature_mask_draw, rule_feature_view -->
# learning.batch — 输入组批与规则特征视图

源码：[batch.py](../../../../src/riichienv/kar/learning/batch.py) · [learning 索引](../README.md)

## 职责与接口

`collate_inputs(examples)` 将 139 字节 Token 流解成按字段存储的 Tensor、padding、完整候选及 DECIDE/ACTION 索引，返回 `ModelBatch`。模型输入不包含 response/rank 真值。

`ModelBatch` 提供 `fields`、`candidates`、索引、context IDs、`legal_mask`、`size` 和设备迁移。`canonical_candidate_ids(candidates,size)` 对32槽 int64 ID 排序，拒绝空集合/重复真实ID/越界，padding=546 允许重复但不计入集合。此规范集同时用于 LEGAL_SET 池化和 K/V 缓存身份；它不是另一张 Embedding 表。真实 ID 仍为 0..545。缺 ACTION 使用索引标记，不伪造真实后缀。

## 特征模式

- `feature_mask_draw(seed, epoch, game_id, decision_id, draw=0, probability=.25)`：稳定 hash 的样本级命中，恢复/分布式不因 microbatch 划分改变。
- `rule_feature_view(batch, mask_samples, enabled=True)`：复制需修改字段，整条样本清除 HAND_COUNT flags[0..7] 和 ROUND_STATE numeric[18..19]/present。
- `enabled=False` 对所有样本禁用这些规则字段，并额外把当前 rank stage 置为 N/A。验证/推理不进行随机抽样置空。

具体字段以当前实现为准；同一条样本的 34 个牌种不可各自随机开关。不修改原 batch，也不改变 Embedding/网络参数尺寸。

## 使用与错误

由 [dataset](../dataset/README.md)和 [inference](../inference/README.md)组批，输入模型；真实标签由 `collate_examples` 单独构造。需要 NumPy/Torch，调用可能分配 Tensor，但不执行优化器。

空 batch、Token 字节长度错误、非法候选或 mask 不是 `[B]` bool 时拒绝，不能用 padding 掩盖错误。

## 测试与状态

对应 [test_legal_set_pool.py](../../../../tests/kar/m1/test_legal_set_pool.py)、[test_model.py](../../../../tests/kar/m1/test_model.py)。本轮未执行 backward/optimizer.step 或阶段任务；仅有界 CPU 输入/前向复验。
