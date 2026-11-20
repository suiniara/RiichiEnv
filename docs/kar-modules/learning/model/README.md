<!-- kar-module: riichienv.kar.learning.model -->
<!-- api: RMSNorm, block_attention_mask, InputEmbedding, TransformerBlock, ModelOutputs, PrefixCache, KARTransformer -->
# learning.model — 三头 Transformer 与共享前缀

源码：[model.py](../../../../src/riichienv/kar/learning/model.py) · [learning 索引](../README.md) · [输入链路合同](../../../kar-input-chain-repair.md)

## 接口与输入

`KARTransformer(config=None)` 按冻结 preset 随机初始化，真实参数数必须等于公式。`forward(ModelBatch)` 输出 policy `[B,546]`、response `[B,21]`、rank `[B,21,4]`，读出保持 FP32。

`InputEmbedding.forward(fields, candidates=None)` 对 LEGAL_SET 行使用完整 `batch.candidates`：复用 ACTION 的五张 scalar 表求组合向量均值。padding=546 不参与均值，真实 ID 唯一/非空/最多32，按全局ID规范排序。非舍牌按 manifest/Schema 使用 N/A 因子；没有新候选表或可训练参数，numeric[16] 数量语义保留。

`encode_prefix(batch)` 与训练 dense 共用 `_hidden`；`score_actions(cache, action_batch)` 的单 ACTION 不重复加入候选池。前缀必须有一个 LEGAL_SET；teacher ACTION 不影响 DECIDE，独立分支互不可见。

## 架构与缓存

五张独立 tile 表、40D 数值投影、RMSNorm/QKV/RoPE/SwiGLU、零 dropout 与三个头均保留。缓存绑定 owner、参数及计算 buffer 版本、完整规范候选集、输入/模型合同、上下文、设备和自身 Tensor 状态。旧输入版本、跨决策、集合改变或缓存/模型被改动均拒绝；CPU TokenHistoryCache 是另一层。

当前 `MODEL_VERSION=kar_transformer_v3_semantic_inputs` 标明模型使用现行语义输入合同，不能因参数形状相同声称旧权重精确兼容。完整 Schema ID/hash 没有因此改名。底层 `PrefixCache` 绑定 dtype/autocast；冻结推理的实际数值后端合同由 [FrozenModelPredictor](../inference/README.md) 在完整前后缀调用外层检查。

## 验证与边界

对应 [test_legal_set_pool.py](../../../../tests/kar/m1/test_legal_set_pool.py)、[test_model.py](../../../../tests/kar/m1/test_model.py)，覆盖 CPU forward、精确向量、dense/cache、隔离及参数量；有界梯度和恢复测试需要 `--allow-parameter-updates` 显式开启。本轮执行范围及结果见[修复记录](../../../kar-nagashi-precision-repair.md)，不等同于真实数据训练或 GPU 验收。
