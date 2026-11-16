<!-- kar-module: riichienv.kar.learning.model -->
<!-- api: RMSNorm, block_attention_mask, InputEmbedding, TransformerBlock, ModelOutputs, PrefixCache, KARTransformer -->
# learning.model — 三头 Transformer 与共享前缀

源码：[model.py](../../../../src/riichienv/kar/learning/model.py) · [learning 索引](../README.md)

## 接口

`KARTransformer(config=None)` 构建并随机初始化网络，真实参数总量必须符合 `ModelConfig.parameter_count`。`forward(ModelBatch)` 返回 `ModelOutputs`：policy `[B,546]`、response `[B,21]`、rank `[B,21,4]` 和 ACTION 存在标记。

`encode_prefix(batch)` 返回 `PrefixCache` 与前缀 policy logits；`score_actions(cache, action_batch)` 共享同一决策 K/V，返回各独立假设 ACTION 的 response/rank logits。

`InputEmbedding`、`TransformerBlock`、`RMSNorm`、`block_attention_mask` 是实现组件，不是独立训练入口。

## 架构与因果合同

五个 tile 字段独立 Embedding、40D 数值投影、RMSNorm、QKV/RoPE/SwiGLU、零 dropout。三头读出 logits 保持 FP32；不能借 BF16 改写 loss 的数值合同。

DECIDE 不可看 ACTION 后缀；多个候选分支互不可见。状态块置换/padding 不应改变有效结果。forward 不接收真实 E/R 标签，真实动作仅以明确的条件后缀出现。

## 缓存与失败

缓存绑定模型 owner、参数/缓冲版本、上下文、设备/配置及缓存 Tensor 状态。更新、load_state_dict、跨观察者/决策或修改缓存后不得复用。CPU `TokenHistoryCache` 是另一层，不是该 K/V 缓存。

非法 block/valid 形状、参数数不符、缺前缀上下文或过期缓存会报错。模型是库原语：构造/forward 本身会占内存、进行计算，不等于已经得到上层作业授权。

## 测试与状态

对应 [m1/test_model.py](../../../../tests/kar/m1/test_model.py)、[core/test_frozen_inference.py](../../../../tests/kar/core/test_frozen_inference.py)。本轮未创建任何网络、运行 forward/backward 或测试；网络结构已编码不代表权重已训练或棋力有效。
