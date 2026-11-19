<!-- kar-module: riichienv.kar.learning.inference -->
<!-- api: model_weights_sha256, FrozenModelPredictor -->
# learning.inference — 冻结网络预测器

源码：[inference.py](../../../../src/riichienv/kar/learning/inference.py) · [learning 索引](../README.md)

## 职责与接口

`FrozenModelPredictor(model, train_manifest_sha256, use_rule_features=True)` 把已加载模型设为 eval，计算真实权重/缓冲 hash 和推理 recipe，生成 `RuntimeBinding`。`model_weights_sha256(model)` 覆盖实际 state_dict 与缓冲，不以文件名替代权重身份。

`policy(snapshot, candidates, context)` 构造本家 prefix、创建 K/V cache并返回原始546 logits；`candidates(snapshot, candidates, supported_discards, context)` 对全部 S ACTION 分支返回21响应及21×4名次 logits。

## 模型与缓存合同

两条路径 `no_grad`、显式关闭完整模型调用的 autocast、要求 FP32 参数/浮点缓冲；推理 recipe 绑定 `fp32_no_autocast_v1`。推理随机规则置空为零；特征模式不改变模型尺寸。原生 K/V 另记录 autocast 模式，不能跨精度给后缀使用。温度、公开响应 mask、支持罚项、KL 与采样由 [runtime](../../runtime/README.md) 协调，不用真实 E/R 标签挑分支。

单次决策 K/V 同时绑定 context ID、snapshot 字节身份、本地历史会话以及完整合法动作集合。模型参数/缓冲、配置、特征模式和训练模式的冻结检查仍在。

另外，`policy()` 使用现有 `TokenHistoryCache` 编码历史。最多保留每一观察席的一个流，使用 `ObserverLedger` 的半庄生命周期身份，不写入 snapshot 序列化或 Token。跨小局只追加；换半庄即使开局完全相同也重建，前缀被修改、历史倒退、positions 改变或迟到 `SELF_PASS` 插入按通用缓存校验失效。没有账本身份的外部快照用新缓存编码。历史缓存不是跨决策 Transformer K/V，不改变注意力。

## 使用与副作用

传入的模型必须已通过来源/构建核验，生产通常从 [deployment](../deployment/README.md)加载。构造会读权重并改变 eval 模式；调用会分配 Tensor并运行真实网络，不因 no_grad 就成为“没有计算”。本模块不创建优化器、更新参数、拟合温度或启动对局。

## 测试与状态

对应 [core/test_frozen_inference.py](../../../../tests/kar/core/test_frozen_inference.py)、[core/test_frozen_history_cache.py](../../../../tests/kar/core/test_frozen_history_cache.py) 与 [m1/test_model.py](../../../../tests/kar/m1/test_model.py)。合成 CPU 前向和缓存实测见[性能报告](../../../kar-history-sampler-performance.md)，不作棋力证据。
