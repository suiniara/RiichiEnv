<!-- kar-module: riichienv.kar.learning.losses -->
<!-- api: Labels, LossSums, loss_sums -->
# learning.losses — 固定 rho 与 sentinel-safe 三头损失

源码：[losses.py](../../../../src/riichienv/kar/learning/losses.py) · [learning 索引](../README.md)

## 接口与形状

`Labels` 保存 action/response/final-rank 目标、有效/适用/观测位与 `[B,21]` bool response_allowed；`response_valid` 是 action_valid、applicable、observed 的合取。`loss_sums(outputs, labels, legal_mask)` 返回三头 sum、真实 batch 大小和各头有效数 `LossSums`。

模型输出为 `[B,546] / [B,21] / [B,21,4]`。`LossSums.objective(global_examples, world_size, rho_e, rho_v)` 使用 `W/global_N * (sum_A + .25 sum_E/rho_E + .1 sum_V/rho_V)`。

## 关键合同

rho 固定取实际 train manifest，不按 microbatch/rank/validation 重算。DDP/累积使用全局真实样本总数，不除本地条件标签数。

无效行先用安全索引和 dummy mask 隔离，再 gather/CE；不能先 gather 非法 sentinel 后乘零。有效行的非法 action/event/rank、全 false mask 或 target 被 mask 排除必须报错，不 clamp 或 unmask。value_valid 必须蕴含真实有效响应。

原始三头 logits 均须有限；不以 nan_to_num 隐藏错误。空 response/value 子集给连图零贡献和零有效数，诊断均值应为 None，不冒充零损失通过。

## 使用与副作用

由历史和正式训练器、验证/标定共同复用。函数计算 Tensor 运算并可形成梯度图，但不自行调用 backward 或 optimizer；实际更新须由受授权训练入口执行。真实标签不进入模型 forward。

## 测试与状态

对应 [m1/test_losses.py](../../../../tests/kar/m1/test_losses.py)、[production/test_training_contracts.py](../../../../tests/kar/production/test_training_contracts.py)、[test_selection_and_roles.py](../../../../tests/kar/production/test_selection_and_roles.py)。本轮未计算 loss、反向传播或运行测试。
