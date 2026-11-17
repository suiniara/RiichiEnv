<!-- kar-module: riichienv.kar.learning.config -->
<!-- api: ModelConfig -->
# learning.config — 模型 preset 与参数合同

源码：[config.py](../../../../src/riichienv/kar/learning/config.py) · [learning 索引](../README.md)

## 职责与接口

`ModelConfig` 是不创建网络的冻结配置数据类。`from_preset(name)`、`parameter_count` 和 `contract_hash` 分别提供标准尺寸、精确参数公式和模型身份。配置模块本身不导入 Torch。

| preset | layers / width / heads / feedforward |
|---|---|
| tiny | 4 / 192 / 3 / 512 |
| small | 6 / 256 / 4 / 704 |
| medium | 8 / 384 / 6 / 1024 |
| large | 8 / 512 / 8 / 1408 |

Small/Medium/Large 参数数为 5,239,435 / 14,790,795 / 26,536,587。实际构造模型还会检查真实参数总量。

```python
from riichienv.kar.learning.config import ModelConfig

config = ModelConfig.from_preset("small")
assert config.parameter_count == 5_239_435
```

## 合同

维度必须正整数；width 可整除 heads，单头维度为偶数以用于 RoPE。norm_eps、rope_base 有范围要求。preset 不是预算许可，选大模型仍需 N_train 阶梯、M1 证据和预算放行。

特征语义版本为 `visible_rule_features_v1_2_2`，随机置空版本另为 `sample_mask_v1`，不再混用。`use_rule_features` 是数据视图，不增加/删除参数。不要为了对齐旧权重手工改维度后保留原 contract hash。

## 接线与状态

由 [model](../model/README.md)、[trainer](../trainer/README.md)、[deployment](../deployment/README.md)消费；完整训练配方另见 [training_config](../training_config/README.md)。对应 [m1/test_model.py](../../../../tests/kar/m1/test_model.py)。本轮未启动阶段训练或棋力实验；对标准 preset 实际构造 CPU 模型并验证参数量。
