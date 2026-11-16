<!-- kar-module: riichienv.kar.contracts -->
<!-- api: Ruleset, Response, canonical_bytes, content_hash, cohort_hash, matches_primary_cohort, model_tier, game_split -->
# contracts — 冻结规则与通用身份

源码：[contracts.py](../../../src/riichienv/kar/contracts.py) · [总索引](../README.md)

## 职责

统一规则、21 类响应枚举、主 cohort、游戏级 split 和 canonical hash。它是协议/审计/训练/推理共同的基础，不包含网络、优化器或牌谱下载逻辑。

## 主要接口

| 接口 | 输入与输出 |
|---|---|
| `Ruleset` / `MJSOUL_RULESET` | 冻结四人雀魂规则，提供规则 hash 和环境创建接口 |
| `Response` | 0..20 的响应语义；全工程共用同一枚举 |
| `canonical_bytes(value)` / `content_hash(value)` | 可 JSON 编码的数据 → 规范 UTF-8 字节 / SHA-256；拒绝 NaN |
| `matches_primary_cohort(metadata)` / `cohort_hash()` | 判断主 cohort 与标识其冻结定义 |
| `model_tier(n_train_half_games)` | 实际 train 半庄数 → small/medium/large |
| `game_split(game_id, salt)` | 稳定游戏身份和 salt → train/validation/test |

## 数据与合同

主 cohort 是 `category=2, mode=2, mode_id=12`，座位信息必须满足冻结要求。`sanchaho_is_draw=False`，保留三家实际荣和；同分按整场初始座位顺序，不按当前庄家。模型阶梯为 `<30000 Small / 30000..49999 Medium / >=50000 Large`，这里只给数据上限，预算仍可要求降级。

```python
from riichienv.kar.contracts import model_tier

tier = model_tier(29_999)  # small；不能用所有 split 总数代替该参数
```

## 调用关系与注意事项

上游提供真实 metadata/game_id；下游 `schema`、`actions`、`audit`、`acceptance`、`learning` 和 `runtime_contracts` 消费同一身份。`content_hash` 是一致性摘要，不是秘密签名。旧目录/catalog 的 split 不自动成为本项目 split。

非法计数、缺 ID/salt 或不可规范编码数据会拒绝处理；不把错误自动降为合格样本。

## 测试与状态

对应 [test_contracts.py](../../../tests/kar/test_contracts.py)、[test_rule_features.py](../../../tests/kar/test_rule_features.py)。本轮仅编写文档，未执行示例、测试或创建环境。共同边界见[总索引](../README.md#共同约束)。
