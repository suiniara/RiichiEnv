<!-- kar-module: riichienv.kar.tokenize -->
<!-- api: TokenizationUnavailableError, TokenSequence, TokenHistoryCache, encode_prefix, append_action -->
# tokenize — 前缀、ACTION 后缀与历史缓存

源码：[tokenize.py](../../../src/riichienv/kar/tokenize.py) · [Schema](../schema/README.md) · [总索引](../README.md)

## 职责与接口

`encode_prefix(snapshot, candidates, history_cache=None)` 把严格的 `PublicSnapshot` 和完整合法全局 ID 编成 `TokenSequence`；`append_action(prefix, action_id)` 只追加一个完整假设 ACTION。`TokenHistoryCache` 缓存 CPU 侧已封存历史行，不是模型 KV cache。

`TokenSequence` 保存行、候选、历史长度、DECIDE/ACTION 索引、schema 身份及二进制序列化能力。序列化包含帧头；下游数据路径区分整个 frame 与 139 字节 Token 流，不能任意切片当成另一种格式。

## 数据合同

前缀为 BOS + H 历史 + 62 状态槽 + DECIDE，共 `H+64` 行；追加 ACTION 后为 `H+65` 行。逻辑位置与块属性由可见事实决定。没有完整可见历史/本家手牌、候选非法或超过 32 时应报错，不静默补字段或截断。

```python
from riichienv.kar.tokenize import encode_prefix, append_action

# snapshot 与 legal_ids 必须来自相同的已核验决策截止点。
def conditional_input(snapshot, legal_ids, action_id):
    return append_action(encode_prefix(snapshot, legal_ids), action_id)
```

## 缓存与杠事务

缓存绑定观察者、起家顺序、Schema、历史前缀等上下文；旧行冻结，不能修改后继续输出陈旧字节。杠接受 Token 使用既有 kind，并由当前公共提交证据产生；合成的原生迟到确认不能重复生成声明。

不要让真实 E/R 标签进入前缀。候选独立 ACTION 分支由模型侧隔离，不把多个未来动作串成一条可互见历史。

## 错误、调用方与测试

`TokenizationUnavailableError` 表示必要可见事实不足。上游为 [observation](../observation/README.md)，下游为 `input_audit`、`learning/export`、`learning/inference`。对应 [test_tokenize.py](../../../tests/kar/test_tokenize.py)、[test_token_cache.py](../../../tests/kar/test_token_cache.py)、[m1/test_model.py](../../../tests/kar/m1/test_model.py)及杠事务回归。本轮未执行示例或测试。
