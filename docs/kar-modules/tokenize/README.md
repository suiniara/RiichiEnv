<!-- kar-module: riichienv.kar.tokenize -->
<!-- api: TokenizationUnavailableError, TokenSequence, TokenHistoryCache, encode_prefix, append_action -->
# tokenize — 前缀、ACTION 后缀与历史缓存

源码：[tokenize.py](../../../src/riichienv/kar/tokenize.py) · [Schema](../schema/README.md) · [总索引](../README.md)

## 职责与接口

`encode_prefix(snapshot, candidates, history_cache=None)` 把严格的 `PublicSnapshot` 和完整合法全局 ID 编成 `TokenSequence`；`append_action(prefix, action_id)` 只追加一个完整假设 ACTION。`TokenHistoryCache` 缓存 CPU 侧已封存历史行，不是模型 KV cache。

`TokenSequence` 保存行、候选、历史长度、DECIDE/ACTION 索引、schema 身份及二进制序列化能力。序列化包含帧头；下游数据路径区分整个 frame 与 139 字节 Token 流，不能任意切片当成另一种格式。

## 数据合同

前缀为 BOS + H 历史 + 62 状态槽 + DECIDE，共 `H+64` 行；追加 ACTION 后为 `H+65` 行。逻辑位置按 `public_semantic_clock_v1`，不按source record或H行数：最后公共语义位置C，S/DECIDE=C+1、ACTION=C+2。缺失/旧public_clock_version拒绝，不默认为当前。没有完整可见历史/本家手牌、候选非法或超过 32 时应报错，不静默补字段或截断。

```python
from riichienv.kar.tokenize import encode_prefix, append_action

# snapshot 与 legal_ids 必须来自相同的已核验决策截止点。
def conditional_input(snapshot, legal_ids, action_id):
    return append_action(encode_prefix(snapshot, legal_ids), action_id)
```

## 缓存与杠事务

缓存绑定观察者、起家顺序、Schema、输入语义版本与复合历史前缀；旧行冻结，不能输出陈旧字节。`snapshot.events` 可含本家确认 SELF_PASS，`processed_events` 是复合事件数而非公共 cutoff。SELF_PASS(kind26) 校验 owner/provenance/确认锚，不推进公共逻辑时钟，批量/增量一致。ACTION 与 LEGAL_SET 共用 Schema 因子工厂。杠接受 Token 仍只由当前公共提交证据产生。

不要让真实 E/R 标签进入前缀。候选独立 ACTION 分支由模型侧隔离，不把多个未来动作串成一条可互见历史。

## 错误、调用方与测试

`TokenizationUnavailableError` 表示必要可见事实不足。上游为 [observation](../observation/README.md)，下游为 `input_audit`、`learning/export`、`learning/inference`。对应 [test_tokenize.py](../../../tests/kar/test_tokenize.py)、[test_token_cache.py](../../../tests/kar/test_token_cache.py)、[m1/test_model.py](../../../tests/kar/m1/test_model.py)及 [test_self_pass_history.py](../../../tests/kar/test_self_pass_history.py)。本轮未执行阶段任务或反向/参数更新；定向历史及缓存对拍结果见[输入链路合同](../../kar-input-chain-repair.md)。
