<!-- kar-module: riichienv.kar.schema -->
<!-- api: load_schema, schema_hash, schema_status, action_factor_ids, token_defaults, validate_token, encode_token, decode_token -->
# schema — Token 字段与二进制合同

源码：[schema.py](../../../src/riichienv/kar/schema.py) · [冻结资源](../assets/README.md) · [总索引](../README.md)

## 职责

加载完整 `token_schema.riichienv.v1.json`，验证注册表、kind-specific 字段使用及跨字段约束，再负责单 Token 的编解码。不是把 overlay 文件 hash 当完整 Schema，也不容许未知标签字段进入输入。

## 主要接口

- `load_schema(path=None)`：读取并验证完整 JSON；缺省使用包内资源。
- `schema_hash(schema)` / `schema_status(path=None)`：规范化身份或带 PASS/BLOCKED/FAIL 含义的状态信息。
- `token_defaults(kind=0, block=0, position=0, valid=None)`：构造字段完整的初始行，仍需按对应 kind 填写。
- `validate_token(row)`：检查字段集合、范围、present 位与 kind-specific 语义。
- `encode_token(row)` / `decode_token(data)`：行 ↔ 固定 139 字节 little-endian Token。
- `action_factor_ids(action_id)`：从同一冻结 manifest 给出 ACTION 五因子，供后缀与 LEGAL_SET 共用；非舍牌的 tile/tsumogiri/riichi 为 N/A，不私改全部动作的因子。

## 数据与合同

活动 Schema 为 `kar-riichienv-v1`，canonical SHA-256 为 `fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673`。设计引用 `kar-v1.2.2` 和 overlay SHA 不作为此 ID/hash 的别名；checkpoint 必须携带 [versioning](../versioning/README.md) 的显式映射与版本合同。`stage`、`flags`、`numeric` 不能当全局通用含义解释；例如 rank/gap 仅出现在规定的 ROUND_STATE 字段。候选容量、Embedding 行数和状态槽布局由完整注册表约束。

```python
from riichienv.kar.schema import schema_status

status = schema_status()  # 只说明注册表可用性，不授予数据或模型运行资格
```

## 调用关系与失败处理

[tokenize](../tokenize/README.md)生成行，[batch](../learning/batch/README.md)按同一布局解码，[runtime_contracts](../runtime_contracts/README.md)保存其身份。未知字段、错误 Token 长度、错误 kind/动作组合或在不适用 kind 写入 present 数值都会被拒绝；不能绕过校验后强行 pack。

## 测试与状态

对应 [test_schema.py](../../../tests/kar/test_schema.py)、[test_tokenize.py](../../../tests/kar/test_tokenize.py)、[m1/test_model.py](../../../tests/kar/m1/test_model.py)。本轮未执行真实数据重导出或阶段训练；CPU 验证结果见[实现覆盖报告](../../kar-implementation-coverage.md)；Schema 可加载仍不等于真实数据和阶段训练已经验收。
