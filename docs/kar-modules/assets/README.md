# KAR 冻结 JSON 资源

[返回模块总索引](../README.md)

## token_schema.riichienv.v1.json

源码资源：[完整 Token Schema](../../../src/riichienv/kar/token_schema.riichienv.v1.json)。由 [schema](../schema/README.md)加载，定义 kind、字段使用、枚举、状态槽、动作因子和二进制合同。活动 ID `kar-riichienv-v1`，canonical SHA-256：

```text
fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673
```

该值基于规范内容，不是 JSON 文件原始空白格式 hash，也不是设计包 overlay 的文件 hash。修改语义需要显式协议升级，不能为了让旧 checkpoint 通过而回填同一身份。

## golden_registry.v1.json

源码资源：[Golden registry](../../../src/riichienv/kar/golden_registry.v1.json)。由 [golden](../golden/README.md)消费，区分必需真实正例与基于真实来源的派生检查。它定义需要的案例，不等于已经存在有效案例，也不是测试执行结果。

真实 raw/protocol/cutoff、期望和当前规则核验仍必须提供。不得用 synthetic 单元 fixture、旧 PASS 字符串或擅自改 N/A 补缺项。

## 使用、测试与状态

它们是包内数据资源而非可执行脚本；不导入下载的 pb2 或旧数据工程。对应 [test_schema.py](../../../tests/kar/test_schema.py)、[test_golden.py](../../../tests/kar/test_golden.py)。本轮没有修改 JSON 资源、协议身份或验收状态。
