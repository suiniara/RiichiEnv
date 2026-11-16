<!-- kar-module: riichienv.kar.readiness -->
<!-- api: feature_readiness, validate_readiness -->
# readiness — 功能、数据、资源状态分离

源码：[readiness.py](../../../src/riichienv/kar/readiness.py) · [总索引](../README.md)

## 职责与接口

`feature_readiness(checks, evidence, source_hash, data_eligible=False, resource_ready=False)` 将明确的布尔检查与证据引用规范化。`validate_readiness(document)` 重建并核对版本、摘要、缺项及派生 flags。

`CORE_REQUIREMENTS` 为 P01..P12 与 U01；`FULL_REMAINING_REQUIREMENTS` 的历史名称对应 T01..T10，不表示今天这些条目仍缺实现。状态来自调用方提供的真实检查结果和引用，而不是本函数自动执行验收。

## 输出语义

`core_feature_ready`、`full_feature_ready`、`data_eligible`、`resource_ready` 分开保存。任一必需检查非 true 或没有证据引用就进入 missing。`parameter_updates_authorized` 固定 false，避免把数据 PASS、代码存在或设备空闲当作用户许可。

## 使用与错误

正确用途是封存当前源码的实际功能/数据/资源证据，供 [authorization](../authorization/README.md)进一步验证。不要为了让训练开始而把所有 checks 手填 true 或复制旧来源。

非法布尔结果、缺源码身份或派生字段与输入不一致会拒绝。纯元数据模块，无 Torch、文件写入或独立 CLI；它不连接硬件，也不运行测试。

## 测试与状态

对应 [core/test_authorization.py](../../../tests/kar/core/test_authorization.py)、[production/test_review_job_guards.py](../../../tests/kar/production/test_review_job_guards.py)。本轮未生成就绪证明或有效执行许可；README 不是验收证据。
